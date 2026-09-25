use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use half::f16;
use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::Dtype;

/// One tensor as it sits in the file.
#[derive(Debug, Clone)]
pub struct RawTensor {
    pub data: Bytes,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

impl RawTensor {
    /// bf16/f16/f32 -> f16, matching `qwen3-asr`'s `RawTensor::to_f16_vec`.
    ///
    /// INT8 (`I8` + `<name>.weight_scale`, see [`dequant_i8`]) is deliberately
    /// *not* handled here: callers that need the whole table go through
    /// [`get_f16`], which dequantizes on the M2 path.  Keeping this narrow
    /// means a stray `to_f16_vec` on an int8 tensor fails loudly instead of
    /// silently dropping the scale.
    pub fn to_f16_vec(&self) -> Result<Vec<half::f16>> {
        Ok(match self.dtype {
            Dtype::F16 => self
                .data
                .chunks_exact(2)
                .map(|c| half::f16::from_ne_bytes([c[0], c[1]]))
                .collect(),
            Dtype::F32 => self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .map(half::f16::from_f32)
                .collect(),
            Dtype::BF16 => self
                .data
                .chunks_exact(2)
                .map(|c| {
                    let b = u16::from_ne_bytes([c[0], c[1]]);
                    half::f16::from_f32(f32::from_bits((b as u32) << 16))
                })
                .collect(),
            other => return Err(anyhow!("unsupported dtype {other:?}")),
        })
    }

    /// Narrow this tensor's elements to little-endian f16 into `dst`.
    ///
    /// The destination form exists so a *fused* matrix (`q|k|v`, `gate|up`) can
    /// be narrowed straight into its final row-major layout: one pass, no
    /// intermediate `Vec<f16>`, and no concatenation copy afterwards.
    pub fn narrow_f16_into(&self, dst: &mut [u8]) -> Result<()> {
        let n = self.data.len() / self.dtype.size();
        anyhow::ensure!(
            dst.len() == n * 2,
            "dst is {} bytes for {n} elements, needs {}",
            dst.len(),
            n * 2
        );
        match self.dtype {
            // safetensors f16 is already little-endian, which is what the
            // kernels read, so this is the file's own bytes.
            Dtype::F16 => dst.copy_from_slice(&self.data),
            Dtype::BF16 => narrow_into(&self.data, dst, 2, |c| {
                f16::from_f32(f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            }),
            Dtype::F32 => narrow_into(&self.data, dst, 4, |c| {
                f16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            }),
            other => return Err(anyhow!("unsupported dtype {other:?} for a weight matrix")),
        }
        Ok(())
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        Ok(match self.dtype {
            Dtype::F32 => self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Dtype::F16 => self
                .data
                .chunks_exact(2)
                .map(|c| half::f16::from_ne_bytes([c[0], c[1]]).to_f32())
                .collect(),
            Dtype::BF16 => self
                .data
                .chunks_exact(2)
                .map(|c| {
                    let b = u16::from_ne_bytes([c[0], c[1]]);
                    f32::from_bits((b as u32) << 16)
                })
                .collect(),
            other => return Err(anyhow!("unsupported dtype {other:?} for to_f32_vec")),
        })
    }

    pub fn as_f16(&self) -> Result<(Vec<half::f16>, Vec<usize>)> {
        Ok((self.to_f16_vec()?, self.shape.clone()))
    }

    pub fn as_f32(&self) -> Result<(Vec<f32>, Vec<usize>)> {
        Ok((self.to_f32_vec()?, self.shape.clone()))
    }

    /// Append row `row` of a `[*, cols]` tensor to `out` as little-endian f16 —
    /// the byte layout the decoder's `prefill` takes.
    ///
    /// An f16 tensor is a straight `memcpy` out of the mapped file; any other
    /// dtype is converted element-wise.  Use this for token lookups instead of
    /// [`Self::to_f16_vec`], which materialises the *whole* table on every call:
    /// `embed_tokens` is 155 M elements at 0.6 B and twice that at 1.7 B, i.e.
    /// ~0.3 s / ~0.6 s of allocation and conversion per transcription.
    pub fn append_f16_row_le(&self, row: usize, cols: usize, out: &mut Vec<u8>) -> Result<()> {
        anyhow::ensure!(self.shape.len() >= 1 && cols == *self.shape.last().unwrap(), "row cols {cols} != {:?}", self.shape);
        let stride = cols * 2;
        let start = row
            .checked_mul(stride)
            .filter(|s| s + stride <= self.data.len())
            .ok_or_else(|| anyhow!("row {row} of {:?} is out of range", self.shape))?;
        match self.dtype {
            Dtype::F16 => out.extend_from_slice(&self.data[start..start + stride]),
            Dtype::BF16 => {
                for c in self.data[start..start + stride].chunks_exact(2) {
                    let b = u16::from_ne_bytes([c[0], c[1]]);
                    let h = half::f16::from_f32(f32::from_bits((b as u32) << 16));
                    out.extend_from_slice(&h.to_le_bytes());
                }
            }
            Dtype::F32 => {
                for c in self.data[start..].chunks_exact(4).take(cols) {
                    let h = half::f16::from_f32(f32::from_ne_bytes([c[0], c[1], c[2], c[3]]));
                    out.extend_from_slice(&h.to_le_bytes());
                }
            }
            other => anyhow::bail!("unsupported dtype {other:?}"),
        }
        Ok(())
    }
}

/// mmap every safetensors shard, zero-copy, mirroring `weights.rs`.
pub fn load_tensors(model_dir: &Path) -> Result<HashMap<String, RawTensor>> {
    let index = model_dir.join("model.safetensors.index.json");
    if index.exists() {
        let idx: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&index)?)?;
        let wm = idx["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow!("invalid index.json"))?;
        let mut shards: Vec<String> = wm
            .values()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        shards.sort();
        shards.dedup();
        let mut all = HashMap::new();
        for s in shards {
            all.extend(load_shard(&model_dir.join(&s))?);
        }
        return Ok(normalize_hf_weight_names(all));
    }
    Ok(normalize_hf_weight_names(load_shard(&model_dir.join("model.safetensors"))?))
}

fn remap_hf_weight_name(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("model.multi_modal_projector.linear_1") {
        return format!("thinker.audio_tower.proj1{rest}");
    }
    if let Some(rest) = name.strip_prefix("model.multi_modal_projector.linear_2") {
        return format!("thinker.audio_tower.proj2{rest}");
    }
    if let Some(rest) = name.strip_prefix("model.audio_tower.") {
        return format!("thinker.audio_tower.{rest}");
    }
    if let Some(rest) = name.strip_prefix("model.language_model.") {
        return format!("thinker.model.{rest}");
    }
    name.to_string()
}

fn normalize_hf_weight_names(weights: HashMap<String, RawTensor>) -> HashMap<String, RawTensor> {
    let is_hf = weights.keys().any(|k| {
        k.starts_with("model.audio_tower.") || k.starts_with("model.language_model.")
    });
    if !is_hf {
        return weights;
    }
    weights
        .into_iter()
        .map(|(k, v)| (remap_hf_weight_name(&k), v))
        .collect()
}

fn load_shard(path: &Path) -> Result<HashMap<String, RawTensor>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("mmap {}", path.display()))?;
    let buf: Bytes = Bytes::from_owner(mmap);
    let st = safetensors::SafeTensors::deserialize(&buf)?;
    let base = buf.as_ptr() as usize;

    let mut out = HashMap::with_capacity(st.len());
    for (name, view) in st.iter() {
        let vd = view.data();
        let offset = vd.as_ptr() as usize - base;
        out.insert(
            name.to_string(),
            RawTensor {
                data: buf.slice(offset..offset + vd.len()),
                shape: view.shape().to_vec(),
                dtype: view.dtype(),
            },
        );
    }
    Ok(out)
}

/// A weight matrix in the final device layout: packed f16, row-major `[rows, cols]`.
///
/// `data` is `Bytes` rather than `Vec<u8>` so an f16 checkpoint can hand the
/// mapped file's own pages to the uploader: `Bytes::clone` is a refcount bump.
/// Every alternative costs whole passes over the weights — a `Vec<f16>` and then
/// a re-encoded `Vec<u8>` — before the bytes have even reached the bus, and at
/// 0.6 B that was ~2 s of the load for nothing.
pub struct PackedWeight {
    pub data: Bytes,
    pub rows: usize,
    pub cols: usize,
}

impl PackedWeight {
    pub fn from_f16(v: &[half::f16], rows: usize, cols: usize) -> Self {
        let mut data = Vec::with_capacity(v.len() * 2);
        for x in v {
            data.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        Self { data: data.into(), rows, cols }
    }

    /// [`RawTensor`] → packed f16, converting in one pass (and not at all when
    /// the tensor is already f16).
    ///
    /// safetensors is little-endian and little-endian f16 is exactly what the
    /// kernels read (the same assumption [`RawTensor::append_f16_row_le`] makes
    /// for token lookups), so an f16 checkpoint hands over the mapped file's own
    /// bytes.
    ///
    /// The converting cases are the measured ones: these checkpoints are bf16,
    /// and the old route (`to_f16_vec` → `Vec<f16>` → re-encoded `Vec<u8>`) ran
    /// three passes over every weight on one core — ~2 s of the load at 0.6 B,
    /// more than the PCIe transfer it was feeding.  bf16 → f32 is exact (a
    /// shift), so narrowing straight to f16 with the same round-to-nearest-even
    /// is bit-identical to what that route produced.
    pub fn from_raw(t: &RawTensor, rows: usize, cols: usize) -> Result<Self> {
        let n = rows * cols;
        anyhow::ensure!(
            t.data.len() == n * t.dtype.size(),
            "{rows}x{cols} needs {} bytes, tensor has {}",
            n * t.dtype.size(),
            t.data.len()
        );
        let data: Bytes = match t.dtype {
            Dtype::F16 => t.data.clone(),
            _ => {
                let mut out = vec![0u8; n * 2];
                t.narrow_f16_into(&mut out)?;
                out.into()
            }
        };
        Ok(Self { data, rows, cols })
    }

    /// Concatenate row-blocks: `[a | b | c]` along the output (row) dimension.
    pub fn concat_rows(parts: &[PackedWeight], cols: usize) -> Result<Self> {
        let rows: usize = parts.iter().map(|p| p.rows).sum();
        let mut data = Vec::with_capacity(rows * cols * 2);
        for p in parts {
            if p.cols != cols {
                return Err(anyhow!(
                    "concat_rows: cols mismatch ({} vs {cols})",
                    p.cols
                ));
            }
            data.extend_from_slice(&p.data);
        }
        Ok(Self { data: data.into(), rows, cols })
    }
}

/// A decoder linear that **stays INT8 on the device** (the M3 layout).
///
/// `data` is the file's own int8 row-major `[rows, cols]` bytes, read by the W8
/// kernels as little-endian u32 words — 4 weights per word, byte `j` of a word
/// is value `4*word + j` (LSB first).  No repacking happens: the layout the
/// kernels want *is* the file's byte sequence reinterpreted, so the tensor's own
/// bytes go to the GPU verbatim.  `scale` is the file's own `F32 [rows]` bytes
/// verbatim (safetensors is little-endian).  Nothing here dequantizes: the f16
/// values M2 produced are now formed in-shader, per value, as
/// `f16(q * scale[row])` — the exact arithmetic [`dequant_i8`] does on the host,
/// so the W8 kernels see the same numbers M2 handed the f16 kernels.
pub struct PackedInt8 {
    pub data: Bytes,
    pub scale: Bytes,
    pub rows: usize,
    pub cols: usize,
}

impl PackedInt8 {
    /// [`RawTensor`] → device int8: validate and hand over both tensors' own
    /// bytes.  `scale_raw` must be the `F32 [rows]` sibling ([`scale_name`]
    /// finds it).  `cols` must be a multiple of 4 so every row starts on a word
    /// boundary — every quantized shape satisfies it (all decoder `cols` are
    /// multiples of 1024).
    pub fn from_raw(t: &RawTensor, scale_raw: &RawTensor) -> Result<Self> {
        anyhow::ensure!(t.shape.len() == 2, "expected 2D int8 weight, got {:?}", t.shape);
        let (rows, cols) = (t.shape[0], t.shape[1]);
        anyhow::ensure!(
            cols % 4 == 0,
            "int8 weight cols {cols} is not a multiple of 4 — the u32 word view needs row-aligned words"
        );
        anyhow::ensure!(
            t.data.len() == rows * cols,
            "int8 weight is {} bytes for {rows}x{cols}",
            t.data.len()
        );
        anyhow::ensure!(
            scale_raw.dtype == Dtype::F32 && scale_raw.shape == vec![rows],
            "int8 scale must be F32 [{rows}], got {:?} {:?}",
            scale_raw.dtype,
            scale_raw.shape
        );
        Ok(Self { data: t.data.clone(), scale: scale_raw.data.clone(), rows, cols })
    }

    /// Concatenate row-blocks along the output dimension — weights and their
    /// scales in the same order, so fused row `r` reads `S[r]`.
    pub fn concat_rows(parts: &[PackedInt8], cols: usize) -> Result<Self> {
        let rows: usize = parts.iter().map(|p| p.rows).sum();
        let mut data = Vec::new();
        let mut scale = Vec::new();
        for p in parts {
            anyhow::ensure!(p.cols == cols, "concat_rows: cols mismatch ({} vs {cols})", p.cols);
            data.extend_from_slice(&p.data);
            scale.extend_from_slice(&p.scale);
        }
        Ok(Self { data: data.into(), scale: scale.into(), rows, cols })
    }
}

/// Suffix for the per-row fp32 scale of an INT8-quantized weight.
///
/// The exporter (`D:\Qwen3-ASR\scripts\quantize_int8.py`) stores each target
/// under its original HF name as `I8 [out, in]`, plus `<name without `.weight`>`
/// + this suffix as `F32 [out]`.  Only decoder linears are quantized
/// (`q/k/v/o`, `gate/up/down`); everything else stays bf16/f16.
pub const INT8_SCALE_SUFFIX: &str = ".weight_scale";

/// Per-row dequantize: `f16(q[i, j] * scale[i])`, one pass, parallel.
///
/// `q` is signed `I8 [rows, cols]`, `scale` is `F32 [rows]` (any float dtype
/// the file carries is widened through f32).  The math matches the Python
/// shim (`wer_compare.py` / `mini_eval.py`): `w = q.to(f32) * s[:, None]`,
/// then round-to-nearest-even to f16 — so M2 text must equal the shim's.
pub fn dequant_i8(q: &[u8], scale_f32: &[f32], rows: usize, cols: usize) -> Result<Vec<half::f16>> {
    anyhow::ensure!(q.len() == rows * cols, "int8 weight is {} bytes for {rows}x{cols}", q.len());
    anyhow::ensure!(scale_f32.len() == rows, "scale is {} for {rows} rows", scale_f32.len());
    let mut out = vec![half::f16::from_f32(0.0); rows * cols];
    out.par_chunks_mut(cols)
        .enumerate()
        .for_each(|(r, dst)| {
            let s = scale_f32[r];
            for (d, &qb) in dst.iter_mut().zip(&q[r * cols..(r + 1) * cols]) {
                *d = half::f16::from_f32((qb as i8) as f32 * s);
            }
        });
    Ok(out)
}

/// Fetch `name`, convert to f16, and hand back both the vector and its shape.
///
/// M2 path: a `Dtype::I8` tensor is dequantized with its
/// `<name>.weight_scale` (`F32 [out]`) sibling before anything else sees it,
/// so every downstream caller (`get_matrix`, `cpu_decoder::Mat::load`,
/// `fused_pieces`) works unchanged.  GPU/CPU constant memory is still f16 —
/// this step saves no VRAM, it only proves the file correct.
pub fn get_f16(
    w: &HashMap<String, RawTensor>,
    name: &str,
) -> Result<(Vec<half::f16>, Vec<usize>)> {
    let t = w
        .get(name)
        .ok_or_else(|| anyhow!("weight not found: {name}"))?;
    if t.dtype == Dtype::I8 {
        let sname = scale_name(name)?;
        let s = w
            .get(&sname)
            .ok_or_else(|| anyhow!("int8 weight {name} needs scale tensor {sname}"))?;
        anyhow::ensure!(t.shape.len() == 2, "expected 2D int8 weight {name}, got {:?}", t.shape);
        let (rows, cols) = (t.shape[0], t.shape[1]);
        let scale_f32 = s.to_f32_vec()?;
        anyhow::ensure!(
            scale_f32.len() == rows,
            "scale {sname} is {} for {rows} rows",
            scale_f32.len()
        );
        return Ok((dequant_i8(&t.data, &scale_f32, rows, cols)?, t.shape.clone()));
    }
    Ok((t.to_f16_vec()?, t.shape.clone()))
}

/// `<weight name>` -> `<scale name>`: strip one trailing `.weight`, append
/// [`INT8_SCALE_SUFFIX`].  Errors (rather than guessing) on names that do not
/// end in `.weight`, so a future format change fails at load, not silently.
fn scale_name(weight_name: &str) -> Result<String> {
    weight_name
        .strip_suffix(".weight")
        .map(|p| format!("{p}{INT8_SCALE_SUFFIX}"))
        .ok_or_else(|| anyhow!("int8 weight {weight_name} does not end in `.weight`"))
}

/// Narrow `src` (little-endian `elem_bytes`-wide elements) to f16 in `dst`, in
/// parallel.
///
/// One pass over each of the source and destination, and the chunking is what
/// makes a 780 M-element narrowing a load-time rounding error instead of the
/// dominant cost.
fn narrow_into(
    src: &[u8],
    dst: &mut [u8],
    elem_bytes: usize,
    to_f16: impl Fn(&[u8]) -> half::f16 + Sync,
) {
    const CHUNK: usize = 1 << 16;
    dst.par_chunks_mut(CHUNK * 2)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let base = ci * CHUNK;
            for (k, slot) in chunk.chunks_exact_mut(2).enumerate() {
                let off = (base + k) * elem_bytes;
                slot.copy_from_slice(&to_f16(&src[off..off + elem_bytes]).to_bits().to_le_bytes());
            }
        });
}

/// Fetch `name` as a packed weight matrix, dequantizing INT8 first.
///
/// For an `I8` tensor this routes through [`get_f16`] (scale-aware) and
/// repacks; for every float dtype it stays on the zero-copy path
/// ([`PackedWeight::from_raw`] hands over the mapped file's own bytes when the
/// tensor is already f16).
pub fn get_matrix(w: &HashMap<String, RawTensor>, name: &str) -> Result<PackedWeight> {
    let t = w
        .get(name)
        .ok_or_else(|| anyhow!("weight not found: {name}"))?;
    if t.shape.len() != 2 {
        return Err(anyhow!("expected 2D weight {name}, got {:?}", t.shape));
    }
    if t.dtype == Dtype::I8 {
        let (v, shape) = get_f16(w, name)?;
        return Ok(PackedWeight::from_f16(&v, shape[0], shape[1]));
    }
    PackedWeight::from_raw(t, t.shape[0], t.shape[1])
}

pub fn get_vector(w: &HashMap<String, RawTensor>, name: &str) -> Result<Vec<half::f16>> {
    Ok(get_f16(w, name)?.0)
}

/// Fetch `name` as a resident-int8 matrix — the M3 path (`get_matrix`'s int8
/// counterpart without the dequantize step).
pub fn get_int8(w: &HashMap<String, RawTensor>, name: &str) -> Result<PackedInt8> {
    let t = w
        .get(name)
        .ok_or_else(|| anyhow!("weight not found: {name}"))?;
    anyhow::ensure!(t.dtype == Dtype::I8, "{name} is {:?}, not I8", t.dtype);
    let sname = scale_name(name)?;
    let s = w
        .get(&sname)
        .ok_or_else(|| anyhow!("int8 weight {name} needs scale tensor {sname}"))?;
    PackedInt8::from_raw(t, s)
}

/// Pack a `Vec<f16>` into the u32-word layout the kernels read.
pub fn pack_words(v: &[half::f16]) -> Vec<u32> {
    assert!(v.len() % 2 == 0, "pack_words needs an even element count");
    v.chunks_exact(2)
        .map(|c| (c[0].to_bits() as u32) | ((c[1].to_bits() as u32) << 16))
        .collect()
}

/// Same, straight from little-endian bytes (avoids a round-trip through f16).
pub fn pack_words_from_bytes(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0, "pack_words_from_bytes needs 4-byte multiples");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Bytes for a `array<u32>` binding built from an f16 slice.
pub fn words_bytes(v: &[half::f16]) -> Vec<u8> {
    let words = pack_words(v);
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn i8_tensor(bytes: &[i8], rows: usize, cols: usize) -> RawTensor {
        let data: Vec<u8> = bytes.iter().map(|&b| b as u8).collect();
        RawTensor { data: Bytes::from(data), shape: vec![rows, cols], dtype: Dtype::I8 }
    }

    fn f32_tensor(v: &[f32]) -> RawTensor {
        let mut data = Vec::with_capacity(v.len() * 4);
        for x in v {
            data.extend_from_slice(&x.to_le_bytes());
        }
        RawTensor { data: Bytes::from(data), shape: vec![v.len()], dtype: Dtype::F32 }
    }

    #[test]
    fn dequant_i8_matches_python_shim() {
        // w = q.to(f32) * scale[:, None], then f16 rounding — the shim's math.
        let (rows, cols) = (3, 4);
        let q: Vec<i8> = vec![-128, -64, -1, 0, 1, 7, 100, 127, -127, 63, -33, 42];
        let scale = vec![0.03125f32, 0.5, 0.0078125];
        let got = dequant_i8(
            &q.iter().map(|&b| b as u8).collect::<Vec<_>>(),
            &scale, rows, cols,
        )
        .unwrap();
        for r in 0..rows {
            for c in 0..cols {
                let want = half::f16::from_f32(q[r * cols + c] as f32 * scale[r]);
                assert_eq!(
                    got[r * cols + c].to_bits(),
                    want.to_bits(),
                    "row {r} col {c}: {:?} vs {:?}",
                    got[r * cols + c],
                    want
                );
            }
        }
    }

    #[test]
    fn dequant_i8_extremes_are_finite_and_scaled() {
        // -128 * the smallest scale must not overflow f16, and 127*s must be
        // exactly representable where f16 allows it.
        let got = dequant_i8(&[128u8, 127, 0, 1], &[0.5f32], 1, 4).unwrap();
        assert_eq!(got[0].to_f32(), -64.0);
        assert_eq!(got[1].to_f32(), 63.5);
        assert_eq!(got[2].to_f32(), 0.0);
        assert_eq!(got[3].to_f32(), 0.5);
    }

    #[test]
    fn dequant_i8_rejects_shape_mismatch() {
        assert!(dequant_i8(&[0u8; 6], &[1.0, 1.0], 2, 4).is_err(), "short data");
        assert!(dequant_i8(&[0u8; 8], &[1.0], 2, 4).is_err(), "short scale");
        assert!(dequant_i8(&[0u8; 8], &[1.0, 1.0], 2, 4).is_ok());
    }

    #[test]
    fn scale_name_round_trip() {
        assert_eq!(
            scale_name("thinker.model.layers.0.self_attn.q_proj.weight").unwrap(),
            "thinker.model.layers.0.self_attn.q_proj.weight_scale"
        );
        assert!(scale_name("thinker.model.layers.0.mlp.down_proj").is_err());
    }

    #[test]
    fn get_f16_dequantizes_int8_with_scale() {
        let mut w = HashMap::new();
        // 2x2 int8 rows scaled by [0.25, -0.5]
        w.insert(
            "a.weight".to_string(),
            i8_tensor(&[4, -8, -2, 3], 2, 2),
        );
        w.insert("a.weight_scale".to_string(), f32_tensor(&[0.25, -0.5]));
        let (v, shape) = get_f16(&w, "a.weight").unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(v[0].to_f32(), 1.0); // 4 * 0.25
        assert_eq!(v[1].to_f32(), -2.0); // -8 * 0.25
        assert_eq!(v[2].to_f32(), 1.0); // -2 * -0.5
        assert_eq!(v[3].to_f32(), -1.5); // 3 * -0.5
    }

    #[test]
    fn get_f16_int8_missing_scale_is_an_error() {
        let mut w = HashMap::new();
        w.insert("a.weight".to_string(), i8_tensor(&[1, 2, 3, 4], 2, 2));
        let err = get_f16(&w, "a.weight").unwrap_err().to_string();
        assert!(err.contains("a.weight_scale"), "got: {err}");
    }

    #[test]
    fn get_matrix_dequantizes_int8() {
        let mut w = HashMap::new();
        w.insert("m.weight".to_string(), i8_tensor(&[127, 0, 0, 127], 2, 2));
        w.insert("m.weight_scale".to_string(), f32_tensor(&[1.0, 2.0]));
        let p = get_matrix(&w, "m.weight").unwrap();
        assert_eq!((p.rows, p.cols), (2, 2));
        assert_eq!(p.data.len(), 8, "packed f16 is rows*cols*2 bytes");
        let words = pack_words_from_bytes(&p.data);
        let h0 = half::f16::from_bits((words[0] & 0xFFFF) as u16);
        let h1 = half::f16::from_bits((words[0] >> 16) as u16);
        assert_eq!(h0.to_f32(), 127.0);
        assert_eq!(h1.to_f32(), 0.0);
    }

    #[test]
    fn int8_bytes_are_the_u32_words_the_kernels_read() {
        // The device layout is the file's byte sequence reinterpreted as
        // little-endian u32 words: unpacking a word's bytes low-to-high gives
        // back the four q values in order, and `data` needs no repacking.
        let q: Vec<u8> = vec![1, 2, 3, 4, 0xFF, 0x80, 0x7F, 0];
        let mut w = HashMap::new();
        w.insert("m.weight".to_string(), i8_tensor(
            &q.iter().map(|&b| b as i8).collect::<Vec<_>>(), 2, 4));
        w.insert("m.weight_scale".to_string(), f32_tensor(&[1.0, 1.0]));
        let p = get_int8(&w, "m.weight").unwrap();
        assert_eq!(p.data.to_vec(), q, "data goes through verbatim");
        for (j, want) in q.iter().enumerate() {
            let word = u32::from_le_bytes(p.data[(j / 4) * 4..(j / 4) * 4 + 4].to_vec().try_into().unwrap());
            let b = ((word >> (8 * (j % 4) as u32)) & 0xFF) as u8;
            assert_eq!(b, *want, "byte {j}");
        }
    }

    #[test]
    fn packed_int8_carries_bytes_and_scale_verbatim() {
        let mut w = HashMap::new();
        w.insert("m.weight".to_string(), i8_tensor(&[4, -8, -2, 3, 1, 0, 0, 2], 2, 4));
        w.insert("m.weight_scale".to_string(), f32_tensor(&[0.25, -0.5]));
        let p = get_int8(&w, "m.weight").unwrap();
        assert_eq!((p.rows, p.cols), (2, 4));
        assert_eq!(p.data.len(), 8, "packed int8 is rows*cols/4 u32 words");
        assert_eq!(p.scale.len(), 8, "scale is rows f32");
        // The scale tensor's own bytes (little-endian f32) go through verbatim.
        assert_eq!(f32::from_le_bytes(p.scale[0..4].try_into().unwrap()), 0.25);
        assert_eq!(f32::from_le_bytes(p.scale[4..8].try_into().unwrap()), -0.5);
    }

    #[test]
    fn packed_int8_rejects_unaligned_cols_and_bad_scale() {
        let mut w = HashMap::new();
        w.insert("m.weight".to_string(), i8_tensor(&[1, 2, 3, 4, 5, 6], 2, 3));
        w.insert("m.weight_scale".to_string(), f32_tensor(&[1.0, 1.0]));
        assert!(get_int8(&w, "m.weight").is_err(), "cols 3 is not word-aligned");
        w.insert("m.weight".to_string(), i8_tensor(&[1, 2, 3, 4], 2, 2));
        w.insert("m.weight_scale".to_string(), f32_tensor(&[1.0]));
        assert!(get_int8(&w, "m.weight").is_err(), "scale must be [rows]");
    }

    #[test]
    fn packed_int8_concat_rows_orders_scales_with_weights() {
        let mk = |rows: usize, s: f32| -> PackedInt8 {
            let mut data = Vec::new();
            for _ in 0..rows * 4 {
                data.push(7u8);
            }
            let mut scale = Vec::new();
            for _ in 0..rows {
                scale.extend_from_slice(&s.to_le_bytes());
            }
            PackedInt8 { data: data.into(), scale: scale.into(), rows, cols: 4 }
        };
        let fused = PackedInt8::concat_rows(&[mk(2, 0.5), mk(3, 2.0)], 4).unwrap();
        assert_eq!(fused.rows, 5);
        assert_eq!(fused.data.len(), 20, "5 rows * 4 cols / 4 bytes-per-word * 4");
        assert_eq!(fused.scale.len(), 20);
        for (r, want) in [0.5, 0.5, 2.0, 2.0, 2.0].iter().enumerate() {
            let s = f32::from_le_bytes(fused.scale[r * 4..r * 4 + 4].try_into().unwrap());
            assert_eq!(s, *want, "scale row {r} must follow its weight block");
        }
    }
}
