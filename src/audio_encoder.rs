//! CPU audio encoder for Qwen3-ASR — f32 GEMM (`gemm` crate + rayon).
//! Safetensors weights are f16; converted once at load.
//!
//! Architecture:
//!   mel [T_mel, 128]  →  conv2d stem (3 × {im2col + gemm + bias + GELU})
//!   → conv_out (Linear + bias)  →  + sinusoidal PE
//!   → 18 × { LN + windowed attn + LN + FFN(GELU-erf) }
//!   → ln_post + proj1 + GELU + proj2
//!   → [n_total, output_dim]
//!
use anyhow::Result;
use gemm::{gemm, Parallelism};
use rayon::prelude::*;
use std::collections::HashMap;

use crate::config::AudioEncoderConfig;
use crate::cpu_tensor::{linear, CpuTensor, CpuWeightF16};
use crate::weights::RawTensor;

// ─── Linear + LayerNorm primitives ─────────────────────────────────

pub(crate) struct CpuAudioLinear {
    pub w_f32: crate::cpu_tensor::CpuWeight,
    pub bias: Option<Vec<f32>>,
}

impl CpuAudioLinear {
    pub(crate) fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
    ) -> Result<Self> {
        let (data, shape) = weights
            .get(&format!("{}.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("weight not found: {}.weight", prefix))?
            .as_f16()?;
        let rows = shape[0];
        let cols = shape[1];
        let bias = if weights.contains_key(&format!("{}.bias", prefix)) {
            let (b, _) = weights
                .get(&format!("{}.bias", prefix))
                .unwrap()
                .as_f32()?;
            Some(b)
        } else {
            None
        };
        let w_f16 = CpuWeightF16 { data, rows, cols };
        Ok(Self { w_f32: w_f16.to_f32(), bias })
    }

    /// x: [..., in_features] → [..., out_features]  (bias added if present)
    pub(crate) fn forward(&self, x: &CpuTensor) -> Result<CpuTensor> {
        let mut y = linear(x, &self.w_f32);
        if let Some(b) = &self.bias {
            let last = y.shape.last().unwrap();
            y.data.par_chunks_mut(*last).for_each(|row| {
                for j in 0..row.len() { row[j] += b[j]; }
            });
        }
        Ok(y)
    }
}

pub(crate) struct CpuAudioLayerNorm {
    pub w: Vec<f32>,
    pub bias: Vec<f32>,
    pub eps: f32,
}

impl CpuAudioLayerNorm {
    pub(crate) fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        eps: f32,
    ) -> Result<Self> {
        let (w, _) = weights
            .get(&format!("{}.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("ln weight not found: {}.weight", prefix))?
            .as_f32()?;
        let (bias, _) = weights
            .get(&format!("{}.bias", prefix))
            .ok_or_else(|| anyhow::anyhow!("ln bias not found: {}.bias", prefix))?
            .as_f32()?;
        Ok(Self { w, bias, eps })
    }

    /// x: [outer, d] → [outer, d]  (LayerNorm over last dim)
    pub(crate) fn forward(&self, x: &CpuTensor) -> CpuTensor {
        let d = *x.shape.last().unwrap();
        let w = &self.w;
        let bias = &self.bias;
        let eps = self.eps;
        let mut out = vec![0.0f32; x.data.len()];
        out.par_chunks_mut(d)
            .zip(x.data.par_chunks(d))
            .for_each(|(o_row, x_row)| {
                let mut mean = 0.0f32;
                for &v in x_row { mean += v; }
                mean /= d as f32;
                let mut var = 0.0f32;
                for &v in x_row { let d_ = v - mean; var += d_ * d_; }
                var /= d as f32;
                let inv_std = 1.0 / (var + eps).sqrt();
                for j in 0..d {
                    o_row[j] = (x_row[j] - mean) * inv_std * w[j] + bias[j];
                }
            });
        CpuTensor::new(out, x.shape.clone())
    }
}

/// Match `torch.nn.functional.gelu` / `ACT2FN["gelu"]` (erf, not tanh).
#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

pub(crate) fn gelu_inplace(x: &mut CpuTensor) {
    x.data.par_iter_mut().for_each(|v| *v = gelu(*v));
}

/// Standard im2col for conv2d(kernel=3, stride=2, pad=1).
/// Input x: [b, c_in, h, w].
/// Output: [col_count, c_in*9] row-major, where col_count = b * h_out * w_out.
/// Column layout: col[ic*9 + kh*3 + kw] matches weight [c_out, c_in, 3, 3] flattened.
pub(crate) fn im2col_3x3_s2p1(x: &[f32], b: usize, c_in: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let h_out = (h + 2 - 3) / 2 + 1;
    let w_out = (w + 2 - 3) / 2 + 1;
    let col_count = b * h_out * w_out;
    let k = c_in * 9;
    let mut cols = vec![0.0f32; col_count * k];
    cols.par_chunks_mut(k)
        .enumerate()
        .for_each(|(col_idx, col)| {
            let ib = col_idx / (h_out * w_out);
            let rem = col_idx % (h_out * w_out);
            let ho = rem / w_out;
            let wo = rem % w_out;
            for ic in 0..c_in {
                for kh in 0..3 {
                    for kw in 0..3 {
                        let ih = (ho * 2 + kh) as isize - 1;
                        let iw = (wo * 2 + kw) as isize - 1;
                        let v = if ih < 0 || ih >= h as isize || iw < 0 || iw >= w as isize {
                            0.0
                        } else {
                            unsafe { *x.get_unchecked(((ib * c_in + ic) * h + ih as usize) * w + iw as usize) }
                        };
                        col[ic * 9 + kh * 3 + kw] = v;
                    }
                }
            }
        });
    (cols, h_out, w_out)
}

#[allow(dead_code)]
pub(crate) struct CpuConvStem {
    c1_w: crate::cpu_tensor::CpuWeight, c1_b: Vec<f32>,
    c2_w: crate::cpu_tensor::CpuWeight, c2_b: Vec<f32>,
    c3_w: crate::cpu_tensor::CpuWeight, c3_b: Vec<f32>,
    co: CpuAudioLinear,
    pe: Vec<f32>,        // [max_pos, d_model] f32
    d_model: usize,
    max_pos: usize,
}

impl CpuConvStem {
    pub(crate) fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        config: &AudioEncoderConfig,
    ) -> Result<Self> {
        let c1_w = load_conv_weight(weights, &format!("{}.conv2d1.weight", prefix))?.to_f32();
        let c1_b = load_bias(weights, &format!("{}.conv2d1.bias", prefix))?;
        let c2_w = load_conv_weight(weights, &format!("{}.conv2d2.weight", prefix))?.to_f32();
        let c2_b = load_bias(weights, &format!("{}.conv2d2.bias", prefix))?;
        let c3_w = load_conv_weight(weights, &format!("{}.conv2d3.weight", prefix))?.to_f32();
        let c3_b = load_bias(weights, &format!("{}.conv2d3.bias", prefix))?;
        let co = CpuAudioLinear::load(weights, &format!("{}.conv_out", prefix))?;

        // Sinusoidal PE — formula identical to `gpu_audio_encoder.rs:220-228`.
        let dm = config.d_model;
        let max_pos = config.max_source_positions;
        let half = dm / 2;
        let lt = (10000.0f64).ln() / (half as f64 - 1.0);
        let mut pe = vec![0.0f32; max_pos * dm];
        for p in 0..max_pos {
            for i in 0..half {
                let a = p as f64 * (-(i as f64) * lt).exp();
                pe[p * dm + i] = a.sin() as f32;
                pe[p * dm + half + i] = a.cos() as f32;
            }
        }
        Ok(Self { c1_w, c1_b, c2_w, c2_b, c3_w, c3_b, co, pe, d_model: dm, max_pos })
    }

    /// Run conv stem on chunked mel input.
    /// mel_chunks: [b_chunks * n_mels * cs] in (chunk, mel_bin, frame) order.
    /// Layout matches GPU: [b_chunks, c=1, h=n_mels, w=cs] in NCHW.
    /// Returns ([b_chunks * t2 * d_model], t2).
    pub(crate) fn forward(
        &self,
        mel_chunks: &[f32],
        b_chunks: usize,
        n_mels: usize,
        cs: usize,
    ) -> Result<(Vec<f32>, usize)> {
        let c1_out = self.c1_w.rows;
        let c2_out = self.c2_w.rows;

        let (x1, h1, w1) = self.conv_block(mel_chunks, b_chunks, 1, n_mels, cs, &self.c1_w, &self.c1_b)?;
        let (x2, h2, w2) = self.conv_block(&x1, b_chunks, c1_out, h1, w1, &self.c2_w, &self.c2_b)?;
        let (x3, h3, w3) = self.conv_block(&x2, b_chunks, c2_out, h2, w2, &self.c3_w, &self.c3_b)?;
        // x3: [b_chunks, c3_out, h3, w3]
        let t2 = w3;

        // Permute [b, c, f, t] → [b, t, c, f] then reshape [b, t, c*f].
        let c_dim = self.c3_w.rows;
        let f_dim = h3;
        let row_len = c_dim * f_dim;
        let mut perm = vec![0.0f32; b_chunks * t2 * c_dim * f_dim];
        perm.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(idx, chunk)| {
                let ib = idx / t2;
                let it = idx % t2;
                for ic in 0..c_dim {
                    for f in 0..f_dim {
                        let src = ((ib * c_dim + ic) * f_dim + f) * t2 + it;
                        let dst = ic * f_dim + f;
                        chunk[dst] = x3[src];
                    }
                }
            });

        // ConvOut (Linear): [b, t2, c*f] → [b, t2, d_model]
        let perm_t = CpuTensor::new(perm, vec![b_chunks, t2, c_dim * f_dim]);
        let co_out = self.co.forward(&perm_t)?;

        // Add PE (broadcast over batch).
        let mut out = co_out.data.clone();
        let dm = self.d_model;
        let pe = &self.pe;
        out.par_chunks_mut(dm)
            .enumerate()
            .for_each(|(idx, chunk)| {
                let it = idx % t2;
                let pe_base = it * dm;
                for j in 0..dm {
                    chunk[j] += pe[pe_base + j];
                }
            });
        Ok((out, t2))
    }

    /// Single conv2d (3×3, stride 2, pad 1) + bias + GELU.
    /// Input x: [b, c_in, h, w]. Returns (flat [b, c_out, h_out, w_out], h_out, w_out).
    fn conv_block(
        &self,
        x: &[f32],
        b: usize, c_in: usize, h: usize, w: usize,
        w_w: &crate::cpu_tensor::CpuWeight, w_b: &[f32],
    ) -> Result<(Vec<f32>, usize, usize)> {
        let c_out = w_w.rows;
        assert_eq!(x.len(), b * c_in * h * w, "conv_block input size mismatch");
        let h_out = (h + 2 - 3) / 2 + 1;
        let w_out = (w + 2 - 3) / 2 + 1;
        let k = c_in * 9;
        assert_eq!(w_w.cols, k, "conv_block weight cols={} != c_in*9={}", w_w.cols, k);
        let plane = h_out * w_out;
        let in_plane = c_in * h * w;
        let mut out4d = vec![0.0f32; b * c_out * plane];
        // Tile batch so conv2/conv3 im2col stays cache-sized (full-b c2 is ~GB).
        const TILE: usize = 8;
        for b0 in (0..b).step_by(TILE) {
            let nb = TILE.min(b - b0);
            let (cols, ho, wo) = im2col_3x3_s2p1(
                &x[b0 * in_plane..(b0 + nb) * in_plane],
                nb, c_in, h, w,
            );
            debug_assert_eq!((ho, wo), (h_out, w_out));
            let col_count = nb * plane;
            let mut gemm_out = vec![0.0f32; c_out * col_count];
            unsafe {
                gemm(
                    c_out, col_count, k,
                    gemm_out.as_mut_ptr(), 1, col_count as isize,
                    false,
                    w_w.data.as_ptr(), 1, k as isize,
                    cols.as_ptr(), k as isize, 1,
                    0.0, 1.0, false, false, false,
                    Parallelism::Rayon(0),
                );
            }
            let dst = &mut out4d[b0 * c_out * plane..(b0 + nb) * c_out * plane];
            dst.par_chunks_mut(plane).enumerate().for_each(|(plane_idx, plane_dst)| {
                let ib = plane_idx / c_out;
                let oc = plane_idx % c_out;
                let bias = w_b[oc];
                let src_row = oc * col_count + ib * plane;
                for i in 0..plane {
                    plane_dst[i] = gelu(gemm_out[src_row + i] + bias);
                }
            });
        }
        Ok((out4d, h_out, w_out))
    }
}

fn load_conv_weight(weights: &HashMap<String, RawTensor>, name: &str) -> Result<CpuWeightF16> {
    let (data, shape) = weights.get(name).ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))?.as_f16()?;
    let c_out = shape[0];
    let k = shape[1..].iter().product::<usize>();
    Ok(CpuWeightF16 { data, rows: c_out, cols: k })
}

fn load_bias(weights: &HashMap<String, RawTensor>, name: &str) -> Result<Vec<f32>> {
    let (data, _) = weights.get(name).ok_or_else(|| anyhow::anyhow!("bias not found: {}", name))?.as_f32()?;
    Ok(data)
}

pub(crate) struct CpuAudioAttention {
    q_proj: CpuAudioLinear,
    k_proj: CpuAudioLinear,
    v_proj: CpuAudioLinear,
    out_proj: CpuAudioLinear,
    num_heads: usize,
    head_dim: usize,
}

impl CpuAudioAttention {
    pub(crate) fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        nh: usize,
        d_model: usize,
    ) -> Result<Self> {
        let hd = d_model / nh;
        Ok(Self {
            q_proj: CpuAudioLinear::load(weights, &format!("{}.q_proj", prefix))?,
            k_proj: CpuAudioLinear::load(weights, &format!("{}.k_proj", prefix))?,
            v_proj: CpuAudioLinear::load(weights, &format!("{}.v_proj", prefix))?,
            out_proj: CpuAudioLinear::load(weights, &format!("{}.out_proj", prefix))?,
            num_heads: nh,
            head_dim: hd,
        })
    }

    /// x [b, s, d_model] → [b, s, d_model]. ws: window size for attention.
    pub(crate) fn forward(
        &self,
        x: &CpuTensor,
        ws: Option<usize>,
    ) -> Result<CpuTensor> {
        let b = x.shape[0];
        let s = x.shape[1];
        let dm = x.shape[2];
        let nh = self.num_heads;
        let hd = self.head_dim;

        // Project Q, K, V.
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let scale = 1.0f32 / (hd as f32).sqrt();
        let window = ws.filter(|&w| w > 0 && w < s);

        let attn_out = if let Some(w) = window {
            // Windowed: process chunks of `w` tokens, each chunk attends only to itself.
            let mut out = vec![0.0f32; b * nh * s * hd];
            for st in (0..s).step_by(w) {
                let len = w.min(s - st);
                let o = attention_window(&q.data, &k.data, &v.data, b, nh, s, dm, hd, len, st, scale);
                // Scatter: out[(ib, ih, st:st+len, hd)] = o[(ib, ih, 0:len, hd)]
                out.par_chunks_mut(s * hd)
                    .zip(o.par_chunks(len * hd))
                    .for_each(|(dst, src)| {
                        let dst_off = st * hd;
                        dst[dst_off..dst_off + len * hd].copy_from_slice(src);
                    });
            }
            out
        } else {
            // Full attention.
            attention_window(&q.data, &k.data, &v.data, b, nh, s, dm, hd, s, 0, scale)
        };

        // Reshape [b, nh, s, hd] → [b, s, nh*hd] (swap dims 1 and 2).
        let flat = {
            let mut out = vec![0.0f32; b * s * nh * hd];
            out.par_chunks_mut(nh * hd)
                .enumerate()
                .for_each(|(idx, chunk)| {
                    let ib = idx / s;
                    let is_ = idx % s;
                    for ih in 0..nh {
                        let src_off = ((ib * nh + ih) * s + is_) * hd;
                        let dst_off = ih * hd;
                        chunk[dst_off..dst_off + hd].copy_from_slice(&attn_out[src_off..src_off + hd]);
                    }
                });
            out
        };
        let attn_flat = CpuTensor::new(flat, vec![b, s, nh * hd]);
        self.out_proj.forward(&attn_flat)
    }
}

/// Scalar attention with rayon parallelism across (batch, head) pairs.
/// q/k/v: [b, s, dm] flat. Processes positions [st, st+len).
/// Returns [b*nh, len*hd] where each chunk is one (batch, head) pair's output for the window.
fn attention_window(
    q: &[f32], k: &[f32], v: &[f32],
    b: usize, nh: usize, s: usize, dm: usize, hd: usize, len: usize, st: usize, scale: f32,
) -> Vec<f32> {
    let bn = b * nh;
    let chunk_len = len * hd;
    let mut out = vec![0.0f32; bn * chunk_len];
    out.par_chunks_mut(chunk_len)
        .enumerate()
        .for_each(|(idx, out_chunk)| {
            let ib = idx / nh;
            let ih = idx % nh;
            let head_off = ih * hd;
            let mut scores = vec![0.0f32; len];
            for qi in 0..len {
                let q_pos = st + qi;
                let q_base = (ib * s + q_pos) * dm + head_off;
                let mut max_s = f32::NEG_INFINITY;
                for ki in 0..len {
                    let k_pos = st + ki;
                    let k_base = (ib * s + k_pos) * dm + head_off;
                    let mut dot = 0.0f32;
                    for jd in 0..hd {
                        unsafe {
                            dot += *q.get_unchecked(q_base + jd) * *k.get_unchecked(k_base + jd);
                        }
                    }
                    let s_ = dot * scale;
                    scores[ki] = s_;
                    if s_ > max_s { max_s = s_; }
                }
                // Softmax.
                let mut sum = 0.0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - max_s).exp();
                    sum += *sc;
                }
                let inv = 1.0 / sum;
                // Weighted sum of V.
                let out_base = qi * hd;
                for ki in 0..len {
                    let w = scores[ki] * inv;
                    let k_pos = st + ki;
                    let v_base = (ib * s + k_pos) * dm + head_off;
                    for jd in 0..hd {
                        unsafe {
                            *out_chunk.get_unchecked_mut(out_base + jd) += w * *v.get_unchecked(v_base + jd);
                        }
                    }
                }
            }
        });
    out
}

pub(crate) struct CpuAudioFfn {
    fc1: CpuAudioLinear,
    fc2: CpuAudioLinear,
}

impl CpuAudioFfn {
    pub(crate) fn load(weights: &HashMap<String, RawTensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1: CpuAudioLinear::load(weights, &format!("{}.fc1", prefix))?,
            fc2: CpuAudioLinear::load(weights, &format!("{}.fc2", prefix))?,
        })
    }

    pub(crate) fn forward(&self, x: &CpuTensor) -> Result<CpuTensor> {
        let mut h = self.fc1.forward(x)?;
        gelu_inplace(&mut h);
        self.fc2.forward(&h)
    }
}

pub(crate) struct CpuAudioLayer {
    sln: CpuAudioLayerNorm,
    attn: CpuAudioAttention,
    fln: CpuAudioLayerNorm,
    ffn: CpuAudioFfn,
}

impl CpuAudioLayer {
    pub(crate) fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        nh: usize,
        d_model: usize,
    ) -> Result<Self> {
        Ok(Self {
            sln: CpuAudioLayerNorm::load(weights, &format!("{}.self_attn_layer_norm", prefix), 1e-5)?,
            attn: CpuAudioAttention::load(weights, &format!("{}.self_attn", prefix), nh, d_model)?,
            fln: CpuAudioLayerNorm::load(weights, &format!("{}.final_layer_norm", prefix), 1e-5)?,
            ffn: CpuAudioFfn::load(weights, prefix)?,
        })
    }

    /// x: [b, s, d_model] consumed; returns post-residual h of same shape.
    pub(crate) fn forward(&self, x: CpuTensor, ws: Option<usize>) -> Result<CpuTensor> {
        let shape = x.shape.clone();
        let normed = self.sln.forward(&x);
        let attn_out = self.attn.forward(&normed, ws)?;
        // Residual add in-place on x.data (no clone needed — x is consumed).
        let mut x_data = x.data;
        x_data.par_iter_mut().zip(&attn_out.data).for_each(|(a, b)| *a += *b);
        let x1 = CpuTensor::new(x_data, shape);
        let normed2 = self.fln.forward(&x1);
        let ffn_out = self.ffn.forward(&normed2)?;
        let mut x2_data = x1.data;
        x2_data.par_iter_mut().zip(&ffn_out.data).for_each(|(a, b)| *a += *b);
        Ok(CpuTensor::new(x2_data, x1.shape))
    }
}

/// Replicate of `gpu_audio_encoder.rs::feo` (line 389-392).
pub(crate) fn feo(ifr: usize) -> usize {
    let f = |l: usize| -> usize { (l - 1) / 2 + 1 };
    f(f(f(ifr)))
}

pub struct CpuAudioEncoder {
    conv_stem: CpuConvStem,
    layers: Vec<CpuAudioLayer>,
    ln_post: CpuAudioLayerNorm,
    proj1: CpuAudioLinear,
    proj2: CpuAudioLinear,
    config: AudioEncoderConfig,
}

impl CpuAudioEncoder {
    pub fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        config: &AudioEncoderConfig,
    ) -> Result<Self> {
        let dm = config.d_model;
        let nh = config.encoder_attention_heads;
        let mut layers = Vec::with_capacity(config.encoder_layers);
        for i in 0..config.encoder_layers {
            layers.push(CpuAudioLayer::load(weights, &format!("{}.layers.{}", prefix, i), nh, dm)?);
        }
        let ln_post = CpuAudioLayerNorm::load(weights, &format!("{}.ln_post", prefix), 1e-5)?;
        let proj1 = CpuAudioLinear::load(weights, &format!("{}.proj1", prefix))?;
        let proj2 = CpuAudioLinear::load(weights, &format!("{}.proj2", prefix))?;
        let conv_stem = CpuConvStem::load(weights, prefix, config)?;
        Ok(Self { conv_stem, layers, ln_post, proj1, proj2, config: config.clone() })
    }

    /// mel: [n_mels * mel_len] flat (mel-bin-major, frame-minor). Returns [n_total, output_dim] flat.
    pub fn forward(&self, mel: &[f32], n_mels: usize, mel_len: usize) -> Result<Vec<f32>> {
        let cs = self.config.n_window * 2;
        let tpc = feo(cs);
        let nfull = mel_len / cs;
        let tail = mel_len % cs;
        let n_chunks = nfull + if tail > 0 { 1 } else { 0 };

        // Build chunked mel buffer [n_chunks * n_mels * cs], zero-padded.
        let mut chunked = vec![0.0f32; n_chunks * n_mels * cs];
        let mut chunk_tokens: Vec<usize> = Vec::with_capacity(n_chunks);
        for i in 0..nfull {
            let s = i * cs;
            for m in 0..n_mels {
                let dst_base = (i * n_mels + m) * cs;
                let src_base = m * mel_len + s;
                for j in 0..cs {
                    chunked[dst_base + j] = mel[src_base + j];
                }
            }
            chunk_tokens.push(tpc);
        }
        if tail > 0 {
            let s = nfull * cs;
            for m in 0..n_mels {
                let dst_base = (nfull * n_mels + m) * cs;
                let src_base = m * mel_len + s;
                for j in 0..tail {
                    chunked[dst_base + j] = mel[src_base + j];
                }
                // Rest is already zero-padded.
            }
            chunk_tokens.push(feo(tail));
        }

        // Conv stem on batched chunks.
        let (conv_data, t2) = self.conv_stem.forward(&chunked, n_chunks, n_mels, cs)?;
        let dm = self.config.d_model;
        let n_total: usize = chunk_tokens.iter().sum();

        // Pack valid tokens from each chunk into [1, n_total, d_model].
        let mut packed = Vec::with_capacity(n_total * dm);
        for (idx, &v) in chunk_tokens.iter().enumerate() {
            let base = idx * t2 * dm;
            packed.extend_from_slice(&conv_data[base..base + v * dm]);
        }

        // Transformer layers.
        let cs2 = self.config.n_window * 2;
        let tpc2 = feo(cs2);
        let cpw = self.config.n_window_infer / cs2;
        let ws = tpc2 * cpw;
        let mut h = CpuTensor::new(packed, vec![1, n_total, dm]);
        for layer in &self.layers {
            h = layer.forward(h, Some(ws))?;
        }

        let h = self.ln_post.forward(&h);
        let mut h = self.proj1.forward(&h)?;
        gelu_inplace(&mut h);
        let h = self.proj2.forward(&h)?;
        Ok(h.data)
    }

    /// Run conv-stem on a single mel chunk of exactly `cs` frames.
    /// mel_chunk: [n_mels * cs] flat (mel-bin-major). cs must equal n_window * 2.
    /// Returns [tpc, d_model] tokens where tpc = feo(cs).
    pub(crate) fn run_conv_stem(
        &self, mel_chunk: &[f32], n_mels: usize, cs: usize,
    ) -> Result<Vec<f32>> {
        let (conv_data, _t2) = self.conv_stem.forward(mel_chunk, 1, n_mels, cs)?;
        // conv_data is [t2, d_model] — all tokens are valid for a full chunk.
        Ok(conv_data)
    }

    /// Run conv-stem on a partial (tail) mel chunk, zero-padded to `cs` frames.
    /// mel_chunk: [n_mels * actual_frames] flat. actual_frames < cs.
    /// Returns [feo(actual_frames), d_model] tokens.
    pub(crate) fn run_conv_stem_tail(
        &self, mel_chunk: &[f32], n_mels: usize, actual_frames: usize, cs: usize,
    ) -> Result<Vec<f32>> {
        let tpc = feo(actual_frames);
        // Zero-pad to cs frames.
        let mut padded = vec![0.0f32; n_mels * cs];
        for m in 0..n_mels {
            let dst_base = m * cs;
            let src_base = m * actual_frames;
            for j in 0..actual_frames {
                padded[dst_base + j] = mel_chunk[src_base + j];
            }
        }
        let (conv_data, _t2) = self.conv_stem.forward(&padded, 1, n_mels, cs)?;
        // Keep only tpc tokens (discard zero-padded extras).
        let dm = self.config.d_model;
        Ok(conv_data[..tpc * dm].to_vec())
    }

    /// Run transformer layers + final projection on packed tokens.
    /// tokens: [n_tokens, d_model] flat. Returns [n_tokens, output_dim] flat.
    pub(crate) fn run_transformer(&self, tokens: &[f32], n_tokens: usize) -> Result<Vec<f32>> {
        let dm = self.config.d_model;
        let cs2 = self.config.n_window * 2;
        let tpc2 = feo(cs2);
        let cpw = self.config.n_window_infer / cs2;
        let ws = tpc2 * cpw;

        let mut h = CpuTensor::new(tokens.to_vec(), vec![1, n_tokens, dm]);
        for layer in &self.layers {
            h = layer.forward(h, Some(ws))?;
        }
        // Final projections.
        let h = self.ln_post.forward(&h);
        let mut h = self.proj1.forward(&h)?;
        gelu_inplace(&mut h);
        let h = self.proj2.forward(&h)?;
        Ok(h.data)
    }

    /// Access config for streaming chunking parameters.
    pub(crate) fn config(&self) -> &AudioEncoderConfig {
        &self.config
    }
}
