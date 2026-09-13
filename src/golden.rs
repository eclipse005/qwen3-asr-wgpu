//! Loading of the CUDA golden dumps used to validate the port.
//!
//! The dumps are produced by the *reference* backend and are plain little-endian
//! f16 / i32 files plus a `meta.json`.  They live under `golden/<tag>/` and travel
//! with this directory, so the harness keeps working after the crate is moved.
//!
//! Producing them requires the reference crate (it has to read private decoder
//! internals); regenerating is a porting-time activity, consuming them is not.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Meta {
    pub model_dir: String,
    pub wav: String,
    pub tag: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub mrope_section: Vec<usize>,
    pub mrope_interleaved: bool,
    pub audio_start_pos: usize,
    pub nat: usize,
    pub seq_len: usize,
    pub max_seq: usize,
    pub max_new_tokens: usize,
    pub first_token: i32,
    /// EOS that ended generation.  Early goldens lack this field; `#[serde(default)]`
    /// plus the 0-check in `expected_sequence` lets those load with the comparison
    /// truncated to the recorded tokens.
    #[serde(default)]
    pub last_token: i32,
    pub n_decode: usize,
    pub n_dumped_step_logits: usize,
}

pub struct Golden {
    pub dir: PathBuf,
    pub meta: Meta,
    pub hidden: Vec<half::f16>,
    pub cos: Vec<half::f16>,
    pub sin: Vec<half::f16>,
    /// `[L, nkvh, seq_len, hd]`
    pub k_cache: Vec<half::f16>,
    pub v_cache: Vec<half::f16>,
    pub tokens: Vec<i32>,
    pub logits_prefill: Vec<half::f16>,
    /// `[S, vocab]`
    pub step_logits: Vec<half::f16>,
    pub step0: HashMap<String, Vec<half::f16>>,
}

fn read_f16(path: &Path) -> Result<Vec<half::f16>> {
    let b = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(b.chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn read_i32(path: &Path) -> Result<Vec<i32>> {
    let b = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

impl Golden {
    pub fn load(dir: &Path) -> Result<Self> {
        let meta: Meta = serde_json::from_str(
            &std::fs::read_to_string(dir.join("meta.json"))
                .with_context(|| format!("read {}", dir.join("meta.json").display()))?,
        )?;
        let mut step0 = HashMap::new();
        for name in [
            "norm1", "qkv", "q_out", "attn_out", "norm2", "gate_up", "activated", "h_out",
            "final_norm",
        ] {
            let p = dir.join(format!("step0_{name}.bin"));
            if p.exists() {
                step0.insert(name.to_string(), read_f16(&p)?);
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            meta,
            hidden: read_f16(&dir.join("hidden.bin"))?,
            cos: read_f16(&dir.join("cos.bin"))?,
            sin: read_f16(&dir.join("sin.bin"))?,
            k_cache: read_f16(&dir.join("k_cache.bin"))?,
            v_cache: read_f16(&dir.join("v_cache.bin"))?,
            tokens: read_i32(&dir.join("tokens.bin"))?,
            logits_prefill: read_f16(&dir.join("logits_prefill.bin"))?,
            step_logits: read_f16(&dir.join("step_logits.bin"))?,
            step0,
        })
    }

    /// The expected decode sequence: the recorded tokens plus the EOS that ended
    /// generation.  Goldens without a recorded EOS compare only over `tokens`.
    pub fn expected_sequence(&self) -> Vec<i32> {
        let mut v: Vec<i32> = self.tokens.clone();
        if self.meta.last_token != 0 {
            v.push(self.meta.last_token);
        }
        v
    }

    /// One layer's K cache rows for all heads: `[nkvh, seq_len, hd]`.
    pub fn layer_slice<'a>(&self, all: &'a [half::f16], layer: usize) -> &'a [half::f16] {
        let m = &self.meta;
        let per_layer = m.num_key_value_heads * m.seq_len * m.head_dim;
        &all[layer * per_layer..(layer + 1) * per_layer]
    }

    pub fn check_integrity(&self) -> Result<()> {
        let m = &self.meta;
        let expect = |got: usize, want: usize, what: &str| -> Result<()> {
            if got != want {
                return Err(anyhow!(
                    "golden {what}: {got} elements, expected {want} (meta says {}x{}x{}x{})",
                    m.num_hidden_layers,
                    m.num_key_value_heads,
                    m.seq_len,
                    m.head_dim
                ));
            }
            Ok(())
        };
        let per_layer = m.num_key_value_heads * m.seq_len * m.head_dim;
        check_opt(expect, self.k_cache.len(), per_layer * m.num_hidden_layers, "k_cache")?;
        check_opt(expect, self.v_cache.len(), per_layer * m.num_hidden_layers, "v_cache")?;
        check_opt(expect, self.hidden.len(), m.seq_len * m.hidden_size, "hidden")?;
        check_opt(expect, self.cos.len(), m.max_seq * m.head_dim, "cos")?;
        check_opt(
            expect,
            self.step_logits.len(),
            m.n_dumped_step_logits * m.vocab_size,
            "step_logits",
        )?;
        Ok(())
    }
}

fn check_opt(
    f: impl Fn(usize, usize, &str) -> Result<()>,
    got: usize,
    want: usize,
    what: &str,
) -> Result<()> {
    f(got, want, what)
}

/// Max abs difference / exact-match count between two f16 tensors.
pub struct Diff {
    pub max_abs: f32,
    pub n_exact: usize,
    pub n: usize,
    pub first_bad: Option<(usize, f32, f32)>,
}

impl Diff {
    pub fn exact(&self) -> bool {
        self.n_exact == self.n
    }
    pub fn describe(&self, label: &str) -> String {
        let pct = 100.0 * self.n_exact as f64 / self.n.max(1) as f64;
        match self.first_bad {
            Some((i, a, b)) if !self.exact() => format!(
                "{label:<22} max|Δ|={:.6e}  exact {}/{} ({:.2}%)  first@{} ref={:.6e} got={:.6e}",
                self.max_abs, self.n_exact, self.n, pct, i, a, b
            ),
            _ => format!(
                "{label:<22} max|Δ|={:.6e}  exact {}/{} ({:.2}%)",
                self.max_abs, self.n_exact, self.n, pct
            ),
        }
    }
}

pub fn diff_f16(a: &[half::f16], b: &[half::f16]) -> Diff {
    let n = a.len().min(b.len());
    let mut d = Diff { max_abs: 0.0, n_exact: 0, n, first_bad: None };
    for i in 0..n {
        let (x, y) = (a[i].to_f32(), b[i].to_f32());
        if a[i].to_bits() == b[i].to_bits() {
            d.n_exact += 1;
        }
        let e = (x - y).abs();
        if e > d.max_abs {
            d.max_abs = e;
        }
        if a[i].to_bits() != b[i].to_bits() && d.first_bad.is_none() {
            d.first_bad = Some((i, x, y));
        }
    }
    d
}
