use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use safetensors::Dtype;

use crate::gpu::{BulkUpload, Gpu, PipeJob};
use crate::shaders;
use crate::shaders::SubgroupMode;
use crate::weights::{self, PackedWeight};

/// Rows per workgroup for the plain-[`shaders::gemv`] pipelines.
///
/// One warp per row is fixed by the alignment contract (each lane owns words
/// `lane, lane+32, …` of its row and the xor butterfly combines the 32 lanes in
/// a fixed tree).  How many rows *share a workgroup* is not: it only moves the
/// workgroup boundary, so the output is bit-identical at any value and the only
/// thing it can change is how many workgroups the dispatch gets.
///
/// That turns out to change nothing.  `gemv_bench`'s chained sweep (32 dispatches
/// per reading, 0.6B shapes) is flat across 2 / 4 / 8 for every shape and only
/// degrades at 16 / 32:
///
/// | shape | 2 | 4 | **8** | 16 | 32 |
/// |---|---|---|---|---|---|
/// | qkv | 240.3 | 240.4 | **239.6** | 228.8 | 197.7 |
/// | o_proj | 203.6 | 205.2 | **203.3** | 190.1 | 169.1 |
/// | gate_up | 251.7 | 251.6 | **248.8** | 244.9 | 217.4 |
/// | down_proj | 216.8 | 217.7 | **217.3** | 207.4 | 187.1 |
/// | lm_head | n/a | 297.8 | **297.7** | 297.2 | 296.0 |
///
/// The spread that *does* exist is between shapes, not between values: 204 GB/s
/// for `o_proj` against 298 for `lm_head`, i.e. a 1024-row matrix against a
/// 151 936-row one.  It is not the resident warp count either -- 2 rows per
/// workgroup gives 4x the workgroups at the same total warp count -- which is why
/// this is a constant and not a tuning knob.
///
/// **The 82 and 105 GB/s this comment used to quote for `o_proj` / `down_proj`
/// were the submit floor, not the kernels**: they came from the `gemv_bench`
/// helper that submitted once per dispatch (~30-50 us of host time), the same
/// artifact the `rows_per_wg` sweep had.  And the shader's tail hard-coded 8 rows
/// / 4 words instead of reading `RPW`, so every non-8 value wrote `rows/2`
/// workgroups' worth of output for `rows/rpw` computed -- the sweep was timing a
/// kernel with a 4x store overrun at rpw 2.  Both are fixed; the table above is
/// what the kernel actually does.
///
/// `QASR_GEMV_RPW` overrides it so A/B runs on one binary.
pub fn gemv_rpw() -> usize {
    static RPW: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *RPW.get_or_init(|| {
        std::env::var("QASR_GEMV_RPW")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| matches!(v, 2 | 4 | 8 | 16 | 32))
            .unwrap_or(8)
    })
}

/// Warps this part holds at once: 15 SMs x 64.  Not queryable -- `wgpu::Limits`
/// has no SM count -- so it is a documented constant, like the tile shapes.
const WARP_SLOTS: usize = 960;

/// `QASR_RECORD_PAD=n`: how many extra times a decode step is recorded into a
/// throwaway encoder that is never submitted.  Pure CPU work, no GPU work, so it
/// prices the recording half of a step -- see `step`.
fn record_pad() -> Option<usize> {
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("QASR_RECORD_PAD")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n| *n > 0)
    })
}

fn env_split(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(default)
}

/// `QASR_SUBMIT_SPLIT=n` (**default 2**, r34): record a decode step in `n`
/// command buffers and submit each one as it is recorded, so the GPU starts
/// executing the first slice while the CPU is still recording the rest.  The
/// step's 229 dispatches are ~0.4 ms of pure CPU work per token and the GPU waits
/// out every microsecond of
/// it (`QASR_RECORD_PAD` prices that: two extra recordings per step cost +129 ms of
/// decode and +124 ms of it shows up as `host submit`, with both arms MATCHing).
/// Slicing is worth it only if a submit is cheaper than the record it hides, and
/// that is what fixes `n`: **a submit costs ~41 us of token time** (submit
/// +13/+26/+45 for 1/3/7 extra submits) against the ~200 us of record, so `n = 2`
/// is the best measured arm (**-1.6%** on 0.6B/90 s_en, **-1.5%** on 0.6B/180 s_zh,
/// three interleaved reps, both MATCH) and `n = 4` / `n = 8` collect less and less
/// (-1.4%, -0.4%).  `QASR_SUBMIT_SPLIT=1` restores the single-submit step, which
/// is what the A/B above was run against.
///
/// `QASR_PASS_SPLIT=n` (default 1): the same slices as `n` passes of *one* encoder
/// and one submit.  That has none of the overlap and all of the pass-boundary
/// barriers, so running the two knobs separately separates the win from its cost --
/// and it measured **+1.3%** (17 us of CPU and 16 us of GPU a boundary), i.e. the
/// win above is the overlap and not a barrier effect, and any design that adds
/// passes rather than submits is priced dead before it is built.
fn submit_split() -> usize {
    env_split("QASR_SUBMIT_SPLIT", 2)
}

fn pass_split() -> usize {
    env_split("QASR_PASS_SPLIT", 1)
}

/// The `c`th of `n` slices of a decode step: the layers cut as evenly as the
/// integer division allows, the embedding prologue riding on the first slice and
/// the `lm_head`/argmax/token-copy epilogue on the last.  The dispatch sequence
/// does not depend on `n`; only where the pass and submit boundaries fall.
fn step_slice(c: usize, n: usize, layers: usize) -> (usize, usize, bool, bool) {
    (layers * c / n, layers * (c + 1) / n, c == 0, c + 1 == n)
}

/// Rows per *warp* for `o_proj` and `down_proj`, which are the two shapes
/// `rows_per_warp` is worth anything on.
///
/// The dispatch needs `rows / rpwr` warps and the machine holds `WARP_SLOTS`, so
/// the question is only whether the shape crosses one wave.  `gemv_bench` prices
/// both sides of that line: at 1024 rows, **two rows per warp is +5.7% for
/// `o_proj` and +6.1% for `down_proj`** (512 warps, one wave, instead of 1024
/// warps that are 1.07 waves with a 64-warp DRAM-starved tail); at 2048 rows
/// (1.7B) two rows per warp is *worse* -- 1024 warps is still over the line, and
/// all it does is halve the resident warp count.  The corpus says the same thing
/// from the other end: r29 shipped `rpwr = 2` unconditionally and the six 0.6B
/// rows improved (decode -1.4% .. -0.1%) while the six 1.7B rows regressed
/// (+1.9% .. +4.6%), which is exactly this rule read backwards.
///
/// So: use two rows per warp *only when that is what takes the dispatch from two
/// waves to one*.  `QASR_GEMV_RPWR=1|2` forces a value so the two can be priced
/// interleaved; unset = the rule.
fn gemv_rpwr_o_dp(rows: usize) -> usize {
    static R: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let forced = *R.get_or_init(|| {
        std::env::var("QASR_GEMV_RPWR")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| matches!(v, 1 | 2))
            .unwrap_or(0)
    });
    if forced != 0 {
        return forced;
    }
    if rows > WARP_SLOTS && rows / 2 <= WARP_SLOTS {
        2
    } else {
        1
    }
}

/// Rows per workgroup for the [`shaders::gemv_norm`] pipelines -- fixed at 8.
///
/// Its prologue is `rms_norm`'s own 256-virtual-thread reduction tree, so the
/// workgroup must stay 256 threads: shrinking it would have to re-split the
/// partials and that is a different sum.
pub const GEMV_NORM_RPW: usize = 8;

/// Text decoder hyper-parameters (mirrors `TextDecoderConfig`).
#[derive(Debug, Clone)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
}

impl TextConfig {
    /// Read text decoder dims from either the original `thinker_config` layout
    /// or a Transformers-native `-hf` `config.json`.
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let cfg = crate::config::AsrConfig::from_file(&dir.join("config.json"))?;
        let t = &cfg.thinker_config.text_config;
        Ok(Self {
            vocab_size: t.vocab_size,
            hidden_size: t.hidden_size,
            intermediate_size: t.intermediate_size,
            num_hidden_layers: t.num_hidden_layers,
            num_attention_heads: t.num_attention_heads,
            num_key_value_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            rms_norm_eps: t.rms_norm_eps as f32,
        })
    }

    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
    pub fn fused_qkv_cols(&self) -> usize {
        self.q_dim() + 2 * self.kv_dim()
    }
    fn scale(&self) -> f32 {
        1.0f32 / (self.head_dim as f32).sqrt()
    }
}

fn block_for_reduction(last: usize) -> u32 {
    let mut bs: u32 = 32;
    let target = last as u32;
    while bs < target && bs < 1024 {
        bs *= 2;
    }
    bs.min(1024).max(32)
}

/// Physical/workgroup and logical/*tree* widths for the prefill's flat causal
/// softmax.
///
/// `block_for_reduction` grows the block with the row so a row is covered in one
/// or two iterations, which at `cur_len = 1280` means **1024 threads** -- and
/// that is one workgroup per SM (`1024 x ~40 registers` does not fit twice), so
/// the whole card runs 15 rows at a time while each row is a 16-barrier
/// reduction tree with two cold row reads inside it.  `QASR_DUP=p_softmax` prices
/// it at **80 ms of a 90 s_en prefill** (2.3% of a run), thirteen times off its
/// byte bound, and the parallelism that would hide it is four times too low.
///
/// Capping the block at 256 by itself takes the prefill from 512 to 451 ms and
/// **passes all twelve fixtures** (`scripts/bench-sm256.tsv`) -- but it *is* a
/// summation-order change, and narrowing it so only this kernel moved made
/// `1.7B / 180s_zh` MISMATCH (r22).  Two perturbation classes at ~1e-7 in
/// different kernels happened to cancel in the first run: that is a razor's
/// edge, not a measurement.
///
/// So the shipped form keeps the **tree width exactly as it was** and only moves
/// the *physical* width: [`softmax_causal_tree`] has each of the 128 threads
/// emulate `tree/128` lanes and fold the tree's first levels inside the thread,
/// in the tree's own operand order.  That is bit-identical by construction, so
/// the gate is a bug detector here rather than an arbiter, and the physical width
/// becomes a legal knob (`QASR_SM_BS`, one of [`SM_BS_CHOICES`]).
///
/// The sweep behind `128` (0.6B, median of three interleaved reps, prefill ms):
///
/// | phys | 90s_en (tree 1024) | 180s_zh (tree 1024) |
/// |---|---|---|
/// | 256 | 461 / 460 | - |
/// | **128** | **455** | **998** |
/// | 64 | 452 | 1005 |
///
/// 64 and 128 are a wash -- each wins one fixture by 0.7%, which is the
/// resolution of the method -- and both are 1-2% ahead of 256.  128 is shipped
/// because it is ahead where the softmax's share is largest and because it never
/// needs to emulate more than 8 lanes (`tree/128`), where 64 needs 16.
fn softmax_blocks(last: usize) -> (u32, u32) {
    let tree = block_for_reduction(last);
    let forced: u32 = std::env::var("QASR_SM_BS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let phys = if forced != 0 {
        assert!(
            SM_BS_CHOICES.contains(&forced),
            "QASR_SM_BS={forced}: only {SM_BS_CHOICES:?} have pipelines built \
             (`softmax_wide`); a width with no pipeline for this row's tree would \
             otherwise fall back to a *different* tree, which is the r22 mistake"
        );
        forced
    } else {
        SOFTMAX_BS_PHYS
    };
    if tree <= phys {
        (tree, tree)
    } else {
        (phys, tree)
    }
}

/// Physical workgroup width for the flat softmax; see [`softmax_blocks`].
const SOFTMAX_BS_PHYS: u32 = 128;

/// The physical widths `softmax_wide` has pipelines for, for every `tree` the
/// row-length ladder can produce (256, 512, 1024).  `softmax_blocks` asserts on
/// anything else rather than silently narrowing the tree.
const SM_BS_CHOICES: [u32; 3] = [64, 128, 256];

pub(crate) fn grid_xy(workgroups: usize) -> (u32, u32) {
    let gx = workgroups.clamp(1, 65_535) as u32;
    let gy = workgroups.div_ceil(gx as usize).max(1) as u32;
    (gx, gy)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsCfg {
    eps: f32,
    _a: f32,
    _b: f32,
    _c: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QkvxCfg {
    max_seq: u32,
    start: u32,
    pos_offset: u32,
    s: u32,
    eps: f32,
    _a: u32,
    _b: u32,
    _c: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GDims {
    m: u32,
    n: u32,
    k: u32,
    ldc: u32,
    bsa: u32,
    bsb: u32,
    bsc: u32,
    beta: u32,
    row0: u32,
    lda: u32,
    q_off: u32,
    gqa_group: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SiluCfg {
    inter2: u32,
    total2: u32,
    gx: u32,
    _b: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RepeatKvCfg {
    nkvh: u32,
    max_seq: u32,
    cur: u32,
    hd: u32,
    npw: u32,
    gx: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxCfg {
    n_w: u32,
    n_x: u32,
    valid: u32,
    m: u32,
    mp: u32,
    scale: f32,
    gx: u32,
    row0: u32,
    q_off: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SlabStatsCfg {
    n_x: u32,
    valid: u32,
    m: u32,
    mp: u32,
    scale: f32,
    row0: u32,
    gx: u32,
    rows: u32,
    q_off: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SlabWeightsCfg {
    rows: u32,
    n_slab: u32,
    gx: u32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SlabMergeCfg {
    rows: u32,
    n_slab: u32,
    gx: u32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GqaCfg {
    cur_len: u32,
    max_seq: u32,
    scale: f32,
    _p: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SplitCfg {
    cur_len: u32,
    max_seq: u32,
    scale: f32,
    n_chunks: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MergeCfg {
    n_chunks: u32,
    _a: u32,
    _b: u32,
    _c: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ArgmaxCfg {
    n: u32,
    slot: u32,
    _a: u32,
    _b: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EmbedCfg {
    slot: u32,
    d2: u32,
    _a: u32,
    _b: u32,
}

/// Grow-only buffers for prefill.  Uniforms and small buffers are fixed-size;
/// every token-sized allocation grows only when a longer clip needs it.
struct PrefillScratch {
    u_gd: wgpu::Buffer,
    u_sl: wgpu::Buffer,
    cap_mp: usize,
    h_buf: wgpu::Buffer,
    normed: wgpu::Buffer,
    norm2: wgpu::Buffer,
    qkv: wgpu::Buffer,
    q_out: wgpu::Buffer,
    attn_flat: wgpu::Buffer,
    gu: wgpu::Buffer,
    activated: wgpu::Buffer,
    cap_scores_bytes: u64,
    scores: wgpu::Buffer,
    cap_attn_bytes: u64,
    attn: wgpu::Buffer,
    slab: Option<SlabScratch>,
}

struct SlabScratch {
    slab_part: wgpu::Buffer,
    slab_stats: wgpu::Buffer,
    slab_w: wgpu::Buffer,
    cap_part_bytes: u64,
    cap_stats_bytes: u64,
    cap_w_bytes: u64,
}

/// Grow-only buffers for the batched prefill body.
struct MultiPrefillScratch {
    u_gd: wgpu::Buffer,
    u_seg: wgpu::Buffer,
    u_sl: wgpu::Buffer,
    cap_mp: usize,
    h_buf: wgpu::Buffer,
    normed: wgpu::Buffer,
    norm2: wgpu::Buffer,
    qkv: wgpu::Buffer,
    attn_flat: wgpu::Buffer,
    gu: wgpu::Buffer,
    activated: wgpu::Buffer,
    cap_mp_attn: usize,
    q_out: wgpu::Buffer,
    cap_scores_bytes: u64,
    scores: wgpu::Buffer,
    cap_attn_bytes: u64,
    attn: wgpu::Buffer,
}

impl PrefillScratch {
    fn new(gpu: &Gpu) -> Self {
        Self {
            u_gd: gpu.uniform("p.gd", (256 * 256) as u64),
            u_sl: gpu.uniform("p.sl", (2 * MAX_SLAB as u64 + 2) * 256),
            cap_mp: 0,
            h_buf: gpu.storage("p.h", 16),
            normed: gpu.storage("p.normed", 16),
            norm2: gpu.storage("p.norm2", 16),
            qkv: gpu.storage("p.qkv", 16),
            q_out: gpu.storage("p.q_out", 16),
            attn_flat: gpu.storage("p.attn_flat", 16),
            gu: gpu.storage("p.gu", 16),
            activated: gpu.storage("p.activated", 16),
            cap_scores_bytes: 0,
            scores: gpu.storage("p.scores", 16),
            cap_attn_bytes: 0,
            attn: gpu.storage("p.attn", 16),
            slab: None,
        }
    }

    fn storage(gpu: &Gpu, label: &str, buf: &mut wgpu::Buffer, cap: &mut u64, need: u64) {
        if need > *cap {
            *buf = gpu.storage(label, need);
            *cap = need;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ensure(
        &mut self,
        gpu: &Gpu,
        words: &impl Fn(usize, usize) -> u64,
        hs: usize,
        fused: usize,
        qh: usize,
        hd: usize,
        inter: usize,
        mp: usize,
        score_words: u64,
        attn_words: u64,
        slab: Option<(u64, u64, u64)>,
    ) {
        if mp > self.cap_mp {
            let storages: [(&str, &mut wgpu::Buffer, u64); 8] = [
                ("p.h", &mut self.h_buf, words(mp, hs)),
                ("p.normed", &mut self.normed, words(mp, hs)),
                ("p.norm2", &mut self.norm2, words(mp, hs)),
                ("p.qkv", &mut self.qkv, words(mp, fused)),
                ("p.q_out", &mut self.q_out, words(qh * mp, hd)),
                ("p.attn_flat", &mut self.attn_flat, words(mp, qh * hd)),
                ("p.gu", &mut self.gu, words(mp, 2 * inter)),
                ("p.activated", &mut self.activated, words(mp, inter)),
            ];
            for (label, buf, need) in storages {
                *buf = gpu.storage(label, need);
            }
            self.cap_mp = mp;
        }
        Self::storage(gpu, "p.scores", &mut self.scores, &mut self.cap_scores_bytes, score_words);
        Self::storage(gpu, "p.attn", &mut self.attn, &mut self.cap_attn_bytes, attn_words);
        if let Some((part, stats, w)) = slab {
            let s = self.slab.get_or_insert_with(|| SlabScratch {
                slab_part: gpu.storage("p.slab_part", 16),
                slab_stats: gpu.storage("p.slab_stats", 16),
                slab_w: gpu.storage("p.slab_w", 16),
                cap_part_bytes: 0,
                cap_stats_bytes: 0,
                cap_w_bytes: 0,
            });
            Self::storage(gpu, "p.slab_part", &mut s.slab_part, &mut s.cap_part_bytes, part);
            Self::storage(gpu, "p.slab_stats", &mut s.slab_stats, &mut s.cap_stats_bytes, stats);
            Self::storage(gpu, "p.slab_w", &mut s.slab_w, &mut s.cap_w_bytes, w);
        }
    }
}

impl MultiPrefillScratch {
    fn new(gpu: &Gpu) -> Self {
        Self {
            u_gd: gpu.uniform("pb.gd", (512 * 256) as u64),
            u_seg: gpu.uniform("pb.seg", 32 * 256),
            u_sl: gpu.uniform("pb.sl", 32 * 256),
            cap_mp: 0,
            h_buf: gpu.storage("pb.h", 16),
            normed: gpu.storage("pb.normed", 16),
            norm2: gpu.storage("pb.norm2", 16),
            qkv: gpu.storage("pb.qkv", 16),
            attn_flat: gpu.storage("pb.attn_flat", 16),
            gu: gpu.storage("pb.gu", 16),
            activated: gpu.storage("pb.activated", 16),
            cap_mp_attn: 0,
            q_out: gpu.storage("pb.q_out", 16),
            cap_scores_bytes: 0,
            scores: gpu.storage("pb.scores", 16),
            cap_attn_bytes: 0,
            attn: gpu.storage("pb.attn", 16),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ensure(
        &mut self,
        gpu: &Gpu,
        words: &impl Fn(usize, usize) -> u64,
        hs: usize,
        fused: usize,
        nqh: usize,
        hd: usize,
        inter: usize,
        mp: usize,
        mp_attn: usize,
        score_words: u64,
        attn_words: u64,
    ) {
        if mp > self.cap_mp {
            let storages: [(&str, &mut wgpu::Buffer, u64); 7] = [
                ("pb.h", &mut self.h_buf, words(mp, hs)),
                ("pb.normed", &mut self.normed, words(mp, hs)),
                ("pb.norm2", &mut self.norm2, words(mp, hs)),
                ("pb.qkv", &mut self.qkv, words(mp, fused)),
                ("pb.attn_flat", &mut self.attn_flat, words(mp, nqh * hd)),
                ("pb.gu", &mut self.gu, words(mp, 2 * inter)),
                ("pb.activated", &mut self.activated, words(mp, inter)),
            ];
            for (label, buf, need) in storages {
                *buf = gpu.storage(label, need);
            }
            self.cap_mp = mp;
        }
        if mp_attn > self.cap_mp_attn {
            self.q_out = gpu.storage("pb.q_out", words(nqh * mp_attn, hd));
            self.cap_mp_attn = mp_attn;
        }
        PrefillScratch::storage(gpu, "pb.scores", &mut self.scores, &mut self.cap_scores_bytes, score_words);
        PrefillScratch::storage(gpu, "pb.attn", &mut self.attn, &mut self.cap_attn_bytes, attn_words);
    }
}

/// Buffers reused by every layer.
pub struct Scratch {
    pub h: wgpu::Buffer,
    pub norm1: wgpu::Buffer,
    pub qkv: wgpu::Buffer,
    pub q_out: wgpu::Buffer,
    pub attn_out: wgpu::Buffer,
    pub norm2: wgpu::Buffer,
    pub gate_up: wgpu::Buffer,
    pub activated: wgpu::Buffer,
    pub final_norm: wgpu::Buffer,
    pub logits: wgpu::Buffer,
    pub token: wgpu::Buffer,
    token_staging: wgpu::Buffer,
    pub cos: wgpu::Buffer,
    pub sin: wgpu::Buffer,
    u_rms: wgpu::Buffer,
    u_qkvx: wgpu::Buffer,
    u_gqa: wgpu::Buffer,
    u_gp1: wgpu::Buffer,
    u_gp2: wgpu::Buffer,
    u_silu: wgpu::Buffer,
    /// Split-K attention partials, `[nqh, max_chunks, …]` — f32.
    pub split_part_out: wgpu::Buffer,
    pub split_part_max: wgpu::Buffer,
    pub split_part_sum: wgpu::Buffer,
}

struct Layer {
    k_cache: wgpu::Buffer,
    v_cache: wgpu::Buffer,
    iln_w: wgpu::Buffer,
    pln_w: wgpu::Buffer,
    qn_w: wgpu::Buffer,
    kn_w: wgpu::Buffer,
    qkv_w: wgpu::Buffer,
    o_w: wgpu::Buffer,
    gu_w: wgpu::Buffer,
    dp_w: wgpu::Buffer,
    /// The per-row f32 scale buffers for the W8 kernels — `Some` iff the
    /// checkpoint is int8 (see [`LayerScales`]); the f16 kernels never see them.
    qkv_s: Option<wgpu::Buffer>,
    o_s: Option<wgpu::Buffer>,
    gu_s: Option<wgpu::Buffer>,
    dp_s: Option<wgpu::Buffer>,
    bg_gemv_qkv: wgpu::BindGroup,
    bg_gemv_o: wgpu::BindGroup,
    /// The same buffers as `bg_gemv_o` / `bg_gemv_dp`, bound against the
    /// `_rpwr2` pipelines: those were built with `None`, so wgpu gave each
    /// pipeline its own layout object and a bind group made from one is rejected
    /// by the other ("Exclusive pipelines don't match").
    bg_gemv_o_rpwr2: wgpu::BindGroup,
    bg_gemv_dp_rpwr2: wgpu::BindGroup,
    bg_gemv_gu: wgpu::BindGroup,
    bg_gemv_dp: wgpu::BindGroup,
    bg_gemv_qkv_norm: wgpu::BindGroup,
    bg_gemv_gu_norm: wgpu::BindGroup,
    /// The int8 forms of the six decode GEMVs, bound against the `_w8`
    /// pipelines with the scale buffers at their extra binding.  Built only for
    /// an int8 checkpoint; the dispatch sites pick by [`WgpuTextDecoder::int8`].
    bg_gemv_w8_qkv: Option<wgpu::BindGroup>,
    bg_gemv_w8_gu: Option<wgpu::BindGroup>,
    bg_gemv_w8_qkv_norm: Option<wgpu::BindGroup>,
    bg_gemv_w8_o: Option<wgpu::BindGroup>,
    bg_gemv_w8_o_rpwr2: Option<wgpu::BindGroup>,
    bg_gemv_w8_gu_norm: Option<wgpu::BindGroup>,
    bg_gemv_w8_dp: Option<wgpu::BindGroup>,
    bg_gemv_w8_dp_rpwr2: Option<wgpu::BindGroup>,
    bg_rms1: wgpu::BindGroup,
    bg_rms2: wgpu::BindGroup,
    bg_extract: wgpu::BindGroup,
    bg_gqa: wgpu::BindGroup,
    bg_gqa_split: wgpu::BindGroup,
    bg_gqa_merge: wgpu::BindGroup,
}

struct Pipes {
    gemv_qkv: wgpu::ComputePipeline,
    gemv_o: wgpu::ComputePipeline,
    gemv_gu: wgpu::ComputePipeline,
    gemv_dp: wgpu::ComputePipeline,
    /// `gemv_o` / `gemv_dp` with two rows per warp; see [`gemv_rpwr_o_dp`].
    gemv_o_rpwr2: wgpu::ComputePipeline,
    gemv_dp_rpwr2: wgpu::ComputePipeline,
    gemv_lm: wgpu::ComputePipeline,
    gemv_qkv_norm: wgpu::ComputePipeline,
    gemv_gu_norm: wgpu::ComputePipeline,
    gemv_lm_norm: wgpu::ComputePipeline,
    /// The int8-resident forms (`shaders::gemv_w8*`, `shaders::prefill_gemm_w8`):
    /// same granule/lane/reduction structure as the f16 kernels, half the weight
    /// bytes on the bus, per-row scale bound alongside.  Built for every model —
    /// an f16 checkpoint never dispatches them, and the compile cost is what
    /// [`load_trace`] can price if it ever matters.
    gemv_w8_o: wgpu::ComputePipeline,
    gemv_w8_o_rpwr2: wgpu::ComputePipeline,
    gemv_w8_dp: wgpu::ComputePipeline,
    gemv_w8_dp_rpwr2: wgpu::ComputePipeline,
    gemv_w8_qkv: wgpu::ComputePipeline,
    gemv_w8_gu: wgpu::ComputePipeline,
    gemv_w8_qkv_norm: wgpu::ComputePipeline,
    gemv_w8_gu_norm: wgpu::ComputePipeline,
    gemm_w8: wgpu::ComputePipeline,
    gemm_w8_acc: wgpu::ComputePipeline,
    rms_norm: wgpu::ComputePipeline,
    extract: wgpu::ComputePipeline,
    gqa256: wgpu::ComputePipeline,
    gqa512: wgpu::ComputePipeline,
     gqa_split128: wgpu::ComputePipeline,
    /// Same phase with the REP q heads that share a kv head done in one pass.
    gqa_split_pair256: wgpu::ComputePipeline,
    gqa_split_pair512: wgpu::ComputePipeline,
    /// Diagnostic: `gqa_split_pair256` with every K read aimed at row 0, so all
    /// loads are L1 hits.  Only reachable through `QASR_DUP`, never on the
    /// default path.
    gqa_split_pair256_row0: wgpu::ComputePipeline,
    /// Diagnostic: `gqa_split_pair256` with the K reads re-aimed at consecutive
    /// addresses across lanes (what a dim-major K layout would look like).  Only
    /// reachable through `QASR_DUP`.
    gqa_split_pair256_coal: wgpu::ComputePipeline,
    gqa_split256: wgpu::ComputePipeline,
    gqa_split512: wgpu::ComputePipeline,
    gqa_merge: wgpu::ComputePipeline,
    silu: wgpu::ComputePipeline,
    argmax: wgpu::ComputePipeline,
    embed: wgpu::ComputePipeline,
    gemm: wgpu::ComputePipeline,
    gemm_acc: wgpu::ComputePipeline,
    gemm_av: wgpu::ComputePipeline,
    gemm_causal: wgpu::ComputePipeline,
    gemm_av_causal: wgpu::ComputePipeline,
    gemm_64: wgpu::ComputePipeline,
    gemm_acc_64: wgpu::ComputePipeline,
    gemm_av_64: wgpu::ComputePipeline,
    gemm_causal_64: wgpu::ComputePipeline,
    gemm_av_causal_64: wgpu::ComputePipeline,
    gemm_w8_64: wgpu::ComputePipeline,
    gemm_w8_acc_64: wgpu::ComputePipeline,
    softmax: std::collections::HashMap<usize, wgpu::ComputePipeline>,
    /// Narrower physical workgroups computing a wider tree, keyed by
    /// `(phys, tree)`; see [`softmax_blocks`] and `softmax_causal_tree`.
    softmax_wide: std::collections::HashMap<(usize, usize), wgpu::ComputePipeline>,
    slab_stats: wgpu::ComputePipeline,
    slab_weights: wgpu::ComputePipeline,
    slab_merge: wgpu::ComputePipeline,
}

/// The three bind groups that read a layer's KV cache: the write side
/// (`extract`, bindings 6/7) and the two attention readers (`gqa` 1/2,
/// `gqa_split` 1/2).  Every other bind group a layer owns is KV-independent, so
/// reallocating the cache means rebuilding exactly these — see
/// [`WgpuTextDecoder::ensure_capacity`].
fn kv_bind_groups(
    gpu: &Gpu,
    pipes: &Pipes,
    scratch: &Scratch,
    qn_w: &wgpu::Buffer,
    kn_w: &wgpu::Buffer,
    k_cache: &wgpu::Buffer,
    v_cache: &wgpu::Buffer,
) -> (wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup) {
    kv_bind_groups_res(
        gpu,
        pipes,
        scratch,
        qn_w,
        kn_w,
        scratch.q_out.as_entire_binding(),
        k_cache.as_entire_binding(),
        v_cache.as_entire_binding(),
        scratch.attn_out.as_entire_binding(),
    )
}

/// [`kv_bind_groups`] with the four attention tensors given as binding
/// resources, so a pooled slot can bind a `BufferBinding` slice of a shared
/// allocation instead of a whole buffer.  Offsets of 0 with no size are the
/// ordinary whole-buffer case.
#[allow(clippy::too_many_arguments)]
fn kv_bind_groups_res(
    gpu: &Gpu,
    pipes: &Pipes,
    scratch: &Scratch,
    qn_w: &wgpu::Buffer,
    kn_w: &wgpu::Buffer,
    q_out: wgpu::BindingResource<'_>,
    k_cache: wgpu::BindingResource<'_>,
    v_cache: wgpu::BindingResource<'_>,
    attn_out: wgpu::BindingResource<'_>,
) -> (wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup) {
    let bg_extract = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("qkv_extract"),
        layout: &pipes.extract.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: scratch.qkv.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: qn_w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: kn_w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: scratch.cos.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: scratch.sin.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: q_out.clone() },
            wgpu::BindGroupEntry { binding: 6, resource: k_cache.clone() },
            wgpu::BindGroupEntry { binding: 7, resource: v_cache.clone() },
            wgpu::BindGroupEntry { binding: 8, resource: scratch.u_qkvx.as_entire_binding() },
        ],
    });
    let bg_gqa = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gqa"),
        layout: &pipes.gqa256.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: scratch.q_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: k_cache.clone() },
            wgpu::BindGroupEntry { binding: 2, resource: v_cache.clone() },
            wgpu::BindGroupEntry { binding: 3, resource: attn_out.clone() },
            wgpu::BindGroupEntry { binding: 4, resource: scratch.u_gqa.as_entire_binding() },
        ],
    });
    let bg_gqa_split = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gqa_split"),
        layout: &pipes.gqa_split256.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: scratch.q_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: k_cache.clone() },
            wgpu::BindGroupEntry { binding: 2, resource: v_cache.clone() },
            wgpu::BindGroupEntry { binding: 3, resource: scratch.split_part_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: scratch.split_part_max.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: scratch.split_part_sum.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: scratch.u_gp1.as_entire_binding() },
        ],
    });
    (bg_extract, bg_gqa, bg_gqa_split)
}

/// Capacity of the single-block attention path.
///
/// Above this `cur_len` attention switches to split-K plus a merge dispatch.
/// The crossover was re-measured with the single path widened to 4096 (same
/// shader, bigger score buffer, byte-identical output): 90 s_en 22.32 vs 23.33
/// RTFx and 180 s_en 18.85 vs 20.87, so the split path's parallelism is worth
/// more than the merge dispatch it costs, and 1024 is the right side of the
/// crossover.
pub const GQA_SINGLE_CAP: usize = 1024;

/// Longest decode KV the batched path will allocate per slot.
///
/// The single attention kernel is capped at [`GQA_SINGLE_CAP`]; above that the
/// batched step runs the same chunked pair + merge the shipped step picks, per
/// slot.  What bounds this instead is memory: a slot's KV is
/// `nkvh * cap * hd * 2 * 2` bytes per layer, i.e. ~470 MiB at 4096 over 28
/// layers, and every slot holds one.
pub const BATCH_SEQ_CAP: usize = 4096;

/// One block of a decoder layer's dispatch sequence, in the order a step issues
/// them.
///
/// The batch driver walks `LAYER_BLOCKS` outermost and slots innermost, so the
/// weight tile one slot pulls into L2 is read by every other slot before the
/// next block replaces it.  A slot's own order is unchanged — `encode_into` with
/// `want = |_| true` issues exactly these seven blocks in this order — which is
/// what keeps a batched token stream bit-identical to a lone one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerBlock {
    Qkv,
    Extract,
    Gqa,
    O,
    Gu,
    Silu,
    Dp,
}

/// [`LayerBlock`] in issue order.
pub const LAYER_BLOCKS: [LayerBlock; 7] = [
    LayerBlock::Qkv,
    LayerBlock::Extract,
    LayerBlock::Gqa,
    LayerBlock::O,
    LayerBlock::Gu,
    LayerBlock::Silu,
    LayerBlock::Dp,
];

/// KV slots allocated at load (112 KiB each, see [`WgpuTextDecoder::load`]).
///
/// `max_seq` is the ceiling, not the allocation: the cache grows from here on
/// demand ([`WgpuTextDecoder::ensure_capacity`]), so a 15-minute request does not
/// make every 15-second clip pay for the 16 384 slots it might need.
pub const KV_INITIAL_CAP: usize = 1024;

/// Capacity grows in multiples of this — a series of clips of one shape then
/// allocates once and pays nothing after that.
pub const KV_STEP: usize = 256;

const SLAB_T: usize = 1024;

const SLAB_BS: usize = 256;

const MAX_SLAB: usize = 16;

/// Slab (bounded-row) tiled prefill attention, or the flat one.
///
/// The two differ in the row width the causal softmax walks: the flat path's is
/// `np/2` words and grows with the prefill, the slab path's is fixed at
/// `slab_t/2 = 512`.  So the crossover is structural at `np/2 = 512`, i.e.
/// `s ~ 900`, and the slab path is the *faster* one above it.  Interleaved A/B,
/// 0.6B, 3 reps, every arm MATCH, prefill ms:
///
/// | s | flat | slab | |
/// |---|---|---|---|
/// | 195 | 145 | 158 | +9.0% |
/// | 407 | 266 | 287 | +7.9% |
/// | 1185 | 720 | **678** | -5.8% |
/// | 2307 | 1541 | **1510** | -2.0% |
///
/// **And it is not a legal default.**  Lowering this threshold to match the
/// crossover was measured on the full gate and the gate rejected it:
/// `1.7B / 180s_zh` MISMATCH (every 0.6B fixture and `1.7B / 180s_en` MATCH, so
/// it is the same signature as `QASR_GQA_COOP`).  The slab softmax is an online
/// one -- per-slab (max, sum) then a rescaled merge -- so its rounding is not
/// the flat path's, and that is inherent to it rather than a bug to fix.
///
/// Diffing the two transcripts says how far apart they are: **the text is
/// character-for-character identical except for punctuation**, three marks
/// where the flat path emits `，`/`。` and the slab path emits `！`.  That is a
/// rounding-level divergence flipping the argmax between two tokens whose
/// logits differ by a hair -- not a structural defect, but enough to fail a
/// byte-exact gate.
///
/// Which leaves a hole worth naming: **`s > 4096` is the default and it is
/// ungated.**  No fixture is that long, so the slab path's numerics have never
/// been checked against `python-hf` anywhere it is actually used -- and what it
/// would do there is exactly what it just did here, i.e. differ in punctuation.
/// The threshold stays at 4096 until either the slab path is made to agree or a
/// long fixture exists; do not lower it to collect the 42 ms.
/// Use the warp-shuffle tail in the prefill softmax's two reduction trees
/// instead of the last five smem rounds.  Each tree crosses warps above distance
/// 32 and stays inside one warp below it, so those levels become
/// `subgroupShuffleXor` -- same operands, same order, same bits -- and the
/// barriers per row go from 20 to 10.  On by default; `QASR_SM_SUBGROUP=0`
/// restores the smem tree so the two can be A/B'd from one binary.
///
/// Worth 4.7% of the prefill because that kernel is 18 960 one-row workgroups
/// and therefore barrier-latency-bound.  **The same surgery on the decode split
/// kernel's stage 2/3 trees was measured at -0.2% on the decode step** and
/// reverted: 8 barriers per stage down to 3 bought 6 ms of 2622, because that
/// kernel has few workgroups with a lot of work each, so its barriers are
/// already amortised.  Same tree, same trick, no transfer -- the mechanism is
/// the workgroup count and the work per workgroup, not the barrier count
/// itself.
fn sg_reduce() -> bool {
    static SG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SG.get_or_init(|| {
        !matches!(
            std::env::var("QASR_SM_SUBGROUP").unwrap_or_default().to_ascii_lowercase().as_str(),
            "0" | "off" | "no"
        )
    })
}
fn slab_path(s: usize) -> bool {
    match std::env::var("QASR_SLAB").unwrap_or_default().to_ascii_lowercase().as_str() {
        "1" | "on" | "yes" | "force" => true,
        "0" | "off" | "no" => false,
        _ => s > 4096,
    }
}

/// Split-attention chunk width.
///
/// The chunk sets both the workgroup count (one per `(q_head, chunk)`, so
/// `nh x ceil(cur_len/chunk)`) and how many partials the merge dispatch has to
/// combine.  `QASR_GQA_CHUNK` pins it to one of the two built kernels so the
/// granularity can be A/B'd from a single binary.
/// Source for the paired split kernel, falling back to the per-q-head one when
/// the config does not have exactly two q heads per kv head.  The paired kernel
/// assumes `REP == 2`; anything else keeps the old (correct, just 2x the KV
/// traffic) path so a different checkpoint cannot silently break.
fn pair_split_src(nqh: usize, nkvh: usize, hd: usize, chunk: usize, row0: bool, coop: bool, pf: bool, coal: bool, depth: usize) -> String {
    if nqh / nkvh == 2 {
        shaders::gqa_decode_split_p1_pair(hd, chunk, 2, row0, coop, pf, coal, depth)
    } else {
        shaders::gqa_decode_split_p1(nqh, nkvh, hd, chunk)
    }
}

/// How many K-row words stage 1 issues before consuming any of them.
///
/// **Default 1, i.e. the shipped `QASR_GQA_PF` form, and that is the engine's
/// verdict, not a placeholder.**  `attn_bench` said unrolling the window was
/// worth a great deal in isolation -- at a 2560-token context, chunk 512,
/// stage 1 measured 96.4 us/layer with one word of slack, 54.8 with four and
/// 55.4 with eight, and the L1-resident and coalesced arms of the same probe
/// bracket that as most of the kernel's cost.  Built into the engine and
/// A/B'd interleaved on 0.6B / 180 s_en (3 reps, all arms MATCH), it is a
/// consistent *loss*:
///
/// | depth | decode ms | vs depth 1 |
/// |---|---|---|
/// | 1 (shipped) | 5692 / 5698 / 5708 | -- |
/// | 2 | 5793 | +1.5% |
/// | 4 | 5783 | +1.6% |
/// | 8 | 5799 | +1.8% |
///
/// Flat in the window width and never positive, so it is a fixed cost (register
/// pressure / scheduling), not a missing-MLP effect: the probe's absolute
/// calibration does not survive the trip into the real kernel, where stage 1
/// sits next to stages 2-4 and the whole layer's state.  The switch is kept so
/// the negative result can be re-checked rather than re-derived, and the probe
/// is kept as an instrument -- but the probe's stage-1 numbers must not be
/// quoted as engine costs.
fn gqa_depth() -> usize {
    static DEPTH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *DEPTH.get_or_init(|| {
        std::env::var("QASR_GQA_DEPTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| (1..=16).contains(v))
            .unwrap_or(1)
    })
}

/// Builds the warp-cooperative stage 1 instead of the per-lane-row one.
///
/// **Off by default, and it must stay that way.**  It is correct arithmetic and
/// MATCHes all six 0.6 B fixtures, but it re-associates the dot product (an
/// 8-lane xor tree instead of the serial chain) and that flips a token on
/// `1.7B / 90s_en` — so it fails the transcript gate and is not a legal default.
/// It is kept behind the switch because it is half of the measurement that
/// located attention's cost: it buys +1.3% where removing the K traffic entirely
/// buys 69%, which is what rules out traffic and lines and points at load
/// latency.  `QASR_GQA_COOP=1` opts in.
fn gqa_coop() -> bool {
    static COOP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *COOP.get_or_init(|| {
        matches!(
            std::env::var("QASR_GQA_COOP").unwrap_or_default().to_ascii_lowercase().as_str(),
            "1" | "on" | "yes"
        )
    })
}

/// Stage 1 issues its `KC4`/`Q4` loads one word early.  The FMA sequence and
/// every operand are unchanged, so this is bit-identical by construction -- it
/// only tests whether load latency is what the kernel waits on, which is the one
/// hypothesis the traffic/lines/chain/occupancy experiments left standing.
/// Measured +0.9% (0.6B / 180 s_en, interleaved 3 reps) and gated 12/12, so it is
/// on by default; `QASR_GQA_PF=0` turns it off.
fn gqa_pf() -> bool {
    static PF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PF.get_or_init(|| {
        !matches!(
            std::env::var("QASR_GQA_PF").unwrap_or_default().to_ascii_lowercase().as_str(),
            "0" | "off" | "no"
        )
    })
}

/// Entry point matching [`pair_split_src`].
fn pair_split_entry(nqh: usize, nkvh: usize) -> &'static str {
    if nqh / nkvh == 2 {
        "gqa_split_p1_pair"
    } else {
        "gqa_split_p1"
    }
}

fn gqa_split_chunk(cur_len: usize) -> usize {
    static FORCED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let forced = *FORCED.get_or_init(|| {
        std::env::var("QASR_GQA_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| matches!(v, 128 | 256 | 512))
            .unwrap_or(0)
    });
    if forced != 0 {
        return forced;
    }
    if cur_len >= 2048 {
        512
    } else {
        256
    }
}

/// One decode run's per-step hidden states, read back as raw little-endian f16.
pub struct DecodeStates {
    /// `[rows, hidden]`, row `k` = `scratch.h` at the end of step `k`.
    pub states: Vec<u8>,
    pub rows: usize,
    pub hidden: usize,
    /// `pos` of row 0 (the prompt length): row `k` ran at `base_pos + k`.
    pub base_pos: usize,
}

pub struct WgpuTextDecoder {
    pub gpu: Gpu,
    pub cfg: TextConfig,
    /// The ceiling on `seq_len + max_new_tokens` — fixed at load, and what the
    /// caller's own bounds check reports against.
    pub max_seq: usize,
    /// KV slots actually allocated: [`KV_INITIAL_CAP`] after load, grown on
    /// demand up to `max_seq`.  Every uniform that carries a KV stride uses
    /// this, not the ceiling.
    pub cap: usize,
    /// Positions already in the KV cache — the next step attends at `cur_len = pos + 1`.
    pub pos: usize,
    /// Host-side time accumulated by [`Self::step`]: (uniform + encode + submit,
    /// token readback).  Purely diagnostic — what the batched-submission idea
    /// below would remove.
    pub host_submit_ms: f64,
    pub host_read_ms: f64,

    /// A recorded-but-unsubmitted step, held so the record can overlap the step
    /// in flight -- see `record_step` / `submit_step`.  A *finished* command
    /// buffer, not an encoder: `finish()` validates, and that is host work the
    /// pipeline can only hide by doing it on the recording side of the await.
    pending: Option<wgpu::CommandBuffer>,
    /// The token map armed for the step in flight, taken by `take_token`.
    pending_map: Option<std::sync::mpsc::Receiver<anyhow::Result<()>>>,
    /// KV and the two attention activations live in a [`SlotPool`] at a fixed
    /// cap: growing them would move the rows out from under the shared buffer,
    /// so [`Self::ensure_capacity`] is a no-op for a pooled slot.
    pooled: bool,
    /// Grow-only allocations for the two prefill paths; reused by every call.
    prefill_scratch: Option<PrefillScratch>,
    multi_scratch: Option<MultiPrefillScratch>,

    embed_table: wgpu::Buffer,
    layers: Vec<Layer>,
    pipes: std::sync::Arc<Pipes>,
    /// The checkpoint quantized the decoder's linears (M3): the weights sit in
    /// VRAM as int8 with per-row scales, and every decode/prefill GEMV/GEMM
    /// dispatch goes through the `_w8` kernels.  `embed`/`lm_head` and the
    /// audio tower are unaffected (they are not quantized by the exporter).
    int8: bool,
    pub scratch: Scratch,

    bg_final_rms: wgpu::BindGroup,
    bg_gemv_lm: wgpu::BindGroup,
    bg_gemv_lm_norm: wgpu::BindGroup,
    bg_silu: wgpu::BindGroup,
    bg_argmax: wgpu::BindGroup,
    bg_embed: wgpu::BindGroup,
    norm_buf: wgpu::Buffer,
    /// port-time scaffolding: the prefill output hidden states [s, hs] f16
    pub debug_prefill_h: Option<wgpu::Buffer>,
    /// port-time scaffolding: layer-0 row-0 hidden state after the layer
    pub debug_l0_h: Option<wgpu::Buffer>,
    /// `[max_new, hidden]` f16: row `pos - decode_h_base_pos` receives
    /// `scratch.h` at the end of that step's tail, i.e. the last layer's
    /// pre-norm residual -- what the fused `bg_gemv_lm_norm` normalizes on its
    /// way to that step's logits.  `Some` only between
    /// [`Self::start_decode_state_dump`] and [`Self::take_decode_states`]; the
    /// copy is one `copy_buffer_to_buffer` in the step's encoder and touches no
    /// dispatch, no binding and no arithmetic.
    pub decode_h_buf: Option<wgpu::Buffer>,
    /// `pos` of row 0 of `decode_h_buf`, so a row is an address and not a
    /// counter the caller has to keep.
    pub decode_h_base_pos: usize,
}

impl WgpuTextDecoder {
    /// Build the decoder and upload `{prefix}.*` weights.
    ///
    /// `max_seq` is the ceiling on `seq_len + max_new_tokens`; the KV cache
    /// starts at [`KV_INITIAL_CAP`] slots and grows up to it on demand.
    /// `rope_positions` sizes the MRoPE tables.
    pub fn load(
        gpu: Gpu,
        model_dir: &Path,
        prefix: &str,
        cfg: TextConfig,
        max_seq: usize,
        rope_positions: usize,
    ) -> Result<Self> {
        let t = std::time::Instant::now();
        let w = weights::load_tensors(model_dir)?;
        crate::load_trace::note("decoder: tensors", t);
        let int8 = checkpoint_is_int8(&w, prefix)?;
        let use_f16 = gpu.use_f16_kernel();
        let hs = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let nqh = cfg.num_attention_heads;
        let nkvh = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let inter = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let nl = cfg.num_hidden_layers;

        let t_pipes = std::time::Instant::now();
        let gqa_pl = family_layout(
            &gpu,
            "gqa_single",
            &[(0, true), (1, true), (2, true), (3, false)],
            4,
        );
        let split_pl = family_layout(
            &gpu,
            "gqa_split",
            &[(0, true), (1, true), (2, true), (3, false), (4, false), (5, false)],
            6,
        );
        let gemm_pl = family_layout_dyn(
            &gpu,
            "prefill_gemm",
            &[(0, true), (1, true), (2, false)],
            3,
            true,
        );
        let gemm_w8_pl = family_layout_dyn(
            &gpu,
            "prefill_gemm_w8",
            &[(0, true), (1, true), (2, false), (4, true)],
            3,
            true,
        );
        let sm_pl = family_layout_dyn(&gpu, "softmax", &[(0, true), (1, false)], 2, true);

        let rms_bs = block_for_reduction(hs) as usize;
        let subgroup = gpu.subgroup32();
        let sg_mode = gpu.subgroup_mode();
        let sm_mode = if sg_reduce() { sg_mode } else { SubgroupMode::None };
        // The three `slab_*` layouts are named here rather than inline in the
        // batch below: a job outlives the expression that made it.
        let slab_stats_pl =
            family_layout_dyn(&gpu, "slab_stats", &[(0, true), (1, false)], 2, true);
        let slab_weights_pl = family_layout(&gpu, "slab_weights", &[(0, true), (1, false)], 2);
        let slab_merge_pl =
            family_layout(&gpu, "slab_merge", &[(0, true), (1, true), (2, false)], 3);
        // One list, built on every core (see [`Gpu::pipelines`]): on DX12 this
        // stage is the driver compiling 48 pipelines and it dominates the load.
        let jobs = [
            PipeJob::new("gemv_qkv", shaders::gemv_mode(cfg.fused_qkv_cols(), hs, false, sg_mode, gemv_rpw()), "gemv", None),
            PipeJob::new("gemv_o", shaders::gemv_mode(hs, q_dim, true, sg_mode, gemv_rpw()), "gemv", None),
            PipeJob::new("gemv_o_rpwr2", shaders::gemv_rpwr_mode(hs, q_dim, true, sg_mode, gemv_rpw(), 2), "gemv", None),
            PipeJob::new("gemv_gu", shaders::gemv_mode(2 * inter, hs, false, sg_mode, gemv_rpw()), "gemv", None),
            PipeJob::new("gemv_dp", shaders::gemv_mode(hs, inter, true, sg_mode, gemv_rpw()), "gemv", None),
            PipeJob::new("gemv_dp_rpwr2", shaders::gemv_rpwr_mode(hs, inter, true, sg_mode, gemv_rpw(), 2), "gemv", None),
            PipeJob::new("gemv_lm", shaders::gemv_mode(vocab, hs, false, sg_mode, gemv_rpw()), "gemv", None),
            PipeJob::new(
                "gemv_qkv_norm",
                shaders::gemv_norm_mode(cfg.fused_qkv_cols(), hs, false, sg_mode, hs, rms_bs, cfg.rms_norm_eps),
                "gemv",
                None,
            ),
            PipeJob::new(
                "gemv_gu_norm",
                shaders::gemv_norm_mode(2 * inter, hs, false, sg_mode, hs, rms_bs, cfg.rms_norm_eps),
                "gemv",
                None,
            ),
            PipeJob::new(
                "gemv_lm_norm",
                shaders::gemv_norm_mode(vocab, hs, false, sg_mode, hs, rms_bs, cfg.rms_norm_eps),
                "gemv",
                None,
            ),
            PipeJob::new("gemv_w8_o", shaders::gemv_w8_mode(hs, q_dim, true, sg_mode, gemv_rpw()), "gemv", None),
            PipeJob::new(
                "gemv_w8_o_rpwr2",
                shaders::gemv_w8_rpwr_mode(hs, q_dim, true, sg_mode, gemv_rpw(), 2),
                "gemv",
                None,
            ),
            PipeJob::new("gemv_w8_dp", shaders::gemv_w8_mode(hs, inter, true, sg_mode, gemv_rpw()), "gemv", None),
            PipeJob::new(
                "gemv_w8_dp_rpwr2",
                shaders::gemv_w8_rpwr_mode(hs, inter, true, sg_mode, gemv_rpw(), 2),
                "gemv",
                None,
            ),
            // The plain (norm-not-folded) qkv/gu forms exist only so the
            // per-op debug probes keep their exact f16 shape on an int8 model;
            // the shipped decode loop uses the `_norm` fused forms above.
            PipeJob::new("gemv_w8_qkv", shaders::gemv_w8_mode(cfg.fused_qkv_cols(), hs, false, sg_mode, gemv_rpw()), "gemv", None),
            PipeJob::new("gemv_w8_gu", shaders::gemv_w8_mode(2 * inter, hs, false, sg_mode, gemv_rpw()), "gemv", None),
            PipeJob::new(
                "gemv_w8_qkv_norm",
                shaders::gemv_norm_w8_mode(cfg.fused_qkv_cols(), hs, false, sg_mode, hs, rms_bs, cfg.rms_norm_eps),
                "gemv",
                None,
            ),
            PipeJob::new(
                "gemv_w8_gu_norm",
                shaders::gemv_norm_w8_mode(2 * inter, hs, false, sg_mode, hs, rms_bs, cfg.rms_norm_eps),
                "gemv",
                None,
            ),
            PipeJob::new("gemm_w8", shaders::prefill_gemm_cfg(false, false, false, false, false, true, false, use_f16), "gemm", Some(&gemm_w8_pl)),
            PipeJob::new("gemm_w8_acc", shaders::prefill_gemm_cfg(false, true, false, false, false, true, false, use_f16), "gemm", Some(&gemm_w8_pl)),
            PipeJob::new("rms_norm", shaders::rms_norm(hs, rms_bs), "rms_norm", None),
            PipeJob::new("qkv_extract", shaders::qkv_extract(nqh, nkvh, hd), "qkv_extract", None),
            PipeJob::new("gqa256", shaders::gqa_decode_single(nqh, nkvh, hd, 256, GQA_SINGLE_CAP), "gqa", Some(&gqa_pl)),
            PipeJob::new("gqa512", shaders::gqa_decode_single(nqh, nkvh, hd, 512, GQA_SINGLE_CAP), "gqa", Some(&gqa_pl)),
            PipeJob::new("gqa_split128", shaders::gqa_decode_split_p1(nqh, nkvh, hd, 128), "gqa_split_p1", Some(&split_pl)),
            PipeJob::new("gqa_split_pair256", pair_split_src(nqh, nkvh, hd, 256, false, subgroup && gqa_coop()  , gqa_pf(), false, gqa_depth()), pair_split_entry(nqh, nkvh), Some(&split_pl)),
            PipeJob::new("gqa_split_pair512", pair_split_src(nqh, nkvh, hd, 512, false, subgroup && gqa_coop()  , gqa_pf(), false, gqa_depth()), pair_split_entry(nqh, nkvh), Some(&split_pl)),
            PipeJob::new("gqa_split_pair256_row0", shaders::gqa_decode_split_p1_pair(hd, 256, 2, true, false, false, false, 1), "gqa_split_p1_pair", Some(&split_pl)),
            PipeJob::new("gqa_split_pair256_coal", shaders::gqa_decode_split_p1_pair(hd, 256, 2, false, false, false, true, 1), "gqa_split_p1_pair", Some(&split_pl)),
            PipeJob::new("gqa_split256", shaders::gqa_decode_split_p1(nqh, nkvh, hd, 256), "gqa_split_p1", Some(&split_pl)),
            PipeJob::new("gqa_split512", shaders::gqa_decode_split_p1(nqh, nkvh, hd, 512), "gqa_split_p1", Some(&split_pl)),
            PipeJob::new("gqa_merge", shaders::gqa_split_merge(hd), "gqa_merge", None),
            PipeJob::new("silu30", shaders::silu_mul_split(inter), "silu_mul_split", None),
            PipeJob::new("argmax", shaders::argmax_into_slot(), "argmax", None),
            PipeJob::new("embed", shaders::embed_lookup_single(), "embed", None),
            PipeJob::new("gemm", shaders::prefill_gemm(false, false), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm_acc", shaders::prefill_gemm(false, true), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm_av", shaders::prefill_gemm(true, false), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm_causal", shaders::prefill_gemm_causal(), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm_av_causal", shaders::prefill_gemm_causal_av(), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm_64", shaders::prefill_gemm_64(false, false, use_f16), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm_acc_64", shaders::prefill_gemm_64(false, true, use_f16), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm_av_64", shaders::prefill_gemm_64(true, false, use_f16), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm_causal_64", shaders::prefill_gemm_causal_64(use_f16), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm_av_causal_64", shaders::prefill_gemm_causal_av_64(use_f16), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm_w8_64", shaders::prefill_gemm_w8_64(false, use_f16), "gemm", Some(&gemm_w8_pl)),
            PipeJob::new("gemm_w8_acc_64", shaders::prefill_gemm_w8_64(true, use_f16), "gemm", Some(&gemm_w8_pl)),
            PipeJob::new("softmax32", shaders::softmax_causal_mode(32, sm_mode), "softmax", Some(&sm_pl)),
            PipeJob::new("softmax64", shaders::softmax_causal_mode(64, sm_mode), "softmax", Some(&sm_pl)),
            PipeJob::new("softmax128", shaders::softmax_causal_mode(128, sm_mode), "softmax", Some(&sm_pl)),
            PipeJob::new("softmax256", shaders::softmax_causal_mode(256, sm_mode), "softmax", Some(&sm_pl)),
            PipeJob::new("softmax512", shaders::softmax_causal_mode(512, sm_mode), "softmax", Some(&sm_pl)),
            PipeJob::new("softmax1024", shaders::softmax_causal_mode(1024, sm_mode), "softmax", Some(&sm_pl)),
            PipeJob::new("softmax64x256", shaders::softmax_causal_tree_mode(64, 256, sm_mode), "softmax", Some(&sm_pl)),
            PipeJob::new("softmax64x512", shaders::softmax_causal_tree_mode(64, 512, sm_mode), "softmax", Some(&sm_pl)),
            PipeJob::new("softmax64x1024", shaders::softmax_causal_tree_mode(64, 1024, sm_mode), "softmax", Some(&sm_pl)),
            PipeJob::new("softmax128x256", shaders::softmax_causal_tree_mode(128, 256, sm_mode), "softmax", Some(&sm_pl)),
            PipeJob::new("softmax128x512", shaders::softmax_causal_tree_mode(128, 512, sm_mode), "softmax", Some(&sm_pl)),
            PipeJob::new("softmax128x1024", shaders::softmax_causal_tree_mode(128, 1024, sm_mode), "softmax", Some(&sm_pl)),
            PipeJob::new("softmax256x512", shaders::softmax_causal_tree_mode(256, 512, sm_mode), "softmax", Some(&sm_pl)),
            PipeJob::new("softmax256x1024", shaders::softmax_causal_tree_mode(256, 1024, sm_mode), "softmax", Some(&sm_pl)),
            PipeJob::new("slab_stats", shaders::slab_stats(SLAB_BS, SLAB_T), "slab_stats", Some(&slab_stats_pl)),
            PipeJob::new("slab_weights", shaders::slab_weights(256), "slab_weights", Some(&slab_weights_pl)),
            PipeJob::new("slab_merge", shaders::slab_merge(nqh, hd), "slab_merge", Some(&slab_merge_pl)),
        ];
        let mut built = gpu.pipelines(&jobs)?;
        let mut take = |name: &str| -> Result<wgpu::ComputePipeline> {
            built
                .remove(name)
                .with_context(|| format!("pipeline {name} was not built"))
        };
        let pipes = Pipes {
            gemv_qkv: take("gemv_qkv")?,
            gemv_o: take("gemv_o")?,
            gemv_o_rpwr2: take("gemv_o_rpwr2")?,
            gemv_gu: take("gemv_gu")?,
            gemv_dp: take("gemv_dp")?,
            gemv_dp_rpwr2: take("gemv_dp_rpwr2")?,
            gemv_lm: take("gemv_lm")?,
            gemv_qkv_norm: take("gemv_qkv_norm")?,
            gemv_gu_norm: take("gemv_gu_norm")?,
            gemv_lm_norm: take("gemv_lm_norm")?,
            gemv_w8_o: take("gemv_w8_o")?,
            gemv_w8_o_rpwr2: take("gemv_w8_o_rpwr2")?,
            gemv_w8_dp: take("gemv_w8_dp")?,
            gemv_w8_dp_rpwr2: take("gemv_w8_dp_rpwr2")?,
            gemv_w8_qkv: take("gemv_w8_qkv")?,
            gemv_w8_gu: take("gemv_w8_gu")?,
            gemv_w8_qkv_norm: take("gemv_w8_qkv_norm")?,
            gemv_w8_gu_norm: take("gemv_w8_gu_norm")?,
            gemm_w8: take("gemm_w8")?,
            gemm_w8_acc: take("gemm_w8_acc")?,
            rms_norm: take("rms_norm")?,
            extract: take("qkv_extract")?,
            gqa256: take("gqa256")?,
            gqa512: take("gqa512")?,
            gqa_split128: take("gqa_split128")?,
            gqa_split_pair256: take("gqa_split_pair256")?,
            gqa_split_pair512: take("gqa_split_pair512")?,
            gqa_split_pair256_row0: take("gqa_split_pair256_row0")?,
            gqa_split_pair256_coal: take("gqa_split_pair256_coal")?,
            gqa_split256: take("gqa_split256")?,
            gqa_split512: take("gqa_split512")?,
            gqa_merge: take("gqa_merge")?,
            silu: take("silu30")?,
            argmax: take("argmax")?,
            embed: take("embed")?,
            gemm: take("gemm")?,
            gemm_acc: take("gemm_acc")?,
            gemm_av: take("gemm_av")?,
            gemm_causal: take("gemm_causal")?,
            gemm_av_causal: take("gemm_av_causal")?,
            gemm_64: take("gemm_64")?,
            gemm_acc_64: take("gemm_acc_64")?,
            gemm_av_64: take("gemm_av_64")?,
            gemm_causal_64: take("gemm_causal_64")?,
            gemm_av_causal_64: take("gemm_av_causal_64")?,
            gemm_w8_64: take("gemm_w8_64")?,
            gemm_w8_acc_64: take("gemm_w8_acc_64")?,
            softmax: std::collections::HashMap::from([
                (32, take("softmax32")?),
                (64, take("softmax64")?),
                (128, take("softmax128")?),
                (256, take("softmax256")?),
                (512, take("softmax512")?),
                (1024, take("softmax1024")?),
            ]),
            // Every `(phys, tree)` the shipped `SOFTMAX_BS_PHYS` can meet must be
            // here: `tree` is the row-length ladder (256/512/1024), and a physical
            // width that lands on a missing pair is a panic, not a fallback.
            softmax_wide: std::collections::HashMap::from([
                ((64, 256), take("softmax64x256")?),
                ((64, 512), take("softmax64x512")?),
                ((64, 1024), take("softmax64x1024")?),
                ((128, 256), take("softmax128x256")?),
                ((128, 512), take("softmax128x512")?),
                ((128, 1024), take("softmax128x1024")?),
                ((256, 512), take("softmax256x512")?),
                ((256, 1024), take("softmax256x1024")?),
            ]),
                    slab_stats: take("slab_stats")?,
            slab_weights: take("slab_weights")?,
            slab_merge: take("slab_merge")?,
        };
        crate::load_trace::note("decoder: pipelines", t_pipes);

        let max_chunks = max_seq.div_ceil(256);
        let t_weights = std::time::Instant::now();
        let mut up = gpu.uploader();
        let scratch = Scratch {
            h: up.storage("h", (hs / 2 * 4) as u64),
            norm1: up.storage("norm1", (hs / 2 * 4) as u64),
            qkv: up.storage("qkv", (cfg.fused_qkv_cols() / 2 * 4) as u64),
            q_out: up.storage("q_out", (q_dim / 2 * 4) as u64),
            attn_out: up.storage("attn_out", (q_dim / 2 * 4) as u64),
            norm2: up.storage("norm2", (hs / 2 * 4) as u64),
            gate_up: up.storage("gate_up", (inter * 4) as u64),
            activated: up.storage("activated", (inter / 2 * 4) as u64),
            final_norm: up.storage("final_norm", (hs / 2 * 4) as u64),
            logits: up.storage("logits", (vocab / 2 * 4) as u64),
            token: up.storage("token", 4),
            token_staging: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("token_staging"),
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            cos: up.storage("cos", (rope_positions * hd / 2 * 4) as u64),
            sin: up.storage("sin", (rope_positions * hd / 2 * 4) as u64),
            u_rms: up.uniform("u_rms", 16),
            u_qkvx: up.uniform("u_qkvx", 32),
            u_silu: up.uniform("u_silu", 16),
            u_gqa: up.uniform("u_gqa", 16),
            u_gp1: up.uniform("u_gp1", 16),
            u_gp2: up.uniform("u_gp2", 16),
            split_part_out: up.storage("split_part_out", (nqh * max_chunks * hd * 4) as u64),
            split_part_max: up.storage("split_part_max", (nqh * max_chunks * 4) as u64),
            split_part_sum: up.storage("split_part_sum", (nqh * max_chunks * 4) as u64),
        };
        up.upload(
            &scratch.u_rms,
            bytemuck::bytes_of(&RmsCfg { eps: cfg.rms_norm_eps, _a: 0.0, _b: 0.0, _c: 0.0 }),
        )?;

        let t_embed = std::time::Instant::now();
        let embed = weights::get_matrix(&w, &format!("{prefix}.embed_tokens.weight"))?;
        crate::load_trace::note("decoder: embed narrow", t_embed);
        if embed.rows != vocab || embed.cols != hs {
            bail!(
                "embed_tokens is [{}x{}], expected [{vocab}x{hs}]",
                embed.rows,
                embed.cols
            );
        }
        let embed_table = upload_weight(&mut up, "embed_tokens", &embed)?;
        crate::load_trace::note("decoder: embed upload", t_embed);
        let norm_buf = upload_vec(
            &mut up,
            "final_norm_w",
            &weights::get_vector(&w, &format!("{prefix}.norm.weight"))?,
        )?;

        let cap = KV_INITIAL_CAP.min(max_seq);
        let kv_words = nkvh * cap * hd / 2;
        let u_argmax = up.uniform("u_argmax", 16);
        let u_embed = up.uniform("u_embed", 16);
        up.upload(
            &u_argmax,
            bytemuck::bytes_of(&ArgmaxCfg { n: vocab as u32, slot: 0, _a: 0, _b: 0 }),
        )?;
        up.upload(
            &u_embed,
            bytemuck::bytes_of(&EmbedCfg { slot: 0, d2: (hs / 2) as u32, _a: 0, _b: 0 }),
        )?;

        let gemv_bg = |pipe: &wgpu::ComputePipeline,
                       wt: &wgpu::Buffer,
                       x: &wgpu::Buffer,
                       y: &wgpu::Buffer|
         -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gemv"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                ],
            })
        };
        // The int8 forms: same weight/activation/output bindings plus the
        // per-row f32 scale buffer (`gemv_w8` binding 3, `gemv_norm_w8`
        // binding 4 — after the norm weights).
        let gemv_w8_bg = |pipe: &wgpu::ComputePipeline,
                          wt: &wgpu::Buffer,
                          x: &wgpu::Buffer,
                          y: &wgpu::Buffer,
                          s: &wgpu::Buffer|
         -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gemv_w8"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: s.as_entire_binding() },
                ],
            })
        };
        let gemv_norm_w8_bg = |pipe: &wgpu::ComputePipeline,
                               wt: &wgpu::Buffer,
                               x: &wgpu::Buffer,
                               y: &wgpu::Buffer,
                               nw: &wgpu::Buffer,
                               s: &wgpu::Buffer|
         -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gemv_norm_w8"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: nw.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: s.as_entire_binding() },
                ],
            })
        };
        let gemv_norm_bg = |pipe: &wgpu::ComputePipeline,
                            wt: &wgpu::Buffer,
                            x: &wgpu::Buffer,
                            y: &wgpu::Buffer,
                            nw: &wgpu::Buffer|
         -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gemv_norm"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: nw.as_entire_binding() },
                ],
            })
        };
        let rms_bg = |x: &wgpu::Buffer, wt: &wgpu::Buffer, out: &wgpu::Buffer| -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rms"),
                layout: &pipes.rms_norm.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: scratch.u_rms.as_entire_binding() },
                ],
            })
        };

        let mut layers = Vec::with_capacity(nl);
        let mut t_conv = std::time::Duration::ZERO;
        let mut t_up = std::time::Duration::ZERO;
        // Conversions run one layer ahead on another thread, so the narrowing of
        // layer i+1 happens while layer i crosses the bus — the two are the only
        // things this load does, and only one of them is on the critical path.
        // The channel bounds the lookahead to a couple of layers' host memory.
        std::thread::scope(|scope| -> Result<()> {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<LayerWeights>>(PREFETCH_LAYERS);
        scope.spawn(move || {
            for i in 0..nl {
                let p = format!("{prefix}.layers.{i}");
                if tx.send(convert_layer(&w, &p, int8)).is_err() {
                    break; // consumer went away
                }
            }
        });
        for i in 0..nl {
            let t = std::time::Instant::now();
            let lw = rx
                .recv()
                .map_err(|_| anyhow!("weight conversion thread died at layer {i}"))??;
            t_conv += t.elapsed();
            let LayerWeights {
                qkv: qkv_p,
                o: o_w,
                gu: gu_p,
                dp: dp_w,
                scales,
                iln: iln_v,
                pln: pln_v,
                qn: qn_v,
                kn: kn_v,
            } = lw;

            let t = std::time::Instant::now();
            let qkv = up.upload_pieces("qkv_w", &qkv_p)?;
            let o = upload_weight(&mut up, "o_w", &o_w)?;
            let gu = up.upload_pieces("gu_w", &gu_p)?;
            let dp = upload_weight(&mut up, "dp_w", &dp_w)?;
            let (qkv_s, o_s, gu_s, dp_s) = match &scales {
                None => (None, None, None, None),
                Some(sc) => (
                    Some(upload_bytes(&mut up, "qkv_s", &sc.qkv)?),
                    Some(upload_bytes(&mut up, "o_s", &sc.o)?),
                    Some(upload_bytes(&mut up, "gu_s", &sc.gu)?),
                    Some(upload_bytes(&mut up, "dp_s", &sc.dp)?),
                ),
            };
            let iln = upload_vec(&mut up, "iln_w", &iln_v)?;
            let pln = upload_vec(&mut up, "pln_w", &pln_v)?;
            let qn = upload_vec(&mut up, "qn_w", &qn_v)?;
            let kn = upload_vec(&mut up, "kn_w", &kn_v)?;
            t_up += t.elapsed();

            let k_cache = gpu.storage("k_cache", (kv_words * 4) as u64);
            let v_cache = gpu.storage("v_cache", (kv_words * 4) as u64);

            let bg_gemv_qkv = gemv_bg(&pipes.gemv_qkv, &qkv, &scratch.norm1, &scratch.qkv);
            let bg_gemv_qkv_norm = gemv_norm_bg(&pipes.gemv_qkv_norm, &qkv, &scratch.h, &scratch.qkv, &iln);
            let bg_gemv_o = gemv_bg(&pipes.gemv_o, &o, &scratch.attn_out, &scratch.h);
            let bg_gemv_o_rpwr2 =
                gemv_bg(&pipes.gemv_o_rpwr2, &o, &scratch.attn_out, &scratch.h);
            let bg_gemv_gu = gemv_bg(&pipes.gemv_gu, &gu, &scratch.norm2, &scratch.gate_up);
            let bg_gemv_gu_norm = gemv_norm_bg(&pipes.gemv_gu_norm, &gu, &scratch.h, &scratch.gate_up, &pln);
            let bg_gemv_dp = gemv_bg(&pipes.gemv_dp, &dp, &scratch.activated, &scratch.h);
            let bg_gemv_dp_rpwr2 =
                gemv_bg(&pipes.gemv_dp_rpwr2, &dp, &scratch.activated, &scratch.h);
            let (bg_gemv_w8_qkv, bg_gemv_w8_gu, bg_gemv_w8_qkv_norm, bg_gemv_w8_o, bg_gemv_w8_o_rpwr2, bg_gemv_w8_gu_norm, bg_gemv_w8_dp, bg_gemv_w8_dp_rpwr2) =
                match (&qkv_s, &o_s, &gu_s, &dp_s) {
                    (Some(qs), Some(os), Some(gs), Some(ds)) => (
                        Some(gemv_w8_bg(&pipes.gemv_w8_qkv, &qkv, &scratch.norm1, &scratch.qkv, qs)),
                        Some(gemv_w8_bg(&pipes.gemv_w8_gu, &gu, &scratch.norm2, &scratch.gate_up, gs)),
                        Some(gemv_norm_w8_bg(&pipes.gemv_w8_qkv_norm, &qkv, &scratch.h, &scratch.qkv, &iln, qs)),
                        Some(gemv_w8_bg(&pipes.gemv_w8_o, &o, &scratch.attn_out, &scratch.h, os)),
                        Some(gemv_w8_bg(&pipes.gemv_w8_o_rpwr2, &o, &scratch.attn_out, &scratch.h, os)),
                        Some(gemv_norm_w8_bg(&pipes.gemv_w8_gu_norm, &gu, &scratch.h, &scratch.gate_up, &pln, gs)),
                        Some(gemv_w8_bg(&pipes.gemv_w8_dp, &dp, &scratch.activated, &scratch.h, ds)),
                        Some(gemv_w8_bg(&pipes.gemv_w8_dp_rpwr2, &dp, &scratch.activated, &scratch.h, ds)),
                    ),
                    _ => (None, None, None, None, None, None, None, None),
                };
            let bg_rms1 = rms_bg(&scratch.h, &iln, &scratch.norm1);
            let bg_rms2 = rms_bg(&scratch.h, &pln, &scratch.norm2);

            let (bg_extract, bg_gqa, bg_gqa_split) =
                kv_bind_groups(&gpu, &pipes, &scratch, &qn, &kn, &k_cache, &v_cache);
            let bg_gqa_merge = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gqa_merge"),
                layout: &pipes.gqa_merge.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: scratch.attn_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: scratch.split_part_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: scratch.split_part_max.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: scratch.split_part_sum.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: scratch.u_gp2.as_entire_binding() },
                ],
            });

            layers.push(Layer {
                k_cache,
                v_cache,
                iln_w: iln,
                pln_w: pln,
                qn_w: qn,
                kn_w: kn,
                qkv_w: qkv,
                o_w: o,
                gu_w: gu,
                dp_w: dp,
                qkv_s,
                o_s,
                gu_s,
                dp_s,
                bg_gemv_qkv,
                bg_gemv_qkv_norm,
                bg_gemv_o,
        bg_gemv_o_rpwr2,
        bg_gemv_dp_rpwr2,
                bg_gemv_gu,
                bg_gemv_gu_norm,
                bg_gemv_dp,
                bg_gemv_w8_qkv,
                bg_gemv_w8_gu,
                bg_gemv_w8_qkv_norm,
                bg_gemv_w8_o,
                bg_gemv_w8_o_rpwr2,
                bg_gemv_w8_gu_norm,
                bg_gemv_w8_dp,
                bg_gemv_w8_dp_rpwr2,
                bg_rms1,
                bg_rms2,
                bg_extract,
                bg_gqa,
                bg_gqa_split,
                bg_gqa_merge,
            });
        }
        Ok(())
        })?;

        let t_tail = std::time::Instant::now();
        let bg_final_rms = rms_bg(&scratch.h, &norm_buf, &scratch.final_norm);
        let bg_gemv_lm = gemv_bg(&pipes.gemv_lm, &embed_table, &scratch.final_norm, &scratch.logits);
        let bg_gemv_lm_norm = gemv_norm_bg(&pipes.gemv_lm_norm, &embed_table, &scratch.h, &scratch.logits, &norm_buf);
        let bg_silu = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("silu"),
            layout: &pipes.silu.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: scratch.gate_up.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scratch.activated.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scratch.u_silu.as_entire_binding() },
            ],
        });
        let bg_argmax = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("argmax"),
            layout: &pipes.argmax.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: scratch.logits.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scratch.token.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: u_argmax.as_entire_binding() },
            ],
        });
        let bg_embed = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embed"),
            layout: &pipes.embed.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: embed_table.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scratch.token.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scratch.h.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: u_embed.as_entire_binding() },
            ],
        });

        up.finish()?;
        crate::load_trace::note("decoder: tail (bind groups + drain)", t_tail);
        crate::load_trace::note_dur("decoder: host convert", t_conv);
        crate::load_trace::note_dur("decoder: gpu upload", t_up);
        crate::load_trace::note("decoder: weights + upload", t_weights);

        Ok(Self {
            gpu,
            cfg,
            max_seq,
            cap,
            pos: 0,
            host_submit_ms: 0.0,
            host_read_ms: 0.0,
            pending: None,
            pending_map: None,
            pooled: false,
            prefill_scratch: None,
            multi_scratch: None,
            embed_table,
            layers,
            pipes: std::sync::Arc::new(pipes),
            int8,
            scratch,
            bg_final_rms,
            bg_gemv_lm,
            bg_gemv_lm_norm,
            bg_silu,
            bg_argmax,
            bg_embed,
            norm_buf,
            debug_prefill_h: None,
            debug_l0_h: None,
            decode_h_buf: None,
            decode_h_base_pos: 0,
        })
    }

    /// Dispatches a single step issues at `cur_len` (the split-attention path
    /// adds one merge dispatch per layer).
    pub fn dispatches_per_step(&self, cur_len: usize) -> usize {
        let per_layer = if cur_len > GQA_SINGLE_CAP { 10 } else { 9 };
        1 + self.cfg.num_hidden_layers * per_layer + 3
    }

    /// The device the decoder lives on — shared with the GPU audio tower.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn k_cache(&self, layer: usize) -> &wgpu::Buffer {
        &self.layers[layer].k_cache
    }
    pub fn v_cache(&self, layer: usize) -> &wgpu::Buffer {
        &self.layers[layer].v_cache
    }

    /// Grow the KV cache so that it holds `need` positions.
    ///
    /// Grow-only and stepped ([`KV_STEP`]): a run of clips of one shape allocates
    /// once, and a request that already fits costs nothing but the comparison.
    /// `need` past the ceiling is clamped here — the caller's own check against
    /// [`Self::max_seq`] is what reports that.
    ///
    /// Only the KV buffers and the three bind groups that read them
    /// ([`kv_bind_groups`]) are rebuilt; weights, pipelines and every other bind
    /// group stay where they are.
    pub fn ensure_capacity(&mut self, need: usize) -> usize {
        // A pooled slot's KV rows are part of a shared allocation; reallocating
        // here would silently unhook them from the batch.
        if self.pooled {
            return self.cap;
        }
        let target = (need.div_ceil(KV_STEP) * KV_STEP).min(self.max_seq).max(self.cap);
        if target == self.cap {
            return self.cap;
        }
        let t0 = std::time::Instant::now();
        let prev = self.cap;
        let kv_words = (self.cfg.num_key_value_heads * target * self.cfg.head_dim / 2) as u64;
        // Preserve the rows already written: a KV-prefix reuse prefill only
        // rewrites its delta, so losing the old cache on growth would corrupt
        // every cached position.  The head stride changes with capacity, so copy
        // each head's valid prefix separately.
        let nkvh = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        let copy_rows = self.pos.min(prev);
        let mut copy = (copy_rows > 0 && target > prev)
            .then(|| self.gpu.device.create_command_encoder(&Default::default()));
        {
            let Self { gpu, layers, pipes, scratch, .. } = self;
            for layer in layers.iter_mut() {
                let k_cache = gpu.storage("k_cache", kv_words * 4);
                let v_cache = gpu.storage("v_cache", kv_words * 4);
                if let Some(cp) = copy.as_mut() {
                    let bytes = (copy_rows * hd * 2) as u64;
                    for h in 0..nkvh {
                        let src = (h * prev * hd * 2) as u64;
                        let dst = (h * target * hd * 2) as u64;
                        cp.copy_buffer_to_buffer(&layer.k_cache, src, &k_cache, dst, bytes);
                        cp.copy_buffer_to_buffer(&layer.v_cache, src, &v_cache, dst, bytes);
                    }
                }
                let (bg_extract, bg_gqa, bg_gqa_split) = {
                    let (qn_w, kn_w) = (&layer.qn_w, &layer.kn_w);
                    kv_bind_groups(gpu, pipes, scratch, qn_w, kn_w, &k_cache, &v_cache)
                };
                layer.k_cache = k_cache;
                layer.v_cache = v_cache;
                layer.bg_extract = bg_extract;
                layer.bg_gqa = bg_gqa;
                layer.bg_gqa_split = bg_gqa_split;
            }
        }
        if let Some(cp) = copy {
            self.gpu.queue.submit([cp.finish()]);
        }
        self.cap = target;
        let mib = (2 * self.cfg.num_hidden_layers as u64 * kv_words * 4) as f64 / (1024.0 * 1024.0);
        eprintln!(
            "[kv] capacity {prev} -> {target} slots ({mib:.0} MiB) in {:.1} ms",
            t0.elapsed().as_secs_f64() * 1000.0,
        );
        self.cap
    }

    /// Upload MRoPE tables — `[rope_positions, head_dim]` f16, row-major.
    pub fn set_rope_tables(&self, cos: &[half::f16], sin: &[half::f16]) {
        self.gpu.upload(&self.scratch.cos, &weights::words_bytes(cos));
        self.gpu.upload(&self.scratch.sin, &weights::words_bytes(sin));
    }

    /// Seed the token that the next `step()` will embed.
    pub fn set_input_token(&self, t: i32) {
        self.gpu.upload(&self.scratch.token, &t.to_le_bytes());
    }

    fn write_step_uniforms(&self, pos: usize) {
        let cur_len = pos + 1;
        self.write_silu_rows(1);
        self.gpu.upload(
            &self.scratch.u_qkvx,
            bytemuck::bytes_of(&QkvxCfg {
                max_seq: self.cap as u32,
                start: pos as u32,
                pos_offset: pos as u32,
                s: 1,
                eps: self.cfg.rms_norm_eps,
                _a: 0,
                _b: 0,
                _c: 0,
            }),
        );
        self.gpu.upload(
            &self.scratch.u_gqa,
            bytemuck::bytes_of(&GqaCfg {
                cur_len: cur_len as u32,
                max_seq: self.cap as u32,
                scale: self.cfg.scale(),
                _p: 0.0,
            }),
        );
        let chunk = gqa_split_chunk(cur_len) as u32;
        let n_chunks = (cur_len as u32).div_ceil(chunk);
        self.gpu.upload(
            &self.scratch.u_gp1,
            bytemuck::bytes_of(&SplitCfg {
                cur_len: cur_len as u32,
                max_seq: self.cap as u32,
                scale: self.cfg.scale(),
                n_chunks,
            }),
        );
        self.gpu.upload(
            &self.scratch.u_gp2,
            bytemuck::bytes_of(&MergeCfg { n_chunks, _a: 0, _b: 0, _c: 0 }),
        );
    }

    fn write_silu_rows(&self, rows: usize) -> (u32, u32) {
        let words = rows * self.cfg.intermediate_size / 2;
        let workgroups = words.div_ceil(256);
        let (gx, gy) = grid_xy(workgroups);
        self.gpu.upload(
            &self.scratch.u_silu,
            bytemuck::bytes_of(&SiluCfg {
                inter2: (self.cfg.intermediate_size / 2) as u32,
                total2: words as u32,
                gx,
                _b: 0,
            }),
        );
        (gx, gy)
    }

    /// One decode step: prologue, all layers, epilogue.
    pub fn encode_step(&self, enc: &mut wgpu::CommandEncoder, pos: usize) {
        self.encode_step_slice(enc, pos, 0, self.layers.len(), true, true);
    }

    /// A slice of a decode step: layers `lo..hi`, plus the embedding prologue if
    /// `embed` and the `lm_head`/argmax/token-copy epilogue if `tail`.  This is
    /// `encode_step` with the boundaries left open so `QASR_SUBMIT_SPLIT` /
    /// `QASR_PASS_SPLIT` can cut the step into several passes or several submits
    /// (`step_slice`); the dispatch sequence is the same for every `n`, so an A/B
    /// of the two prices the boundary and not the work.
    fn encode_step_slice(
        &self,
        enc: &mut wgpu::CommandEncoder,
        pos: usize,
        lo: usize,
        hi: usize,
        embed: bool,
        tail: bool,
    ) {
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            self.encode_into(&mut cp, pos, lo, hi, embed, tail, &|_| true);
        }
        if tail {
            enc.copy_buffer_to_buffer(&self.scratch.token, 0, &self.scratch.token_staging, 0, 4);
            self.encode_state_row(enc, pos);
        }
    }

    /// Copy this step's `scratch.h` into its row of [`Self::decode_h_buf`], when a
    /// state dump is armed.
    ///
    /// `scratch.h` is the residual stream *after* the last layer, which is what a
    /// standalone `rms_norm` + `gemv_lm` tail would read; the shipped tail fuses
    /// that norm into `bg_gemv_lm_norm` (so `scratch.final_norm` is never written
    /// on the b=1 path), and this copy is the only way to see it without adding a
    /// dispatch.
    fn encode_state_row(&self, enc: &mut wgpu::CommandEncoder, pos: usize) {
        let Some(dst) = &self.decode_h_buf else { return };
        let bytes = (self.cfg.hidden_size * 2) as u64;
        let off = (pos.saturating_sub(self.decode_h_base_pos) as u64) * bytes;
        debug_assert_eq!(bytes % 4, 0, "a state row must be 4-byte aligned");
        debug_assert!(off + bytes <= dst.size(), "state row past the dump buffer");
        if off + bytes > dst.size() {
            return;
        }
        enc.copy_buffer_to_buffer(&self.scratch.h, 0, dst, off, bytes);
    }

    /// This slice's dispatches into an already-open compute pass.
    ///
    /// [`Self::encode_step_slice`] is this plus the pass and the token copy, so
    /// the b=1 path issues exactly the dispatches it always did.  The batch
    /// driver needs the pass-level form: several slots' steps share one pass, so
    /// the dispatches of layer `l` for every slot land next to each other and the
    /// weights slot 0 pulls in are the ones the other slots find in L2.
    pub fn encode_into<'pass>(
        &'pass self,
        cp: &mut wgpu::ComputePass<'pass>,
        pos: usize,
        lo: usize,
        hi: usize,
        embed: bool,
        tail: bool,
        want: &dyn Fn(LayerBlock) -> bool,
    ) {
        let cfg = &self.cfg;
        let cur_len = pos + 1;
        let gemv_grid = |rows: usize| (rows / gemv_rpw()) as u32;
        let norm_grid = |rows: usize| (rows / GEMV_NORM_RPW) as u32;
        let silu_grid = grid_xy((cfg.intermediate_size / 2).div_ceil(256));

        let dup = std::env::var("QASR_DUP").unwrap_or_default();
        let d = |name: &str| dup == name;
        // `QASR_SKIP` drops one dispatch per layer instead of duplicating it.
        // Skipping changes the numbers and therefore the tokens, so the figure to
        // read is `decode ms / token`, not the absolute decode time.
        let skip = std::env::var("QASR_SKIP").unwrap_or_default();
        let k = |name: &str| skip == name;

        if embed {
            cp.set_pipeline(&self.pipes.embed);
            cp.set_bind_group(0, &self.bg_embed, &[]);
            cp.dispatch_workgroups(1, 1, 1);
        }

        for l in &self.layers[lo..hi] {
            if want(LayerBlock::Qkv) && !k("qkv") {
                if self.int8 {
                    cp.set_pipeline(&self.pipes.gemv_w8_qkv_norm);
                    cp.set_bind_group(0, l.bg_gemv_w8_qkv_norm.as_ref().expect("int8 bind groups"), &[]);
                } else {
                    cp.set_pipeline(&self.pipes.gemv_qkv_norm);
                    cp.set_bind_group(0, &l.bg_gemv_qkv_norm, &[]);
                }
                cp.dispatch_workgroups(norm_grid(cfg.fused_qkv_cols()), 1, 1);
            }
            if want(LayerBlock::Qkv) && d("qkv") {
                cp.dispatch_workgroups(norm_grid(cfg.fused_qkv_cols()), 1, 1);
            }

            if want(LayerBlock::Extract) && !k("extract") {
                cp.set_pipeline(&self.pipes.extract);
                cp.set_bind_group(0, &l.bg_extract, &[]);
                cp.dispatch_workgroups(1, (cfg.num_attention_heads + cfg.num_key_value_heads) as u32, 1);
            }
            if want(LayerBlock::Extract) && d("extract") {
                cp.dispatch_workgroups(1, (cfg.num_attention_heads + cfg.num_key_value_heads) as u32, 1);
            }

            if want(LayerBlock::Gqa) && !k("gqa") {
                self.encode_gqa(cp, l, cur_len);
            }
            if want(LayerBlock::Gqa) && d("gqa")
                && want(LayerBlock::Gqa) && !k("gqa") {
                    self.encode_gqa(cp, l, cur_len);
                }

            let rpwr = gemv_rpwr_o_dp(cfg.hidden_size);
            if want(LayerBlock::O) && !k("o") {
                if self.int8 {
                    let bg = if rpwr == 2 {
                        &l.bg_gemv_w8_o_rpwr2
                    } else {
                        &l.bg_gemv_w8_o
                    };
                    cp.set_pipeline(if rpwr == 2 {
                        &self.pipes.gemv_w8_o_rpwr2
                    } else {
                        &self.pipes.gemv_w8_o
                    });
                    cp.set_bind_group(0, bg.as_ref().expect("int8 bind groups"), &[]);
                } else if rpwr == 2 {
                    cp.set_pipeline(&self.pipes.gemv_o_rpwr2);
                    cp.set_bind_group(0, &l.bg_gemv_o_rpwr2, &[]);
                } else {
                    cp.set_pipeline(&self.pipes.gemv_o);
                    cp.set_bind_group(0, &l.bg_gemv_o, &[]);
                }
                cp.dispatch_workgroups(gemv_grid(cfg.hidden_size) / rpwr as u32, 1, 1);
            }
            if want(LayerBlock::O) && d("o") {
                cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1);
            }

            if want(LayerBlock::Gu) && !k("gu") {
                if self.int8 {
                    cp.set_pipeline(&self.pipes.gemv_w8_gu_norm);
                    cp.set_bind_group(0, l.bg_gemv_w8_gu_norm.as_ref().expect("int8 bind groups"), &[]);
                } else {
                    cp.set_pipeline(&self.pipes.gemv_gu_norm);
                    cp.set_bind_group(0, &l.bg_gemv_gu_norm, &[]);
                }
                cp.dispatch_workgroups(norm_grid(2 * cfg.intermediate_size), 1, 1);
            }
            if want(LayerBlock::Gu) && d("gu") {
                cp.dispatch_workgroups(norm_grid(2 * cfg.intermediate_size), 1, 1);
            }

            if want(LayerBlock::Silu) && !k("silu") {
                cp.set_pipeline(&self.pipes.silu);
                cp.set_bind_group(0, &self.bg_silu, &[]);
                cp.dispatch_workgroups(silu_grid.0, silu_grid.1, 1);
            }
            if want(LayerBlock::Silu) && d("silu") {
                cp.dispatch_workgroups(silu_grid.0, silu_grid.1, 1);
            }

            if want(LayerBlock::Dp) && !k("dp") {
                if self.int8 {
                    let bg = if rpwr == 2 {
                        &l.bg_gemv_w8_dp_rpwr2
                    } else {
                        &l.bg_gemv_w8_dp
                    };
                    cp.set_pipeline(if rpwr == 2 {
                        &self.pipes.gemv_w8_dp_rpwr2
                    } else {
                        &self.pipes.gemv_w8_dp
                    });
                    cp.set_bind_group(0, bg.as_ref().expect("int8 bind groups"), &[]);
                } else if rpwr == 2 {
                    cp.set_pipeline(&self.pipes.gemv_dp_rpwr2);
                    cp.set_bind_group(0, &l.bg_gemv_dp_rpwr2, &[]);
                } else {
                    cp.set_pipeline(&self.pipes.gemv_dp);
                    cp.set_bind_group(0, &l.bg_gemv_dp, &[]);
                }
                cp.dispatch_workgroups(gemv_grid(cfg.hidden_size) / rpwr as u32, 1, 1);
            }
            if want(LayerBlock::Dp) && d("dp") {
                cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1);
            }
        }

        if tail {
        if !k("lm") {
        cp.set_pipeline(&self.pipes.gemv_lm_norm);
        cp.set_bind_group(0, &self.bg_gemv_lm_norm, &[]);
        cp.dispatch_workgroups(norm_grid(cfg.vocab_size), 1, 1);
        }
        if d("lm") {
            cp.dispatch_workgroups(norm_grid(cfg.vocab_size), 1, 1);
        }

        if !k("argmax") {
        cp.set_pipeline(&self.pipes.argmax);
        cp.set_bind_group(0, &self.bg_argmax, &[]);
        cp.dispatch_workgroups(1, 1, 1);
        }

        }
    }

    /// Which split-attention pipeline a `cur_len` needs, its grid width and the
    /// chunk count (`write_step_uniforms` puts the same numbers in `u_gp1`).
    ///
    /// Shared by the shipped step and the batched one: both must make the exact
    /// choice `cur_len` implies, or a slot's tokens stop matching its lone run.
    fn split_attn_plan(&self, cur_len: usize) -> (&wgpu::ComputePipeline, u32, u32) {
        let paired = self.cfg.num_attention_heads == 2 * self.cfg.num_key_value_heads
            && !matches!(
                std::env::var("QASR_GQA_PAIR").unwrap_or_default().to_ascii_lowercase().as_str(),
                "0" | "off" | "no"
            );
        let chunk = if paired {
            gqa_split_chunk(cur_len).max(256)
        } else {
            gqa_split_chunk(cur_len)
        } as u32;
        let n_chunks = (cur_len as u32).div_ceil(chunk);
        let split = match (paired, chunk) {
            (true, 512) => &self.pipes.gqa_split_pair512,
            (true, _) => &self.pipes.gqa_split_pair256,
            (false, 128) => &self.pipes.gqa_split128,
            (false, 512) => &self.pipes.gqa_split512,
            (false, _) => &self.pipes.gqa_split256,
        };
        let split_x = if paired {
            self.cfg.num_key_value_heads
        } else {
            self.cfg.num_attention_heads
        } as u32;
        (split, split_x, n_chunks)
    }

    fn encode_gqa<'pass>(&self, cp: &mut wgpu::ComputePass<'pass>, l: &Layer, cur_len: usize) {
        if cur_len <= GQA_SINGLE_CAP {
            let gqa = if cur_len > 512 { &self.pipes.gqa512 } else { &self.pipes.gqa256 };
            cp.set_pipeline(gqa);
            cp.set_bind_group(0, &l.bg_gqa, &[]);
            cp.dispatch_workgroups(self.cfg.num_attention_heads as u32, 1, 1);
            return;
        }
        // The paired kernel covers both q heads of a kv head per workgroup, so
        // its grid is `nkvh` wide and it has no 128-wide build.
        // `QASR_GQA_PAIR=0` forces the per-q-head kernel so the two can be A/B'd
        // from one binary.
        // `REP == 2` is what makes the pair-named pipeline the paired kernel (see
        // `pair_split_src`); `QASR_GQA_PAIR=0` forces the per-q-head kernel so
        // the two can be A/B'd from one binary.
        let paired = self.cfg.num_attention_heads == 2 * self.cfg.num_key_value_heads
            && !matches!(
                std::env::var("QASR_GQA_PAIR").unwrap_or_default().to_ascii_lowercase().as_str(),
                "0" | "off" | "no"
            );
        let chunk = if paired {
            gqa_split_chunk(cur_len).max(256)
        } else {
            gqa_split_chunk(cur_len)
        } as u32;
        let n_chunks = (cur_len as u32).div_ceil(chunk);
        let split = match (paired, chunk) {
            (true, 512) => &self.pipes.gqa_split_pair512,
            (true, _) => &self.pipes.gqa_split_pair256,
            (false, 128) => &self.pipes.gqa_split128,
            (false, 512) => &self.pipes.gqa_split512,
            (false, _) => &self.pipes.gqa_split256,
        };
        let split_x = if paired {
            self.cfg.num_key_value_heads
        } else {
            self.cfg.num_attention_heads
        } as u32;
        let dup = std::env::var("QASR_DUP").unwrap_or_default();
        cp.set_pipeline(split);
        cp.set_bind_group(0, &l.bg_gqa_split, &[]);
        cp.dispatch_workgroups(split_x, n_chunks, 1);
        if dup == "gqa_p1" {
            cp.dispatch_workgroups(split_x, n_chunks, 1);
        }
        // Diagnostic: the same dispatch with every K read aimed at row 0, so all
        // loads are L1 hits.  Read as the QASR_DUP delta against this arm; the
        // first dispatch is the real one so the output is unaffected.
        if dup == "gqa_p1_coal" {
            cp.set_pipeline(&self.pipes.gqa_split_pair256_coal);
            cp.set_bind_group(0, &l.bg_gqa_split, &[]);
            cp.dispatch_workgroups(split_x, n_chunks, 1);
            cp.set_pipeline(split);
        }
        if dup == "gqa_p1_row0" {
            cp.set_pipeline(&self.pipes.gqa_split_pair256_row0);
            cp.set_bind_group(0, &l.bg_gqa_split, &[]);
            cp.dispatch_workgroups(split_x, n_chunks, 1);
            cp.set_pipeline(split);
        }
        cp.set_pipeline(&self.pipes.gqa_merge);
        cp.set_bind_group(0, &l.bg_gqa_merge, &[]);
        cp.dispatch_workgroups(self.cfg.num_attention_heads as u32, 1, 1);
        if dup == "gqa_merge" {
            cp.dispatch_workgroups(self.cfg.num_attention_heads as u32, 1, 1);
        }
    }

    /// Record the next decode step into a held encoder, without submitting it.
    ///
    /// `step` does this and then waits, so the GPU sits idle for the ~0.4 ms of
    /// CPU the step's 229 dispatches need -- `QASR_RECORD_PAD` prices exactly that
    /// (two extra recordings per step, zero GPU work: +129 ms of decode, +124 of
    /// it in `host submit`, both arms MATCH).  Recording under this API instead
    /// lets the caller do it while the *previous* step is still executing: nothing
    /// here reads a decode result, because the embedding for the next step is read
    /// on the GPU out of `token_buf`.  The one thing that must wait is the uniform
    /// write, which `submit_step` does after the previous token has come back --
    /// the shaders read those uniforms while the step runs, so rewriting them with
    /// a step in flight would corrupt it.
    ///
    /// So a pipelined loop is: `record_step` (overlapped) -> `take_token` (blocks
    /// on the step in flight) -> `submit_step`.  Bit-identical to `step`: same
    /// dispatches in the same order into the same pass, and the uniforms are still
    /// written before the submit that reads them.
    ///
    /// `finish()` is called *here*, and the held value is the finished
    /// `CommandBuffer` — a CB is validated, and still just a value that can be
    /// submitted later.  The first version of this pipeline held the encoder and
    /// finished it in `submit_step`, which left ~0.14 ms/token of validation on
    /// the critical side of the await and made the whole change worth only -15 ms
    /// of a 2586 ms decode; moving it across is the difference between hiding the
    /// record and hiding the record *and* its validation.
    pub fn record_step(&mut self) -> Result<()> {
        anyhow::ensure!(self.pending.is_none(), "step already recorded");
        let pos = self.pos;
        let t_host = std::time::Instant::now();
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        self.encode_step(&mut enc, pos);
        self.pending = Some(enc.finish());
        // Charged to `host submit` even though nothing is submitted yet: it is the
        // same host work `step()` charges there, and it has to stay in the line or
        // the phase split stops meaning anything in this loop (`submit` would drop
        // from ~0.4 ms/token to ~0.03 and the record would vanish from the
        // accounting while still costing time whenever it exceeds the overlap).
        // What differs is *when* it runs -- under the previous step -- so in the
        // pipelined loop `submit` and `read` overlap and their sum is longer than
        // the wall by the overlap; `read` is the part of the previous step's
        // execution the record did not cover.
        self.host_submit_ms += t_host.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// Write the recorded step's uniforms, submit it, arm its token readback, and
    /// advance `pos`.
    pub fn submit_step(&mut self) -> Result<()> {
        let cb = self.pending.take().context("no recorded step")?;
        let t_host = std::time::Instant::now();
        self.write_step_uniforms(self.pos);
        self.gpu.queue.submit([cb]);
        // Arm the map now so it resolves with *this* step: the copy that fills
        // `token_staging` is the last command in the buffer just submitted.
        let slice = self.scratch.token_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r.map_err(|e| anyhow::anyhow!("map token: {e}")));
        });
        self.pending_map = Some(rx);
        self.pos += 1;
        self.host_submit_ms += t_host.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// Drop a recorded-but-unsubmitted step without running it.
    ///
    /// The pipelined loop takes step `k-1`'s token *before* submitting step `k`,
    /// so an EOS there means the step just recorded must not run.  It also has to
    /// stop being held: `record_step`'s invariant is one recorded step at a time,
    /// and a decode that ends early leaves the decoder to be used again (the next
    /// `transcribe` on the same instance, batched or streaming).  Not dropping it
    /// made that *second* decode fail with "step already recorded" -- and since
    /// every fixture EOS-terminates before `max_new`, that was every second call on
    /// an instance, on the default path.
    pub fn discard_step(&mut self) {
        self.pending = None;
    }

    /// The token of the step in flight, blocking until the GPU hands it over.
    pub fn take_token(&mut self) -> Result<i32> {
        let rx = self.pending_map.take().context("no step in flight")?;
        let t_read = std::time::Instant::now();
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("poll token")?;
        rx.recv().context("map callback dropped")??;
        let slice = self.scratch.token_staging.slice(..);
        let v = {
            let d = slice.get_mapped_range()?;
            i32::from_le_bytes([d[0], d[1], d[2], d[3]])
        };
        self.scratch.token_staging.unmap();
        self.host_read_ms += t_read.elapsed().as_secs_f64() * 1000.0;
        Ok(v)
    }

    /// One decode step at `self.pos`, then `pos += 1`.  Returns the token argmax
    /// wrote into `token_buf[0]` (which the *next* step will embed).
    pub fn step(&mut self) -> Result<i32> {
        let pos = self.pos;
        let t_host = std::time::Instant::now();
        self.write_step_uniforms(pos);
        let layers = self.layers.len();
        let n = submit_split().min(layers.max(1));
        if n > 1 {
            // Record slice by slice and submit each as it is recorded: the GPU
            // starts on slice 0 (and on the shared-memory latency behind it) while
            // the CPU is still recording the rest of the step.  Every slice's
            // uniforms were written above, before the first submit, so nothing
            // here races a step in flight.
            for c in 0..n {
                let (lo, hi, embed, tail) = step_slice(c, n, layers);
                let mut enc = self.gpu.device.create_command_encoder(&Default::default());
                self.encode_step_slice(&mut enc, pos, lo, hi, embed, tail);
                self.gpu.queue.submit([enc.finish()]);
            }
        } else {
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            let m = pass_split().min(layers.max(1));
            if m > 1 {
                for c in 0..m {
                    let (lo, hi, embed, tail) = step_slice(c, m, layers);
                    self.encode_step_slice(&mut enc, pos, lo, hi, embed, tail);
                }
            } else {
                self.encode_step(&mut enc, pos);
            }
            // Pessimisation probe for the step's *record* cost.  `QASR_RECORD_PAD=n`
            // records the same step `n` more times into a throwaway encoder and never
            // submits it, so it adds pure CPU work and exactly zero GPU work: a 1:1
            // rise in the step time says the record is dead time the GPU waits out,
            // and a smaller rise says part of it is already hidden.  Both arms have to
            // MATCH -- the probe cannot change the tokens.
            //
            // It lives in the single-submit arm, so reading it as it was read for the
            // +129 ms above needs `QASR_SUBMIT_SPLIT=1` as well as `QASR_DEC_PIPE=0`:
            // the shipped default is the two-submit step, which returns at `n > 1`
            // above and never reaches this.  The probe prices *this* arm's record;
            // the pipelined arm's is priced by the A/B in `pipelined_decode`.
            if let Some(n) = record_pad() {
                for _ in 0..n {
                    let mut throw = self.gpu.device.create_command_encoder(&Default::default());
                    self.encode_step(&mut throw, pos);
                    std::mem::drop(throw.finish());
                }
            }
            self.gpu.queue.submit([enc.finish()]);
        }
        self.host_submit_ms += t_host.elapsed().as_secs_f64() * 1000.0;
        let t_read = std::time::Instant::now();
        let v = self.read_token()?;
        self.host_read_ms += t_read.elapsed().as_secs_f64() * 1000.0;
        self.pos += 1;
        Ok(v)
    }

    /// Dispatch only the embedding lookup.
    pub fn debug_encode_embed(&self, enc: &mut wgpu::CommandEncoder) {
        let mut cp = enc.begin_compute_pass(&Default::default());
        self.debug_dispatch_embed(&mut cp);
    }

    /// Embed dispatch into an already-open compute pass.
    pub fn debug_dispatch_embed(&self, cp: &mut wgpu::ComputePass) {
        cp.set_pipeline(&self.pipes.embed);
        cp.set_bind_group(0, &self.bg_embed, &[]);
        cp.dispatch_workgroups(1, 1, 1);
    }

    /// The single-block attention pipeline (probe scaffolding).
    pub fn gqa512_pipe(&self) -> &wgpu::ComputePipeline {
        &self.pipes.gqa512
    }

    /// Layer `l`'s single-block attention bind group (probe scaffolding).
    pub fn gqa_bind(&self, l: usize) -> &wgpu::BindGroup {
        &self.layers[l].bg_gqa
    }

    /// Per-step uniforms for `pos` (probe scaffolding).
    pub fn debug_write_step_uniforms(&self, pos: usize) {
        self.write_step_uniforms(pos);
    }

    /// Dispatch only layer `l`'s ops.  `h` must already hold the incoming
    /// residual; per-step uniforms are written for `pos`.
    pub fn debug_encode_layer(&self, enc: &mut wgpu::CommandEncoder, l: usize, pos: usize) {
        self.write_step_uniforms(pos);
        let cfg = &self.cfg;
        let cur_len = pos + 1;
        let gemv_grid = |rows: usize| (rows / gemv_rpw()) as u32;
        let silu_grid = grid_xy((cfg.intermediate_size / 2).div_ceil(256));
        let layer = &self.layers[l];
        let mut cp = enc.begin_compute_pass(&Default::default());
        let w8 = self.int8;
        cp.set_pipeline(&self.pipes.rms_norm);
        cp.set_bind_group(0, &layer.bg_rms1, &[]);
        cp.dispatch_workgroups(1, 1, 1);
        cp.set_pipeline(if w8 { &self.pipes.gemv_w8_qkv } else { &self.pipes.gemv_qkv });
        cp.set_bind_group(
            0,
            if w8 {
                layer.bg_gemv_w8_qkv.as_ref().expect("int8 bind groups")
            } else {
                &layer.bg_gemv_qkv
            },
            &[],
        );
        cp.dispatch_workgroups(gemv_grid(cfg.fused_qkv_cols()), 1, 1);
        cp.set_pipeline(&self.pipes.extract);
        cp.set_bind_group(0, &layer.bg_extract, &[]);
        cp.dispatch_workgroups(1, (cfg.num_attention_heads + cfg.num_key_value_heads) as u32, 1);
        self.encode_gqa(&mut cp, layer, cur_len);
        cp.set_pipeline(if w8 { &self.pipes.gemv_w8_o } else { &self.pipes.gemv_o });
        cp.set_bind_group(
            0,
            if w8 {
                layer.bg_gemv_w8_o.as_ref().expect("int8 bind groups")
            } else {
                &layer.bg_gemv_o
            },
            &[],
        );
        cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1);
        cp.set_pipeline(&self.pipes.rms_norm);
        cp.set_bind_group(0, &layer.bg_rms2, &[]);
        cp.dispatch_workgroups(1, 1, 1);
        cp.set_pipeline(if w8 { &self.pipes.gemv_w8_gu } else { &self.pipes.gemv_gu });
        cp.set_bind_group(
            0,
            if w8 {
                layer.bg_gemv_w8_gu.as_ref().expect("int8 bind groups")
            } else {
                &layer.bg_gemv_gu
            },
            &[],
        );
        cp.dispatch_workgroups(gemv_grid(2 * cfg.intermediate_size), 1, 1);
        cp.set_pipeline(&self.pipes.silu);
        cp.set_bind_group(0, &self.bg_silu, &[]);
        cp.dispatch_workgroups(silu_grid.0, silu_grid.1, 1);
        cp.set_pipeline(if w8 { &self.pipes.gemv_w8_dp } else { &self.pipes.gemv_dp });
        cp.set_bind_group(
            0,
            if w8 {
                layer.bg_gemv_w8_dp.as_ref().expect("int8 bind groups")
            } else {
                &layer.bg_gemv_dp
            },
            &[],
        );
        cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1);
    }

    /// Dispatch the decoder tail: final norm + lm_head + argmax.
    pub fn debug_encode_tail(&self, enc: &mut wgpu::CommandEncoder) {
        let mut cp = enc.begin_compute_pass(&Default::default());
        self.debug_dispatch_tail(&mut cp);
    }

    /// Tail dispatches into an already-open compute pass.
    pub fn debug_dispatch_tail(&self, cp: &mut wgpu::ComputePass) {
        cp.set_pipeline(&self.pipes.rms_norm);
        cp.set_bind_group(0, &self.bg_final_rms, &[]);
        cp.dispatch_workgroups(1, 1, 1);
        cp.set_pipeline(&self.pipes.gemv_lm);
        cp.set_bind_group(0, &self.bg_gemv_lm, &[]);
        cp.dispatch_workgroups((self.cfg.vocab_size / 8) as u32, 1, 1);
        cp.set_pipeline(&self.pipes.argmax);
        cp.set_bind_group(0, &self.bg_argmax, &[]);
        cp.dispatch_workgroups(1, 1, 1);
    }

    /// Read back the first bytes of the embed table (probe scaffolding).
    pub fn debug_embed_table_bytes(&self, n: u64) -> Result<Vec<u8>> {
        self.gpu.readback(&self.embed_table, n)
    }

    /// Dispatch a single one of layer `l`'s nine ops (probe scaffolding):
    /// 0=rms1, 1=gemv_qkv, 2=extract, 3=gqa, 4=gemv_o, 5=rms2, 6=gemv_gu,
    /// 7=silu, 8=gemv_dp.  Per-step uniforms are written for `pos`.
    pub fn debug_encode_op(&self, enc: &mut wgpu::CommandEncoder, l: usize, pos: usize, op: usize) {
        self.write_step_uniforms(pos);
        let mut cp = enc.begin_compute_pass(&Default::default());
        self.debug_dispatch_op(&mut cp, l, pos, op);
    }

    /// Single layer-op dispatch into an already-open compute pass.
    pub fn debug_dispatch_op(&self, cp: &mut wgpu::ComputePass, l: usize, pos: usize, op: usize) {
        let cfg = &self.cfg;
        let gemv_grid = |rows: usize| (rows / gemv_rpw()) as u32;
        let layer = &self.layers[l];
        let w8 = self.int8;
        match op {
            0 => { cp.set_pipeline(&self.pipes.rms_norm); cp.set_bind_group(0, &layer.bg_rms1, &[]); cp.dispatch_workgroups(1, 1, 1); }
            1 => {
                cp.set_pipeline(if w8 { &self.pipes.gemv_w8_qkv } else { &self.pipes.gemv_qkv });
                cp.set_bind_group(0, if w8 { layer.bg_gemv_w8_qkv.as_ref().expect("int8 bind groups") } else { &layer.bg_gemv_qkv }, &[]);
                cp.dispatch_workgroups(gemv_grid(cfg.fused_qkv_cols()), 1, 1);
            }
            2 => { cp.set_pipeline(&self.pipes.extract); cp.set_bind_group(0, &layer.bg_extract, &[]); cp.dispatch_workgroups(1, (cfg.num_attention_heads + cfg.num_key_value_heads) as u32, 1); }
            3 => self.encode_gqa(cp, layer, pos + 1),
            4 => {
                cp.set_pipeline(if w8 { &self.pipes.gemv_w8_o } else { &self.pipes.gemv_o });
                cp.set_bind_group(0, if w8 { layer.bg_gemv_w8_o.as_ref().expect("int8 bind groups") } else { &layer.bg_gemv_o }, &[]);
                cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1);
            }
            5 => { cp.set_pipeline(&self.pipes.rms_norm); cp.set_bind_group(0, &layer.bg_rms2, &[]); cp.dispatch_workgroups(1, 1, 1); }
            6 => {
                cp.set_pipeline(if w8 { &self.pipes.gemv_w8_gu } else { &self.pipes.gemv_gu });
                cp.set_bind_group(0, if w8 { layer.bg_gemv_w8_gu.as_ref().expect("int8 bind groups") } else { &layer.bg_gemv_gu }, &[]);
                cp.dispatch_workgroups(gemv_grid(2 * cfg.intermediate_size), 1, 1);
            }
            7 => { cp.set_pipeline(&self.pipes.silu); cp.set_bind_group(0, &self.bg_silu, &[]); cp.dispatch_workgroups(((cfg.intermediate_size / 2).div_ceil(256)) as u32, 1, 1); }
            8 => {
                cp.set_pipeline(if w8 { &self.pipes.gemv_w8_dp } else { &self.pipes.gemv_dp });
                cp.set_bind_group(0, if w8 { layer.bg_gemv_w8_dp.as_ref().expect("int8 bind groups") } else { &layer.bg_gemv_dp }, &[]);
                cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1);
            }
            _ => panic!("debug_encode_op: op {op} out of range"),
        }
    }

    fn read_token(&self) -> Result<i32> {
        let slice = self.scratch.token_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("poll token")?;
        rx.recv().context("map callback dropped")?.context("map token")?;
        let v = {
            let d = slice.get_mapped_range()?;
            i32::from_le_bytes([d[0], d[1], d[2], d[3]])
        };
        self.scratch.token_staging.unmap();
        Ok(v)
    }

    /// Read any scratch buffer back as f16 values.
    pub fn read_f16(&self, buf: &wgpu::Buffer, elems: usize) -> Result<Vec<half::f16>> {
        let bytes = self.gpu.readback(buf, (elems * 2) as u64)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
            .collect())
    }

    /// Arm the per-step hidden-state dump: `max_rows` rows of `hidden` f16, one
    /// per decode step from here on (see [`Self::decode_h_buf`]).
    ///
    /// Rows are addressed by `pos`, so the caller only has to keep `self.pos`
    /// honest -- which it already is: `step` and `submit_step` advance it once per
    /// executed step and nothing else touches it.
    pub fn start_decode_state_dump(&mut self, max_rows: usize) -> Result<()> {
        let bytes = max_rows.max(1) as u64 * (self.cfg.hidden_size * 2) as u64;
        self.decode_h_base_pos = self.pos;
        self.decode_h_buf = Some(self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("decode_h"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        Ok(())
    }

    /// Read the armed dump back and disarm it.
    ///
    /// `rows` is `pos - base`, i.e. the steps that actually executed (a
    /// `discard_step`'d record wrote no row and advanced nothing), and row `k` is
    /// the hidden state whose logits predicted the token *after* the first one.
    pub fn take_decode_states(&mut self) -> Result<Option<DecodeStates>> {
        let Some(buf) = self.decode_h_buf.take() else {
            return Ok(None);
        };
        let hidden = self.cfg.hidden_size;
        let rows = self.pos.saturating_sub(self.decode_h_base_pos);
        let states = if rows == 0 {
            Vec::new()
        } else {
            self.gpu.readback_at(&buf, 0, (rows * hidden * 2) as u64)?
        };
        Ok(Some(DecodeStates { states, rows, hidden, base_pos: self.decode_h_base_pos }))
    }

    /// Timed decode without per-step host synchronisation: the token stays on the
    /// GPU (embed reads `token_buf`), so this measures pure GPU-bound ms/token.
    pub fn bench_steps(&mut self, steps: usize, warmup: usize) -> Result<f64> {
        for _ in 0..warmup {
            let pos = self.pos;
            self.write_step_uniforms(pos);
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            self.encode_step(&mut enc, pos);
            self.gpu.queue.submit([enc.finish()]);
            self.pos += 1;
        }
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("warmup poll")?;

        let t0 = Instant::now();
        for _ in 0..steps {
            let pos = self.pos;
            self.write_step_uniforms(pos);
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            self.encode_step(&mut enc, pos);
            self.gpu.queue.submit([enc.finish()]);
            self.pos += 1;
        }
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("bench poll")?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
        Ok(ms)
    }

    /// Bytes of weight traffic per decode step (for the bandwidth figure).
    pub fn step_weight_bytes(&self) -> u64 {
        let c = &self.cfg;
        let elems = (c.fused_qkv_cols() * c.hidden_size
            + c.hidden_size * c.q_dim()
            + 2 * c.intermediate_size * c.hidden_size
            + c.hidden_size * c.intermediate_size) as u64;
        // int8 residency: one byte per weight plus the per-row f32 scale of
        // each matrix (`lm_head`/embed stay f16 either way).
        let per_layer = if self.int8 {
            elems + (c.fused_qkv_cols() + c.hidden_size + 2 * c.intermediate_size + c.hidden_size) as u64 * 4
        } else {
            elems * 2
        };
        per_layer * c.num_hidden_layers as u64 + (c.vocab_size * c.hidden_size) as u64 * 2
    }

    pub fn embed_table(&self) -> &wgpu::Buffer {
        &self.embed_table
    }

    /// A second decoder that shares **this one's weights and pipelines**.
    ///
    /// `wgpu::Buffer` clones alias the same allocation, so N slots cost one copy
    /// of the weights; only the KV cache, the scratch activations and the bind
    /// groups that reference them are new.  Nothing about a fork changes what a
    /// dispatch computes: it is the same kernel, the same bindings and the same
    /// order as [`Self::step`], so a forked slot's token stream is bit-identical
    /// to running it alone.  That is the property batching is built on -- it may
    /// change *when* the GPU sees a dispatch, never *what* it computes.
    pub fn fork(&self) -> Result<Self> {
        self.fork_inner(None)
    }

    /// A fork whose KV is allocated at exactly `cap` positions.
    ///
    /// [`Self::fork`] inherits this decoder's `cap`, and `ensure_capacity` only
    /// grows -- so a fork cannot be made *smaller* through it.  The batch path
    /// needs the prefill's KV and the slot's region to share a stride, which means
    /// the prefill fork has to be told the exact cap rather than inherit one.
    pub fn fork_capped(&self, cap: usize) -> Result<Self> {
        self.fork_inner(Some(cap))
    }

    fn fork_inner(&self, cap_override: Option<usize>) -> Result<Self> {
        let gpu = &self.gpu;
        let cfg = &self.cfg;
        let pipes = &self.pipes;
        let cap = cap_override.unwrap_or(self.cap);
        let hs = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let inter = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let nqh = cfg.num_attention_heads;
        let nkvh = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let kv_words = nkvh * cap * hd / 2;
        let max_chunks = self.max_seq.div_ceil(256);

        let mut up = gpu.uploader();
        let scratch = Scratch {
            h: up.storage("h", (hs / 2 * 4) as u64),
            norm1: up.storage("norm1", (hs / 2 * 4) as u64),
            qkv: up.storage("qkv", (cfg.fused_qkv_cols() / 2 * 4) as u64),
            q_out: up.storage("q_out", (q_dim / 2 * 4) as u64),
            attn_out: up.storage("attn_out", (q_dim / 2 * 4) as u64),
            norm2: up.storage("norm2", (hs / 2 * 4) as u64),
            gate_up: up.storage("gate_up", (inter * 4) as u64),
            activated: up.storage("activated", (inter / 2 * 4) as u64),
            final_norm: up.storage("final_norm", (hs / 2 * 4) as u64),
            logits: up.storage("logits", (vocab / 2 * 4) as u64),
            token: up.storage("token", 4),
            token_staging: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("token_staging"),
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            // Same size as the source's: the tables themselves are copied over
            // below, because they are the model's and not the slot's.
            cos: up.storage("cos", self.scratch.cos.size()),
            sin: up.storage("sin", self.scratch.sin.size()),
            u_rms: up.uniform("u_rms", 16),
            u_qkvx: up.uniform("u_qkvx", 32),
            u_silu: up.uniform("u_silu", 16),
            u_gqa: up.uniform("u_gqa", 16),
            u_gp1: up.uniform("u_gp1", 16),
            u_gp2: up.uniform("u_gp2", 16),
            split_part_out: up.storage("split_part_out", (nqh * max_chunks * hd * 4) as u64),
            split_part_max: up.storage("split_part_max", (nqh * max_chunks * 4) as u64),
            split_part_sum: up.storage("split_part_sum", (nqh * max_chunks * 4) as u64),
        };
        let q_res = scratch.q_out.as_entire_binding();
        let out_res = scratch.attn_out.as_entire_binding();
        up.upload(
            &scratch.u_rms,
            bytemuck::bytes_of(&RmsCfg { eps: cfg.rms_norm_eps, _a: 0.0, _b: 0.0, _c: 0.0 }),
        )?;
        let u_argmax = up.uniform("u_argmax", 16);
        let u_embed = up.uniform("u_embed", 16);
        up.upload(
            &u_argmax,
            bytemuck::bytes_of(&ArgmaxCfg { n: vocab as u32, slot: 0, _a: 0, _b: 0 }),
        )?;
        up.upload(
            &u_embed,
            bytemuck::bytes_of(&EmbedCfg { slot: 0, d2: (hs / 2) as u32, _a: 0, _b: 0 }),
        )?;

        let gemv_bg = |pipe: &wgpu::ComputePipeline,
                       wt: &wgpu::Buffer,
                       x: &wgpu::Buffer,
                       y: &wgpu::Buffer|
         -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gemv"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                ],
            })
        };
        let gemv_w8_bg = |pipe: &wgpu::ComputePipeline,
                          wt: &wgpu::Buffer,
                          x: &wgpu::Buffer,
                          y: &wgpu::Buffer,
                          s: &wgpu::Buffer|
         -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gemv_w8"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: s.as_entire_binding() },
                ],
            })
        };
        let gemv_norm_w8_bg = |pipe: &wgpu::ComputePipeline,
                               wt: &wgpu::Buffer,
                               x: &wgpu::Buffer,
                               y: &wgpu::Buffer,
                               nw: &wgpu::Buffer,
                               s: &wgpu::Buffer|
         -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gemv_norm_w8"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: nw.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: s.as_entire_binding() },
                ],
            })
        };
        let gemv_norm_bg = |pipe: &wgpu::ComputePipeline,
                            wt: &wgpu::Buffer,
                            x: &wgpu::Buffer,
                            y: &wgpu::Buffer,
                            nw: &wgpu::Buffer|
         -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gemv_norm"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: nw.as_entire_binding() },
                ],
            })
        };
        let rms_bg = |x: &wgpu::Buffer, wt: &wgpu::Buffer, out: &wgpu::Buffer| -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rms"),
                layout: &pipes.rms_norm.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: scratch.u_rms.as_entire_binding() },
                ],
            })
        };

        let mut layers = Vec::with_capacity(self.layers.len());
        for src in self.layers.iter() {
            let k_cache = gpu.storage("k_cache", (kv_words * 4) as u64);
            let v_cache = gpu.storage("v_cache", (kv_words * 4) as u64);
            let (k_res, v_res) = (
                k_cache.as_entire_binding(),
                v_cache.as_entire_binding(),
            );
            let (qkv, o, gu, dp) =
                (src.qkv_w.clone(), src.o_w.clone(), src.gu_w.clone(), src.dp_w.clone());
            let (iln, pln, qn, kn) =
                (src.iln_w.clone(), src.pln_w.clone(), src.qn_w.clone(), src.kn_w.clone());
            let (qkv_s, o_s, gu_s, dp_s) =
                (src.qkv_s.clone(), src.o_s.clone(), src.gu_s.clone(), src.dp_s.clone());

            let bg_gemv_qkv = gemv_bg(&pipes.gemv_qkv, &qkv, &scratch.norm1, &scratch.qkv);
            let bg_gemv_qkv_norm =
                gemv_norm_bg(&pipes.gemv_qkv_norm, &qkv, &scratch.h, &scratch.qkv, &iln);
            let bg_gemv_o = gemv_bg(&pipes.gemv_o, &o, &scratch.attn_out, &scratch.h);
            let bg_gemv_o_rpwr2 =
                gemv_bg(&pipes.gemv_o_rpwr2, &o, &scratch.attn_out, &scratch.h);
            let bg_gemv_gu = gemv_bg(&pipes.gemv_gu, &gu, &scratch.norm2, &scratch.gate_up);
            let bg_gemv_gu_norm =
                gemv_norm_bg(&pipes.gemv_gu_norm, &gu, &scratch.h, &scratch.gate_up, &pln);
            let bg_gemv_dp = gemv_bg(&pipes.gemv_dp, &dp, &scratch.activated, &scratch.h);
            let bg_gemv_dp_rpwr2 =
                gemv_bg(&pipes.gemv_dp_rpwr2, &dp, &scratch.activated, &scratch.h);
            let (bg_gemv_w8_qkv, bg_gemv_w8_gu, bg_gemv_w8_qkv_norm, bg_gemv_w8_o, bg_gemv_w8_o_rpwr2, bg_gemv_w8_gu_norm, bg_gemv_w8_dp, bg_gemv_w8_dp_rpwr2) =
                match (&qkv_s, &o_s, &gu_s, &dp_s) {
                    (Some(qs), Some(os), Some(gs), Some(ds)) => (
                        Some(gemv_w8_bg(&pipes.gemv_w8_qkv, &qkv, &scratch.norm1, &scratch.qkv, qs)),
                        Some(gemv_w8_bg(&pipes.gemv_w8_gu, &gu, &scratch.norm2, &scratch.gate_up, gs)),
                        Some(gemv_norm_w8_bg(&pipes.gemv_w8_qkv_norm, &qkv, &scratch.h, &scratch.qkv, &iln, qs)),
                        Some(gemv_w8_bg(&pipes.gemv_w8_o, &o, &scratch.attn_out, &scratch.h, os)),
                        Some(gemv_w8_bg(&pipes.gemv_w8_o_rpwr2, &o, &scratch.attn_out, &scratch.h, os)),
                        Some(gemv_norm_w8_bg(&pipes.gemv_w8_gu_norm, &gu, &scratch.h, &scratch.gate_up, &pln, gs)),
                        Some(gemv_w8_bg(&pipes.gemv_w8_dp, &dp, &scratch.activated, &scratch.h, ds)),
                        Some(gemv_w8_bg(&pipes.gemv_w8_dp_rpwr2, &dp, &scratch.activated, &scratch.h, ds)),
                    ),
                    _ => (None, None, None, None, None, None, None, None),
                };
            let bg_rms1 = rms_bg(&scratch.h, &iln, &scratch.norm1);
            let bg_rms2 = rms_bg(&scratch.h, &pln, &scratch.norm2);
            let (bg_extract, bg_gqa, bg_gqa_split) = kv_bind_groups_res(
                gpu,
                pipes,
                &scratch,
                &qn,
                &kn,
                q_res.clone(),
                k_res,
                v_res,
                out_res.clone(),
            );
            let bg_gqa_merge = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gqa_merge"),
                layout: &pipes.gqa_merge.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: scratch.attn_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: scratch.split_part_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: scratch.split_part_max.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: scratch.split_part_sum.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: scratch.u_gp2.as_entire_binding() },
                ],
            });
            layers.push(Layer {
                k_cache: k_cache.clone(),
                v_cache: v_cache.clone(),
                iln_w: iln,
                pln_w: pln,
                qn_w: qn,
                kn_w: kn,
                qkv_w: qkv,
                o_w: o,
                gu_w: gu,
                dp_w: dp,
                qkv_s,
                o_s,
                gu_s,
                dp_s,
                bg_gemv_qkv,
                bg_gemv_qkv_norm,
                bg_gemv_o,
                bg_gemv_o_rpwr2,
                bg_gemv_gu,
                bg_gemv_gu_norm,
                bg_gemv_dp,
                bg_gemv_dp_rpwr2,
                bg_gemv_w8_qkv,
                bg_gemv_w8_gu,
                bg_gemv_w8_qkv_norm,
                bg_gemv_w8_o,
                bg_gemv_w8_o_rpwr2,
                bg_gemv_w8_gu_norm,
                bg_gemv_w8_dp,
                bg_gemv_w8_dp_rpwr2,
                bg_rms1,
                bg_rms2,
                bg_extract,
                bg_gqa,
                bg_gqa_split,
                bg_gqa_merge,
            });
        }

        let norm_buf = self.norm_buf.clone();
        let embed_table = self.embed_table.clone();
        let bg_final_rms = rms_bg(&scratch.h, &norm_buf, &scratch.final_norm);
        let bg_gemv_lm = gemv_bg(&pipes.gemv_lm, &embed_table, &scratch.final_norm, &scratch.logits);
        let bg_gemv_lm_norm =
            gemv_norm_bg(&pipes.gemv_lm_norm, &embed_table, &scratch.h, &scratch.logits, &norm_buf);
        let bg_silu = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("silu"),
            layout: &pipes.silu.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: scratch.gate_up.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scratch.activated.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scratch.u_silu.as_entire_binding() },
            ],
        });
        let bg_argmax = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("argmax"),
            layout: &pipes.argmax.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: scratch.logits.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scratch.token.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: u_argmax.as_entire_binding() },
            ],
        });
        let bg_embed = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embed"),
            layout: &pipes.embed.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: embed_table.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scratch.token.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scratch.h.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: u_embed.as_entire_binding() },
            ],
        });
        up.finish()?;

        // The MRoPE tables are a property of the model, not of a slot: copy
        // them so a fork is ready to decode without re-uploading from the host.
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let tbl = self.scratch.cos.size().min(scratch.cos.size());
        enc.copy_buffer_to_buffer(&self.scratch.cos, 0, &scratch.cos, 0, tbl);
        enc.copy_buffer_to_buffer(&self.scratch.sin, 0, &scratch.sin, 0, tbl);
        gpu.queue.submit([enc.finish()]);

        Ok(Self {
            gpu: self.gpu.clone(),
            cfg: self.cfg.clone(),
            max_seq: self.max_seq,
            cap,
            pos: 0,
            host_submit_ms: 0.0,
            host_read_ms: 0.0,
            pending: None,
            pending_map: None,
            pooled: false,
            prefill_scratch: None,
            multi_scratch: None,
            embed_table,
            layers,
            pipes: self.pipes.clone(),
            int8: self.int8,
            scratch,
            bg_final_rms,
            bg_gemv_lm,
            bg_gemv_lm_norm,
            bg_silu,
            bg_argmax,
            bg_embed,
            norm_buf,
            debug_prefill_h: None,
            debug_l0_h: None,
            decode_h_buf: None,
            decode_h_base_pos: 0,
        })
    }

    /// One decode step for every slot in `slots`.
    ///
    /// Every slot's uniforms are written first, then **one** command buffer and
    /// **one** compute pass carry the dispatches of layer `l` for all slots
    /// back-to-back, then the epilogue for all slots, then **one** submit and
    /// **one** token readback.  A slot's arithmetic never depends on another
    /// slot, so the token stream is bit-identical to calling [`Self::step`] on
    /// each in turn; what changes is that the weight tile slot 0 pulls into L2 is
    /// the one the rest find there.
    ///
    /// `n == 1` dispatches to [`Self::step`] unchanged: the b=1 path is the old
    /// code, not this one with a loop of length one.
    pub fn step_batch(slots: &mut [WgpuTextDecoder]) -> Result<Vec<i32>> {
        if slots.len() == 1 {
            return Ok(vec![slots[0].step()?]);
        }
        let n = slots.len();
        for d in slots.iter() {
            d.write_step_uniforms(d.pos);
        }
        let gpu = slots[0].gpu.clone();
        let nl = slots[0].layers.len();
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_tokens"),
            size: (4 * n) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            for l in 0..nl {
                for block in LAYER_BLOCKS {
                    for d in slots.iter() {
                        d.encode_into(
                            &mut cp,
                            d.pos,
                            l,
                            l + 1,
                            l == 0 && block == LayerBlock::Qkv,
                            false,
                            &|b| b == block,
                        );
                    }
                }
            }
            for d in slots.iter() {
                d.encode_into(&mut cp, d.pos, nl, nl, false, true, &|_| true);
            }
        }
        for (s, d) in slots.iter().enumerate() {
            enc.copy_buffer_to_buffer(&d.scratch.token, 0, &staging, (s * 4) as u64, 4);
        }
        gpu.queue.submit([enc.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).context("poll batch")?;
        rx.recv().context("map callback dropped")?.context("map batch")?;
        let out: Vec<i32> = {
            let d = slice.get_mapped_range()?;
            (0..n)
                .map(|s| i32::from_le_bytes([d[s * 4], d[s * 4 + 1], d[s * 4 + 2], d[s * 4 + 3]]))
                .collect()
        };
        staging.unmap();
        for d in slots.iter_mut() {
            d.pos += 1;
        }
        Ok(out)
    }
}

/// One allocation per layer holding every slot's KV, plus the two per-slot
/// activation tensors attention needs to address by slot index.
///
/// One dispatch for every slot, each slot reading **its own buffers**.
///
/// The batched kernels take `binding_array`s and pick the slot with `wgid.z`
/// ([`shaders::gemv_batch_slots`]), so a slot is *which buffer* and never an
/// offset into a shared one.  That is the whole design: no kernel re-derives a
/// stride, a slot reads the exact bytes it reads alone, and b only changes
/// **when** the GPU sees a dispatch -- never what it computes.  The slots are
/// plain [`WgpuTextDecoder::fork`]s: b costs b KV caches and b scratch sets,
/// one copy of the weights, and nothing has to be copied from a whole-run buffer
/// into a decode-time one.
///
/// What stays per slot is the work that is not a weight stream: `embed`, the
/// three RMSNorms (one workgroup each, the shipped kernel and the shipped bind
/// group), and `argmax`.  What is batched is exactly the linear algebra whose
/// cost is the weight read -- `qkv`, `o`, `gu`, `dp`, `lm` -- plus `extract`,
/// `gqa` and `silu`, which only touch per-slot rows.
/// Most slots one batched dispatch can address (`binding_array<T, 16>`).
///
/// This used to be 8 because the fused norm staged every slot's row in
/// workgroup shared memory; the weight streams now read a normalized row that
/// already sits in each slot's own buffer, so the limit is the binding array's
/// declared width (`binding_array<T, 32>` in the batched kernels).
pub const MAX_BATCH: usize = 32;

const MAX_BATCH_U32: u32 = MAX_BATCH as u32;

pub struct BatchProjections {
    qkv: wgpu::ComputePipeline,
    gu: wgpu::ComputePipeline,
    lm: wgpu::ComputePipeline,
    o: wgpu::ComputePipeline,
    dp: wgpu::ComputePipeline,
    qkv_bg: Vec<wgpu::BindGroup>,
    gu_bg: Vec<wgpu::BindGroup>,
    lm_bg: wgpu::BindGroup,
    o_bg: Vec<wgpu::BindGroup>,
    dp_bg: Vec<wgpu::BindGroup>,
    gqa256: wgpu::ComputePipeline,
    gqa512: wgpu::ComputePipeline,
    /// `[block width][layer]` -- width 0 is the 256-thread kernel, 1 the 512.
    gqa_bg: [Vec<wgpu::BindGroup>; 2],
    /// Per width: every slot's `cur_len`, indexed by the slot the `PICK` table
    /// names.
    gqa_cl: [wgpu::Buffer; 2],
    /// Per width: the slots that dispatch owns, compacted.
    gqa_pick: [wgpu::Buffer; 2],
    gqa_u_bg: wgpu::BindGroup,
    extract: wgpu::ComputePipeline,
    extract_u_bg: wgpu::BindGroup,
    extract_bg: Vec<wgpu::BindGroup>,
    /// One `u32` per slot: this step's write position for `extract`.
    extract_pos: wgpu::Buffer,
    silu: wgpu::ComputePipeline,
    silu_bg: wgpu::BindGroup,
    silu_u_bg: wgpu::BindGroup,
    nqh: u32,
    g_qkv: u32,
    g_o: u32,
    g_gu: u32,
    g_dp: u32,
    g_lm: u32,
    g_silu: (u32, u32),
    pub batch: usize,
}

/// The slots' own buffers as one binding array, padded to the shader's 16 with
/// slot 0: `wgid.z` never reaches the padding because the grid is `batch` deep.
fn bind_array<'a>(buffers: &[&'a wgpu::Buffer]) -> Vec<wgpu::BufferBinding<'a>> {
    let pad = *buffers.first().expect("a batch has at least one slot");
    (0..MAX_BATCH)
        .map(|i| {
            let buffer = buffers.get(i).copied().unwrap_or(pad);
            wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: wgpu::BufferSize::new(buffer.size()),
            }
        })
        .collect()
}

fn batch_subgroup_mode(gpu: &Gpu) -> SubgroupMode {
    match std::env::var("QASR_BATCH_SUBGROUP")
        .ok()
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("0" | "off" | "no" | "false") => SubgroupMode::None,
        _ => gpu.subgroup_mode(),
    }
}

impl BatchProjections {
    /// Build the batched dispatches for `slots`.
    ///
    /// Every slot must be a [`WgpuTextDecoder::fork`] of the same decoder; the
    /// weight buffers are read from slot 0 because a fork aliases them.
    pub fn new(slots: &[WgpuTextDecoder]) -> Result<Self> {
        let batch = slots.len();
        anyhow::ensure!(batch >= 2, "BatchProjections needs two or more slots");
        anyhow::ensure!(
            batch <= MAX_BATCH,
            "batch decode fits at most {MAX_BATCH} slots; use a smaller --batch"
        );
        let base = &slots[0];
        let gpu = &base.gpu;
        let sg_mode = batch_subgroup_mode(gpu);
        anyhow::ensure!(
            gpu.supports_binding_array(),
            "BatchProjections requires wgpu_binding_array support"
        );
        let cfg = &base.cfg;
        let hs = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let inter = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let nl = base.layers.len();
        let buf = |f: &dyn Fn(&WgpuTextDecoder) -> &wgpu::Buffer| -> Vec<&wgpu::Buffer> {
            slots.iter().map(f).collect()
        };

        // One layout for every weight stream: binding 0 is the matrix, 1/2 are
        // the per-slot input/output arrays.
        let (gemv_pl, _) = batch_layout(
            gpu,
            "batch_gemv",
            &[(0, true, None), (1, true, Some(MAX_BATCH_U32)), (2, false, Some(MAX_BATCH_U32))],
            false,
        );
        let o = gpu.pipeline(
            "batch_o",
            &shaders::gemv_batch_slots_mode(hs, q_dim, batch, true, sg_mode),
            "gemv_batch_slots",
            Some(&gemv_pl),
        )?;
        let qkv = gpu.pipeline(
            "batch_qkv",
            &shaders::gemv_batch_slots_mode(cfg.fused_qkv_cols(), hs, batch, false, sg_mode),
            "gemv_batch_slots",
            Some(&gemv_pl),
        )?;
        let gu = gpu.pipeline(
            "batch_gu",
            &shaders::gemv_batch_slots_mode(2 * inter, hs, batch, false, sg_mode),
            "gemv_batch_slots",
            Some(&gemv_pl),
        )?;
        let lm = gpu.pipeline(
            "batch_lm",
            &shaders::gemv_batch_slots_mode(vocab, hs, batch, false, sg_mode),
            "gemv_batch_slots",
            Some(&gemv_pl),
        )?;
        let dp = gpu.pipeline(
            "batch_dp",
            &shaders::gemv_batch_slots_mode(hs, inter, batch, true, sg_mode),
            "gemv_batch_slots",
            Some(&gemv_pl),
        )?;
        let nqh = cfg.num_attention_heads;
        // One shared layout for both block widths: a bind group built from one
        // pipeline's implicit layout is rejected by the other.
        let (gqa_pl, gqa_ubgl) = batch_layout(
            gpu,
            "gqa_batch",
            &[
                (0, true, Some(MAX_BATCH_U32)),
                (1, true, Some(MAX_BATCH_U32)),
                (2, true, Some(MAX_BATCH_U32)),
                (3, false, Some(MAX_BATCH_U32)),
                (5, true, None),
                (6, true, None),
            ],
            true,
        );
        let gqa256 = gpu.pipeline(
            "batch_gqa256",
            &shaders::gqa_decode_single_slots(
                nqh,
                cfg.num_key_value_heads,
                cfg.head_dim,
                256,
                GQA_SINGLE_CAP,
            ),
            "gqa",
            Some(&gqa_pl),
        )?;
        let gqa512 = gpu.pipeline(
            "batch_gqa512",
            &shaders::gqa_decode_single_slots(
                nqh,
                cfg.num_key_value_heads,
                cfg.head_dim,
                512,
                GQA_SINGLE_CAP,
            ),
            "gqa",
            Some(&gqa_pl),
        )?;

        // `0` is the weight matrix (shared, read from slot 0); `1` and `2` are
        // every slot's input / output buffer.
        let bind = |pipe: &wgpu::ComputePipeline,
                    w: &wgpu::Buffer,
                    xs: &[&wgpu::Buffer],
                    ys: &[&wgpu::Buffer]|
         -> wgpu::BindGroup {
            let xb = bind_array(xs);
            let yb = bind_array(ys);
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("batch_proj"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: w.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::BufferArray(&xb),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::BufferArray(&yb),
                    },
                ],
            })
        };

        let mut qkv_bg = Vec::with_capacity(nl);
        let mut gu_bg = Vec::with_capacity(nl);
        for l in 0..nl {
            qkv_bg.push(bind(
                &qkv,
                &base.layers[l].qkv_w,
                &buf(&|d| &d.scratch.norm1),
                &buf(&|d| &d.scratch.qkv),
            ));
            gu_bg.push(bind(
                &gu,
                &base.layers[l].gu_w,
                &buf(&|d| &d.scratch.norm2),
                &buf(&|d| &d.scratch.gate_up),
            ));
        }
        let lm_bg = bind(
            &lm,
            &base.embed_table,
            &buf(&|d| &d.scratch.final_norm),
            &buf(&|d| &d.scratch.logits),
        );

        let mut o_bg = Vec::with_capacity(nl);
        let mut dp_bg = Vec::with_capacity(nl);
        for l in 0..nl {
            o_bg.push(bind(
                &o,
                &base.layers[l].o_w,
                &buf(&|d| &d.scratch.attn_out),
                &buf(&|d| &d.scratch.h),
            ));
            dp_bg.push(bind(
                &dp,
                &base.layers[l].dp_w,
                &buf(&|d| &d.scratch.activated),
                &buf(&|d| &d.scratch.h),
            ));
        }

        let gqa_cl = [
            gpu.storage("batch_cur_len", 4 * MAX_BATCH as u64),
            gpu.storage("batch_cur_len_hi", 4 * MAX_BATCH as u64),
        ];
        let gqa_pick = [
            gpu.storage("batch_pick", 4 * MAX_BATCH as u64),
            gpu.storage("batch_pick_b", 4 * MAX_BATCH as u64),
        ];
        let q = buf(&|d| &d.scratch.q_out);
        let out = buf(&|d| &d.scratch.attn_out);
        let mut bg_a = Vec::with_capacity(nl);
        let mut bg_b = Vec::with_capacity(nl);
        for l in 0..nl {
            let k = buf(&|d| &d.layers[l].k_cache);
            let v = buf(&|d| &d.layers[l].v_cache);
            for (g, dst) in [(0usize, &mut bg_a), (1, &mut bg_b)] {
                let qb = bind_array(&q);
                let kb = bind_array(&k);
                let vb = bind_array(&v);
                let ob = bind_array(&out);
                dst.push(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("batch_gqa"),
                    layout: &gqa512.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::BufferArray(&qb),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::BufferArray(&kb),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::BufferArray(&vb),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::BufferArray(&ob),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: gqa_cl[g].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: gqa_pick[g].as_entire_binding(),
                        },
                    ],
                }));
            }
        }
        let gqa_bg = [bg_a, bg_b];

        let (extract_pl, extract_ubgl) = batch_layout(
            gpu,
            "batch_extract",
            &[
                (0, true, Some(MAX_BATCH_U32)),
                (1, true, None),
                (2, true, None),
                (3, true, None),
                (4, true, None),
                (5, false, Some(MAX_BATCH_U32)),
                (6, false, Some(MAX_BATCH_U32)),
                (7, false, Some(MAX_BATCH_U32)),
                (9, true, None),
            ],
            true,
        );
        let extract = gpu.pipeline(
            "batch_extract",
            &shaders::qkv_extract_slots(
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
            ),
            "qkv_extract",
            Some(&extract_pl),
        )?;
        let extract_pos = gpu.storage("batch_extract_pos", 4 * MAX_BATCH as u64);
        let mut extract_bg = Vec::with_capacity(nl);
        for l in 0..nl {
            let qkvb = bind_array(&buf(&|d| &d.scratch.qkv));
            let qb = bind_array(&q);
            let kb = bind_array(&buf(&|d| &d.layers[l].k_cache));
            let vb = bind_array(&buf(&|d| &d.layers[l].v_cache));
            extract_bg.push(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("batch_extract"),
                layout: &extract.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::BufferArray(&qkvb),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: base.layers[l].qn_w.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: base.layers[l].kn_w.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: slots[0].scratch.cos.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: slots[0].scratch.sin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::BufferArray(&qb),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::BufferArray(&kb),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::BufferArray(&vb),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: extract_pos.as_entire_binding(),
                    },
                ],
            }));
        }

        let (silu_pl, silu_ubgl) = batch_layout(
            gpu,
            "batch_silu",
            &[(0, true, Some(MAX_BATCH_U32)), (1, false, Some(MAX_BATCH_U32))],
            true,
        );
        let silu = gpu.pipeline(
            "batch_silu",
            &shaders::silu_mul_split_slots(inter),
            "silu_mul_split_slots",
            Some(&silu_pl),
        )?;
        let gub = bind_array(&buf(&|d| &d.scratch.gate_up));
        let actb = bind_array(&buf(&|d| &d.scratch.activated));
        let silu_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch_silu"),
            layout: &silu.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::BufferArray(&gub),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::BufferArray(&actb),
                },
            ],
        });

        // Group 1: the per-step config (the per-slot arrays own group 0).
        let uni_bg = |layout: &wgpu::BindGroupLayout,
                      buf: &wgpu::Buffer,
                      label: &str|
         -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            })
        };
        let gqa_u_bg = uni_bg(
            gqa_ubgl.as_ref().expect("gqa has a uniform"),
            &slots[0].scratch.u_gqa,
            "batch_gqa_cfg",
        );
        let extract_u_bg = uni_bg(
            extract_ubgl.as_ref().expect("extract has a uniform"),
            &slots[0].scratch.u_qkvx,
            "batch_extract_cfg",
        );
        let silu_u_bg = uni_bg(
            silu_ubgl.as_ref().expect("silu has a uniform"),
            &slots[0].scratch.u_silu,
            "batch_silu_cfg",
        );

        let out = Self {
            g_qkv: (cfg.fused_qkv_cols() / 8) as u32,
            g_o: (hs / 8) as u32,
            g_gu: (2 * inter / 8) as u32,
            g_dp: (hs / 8) as u32,
            g_lm: (vocab / 8) as u32,
            g_silu: grid_xy((inter / 2).div_ceil(256)),
            qkv,
            gu,
            lm,
            o,
            dp,
            qkv_bg,
            gu_bg,
            lm_bg,
            o_bg,
            dp_bg,
            gqa_bg,
            gqa_cl,
            gqa_pick,
            gqa_u_bg,
            silu,
            silu_bg,
            silu_u_bg,
            extract,
            extract_bg,
            extract_u_bg,
            extract_pos,
            nqh: nqh as u32,
            gqa256,
            gqa512,
            batch,
        };
        if let Err(e) = gpu.save_pipeline_cache() {
            eprintln!("[pipeline cache] save skipped: {e:#}");
        }
        Ok(out)
    }

    /// One step for every slot: five weight streams once, everything else per
    /// slot with the shipped kernel.
    pub fn step(&self, slots: &mut [WgpuTextDecoder]) -> Result<Vec<i32>> {
        let n = slots.len();
        anyhow::ensure!(n == self.batch, "BatchProjections: slot count changed");
        let gpu = slots[0].gpu.clone();
        for d in slots.iter() {
            d.write_step_uniforms(d.pos);
        }
        let nl = slots[0].layers.len();

        // The only per-slot scalars the batched kernels need: this step's write
        // position, and which block width owns each slot.
        let mut pos = [0u32; MAX_BATCH];
        let mut lo = [0u32; MAX_BATCH];
        let mut hi = [0u32; MAX_BATCH];
        for (i, d) in slots.iter().enumerate() {
            pos[i] = d.pos as u32;
            let cur_len = (d.pos + 1) as u32;
            if cur_len > 512 {
                hi[i] = cur_len;
            } else {
                lo[i] = cur_len;
            }
        }
        gpu.queue
            .write_buffer(&self.extract_pos, 0, bytemuck::cast_slice(&pos));
        // One line per width: which slots it owns, and their lengths.
        let mut pick = [[0u32; MAX_BATCH]; 2];
        let mut counts = [0u32; 2];
        for (i, h) in hi.iter().enumerate() {
            let g = usize::from(*h > 0);
            pick[g][counts[g] as usize] = i as u32;
            counts[g] += 1;
        }
        gpu.queue
            .write_buffer(&self.gqa_cl[0], 0, bytemuck::cast_slice(&lo));
        gpu.queue
            .write_buffer(&self.gqa_cl[1], 0, bytemuck::cast_slice(&hi));
        gpu.queue
            .write_buffer(&self.gqa_pick[0], 0, bytemuck::cast_slice(&pick[0]));
        gpu.queue
            .write_buffer(&self.gqa_pick[1], 0, bytemuck::cast_slice(&pick[1]));

        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_tokens"),
            size: (4 * n) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            // The token the caller seeded becomes layer 0's input, exactly the
            // dispatch `encode_step_slice(embed = true)` issues for b=1.
            for d in slots.iter() {
                cp.set_pipeline(&d.pipes.embed);
                cp.set_bind_group(0, &d.bg_embed, &[]);
                cp.dispatch_workgroups(1, 1, 1);
            }
            for l in 0..nl {
                // The norm is one workgroup per slot (the shipped kernel), then
                // one weight read for all of them.  Fusing the norm into the GEMV
                // (`gemv_norm_batch_slots`) makes every workgroup re-stage every
                // slot's row in shared memory, so its cost grows with b and the
                // step stops getting cheaper past b=4; reading a normalized row
                // that already sits in each slot's own buffer costs one pass.
                for d in slots.iter() {
                    cp.set_pipeline(&d.pipes.rms_norm);
                    cp.set_bind_group(0, &d.layers[l].bg_rms1, &[]);
                    cp.dispatch_workgroups(1, 1, 1);
                }
                cp.set_pipeline(&self.qkv);
                cp.set_bind_group(0, &self.qkv_bg[l], &[]);
                cp.dispatch_workgroups(self.g_qkv, 1, 1);
                // RoPE + KV write, then attention: `wgid.z` is the slot, and the
                // KV row is `POS[slot]`/`CL[slot]`, not slot 0's.
                cp.set_pipeline(&self.extract);
                cp.set_bind_group(0, &self.extract_bg[l], &[]);
                cp.set_bind_group(1, &self.extract_u_bg, &[]);
                cp.dispatch_workgroups(
                    1,
                    self.nqh + slots[0].cfg.num_key_value_heads as u32,
                    self.batch as u32,
                );
                // The shipped step picks 256 threads for `cur_len <= 512` and 512
                // above it, and the two reduction trees differ -- so a slot keeps
                // the width its lone run would use.  Both dispatches cover every
                // slot; the ones the other width owns return on `CL == 0`.
                for (g, pipe) in [(0usize, &self.gqa256), (1usize, &self.gqa512)] {
                    if counts[g] == 0 {
                        continue;
                    }
                    cp.set_pipeline(pipe);
                    cp.set_bind_group(0, &self.gqa_bg[g][l], &[]);
                    cp.set_bind_group(1, &self.gqa_u_bg, &[]);
                    cp.dispatch_workgroups(self.nqh, 1, counts[g]);
                }
                // Above the single kernel's cap the shipped step switches to the
                // chunked pair + merge.  Those bind a slot's own buffers and cache
                // state, so they run per slot rather than fused -- the same
                // dispatches, in the same order, as that slot alone.
                for d in slots.iter() {
                    let cur_len = d.pos + 1;
                    if cur_len <= crate::decoder::GQA_SINGLE_CAP {
                        continue;
                    }
                    let (split, split_x, n_chunks) = d.split_attn_plan(cur_len);
                    cp.set_pipeline(split);
                    cp.set_bind_group(0, &d.layers[l].bg_gqa_split, &[]);
                    cp.dispatch_workgroups(split_x, n_chunks, 1);
                    cp.set_pipeline(&d.pipes.gqa_merge);
                    cp.set_bind_group(0, &d.layers[l].bg_gqa_merge, &[]);
                    cp.dispatch_workgroups(self.nqh, 1, 1);
                }
                cp.set_pipeline(&self.o);
                cp.set_bind_group(0, &self.o_bg[l], &[]);
                cp.dispatch_workgroups(self.g_o, 1, 1);
                for d in slots.iter() {
                    cp.set_pipeline(&d.pipes.rms_norm);
                    cp.set_bind_group(0, &d.layers[l].bg_rms2, &[]);
                    cp.dispatch_workgroups(1, 1, 1);
                }
                cp.set_pipeline(&self.gu);
                cp.set_bind_group(0, &self.gu_bg[l], &[]);
                cp.dispatch_workgroups(self.g_gu, 1, 1);
                cp.set_pipeline(&self.silu);
                cp.set_bind_group(0, &self.silu_bg, &[]);
                cp.set_bind_group(1, &self.silu_u_bg, &[]);
                cp.dispatch_workgroups(self.g_silu.0, self.g_silu.1, self.batch as u32);
                cp.set_pipeline(&self.dp);
                cp.set_bind_group(0, &self.dp_bg[l], &[]);
                cp.dispatch_workgroups(self.g_dp, 1, 1);
            }
            for d in slots.iter() {
                cp.set_pipeline(&d.pipes.rms_norm);
                cp.set_bind_group(0, &d.bg_final_rms, &[]);
                cp.dispatch_workgroups(1, 1, 1);
            }
            cp.set_pipeline(&self.lm);
            cp.set_bind_group(0, &self.lm_bg, &[]);
            cp.dispatch_workgroups(self.g_lm, 1, 1);
            for d in slots.iter() {
                cp.set_pipeline(&d.pipes.argmax);
                cp.set_bind_group(0, &d.bg_argmax, &[]);
                cp.dispatch_workgroups(1, 1, 1);
            }
        }
        for (s, d) in slots.iter().enumerate() {
            enc.copy_buffer_to_buffer(&d.scratch.token, 0, &staging, (s * 4) as u64, 4);
        }
        gpu.queue.submit([enc.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("poll batch")?;
        rx.recv().context("map callback dropped")?.context("map batch")?;
        let out: Vec<i32> = {
            let d = slice.get_mapped_range()?;
            (0..n)
                .map(|s| i32::from_le_bytes([d[s * 4], d[s * 4 + 1], d[s * 4 + 2], d[s * 4 + 3]]))
                .collect()
        };
        staging.unmap();
        for d in slots.iter_mut() {
            d.pos += 1;
        }
        Ok(out)
    }
}

/// Layout for the batched dispatches: `stor` entries carry their binding-array
/// `count`, which is what the shader's `binding_array<T, N>` declares and what a
/// `BufferArray` bind group has to match.  `uni` is the optional uniform.
fn batch_layout(
    gpu: &Gpu,
    label: &str,
    stor: &[(u32, bool, Option<u32>)],
    uniform: bool,
) -> (wgpu::PipelineLayout, Option<wgpu::BindGroupLayout>) {
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> = stor
        .iter()
        .map(|&(binding, read_only, count)| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: count.and_then(std::num::NonZeroU32::new),
        })
        .collect();
    entries.sort_by_key(|e| e.binding);
    let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}_bgl")),
        entries: &entries,
    });
    // wgpu: "bind groups may not contain both a binding array and a uniform
    // buffer", so the config rides group 1 on its own.
    let ubgl = uniform.then(|| {
        gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}_ubgl")),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    });
    let layouts: Vec<Option<&wgpu::BindGroupLayout>> =
        std::iter::once(Some(&bgl)).chain(ubgl.as_ref().map(Some)).collect();
    let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_pl")),
        bind_group_layouts: &layouts,
        immediate_size: 0,
    });
    (pl, ubgl)
}

/// One segment of a batched prefill: its hidden rows, and the KV it writes into.
///
/// The KV lives in each slot's own cache ([`WgpuTextDecoder::fork_capped`] made
/// it), so attention for a segment reads a whole buffer and never an offset into
/// a shared one.
pub struct PrefillSeg<'a> {
    pub hidden: &'a [u8],
    pub len: usize,
    /// Per layer, this segment's K/V caches.
    pub k: Vec<wgpu::Buffer>,
    pub v: Vec<wgpu::Buffer>,
}

/// Prefill several segments at once, one first-token id per segment.
///
/// This exists because the two halves of prefill want different shapes:
///
/// * **The projections are row-independent.**  `rms_norm`, `qkv`, `o`, `gu`,
///   `silu`, `dp` compute row `i` without looking at any other row, so all
///   segments' rows go through **one** GEMM per layer (`m = sum of lengths`)
///   instead of one GEMM per segment.  Measured on 27x40 s segments: the
///   projections are 89% of this phase, and they were being issued as 27 small
///   GEMMs each layer.
/// * **Attention is not.**  A row attends only inside its own segment, and each
///   segment's KV is a different buffer, so this half stays per segment -- and it
///   is the cheap half (11%).
///
/// The arithmetic per row is the same statements in the same order as
/// [`WgpuTextDecoder::prefill`]; only the dispatch grouping changes.  Segments
/// that need the slab attention path (>4096 rows) or an int8 checkpoint take the
/// single-segment path, because those two variants would need per-segment slab
/// bookkeeping for no measurable gain at segment lengths.
pub fn prefill_batch(
    slots: &mut [&mut WgpuTextDecoder],
    plans: &[(usize, &[u8])],
) -> Result<Vec<i32>> {
    anyhow::ensure!(!slots.is_empty(), "prefill_batch: no slots");
    anyhow::ensure!(slots.len() == plans.len(), "prefill_batch: slot/plan mismatch");
    let nl = slots[0].layers.len();
    anyhow::ensure!(nl == slots[0].cfg.num_hidden_layers, "layer count");

    // The clones are handles: `wgpu::Buffer` is refcounted, so a slot's KV is
    // reachable from here without borrowing the slot itself.
    let kv: Vec<Vec<(wgpu::Buffer, wgpu::Buffer)>> = (0..nl)
        .map(|l| {
            slots
                .iter()
                .map(|d| (d.layers[l].k_cache.clone(), d.layers[l].v_cache.clone()))
                .collect()
        })
        .collect();

    let flat = !slots[0].int8 && plans.iter().all(|(len, _)| !slab_path(*len));
    if plans.len() < 2 || !flat {
        return (0..slots.len())
            .map(|i| slots[i].prefill(plans[i].1, plans[i].0, 0))
            .collect();
    }

    let segs: Vec<PrefillSeg<'_>> = plans
        .iter()
        .enumerate()
        .map(|(i, (len, hidden))| PrefillSeg {
            hidden,
            len: *len,
            k: kv.iter().map(|per_slot| per_slot[i].0.clone()).collect(),
            v: kv.iter().map(|per_slot| per_slot[i].1.clone()).collect(),
        })
        .collect();
    slots[0].prefill_multi(&segs)
}

impl WgpuTextDecoder {
    /// The batched body of [`prefill_batch`]; see that function for why the work
    /// splits the way it does.
    fn prefill_multi(&mut self, segs: &[PrefillSeg<'_>]) -> Result<Vec<i32>> {
        let cfg = self.cfg.clone();
        let hs = cfg.hidden_size;
        let nqh = cfg.num_attention_heads;
        let nkvh = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let inter = cfg.intermediate_size;

        // Row layout: segment i owns rows `r0[i] .. r0[i]+len[i]` of the shared
        // activation buffers, exactly as consecutive rows of one sequence.
        let lens: Vec<usize> = segs.iter().map(|g| g.len).collect();
        let mut r0 = Vec::with_capacity(lens.len());
        let mut acc = 0usize;
        for &l in &lens {
            r0.push(acc);
            acc += l;
        }
        let s = acc;
        let mp = s.div_ceil(128) * 128;
        // Per-segment attention geometry, one segment at a time: sized by the
        // longest, reused by all, so the scratch stays constant.
        let len_max = *lens.iter().max().expect("segments");
        let mp_max = len_max.div_ceil(128) * 128;
        let cur_max = len_max;
        let np_max = cur_max.div_ceil(128) * 128;
        let cur16_max = cur_max.div_ceil(16) * 16;

        let bm = if self.gpu.use_tile64(s) { 64 } else { 128 };
        let bn = bm;
        const MAX_GEMMS: u64 = 512;
        // Every forked slot shares `self.cap`, so the KV stride of all segments
        // is the same; the attention buffers only need the longest segment.
        anyhow::ensure!(self.cap >= np_max, "prefill_multi: KV capacity below segment length");
        let words = |rows: usize, cols: usize| (rows * cols / 2 * 4) as u64;
        // Clone handles before mutably borrowing the pool.
        let sc = self.multi_scratch.get_or_insert_with(|| MultiPrefillScratch::new(&self.gpu));
        sc.ensure(
            &self.gpu, &words, hs, cfg.fused_qkv_cols(), nqh, hd, inter,
            mp, mp_max, words(nqh * mp_max, np_max), words(nqh * mp_max, cur16_max),
        );
        let u_gd = sc.u_gd.clone();
        let u_seg = sc.u_seg.clone();
        let u_sl = sc.u_sl.clone();
        let h_buf = sc.h_buf.clone();
        let normed = sc.normed.clone();
        let norm2 = sc.norm2.clone();
        let qkv = sc.qkv.clone();
        let attn_flat = sc.attn_flat.clone();
        let gu = sc.gu.clone();
        let activated = sc.activated.clone();
        let q_out = sc.q_out.clone();
        let scores = sc.scores.clone();
        let attn = sc.attn.clone();

        let mut hidden = Vec::with_capacity(s * hs * 2);
        for g in segs {
            hidden.extend_from_slice(&g.hidden[..g.len * hs * 2]);
        }
        self.gpu.upload(&h_buf, &hidden);

        let gpu = &self.gpu;
        let (gemm_pipe, gemm_acc_pipe, gemm_causal_pipe, gemm_av_causal_pipe) = if bm == 64 {
            (&self.pipes.gemm_64, &self.pipes.gemm_acc_64, &self.pipes.gemm_causal_64, &self.pipes.gemm_av_causal_64)
        } else {
            (&self.pipes.gemm, &self.pipes.gemm_acc, &self.pipes.gemm_causal, &self.pipes.gemm_av_causal)
        };
        let scale = cfg.scale();
        let eps = cfg.rms_norm_eps;
        let max_seq = self.cap as u32;

        // One config per segment per kernel family: the dispatches are per
        // segment, and a single uniform written N times before submit would
        // reach every one of them with the last value.
        let cur16s: Vec<usize> = lens.iter().map(|l| l.div_ceil(16) * 16).collect();
        let nps: Vec<usize> = lens.iter().map(|l| l.div_ceil(128) * 128).collect();
        let mps: Vec<usize> = lens.iter().map(|l| l.div_ceil(128) * 128).collect();
        for (i, _) in lens.iter().enumerate() {
            self.gpu.write_at(
                &u_seg,
                (i * 256) as u64,
                bytemuck::bytes_of(&QkvxCfg {
                    max_seq,
                    start: 0,
                    pos_offset: 0,
                    // `q_out` row stride: this segment's own padded row count, so
                    // `extract` writes and the scores GEMM reads the same layout
                    // (`bsa = mps[i]*hd/2`).  The buffer is sized by the longest
                    // segment and each segment uses its prefix.
                    s: mps[i] as u32,
                    eps,
                    _a: 0,
                    _b: 0,
                    _c: 0,
                }),
            );
            let sm_grid = grid_xy(nqh * lens[i]);
            self.gpu.write_at(
                &u_sl,
                (i * 256) as u64,
                bytemuck::bytes_of(&SoftmaxCfg {
                    n_w: (cur16s[i] / 2) as u32,
                    n_x: (nps[i] / 2) as u32,
                    valid: lens[i] as u32,
                    m: lens[i] as u32,
                    mp: mps[i] as u32,
                    scale,
                    gx: sm_grid.0,
                    row0: 0,
                    q_off: 0,
                }),
            );
        }

        let gemm_bg = |pipe: &wgpu::ComputePipeline,
                       a: &wgpu::Buffer,
                       w: &wgpu::Buffer,
                       c: &wgpu::Buffer,
                       coff: u64|
         -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pb.gemm"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: if coff == 0 {
                            c.as_entire_binding()
                        } else {
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: c,
                                offset: coff,
                                size: None,
                            })
                        },
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &u_gd,
                            offset: 0,
                            size: std::num::NonZeroU64::new(64),
                        }),
                    },
                ],
            })
        };

        let mut gd_slot = 0usize;
        macro_rules! gemm {
            ($cp:expr, $pipe:expr, $a:expr, $w:expr, $c:expr, $coff:expr, $bounds:expr,
             $gx:expr, $gy:expr, $gz:expr) => {{
                let slot = gd_slot;
                gd_slot += 1;
                assert!(slot < MAX_GEMMS as usize, "prefill_batch: GEMM slots exhausted");
                let off = (slot * 256) as u64;
                gpu.queue.write_buffer(&u_gd, off, bytemuck::bytes_of(&$bounds));
                let bg = gemm_bg($pipe, $a, $w, $c, $coff);
                $cp.set_pipeline($pipe);
                $cp.set_bind_group(0, &bg, &[off as u32]);
                $cp.dispatch_workgroups($gx, $gy, $gz);
            }};
        }

        let (silu_gx, silu_gy) = self.write_silu_rows(s);
        let rms_bg = |x: &wgpu::Buffer, w: &wgpu::Buffer, y: &wgpu::Buffer| -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pb.rms"),
                layout: &self.pipes.rms_norm.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.scratch.u_rms.as_entire_binding(),
                    },
                ],
            })
        };

        // Every segment's AV sweeps `cur16i` rows while `len` were written; zero
        // each segment's V tail per KV head so a reused slot cache cannot leak
        // stale (or NaN) values into the padded columns.
        {
            let mut any = false;
            let mut zenc = gpu.device.create_command_encoder(&Default::default());
            for (i, seg) in segs.iter().enumerate() {
                if cur16s[i] > lens[i] {
                    let bytes = ((cur16s[i] - lens[i]) * hd * 2) as u64;
                    for cache in &seg.v {
                        for h in 0..nkvh {
                            let off = ((h * self.cap + lens[i]) * hd * 2) as u64;
                            zenc.clear_buffer(cache, off, Some(bytes));
                            any = true;
                        }
                    }
                }
            }
            if any {
                gpu.queue.submit([zenc.finish()]);
            }
        }

        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let mut cp = enc.begin_compute_pass(&Default::default());

        for (li, layer) in self.layers.iter().enumerate() {
            // front: norm + qkv over every segment's rows
            let bg = rms_bg(&h_buf, &layer.iln_w, &normed);
            cp.set_pipeline(&self.pipes.rms_norm);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(s as u32, 1, 1);
            gemm!(
                &mut cp, gemm_pipe, &normed, &layer.qkv_w, &qkv, 0u64,
                GDims {
                    m: s as u32,
                    n: cfg.fused_qkv_cols() as u32,
                    k: hs as u32,
                    ldc: cfg.fused_qkv_cols() as u32,
                    bsa: 0,
                    bsb: 0,
                    bsc: 0,
                    beta: 0,
                    row0: 0,
                    lda: hs as u32,
                    q_off: 0,
                    gqa_group: 1,
                },
                (cfg.fused_qkv_cols() / bn) as u32, (mp / bm) as u32, 1
            );

            // attention: per segment, each into its own KV
            for (i, seg) in segs.iter().enumerate() {
                let len = seg.len;
                let mpi = mps[i];
                let npi = nps[i];
                let cur16i = cur16s[i];
                let row0 = r0[i];

                // RoPE + KV write, reading this segment's rows of `qkv`.
                let qkv_off = (row0 * cfg.fused_qkv_cols() * 2) as u64;
                let bg_ex = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pb.extract"),
                    layout: &self.pipes.extract.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &qkv,
                                offset: qkv_off,
                                size: None,
                            }),
                        },
                        wgpu::BindGroupEntry { binding: 1, resource: layer.qn_w.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: layer.kn_w.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.scratch.cos.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.scratch.sin.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry { binding: 5, resource: q_out.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: seg.k[li].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 7, resource: seg.v[li].as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &u_seg,
                                offset: (i * 256) as u64,
                                size: std::num::NonZeroU64::new(64),
                            }),
                        },
                    ],
                });
                cp.set_pipeline(&self.pipes.extract);
                cp.set_bind_group(0, &bg_ex, &[]);
                cp.dispatch_workgroups(len as u32, (nqh + nkvh) as u32, 1);

                let softmax_grid = grid_xy(nqh * len);

                // scores = Q . K^T, causal inside the segment.
                gemm!(
                    &mut cp, gemm_causal_pipe, &q_out, &seg.k[li], &scores, 0u64,
                    GDims {
                        m: len as u32,
                        n: len as u32,
                        k: hd as u32,
                        ldc: npi as u32,
                        bsa: (mpi * hd / 2) as u32,
                        bsb: (self.cap * hd / 2) as u32,
                        bsc: (mpi * npi) as u32,
                        beta: 0,
                        row0: 0,
                        lda: hd as u32,
                        q_off: 0,
                        gqa_group: (nqh / nkvh) as u32,
                    },
                    (npi / bn) as u32, (mpi / bm) as u32, nqh as u32
                );
                let (phys, tree) = softmax_blocks(len);
                let sm_pipe = if phys == tree {
                    &self.pipes.softmax[&(tree as usize)]
                } else {
                    &self.pipes.softmax_wide[&(phys as usize, tree as usize)]
                };
                let bg_sm = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pb.sm"),
                    layout: &sm_pipe.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: scores.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: attn.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &u_sl,
                                offset: (i * 256) as u64,
                                size: std::num::NonZeroU64::new(64),
                            }),
                        },
                    ],
                });
                cp.set_pipeline(sm_pipe);
                // The softmax layout's uniform is a dynamic-offset binding (the
                // slab path walks slab configs with it); this path puts the
                // segment's slot in the binding itself.
                cp.set_bind_group(0, &bg_sm, &[0]);
                cp.dispatch_workgroups(softmax_grid.0, softmax_grid.1, 1);

                // attention . V into this segment's rows of the shared buffer,
                // which is what lets the `o` GEMM see every segment at once.
                let out_off = (row0 * nqh * hd * 2) as u64;
                gemm!(
                    &mut cp, gemm_av_causal_pipe, &attn, &seg.v[li], &attn_flat, out_off,
                    GDims {
                        m: len as u32,
                        n: hd as u32,
                        k: cur16i as u32,
                        ldc: (nqh * hd) as u32,
                        bsa: (mpi * cur16i / 2) as u32,
                        bsb: (self.cap * hd / 2) as u32,
                        bsc: hd as u32,
                        beta: 0,
                        row0: 0,
                        lda: cur16i as u32,
                        q_off: 0,
                        gqa_group: (nqh / nkvh) as u32,
                    },
                    (hd / bn) as u32, (mpi / bm) as u32, nqh as u32
                );
            }

            // o over every row, then the FFN
            gemm!(
                &mut cp, gemm_acc_pipe, &attn_flat, &layer.o_w, &h_buf, 0u64,
                GDims {
                    m: s as u32,
                    n: hs as u32,
                    k: (nqh * hd) as u32,
                    ldc: hs as u32,
                    bsa: 0,
                    bsb: 0,
                    bsc: 0,
                    beta: 0,
                    row0: 0,
                    lda: (nqh * hd) as u32,
                    q_off: 0,
                    gqa_group: 1,
                },
                (hs / bn) as u32, (mp / bm) as u32, 1
            );
            let bg2 = rms_bg(&h_buf, &layer.pln_w, &norm2);
            cp.set_pipeline(&self.pipes.rms_norm);
            cp.set_bind_group(0, &bg2, &[]);
            cp.dispatch_workgroups(s as u32, 1, 1);
            gemm!(
                &mut cp, gemm_pipe, &norm2, &layer.gu_w, &gu, 0u64,
                GDims {
                    m: s as u32,
                    n: (2 * inter) as u32,
                    k: hs as u32,
                    ldc: (2 * inter) as u32,
                    bsa: 0,
                    bsb: 0,
                    bsc: 0,
                    beta: 0,
                    row0: 0,
                    lda: hs as u32,
                    q_off: 0,
                    gqa_group: 1,
                },
                ((2 * inter) / bn) as u32, (mp / bm) as u32, 1
            );
            let bg_silu = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pb.silu"),
                layout: &self.pipes.silu.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: gu.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: activated.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.scratch.u_silu.as_entire_binding(),
                    },
                ],
            });
            cp.set_pipeline(&self.pipes.silu);
            cp.set_bind_group(0, &bg_silu, &[]);
            cp.dispatch_workgroups(silu_gx, silu_gy, 1);
            gemm!(
                &mut cp, gemm_acc_pipe, &activated, &layer.dp_w, &h_buf, 0u64,
                GDims {
                    m: s as u32,
                    n: hs as u32,
                    k: inter as u32,
                    ldc: hs as u32,
                    bsa: 0,
                    bsb: 0,
                    bsc: 0,
                    beta: 0,
                    row0: 0,
                    lda: inter as u32,
                    q_off: 0,
                    gqa_group: 1,
                },
                (hs / bn) as u32, (mp / bm) as u32, 1
            );

            // The GEMM uniform is a slot array; a layer spends `4 + 2*segs.len()`
            // of them, so submit and recycle well before it fills.
            if (li + 1) % 4 == 0 && li + 1 < self.layers.len() {
                drop(cp);
                gpu.queue.submit([enc.finish()]);
                enc = gpu.device.create_command_encoder(&Default::default());
                cp = enc.begin_compute_pass(&Default::default());
                gd_slot = 0;
            }
        }

        // Tail: each segment's last row through the final norm, `lm_head` and
        // argmax.  The last row is the only one that predicts a next token.
        let mut tokens = Vec::with_capacity(segs.len());
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pb.tokens"),
            size: (4 * segs.len()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        drop(cp);
        for (i, seg) in segs.iter().enumerate() {
            let mut cp = enc.begin_compute_pass(&Default::default());
            let last = ((r0[i] + seg.len - 1) * hs * 2) as u64;
            let bg_fn = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pb.final_rms"),
                layout: &self.pipes.rms_norm.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &h_buf,
                            offset: last,
                            size: std::num::NonZeroU64::new((hs * 2) as u64),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: self.norm_buf.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.scratch.final_norm.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.scratch.u_rms.as_entire_binding(),
                    },
                ],
            });
            cp.set_pipeline(&self.pipes.rms_norm);
            cp.set_bind_group(0, &bg_fn, &[]);
            cp.dispatch_workgroups(1, 1, 1);
            cp.set_pipeline(&self.pipes.gemv_lm);
            cp.set_bind_group(0, &self.bg_gemv_lm, &[]);
            cp.dispatch_workgroups((cfg.vocab_size / 8) as u32, 1, 1);
            cp.set_pipeline(&self.pipes.argmax);
            cp.set_bind_group(0, &self.bg_argmax, &[]);
            cp.dispatch_workgroups(1, 1, 1);
            drop(cp);
            enc.copy_buffer_to_buffer(&self.scratch.token, 0, &staging, (i * 4) as u64, 4);
        }
        gpu.queue.submit([enc.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("poll prefill_batch")?;
        rx.recv().context("map callback dropped")?.context("map prefill_batch")?;
        {
            let d = slice.get_mapped_range()?;
            for i in 0..segs.len() {
                tokens.push(i32::from_le_bytes([
                    d[i * 4],
                    d[i * 4 + 1],
                    d[i * 4 + 2],
                    d[i * 4 + 3],
                ]));
            }
        }
        staging.unmap();
        Ok(tokens)
    }
}

fn family_layout(
    gpu: &Gpu,
    label: &str,
    storage: &[(u32, bool)],
    uniform_binding: u32,
) -> wgpu::PipelineLayout {
    family_layout_dyn(gpu, label, storage, uniform_binding, false)
}

fn family_layout_dyn(
    gpu: &Gpu,
    label: &str,
    storage: &[(u32, bool)],
    uniform_binding: u32,
    uniform_dynamic: bool,
) -> wgpu::PipelineLayout {
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> = storage
        .iter()
        .map(|&(binding, read_only)| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: uniform_binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: uniform_dynamic,
            min_binding_size: None,
        },
        count: None,
    });
    let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}_bgl")),
        entries: &entries,
    });
    gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_pl")),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    })
}

fn upload_weight(up: &mut BulkUpload, label: &str, w: &PackedWeight) -> Result<wgpu::Buffer> {
    let b = up.storage(label, w.data.len() as u64);
    up.upload(&b, &w.data)?;
    Ok(b)
}

/// Upload raw bytes (the int8 layers' scale buffers) as a storage binding.
fn upload_bytes(up: &mut BulkUpload, label: &str, v: &[u8]) -> Result<wgpu::Buffer> {
    let b = up.storage(label, v.len() as u64);
    up.upload(&b, v)?;
    Ok(b)
}

fn upload_vec(up: &mut BulkUpload, label: &str, v: &[half::f16]) -> Result<wgpu::Buffer> {
    let b = up.storage(label, (v.len() * 2) as u64);
    up.upload(&b, &weights::words_bytes(v))?;
    Ok(b)
}

/// Fuse `parts` into one row-major `[part0 | part1 | ...]` matrix, as
/// `(offset, bytes)` pieces ready for a single destination buffer.
///
/// The concatenation used to be a `PackedWeight` of its own: a full copy of
/// every fused matrix (890 MiB of the 0.6 B decoder) made purely so the upload
/// could see one slice.  The kernels only need the parts to be *contiguous*, so
/// an f16 checkpoint contributes its own mapped bytes at their offsets (no copy
/// at all), and a bf16/f32 one is narrowed straight into the fused layout —
/// without the per-part buffers the piecewise version needed.
fn fused_pieces(
    w: &HashMap<String, weights::RawTensor>,
    prefix: &str,
    parts: &[&str],
) -> Result<Vec<(u64, bytes::Bytes)>> {
    let tensors: Vec<&weights::RawTensor> = parts
        .iter()
        .map(|p| {
            let name = format!("{prefix}.{p}.weight");
            w.get(&name).ok_or_else(|| anyhow!("weight not found: {name}"))
        })
        .collect::<Result<_>>()?;

    if tensors.iter().all(|t| t.dtype == Dtype::F16) {
        let mut pieces = Vec::with_capacity(tensors.len());
        let mut off = 0u64;
        for t in &tensors {
            pieces.push((off, t.data.clone()));
            off += t.data.len() as u64;
        }
        return Ok(pieces);
    }

    // An I8 part in an f16 fusion group is a malformed file: the GPU path for
    // int8 checkpoints is [`fused_i8_pieces`] (weights stay resident), so a
    // silent dequantize here would put f16 bytes where the W8 kernels expect
    // int8 — fail loudly instead.
    anyhow::ensure!(
        tensors.iter().all(|t| t.dtype != Dtype::I8),
        "mixed I8/float fusion for {prefix}: the exporter quantizes whole groups"
    );

    let mut out = Vec::with_capacity(total_bytes(&tensors)?);
    for t in tensors.iter() {
        let start = out.len();
        out.resize(start + t.data.len() / t.dtype.size() * 2, 0);
        t.narrow_f16_into(&mut out[start..])?;
    }
    Ok(vec![(0, out.into())])
}

/// Fuse an I8 fusion group into the device's resident-int8 form: per-part zero-
/// copy byte pieces (the file's own rows, u32-word aligned because every part's
/// `cols` is a multiple of 4) plus the per-row f32 scales concatenated in the
/// *same part order* — fused row `r` must read `S[r]`, and the ordering bug M2
/// made impossible is exactly the one this function could reintroduce, so
/// weights and scales travel together and `concat_rows`-style order is the
/// only order they can have.
fn fused_i8_pieces(
    w: &HashMap<String, weights::RawTensor>,
    prefix: &str,
    parts: &[&str],
) -> Result<(Vec<(u64, bytes::Bytes)>, bytes::Bytes)> {
    let mut pieces = Vec::with_capacity(parts.len());
    let mut scales = Vec::new();
    let mut off = 0u64;
    let mut cols: Option<usize> = None;
    for p in parts {
        let name = format!("{prefix}.{p}.weight");
        let m = weights::get_int8(w, &name)?;
        anyhow::ensure!(
            cols.map(|c| c == m.cols).unwrap_or(true),
            "{prefix}: fused int8 parts disagree on cols ({cols:?} vs {})",
            m.cols
        );
        cols = Some(m.cols);
        pieces.push((off, m.data.clone()));
        off += m.data.len() as u64;
        scales.extend_from_slice(&m.scale);
    }
    Ok((pieces, scales.into()))
}

/// How many layers of converted weights the prefetch thread may run ahead with.
///
/// Two layers is ~34 MiB at 0.6 B (~90 MiB at 1.7 B): enough that the narrowing
/// of the next layer always overlaps the transfer of the current one, and small
/// enough not to be a memory decision.
const PREFETCH_LAYERS: usize = 2;

/// One decoder layer's weights, converted and ready to be staged.
struct LayerWeights {
    qkv: Vec<(u64, bytes::Bytes)>,
    o: PackedWeight,
    gu: Vec<(u64, bytes::Bytes)>,
    dp: PackedWeight,
    /// `Some` iff this layer's linears are int8-resident (the M3 layout): f32
    /// LE per-row scales for `qkv | o | gu | dp` in fused row order, while
    /// `qkv`/`o`/`gu`/`dp` then carry the raw int8 bytes the W8 kernels read.
    /// `None` is the f16 checkpoint.  The exporter quantizes a layer's linears
    /// all-or-none, and [`convert_layer`] fails anything in between.
    scales: Option<LayerScales>,
    iln: Vec<half::f16>,
    pln: Vec<half::f16>,
    qn: Vec<half::f16>,
    kn: Vec<half::f16>,
}

/// The four scale buffers of an int8 layer, in fused row order — the buffer
/// the W8 kernels' `S` binding reads.
struct LayerScales {
    qkv: bytes::Bytes,
    o: bytes::Bytes,
    gu: bytes::Bytes,
    dp: bytes::Bytes,
}

/// Whether the checkpoint quantized the decoder's linears at all.  The
/// exporter does so checkpoint-wide, so layer 0's `q_proj` decides; every
/// layer is held to that answer by [`convert_layer`].
fn checkpoint_is_int8(w: &HashMap<String, weights::RawTensor>, prefix: &str) -> Result<bool> {
    let name = format!("{prefix}.layers.0.self_attn.q_proj.weight");
    let t = w
        .get(&name)
        .ok_or_else(|| anyhow!("weight not found: {name}"))?;
    Ok(t.dtype == Dtype::I8)
}

/// Read one layer's tensors out of the checkpoint in upload-ready form.
///
/// `int8` is the checkpoint-wide mode ([`checkpoint_is_int8`]); an f16 layer
/// with an I8 part (or the reverse) is a malformed file and an error here, not
/// a silent dequantize — the W8 kernels would read the wrong bytes.
fn convert_layer(w: &HashMap<String, weights::RawTensor>, p: &str, int8: bool) -> Result<LayerWeights> {
    let iln = weights::get_vector(w, &format!("{p}.input_layernorm.weight"))?;
    let pln = weights::get_vector(w, &format!("{p}.post_attention_layernorm.weight"))?;
    let qn = weights::get_vector(w, &format!("{p}.self_attn.q_norm.weight"))?;
    let kn = weights::get_vector(w, &format!("{p}.self_attn.k_norm.weight"))?;
    if int8 {
        let (qkv, qkv_s) =
            fused_i8_pieces(w, &format!("{p}.self_attn"), &["q_proj", "k_proj", "v_proj"])?;
        let (gu, gu_s) = fused_i8_pieces(w, &format!("{p}.mlp"), &["gate_proj", "up_proj"])?;
        let o = weights::get_int8(w, &format!("{p}.self_attn.o_proj.weight"))?;
        let dp = weights::get_int8(w, &format!("{p}.mlp.down_proj.weight"))?;
        return Ok(LayerWeights {
            // The int8 bytes ride in `PackedWeight`'s data slot (the upload is
            // dtype-blind); `scales` is what says how to read them back.
            o: PackedWeight { data: o.data.clone(), rows: o.rows, cols: o.cols },
            dp: PackedWeight { data: dp.data.clone(), rows: dp.rows, cols: dp.cols },
            qkv,
            gu,
            scales: Some(LayerScales {
                qkv: qkv_s,
                o: o.scale.clone(),
                gu: gu_s,
                dp: dp.scale.clone(),
            }),
            iln,
            pln,
            qn,
            kn,
        });
    }
    let o_name = format!("{p}.self_attn.o_proj.weight");
    anyhow::ensure!(
        w.get(&o_name).map(|t| t.dtype != Dtype::I8).unwrap_or(false),
        "{o_name} is I8 in an otherwise-f16 checkpoint; quantize the whole decoder or none"
    );
    let dp_name = format!("{p}.mlp.down_proj.weight");
    anyhow::ensure!(
        w.get(&dp_name).map(|t| t.dtype != Dtype::I8).unwrap_or(false),
        "{dp_name} is I8 in an otherwise-f16 checkpoint; quantize the whole decoder or none"
    );
    Ok(LayerWeights {
        o: weights::get_matrix(w, &o_name)?,
        dp: weights::get_matrix(w, &dp_name)?,
        qkv: fused_pieces(w, &format!("{p}.self_attn"), &["q_proj", "k_proj", "v_proj"])?,
        gu: fused_pieces(w, &format!("{p}.mlp"), &["gate_proj", "up_proj"])?,
        scales: None,
        iln,
        pln,
        qn,
        kn,
    })
}

fn total_bytes(tensors: &[&weights::RawTensor]) -> Result<usize> {
    let mut total = 0usize;
    for t in tensors {
        if t.shape.len() != 2 {
            return Err(anyhow!("expected 2D weight, got {:?}", t.shape));
        }
        total += t.shape[0] * t.shape[1] * 2;
    }
    Ok(total)
}

impl WgpuTextDecoder {
    /// Prefill: run `s` input positions (hidden states `[s, hs]` f16
    /// little-endian words) through every layer with causal attention, writing
    /// KV slots `kv_start..kv_start+s`, then final-norm + lm-head the last
    /// position.  Returns the argmax token (the first decode token) and leaves
    /// `self.pos` at `kv_start + s` so `step()` continues the sequence.
    ///
    /// The prefill GEMM chain: f32 accumulate with a single f16 rounding, in
    /// this engine's own accumulation order.
    ///
    /// # GPU 资源失败必须变成 `Err`
    ///
    /// wgpu 的 `create_buffer` / `create_bind_group` 是 **infallible**：显存不足时
    /// 返回的是 invalid handle，失败通过异步 validation error 报告，而不是同步
    /// `Result`。不主动接 error scope 的话，代码会拿着无效 buffer 继续 dispatch，
    /// prefill 变成空转，decode 随后吐出复读垃圾并返回 `Ok` —— 失败被伪装成
    /// 成功，比报错糟糕得多。
    ///
    /// 实测（13.8 min 中文 / fp16 1.7B / 8 GB 卡）：451 次 uncaptured validation
    /// error、prefill 只用了 172 ms（正常约 16.5 s）、输出 `damenopdamenop...`。
    /// 接上 scope 后这里必须返回 `Err`。
    ///
    /// **只在运行期真正失败时报**：这里不预估显存、不提前拦。两个 scope 分别接
    /// wgpu 的 OutOfMemory 与 Validation（buffer 分配失败在 wgpu 里表现为 invalid
    /// handle + validation error，而不是同步 OOM 异常），捕获后翻成人能看懂的话。
    ///
    /// The prefill GEMM chain: f32 accumulate with a single f16 rounding, in
    /// this engine's own accumulation order.
    pub fn prefill(&mut self, hidden_words: &[u8], s: usize, kv_start: usize) -> Result<i32> {
        // 栈式 scope：内层先 pop。Validation 错误被内层吃掉，OOM 穿过它落到外层。
        let oom = self
            .gpu
            .device
            .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = self
            .gpu
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let result = self.prefill_impl(hidden_words, s, kv_start);
        let validation_err = pollster::block_on(validation.pop());
        let oom_err = pollster::block_on(oom.pop());
        let audio_min = s as f64 / 13.0 / 60.0;
        if let Some(e) = oom_err {
            return Err(anyhow::anyhow!(
                "GPU out of memory during prefill ({s} tokens, about {audio_min:.1} min of audio). \
                 Try the int8 checkpoint, shorter segments, or the CPU backend. Caused by: {e}"
            ));
        }
        if let Some(e) = validation_err {
            return Err(anyhow::anyhow!(
                "GPU resource allocation failed during prefill ({s} tokens, about {audio_min:.1} \
                 min of audio) — usually VRAM exhausted, so a buffer came back invalid. \
                 Try the int8 checkpoint, shorter segments, or the CPU backend. Caused by: {e}"
            ));
        }
        result
    }

    #[allow(clippy::too_many_lines)]
    fn prefill_impl(&mut self, hidden_words: &[u8], s: usize, kv_start: usize) -> Result<i32> {
        let cfg = self.cfg.clone();
        let hs = cfg.hidden_size;
        let nqh = cfg.num_attention_heads;
        let nkvh = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let inter = cfg.intermediate_size;
        let cur = kv_start + s;
        let mp = s.div_ceil(128) * 128;
        let np = cur.div_ceil(128) * 128;
        let cur16 = cur.div_ceil(16) * 16;
        anyhow::ensure!(hidden_words.len() >= s * hs * 2, "prefill hidden size mismatch");
        let slab_t = SLAB_T;
        let n_slab = if slab_path(s) { cur.div_ceil(slab_t) } else { 0 };
        let slabbed = n_slab > 0;
        let nkv_rows = if slabbed { n_slab * slab_t } else { np };
        // Slabbed attention reads whole slabs past `cur`, so the cache must hold
        // `nkv_rows` per KV head *before* any buffer handle is taken below --
        // with `QASR_SLAB=on` that is more than `cur`, and more than a previous
        // `cap`, which only grows through here.
        self.ensure_capacity(nkv_rows);
        let bind_limit = self.gpu.limits.max_storage_buffer_binding_size;
        if slabbed {
            anyhow::ensure!(
                n_slab <= MAX_SLAB,
                "prefill: {n_slab} key slabs of {slab_t} exceed the {MAX_SLAB} uniform slots"
            );
            let part = (n_slab * mp * nqh * hd * 2) as u64;
            anyhow::ensure!(
                part <= bind_limit,
                "prefill slab scratch is {:.2} GiB for {s} tokens, over the {:.2} GiB per-binding limit",
                part as f64 / (1u64 << 30) as f64,
                bind_limit as f64 / (1u64 << 30) as f64,
            );
        } else {
            let scratch = (nqh * mp * cur16 * 2) as u64;
            anyhow::ensure!(
                scratch <= bind_limit,
                "prefill attention scratch is {:.2} GiB for {s} tokens ({:.1} min of audio), over the {:.2} GiB per-binding limit — \
                 the O(s²) prefill attention supports at most ~{} tokens",
                scratch as f64 / (1u64 << 30) as f64,
                s as f64 / 12.5 / 60.0,
                bind_limit as f64 / (1u64 << 30) as f64,
                ((bind_limit / (2 * nqh as u64)) as f64).sqrt() as usize,
            );
        }

        let bm = if self.gpu.use_tile64(s) { 64 } else { 128 };
        let bn = bm;
        let words = |rows: usize, cols: usize| (rows * cols / 2 * 4) as u64;
        let slab_cols = if slabbed { slab_t } else { np };
        let sc = self.prefill_scratch.get_or_insert_with(|| PrefillScratch::new(&self.gpu));
        sc.ensure(
            &self.gpu, &words, hs, cfg.fused_qkv_cols(), nqh, hd, inter, mp,
            words(nqh * mp, slab_cols), words(nqh * mp, if slabbed { slab_t } else { cur16 }),
            if slabbed {
                Some((
                    words(n_slab * mp, nqh * hd),
                    (n_slab * nqh * mp * 2 * 4) as u64,
                    (n_slab * nqh * mp * 4) as u64,
                ))
            } else {
                None
            },
        );
        let u_gd = sc.u_gd.clone();
        let u_sl = sc.u_sl.clone();
        let h_buf = sc.h_buf.clone();
        let normed = sc.normed.clone();
        let norm2 = sc.norm2.clone();
        let qkv = sc.qkv.clone();
        let q_out = sc.q_out.clone();
        let scores = sc.scores.clone();
        let attn = sc.attn.clone();
        let attn_flat = sc.attn_flat.clone();
        let gu = sc.gu.clone();
        let activated = sc.activated.clone();
        let (slab_part, slab_stats, slab_w) = match &sc.slab {
            Some(s) => (
                Some(s.slab_part.clone()),
                Some(s.slab_stats.clone()),
                Some(s.slab_w.clone()),
            ),
            None => (None, None, None),
        };
        const MAX_GEMMS: u64 = 256;
        let mut up = self.gpu.uploader();
        up.upload(&h_buf, &hidden_words[..s * hs * 2])?;
        up.finish()?;

        let gpu = &self.gpu;
        let (gemm_pipe, gemm_acc_pipe, gemm_causal_pipe, gemm_av_causal_pipe, gemm_av_pipe, gemm_w8_pipe, gemm_w8_acc_pipe) = if bm == 64 {
            (&self.pipes.gemm_64, &self.pipes.gemm_acc_64, &self.pipes.gemm_causal_64, &self.pipes.gemm_av_causal_64, &self.pipes.gemm_av_64, &self.pipes.gemm_w8_64, &self.pipes.gemm_w8_acc_64)
        } else {
            (&self.pipes.gemm, &self.pipes.gemm_acc, &self.pipes.gemm_causal, &self.pipes.gemm_av_causal, &self.pipes.gemm_av, &self.pipes.gemm_w8, &self.pipes.gemm_w8_acc)
        };
        let gemm_bg = |pipe: &wgpu::ComputePipeline,
                       a: &wgpu::Buffer,
                       w: &wgpu::Buffer,
                       c: &wgpu::Buffer,
                       coff: u64,
                       u_gd: &wgpu::Buffer|
         -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("p.gemm"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: if coff == 0 {
                            c.as_entire_binding()
                        } else {
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: c,
                                offset: coff,
                                size: None,
                            })
                        },
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: u_gd,
                            offset: 0,
                            size: std::num::NonZeroU64::new(64),
                        }),
                    },
                ],
            })
        };
        // The int8 weight GEMM: `gemm_bg` plus the per-row f32 scale buffer at
        // binding 4 (the shader's `S8`).
        let gemm_w8_bg = |pipe: &wgpu::ComputePipeline,
                          a: &wgpu::Buffer,
                          w: &wgpu::Buffer,
                          s: &wgpu::Buffer,
                          c: &wgpu::Buffer,
                          coff: u64,
                          u_gd: &wgpu::Buffer|
         -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("p.gemm_w8"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: if coff == 0 {
                            c.as_entire_binding()
                        } else {
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: c,
                                offset: coff,
                                size: None,
                            })
                        },
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: u_gd,
                            offset: 0,
                            size: std::num::NonZeroU64::new(64),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 4, resource: s.as_entire_binding() },
                ],
            })
        };
        let mut gd_slot = 2 * MAX_SLAB;
        macro_rules! gemm_at {
            ($cp:expr, $pipe:expr, $a:expr, $w:expr, $c:expr, $coff:expr, $slot:expr, $gdims:expr,
             $gx:expr, $gy:expr, $gz:expr) => {{
                assert!($slot < MAX_GEMMS as usize, "prefill: GEMM uniform slots exhausted");
                let off = (($slot) * 256) as u64;
                gpu.queue.write_buffer(&u_gd, off, bytemuck::bytes_of(&$gdims));
                let bg = gemm_bg($pipe, $a, $w, $c, $coff, &u_gd);
                $cp.set_pipeline($pipe);
                $cp.set_bind_group(0, &bg, &[off as u32]);
                $cp.dispatch_workgroups($gx, $gy, $gz);
            }};
        }
        macro_rules! gemm {
            ($cp:expr, $pipe:expr, $a:expr, $w:expr, $c:expr, $m:expr, $n:expr, $k:expr,
             $ldc:expr, $bsa:expr, $bsb:expr, $bsc:expr, $gx:expr, $gy:expr, $gz:expr) => {{
                let slot = gd_slot;
                gd_slot += 1;
                gemm_at!(
                    $cp, $pipe, $a, $w, $c, 0u64, slot,
                    GDims {
                        m: $m as u32,
                        n: $n as u32,
                        k: $k as u32,
                        ldc: $ldc as u32,
                        bsa: $bsa as u32,
                        bsb: $bsb as u32,
                        bsc: $bsc as u32,
                        beta: 0,
                        row0: 0,
                        lda: $k as u32,
                        q_off: kv_start as u32,
                        gqa_group: 1,
                    },
                    $gx, $gy, $gz
                );
            }};
        }
        // The int8 weight GEMM — `gemm!` with the scale buffer.  Dispatched only
        // on an int8 checkpoint (`self.int8`), and only for the four weight
        // operands; the attention GEMMs stay f16.
        macro_rules! gemm_w8 {
            ($cp:expr, $pipe:expr, $a:expr, $w:expr, $s:expr, $c:expr, $m:expr, $n:expr, $k:expr,
             $ldc:expr, $bsa:expr, $bsb:expr, $bsc:expr, $gx:expr, $gy:expr, $gz:expr) => {{
                let slot = gd_slot;
                gd_slot += 1;
                assert!(slot < MAX_GEMMS as usize, "prefill: GEMM uniform slots exhausted");
                let off = ((slot) * 256) as u64;
                gpu.queue.write_buffer(&u_gd, off, bytemuck::bytes_of(&GDims {
                    m: $m as u32,
                    n: $n as u32,
                    k: $k as u32,
                    ldc: $ldc as u32,
                    bsa: $bsa as u32,
                    bsb: $bsb as u32,
                    bsc: $bsc as u32,
                    beta: 0,
                    row0: 0,
                    lda: $k as u32,
                    q_off: kv_start as u32,
                    gqa_group: 1,
                }));
                let bg = gemm_w8_bg($pipe, $a, $w, $s, $c, 0u64, &u_gd);
                $cp.set_pipeline($pipe);
                $cp.set_bind_group(0, &bg, &[off as u32]);
                $cp.dispatch_workgroups($gx, $gy, $gz);
            }};
        }

        let silu_grid = self.write_silu_rows(s);
        let softmax_grid = grid_xy(nqh * s);
        gpu.upload(
            &self.scratch.u_qkvx,
            bytemuck::bytes_of(&QkvxCfg {
                max_seq: self.cap as u32,
                start: kv_start as u32,
                pos_offset: kv_start as u32,
                s: mp as u32,
                eps: cfg.rms_norm_eps,
                _a: 0,
                _b: 0,
                _c: 0,
            }),
        );
        let w_grid = grid_xy((nqh * mp).div_ceil(256));
        let slab_m_grid = (mp * hd / 2).div_ceil(256) as u32;
        if slabbed {
            for t in 0..n_slab {
                let t0 = t * slab_t;
                let tl = (cur16 - t0).min(slab_t);
                gpu.write_at(
                    &u_sl,
                    (t * 256) as u64,
                    bytemuck::bytes_of(&SoftmaxCfg {
                        n_w: (slab_t / 2) as u32,
                        n_x: (slab_t / 2) as u32,
                        valid: tl as u32,
                        m: s as u32,
                        mp: mp as u32,
                        scale: cfg.scale(),
                        gx: softmax_grid.0,
                        row0: t0 as u32,
                        q_off: kv_start as u32,
                    }),
                );
                gpu.write_at(
                    &u_sl,
                    ((MAX_SLAB + t) * 256) as u64,
                    bytemuck::bytes_of(&SlabStatsCfg {
                        n_x: (slab_t / 2) as u32,
                        valid: tl as u32,
                        m: s as u32,
                        mp: mp as u32,
                        scale: cfg.scale(),
                        row0: t0 as u32,
                        gx: softmax_grid.0,
                        rows: (nqh * mp) as u32,
                        q_off: kv_start as u32,
                    }),
                );
            }
            gpu.write_at(
                &u_sl,
                (2 * MAX_SLAB * 256) as u64,
                bytemuck::bytes_of(&SlabWeightsCfg {
                    rows: (nqh * mp) as u32,
                    n_slab: n_slab as u32,
                    gx: w_grid.0,
                    _p: 0,
                }),
            );
            gpu.write_at(
                &u_sl,
                ((2 * MAX_SLAB + 1) * 256) as u64,
                bytemuck::bytes_of(&SlabMergeCfg {
                    rows: mp as u32,
                    n_slab: n_slab as u32,
                    gx: slab_m_grid,
                    _p: 0,
                }),
            );
        } else {
            gpu.write_at(
                &u_sl,
                0,
                bytemuck::bytes_of(&SoftmaxCfg {
                    n_w: (cur16 / 2) as u32,
                    n_x: (np / 2) as u32,
                    valid: cur as u32,
                    m: s as u32,
                    mp: mp as u32,
                    scale: cfg.scale(),
                    gx: softmax_grid.0,
                    row0: 0,
                    q_off: kv_start as u32,
                }),
            );
        }
        // The AV pass sweeps `k = cur16` while only `cur` KV rows were written,
        // and the softmax's zero columns only make those tail reads harmless if
        // the rows themselves are numbers: a reused cache holds whatever the
        // previous clip left there, including NaN.  Zero `cur..cur16` per KV
        // head before the compute pass, exactly what a fresh buffer would be.
        if cur16 > cur {
            let mut zenc = gpu.device.create_command_encoder(&Default::default());
            let bytes = ((cur16 - cur) * hd * 2) as u64;
            for layer in &self.layers {
                for h in 0..nkvh {
                    let off = ((h * self.cap + cur) * hd * 2) as u64;
                    zenc.clear_buffer(&layer.v_cache, off, Some(bytes));
                }
            }
            gpu.queue.submit([zenc.finish()]);
        }

        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let mut cp = enc.begin_compute_pass(&Default::default());

        let dup = std::env::var("QASR_DUP").unwrap_or_default();
        // `QASR_SKIP` on the prefill's four *projections* -- the phase's dominant
        // cost had no exact price at all (only `QASR_DUP`, which inflates a GEMM
        // by 35-65%), so their share of `pre` was an inference from `gemm_bench`.
        // Skipping changes the transcript, not the phase line: `pre` stays
        // comparable, which is the same argument the encoder's skips rest on.
        let skip = std::env::var("QASR_SKIP").unwrap_or_default();
        let sk = |name: &str| skip == name;

        for (li, layer) in self.layers.iter().enumerate() {
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("p.rms1"),
                layout: &self.pipes.rms_norm.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: h_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: layer.iln_w.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: normed.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.scratch.u_rms.as_entire_binding() },
                ],
            });
            cp.set_pipeline(&self.pipes.rms_norm);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(s as u32, 1, 1);

            if !sk("p_qkv") {
                if self.int8 {
                    gemm_w8!(
                        &mut cp, gemm_w8_pipe, &normed, &layer.qkv_w,
                        layer.qkv_s.as_ref().expect("int8 scale buffers"), &qkv,
                        s, cfg.fused_qkv_cols(), hs, cfg.fused_qkv_cols(), 0, 0, 0,
                        (cfg.fused_qkv_cols() / bn) as u32, (mp / bm) as u32, 1
                    );
                } else {
                    gemm!(
                        &mut cp, gemm_pipe, &normed, &layer.qkv_w, &qkv,
                        s, cfg.fused_qkv_cols(), hs, cfg.fused_qkv_cols(), 0, 0, 0,
                        (cfg.fused_qkv_cols() / bn) as u32, (mp / bm) as u32, 1
                    );
                }
            }

            let bg_ex = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("p.extract"),
                layout: &self.pipes.extract.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: qkv.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: layer.qn_w.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: layer.kn_w.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.scratch.cos.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.scratch.sin.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: q_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: layer.k_cache.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: layer.v_cache.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: self.scratch.u_qkvx.as_entire_binding() },
                ],
            });
            cp.set_pipeline(&self.pipes.extract);
            cp.set_bind_group(0, &bg_ex, &[]);
            cp.dispatch_workgroups(s as u32, (nqh + nkvh) as u32, 1);
            if dup == "p_extract" {
                cp.dispatch_workgroups(s as u32, (nqh + nkvh) as u32, 1);
            }

            if slabbed {
                let spart = slab_part.as_ref().unwrap();
                let sstats = slab_stats.as_ref().unwrap();
                let swt = slab_w.as_ref().unwrap();
                for t in 0..n_slab {
                    let t0 = t * slab_t;
                    let tl = (cur16 - t0).min(slab_t);
                    let sg = GDims {
                        m: s as u32,
                        n: slab_t as u32,
                        k: hd as u32,
                        ldc: slab_t as u32,
                        bsa: (mp * hd / 2) as u32,
                        bsb: (self.cap * hd / 2) as u32,
                        bsc: (mp * slab_t) as u32,
                        beta: 0,
                        row0: t0 as u32,
                        lda: hd as u32,
                        q_off: kv_start as u32,
                        gqa_group: (nqh / nkvh) as u32,
                    };
                    let sgrid = ((slab_t / bn) as u32, (mp / bm) as u32, nqh as u32);
                    gemm_at!(
                        &mut cp, gemm_causal_pipe, &q_out, &layer.k_cache, &scores,
                        0u64, t, sg,
                        sgrid.0, sgrid.1, sgrid.2
                    );
                    if dup == "p_scores" {
                        gemm_at!(
                            &mut cp, gemm_causal_pipe, &q_out, &layer.k_cache, &scores,
                            0u64, t, sg,
                            sgrid.0, sgrid.1, sgrid.2
                        );
                    }

                    let bg_st = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("p.slab_stats"),
                        layout: &self.pipes.slab_stats.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: scores.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: sstats.as_entire_binding() },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &u_sl,
                                    offset: 0,
                                    size: std::num::NonZeroU64::new(40),
                                }),
                            },
                        ],
                    });
                    cp.set_pipeline(&self.pipes.slab_stats);
                    cp.set_bind_group(0, &bg_st, &[((MAX_SLAB + t) * 256) as u32]);
                    cp.dispatch_workgroups(softmax_grid.0, softmax_grid.1, 1);

                    let bg_sm = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("p.slab_sm"),
                        layout: &self.pipes.softmax[&slab_t].get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: scores.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: attn.as_entire_binding() },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &u_sl,
                                    offset: 0,
                                    size: std::num::NonZeroU64::new(40),
                                }),
                            },
                        ],
                    });
                    cp.set_pipeline(&self.pipes.softmax[&SLAB_BS]);
                    cp.set_bind_group(0, &bg_sm, &[(t * 256) as u32]);
                    cp.dispatch_workgroups(softmax_grid.0, softmax_grid.1, 1);
                    if dup == "p_softmax" {
                        cp.dispatch_workgroups(softmax_grid.0, softmax_grid.1, 1);
                    }

                    let ag = GDims {
                        m: s as u32,
                        n: hd as u32,
                        k: tl as u32,
                        ldc: (nqh * hd) as u32,
                        bsa: (mp * slab_t / 2) as u32,
                        bsb: (self.cap * hd / 2) as u32,
                        bsc: hd as u32,
                        beta: 0,
                        row0: t0 as u32,
                        lda: slab_t as u32,
                        q_off: kv_start as u32,
                        gqa_group: (nqh / nkvh) as u32,
                    };
                    // Slab AV: `n = hd` is `hd/bn` tiles (never a hardcoded 1).
                    gemm_at!(
                        &mut cp, gemm_av_causal_pipe, &attn, &layer.v_cache, spart,
                        (t * mp * nqh * hd * 2) as u64, n_slab + t, ag,
                        (hd / bn) as u32, (mp / bm) as u32, nqh as u32
                    );
                    if dup == "p_av" {
                        gemm_at!(
                            &mut cp, gemm_av_causal_pipe, &attn, &layer.v_cache, spart,
                            (t * mp * nqh * hd * 2) as u64, n_slab + t, ag,
                            (hd / bn) as u32, (mp / bm) as u32, nqh as u32
                        );
                    }
                }

                let bg_w = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("p.slab_w"),
                    layout: &self.pipes.slab_weights.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: sstats.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: swt.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &u_sl,
                                offset: (2 * MAX_SLAB * 256) as u64,
                                size: std::num::NonZeroU64::new(40),
                            }),
                        },
                    ],
                });
                cp.set_pipeline(&self.pipes.slab_weights);
                cp.set_bind_group(0, &bg_w, &[]);
                cp.dispatch_workgroups(w_grid.0, w_grid.1, 1);

                let bg_m = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("p.slab_merge"),
                    layout: &self.pipes.slab_merge.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: spart.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: swt.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: attn_flat.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &u_sl,
                                offset: ((2 * MAX_SLAB + 1) * 256) as u64,
                                size: std::num::NonZeroU64::new(40),
                            }),
                        },
                    ],
                });
                cp.set_pipeline(&self.pipes.slab_merge);
                cp.set_bind_group(0, &bg_m, &[]);
                cp.dispatch_workgroups(slab_m_grid, nqh as u32, 1);
            } else {
                // K is read in place: one KV head serves `nqh/nkvh` query heads
                // (`gqa_group` divides the batch index into `bsb`), so the
                // repeated [nqh, ...] copy `repeat_kv` used to build is gone.
                let qk = GDims {
                    m: s as u32,
                    n: cur as u32,
                    k: hd as u32,
                    ldc: np as u32,
                    bsa: (mp * hd / 2) as u32,
                    bsb: (self.cap * hd / 2) as u32,
                    bsc: (mp * np) as u32,
                    beta: 0,
                    row0: 0,
                    lda: hd as u32,
                    q_off: kv_start as u32,
                    gqa_group: (nqh / nkvh) as u32,
                };
                let slot = gd_slot;
                gd_slot += 1;
                gemm_at!(
                    &mut cp, gemm_causal_pipe, &q_out, &layer.k_cache, &scores,
                    0u64, slot, qk,
                    (np / bn) as u32, (mp / bm) as u32, nqh as u32
                );
                if dup == "p_scores" {
                    let slot = gd_slot;
                    gd_slot += 1;
                    gemm_at!(
                        &mut cp, gemm_pipe, &q_out, &layer.k_cache, &scores,
                        0u64, slot, qk,
                        (np / bn) as u32, (mp / bm) as u32, nqh as u32
                    );
                }

                let (phys, tree) = softmax_blocks(cur);
                // `phys == tree` is the shipped kernel; otherwise the physical
                // width is narrower than the arithmetic's tree and each thread
                // emulates `tree/phys` lanes (see `softmax_causal_tree`).
                let sm_pipe = if phys == tree {
                    &self.pipes.softmax[&(tree as usize)]
                } else {
                    &self.pipes.softmax_wide[&(phys as usize, tree as usize)]
                };
                let bg_sm = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("p.sm"),
                    layout: &sm_pipe.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: scores.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: attn.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &u_sl,
                                offset: 0,
                                size: std::num::NonZeroU64::new(40),
                            }),
                        },
                    ],
                });
                cp.set_pipeline(sm_pipe);
                cp.set_bind_group(0, &bg_sm, &[0]);
                cp.dispatch_workgroups(softmax_grid.0, softmax_grid.1, 1);
                if dup == "p_softmax" {
                    cp.dispatch_workgroups(softmax_grid.0, softmax_grid.1, 1);
                }

                let av = GDims {
                    m: s as u32,
                    n: hd as u32,
                    k: cur16 as u32,
                    ldc: (nqh * hd) as u32,
                    bsa: (mp * cur16 / 2) as u32,
                    bsb: (self.cap * hd / 2) as u32,
                    bsc: hd as u32,
                    beta: 0,
                    row0: 0,
                    lda: cur16 as u32,
                    q_off: kv_start as u32,
                    gqa_group: (nqh / nkvh) as u32,
                };
                // `n = hd` is one 128- or two 64-wide tiles, so `gx` follows the
                // tile width -- a hardcoded 1 leaves half of a 64x64 AV unwritten.
                let slot = gd_slot;
                gd_slot += 1;
                gemm_at!(
                    &mut cp, gemm_av_causal_pipe, &attn, &layer.v_cache, &attn_flat,
                    0u64, slot, av,
                    (hd / bn) as u32, (mp / bm) as u32, nqh as u32
                );
                if dup == "p_av" {
                    let slot = gd_slot;
                    gd_slot += 1;
                    gemm_at!(
                        &mut cp, gemm_av_pipe, &attn, &layer.v_cache, &attn_flat,
                        0u64, slot, av,
                        (hd / bn) as u32, (mp / bm) as u32, nqh as u32
                    );
                }
            }

            if !sk("p_o") {
                if self.int8 {
                    gemm_w8!(
                        &mut cp, gemm_w8_acc_pipe, &attn_flat, &layer.o_w,
                        layer.o_s.as_ref().expect("int8 scale buffers"), &h_buf,
                        s, hs, nqh * hd, hs, 0, 0, 0,
                        (hs / bn) as u32, (mp / bm) as u32, 1
                    );
                } else {
                    gemm!(
                        &mut cp, gemm_acc_pipe, &attn_flat, &layer.o_w, &h_buf,
                        s, hs, nqh * hd, hs, 0, 0, 0,
                        (hs / bn) as u32, (mp / bm) as u32, 1
                    );
                }
            }

            let bg_rms2 = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("p.rms2"),
                layout: &self.pipes.rms_norm.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: h_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: layer.pln_w.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: norm2.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.scratch.u_rms.as_entire_binding() },
                ],
            });
            cp.set_pipeline(&self.pipes.rms_norm);
            cp.set_bind_group(0, &bg_rms2, &[]);
            cp.dispatch_workgroups(s as u32, 1, 1);
            if dup == "p_rms" {
                cp.dispatch_workgroups(s as u32, 1, 1);
            }

            if !sk("p_gu") {
                if self.int8 {
                    gemm_w8!(
                        &mut cp, gemm_w8_pipe, &norm2, &layer.gu_w,
                        layer.gu_s.as_ref().expect("int8 scale buffers"), &gu,
                        s, 2 * inter, hs, 2 * inter, 0, 0, 0,
                        ((2 * inter) / bn) as u32, (mp / bm) as u32, 1
                    );
                } else {
                    gemm!(
                        &mut cp, gemm_pipe, &norm2, &layer.gu_w, &gu,
                        s, 2 * inter, hs, 2 * inter, 0, 0, 0,
                        ((2 * inter) / bn) as u32, (mp / bm) as u32, 1
                    );
                }
            }

            let bg_silu = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("p.silu"),
                layout: &self.pipes.silu.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: gu.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: activated.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.scratch.u_silu.as_entire_binding() },
                ],
            });
            cp.set_pipeline(&self.pipes.silu);
            cp.set_bind_group(0, &bg_silu, &[]);
            cp.dispatch_workgroups(silu_grid.0, silu_grid.1, 1);
            if dup == "p_silu" {
                cp.dispatch_workgroups(silu_grid.0, silu_grid.1, 1);
            }

            if !sk("p_dp") {
                if self.int8 {
                    gemm_w8!(
                        &mut cp, gemm_w8_acc_pipe, &activated, &layer.dp_w,
                        layer.dp_s.as_ref().expect("int8 scale buffers"), &h_buf,
                        s, hs, inter, hs, 0, 0, 0,
                        (hs / bn) as u32, (mp / bm) as u32, 1
                    );
                } else {
                    gemm!(
                        &mut cp, gemm_acc_pipe, &activated, &layer.dp_w, &h_buf,
                        s, hs, inter, hs, 0, 0, 0,
                        (hs / bn) as u32, (mp / bm) as u32, 1
                    );
                }
            }

            let submit_every = if s >= 4096 { 2 } else { 4 };
            if s >= 512 && (li + 1) % submit_every == 0 {
                drop(cp);
                gpu.queue.submit([enc.finish()]);
                if let Err(e) = gpu.device.poll(wgpu::PollType::wait_indefinitely()) {
                    anyhow::bail!("prefill: device lost after layer {li}: {e:?}");
                }
                enc = gpu.device.create_command_encoder(&Default::default());
                cp = enc.begin_compute_pass(&Default::default());
                gd_slot = 2 * MAX_SLAB;
            }
        }

        let last_off = ((s - 1) * hs * 2) as u64;
        let bg_fn = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("p.final_rms"),
            layout: &self.pipes.rms_norm.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &h_buf,
                        offset: last_off,
                        size: std::num::NonZeroU64::new((hs * 2) as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: self.norm_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.scratch.final_norm.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.scratch.u_rms.as_entire_binding() },
            ],
        });
        cp.set_pipeline(&self.pipes.rms_norm);
        cp.set_bind_group(0, &bg_fn, &[]);
        cp.dispatch_workgroups(1, 1, 1);
        cp.set_pipeline(&self.pipes.gemv_lm);
        cp.set_bind_group(0, &self.bg_gemv_lm, &[]);
        cp.dispatch_workgroups((cfg.vocab_size / 8) as u32, 1, 1);
        cp.set_pipeline(&self.pipes.argmax);
        cp.set_bind_group(0, &self.bg_argmax, &[]);
        cp.dispatch_workgroups(1, 1, 1);
        drop(cp);
        enc.copy_buffer_to_buffer(&self.scratch.token, 0, &self.scratch.token_staging, 0, 4);
        gpu.queue.submit([enc.finish()]);

        {
            let dbg = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("p.dbg_h0"),
                size: (hs * 2) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut denc = gpu.device.create_command_encoder(&Default::default());
            denc.copy_buffer_to_buffer(&h_buf, 0, &dbg, 0, (hs * 2) as u64);
            gpu.queue.submit([denc.finish()]);
            self.debug_l0_h = Some(dbg);
        }
        self.debug_prefill_h = Some(h_buf);
        let tok = self.read_token()?;
        self.pos = kv_start + s;
        Ok(tok)
    }
}
