use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use safetensors::Dtype;

use crate::gpu::{BulkUpload, Gpu};
use crate::shaders;
use crate::weights::{self, PackedWeight};

/// Rows per workgroup for the plain-[`shaders::gemv`] pipelines.
///
/// One warp per row is fixed by the alignment contract (each lane owns words
/// `lane, lane+32, …` of its row and the xor butterfly combines the 32 lanes in
/// a fixed tree).  How many rows *share a workgroup* is not: it only decides how
/// many workgroups the dispatch gets, and `o_proj` / `down_proj` are the two
/// shapes small enough that the machine is unfilled at 8 rows (measured in
/// `gemv_bench`: 82 and 105 GB/s at 128 workgroups vs 289 GB/s for `lm_head` at
/// 18 992).  Every row is still produced by the same warp with the same lane
/// mapping, so the output is bit-identical at any value.
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
    bg_gemv_qkv: wgpu::BindGroup,
    bg_gemv_o: wgpu::BindGroup,
    bg_gemv_gu: wgpu::BindGroup,
    bg_gemv_dp: wgpu::BindGroup,
    bg_gemv_qkv_norm: wgpu::BindGroup,
    bg_gemv_gu_norm: wgpu::BindGroup,
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
    gemv_lm: wgpu::ComputePipeline,
    gemv_qkv_norm: wgpu::ComputePipeline,
    gemv_gu_norm: wgpu::ComputePipeline,
    gemv_lm_norm: wgpu::ComputePipeline,
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
    softmax: std::collections::HashMap<usize, wgpu::ComputePipeline>,
    /// Narrower physical workgroups computing a wider tree, keyed by
    /// `(phys, tree)`; see [`softmax_blocks`] and `softmax_causal_tree`.
    softmax_wide: std::collections::HashMap<(usize, usize), wgpu::ComputePipeline>,
    repeat_kv: wgpu::ComputePipeline,
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
    let bg_extract = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("qkv_extract"),
        layout: &pipes.extract.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: scratch.qkv.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: qn_w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: kn_w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: scratch.cos.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: scratch.sin.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: scratch.q_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: k_cache.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: v_cache.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: scratch.u_qkvx.as_entire_binding() },
        ],
    });
    let bg_gqa = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gqa"),
        layout: &pipes.gqa256.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: scratch.q_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: k_cache.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: v_cache.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: scratch.attn_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: scratch.u_gqa.as_entire_binding() },
        ],
    });
    let bg_gqa_split = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gqa_split"),
        layout: &pipes.gqa_split256.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: scratch.q_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: k_cache.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: v_cache.as_entire_binding() },
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

/// KV slots allocated at load (112 KiB each, see [`WgpuTextDecoder::load`]).
///
/// `max_seq` is the ceiling, not the allocation: the cache grows from here on
/// demand ([`WgpuTextDecoder::ensure_capacity`]), so a 15-minute request does not
/// make every 15-second clip pay for the 16 384 slots it might need.
pub const KV_INITIAL_CAP: usize = 1024;

/// Capacity grows in multiples of this — a series of clips of one shape then
/// allocates once and pays nothing after that.
pub(crate) const KV_STEP: usize = 256;

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

    embed_table: wgpu::Buffer,
    layers: Vec<Layer>,
    pipes: Pipes,
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
        let hs = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let nqh = cfg.num_attention_heads;
        let nkvh = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let inter = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let nl = cfg.num_hidden_layers;

        let build =
            |label: &str, src: &str, entry: &str, pl: Option<&wgpu::PipelineLayout>| -> Result<wgpu::ComputePipeline> {
                gpu.pipeline(label, src, entry, pl)
                    .with_context(|| format!("build pipeline {label}"))
            };

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
        let sm_pl = family_layout_dyn(&gpu, "softmax", &[(0, true), (1, false)], 2, true);
        let rk_pl = family_layout(&gpu, "repeat_kv", &[(0, true), (1, false)], 2);

        let rms_bs = block_for_reduction(hs) as usize;
        let subgroup = gpu.features.contains(wgpu::Features::SUBGROUP)
            && gpu.info.subgroup_min_size == 32
            && gpu.info.subgroup_max_size == 32;
        let pipes = Pipes {
            gemv_qkv: build("gemv_qkv", &shaders::gemv(cfg.fused_qkv_cols(), hs, false, subgroup, gemv_rpw()), "gemv", None)?,
            gemv_o: build("gemv_o", &shaders::gemv(hs, q_dim, true, subgroup, gemv_rpw()), "gemv", None)?,
            gemv_gu: build("gemv_gu", &shaders::gemv(2 * inter, hs, false, subgroup, gemv_rpw()), "gemv", None)?,
            gemv_dp: build("gemv_dp", &shaders::gemv(hs, inter, true, subgroup, gemv_rpw()), "gemv", None)?,
            gemv_lm: build("gemv_lm", &shaders::gemv(vocab, hs, false, subgroup, gemv_rpw()), "gemv", None)?,
            gemv_qkv_norm: build(
                "gemv_qkv_norm",
                &shaders::gemv_norm(cfg.fused_qkv_cols(), hs, false, subgroup, hs, rms_bs as usize, cfg.rms_norm_eps),
                "gemv",
                None,
            )?,
            gemv_gu_norm: build(
                "gemv_gu_norm",
                &shaders::gemv_norm(2 * inter, hs, false, subgroup, hs, rms_bs as usize, cfg.rms_norm_eps),
                "gemv",
                None,
            )?,
            gemv_lm_norm: build(
                "gemv_lm_norm",
                &shaders::gemv_norm(vocab, hs, false, subgroup, hs, rms_bs as usize, cfg.rms_norm_eps),
                "gemv",
                None,
            )?,
            rms_norm: build("rms_norm", &shaders::rms_norm(hs, rms_bs), "rms_norm", None)?,
            extract: build("qkv_extract", &shaders::qkv_extract(nqh, nkvh, hd), "qkv_extract", None)?,
            gqa256: build("gqa256", &shaders::gqa_decode_single(nqh, nkvh, hd, 256, GQA_SINGLE_CAP), "gqa", Some(&gqa_pl))?,
            gqa512: build("gqa512", &shaders::gqa_decode_single(nqh, nkvh, hd, 512, GQA_SINGLE_CAP), "gqa", Some(&gqa_pl))?,
            gqa_split128: build("gqa_split128", &shaders::gqa_decode_split_p1(nqh, nkvh, hd, 128), "gqa_split_p1", Some(&split_pl))?,
            gqa_split_pair256: build("gqa_split_pair256", &pair_split_src(nqh, nkvh, hd, 256, false, subgroup && gqa_coop()  , gqa_pf(), false, gqa_depth()), pair_split_entry(nqh, nkvh), Some(&split_pl))?,
            gqa_split_pair512: build("gqa_split_pair512", &pair_split_src(nqh, nkvh, hd, 512, false, subgroup && gqa_coop()  , gqa_pf(), false, gqa_depth()), pair_split_entry(nqh, nkvh), Some(&split_pl))?,
            gqa_split_pair256_row0: build("gqa_split_pair256_row0", &shaders::gqa_decode_split_p1_pair(hd, 256, 2, true, false, false, false, 1), "gqa_split_p1_pair", Some(&split_pl))?,
            gqa_split_pair256_coal: build("gqa_split_pair256_coal", &shaders::gqa_decode_split_p1_pair(hd, 256, 2, false, false, false, true, 1), "gqa_split_p1_pair", Some(&split_pl))?,
            gqa_split256: build("gqa_split256", &shaders::gqa_decode_split_p1(nqh, nkvh, hd, 256), "gqa_split_p1", Some(&split_pl))?,
            gqa_split512: build("gqa_split512", &shaders::gqa_decode_split_p1(nqh, nkvh, hd, 512), "gqa_split_p1", Some(&split_pl))?,
            gqa_merge: build("gqa_merge", &shaders::gqa_split_merge(hd), "gqa_merge", None)?,
            silu: build("silu30", &shaders::silu_mul_split(inter), "silu_mul_split", None)?,
            argmax: build("argmax", &shaders::argmax_into_slot(), "argmax", None)?,
            embed: build("embed", &shaders::embed_lookup_single(), "embed", None)?,
            gemm: build("gemm", &shaders::prefill_gemm(false, false), "gemm", Some(&gemm_pl))?,
            gemm_acc: build("gemm_acc", &shaders::prefill_gemm(false, true), "gemm", Some(&gemm_pl))?,
            gemm_av: build("gemm_av", &shaders::prefill_gemm(true, false), "gemm", Some(&gemm_pl))?,
            gemm_causal: build("gemm_causal", &shaders::prefill_gemm_causal(), "gemm", Some(&gemm_pl))?,
            gemm_av_causal: build("gemm_av_causal", &shaders::prefill_gemm_causal_av(), "gemm", Some(&gemm_pl))?,
            softmax: std::collections::HashMap::from([
                (32, build("softmax32", &shaders::softmax_causal(32, subgroup && sg_reduce()), "softmax", Some(&sm_pl))?),
                (64, build("softmax64", &shaders::softmax_causal(64, subgroup && sg_reduce()), "softmax", Some(&sm_pl))?),
                (128, build("softmax128", &shaders::softmax_causal(128, subgroup && sg_reduce()), "softmax", Some(&sm_pl))?),
                (256, build("softmax256", &shaders::softmax_causal(256, subgroup && sg_reduce()), "softmax", Some(&sm_pl))?),
                (512, build("softmax512", &shaders::softmax_causal(512, subgroup && sg_reduce()), "softmax", Some(&sm_pl))?),
                (1024, build("softmax1024", &shaders::softmax_causal(1024, subgroup && sg_reduce()), "softmax", Some(&sm_pl))?),
            ]),
            // Every `(phys, tree)` the shipped `SOFTMAX_BS_PHYS` can meet must be
            // here: `tree` is the row-length ladder (256/512/1024), and a physical
            // width that lands on a missing pair is a panic, not a fallback.
            softmax_wide: std::collections::HashMap::from([
                ((64, 256), build("softmax64x256", &shaders::softmax_causal_tree(64, 256, subgroup && sg_reduce()), "softmax", Some(&sm_pl))?),
                ((64, 512), build("softmax64x512", &shaders::softmax_causal_tree(64, 512, subgroup && sg_reduce()), "softmax", Some(&sm_pl))?),
                ((64, 1024), build("softmax64x1024", &shaders::softmax_causal_tree(64, 1024, subgroup && sg_reduce()), "softmax", Some(&sm_pl))?),
                ((128, 256), build("softmax128x256", &shaders::softmax_causal_tree(128, 256, subgroup && sg_reduce()), "softmax", Some(&sm_pl))?),
                ((128, 512), build("softmax128x512", &shaders::softmax_causal_tree(128, 512, subgroup && sg_reduce()), "softmax", Some(&sm_pl))?),
                ((128, 1024), build("softmax128x1024", &shaders::softmax_causal_tree(128, 1024, subgroup && sg_reduce()), "softmax", Some(&sm_pl))?),
                ((256, 512), build("softmax256x512", &shaders::softmax_causal_tree(256, 512, subgroup && sg_reduce()), "softmax", Some(&sm_pl))?),
                ((256, 1024), build("softmax256x1024", &shaders::softmax_causal_tree(256, 1024, subgroup && sg_reduce()), "softmax", Some(&sm_pl))?),
            ]),
            repeat_kv: build("repeat_kv", &shaders::repeat_kv(nqh / nkvh), "repeat_kv", Some(&rk_pl))?,
            slab_stats: build(
                "slab_stats",
                &shaders::slab_stats(SLAB_BS, SLAB_T),
                "slab_stats",
                Some(&family_layout_dyn(&gpu, "slab_stats", &[(0, true), (1, false)], 2, true)),
            )?,
            slab_weights: build(
                "slab_weights",
                &shaders::slab_weights(256),
                "slab_weights",
                Some(&family_layout(&gpu, "slab_weights", &[(0, true), (1, false)], 2)),
            )?,
            slab_merge: build(
                "slab_merge",
                &shaders::slab_merge(nqh, hd),
                "slab_merge",
                Some(&family_layout(&gpu, "slab_merge", &[(0, true), (1, true), (2, false)], 3)),
            )?,
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
                if tx.send(convert_layer(&w, &p)).is_err() {
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
            let LayerWeights { qkv: qkv_p, o: o_w, gu: gu_p, dp: dp_w, iln: iln_v, pln: pln_v, qn: qn_v, kn: kn_v } = lw;

            let t = std::time::Instant::now();
            let qkv = up.upload_pieces("qkv_w", &qkv_p)?;
            let o = upload_weight(&mut up, "o_w", &o_w)?;
            let gu = up.upload_pieces("gu_w", &gu_p)?;
            let dp = upload_weight(&mut up, "dp_w", &dp_w)?;
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
            let bg_gemv_gu = gemv_bg(&pipes.gemv_gu, &gu, &scratch.norm2, &scratch.gate_up);
            let bg_gemv_gu_norm = gemv_norm_bg(&pipes.gemv_gu_norm, &gu, &scratch.h, &scratch.gate_up, &pln);
            let bg_gemv_dp = gemv_bg(&pipes.gemv_dp, &dp, &scratch.activated, &scratch.h);
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
                bg_gemv_qkv,
                bg_gemv_qkv_norm,
                bg_gemv_o,
                bg_gemv_gu,
                bg_gemv_gu_norm,
                bg_gemv_dp,
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
            embed_table,
            layers,
            pipes,
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
        let target = (need.div_ceil(KV_STEP) * KV_STEP).min(self.max_seq).max(self.cap);
        if target == self.cap {
            return self.cap;
        }
        let t0 = std::time::Instant::now();
        let prev = self.cap;
        let kv_words = (self.cfg.num_key_value_heads * target * self.cfg.head_dim / 2) as u64;
        {
            let Self { gpu, layers, pipes, scratch, .. } = self;
            for layer in layers.iter_mut() {
                let k_cache = gpu.storage("k_cache", kv_words * 4);
                let v_cache = gpu.storage("v_cache", kv_words * 4);
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

    pub fn encode_step(&self, enc: &mut wgpu::CommandEncoder, pos: usize) {
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

        let mut cp = enc.begin_compute_pass(&Default::default());

        cp.set_pipeline(&self.pipes.embed);
        cp.set_bind_group(0, &self.bg_embed, &[]);
        cp.dispatch_workgroups(1, 1, 1);

        for l in &self.layers {
            if !k("qkv") {
                cp.set_pipeline(&self.pipes.gemv_qkv_norm);
                cp.set_bind_group(0, &l.bg_gemv_qkv_norm, &[]);
                cp.dispatch_workgroups(norm_grid(cfg.fused_qkv_cols()), 1, 1);
            }
            if d("qkv") {
                cp.dispatch_workgroups(norm_grid(cfg.fused_qkv_cols()), 1, 1);
            }

            if !k("extract") {
                cp.set_pipeline(&self.pipes.extract);
                cp.set_bind_group(0, &l.bg_extract, &[]);
                cp.dispatch_workgroups(1, (cfg.num_attention_heads + cfg.num_key_value_heads) as u32, 1);
            }
            if d("extract") {
                cp.dispatch_workgroups(1, (cfg.num_attention_heads + cfg.num_key_value_heads) as u32, 1);
            }

            if !k("gqa") {
                self.encode_gqa(&mut cp, l, cur_len);
            }
            if d("gqa") {
                if !k("gqa") {
                self.encode_gqa(&mut cp, l, cur_len);
            }
            }

            if !k("o") {
                cp.set_pipeline(&self.pipes.gemv_o);
                cp.set_bind_group(0, &l.bg_gemv_o, &[]);
                cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1);
            }
            if d("o") {
                cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1);
            }

            if !k("gu") {
                cp.set_pipeline(&self.pipes.gemv_gu_norm);
                cp.set_bind_group(0, &l.bg_gemv_gu_norm, &[]);
                cp.dispatch_workgroups(norm_grid(2 * cfg.intermediate_size), 1, 1);
            }
            if d("gu") {
                cp.dispatch_workgroups(norm_grid(2 * cfg.intermediate_size), 1, 1);
            }

            if !k("silu") {
                cp.set_pipeline(&self.pipes.silu);
                cp.set_bind_group(0, &self.bg_silu, &[]);
                cp.dispatch_workgroups(silu_grid.0, silu_grid.1, 1);
            }
            if d("silu") {
                cp.dispatch_workgroups(silu_grid.0, silu_grid.1, 1);
            }

            if !k("dp") {
                cp.set_pipeline(&self.pipes.gemv_dp);
                cp.set_bind_group(0, &l.bg_gemv_dp, &[]);
                cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1);
            }
            if d("dp") {
                cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1);
            }
        }

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

        drop(cp);
        enc.copy_buffer_to_buffer(&self.scratch.token, 0, &self.scratch.token_staging, 0, 4);
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

    /// One decode step at `self.pos`, then `pos += 1`.  Returns the token argmax
    /// wrote into `token_buf[0]` (which the *next* step will embed).
    pub fn step(&mut self) -> Result<i32> {
        let pos = self.pos;
        let t_host = std::time::Instant::now();
        self.write_step_uniforms(pos);
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        self.encode_step(&mut enc, pos);
        self.gpu.queue.submit([enc.finish()]);
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
        cp.set_pipeline(&self.pipes.rms_norm);
        cp.set_bind_group(0, &layer.bg_rms1, &[]);
        cp.dispatch_workgroups(1, 1, 1);
        cp.set_pipeline(&self.pipes.gemv_qkv);
        cp.set_bind_group(0, &layer.bg_gemv_qkv, &[]);
        cp.dispatch_workgroups(gemv_grid(cfg.fused_qkv_cols()), 1, 1);
        cp.set_pipeline(&self.pipes.extract);
        cp.set_bind_group(0, &layer.bg_extract, &[]);
        cp.dispatch_workgroups(1, (cfg.num_attention_heads + cfg.num_key_value_heads) as u32, 1);
        self.encode_gqa(&mut cp, layer, cur_len);
        cp.set_pipeline(&self.pipes.gemv_o);
        cp.set_bind_group(0, &layer.bg_gemv_o, &[]);
        cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1);
        cp.set_pipeline(&self.pipes.rms_norm);
        cp.set_bind_group(0, &layer.bg_rms2, &[]);
        cp.dispatch_workgroups(1, 1, 1);
        cp.set_pipeline(&self.pipes.gemv_gu);
        cp.set_bind_group(0, &layer.bg_gemv_gu, &[]);
        cp.dispatch_workgroups(gemv_grid(2 * cfg.intermediate_size), 1, 1);
        cp.set_pipeline(&self.pipes.silu);
        cp.set_bind_group(0, &self.bg_silu, &[]);
        cp.dispatch_workgroups(silu_grid.0, silu_grid.1, 1);
        cp.set_pipeline(&self.pipes.gemv_dp);
        cp.set_bind_group(0, &layer.bg_gemv_dp, &[]);
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
        match op {
            0 => { cp.set_pipeline(&self.pipes.rms_norm); cp.set_bind_group(0, &layer.bg_rms1, &[]); cp.dispatch_workgroups(1, 1, 1); }
            1 => { cp.set_pipeline(&self.pipes.gemv_qkv); cp.set_bind_group(0, &layer.bg_gemv_qkv, &[]); cp.dispatch_workgroups(gemv_grid(cfg.fused_qkv_cols()), 1, 1); }
            2 => { cp.set_pipeline(&self.pipes.extract); cp.set_bind_group(0, &layer.bg_extract, &[]); cp.dispatch_workgroups(1, (cfg.num_attention_heads + cfg.num_key_value_heads) as u32, 1); }
            3 => self.encode_gqa(cp, layer, pos + 1),
            4 => { cp.set_pipeline(&self.pipes.gemv_o); cp.set_bind_group(0, &layer.bg_gemv_o, &[]); cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1); }
            5 => { cp.set_pipeline(&self.pipes.rms_norm); cp.set_bind_group(0, &layer.bg_rms2, &[]); cp.dispatch_workgroups(1, 1, 1); }
            6 => { cp.set_pipeline(&self.pipes.gemv_gu); cp.set_bind_group(0, &layer.bg_gemv_gu, &[]); cp.dispatch_workgroups(gemv_grid(2 * cfg.intermediate_size), 1, 1); }
            7 => { cp.set_pipeline(&self.pipes.silu); cp.set_bind_group(0, &self.bg_silu, &[]); cp.dispatch_workgroups(((cfg.intermediate_size / 2).div_ceil(256)) as u32, 1, 1); }
            8 => { cp.set_pipeline(&self.pipes.gemv_dp); cp.set_bind_group(0, &layer.bg_gemv_dp, &[]); cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1); }
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
        let per_layer = (c.fused_qkv_cols() * c.hidden_size
            + c.hidden_size * c.q_dim()
            + 2 * c.intermediate_size * c.hidden_size
            + c.hidden_size * c.intermediate_size) as u64
            * 2;
        per_layer * c.num_hidden_layers as u64 + (c.vocab_size * c.hidden_size) as u64 * 2
    }

    pub fn embed_table(&self) -> &wgpu::Buffer {
        &self.embed_table
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

    let mut out = Vec::with_capacity(total_bytes(&tensors)?);
    for t in &tensors {
        let start = out.len();
        out.resize(start + t.data.len() / t.dtype.size() * 2, 0);
        t.narrow_f16_into(&mut out[start..])?;
    }
    Ok(vec![(0, out.into())])
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
    iln: Vec<half::f16>,
    pln: Vec<half::f16>,
    qn: Vec<half::f16>,
    kn: Vec<half::f16>,
}

/// Read one layer's tensors out of the checkpoint in upload-ready form.
fn convert_layer(w: &HashMap<String, weights::RawTensor>, p: &str) -> Result<LayerWeights> {
    Ok(LayerWeights {
        o: weights::get_matrix(w, &format!("{p}.self_attn.o_proj.weight"))?,
        dp: weights::get_matrix(w, &format!("{p}.mlp.down_proj.weight"))?,
        qkv: fused_pieces(w, &format!("{p}.self_attn"), &["q_proj", "k_proj", "v_proj"])?,
        gu: fused_pieces(w, &format!("{p}.mlp"), &["gate_proj", "up_proj"])?,
        iln: weights::get_vector(w, &format!("{p}.input_layernorm.weight"))?,
        pln: weights::get_vector(w, &format!("{p}.post_attention_layernorm.weight"))?,
        qn: weights::get_vector(w, &format!("{p}.self_attn.q_norm.weight"))?,
        kn: weights::get_vector(w, &format!("{p}.self_attn.k_norm.weight"))?,
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
    #[allow(clippy::too_many_lines)]
    pub fn prefill(&mut self, hidden_words: &[u8], s: usize, kv_start: usize) -> Result<i32> {
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
        let bind_limit = self.gpu.limits.max_storage_buffer_binding_size as u64;
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

        let words = |rows: usize, cols: usize| (rows * cols / 2 * 4) as u64;
        let mut up = self.gpu.uploader();
        let h_buf = up.storage("p.h", words(mp, hs));
        let normed = up.storage("p.normed", words(mp, hs));
        let norm2 = up.storage("p.norm2", words(mp, hs));
        let qkv = up.storage("p.qkv", words(mp, cfg.fused_qkv_cols()));
        let q_out = up.storage("p.q_out", words(nqh * mp, hd));
        let slab_cols = if slabbed { slab_t } else { np };
        let scores = up.storage("p.scores", words(nqh * mp, slab_cols));
        let k_rep = up.storage("p.k_rep", words(nqh * nkv_rows, hd));
        let v_rep = up.storage("p.v_rep", words(nqh * nkv_rows, hd));
        let attn = up.storage("p.attn", words(nqh * mp, if slabbed { slab_t } else { cur16 }));
        let slab_part = slabbed.then(|| up.storage("p.slab_part", words(n_slab * mp, nqh * hd)));
        let slab_stats = slabbed.then(|| up.storage("p.slab_stats", (n_slab * nqh * mp * 2 * 4) as u64));
        let slab_w = slabbed.then(|| up.storage("p.slab_w", (n_slab * nqh * mp * 4) as u64));
        let attn_flat = up.storage("p.attn_flat", words(mp, nqh * hd));
        let gu = up.storage("p.gu", words(mp, 2 * inter));
        let activated = up.storage("p.activated", words(mp, inter));
        const MAX_GEMMS: u64 = 256;
        let u_gd = up.uniform("p.gd", MAX_GEMMS * 256);
        let u_sl = up.uniform("p.sl", (2 * MAX_SLAB as u64 + 2) * 256);
        let u_rk = up.uniform("p.rk", 32);
        up.upload(&h_buf, hidden_words)?;
        up.finish()?;

        let gpu = &self.gpu;
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
                    },
                    $gx, $gy, $gz
                );
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
                }),
            );
        }
        let tail_rows = nkv_rows - cur;
        if tail_rows > 0 {
            let zero = vec![0u8; tail_rows * hd * 2];
            for h in 0..nqh {
                let off = ((h * nkv_rows + cur) * hd * 2) as u64;
                gpu.write_at(&k_rep, off, &zero);
                gpu.write_at(&v_rep, off, &zero);
            }
        }
        let repeat_grid = grid_xy((nqh * cur * hd / 2).div_ceil(256));
        gpu.upload(
            &u_rk,
            bytemuck::bytes_of(&RepeatKvCfg {
                nkvh: nkvh as u32,
                max_seq: self.cap as u32,
                cur: cur as u32,
                hd: hd as u32,
                npw: (nkv_rows * hd / 2) as u32,
                gx: repeat_grid.0,
            }),
        );

        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let mut cp = enc.begin_compute_pass(&Default::default());

        let dup = std::env::var("QASR_DUP").unwrap_or_default();

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

            gemm!(
                &mut cp, &self.pipes.gemm, &normed, &layer.qkv_w, &qkv,
                s, cfg.fused_qkv_cols(), hs, cfg.fused_qkv_cols(), 0, 0, 0,
                (cfg.fused_qkv_cols() / 128) as u32, (mp / 128) as u32, 1
            );

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

            for (cache, out) in [(&layer.k_cache, &k_rep), (&layer.v_cache, &v_rep)] {
                let bg_rk = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("p.rk"),
                    layout: &self.pipes.repeat_kv.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: cache.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: u_rk.as_entire_binding() },
                    ],
                });
                cp.set_pipeline(&self.pipes.repeat_kv);
                cp.set_bind_group(0, &bg_rk, &[]);
                cp.dispatch_workgroups(repeat_grid.0, repeat_grid.1, 1);
                if dup == "p_repeat" {
                    cp.dispatch_workgroups(repeat_grid.0, repeat_grid.1, 1);
                }
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
                        bsb: (nkv_rows * hd / 2) as u32,
                        bsc: (mp * slab_t) as u32,
                        beta: 0,
                        row0: t0 as u32,
                        lda: hd as u32,
                    };
                    let sgrid = ((slab_t / 128) as u32, (mp / 128) as u32, nqh as u32);
                    gemm_at!(
                        &mut cp, &self.pipes.gemm_causal, &q_out, &k_rep, &scores,
                        0u64, t, sg,
                        sgrid.0, sgrid.1, sgrid.2
                    );
                    if dup == "p_scores" {
                        gemm_at!(
                            &mut cp, &self.pipes.gemm_causal, &q_out, &k_rep, &scores,
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
                                    size: std::num::NonZeroU64::new(32),
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
                                    size: std::num::NonZeroU64::new(32),
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
                        bsb: (nkv_rows * hd / 2) as u32,
                        bsc: hd as u32,
                        beta: 0,
                        row0: t0 as u32,
                        lda: slab_t as u32,
                    };
                    gemm_at!(
                        &mut cp, &self.pipes.gemm_av_causal, &attn, &v_rep, spart,
                        (t * mp * nqh * hd * 2) as u64, n_slab + t, ag,
                        1, (mp / 128) as u32, nqh as u32
                    );
                    if dup == "p_av" {
                        gemm_at!(
                            &mut cp, &self.pipes.gemm_av_causal, &attn, &v_rep, spart,
                            (t * mp * nqh * hd * 2) as u64, n_slab + t, ag,
                            1, (mp / 128) as u32, nqh as u32
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
                                size: std::num::NonZeroU64::new(32),
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
                                size: std::num::NonZeroU64::new(32),
                            }),
                        },
                    ],
                });
                cp.set_pipeline(&self.pipes.slab_merge);
                cp.set_bind_group(0, &bg_m, &[]);
                cp.dispatch_workgroups(slab_m_grid, nqh as u32, 1);
            } else {
                gemm!(
                    &mut cp, &self.pipes.gemm_causal, &q_out, &k_rep, &scores,
                    s, cur, hd, np, mp * hd / 2, np * hd / 2, mp * np,
                    (np / 128) as u32, (mp / 128) as u32, nqh as u32
                );
                if dup == "p_scores" {
                    gemm!(
                        &mut cp, &self.pipes.gemm, &q_out, &k_rep, &scores,
                        s, cur, hd, np, mp * hd / 2, np * hd / 2, mp * np,
                        (np / 128) as u32, (mp / 128) as u32, nqh as u32
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
                                size: std::num::NonZeroU64::new(32),
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

                gemm!(
                    &mut cp, &self.pipes.gemm_av_causal, &attn, &v_rep, &attn_flat,
                    s, hd, cur16, nqh * hd, mp * cur16 / 2, np * hd / 2, hd,
                    1, (mp / 128) as u32, nqh as u32
                );
                if dup == "p_av" {
                    gemm!(
                        &mut cp, &self.pipes.gemm_av, &attn, &v_rep, &attn_flat,
                        s, hd, cur16, nqh * hd, mp * cur16 / 2, np * hd / 2, hd,
                        1, (mp / 128) as u32, nqh as u32
                    );
                }
            }

            gemm!(
                &mut cp, &self.pipes.gemm_acc, &attn_flat, &layer.o_w, &h_buf,
                s, hs, nqh * hd, hs, 0, 0, 0,
                (hs / 128) as u32, (mp / 128) as u32, 1
            );

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

            gemm!(
                &mut cp, &self.pipes.gemm, &norm2, &layer.gu_w, &gu,
                s, 2 * inter, hs, 2 * inter, 0, 0, 0,
                ((2 * inter) / 128) as u32, (mp / 128) as u32, 1
            );

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

            gemm!(
                &mut cp, &self.pipes.gemm_acc, &activated, &layer.dp_w, &h_buf,
                s, hs, inter, hs, 0, 0, 0,
                (hs / 128) as u32, (mp / 128) as u32, 1
            );

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
