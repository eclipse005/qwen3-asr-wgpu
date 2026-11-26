#![allow(dead_code)]

/// Which subgroup (warp) shape the adapter exposes.
///
/// Stage 1 only classifies the device; the shipped 32-lane shuffles stay
/// unchanged. Future kernels can select a reduction path from this value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgroupMode {
    /// Exactly 32 lanes: the existing XOR-butterfly shuffles are valid.
    Warp32,
    /// Subgroups are a multiple of 32 (64..=64 or RDNA's 32..=64): a
    /// portable subgroup reduction covering more than one warp.
    Warp64,
    /// Subgroups exist but in some other width range: use portable
    /// subgroup builtins rather than 32-lane XOR shuffles.
    Generic,
    /// No subgroup feature: fall back to the shared-memory tree.
    None,
}

pub const LOG2_E: f32 = std::f32::consts::LOG2_E;

/// Reduction fragments shared by every 32-lane-per-row GEMV kernel.
///
/// `Warp32` and `Warp64` emit the legacy XOR tree verbatim: XOR deltas 16..1
/// never cross a 32-lane boundary, even when the hardware subgroup is 64 wide.
/// `Generic` reduces within each subgroup and folds those sums through one
/// workgroup array; both barriers stay in uniform control flow at the top of
/// `bfly` so Naga's uniformity analysis accepts them across repeated calls.
/// `None` keeps the original `bt0`/`bt1` tree byte for byte.
struct BflyFragments {
    scratch: String,
    signature: &'static str,
    fn_tail: &'static str,
    args: &'static str,
    body: &'static str,
}

// These strings use explicit `\x20` for leading spaces because a Rust `\` line
// continuation strips them; the generated WGSL must keep every indent.
const XOR_BFLY_BODY: &str = "\x20\x20\x20\x20var t = v;\n\x20\x20\x20\x20t = t + subgroupShuffleXor(t, 16u);\n\x20\x20\x20\x20t = t + subgroupShuffleXor(t, 8u);\n\x20\x20\x20\x20t = t + subgroupShuffleXor(t, 4u);\n\x20\x20\x20\x20t = t + subgroupShuffleXor(t, 2u);\n\x20\x20\x20\x20t = t + subgroupShuffleXor(t, 1u);\n\x20\x20\x20\x20return t;";

const SHARED_BFLY_BODY: &str = "\x20\x20\x20\x20let wb = lid & 0xFFFFFFE0u;\n\x20\x20\x20\x20bt0[lid] = v;\n\x20\x20\x20\x20workgroupBarrier();\n\x20\x20\x20\x20var t = bt0[lid] + bt0[wb + (lane ^ 16u)];\n\x20\x20\x20\x20bt1[lid] = t;\n\x20\x20\x20\x20workgroupBarrier();\n\x20\x20\x20\x20t = bt1[lid] + bt1[wb + (lane ^ 8u)];\n\x20\x20\x20\x20bt0[lid] = t;\n\x20\x20\x20\x20workgroupBarrier();\n\x20\x20\x20\x20t = bt0[lid] + bt0[wb + (lane ^ 4u)];\n\x20\x20\x20\x20bt1[lid] = t;\n\x20\x20\x20\x20workgroupBarrier();\n\x20\x20\x20\x20t = bt1[lid] + bt1[wb + (lane ^ 2u)];\n\x20\x20\x20\x20bt0[lid] = t;\n\x20\x20\x20\x20workgroupBarrier();\n\x20\x20\x20\x20t = bt0[lid] + bt0[wb + (lane ^ 1u)];\n\x20\x20\x20\x20return t;";

const GENERIC_BFLY_BODY: &str = "\x20\x20\x20\x20let sg_sum = subgroupAdd(v);\n\x20\x20\x20\x20let sg_id = lid / sg_size;\n\x20\x20\x20\x20if (sg_lid == 0u) {\n\x20\x20\x20\x20\x20\x20\x20\x20sg_red[sg_id] = sg_sum;\n\x20\x20\x20\x20}\n\x20\x20\x20\x20workgroupBarrier();\n\x20\x20\x20\x20var total = 0.0;\n\x20\x20\x20\x20if (lane == 0u) {\n\x20\x20\x20\x20\x20\x20\x20\x20let sg_per_row = max(32u / sg_size, 1u);\n\x20\x20\x20\x20\x20\x20\x20\x20let base_sg = (lid >> 5u) * sg_per_row;\n\x20\x20\x20\x20\x20\x20\x20\x20for (var k = 0u; k < sg_per_row; k = k + 1u) {\n\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20total = total + sg_red[base_sg + k];\n\x20\x20\x20\x20\x20\x20\x20\x20}\n\x20\x20\x20\x20}\n\x20\x20\x20\x20workgroupBarrier();\n\x20\x20\x20\x20return total;";

/// `scratch_kind` selects the shared-array declaration style: some kernels
/// spell the workgroup size `THREADS`, the fixed-256 ones spell it as `256`.
fn bfly_fragments(mode: SubgroupMode, threads: usize, scratch_kind: &str) -> BflyFragments {
    match mode {
        SubgroupMode::Warp32 | SubgroupMode::Warp64 => BflyFragments {
            scratch: String::new(),
            signature: "",
            fn_tail: "",
            args: "",
            body: XOR_BFLY_BODY,
        },
        SubgroupMode::None => BflyFragments {
            scratch: format!(
                "var<workgroup> bt0: array<f32, {scratch_kind}>;\nvar<workgroup> bt1: array<f32, {scratch_kind}>;\n"
            ),
            signature: "",
            fn_tail: "",
            args: "",
            body: SHARED_BFLY_BODY,
        },
        SubgroupMode::Generic => BflyFragments {
            scratch: format!("var<workgroup> sg_red: array<f32, {threads}>;\n"),
            signature: ",\n        @builtin(subgroup_invocation_id) sg_lid: u32,\n        @builtin(subgroup_size) sg_size: u32",
            fn_tail: ", sg_lid: u32, sg_size: u32",
            args: ", sg_lid, sg_size",
            body: GENERIC_BFLY_BODY,
        },
    }
}

const HALF_AT: &str = "\
fn half_at(v: vec2<f32>, i: u32) -> f32 { return select(v.y, v.x, (i & 1u) == 0u); }
";

/// `rms_norm_f16` — one workgroup per row, block tree reduction over `LAST`.
///
/// Bits that matter: `local += vx*vx + vy*vy` per `__half2` word (the odd/even
/// tail is irrelevant because every model dim is even), the `s`-halving tree, and
/// the two passes reading `x` twice.
pub fn rms_norm(last: usize, bs: usize) -> String {
    assert!(
        last % 2 == 0 && last >= bs,
        "rms_norm: last must be even and >= bs"
    );
    format!(
        "{HALF_AT}
struct Cfg {{ eps: f32, _a: f32, _b: f32, _c: f32 }};

@group(0) @binding(0) var<storage, read>       X:   array<u32>;
@group(0) @binding(1) var<storage, read>       Wt:  array<u32>;
@group(0) @binding(2) var<storage, read_write> Out: array<u32>;
@group(0) @binding(3) var<uniform>             cfg: Cfg;

const LAST: u32 = {last}u;
const BS: u32 = {bs}u;
const LAST2: u32 = {last2}u;

var<workgroup> red: array<f32, {bs}>;

@compute @workgroup_size({bs})
fn rms_norm(@builtin(workgroup_id) wgid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {{
    // row = workgroup id: decode dispatches one row; prefill dispatches s rows
    // of a [s, LAST] activation.  The reduction order is row-independent.
    let row = wgid.x * LAST2;
    var local = 0.0;
    for (var j = lid.x; j < LAST2; j = j + BS) {{
        let v = unpack2x16float(X[row + j]);
        local = local + (v.x * v.x + v.y * v.y);
    }}
    red[lid.x] = local;
    workgroupBarrier();
    for (var s = BS >> 1u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{ red[lid.x] = red[lid.x] + red[lid.x + s]; }}
        workgroupBarrier();
    }}
    let inv_rms = inverseSqrt(red[0] / f32(LAST) + cfg.eps);
    workgroupBarrier();
    for (var j = lid.x; j < LAST2; j = j + BS) {{
        let xv = unpack2x16float(X[row + j]);
        let wv = unpack2x16float(Wt[j]);
        Out[row + j] = pack2x16float(vec2<f32>(
            xv.x * inv_rms * wv.x,
            xv.y * inv_rms * wv.y));
    }}
}}
",
        last2 = last / 2
    )
}

/// [`rms_norm`] over `batch` sequences in one dispatch, `wgid.z` = the slot.
///
/// `X` and `Out` are `[slot, ...]` rows of one allocation, so the slot is an
/// address, not a branch: `base = wgid.z * SLOT` and every index is the shipped
/// kernel's plus that base.  The reduction tree, the rounding and the write are
/// unchanged, which is what keeps a batched norm bit-identical to a lone one.
///
/// Why: at b=8 the step issued 456 one-workgroup norm dispatches (two per layer
/// per slot plus the final one).
pub fn rms_norm_slots(last: usize, bs: usize, slot_words: usize, batch: usize) -> String {
    assert!(
        last % 2 == 0 && last >= bs,
        "rms_norm_slots: last must be even and >= bs"
    );
    assert!(
        (1..=16).contains(&batch),
        "rms_norm_slots: batch must be 1..=16"
    );
    assert_eq!(
        slot_words,
        last / 2,
        "rms_norm_slots: slot stride must be last/2 words"
    );
    format!(
        "{HALF_AT}
struct Cfg {{ eps: f32, _a: f32, _b: f32, _c: f32 }};

@group(0) @binding(0) var<storage, read>       Wt:  array<u32>;
@group(0) @binding(1) var<storage, read>       X:   array<u32>;
@group(0) @binding(2) var<storage, read_write> Out: array<u32>;
@group(0) @binding(3) var<uniform>             cfg: Cfg;

const LAST: u32 = {last}u;
const BS: u32 = {bs}u;
const LAST2: u32 = {last2}u;
const SLOT: u32 = {slot}u;

var<workgroup> red: array<f32, {bs}>;

@compute @workgroup_size({bs})
fn rms_norm_slots(@builtin(workgroup_id) wgid: vec3<u32>,
                  @builtin(local_invocation_id) lid: vec3<u32>) {{
    // row = workgroup id, offset by the slot: one row per (slot, row) pair.
    let row = wgid.z * SLOT + wgid.x * LAST2;
    var local = 0.0;
    for (var j = lid.x; j < LAST2; j = j + BS) {{
        let v = unpack2x16float(X[row + j]);
        local = local + (v.x * v.x + v.y * v.y);
    }}
    red[lid.x] = local;
    workgroupBarrier();
    for (var sh = BS >> 1u; sh > 0u; sh = sh >> 1u) {{
        if (lid.x < sh) {{ red[lid.x] = red[lid.x] + red[lid.x + sh]; }}
        workgroupBarrier();
    }}
    let inv_rms = inverseSqrt(red[0] / f32(LAST) + cfg.eps);
    workgroupBarrier();
    for (var j = lid.x; j < LAST2; j = j + BS) {{
        let xv = unpack2x16float(X[row + j]);
        let wv = unpack2x16float(Wt[j]);
        Out[row + j] = pack2x16float(vec2<f32>(
            xv.x * inv_rms * wv.x,
            xv.y * inv_rms * wv.y));
    }}
}}
",
        last = last,
        bs = bs,
        last2 = last / 2,
        slot = slot_words,
    )
}

/// [`gemv`] with a split-K dimension: `splits` workgroups (distinguished by
/// `wgid.z`) each own a contiguous range of the row's K granules and write an f32
/// partial, which [`gemv_merge`] combines.
///
/// **Measured and rejected** (`gemv_bench`, 2026-09-14): splitting K does not buy
/// the parallelism the shapes lack — `o_proj` gets *slower* (64 vs 82 GB/s) and
/// the other shapes are 0.96-1.01x, because the extra dispatch and the partial
/// traffic cost more than the added workgroups.  Kept only as the bench's A/B
/// variant; do not wire it into the decoder.  (Also not bit-identical: the
/// per-lane accumulator covers a sub-range of K.)
pub fn gemv_split(n: usize, k: usize, subgroup: bool, splits: usize) -> String {
    gemv_split_mode(
        n,
        k,
        if subgroup {
            SubgroupMode::Warp32
        } else {
            SubgroupMode::None
        },
        splits,
    )
}

pub fn gemv_split_mode(n: usize, k: usize, mode: SubgroupMode, splits: usize) -> String {
    assert_eq!(n % 8, 0, "gemv_split: rows must be a multiple of 8");
    assert_eq!(k % 8, 0, "gemv_split: k must be a multiple of 8");
    let kg = k / 8;
    assert_eq!(kg % 32, 0, "gemv_split: k/8 must be a multiple of 32");
    let granules_per_lane = kg / 32;
    assert_eq!(
        granules_per_lane % splits,
        0,
        "gemv_split: granules/lane must divide by splits"
    );
    let gpt = granules_per_lane / splits;

    let bfly = bfly_fragments(mode, 256, "256");
    let subgroup_body = bfly.body;
    let bfly_scratch = bfly.scratch;
    let builtins = bfly.signature;
    let fn_tail = bfly.fn_tail;
    let bfly_args = bfly.args;
    let mut body = String::new();
    for g in 0..gpt {
        body.push_str(&format!(
            "        let wv{g} = Wt[wbase + (sp * {gpt}u + {g}u) * 32u + lane];\n\
             \x20        let xv{g} = X[(sp * {gpt}u + {g}u) * 32u + lane];\n",
        ));
        for c in 0..4 {
            let sel = ["x", "y", "z", "w"][c];
            body.push_str(&format!(
                "        let w{g}{c} = unpack2x16float(wv{g}.{sel});\n\
                 \x20        let x{g}{c} = unpack2x16float(xv{g}.{sel});\n\
                 \x20        a{c} = fma(w{g}{c}.x, x{g}{c}.x, fma(w{g}{c}.y, x{g}{c}.y, a{c}));\n",
            ));
        }
    }
    format!(
        "@group(0) @binding(0) var<storage, read>       Wt: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       X:  array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> P:  array<f32>;

const KG: u32 = {kg}u;
const SPLITS: u32 = {splits}u;
const SUBGROUP: u32 = {subgroup_lit}u;

{bfly_scratch}var<workgroup> rows_out: array<f32, 8>;

fn bfly(v: f32, lid: u32, lane: u32{fn_tail}) -> f32 {{
{subgroup_body}
}}

@compute @workgroup_size(256)
fn gemv(@builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>{builtins}) {{
    let lane = lid.x & 31u;
    let warp = lid.x >> 5u;
    let row = workgroup_id.x * 8u + warp;
    let sp = workgroup_id.z;
    let wbase = row * KG;
    var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
    {{
{body}    }}
    let acc = (a0 + a1) + (a2 + a3);
    let r = bfly(acc, lid.x, lane{bfly_args});
    if (lane == 0u) {{ rows_out[warp] = r; }}
    workgroupBarrier();
    if (lid.x < 8u) {{
        P[(workgroup_id.x * 8u + lid.x) * SPLITS + sp] = rows_out[lid.x];
    }}
}}
",
        subgroup_lit = u32::from(mode != SubgroupMode::None),
        builtins = builtins,
        fn_tail = fn_tail,
        bfly_args = bfly_args,
    )
}

/// Combine the [`gemv_split`] partials: `Y[i] = sum_s P[i*SPLITS + s]`, plus the
/// residual word when `accum`.  One thread per packed f16 output word (two rows).
pub fn gemv_merge(accum: bool) -> String {
    format!(
        "struct Cfg {{ words: u32, splits: u32 }};

@group(0) @binding(0) var<storage, read>       P: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;

const ACCUM: u32 = {accum}u;

@compute @workgroup_size(256)
fn gemv_merge(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= cfg.words) {{ return; }}
    var va = 0.0;
    var vb = 0.0;
    for (var s = 0u; s < cfg.splits; s = s + 1u) {{
        va = va + P[(2u * i) * cfg.splits + s];
        vb = vb + P[(2u * i + 1u) * cfg.splits + s];
    }}
    if (ACCUM == 1u) {{
        let old = unpack2x16float(Y[i]);
        va = va + old.x;
        vb = vb + old.y;
    }}
    Y[i] = pack2x16float(vec2<f32>(va, vb));
}}
",
        accum = u32::from(accum),
    )
}

/// `gemv_f16` -- warp-per-row, `uint4` (8-half) lane-strided loads on both the
/// weight row and the activation vector, four independent f32 accumulators,
/// 5-round xor butterfly over 32-lane warps, residual add folded into the epilogue.
///
/// `n` must be a multiple of 8 (all model shapes are) so no partial workgroup
/// exists; `k/8` must be a multiple of 32 so the granule loop divides evenly.
pub fn gemv(n: usize, k: usize, accum: bool, subgroup: bool, rows_per_wg: usize) -> String {
    gemv_mode(
        n,
        k,
        accum,
        if subgroup {
            SubgroupMode::Warp32
        } else {
            SubgroupMode::None
        },
        rows_per_wg,
    )
}

pub fn gemv_mode(
    n: usize,
    k: usize,
    accum: bool,
    mode: SubgroupMode,
    rows_per_wg: usize,
) -> String {
    gemv_impl(n, k, accum, mode, rows_per_wg, 1, false)
}

/// [`gemv`] with `rows_per_warp` rows per warp instead of one.
///
/// One warp per row is fixed by the alignment contract -- each lane owns words
/// `lane, lane+32, ...` of its row and the xor butterfly combines the 32 lanes in
/// a fixed tree -- but *how many rows a warp walks* is not: a warp that walks each
/// of its rows in turn produces every row with the same lane mapping, the same
/// add order and the same butterfly, so this is bit-identical at any value.
///
/// Why it is worth a knob: the warp count per dispatch is `rows / rows_per_warp`,
/// and this part holds 64 warps per SM (960 across 15).  `o_proj` and `down_proj`
/// are 1024 rows, so at one row per warp they need 1024 warps -- 1.07 waves, and
/// the leaked 0.07 is a *latency*-bound tail of 64 warps that cannot saturate
/// DRAM.  Two rows per warp is 512 warps in one wave.  Whether that is what the
/// 204 GB/s is, or whether it is a fixed per-dispatch ramp, is what
/// `gemv_bench`'s `rpw2` column answers.
pub fn gemv_rpwr(
    n: usize,
    k: usize,
    accum: bool,
    subgroup: bool,
    rows_per_wg: usize,
    rows_per_warp: usize,
) -> String {
    gemv_rpwr_mode(
        n,
        k,
        accum,
        if subgroup {
            SubgroupMode::Warp32
        } else {
            SubgroupMode::None
        },
        rows_per_wg,
        rows_per_warp,
    )
}

pub fn gemv_rpwr_mode(
    n: usize,
    k: usize,
    accum: bool,
    mode: SubgroupMode,
    rows_per_wg: usize,
    rows_per_warp: usize,
) -> String {
    gemv_impl(n, k, accum, mode, rows_per_wg, rows_per_warp, false)
}

/// [`gemv`] reading an **int8-resident** weight (the M3 layout): the same
/// granule, lane map, unroll, accumulators, butterfly and epilogue as the f16
/// kernel — `KG` is still `k/8`, a granule still covers 8 values — with the
/// weight word now a `vec2<u32>` of 8 int8 and a per-row f32 scale bound
/// alongside.  Per value the kernel computes `f16(q * scale[row])`: the two
/// half-products are formed in f32 and rounded to f16 by `pack2x16float`
/// (round-to-nearest-even, the same rounding the host dequant `half::from_f32`
/// applies), then fed to the f32 fma.  So this kernel consumes exactly the
/// values M2 materialized, in exactly the f16 kernel's order — the decode
/// arithmetic is unchanged and only the bytes on the bus are half.
///
/// Layout: row `r`'s bytes start at word `r * KG` of `Wt` (each byte of a word
/// is value `4*word + j`, LSB first — the file's own sequence, no repacking);
/// `cols` being a multiple of 4 keeps rows word-aligned, and `k % 8 == 0` keeps
/// the granule grid exact.
pub fn gemv_w8(n: usize, k: usize, accum: bool, subgroup: bool, rows_per_wg: usize) -> String {
    gemv_w8_mode(
        n,
        k,
        accum,
        if subgroup {
            SubgroupMode::Warp32
        } else {
            SubgroupMode::None
        },
        rows_per_wg,
    )
}

pub fn gemv_w8_mode(
    n: usize,
    k: usize,
    accum: bool,
    mode: SubgroupMode,
    rows_per_wg: usize,
) -> String {
    gemv_impl(n, k, accum, mode, rows_per_wg, 1, true)
}

/// [`gemv_w8`] with `rows_per_warp` rows per warp — [`gemv_rpwr`]'s knob on the
/// int8 kernel, bit-identical for the same reason the f16 one is.
pub fn gemv_w8_rpwr(
    n: usize,
    k: usize,
    accum: bool,
    subgroup: bool,
    rows_per_wg: usize,
    rows_per_warp: usize,
) -> String {
    gemv_w8_rpwr_mode(
        n,
        k,
        accum,
        if subgroup {
            SubgroupMode::Warp32
        } else {
            SubgroupMode::None
        },
        rows_per_wg,
        rows_per_warp,
    )
}

pub fn gemv_w8_rpwr_mode(
    n: usize,
    k: usize,
    accum: bool,
    mode: SubgroupMode,
    rows_per_wg: usize,
    rows_per_warp: usize,
) -> String {
    gemv_impl(n, k, accum, mode, rows_per_wg, rows_per_warp, true)
}

/// The shared GEMV tile body: the eight prefetch loads of one iteration come
/// first, then per granule the weight unpack, the activation unpack and the
/// four fma lines — the same lines in both modes, so value `j` of a granule
/// feeds accumulator `j/2` identically whichever form the weights came in.
/// `xsrc` names the activation array (`X`, global, in `gemv`; the staged `xs`
/// workgroup array in `gemv_norm`); `wbase`, `i`, `s` and `q8` are the caller's.
fn gemv_tile_body(w8: bool, xsrc: &str) -> String {
    let mut body = String::new();
    for t in 0..4 {
        let off = ["", " + 32u", " + 64u", " + 96u"][t];
        body.push_str(&format!(
            "        let wv{t} = Wt[wbase + i{off}];\n        let xv{t} = {xsrc}[i{off}];\n"
        ));
    }
    for t in 0..4 {
        for wi in 0..4 {
            if w8 {
                // `wv.{comp}` holds values 4*wi..4*wi+3; the pair for `w{t}{wi}`
                // is the first two of them, each `f16(q * s)` via pack2x16float
                // (round-to-nearest-even — the host dequant's own rounding).
                let (comp, sha, shb) =
                    [("x", 24u32, 16u32), ("x", 8, 0), ("y", 24, 16), ("y", 8, 0)][wi];
                body.push_str(&format!(
                    "        let w{t}{wi} = unpack2x16float(pack2x16float(vec2<f32>(q8(wv{t}.{comp}, {sha}u) * s, q8(wv{t}.{comp}, {shb}u) * s)));\n"
                ));
            } else {
                let comp = ["x", "y", "z", "w"][wi];
                body.push_str(&format!(
                    "        let w{t}{wi} = unpack2x16float(wv{t}.{comp});\n"
                ));
            }
            let xcomp = ["x", "y", "z", "w"][wi];
            body.push_str(&format!(
                "        let x{t}{wi} = unpack2x16float(xv{t}.{xcomp});\n"
            ));
        }
        for a in 0..4 {
            body.push_str(&format!(
                "        a{a} = fma(w{t}{a}.x, x{t}{a}.x, fma(w{t}{a}.y, x{t}{a}.y, a{a}));\n"
            ));
        }
    }
    body.push_str("        i = i + 128u;\n");
    body
}

fn gemv_impl(
    n: usize,
    k: usize,
    accum: bool,
    mode: SubgroupMode,
    rows_per_wg: usize,
    rows_per_warp: usize,
    w8: bool,
) -> String {
    assert_eq!(n % 8, 0, "gemv: rows must be a multiple of 8");
    let kg = k / 8;
    assert_eq!(kg % 32, 0, "gemv: k/8 must be a multiple of 32");
    assert_eq!(k % 8, 0, "gemv: k must be a multiple of 8");
    let tiles = kg / 32;
    assert_eq!(
        tiles % 4,
        0,
        "gemv: k/256 must be a multiple of 4 (unrolled x4)"
    );
    // One warp per row is what fixes the reduction tree, so `rows_per_wg` only
    // chooses how many rows share a workgroup -- and therefore how many
    // workgroups the dispatch has.  Every warp still walks its own row with the
    // same lane->word mapping and the same xor butterfly; only the
    // workgroup-level bookkeeping moves.  Row `r` is written by the warp that
    // owns it either way, so the arithmetic (and the bits) are unchanged.
    assert!(
        rows_per_wg.is_power_of_two() && (2..=32).contains(&rows_per_wg),
        "gemv: rows_per_wg must be 2/4/8/16/32"
    );
    assert_eq!(
        n % rows_per_wg,
        0,
        "gemv: rows must be a multiple of rows_per_wg"
    );
    assert!(
        rows_per_warp.is_power_of_two() && (1..=4).contains(&rows_per_warp),
        "gemv: rows_per_warp must be 1/2/4"
    );
    assert_eq!(
        n % rows_per_warp,
        0,
        "gemv: rows must be a multiple of rows_per_warp"
    );
    let rows_per_wg_tot = rows_per_wg * rows_per_warp;
    let threads = rows_per_wg * 32;
    let accum_lit = if accum { 1u32 } else { 0u32 };
    let subgroup_lit = u32::from(mode != SubgroupMode::None);
    let bfly = bfly_fragments(mode, threads, "THREADS");
    let subgroup_body = bfly.body;
    let bfly_scratch = bfly.scratch;
    let builtins = bfly.signature;
    let fn_tail = bfly.fn_tail;
    let bfly_args = bfly.args;
    // The two weight forms share every structural constant — `KG` is `k/8`
    // granules of 8 values in both, so the lane map, the unroll, the
    // accumulators and the reduction tree are one shader; only the weight word
    // (8 f16 in a vec4, 8 int8 in a vec2), the per-value unpack and the scale
    // binding differ.
    let (wty, sbind, q8fn, sload) = if w8 {
        (
            "vec2<u32>",
            "@group(0) @binding(3) var<storage, read>       S:  array<f32>;\n",
            "\n/// Byte `j` of `w` as a signed f32: shift it to the top of the word,\n/// bitcast, and let the arithmetic shift right do the sign extension.\nfn q8(w: u32, sh: u32) -> f32 {\n    return f32((bitcast<i32>(w << sh)) >> 24u);\n}\n",
            "let s = S[row];\n    ",
        )
    } else {
        ("vec4<u32>", "", "", "    ")
    };
    let body = gemv_tile_body(w8, "X");
    format!(
        "@group(0) @binding(0) var<storage, read>       Wt: array<{wty}>;
@group(0) @binding(1) var<storage, read>       X:  array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> Y:  array<u32>;
{sbind}const KG: u32 = {kg}u;
const TILES: u32 = {tiles}u;
const ACCUM: u32 = {accum_lit}u;
const SUBGROUP: u32 = {subgroup_lit}u;
const THREADS: u32 = {threads}u;
const RPW: u32 = {rpw}u;
const RPWR: u32 = {rpwr}u;
const ROWSW: u32 = {rowsw}u;
{q8fn}{bfly_scratch}var<workgroup> rows_out: array<f32, {rowsn}>;

/// 5-round xor butterfly over the 32 lanes of one warp -- the exact tree
/// `__shfl_xor_sync(acc, [16,8,4,2,1])` produces.
///
/// `SUBGROUP=1` emits `subgroupShuffleXor` (gated on `Features::SUBGROUP`,
/// measured available on this Pascal/Vulkan stack); `SUBGROUP=0` runs the same
/// tree through shared memory with 5 `workgroupBarrier()`s.
///
/// **Both forms are bit-identical**: the xor order is unchanged and every step
/// is one f32 add of the same two operands, so the reduction tree -- and every
/// bit of the result -- is preserved.  A/B measured 1.17-1.18x on both the
/// 512-row and the 18992-row shape with 0 differing outputs
/// (`cargo run --release --bin subgroup_bfly_bench`).
fn bfly(v: f32, lid: u32, lane: u32{fn_tail}) -> f32 {{
{subgroup_body}
}}

@compute @workgroup_size({threads})
fn gemv(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>{builtins}) {{
    let lane = lid.x & 31u;
    let warp = lid.x >> 5u;
    // `RPWR` rows per warp, walked in turn.  Each row keeps its own
    // accumulators, its own `bfly` and its own slot, so nothing about a row's
    // arithmetic depends on how many other rows the warp also walked.
    let rbase = (wgid.x * RPW + warp) * RPWR;
    for (var rr = 0u; rr < RPWR; rr = rr + 1u) {{
    let row = rbase + rr;

    let wbase = row * KG;
    {sload}var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
    // Tiles unrolled x4 with prefetched loads: each warp keeps 4 (Wt, X) pairs
    // in flight to hide DRAM latency (rows == warps leaves the machine
    // underfilled on the small projections).  The fma chain still consumes
    // tiles in ascending order, so the reduction bits are unchanged.
    var i = lane;
    for (var g = 0u; g < TILES; g = g + 4u) {{
{body}    }}
    let acc = (a0 + a1) + (a2 + a3);
    let r = bfly(acc, lid.x, lane{bfly_args});
    if (lane == 0u) {{ rows_out[warp * RPWR + rr] = r; }}
    }}
    workgroupBarrier();
    if (lid.x == 0u) {{
        let wordbase = (wgid.x * RPW * RPWR) >> 1u;
        for (var w = 0u; w < ROWSW; w = w + 1u) {{
            var va = rows_out[2u * w];
            var vb = rows_out[2u * w + 1u];
            if (ACCUM == 1u) {{
                let old = unpack2x16float(Y[wordbase + w]);
                va = va + old.x;
                vb = vb + old.y;
            }}
            Y[wordbase + w] = pack2x16float(vec2<f32>(va, vb));
        }}
    }}
}}
",
        kg = kg,
        tiles = tiles,
        accum_lit = accum_lit,
        subgroup_lit = subgroup_lit,
        subgroup_body = subgroup_body,
        bfly_scratch = bfly_scratch,
        builtins = builtins,
        fn_tail = fn_tail,
        bfly_args = bfly_args,
        rpw = rows_per_wg,
        rowsw = rows_per_wg_tot / 2,
        rowsn = rows_per_wg_tot,
        rpwr = rows_per_warp,
        threads = threads,
        wty = wty,
        sbind = sbind,
        q8fn = q8fn,
        sload = sload,
        body = body,
    )
}

/// Batched GEMV: one weight read, `batch` activation vectors.
///
/// [`gemv`] streams `n * k * 2` bytes of weights per dispatch and produces one
/// token's projection, so at b=1 the step is a single weight stream and the
/// hardware has nothing else to overlap.  This kernel keeps [`gemv`]'s granule,
/// lane map, accumulator order and xor butterfly -- value `j` of a granule is
/// still consumed by the same `fma` chain in the same order -- but each warp
/// holds `batch` accumulator quads and multiplies every weight word against
/// every vector's word before moving to the next one.  Weight traffic per
/// *step* is unchanged while `batch` tokens come out of it, which is the whole
/// arithmetic of decode batching.
///
/// Layout follows [`gemv`]: `Wt` is `[n, k/8]` words of four f16 with row `r` at
/// word `r*KG`; `X` is `batch` rows of those same `KG` words; `Y` is `batch`
/// rows of `n` f16, row `b` starting at word `b * (n/2)`.  `n` must be a
/// multiple of `RPW*2 = 16` so each workgroup writes whole f16 pairs, and `k` a
/// multiple of 256 so the x4 unroll closes evenly.
pub fn gemv_batch(n: usize, k: usize, batch: usize) -> String {
    const RPW: usize = 8;
    const THREADS: usize = RPW * 32;
    let kg = k / 8;
    assert_eq!(k % 256, 0, "gemv_batch: k must be a multiple of 256");
    assert_eq!(kg % 32, 0, "gemv_batch: k/8 must be a multiple of 32");
    let tiles = kg / 32;
    assert_eq!(tiles % 4, 0, "gemv_batch: k/256 must be a multiple of 4");
    assert_eq!(
        n % (RPW * 2),
        0,
        "gemv_batch: rows must be a multiple of 16"
    );
    assert!(
        (1..=16).contains(&batch),
        "gemv_batch: batch must be 1..=16"
    );
    let rowsw = RPW / 2;
    let rowsn = batch * THREADS;

    let mut accs = String::new();
    let mut body = String::new();
    for b in 0..batch {
        for c in 0..4 {
            accs.push_str(&format!("    var a{b}_{c} = 0.0;\n"));
        }
    }
    // The weight words of one tile first: they are what every batch shares.
    for t in 0..4 {
        let off = ["", " + 32u", " + 64u", " + 96u"][t];
        body.push_str(&format!("        let wv{t} = Wt[wbase + i{off}];\n"));
    }
    for t in 0..4 {
        for (wi, comp) in ["x", "y", "z", "w"].iter().enumerate() {
            body.push_str(&format!(
                "        let w{t}{wi} = unpack2x16float(wv{t}.{comp});\n"
            ));
        }
    }
    for b in 0..batch {
        let xb = b * kg;
        for t in 0..4 {
            let off = ["", " + 32u", " + 64u", " + 96u"][t];
            body.push_str(&format!("        let x{b}_{t} = X[{xb}u + i{off}];\n"));
        }
        for t in 0..4 {
            for (wi, comp) in ["x", "y", "z", "w"].iter().enumerate() {
                body.push_str(&format!(
                    "        let x{b}{t}{wi} = unpack2x16float(x{b}_{t}.{comp});\n"
                ));
            }
        }
        for t in 0..4 {
            for c in 0..4 {
                body.push_str(&format!(
                    "        a{b}_{c} = fma(w{t}{c}.x, x{b}{t}{c}.x, fma(w{t}{c}.y, x{b}{t}{c}.y, a{b}_{c}));\n"
                ));
            }
        }
    }
    body.push_str("        i = i + 128u;\n");

    let mut reduce = String::new();
    for b in 0..batch {
        reduce.push_str(&format!(
            "    let acc{b} = (a{b}_0 + a{b}_1) + (a{b}_2 + a{b}_3);\n    let r{b} = bfly(acc{b}, lid.x, lane);\n    if (lane == 0u) {{ rows_out[{o}u + warp] = r{b}; }}\n",
            o = b * THREADS
        ));
    }
    let mut epilogue = String::new();
    for b in 0..batch {
        let base = b * THREADS;
        epilogue.push_str(&format!(
            "        let wb{b} = {b}u * (N >> 1u) + (wgid.x * {rowsw}u);\n"
        ));
        for w in 0..rowsw {
            epilogue.push_str(&format!(
                "        Y[wb{b} + {w}u] = pack2x16float(vec2<f32>(rows_out[{a}u], rows_out[{c}u]));\n",
                a = base + 2 * w,
                c = base + 2 * w + 1
            ));
        }
    }

    format!(
        "@group(0) @binding(0) var<storage, read>       Wt: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       X:  array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> Y:  array<u32>;

const KG: u32 = {kg}u;
const TILES: u32 = {tiles}u;
const THREADS: u32 = {threads}u;
const RPW: u32 = {rpw}u;
const N: u32 = {n}u;

var<workgroup> bt0: array<f32, THREADS>;
var<workgroup> bt1: array<f32, THREADS>;
var<workgroup> rows_out: array<f32, {rowsn}>;

/// The same 5-round xor butterfly [`gemv`] uses, through shared memory: the
/// reduction tree has to stay bit-identical for b=1 to be a regression-free
/// reference.
fn bfly(v: f32, lid: u32, lane: u32) -> f32 {{
    let wb = lid & 0xFFFFFFE0u;
    bt0[lid] = v;
    workgroupBarrier();
    var t = bt0[lid] + bt0[wb + (lane ^ 16u)];
    bt1[lid] = t;
    workgroupBarrier();
    t = bt1[lid] + bt1[wb + (lane ^ 8u)];
    bt0[lid] = t;
    workgroupBarrier();
    t = bt0[lid] + bt0[wb + (lane ^ 4u)];
    bt1[lid] = t;
    workgroupBarrier();
    t = bt1[lid] + bt1[wb + (lane ^ 2u)];
    bt0[lid] = t;
    workgroupBarrier();
    t = bt0[lid] + bt0[wb + (lane ^ 1u)];
    return t;
}}

@compute @workgroup_size({threads})
fn gemv_batch(@builtin(workgroup_id) wgid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {{
    let lane = lid.x & 31u;
    let warp = lid.x >> 5u;
    let row = wgid.x * RPW + warp;
    let wbase = row * KG;
{accs}    var i = lane;
    // x4 unroll with the four weight words prefetched: the fma chain still
    // consumes tiles in ascending order, so the reduction bits are unchanged.
    for (var g = 0u; g < TILES; g = g + 4u) {{
{body}    }}
{reduce}    workgroupBarrier();
    if (lid.x == 0u) {{
{epilogue}    }}
}}
",
        kg = kg,
        tiles = tiles,
        threads = THREADS,
        rpw = RPW,
        n = n,
        rowsn = rowsn,
        accs = accs,
        body = body,
        reduce = reduce,
        epilogue = epilogue,
    )
}

/// Batched GEMV over **separate per-slot buffers**: one weight read, `batch`
/// dot products, no change to any slot's buffer layout.
///
/// [`gemv_batch`] takes its activations as one `[batch, k]` buffer, which would
/// force every per-slot tensor to become a slice of a shared allocation.  This
/// form takes one binding per slot instead, so a batch can run on the buffers
/// the single-sequence path already allocates -- the per-slot kernels
/// (`extract`, `gqa`, `silu`, `argmax`, `rms_norm`) never learn about batching.
///
/// The arithmetic per slot is [`gemv`]'s: same granule walk, same four
/// accumulators, same `(a0+a1)+(a2+a3)`, same xor butterfly, same epilogue.  The
/// only thing that moves is where the weight word is loaded -- once per tile for
/// all slots instead of once per slot -- so a batched token is bit-identical to a
/// lone one.
///
/// Bindings: `0` = W, `1..=batch` = each slot's activation, `1+batch..` = each
/// slot's output.
pub fn gemv_batch_slots(n: usize, k: usize, batch: usize, accum: bool) -> String {
    gemv_batch_slots_mode(n, k, batch, accum, SubgroupMode::None)
}

pub fn gemv_batch_slots_mode(
    n: usize,
    k: usize,
    batch: usize,
    accum: bool,
    mode: SubgroupMode,
) -> String {
    const RPW: usize = 8;
    const THREADS: usize = RPW * 32;
    let kg = k / 8;
    assert_eq!(k % 256, 0, "gemv_batch_slots: k must be a multiple of 256");
    assert_eq!(kg % 32, 0, "gemv_batch_slots: k/8 must be a multiple of 32");
    let tiles = kg / 32;
    assert_eq!(
        tiles % 4,
        0,
        "gemv_batch_slots: k/256 must be a multiple of 4"
    );
    assert_eq!(
        n % (RPW * 2),
        0,
        "gemv_batch_slots: rows must be a multiple of 16"
    );
    assert!(
        (2..=32).contains(&batch),
        "gemv_batch_slots: batch must be 2..=32"
    );
    let rowsw = RPW / 2;
    let rowsn = batch * THREADS;
    let accum_lit = if accum { 1u32 } else { 0u32 };
    let bfly = bfly_fragments(mode, THREADS, "THREADS");
    let subgroup_body = bfly.body;
    let bfly_scratch = bfly.scratch;
    let builtins = bfly.signature;
    let fn_tail = bfly.fn_tail;
    let bfly_args = bfly.args;

    // One binding covers every slot: the slot picks *which buffer*, not an offset
    // into a shared one.  `wgid.z` indexes the binding array, so a slot reads
    // exactly the bytes its lone run reads and no kernel carries a stride.
    let decl = String::from(
        "\
@group(0) @binding(0) var<storage, read>       Wt: array<vec4<u32>>;\
@group(0) @binding(1) var<storage, read>       Xs: binding_array<Arr4, 32>;\
@group(0) @binding(2) var<storage, read_write> Ys: binding_array<Arr32, 32>;\
",
    );

    let mut accs = String::new();
    let mut body = String::new();
    for s in 0..batch {
        for c in 0..4 {
            accs.push_str(&format!("    var a{s}_{c} = 0.0;\n"));
        }
    }
    for t in 0..4 {
        let off = ["", " + 32u", " + 64u", " + 96u"][t];
        body.push_str(&format!("        let wv{t} = Wt[wbase + i{off}];\n"));
    }
    for t in 0..4 {
        for (wi, comp) in ["x", "y", "z", "w"].iter().enumerate() {
            body.push_str(&format!(
                "        let w{t}{wi} = unpack2x16float(wv{t}.{comp});\n"
            ));
        }
    }
    for s in 0..batch {
        for t in 0..4 {
            let off = ["", " + 32u", " + 64u", " + 96u"][t];
            body.push_str(&format!("        let x{s}_{t} = Xs[{s}].v[i{off}];\n"));
        }
        for t in 0..4 {
            for (wi, comp) in ["x", "y", "z", "w"].iter().enumerate() {
                body.push_str(&format!(
                    "        let x{s}{t}{wi} = unpack2x16float(x{s}_{t}.{comp});\n"
                ));
            }
        }
        for t in 0..4 {
            for c in 0..4 {
                body.push_str(&format!(
                    "        a{s}_{c} = fma(w{t}{c}.x, x{s}{t}{c}.x, fma(w{t}{c}.y, x{s}{t}{c}.y, a{s}_{c}));\n"
                ));
            }
        }
    }
    body.push_str("        i = i + 128u;\n");

    let mut reduce = String::new();
    for s in 0..batch {
        reduce.push_str(&format!(
            "    let acc{s} = (a{s}_0 + a{s}_1) + (a{s}_2 + a{s}_3);\n    let r{s} = bfly(acc{s}, lid.x, lane{args});\n    if (lane == 0u) {{ rows_out[{o}u + warp] = r{s}; }}\n",
            o = s * THREADS,
            args = bfly_args
        ));
    }
    let mut epilogue = String::new();
    for s in 0..batch {
        let base = s * THREADS;
        for w in 0..rowsw {
            epilogue.push_str(&format!(
                "        var va{s}_{w} = rows_out[{a}u];\n        var vb{s}_{w} = rows_out[{c}u];\n",
                a = base + 2 * w,
                c = base + 2 * w + 1
            ));
            epilogue.push_str(&format!(
                "        if (ACCUM == 1u) {{ let old = unpack2x16float(Ys[{s}].v[wb + {w}u]); va{s}_{w} = va{s}_{w} + old.x; vb{s}_{w} = vb{s}_{w} + old.y; }}\n"
            ));
            epilogue.push_str(&format!(
                "        Ys[{s}].v[wb + {w}u] = pack2x16float(vec2<f32>(va{s}_{w}, vb{s}_{w}));\n"
            ));
        }
    }

    format!(
        "enable wgpu_binding_array;
/// naga requires the binding-array element to be a struct; one struct per word
/// shape keeps the bodies the same `array<...>` the lone kernels use.
struct Arr4  {{ v: array<vec4<u32>> }};
struct Arr32 {{ v: array<u32> }};
{decl}
const KG: u32 = {kg}u;
const TILES: u32 = {tiles}u;
const THREADS: u32 = {threads}u;
const RPW: u32 = {rpw}u;
const ACCUM: u32 = {accum}u;

{bfly_scratch}var<workgroup> rows_out: array<f32, {rowsn}>;

/// [`gemv`]'s 5-round xor butterfly, through shared memory: the reduction tree
/// is what makes a batched slot's sum bit-identical to its lone sum.
fn bfly(v: f32, lid: u32, lane: u32{fn_tail}) -> f32 {{
{subgroup_body}
}}

@compute @workgroup_size({threads})
fn gemv_batch_slots(@builtin(workgroup_id) wgid: vec3<u32>,
                    @builtin(local_invocation_id) lid: vec3<u32>{builtins}) {{
    let lane = lid.x & 31u;
    let warp = lid.x >> 5u;
    let row = wgid.x * RPW + warp;
    let wbase = row * KG;
    let wb = (wgid.x * {rowsw}u);
{accs}    var i = lane;
    for (var g = 0u; g < TILES; g = g + 4u) {{
{body}    }}
{reduce}    workgroupBarrier();
    if (lid.x == 0u) {{
{epilogue}    }}
}}
",
        decl = decl,
        kg = kg,
        tiles = tiles,
        threads = THREADS,
        rpw = RPW,
        accum = accum_lit,
        rowsn = rowsn,
        rowsw = rowsw,
        subgroup_body = subgroup_body,
        bfly_scratch = bfly_scratch,
        builtins = builtins,
        fn_tail = fn_tail,
        accs = accs,
        body = body,
        reduce = reduce,
        epilogue = epilogue,
    )
}

/// [`gemv`] with the RMSNorm that feeds it folded into the workgroup prologue.
///
/// The norm is one 1-workgroup dispatch per site in the decode loop and each of
/// those costs ~30 µs of critical path (measured by ablation: dropping the two
/// per-layer norms saves 1.1 ms/step at a 180 s context), which is what this
/// removes.  Two things have to be exactly right or the token stream changes:
///
/// * **The reduction tree.**  `rms_norm` runs `bs` (= `block_for_reduction(hs)`,
///   1024 at both shipped sizes) threads, so each *virtual* thread's partial is
///   `sum over j = t, t+bs, …` of that word's `x²+y²`, followed by
///   `red[t] += red[t+s]` for `s = bs/2 … 1`.  Here a 256-thread workgroup
///   computes `vc = bs/256` virtual partials per thread and folds the first
///   `log2(vc)` rounds into local adds — the pairing and the add order are the
///   arithmetic's, not an approximation of it.
/// * **The value that is consumed.**  `rms_norm` writes the normalized row back
///   as f16, so the GEMV must read `pack2x16float(x * inv_rms * w)` — not an
///   f32 intermediate.  The staging loop below writes exactly that expression,
///   in the same order, into shared memory.
///
/// Writes `Y` like `gemv` (same epilogue, same reduction), so the decode
/// arithmetic is unchanged; only the 1-workgroup dispatches disappear.
pub fn gemv_norm(
    n: usize,
    k: usize,
    accum: bool,
    subgroup: bool,
    last: usize,
    bs: usize,
    eps: f32,
) -> String {
    gemv_norm_mode(
        n,
        k,
        accum,
        if subgroup {
            SubgroupMode::Warp32
        } else {
            SubgroupMode::None
        },
        last,
        bs,
        eps,
    )
}

pub fn gemv_norm_mode(
    n: usize,
    k: usize,
    accum: bool,
    mode: SubgroupMode,
    last: usize,
    bs: usize,
    eps: f32,
) -> String {
    gemv_norm_impl(n, k, accum, mode, last, bs, eps, false)
}

/// [`gemv_norm`] on the int8-resident weights — [`gemv_w8`]'s weight form with
/// [`gemv_norm`]'s folded RMSNorm prologue, unchanged in every other respect.
pub fn gemv_norm_w8(
    n: usize,
    k: usize,
    accum: bool,
    subgroup: bool,
    last: usize,
    bs: usize,
    eps: f32,
) -> String {
    gemv_norm_w8_mode(
        n,
        k,
        accum,
        if subgroup {
            SubgroupMode::Warp32
        } else {
            SubgroupMode::None
        },
        last,
        bs,
        eps,
    )
}

pub fn gemv_norm_w8_mode(
    n: usize,
    k: usize,
    accum: bool,
    mode: SubgroupMode,
    last: usize,
    bs: usize,
    eps: f32,
) -> String {
    gemv_norm_impl(n, k, accum, mode, last, bs, eps, true)
}

fn gemv_norm_impl(
    n: usize,
    k: usize,
    accum: bool,
    mode: SubgroupMode,
    last: usize,
    bs: usize,
    eps: f32,
    w8: bool,
) -> String {
    assert_eq!(n % 8, 0, "gemv_norm: rows must be a multiple of 8");
    let kg = k / 8;
    assert_eq!(kg % 32, 0, "gemv_norm: k/8 must be a multiple of 32");
    assert_eq!(k % 8, 0, "gemv_norm: k must be a multiple of 8");
    let tiles = kg / 32;
    assert_eq!(tiles % 4, 0, "gemv_norm: k/256 must be a multiple of 4");
    let last2 = last / 2;
    assert_eq!(
        last2,
        kg * 4,
        "gemv_norm: activation row {last2} words != k/2 {kg_expected}",
        kg_expected = k / 2
    );
    let vc = bs / 256;
    assert!(
        [1usize, 2, 4].contains(&vc) && bs == vc * 256,
        "gemv_norm: block size {bs} is not 256/512/1024"
    );
    let accum_lit = u32::from(accum);
    let subgroup_lit = u32::from(mode != SubgroupMode::None);
    let bfly = bfly_fragments(mode, 256, "256");
    let subgroup_body = bfly.body;
    let bfly_scratch = bfly.scratch;
    let builtins = bfly.signature;
    let fn_tail = bfly.fn_tail;
    let bfly_args = bfly.args;
    let folded = match vc {
        1 => "l0".to_string(),
        2 => "l0 + l1".to_string(),
        _ => "(l0 + l2) + (l1 + l3)".to_string(),
    };
    let mut locals = String::new();
    for i in 0..vc {
        if i == 0 {
            locals.push_str("    var l0 = 0.0;\n");
        } else {
            locals.push_str(&format!("    var l{i} = 0.0;\n"));
        }
    }
    let mut sums = String::new();
    for i in 0..vc {
        let off = i * 256;
        sums.push_str(&format!(
            "    for (var j = lid.x + {off}u; j < LAST2; j = j + BS) {{\n\
             \x20       let v = unpack2x16float(Xr[j]);\n\
             \x20       l{i} = l{i} + (v.x * v.x + v.y * v.y);\n\
             \x20   }}\n"
        ));
    }
    let (wty, sbind, q8fn, sload) = if w8 {
        (
            "vec2<u32>",
            "@group(0) @binding(4) var<storage, read>       S:   array<f32>;\n",
            "\n/// Byte `j` of `w` as a signed f32: shift it to the top of the word,\n/// bitcast, and let the arithmetic shift right do the sign extension.\nfn q8(w: u32, sh: u32) -> f32 {\n    return f32((bitcast<i32>(w << sh)) >> 24u);\n}\n",
            "let s = S[row];\n    ",
        )
    } else {
        ("vec4<u32>", "", "", "    ")
    };
    let body = gemv_tile_body(w8, "xs");
    format!(
        "@group(0) @binding(0) var<storage, read>       Wt:  array<{wty}>;
@group(0) @binding(1) var<storage, read>       Xr:  array<u32>;
@group(0) @binding(2) var<storage, read_write> Y:   array<u32>;
@group(0) @binding(3) var<storage, read>       NW:  array<u32>;
{sbind}
const KG: u32 = {kg}u;
const TILES: u32 = {tiles}u;
const ACCUM: u32 = {accum_lit}u;
const SUBGROUP: u32 = {subgroup_lit}u;
const LAST: u32 = {last}u;
const LAST2: u32 = {last2}u;
const BS: u32 = {bs}u;
const EPS: f32 = {eps:?}f;

{q8fn}{bfly_scratch}var<workgroup> rows_out: array<f32, 8>;
var<workgroup> red: array<f32, 256>;
/// The normalized row, f16-packed exactly as `rms_norm` would have written it.
var<workgroup> xs: array<vec4<u32>, {kg}u>;

fn bfly(v: f32, lid: u32, lane: u32{fn_tail}) -> f32 {{
{subgroup_body}
}}

/// `rms_norm`'s output word `j`: the same two multiplies, in the same order,
/// so the f16 rounding matches the buffer it used to go through.
fn norm_word(j: u32, inv_rms: f32) -> u32 {{
    let xv = unpack2x16float(Xr[j]);
    let wv = unpack2x16float(NW[j]);
    return pack2x16float(vec2<f32>(xv.x * inv_rms * wv.x, xv.y * inv_rms * wv.y));
}}

@compute @workgroup_size(256)
fn gemv(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>{builtins}) {{
    let lane = lid.x & 31u;
    let warp = lid.x >> 5u;
    let row = wgid.x * 8u + warp;

    // ── prologue: the RMSNorm this GEMV used to wait for ──
{locals}{sums}    red[lid.x] = {folded};
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{ red[lid.x] = red[lid.x] + red[lid.x + s]; }}
        workgroupBarrier();
    }}
    let inv_rms = inverseSqrt(red[0] / f32(LAST) + EPS);
    workgroupBarrier();
    for (var w4 = lid.x; w4 < KG; w4 = w4 + 256u) {{
        let b0 = norm_word(4u * w4, inv_rms);
        let b1 = norm_word(4u * w4 + 1u, inv_rms);
        let b2 = norm_word(4u * w4 + 2u, inv_rms);
        let b3 = norm_word(4u * w4 + 3u, inv_rms);
        xs[w4] = vec4<u32>(b0, b1, b2, b3);
    }}
    workgroupBarrier();

    let wbase = row * KG;
    {sload}var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
    // Body identical to `gemv`; only the activation comes from `xs`.
    var i = lane;
    for (var g = 0u; g < TILES; g = g + 4u) {{
{body}    }}
    let acc = (a0 + a1) + (a2 + a3);
    let r = bfly(acc, lid.x, lane{bfly_args});
    if (lane == 0u) {{ rows_out[warp] = r; }}
    workgroupBarrier();
    if (lid.x == 0u) {{
        let wordbase = (wgid.x * 8u) >> 1u;
        for (var w = 0u; w < 4u; w = w + 1u) {{
            var va = rows_out[2u * w];
            var vb = rows_out[2u * w + 1u];
            if (ACCUM == 1u) {{
                let old = unpack2x16float(Y[wordbase + w]);
                va = va + old.x;
                vb = vb + old.y;
            }}
            Y[wordbase + w] = pack2x16float(vec2<f32>(va, vb));
        }}
    }}
}}
",
        kg = kg,
        tiles = tiles,
        accum_lit = accum_lit,
        subgroup_lit = subgroup_lit,
        last = last,
        last2 = last2,
        bs = bs,
        eps = eps,
        locals = locals,
        sums = sums,
        folded = folded,
        subgroup_body = subgroup_body,
        bfly_scratch = bfly_scratch,
        builtins = builtins,
        fn_tail = fn_tail,
        bfly_args = bfly_args,
        wty = wty,
        sbind = sbind,
        q8fn = q8fn,
        sload = sload,
        body = body,
    )
}

/// [`gemv_norm`] over `batch` sequences: one weight read, `batch` norm+GEMVs.
///
/// Same prologue statements as [`gemv_norm`] -- the `vc = bs/256` folded partials,
/// the 256-wide tree, `norm_word`'s two multiplies and its f16 packing -- but run
/// once per slot into that slot's shared row, and then one GEMV loop that reads
/// each weight word once and multiplies it into every slot's staged row.  That
/// keeps a batched token bit-identical to a lone one, which the `rms_norm` +
/// `gemv` split does **not**: the standalone norm reduces with `bs` threads where
/// this one folds `vc` partials per thread, so the two f32 sums associate
/// differently and an occasional f16 rounding lands the other way.
///
/// `Xr`/`Y` are one binding per slot (the decoder keeps its per-slot buffers);
/// the norm weights are shared, and `EPS` is baked in exactly as it is for the
/// single-vector kernel.
pub fn gemv_norm_batch_slots(
    n: usize,
    k: usize,
    batch: usize,
    accum: bool,
    last: usize,
    bs: usize,
    eps: f32,
) -> String {
    gemv_norm_batch_slots_mode(n, k, batch, accum, last, bs, eps, SubgroupMode::None)
}

pub fn gemv_norm_batch_slots_mode(
    n: usize,
    k: usize,
    batch: usize,
    accum: bool,
    last: usize,
    bs: usize,
    eps: f32,
    mode: SubgroupMode,
) -> String {
    assert_eq!(
        n % 8,
        0,
        "gemv_norm_batch_slots: rows must be a multiple of 8"
    );
    let kg = k / 8;
    assert_eq!(
        k % 256,
        0,
        "gemv_norm_batch_slots: k must be a multiple of 256"
    );
    let tiles = kg / 32;
    assert_eq!(
        tiles % 4,
        0,
        "gemv_norm_batch_slots: k/256 must be a multiple of 4"
    );
    let last2 = last / 2;
    assert_eq!(
        last2,
        kg * 4,
        "gemv_norm_batch_slots: activation row != k/2 words"
    );
    assert!(
        (2..=32).contains(&batch),
        "gemv_norm_batch_slots: batch must be 2..=16"
    );
    let vc = bs / 256;
    assert_eq!(
        bs,
        vc * 256,
        "gemv_norm_batch_slots: block size must be 256/512/1024"
    );
    assert!(
        [1usize, 2, 4].contains(&vc),
        "gemv_norm_batch_slots: block size"
    );
    let accum_lit = u32::from(accum);
    let bfly = bfly_fragments(mode, 256, "256");
    let subgroup_body = bfly.body;
    let bfly_scratch = bfly.scratch;
    let builtins = bfly.signature;
    let fn_tail = bfly.fn_tail;
    let bfly_args = bfly.args;
    let folded = match vc {
        1 => "l0".to_string(),
        2 => "l0 + l1".to_string(),
        _ => "(l0 + l2) + (l1 + l3)".to_string(),
    };

    // Same shape as `gemv_batch_slots`: one binding holds every slot's row, so
    // the slot is the buffer the dispatch names and never a stride.
    let decl = String::from(
        "enable wgpu_binding_array;
struct Arr4  { v: array<vec4<u32>> };
struct Arr32 { v: array<u32> };
@group(0) @binding(0) var<storage, read>       Wt:  array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       Xs:  binding_array<Arr32, 32>;
@group(0) @binding(2) var<storage, read_write> Ys:  binding_array<Arr32, 32>;
@group(0) @binding(3) var<storage, read>       NW:  array<u32>;
",
    );

    let mut xs_decl = String::new();
    for s in 0..batch {
        xs_decl.push_str(&format!("var<workgroup> xs{s}: array<vec4<u32>, {kg}u>;\n"));
    }
    // One helper, not one per slot: `wgid.z` already picks the row.
    let normfns = "fn norm_word(z: u32, j: u32, inv_rms: f32) -> u32 {
    let xv = unpack2x16float(Xs[z].v[j]);
    let wv = unpack2x16float(NW[j]);
    return pack2x16float(vec2<f32>(xv.x * inv_rms * wv.x, xv.y * inv_rms * wv.y));
}
";

    // Prologue per slot: the shipped statements, against that slot's row.
    let mut pro = String::new();
    for s in 0..batch {
        pro.push_str(&format!("    // slot {s}\n    {{\n"));
        for i in 0..vc {
            pro.push_str(&format!("    var l{i} = 0.0;\n"));
        }
        for i in 0..vc {
            let off = i * 256;
            pro.push_str(&format!(
                "    for (var j = lid.x + {off}u; j < LAST2; j = j + BS) {{\n        let v = unpack2x16float(Xs[{s}].v[j]);\n        l{i} = l{i} + (v.x * v.x + v.y * v.y);\n    }}\n"
            ));
        }
        pro.push_str(&format!(
            "    red[lid.x] = {folded};
    workgroupBarrier();
    for (var t = 128u; t > 0u; t = t >> 1u) {{
        if (lid.x < t) {{ red[lid.x] = red[lid.x] + red[lid.x + t]; }}
        workgroupBarrier();
    }}
    let inv_rms{s} = inverseSqrt(red[0] / f32(LAST) + EPS);
    workgroupBarrier();
    for (var w4 = lid.x; w4 < KG; w4 = w4 + 256u) {{
        let b0 = norm_word({s}, 4u * w4, inv_rms{s});
        let b1 = norm_word({s}, 4u * w4 + 1u, inv_rms{s});
        let b2 = norm_word({s}, 4u * w4 + 2u, inv_rms{s});
        let b3 = norm_word({s}, 4u * w4 + 3u, inv_rms{s});
        xs{s}[w4] = vec4<u32>(b0, b1, b2, b3);
    }}
    workgroupBarrier();
    }}
"
        ));
    }

    let mut accs = String::new();
    let mut body = String::new();
    for s in 0..batch {
        for c in 0..4 {
            accs.push_str(&format!("    var a{s}_{c} = 0.0;\n"));
        }
    }
    for t in 0..4 {
        let off = ["", " + 32u", " + 64u", " + 96u"][t];
        body.push_str(&format!("        let wv{t} = Wt[wbase + i{off}];\n"));
    }
    for t in 0..4 {
        for (wi, comp) in ["x", "y", "z", "w"].iter().enumerate() {
            body.push_str(&format!(
                "        let w{t}{wi} = unpack2x16float(wv{t}.{comp});\n"
            ));
        }
    }
    for s in 0..batch {
        for t in 0..4 {
            let off = ["", " + 32u", " + 64u", " + 96u"][t];
            body.push_str(&format!("        let x{s}_{t} = xs{s}[i{off}];\n"));
        }
        for t in 0..4 {
            for (wi, comp) in ["x", "y", "z", "w"].iter().enumerate() {
                body.push_str(&format!(
                    "        let x{s}{t}{wi} = unpack2x16float(x{s}_{t}.{comp});\n"
                ));
            }
        }
        for t in 0..4 {
            for c in 0..4 {
                body.push_str(&format!(
                    "        a{s}_{c} = fma(w{t}{c}.x, x{s}{t}{c}.x, fma(w{t}{c}.y, x{s}{t}{c}.y, a{s}_{c}));\n"
                ));
            }
        }
    }
    body.push_str("        i = i + 128u;\n");

    let mut reduce = String::new();
    for s in 0..batch {
        reduce.push_str(&format!(
            "    let acc{s} = (a{s}_0 + a{s}_1) + (a{s}_2 + a{s}_3);\n    let r{s} = bfly(acc{s}, lid.x, lane{args});\n    if (lane == 0u) {{ rows_out[{o}u + warp] = r{s}; }}\n",
            o = s * 8,
            args = bfly_args
        ));
    }
    let mut epi = String::new();
    for s in 0..batch {
        for w in 0..4 {
            epi.push_str(&format!(
                "        var va{s}_{w} = rows_out[{a}u];\n        var vb{s}_{w} = rows_out[{c}u];\n",
                a = s * 8 + 2 * w,
                c = s * 8 + 2 * w + 1
            ));
            epi.push_str(&format!(
                "        if (ACCUM == 1u) {{ let old = unpack2x16float(Ys[{s}].v[wordbase + {w}u]); va{s}_{w} = va{s}_{w} + old.x; vb{s}_{w} = vb{s}_{w} + old.y; }}\n"
            ));
            epi.push_str(&format!(
                "        Ys[{s}].v[wordbase + {w}u] = pack2x16float(vec2<f32>(va{s}_{w}, vb{s}_{w}));\n"
            ));
        }
    }

    format!(
        "{decl}
const KG: u32 = {kg}u;
const TILES: u32 = {tiles}u;
const ACCUM: u32 = {accum}u;
const LAST: u32 = {last}u;
const LAST2: u32 = {last2}u;
const BS: u32 = {bs}u;
const EPS: f32 = {eps:?}f;

{bfly_scratch}var<workgroup> rows_out: array<f32, {rowsn}>;
var<workgroup> red: array<f32, 256>;
{xs_decl}
/// The same 5-round xor butterfly as `gemv_norm`, through shared memory.
fn bfly(v: f32, lid: u32, lane: u32{fn_tail}) -> f32 {{
{subgroup_body}
}}

/// `rms_norm`'s output word `j`, per slot: the same two multiplies in the same
/// order, so the f16 rounding matches the buffer it used to go through.
{normfns}
@compute @workgroup_size(256)
fn gemv_norm_batch_slots(@builtin(workgroup_id) wgid: vec3<u32>,
                         @builtin(local_invocation_id) lid: vec3<u32>{builtins}) {{
    let lane = lid.x & 31u;
    let warp = lid.x >> 5u;
    let row = wgid.x * 8u + warp;

{pro}
    let wbase = row * KG;
{accs}    var i = lane;
    for (var g = 0u; g < TILES; g = g + 4u) {{
{body}    }}
{reduce}    workgroupBarrier();
    if (lid.x == 0u) {{
        let wordbase = (wgid.x * 8u) >> 1u;
{epi}    }}
}}
",
        decl = decl,
        kg = kg,
        tiles = tiles,
        accum = accum_lit,
        last = last,
        last2 = last2,
        bs = bs,
        eps = eps,
        rowsn = batch * 8,
        subgroup_body = subgroup_body,
        bfly_scratch = bfly_scratch,
        builtins = builtins,
        fn_tail = fn_tail,
        xs_decl = xs_decl,
        normfns = normfns,
        pro = pro,
        accs = accs,
        body = body,
        reduce = reduce,
        epi = epi,
    )
}

/// `qkv_extract_qkv_norm_rotary_cache_f16` — one workgroup per head slot,
/// `grid = (1, nqh + nkvh)`.  Q heads land in `QOut`, K heads write the roped K
/// into `KCache` and copy V through verbatim.
pub fn qkv_extract(nqh: usize, nkvh: usize, d: usize) -> String {
    assert_eq!(d % 2, 0);
    let total_cols = (nqh + 2 * nkvh) * d;
    let q_dim = nqh * d;
    let kv_dim = nkvh * d;
    format!(
        "{HALF_AT}
struct Cfg {{ max_seq: u32, start: u32, pos_offset: u32, s: u32, eps: f32, _a: u32, _b: u32, _c: u32 }};

@group(0) @binding(0) var<storage, read>       Qkv:    array<u32>;
@group(0) @binding(1) var<storage, read>       QnW:    array<u32>;
@group(0) @binding(2) var<storage, read>       KnW:    array<u32>;
@group(0) @binding(3) var<storage, read>       Cos:    array<u32>;
@group(0) @binding(4) var<storage, read>       Sin:    array<u32>;
@group(0) @binding(5) var<storage, read_write> QOut:   array<u32>;
@group(0) @binding(6) var<storage, read_write> KCache: array<u32>;
@group(0) @binding(7) var<storage, read_write> VCache: array<u32>;
@group(0) @binding(8) var<uniform>             cfg:    Cfg;

const D: u32 = {d}u;
const D2: u32 = {d2}u;
const HD: u32 = {hd}u;
const NQH: u32 = {nqh}u;
const NKVH: u32 = {nkvh}u;
const TOT2: u32 = {tot2}u;
const QD2: u32 = {qd2}u;
const KVD2: u32 = {kvd2}u;
const BS: u32 = 128u;

var<workgroup> red: array<f32, 128>;

/// Sum of squares + tree reduction + inverse RMS (one element per thread,
/// `s` halving from `bs/2`).
fn head_inv_rms(base: u32, lid: u32) -> f32 {{
    var local = 0.0;
    for (var j = lid; j < D; j = j + BS) {{
        let e = half_at(unpack2x16float(Qkv[base + (j >> 1u)]), j);
        local = local + e * e;
    }}
    red[lid] = local;
    workgroupBarrier();
    for (var s = BS >> 1u; s > 0u; s = s >> 1u) {{
        if (lid < s) {{ red[lid] = red[lid] + red[lid + s]; }}
        workgroupBarrier();
    }}
    let r = inverseSqrt(red[0] / f32(D) + cfg.eps);
    workgroupBarrier();
    return r;
}}

/// One roped output element: norm over the head row, then rotate-half RoPE.
/// `use_q` picks the Q norm weights over the K norm weights.
fn rope_elem(use_q: bool, base: u32, csbase: u32, j: u32, inv_rms: f32) -> f32 {{
    let xj = half_at(unpack2x16float(Qkv[base + (j >> 1u)]), j);
    var wj = 0.0;
    if (use_q) {{
        wj = half_at(unpack2x16float(QnW[j >> 1u]), j);
    }} else {{
        wj = half_at(unpack2x16float(KnW[j >> 1u]), j);
    }}
    let x_val_j = xj * inv_rms * wj;
    let pj = select(j - HD, j + HD, j < HD);
    let xp = half_at(unpack2x16float(Qkv[base + (pj >> 1u)]), pj);
    var wp = 0.0;
    if (use_q) {{
        wp = half_at(unpack2x16float(QnW[pj >> 1u]), pj);
    }} else {{
        wp = half_at(unpack2x16float(KnW[pj >> 1u]), pj);
    }}
    let x_pair = xp * inv_rms * wp;
    let pair_val = select(x_pair, -x_pair, j < HD);
    let c = half_at(unpack2x16float(Cos[csbase + (j >> 1u)]), j);
    let si = half_at(unpack2x16float(Sin[csbase + (j >> 1u)]), j);
    return x_val_j * c + pair_val * si;
}}

@compute @workgroup_size(128)
fn qkv_extract(@builtin(workgroup_id) wgid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {{
    // wgid.x = position within the batch (0 for decode, 0..s for prefill)
    let is = wgid.x;
    let hy = wgid.y;
    // cos/sin are `array<u32>` (2 f16 per word): row `p` starts at word p*D2.
    let csbase = (cfg.pos_offset + is) * D2;

    if (hy < NQH) {{
        // Q head `hy` of position `is`: word is*TOT2 + hy*D2 (D2 words = D f16).
        let base = is * TOT2 + hy * D2;
        let inv_rms = head_inv_rms(base, lid.x);
        for (var w = lid.x; w < D2; w = w + BS) {{
            let j0 = w * 2u;
            QOut[(hy * cfg.s + is) * D2 + w] = pack2x16float(vec2<f32>(
                rope_elem(true, base, csbase, j0, inv_rms),
                rope_elem(true, base, csbase, j0 + 1u, inv_rms)));
        }}
    }} else {{
        let ih = hy - NQH;
        if (ih < NKVH) {{
            let kbase = is * TOT2 + QD2 + ih * D2;
            let vbase = is * TOT2 + QD2 + KVD2 + ih * D2;
            let cache = (ih * cfg.max_seq + cfg.start + is) * D2;
            let inv_rms = head_inv_rms(kbase, lid.x);
            for (var w = lid.x; w < D2; w = w + BS) {{
                let j0 = w * 2u;
                KCache[cache + w] = pack2x16float(vec2<f32>(
                    rope_elem(false, kbase, csbase, j0, inv_rms),
                    rope_elem(false, kbase, csbase, j0 + 1u, inv_rms)));
                VCache[cache + w] = Qkv[vbase + w];
            }}
        }}
    }}
}}
",
        d2 = d / 2,
        hd = d / 2,
        tot2 = total_cols / 2,
        qd2 = q_dim / 2,
        kvd2 = kv_dim / 2,
    )
}
/// [`qkv_extract`] over `batch` sequences in one dispatch, `wgid.z` = the slot.
///
/// The pooled buffers carry a slot stride for the qkv read, the q_out write and
/// the two cache writes; every other expression -- the head RMSNorm, the
/// rotate-half RoPE, the f16 packing -- is the shipped kernel's, so each slot's
/// KV rows and q_out bytes are the ones a lone run would have written.
pub fn qkv_extract_slots(nqh: usize, nkvh: usize, d: usize) -> String {
    assert_eq!(d % 2, 0);
    let total_cols = (nqh + 2 * nkvh) * d;
    let q_dim = nqh * d;
    let kv_dim = nkvh * d;
    format!(
        "enable wgpu_binding_array;
/// naga requires the binding-array element to be a struct; one struct per word
/// shape keeps the bodies the same `array<...>` the lone kernels use.
struct Arr4  {{ v: array<vec4<u32>> }};
struct Arr32 {{ v: array<u32> }};
{HALF_AT}
struct Cfg {{ max_seq: u32, start: u32, pos_offset: u32, s: u32, eps: f32, _a: u32, _b: u32, _c: u32 }};

@group(0) @binding(0) var<storage, read>       Qkv:    binding_array<Arr32, 32>;
@group(0) @binding(1) var<storage, read>       QnW:    array<u32>;
@group(0) @binding(2) var<storage, read>       KnW:    array<u32>;
@group(0) @binding(3) var<storage, read>       Cos:    array<u32>;
@group(0) @binding(4) var<storage, read>       Sin:    array<u32>;
@group(0) @binding(5) var<storage, read_write> QOut:   binding_array<Arr32, 32>;
@group(0) @binding(6) var<storage, read_write> KCache: binding_array<Arr32, 32>;
@group(0) @binding(7) var<storage, read_write> VCache: binding_array<Arr32, 32>;
/// Group 1: wgpu refuses a bind group that mixes a binding array with a uniform.
@group(1) @binding(0) var<uniform>             cfg:    Cfg;
/// Each slot's own write position: the segments in a batch have different
/// lengths, so slot 0's `cfg.start` is not it.
@group(0) @binding(9) var<storage, read>       POS:    array<u32>;

const D: u32 = {d}u;
const D2: u32 = {d2}u;
const HD: u32 = {hd}u;
const NQH: u32 = {nqh}u;
const NKVH: u32 = {nkvh}u;
const TOT2: u32 = {tot2}u;
const QD2: u32 = {qd2}u;
const KVD2: u32 = {kvd2}u;
const BS: u32 = 128u;

var<workgroup> red: array<f32, 128>;

/// Sum of squares + tree reduction + inverse RMS (one element per thread,
/// `s` halving from `bs/2`).
fn head_inv_rms(z: u32, base: u32, lid: u32) -> f32 {{
    var local = 0.0;
    for (var j = lid; j < D; j = j + BS) {{
        let e = half_at(unpack2x16float(Qkv[z].v[base + (j >> 1u)]), j);
        local = local + e * e;
    }}
    red[lid] = local;
    workgroupBarrier();
    for (var s = BS >> 1u; s > 0u; s = s >> 1u) {{
        if (lid < s) {{ red[lid] = red[lid] + red[lid + s]; }}
        workgroupBarrier();
    }}
    let r = inverseSqrt(red[0] / f32(D) + cfg.eps);
    workgroupBarrier();
    return r;
}}

/// One roped output element: norm over the head row, then rotate-half RoPE.
/// `use_q` picks the Q norm weights over the K norm weights.
fn rope_elem(z: u32, use_q: bool, base: u32, csbase: u32, j: u32, inv_rms: f32) -> f32 {{
    let xj = half_at(unpack2x16float(Qkv[z].v[base + (j >> 1u)]), j);
    var wj = 0.0;
    if (use_q) {{
        wj = half_at(unpack2x16float(QnW[j >> 1u]), j);
    }} else {{
        wj = half_at(unpack2x16float(KnW[j >> 1u]), j);
    }}
    let x_val_j = xj * inv_rms * wj;
    let pj = select(j - HD, j + HD, j < HD);
    let xp = half_at(unpack2x16float(Qkv[z].v[base + (pj >> 1u)]), pj);
    var wp = 0.0;
    if (use_q) {{
        wp = half_at(unpack2x16float(QnW[pj >> 1u]), pj);
    }} else {{
        wp = half_at(unpack2x16float(KnW[pj >> 1u]), pj);
    }}
    let x_pair = xp * inv_rms * wp;
    let pair_val = select(x_pair, -x_pair, j < HD);
    let c = half_at(unpack2x16float(Cos[csbase + (j >> 1u)]), j);
    let si = half_at(unpack2x16float(Sin[csbase + (j >> 1u)]), j);
    return x_val_j * c + pair_val * si;
}}

@compute @workgroup_size(128)
fn qkv_extract(@builtin(workgroup_id) wgid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {{
    // wgid.x = position within the batch (0 for decode, 0..s for prefill)
    let is = wgid.x;
    let slot = wgid.z;
    let hy = wgid.y;
    // cos/sin are `array<u32>` (2 f16 per word): row `p` starts at word p*D2.
    let csbase = (POS[slot] + is) * D2;

    if (hy < NQH) {{
        // Q head `hy` of position `is`: word is*TOT2 + hy*D2 (D2 words = D f16).
        let base = is * TOT2 + hy * D2;
        let inv_rms = head_inv_rms(slot, base, lid.x);
        for (var w = lid.x; w < D2; w = w + BS) {{
            let j0 = w * 2u;
            QOut[slot].v[(hy * cfg.s + is) * D2 + w] = pack2x16float(vec2<f32>(
                rope_elem(slot, true, base, csbase, j0, inv_rms),
                rope_elem(slot, true, base, csbase, j0 + 1u, inv_rms)));
        }}
    }} else {{
        let ih = hy - NQH;
        if (ih < NKVH) {{
            let kbase = is * TOT2 + QD2 + ih * D2;
            let vbase = is * TOT2 + QD2 + KVD2 + ih * D2;
            let cache = (ih * cfg.max_seq + POS[slot] + is) * D2;
            let inv_rms = head_inv_rms(slot, kbase, lid.x);
            for (var w = lid.x; w < D2; w = w + BS) {{
                let j0 = w * 2u;
                KCache[slot].v[cache + w] = pack2x16float(vec2<f32>(
                    rope_elem(slot, false, kbase, csbase, j0, inv_rms),
                    rope_elem(slot, false, kbase, csbase, j0 + 1u, inv_rms)));
                VCache[slot].v[cache + w] = Qkv[slot].v[vbase + w];
            }}
        }}
    }}
}}
",
        d2 = d / 2,
        hd = d / 2,
        tot2 = total_cols / 2,
        qd2 = q_dim / 2,
        kvd2 = kv_dim / 2,
    )
}

/// `fused_gqa_decode_f16` — single-block-per-(b,q_head) flash-style attention.
/// `bs` follows the adaptive choice (256 for `cur_len <= 512`, else 512); a
/// 1024 case exists that only the split path ever uses.
const EXP_BT: &str = "
fn lead32(v0: u32) -> i32 {
    var n = 0i;
    var v = v0;
    if ((v & 0xFFFF0000u) != 0u) { n = n + 16; v = v >> 16u; }
    if ((v & 0xFF00u) != 0u)     { n = n + 8;  v = v >> 8u;  }
    if ((v & 0xF0u) != 0u)       { n = n + 4;  v = v >> 4u;  }
    if ((v & 0xCu) != 0u)        { n = n + 2;  v = v >> 2u;  }
    if ((v & 0x2u) != 0u)        { n = n + 1; }
    return n;
}
fn shr64(hi0: u32, lo0: u32, t0: i32) -> vec3<u32> {
    if (t0 <= 0) { return vec3<u32>(hi0, lo0, 0u); }
    if (t0 >= 64) { return vec3<u32>(0u, 0u, select(0u, 1u, (hi0 | lo0) != 0u)); }
    if (t0 >= 32) {
        let s = u32(t0) - 32u;
        let lost = select(0u, 1u, lo0 != 0u) | select(0u, 1u, (hi0 & ((1u << s) - 1u)) != 0u);
        return vec3<u32>(0u, hi0 >> s, lost);
    }
    let s = u32(t0);
    let lost = select(0u, 1u, (lo0 & ((1u << s) - 1u)) != 0u);
    return vec3<u32>(hi0 >> s, (lo0 >> s) | (hi0 << (32u - s)), lost);
}
fn shl64(hi0: u32, lo0: u32, t0: i32) -> vec2<u32> {
    if (t0 <= 0) { return vec2<u32>(hi0, lo0); }
    if (t0 >= 32) {
        let s = u32(t0) - 32u;
        return vec2<u32>(lo0 << s, 0u);
    }
    let s = u32(t0);
    return vec2<u32>((hi0 << s) | (lo0 >> (32u - s)), lo0 << s);
}
fn bit64(hi: u32, lo: u32, pos0: i32) -> u32 {
    if (pos0 >= 32) { return (hi >> u32(pos0 - 32)) & 1u; }
    return (lo >> u32(pos0)) & 1u;
}
fn stickyBelow(hi: u32, lo: u32, pos0: i32) -> u32 {
    if (pos0 <= 0) { return 0u; }
    if (pos0 >= 64) { return select(0u, 1u, (hi | lo) != 0u); }
    if (pos0 >= 32) {
        let s = u32(pos0) - 32u;
        return select(0u, 1u, lo != 0u) | select(0u, 1u, (hi & ((1u << s) - 1u)) != 0u);
    }
    let s = u32(pos0);
    return select(0u, 1u, (lo & ((1u << s) - 1u)) != 0u);
}
// correctly-rounded f32 fma: rn(a*b + c), pure integer
fn fma_int(a: f32, b: f32, c: f32) -> f32 {
    let ba = bitcast<u32>(a);
    let bb = bitcast<u32>(b);
    let bc = bitcast<u32>(c);
    let ea0 = i32((ba >> 23u) & 0xFFu);
    let eb0 = i32((bb >> 23u) & 0xFFu);
    let ec0 = i32((bc >> 23u) & 0xFFu);
    if (ea0 == 255 || eb0 == 255 || ec0 == 255) { return a * b + c; }
    var ma = ba & 0x7FFFFFu;
    var mb = bb & 0x7FFFFFu;
    var mc = bc & 0x7FFFFFu;
    var ea = ea0; var eb = eb0; var ec = ec0;
    if (ea0 == 0) { ea = 1; } else { ma = ma | 0x800000u; }
    if (eb0 == 0) { eb = 1; } else { mb = mb | 0x800000u; }
    if (ec0 == 0) { ec = 1; } else { mc = mc | 0x800000u; }
    let sa = ba >> 31u;
    let sbS = bb >> 31u;
    let scS = bc >> 31u;
    if (ma == 0u || mb == 0u) { return c; }
    let a1 = ma >> 12u; let a0 = ma & 0xFFFu;
    let b1 = mb >> 12u; let b0 = mb & 0xFFFu;
    let t0 = a0 * b0;
    let t1 = a0 * b1 + a1 * b0;
    let t2 = a1 * b1;
    let c1w = (t0 >> 12u) + t1;
    let w0 = t0 & 0xFFFu;
    let w1 = c1w & 0xFFFu;
    let w2 = (c1w >> 12u) + t2;
    var pHi = w2 >> 8u;
    var pLo = ((w2 & 0xFFu) << 24u) | (w1 << 12u) | w0;
    var ep = ea + eb - 300;
    if (pHi < 0x8000u) {
        pHi = (pHi << 1u) | (pLo >> 31u);
        pLo = pLo << 1u;
        ep = ep - 1;
    }
    let EP = ep + 47;
    var E = EP + 4;
    var hasC = (mc != 0u);
    if (hasC) { E = max(E, ec - 127 + 4); }
    let shP = E - EP;
    var sticky = 0u;
    var Mp = shl64(pHi, pLo, 16 - shP);
    if (shP > 16) {
        let r = shr64(pHi, pLo, shP - 16);
        Mp = vec2<u32>(r.x, r.y);
        sticky = r.z;
    }
    var McHi = 0u; var McLo = 0u;
    if (hasC) {
        let shC = E - (ec - 127);
        if (shC <= 40) {
            let r = shl64(0u, mc, 40 - shC);
            McHi = r.x; McLo = r.y;
        } else {
            let r = shr64(0u, mc, shC - 40);
            McHi = r.x; McLo = r.y;
            sticky = sticky | r.z;
        }
    }
    let spSign = sa ^ sbS;
    var rHi = 0u; var rLo = 0u; var rSign = 0u;
    if (!hasC) {
        rHi = Mp.x; rLo = Mp.y; rSign = spSign;
    } else if (spSign == scS) {
        let lo = Mp.y + McLo;
        let carry = select(0u, 1u, lo < Mp.y);
        rHi = Mp.x + McHi + carry;
        rLo = lo;
        rSign = spSign;
    } else {
        var gt = (Mp.x > McHi) || (Mp.x == McHi && Mp.y > McLo);
        var eq = (Mp.x == McHi && Mp.y == McLo);
        if (eq) { return bitcast<f32>(0u); }
        var bigHi = Mp.x; var bigLo = Mp.y; var smlHi = McHi; var smlLo = McLo;
        rSign = spSign;
        if (!gt) {
            bigHi = McHi; bigLo = McLo; smlHi = Mp.x; smlLo = Mp.y;
            rSign = scS;
        }
        let borrow = select(0u, 1u, bigLo < smlLo);
        rHi = bigHi - smlHi - borrow;
        rLo = bigLo - smlLo;
    }
    if (rHi == 0u && rLo == 0u) { return bitcast<f32>(0u); }
    var lb: i32;
    if (rHi != 0u) { lb = lead32(rHi) + 32; } else { lb = lead32(rLo); }
    let F = E - 63;
    var eUnb = F + lb;
    let signBit = rSign << 31u;
    if (eUnb >= 128) { return bitcast<f32>(signBit | 0x7F800000u); }
    if (eUnb >= -126) {
        var mR: u32;
        let R = lb - 23;
        if (R <= 0) {
            let r = shl64(rHi, rLo, -R);
            mR = r.y;
        } else {
            if (R >= 32) { mR = rHi >> u32(R - 32); }
            else { mR = (rHi << u32(32 - R)) | (rLo >> u32(R)); }
            var g = 0u; var rr = 0u; var st = sticky;
            if (R >= 2) {
                g = bit64(rHi, rLo, R - 1);
                rr = bit64(rHi, rLo, R - 2);
                st = st | stickyBelow(rHi, rLo, R - 2);
            } else {
                g = bit64(rHi, rLo, 0);
            }
            if (g != 0u && (rr != 0u || st != 0u || (mR & 1u) == 1u)) {
                mR = mR + 1u;
                if (mR == 0x1000000u) { mR = 0x800000u; eUnb = eUnb + 1; }
                if (eUnb >= 128) { return bitcast<f32>(signBit | 0x7F800000u); }
            }
        }
        return bitcast<f32>(signBit | (u32(eUnb + 127) << 23u) | (mR & 0x7FFFFFu));
    }
    let shiftS = F + 149;
    var k: u32;
    if (shiftS >= 0) {
        let r = shl64(rHi, rLo, shiftS);
        k = r.y;
    } else {
        let t = -shiftS;
        if (t >= 64) { return bitcast<f32>(signBit); }
        let r = shr64(rHi, rLo, t);
        k = r.y;
        var g = 0u; var rr = 0u; var st = sticky;
        if (t >= 2) {
            g = bit64(rHi, rLo, t - 1);
            rr = bit64(rHi, rLo, t - 2);
            st = st | stickyBelow(rHi, rLo, t - 2);
        } else {
            g = bit64(rHi, rLo, 0);
        }
        if (g != 0u && (rr != 0u || st != 0u || (k & 1u) == 1u)) {
            k = k + 1u;
            if (k == 0x800000u) { return bitcast<f32>(signBit | 0x800000u); }
        }
    }
    if (k == 0u) { return bitcast<f32>(signBit); }
    return bitcast<f32>(signBit | k);
}
// rn(A * 2^(q-126)) for normal A > 0, q in [0,252]: power-of-two scaling with
// integer GRS rounding in the subnormal grid (the driver's OpFMul flushes
// subnormal outputs; a correctly-rounded multiply does not).
fn scale_pow2(Abits: u32, q: u32) -> u32 {
    let sgn = Abits & 0x80000000u;
    if ((Abits & 0x7FFFFFFFu) == 0u) { return sgn; }
    let mA = (Abits & 0x7FFFFFu) | 0x800000u;
    let eA = i32((Abits >> 23u) & 0xFFu) - 127;
    let eRes = eA + i32(q) - 126;
    if (eRes >= 128) { return sgn | 0x7F800000u; }
    if (eRes >= -126) {
        return sgn | (u32(eRes + 127) << 23u) | (Abits & 0x7FFFFFu);
    }
    let s = -126 - eRes;
    if (s >= 25) { return sgn; }
    let k = mA >> u32(s);
    var kR = k;
    var g = 0u; var rr = 0u; var st = 0u;
    if (s >= 2) {
        g = (mA >> u32(s - 1)) & 1u;
        rr = (mA >> u32(s - 2)) & 1u;
        st = select(0u, 1u, (mA & ((1u << u32(s - 2)) - 1u)) != 0u);
    } else {
        g = (mA >> u32(s - 1)) & 1u;
    }
    if (g != 0u && (rr != 0u || st != 0u || (k & 1u) == 1u)) { kR = k + 1u; }
    if (kR >= 0x800000u) { return sgn | 0x800000u; }
    return sgn | kR;
}
fn expf_bt(x: f32) -> f32 {
    let f5 = clamp(fma_int(x, bitcast<f32>(0x3BBB989Du), 0.5), 0.0, 1.0);
    let b5 = bitcast<u32>(f5);
    var f8 = 12582913u;
    if ((b5 & 0x7FFFFFFFu) != 0u) {
        let raw_e = i32((b5 >> 23u) & 0xFFu) - 127;
        let m = select(b5 & 0x7FFFFFu, (b5 & 0x7FFFFFu) | 0x800000u, raw_e != -127);
        let ee = select(-126, raw_e, raw_e != -127);
        let pp = (m << 8u) - (m << 2u);
        let shift = u32(23 - ee);
        let q = select(0u, pp >> shift, shift < 32u);
        f8 = 12582913u + q;
    }
    let f8f = bitcast<f32>(0x4B000000u | (f8 - 0x800000u));
    let f10 = -(f8f + bitcast<f32>(0xCB40007Fu));
    let f12 = fma_int(x, bitcast<f32>(0x3FB8AA3Bu), f10);
    let f14 = fma_int(x, bitcast<f32>(0x32A57060u), f12);
    let q = f8 - 12582913u;
    return bitcast<f32>(scale_pow2(bitcast<u32>(exp2(f14)), q));
}
";

pub fn gqa_decode_single(nqh: usize, nkvh: usize, d: usize, bs: usize, cap: usize) -> String {
    let tchunks = (bs / d).max(1);
    assert_eq!(d % 2, 0);
    assert_eq!(bs % d, 0);
    format!(
        "{HALF_AT}
{EXP_BT}
struct Cfg {{ cur_len: u32, max_seq: u32, scale: f32, _p: f32 }};

@group(0) @binding(0) var<storage, read>       Q4:  array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       KC4: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       VC:  array<u32>;
@group(0) @binding(3) var<storage, read_write> Out: array<u32>;
@group(0) @binding(4) var<uniform>             cfg: Cfg;

const D: u32 = {d}u;
const D2: u32 = {d2}u;
const D4: u32 = {d4}u;
const BS: u32 = {bs}u;
const TCH: u32 = {tchunks}u;
const REP: u32 = {rep}u;
const CAP: u32 = {cap}u;

var<workgroup> sc:      array<f32, {cap}>;
var<workgroup> partial: array<f32, {d} * {tchunks}>;
var<workgroup> red_max: array<f32, {bs}>;
var<workgroup> red_sum: array<f32, {bs}>;

@compute @workgroup_size({bs})
fn gqa(@builtin(workgroup_id) wgid: vec3<u32>,
       @builtin(local_invocation_id) lid: vec3<u32>) {{
    let qh = wgid.x;
    let kh = qh / REP;
    let qbase = qh * D2;
    let kbase = kh * cfg.max_seq * D2;

    // Stage 1 — scores[t] = (Q . K[t]) * scale.  Same word order and the same
    // f32 adds as the scalar form; four words per load instead of one (see the
    // split kernel's stage 1 for why the transactions matter).
    let q4 = qbase >> 2u;
    for (var t = lid.x; t < cfg.cur_len; t = t + BS) {{
        var dot = 0.0;
        let row4 = (kbase + t * D2) >> 2u;
        for (var j4 = 0u; j4 < D4; j4 = j4 + 1u) {{
            let qv = Q4[q4 + j4];
            let kv = KC4[{krow} + j4];
            let q0 = unpack2x16float(qv.x);
            let k0 = unpack2x16float(kv.x);
            dot = dot + (q0.x * k0.x + q0.y * k0.y);
            let q1 = unpack2x16float(qv.y);
            let k1 = unpack2x16float(kv.y);
            dot = dot + (q1.x * k1.x + q1.y * k1.y);
            let q2 = unpack2x16float(qv.z);
            let k2 = unpack2x16float(kv.z);
            dot = dot + (q2.x * k2.x + q2.y * k2.y);
            let q3 = unpack2x16float(qv.w);
            let k3 = unpack2x16float(kv.w);
            dot = dot + (q3.x * k3.x + q3.y * k3.y);
        }}
        sc[t] = dot * cfg.scale;
    }}
    workgroupBarrier();

    // Stage 2 — row max
    var lmax = bitcast<f32>(0xFF800000u);
    for (var t = lid.x; t < cfg.cur_len; t = t + BS) {{
        if (sc[t] > lmax) {{ lmax = sc[t]; }}
    }}
    red_max[lid.x] = lmax;
    workgroupBarrier();
    for (var s = BS >> 1u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{ red_max[lid.x] = max(red_max[lid.x], red_max[lid.x + s]); }}
        workgroupBarrier();
    }}
    let row_max = red_max[0];
    workgroupBarrier();

    // Stage 3 — exp + sum
    var lsum = 0.0;
    for (var t = lid.x; t < cfg.cur_len; t = t + BS) {{
        let e = expf_bt(sc[t] - row_max);
        sc[t] = e;
        lsum = lsum + e;
    }}
    red_sum[lid.x] = lsum;
    workgroupBarrier();
    for (var s = BS >> 1u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{ red_sum[lid.x] = red_sum[lid.x] + red_sum[lid.x + s]; }}
        workgroupBarrier();
    }}
    let inv_sum = 1.0 / red_sum[0];
    workgroupBarrier();

    // Stage 4 — partial[t_idx][j] then cross-t_chunk merge.  One thread owns a
    // *pair* of dims: both halves sit in the same f16 word, so a single LDS and
    // a single unpack feed two FMAs instead of one.  Each dim still walks the
    // same stride-`TCH` key set in the same order, so every accumulator keeps its
    // original summation order — only the instruction count per element drops
    // (the block's second half has no pair left to own and idles here).
    let jp = lid.x % D2;
    let t_idx = lid.x / D2;
    if (t_idx < TCH) {{
        var a0 = 0.0;
        var a1 = 0.0;
        for (var t = t_idx; t < cfg.cur_len; t = t + TCH) {{
            let row = kbase + t * D2;
            let v = unpack2x16float(VC[row + jp]);
            a0 = a0 + sc[t] * v.x;
            a1 = a1 + sc[t] * v.y;
        }}
        partial[t_idx * D + jp * 2u] = a0;
        partial[t_idx * D + jp * 2u + 1u] = a1;
    }}
    workgroupBarrier();

    // exactly one element per thread, written as whole f16 words — the
    // per-element arithmetic, and its order, is unchanged.
    if (lid.x < D2) {{
        var a0 = 0.0;
        var a1 = 0.0;
        for (var ti = 0u; ti < TCH; ti = ti + 1u) {{
            a0 = a0 + partial[ti * D + lid.x * 2u];
            a1 = a1 + partial[ti * D + lid.x * 2u + 1u];
        }}
        Out[qbase + lid.x] = pack2x16float(vec2<f32>(a0 * inv_sum, a1 * inv_sum));
    }}
}}
",
        d2 = d / 2,
        d4 = d / 8,
        rep = nqh / nkvh,
        krow = "row4",
    )
}

/// [`gqa_decode_single`] with a slot axis: `wgid.z` is the slot.
///
/// One dispatch of `nqh * slots` workgroups replaces `slots` dispatches of
/// `nqh`.  At the 1.7B shape `nqh` is 16, so the shipped dispatch is 16
/// workgroups on a 96-CU card -- latency-bound at ~49 GB/s of KV traffic; eight
/// of those run one after another (measured 14.51 ms per 28 layers) while one
/// 128-workgroup dispatch measures 1.97 ms for the same work.  Each workgroup
/// still walks its own slot's rows in the same order with the same trees, and
/// the only new input is that slot's `cur_len`, so a batched slot is
/// bit-identical to a lone one.
pub fn gqa_decode_single_slots(nqh: usize, nkvh: usize, d: usize, bs: usize, cap: usize) -> String {
    let tchunks = (bs / d).max(1);
    assert_eq!(d % 2, 0);
    assert_eq!(bs % d, 0);
    format!(
        "enable wgpu_binding_array;
/// naga requires the binding-array element to be a struct; one struct per word
/// shape keeps the bodies the same `array<...>` the lone kernels use.
struct Arr4  {{ v: array<vec4<u32>> }};
struct Arr32 {{ v: array<u32> }};
{HALF_AT}
{EXP_BT}
struct Cfg {{ cur_len: u32, max_seq: u32, scale: f32, _p: f32 }};

@group(0) @binding(0) var<storage, read>       Q4:  binding_array<Arr4, 32>;
@group(0) @binding(1) var<storage, read>       KC4: binding_array<Arr4, 32>;
@group(0) @binding(2) var<storage, read>       VC:  binding_array<Arr32, 32>;
@group(0) @binding(3) var<storage, read_write> Out: binding_array<Arr32, 32>;
/// Group 1: wgpu refuses a bind group that mixes a binding array with a
/// uniform, and the per-slot arrays already fill group 0.
@group(1) @binding(0) var<uniform>             cfg: Cfg;
/// Each slot's own `cur_len`: a batch holds segments of different lengths, so
/// this cannot ride the one shared uniform.
@group(0) @binding(5) var<storage, read>       CL:   array<u32>;
/// The slots this dispatch owns, compacted: `wgid.z` -> slot.  A step runs one
/// dispatch per block width (the shipped kernel picks 256 or 512 above/below
/// 512, and the two trees differ), so this is the only place a dispatch is told
/// which slots are its own.
@group(0) @binding(6) var<storage, read>       PICK: array<u32>;

const D: u32 = {d}u;
const D2: u32 = {d2}u;
const D4: u32 = {d4}u;
const BS: u32 = {bs}u;
const TCH: u32 = {tchunks}u;
const REP: u32 = {rep}u;
const CAP: u32 = {cap}u;

var<workgroup> sc:      array<f32, {cap}>;
var<workgroup> partial: array<f32, {d} * {tchunks}>;
var<workgroup> red_max: array<f32, {bs}>;
var<workgroup> red_sum: array<f32, {bs}>;

@compute @workgroup_size({bs})
fn gqa(@builtin(workgroup_id) wgid: vec3<u32>,
       @builtin(local_invocation_id) lid: vec3<u32>) {{
    let qh = wgid.x;
    let kh = qh / REP;
    let z = PICK[wgid.z];
    let cur_len = CL[z];
    let qbase = qh * D2;
    let kbase = kh * cfg.max_seq * D2;

    // Stage 1 — scores[t] = (Q . K[t]) * scale.  Same word order and the same
    // f32 adds as the scalar form; four words per load instead of one (see the
    // split kernel's stage 1 for why the transactions matter).
    let q4 = qbase >> 2u;
    for (var t = lid.x; t < cur_len; t = t + BS) {{
        var dot = 0.0;
        let row4 = (kbase + t * D2) >> 2u;
        for (var j4 = 0u; j4 < D4; j4 = j4 + 1u) {{
            let qv = Q4[z].v[q4 + j4];
            let kv = KC4[z].v[{krow} + j4];
            let q0 = unpack2x16float(qv.x);
            let k0 = unpack2x16float(kv.x);
            dot = dot + (q0.x * k0.x + q0.y * k0.y);
            let q1 = unpack2x16float(qv.y);
            let k1 = unpack2x16float(kv.y);
            dot = dot + (q1.x * k1.x + q1.y * k1.y);
            let q2 = unpack2x16float(qv.z);
            let k2 = unpack2x16float(kv.z);
            dot = dot + (q2.x * k2.x + q2.y * k2.y);
            let q3 = unpack2x16float(qv.w);
            let k3 = unpack2x16float(kv.w);
            dot = dot + (q3.x * k3.x + q3.y * k3.y);
        }}
        sc[t] = dot * cfg.scale;
    }}
    workgroupBarrier();

    // Stage 2 — row max
    var lmax = bitcast<f32>(0xFF800000u);
    for (var t = lid.x; t < cur_len; t = t + BS) {{
        if (sc[t] > lmax) {{ lmax = sc[t]; }}
    }}
    red_max[lid.x] = lmax;
    workgroupBarrier();
    for (var s = BS >> 1u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{ red_max[lid.x] = max(red_max[lid.x], red_max[lid.x + s]); }}
        workgroupBarrier();
    }}
    let row_max = red_max[0];
    workgroupBarrier();

    // Stage 3 — exp + sum
    var lsum = 0.0;
    for (var t = lid.x; t < cur_len; t = t + BS) {{
        let e = expf_bt(sc[t] - row_max);
        sc[t] = e;
        lsum = lsum + e;
    }}
    red_sum[lid.x] = lsum;
    workgroupBarrier();
    for (var s = BS >> 1u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{ red_sum[lid.x] = red_sum[lid.x] + red_sum[lid.x + s]; }}
        workgroupBarrier();
    }}
    let inv_sum = 1.0 / red_sum[0];
    workgroupBarrier();

    // Stage 4 — partial[t_idx][j] then cross-t_chunk merge.  One thread owns a
    // *pair* of dims: both halves sit in the same f16 word, so a single LDS and
    // a single unpack feed two FMAs instead of one.  Each dim still walks the
    // same stride-`TCH` key set in the same order, so every accumulator keeps its
    // original summation order — only the instruction count per element drops
    // (the block's second half has no pair left to own and idles here).
    let jp = lid.x % D2;
    let t_idx = lid.x / D2;
    if (t_idx < TCH) {{
        var a0 = 0.0;
        var a1 = 0.0;
        for (var t = t_idx; t < cur_len; t = t + TCH) {{
            let row = kbase + t * D2;
            let v = unpack2x16float(VC[z].v[row + jp]);
            a0 = a0 + sc[t] * v.x;
            a1 = a1 + sc[t] * v.y;
        }}
        partial[t_idx * D + jp * 2u] = a0;
        partial[t_idx * D + jp * 2u + 1u] = a1;
    }}
    workgroupBarrier();

    // exactly one element per thread, written as whole f16 words — the
    // per-element arithmetic, and its order, is unchanged.
    if (lid.x < D2) {{
        var a0 = 0.0;
        var a1 = 0.0;
        for (var ti = 0u; ti < TCH; ti = ti + 1u) {{
            a0 = a0 + partial[ti * D + lid.x * 2u];
            a1 = a1 + partial[ti * D + lid.x * 2u + 1u];
        }}
        Out[z].v[qbase + lid.x] = pack2x16float(vec2<f32>(a0 * inv_sum, a1 * inv_sum));
    }}
}}
",
        d2 = d / 2,
        d4 = d / 8,
        rep = nqh / nkvh,
        krow = "row4",
    )
}

/// `silu_mul_split_f16` — `gu` holds `[gate | up]` per row.  The reference's
/// approx exp is `ex2.approx(x * log2(e))`, so `exp2` is used rather than `exp`.
pub fn silu_mul_split(inter: usize) -> String {
    assert!(inter % 2 == 0);
    let inter2 = inter / 2;
    let threads = 256usize;
    let grid = inter2.div_ceil(threads);
    let _ = grid;
    format!(
        "@group(0) @binding(0) var<storage, read>       Gu:  array<u32>;
@group(0) @binding(1) var<storage, read_write> Out: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;

struct Cfg {{ inter2: u32, total2: u32, gx: u32, _b: u32 }};

const INTER2: u32 = {inter2}u;
const LOG2E: f32 = {log2e};
const THREADS: u32 = {threads}u;

@compute @workgroup_size({threads})
fn silu_mul_split(@builtin(global_invocation_id) gid: vec3<u32>) {{
    // Two grid axes: `rows · inter/2` words overflows wgpu's 65535-per-dimension
    // limit for prefills longer than ~5.8 minutes, so x is capped and y continues
    // the flat index space (`gx · THREADS` words per row).
    let i = gid.x + gid.y * (cfg.gx * THREADS);
    if (i >= cfg.total2) {{ return; }}
    // rows of [row_gate | row_up]: decode runs one row, prefill s rows
    let row = i / cfg.inter2;
    let c = i - row * cfg.inter2;
    let base = row * cfg.inter2 * 2u;
    let gv = unpack2x16float(Gu[base + c]);
    let uv = unpack2x16float(Gu[base + cfg.inter2 + c]);
    let sx = 1.0 / (1.0 + exp2(-gv.x * LOG2E));
    let sy = 1.0 / (1.0 + exp2(-gv.y * LOG2E));
    Out[i] = pack2x16float(vec2<f32>(gv.x * sx * uv.x, gv.y * sy * uv.y));
}}
",
        log2e = format_f32(LOG2_E),
    )
}

/// [`silu_mul_split`] over `batch` sequences in one dispatch, `wgid.z` = slot.
///
/// `Gu` and `Out` are `[slot, ...]` rows of one allocation: the slot is added to
/// both addresses, and the element's own arithmetic (`pack2x16float(gv * s * uv)`
/// with the same approx exp) is untouched.
pub fn silu_mul_split_slots(inter: usize) -> String {
    assert!(inter % 2 == 0);
    let inter2 = inter / 2;
    let threads = 256usize;
    format!(
        "enable wgpu_binding_array;
/// naga requires the binding-array element to be a struct; one struct per word
/// shape keeps the bodies the same `array<...>` the lone kernels use.
struct Arr4  {{ v: array<vec4<u32>> }};
struct Arr32 {{ v: array<u32> }};
@group(0) @binding(0) var<storage, read>       Gu:  binding_array<Arr32, 32>;
@group(0) @binding(1) var<storage, read_write> Out: binding_array<Arr32, 32>;
/// Group 1: wgpu refuses a bind group that mixes a binding array with a uniform.
@group(1) @binding(0) var<uniform>             cfg: Cfg;

struct Cfg {{ inter2: u32, total2: u32, gx: u32, _b: u32 }};

const INTER2: u32 = {inter2}u;
const LOG2E: f32 = {log2e};
const THREADS: u32 = {threads}u;

@compute @workgroup_size({threads})
fn silu_mul_split_slots(@builtin(workgroup_id) wgid: vec3<u32>,
                        @builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x + gid.y * (cfg.gx * THREADS);
    if (i >= cfg.total2) {{ return; }}
    let row = i / cfg.inter2;
    let c = i - row * cfg.inter2;
    let base = row * cfg.inter2 * 2u;
    let gv = unpack2x16float(Gu[wgid.z].v[base + c]);
    let uv = unpack2x16float(Gu[wgid.z].v[base + cfg.inter2 + c]);
    let sx = 1.0 / (1.0 + exp2(-gv.x * LOG2E));
    let sy = 1.0 / (1.0 + exp2(-gv.y * LOG2E));
    Out[wgid.z].v[i] = pack2x16float(vec2<f32>(gv.x * sx * uv.x, gv.y * sy * uv.y));
}}
",
        log2e = format_f32(LOG2_E),
    )
}

/// `fused_gqa_decode_split_p1_f16` — long-context (cur_len > 1024) split-K
/// attention, phase 1: one workgroup per (q_head, chunk).  Chunk-local scores →
/// chunk max/sum via the shared block-reduction tree → unnormalized partial
/// numerator.  Block size is fixed at 256 regardless of chunk width;
/// `t_split = 256/d = 2` threads cooperate per output element.
///
/// Empty chunks (t_start >= cur_len) write max=-inf, sum=0, partial=0 — the
/// merge kernel's `> -inf` guard depends on it.
pub fn gqa_decode_split_p1(nqh: usize, nkvh: usize, d: usize, chunk: usize) -> String {
    assert_eq!(d % 2, 0);
    let t_split = 256 / d;
    format!(
        "{HALF_AT}
{EXP_BT}
struct Cfg {{ cur_len: u32, max_seq: u32, scale: f32, n_chunks: u32 }};

@group(0) @binding(0) var<storage, read>       Q4:   array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       KC4:  array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       VC:   array<u32>;
@group(0) @binding(3) var<storage, read_write> POut: array<f32>;
@group(0) @binding(4) var<storage, read_write> PMax: array<f32>;
@group(0) @binding(5) var<storage, read_write> PSum: array<f32>;
@group(0) @binding(6) var<uniform>             cfg:  Cfg;

const D: u32 = {d}u;
const D2: u32 = {d2}u;
const D4: u32 = {d4}u;
const CHUNK: u32 = {chunk}u;
const BS: u32 = 256u;
const T_SPLIT: u32 = {t_split}u;
const REP: u32 = {rep}u;

var<workgroup> sc:      array<f32, {chunk}u>;
var<workgroup> partial: array<f32, {d} * {t_split}u>;
var<workgroup> red_max: array<f32, 256u>;
var<workgroup> red_sum: array<f32, 256u>;

@compute @workgroup_size(256)
fn gqa_split_p1(@builtin(workgroup_id) wgid: vec3<u32>,
                @builtin(local_invocation_id) lid: vec3<u32>) {{
    let qh = wgid.x;
    let by = wgid.y;
    let kh = qh / REP;
    let qbase = qh * D2;
    let kbase = kh * cfg.max_seq * D2;
    let t_start = by * CHUNK;
    let mi = qh * cfg.n_chunks + by;

    if (t_start >= cfg.cur_len) {{
        if (lid.x == 0u) {{
            PMax[mi] = bitcast<f32>(0xFF800000u);
            PSum[mi] = 0.0;
        }}
        if (lid.x < D) {{
            POut[mi * D + lid.x] = 0.0;
        }}
        return;
    }}
    let chunk_len = min(CHUNK, cfg.cur_len - t_start);

    // Stage 1 — scores[t] = (Q . K[t_start + t]) * scale, chunk-local t
    //
    // Read through `vec4<u32>` (four words per instruction, 16 B per lane).
    // The per-key *word order* is untouched — each loaded word is unpacked and
    // accumulated in exactly the sequence the scalar loop used — so the f32
    // reduction is bit-identical.  What changes is the transaction count: the
    // scalar form had each lane walking its own 256 B row, so every warp load
    // touched 32 different cache lines (measured ~30 GB/s effective on this
    // Pascal part, against 289 GB/s for the vectorised GEMV shape).
    let q4 = qbase >> 2u;
    for (var t = lid.x; t < chunk_len; t = t + BS) {{
        var dot = 0.0;
        let row4 = (kbase + (t_start + t) * D2) >> 2u;
        for (var j4 = 0u; j4 < D4; j4 = j4 + 1u) {{
            let qv = Q4[q4 + j4];
            let kv = KC4[{krow} + j4];
            let q0 = unpack2x16float(qv.x);
            let k0 = unpack2x16float(kv.x);
            dot = dot + (q0.x * k0.x + q0.y * k0.y);
            let q1 = unpack2x16float(qv.y);
            let k1 = unpack2x16float(kv.y);
            dot = dot + (q1.x * k1.x + q1.y * k1.y);
            let q2 = unpack2x16float(qv.z);
            let k2 = unpack2x16float(kv.z);
            dot = dot + (q2.x * k2.x + q2.y * k2.y);
            let q3 = unpack2x16float(qv.w);
            let k3 = unpack2x16float(kv.w);
            dot = dot + (q3.x * k3.x + q3.y * k3.y);
        }}
        sc[t] = dot * cfg.scale;
    }}
    workgroupBarrier();

    // Stage 2 — chunk max
    var lmax = bitcast<f32>(0xFF800000u);
    for (var t = lid.x; t < chunk_len; t = t + BS) {{
        if (sc[t] > lmax) {{ lmax = sc[t]; }}
    }}
    red_max[lid.x] = lmax;
    workgroupBarrier();
    for (var s = BS >> 1u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{ red_max[lid.x] = max(red_max[lid.x], red_max[lid.x + s]); }}
        workgroupBarrier();
    }}
    let chunk_max = red_max[0];
    workgroupBarrier();

    // Stage 3 — exp + sum
    var lsum = 0.0;
    for (var t = lid.x; t < chunk_len; t = t + BS) {{
        let e = expf_bt(sc[t] - chunk_max);
        sc[t] = e;
        lsum = lsum + e;
    }}
    red_sum[lid.x] = lsum;
    workgroupBarrier();
    for (var s = BS >> 1u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{ red_sum[lid.x] = red_sum[lid.x] + red_sum[lid.x + s]; }}
        workgroupBarrier();
    }}
    let chunk_sum = red_sum[0];
    workgroupBarrier();

    // Stage 4 — partial numerator, unnormalized; V read at the global position.
    // Same dim-pair split as the single-block kernel: one thread owns two dims
    // that share an f16 word, so one LDS + one unpack feed two FMAs, and every
    // dim keeps its original stride-`T_SPLIT` key order (bit-identical).
    let jp = lid.x % D2;
    let t_idx = lid.x / D2;
    if (t_idx < T_SPLIT) {{
        var a0 = 0.0;
        var a1 = 0.0;
        for (var t = t_idx; t < chunk_len; t = t + T_SPLIT) {{
            let row = kbase + (t_start + t) * D2;
            let v = unpack2x16float(VC[row + jp]);
            a0 = a0 + sc[t] * v.x;
            a1 = a1 + sc[t] * v.y;
        }}
        partial[t_idx * D + jp * 2u] = a0;
        partial[t_idx * D + jp * 2u + 1u] = a1;
    }}
    workgroupBarrier();

    if (lid.x < D) {{
        var acc = 0.0;
        for (var ti = 0u; ti < T_SPLIT; ti = ti + 1u) {{
            acc = acc + partial[ti * D + lid.x];
        }}
        POut[mi * D + lid.x] = acc;
    }}
    if (lid.x == 0u) {{
        PMax[mi] = chunk_max;
        PSum[mi] = chunk_sum;
    }}
}}
",
        d2 = d / 2,
        d4 = d / 8,
        t_split = t_split,
        rep = nqh / nkvh,
        krow = "row4",
    )
}

/// [`gqa_decode_split_p1`] with the `REP` q heads that share a kv head computed
/// together, so each K/V element is read once instead of once per q head.
///
/// The KV read is the whole cost of this stage — its measured marginal price
/// scales linearly with `cur_len` (0 us/layer at a 250-token context, 66 at
/// 1340, 122 at 2600) — and GQA makes `REP` of those reads redundant: with
/// `nqh = 16`, `nkvh = 8`, every kv head is fetched twice.  Here one workgroup
/// covers a `(kv_head, chunk)` pair and thread `lid` computes *both* q heads'
/// dot products from the one K row it loaded, then both weighted-V sums from the
/// one V element it loaded.
///
/// Bit-identical to the per-q-head kernel: each head keeps its own score array,
/// its own 256-wide reduction tree (the two trees run side by side in one pass,
/// indices `0..256` for the first head and `256..512` for the second, so each
/// half sees exactly the same sequence of ops), and the same per-head key and
/// dim strides.  Only the number of workgroups changes: `nkvh` instead of `nqh`
/// per chunk, each doing `REP` heads' worth of work.
pub fn gqa_decode_split_p1_pair(
    d: usize,
    chunk: usize,
    rep: usize,
    row0: bool,
    coop: bool,
    pf: bool,
    coal: bool,
    depth: usize,
) -> String {
    assert_eq!(d % 2, 0);
    assert_eq!(
        rep, 2,
        "paired split handles exactly the 2 q heads of a kv head"
    );
    let t_split = 256 / d;
    assert!(t_split >= 1);
    // `row0` points every K read at row 0, so the whole workgroup shares one
    // 256 B line set and every load is an L1 hit.  Same instruction count, same
    // FMAs, same reduction -- it separates "the strided row pattern costs" from
    // "the loads cost", the question the paired kernel (+1.0%) and the
    // accumulator depth (flat) between them could not settle.
    // `coal` is a second diagnostic: it keeps the 16 loads per lane and the loop
    // shape but re-aims them at consecutive addresses across lanes, i.e. the
    // access pattern a dim-major K layout would produce.  Values are wrong (it is
    // only ever a duplicated dispatch), so it answers one question: is the
    // *pattern* worth more than the ~1.3% the cooperative form got?
    let kv_index = if row0 {
        "0u + j4".to_string()
    } else if coal {
        // consecutive lanes read consecutive 16 B words, so one warp instruction
        // covers 512 B = 4 cache lines instead of 32 -- but each lane still does
        // all 16 loads, unlike the cooperative form which cut the per-lane load
        // count.  That is what a dim-major K layout would look like.
        "((row4 >> 4u) << 4u) + j4 * 32u + (lid.x & 31u)".to_string()
    } else {
        "row4 + j4".to_string()
    };
    // The same three address forms with the loop index baked in, for the
    // unrolled window below.
    let kv_at = |j: usize| -> String {
        if row0 {
            format!("{j}u")
        } else if coal {
            format!("((row4 >> 4u) << 4u) + {j}u * 32u + (lid.x & 31u)")
        } else {
            format!("row4 + {j}u")
        }
    };
    // `coop` is the fix that probe implies: a group of 8 lanes walks one key's
    // 256 B row together, each lane taking two `vec4`, then a 3-round xor tree
    // folds the eight partial dots.  A warp then covers four *contiguous* rows
    // per instruction instead of 32 scattered ones -- 4 cache lines instead of
    // 32 -- and the warp-instruction count per key is unchanged (`16.5` either
    // way: the serial form spends its extra lanes re-loading the same Q vector
    // 32 times over, which is what the cooperative form saves).  Requires
    // `subgroupShuffleXor`, i.e. `Features::SUBGROUP`.
    assert!(!(row0 && coop), "row0 is the serial diagnostic");
    let s1: String = if coop {
        let mut t = String::new();
        t.push_str(
            "        let lg = lid.x / 8u;\n\
             \x20       let li = lid.x % 8u;\n\
             \x20       let j0 = li * 2u;\n\
             \x20       let niter = (chunk_len + 31u) / 32u;\n\
             \x20       for (var it = 0u; it < niter; it = it + 1u) {\n\
             \x20           let t = lg + it * 32u;\n\
             \x20           let live = t < chunk_len;\n\
             \x20           let tt = select(0u, t, live);\n\
             \x20           let row4 = (kbase + (t_start + tt) * D2) >> 2u;\n\
             \x20           let kv0 = KC4[row4 + j0];\n\
             \x20           let kv1 = KC4[row4 + j0 + 1u];\n\
             \x20           let qa0 = Q4[qa4 + j0];\n\
             \x20           let qa1 = Q4[qa4 + j0 + 1u];\n\
             \x20           let qb0 = Q4[qb4 + j0];\n\
             \x20           let qb1 = Q4[qb4 + j0 + 1u];\n\
             \x20           var sa = 0.0;\n\
             \x20           var sb = 0.0;\n",
        );
        for (w, kv, qv, acc) in [("0", "kv0", "qa0", "sa"), ("1", "kv1", "qa1", "sa")] {
            for j in 0..4 {
                let f = ['x', 'y', 'z', 'w'][j];
                t.push_str(&format!(
                    "            let k{w}_{j} = unpack2x16float({kv}.{f});\n"
                ));
                t.push_str(&format!(
                    "            let a{w}_{j} = unpack2x16float({qv}.{f});\n"
                ));
                t.push_str(&format!(
                    "            {acc} = {acc} + (a{w}_{j}.x * k{w}_{j}.x + a{w}_{j}.y * k{w}_{j}.y);\n"
                ));
            }
        }
        // head B reuses the K values loaded for head A
        for (w, qv) in [("0", "qb0"), ("1", "qb1")] {
            for j in 0..4 {
                let f = ['x', 'y', 'z', 'w'][j];
                t.push_str(&format!(
                    "            let b{w}_{j} = unpack2x16float({qv}.{f});\n"
                ));
                t.push_str(&format!(
                    "            sb = sb + (b{w}_{j}.x * k{w}_{j}.x + b{w}_{j}.y * k{w}_{j}.y);\n"
                ));
            }
        }
        t.push_str(
            "            sa = sa + subgroupShuffleXor(sa, 1u);\n\
             \x20           sa = sa + subgroupShuffleXor(sa, 2u);\n\
             \x20           sa = sa + subgroupShuffleXor(sa, 4u);\n\
             \x20           sb = sb + subgroupShuffleXor(sb, 1u);\n\
             \x20           sb = sb + subgroupShuffleXor(sb, 2u);\n\
             \x20           sb = sb + subgroupShuffleXor(sb, 4u);\n\
             \x20           if (live && li == 0u) {\n\
             \x20               sc_a[t] = sa * cfg.scale;\n\
             \x20               sc_b[t] = sb * cfg.scale;\n\
             \x20           }\n\
             \x20       }\n",
        );
        t
    } else {
        let mut t = String::new();
        if depth > 1 {
            // The load window.  `depth` K-row words are issued *before* any of
            // them is consumed, then the identical add sequence replays in the
            // original order -- same operands, same association, only when the
            // fetch happens moves.  The `pf` form below leaves one word of
            // slack; measured on `attn_bench` at a 2560-token context that is
            // nowhere near enough (chunk 512: 96.4 us/layer with one word of
            // slack, 54.8 with four, 55.4 with eight, 110 with sixteen -- the
            // last one spills).
            t.push_str(
                "        for (var t = lid.x; t < chunk_len; t = t + BS) {\n\
                 \x20       var da = 0.0;\n\
                 \x20       var db = 0.0;\n\
                 \x20       let row4 = (kbase + (t_start + t) * D2) >> 2u;\n",
            );
            let mut j0 = 0usize;
            while j0 < 16 {
                let n = depth.min(16 - j0);
                t.push_str("        {\n");
                for k in 0..n {
                    let j = j0 + k;
                    t.push_str(&format!("            let kv{j} = KC4[{}];\n", kv_at(j)));
                }
                for k in 0..n {
                    let j = j0 + k;
                    t.push_str("            {\n");
                    t.push_str(&format!("            let qa = Q4[qa4 + {j}u];\n"));
                    t.push_str(&format!("            let qb = Q4[qb4 + {j}u];\n"));
                    for s in 0..4 {
                        let f = ['x', 'y', 'z', 'w'][s];
                        t.push_str(&format!(
                            "            let k{j}_{s} = unpack2x16float(kv{j}.{f});\n"
                        ));
                        t.push_str(&format!(
                            "            let a{j}_{s} = unpack2x16float(qa.{f});\n"
                        ));
                    }
                    for s in 0..4 {
                        t.push_str(&format!(
                            "            da = da + (a{j}_{s}.x * k{j}_{s}.x + a{j}_{s}.y * k{j}_{s}.y);\n"
                        ));
                    }
                    for s in 0..4 {
                        let f = ['x', 'y', 'z', 'w'][s];
                        t.push_str(&format!(
                            "            let b{j}_{s} = unpack2x16float(qb.{f});\n"
                        ));
                    }
                    for s in 0..4 {
                        t.push_str(&format!(
                            "            db = db + (b{j}_{s}.x * k{j}_{s}.x + b{j}_{s}.y * k{j}_{s}.y);\n"
                        ));
                    }
                    t.push_str("            }\n");
                }
                t.push_str("        }\n");
                j0 += n;
            }
            t.push_str(
                "        sc_a[t] = da * cfg.scale;\n\
                 \x20       sc_b[t] = db * cfg.scale;\n\
                 \x20       }\n",
            );
        } else {
            if pf {
                // Same arithmetic, loads issued one word early: the FMA sequence and
                // every operand are untouched, only when the next `KC4`/`Q4` word is
                // fetched moves.  If load latency is what costs, this is the cheapest
                // bit-identical way to give it something to overlap with.
                t.push_str(
                    "        for (var t = lid.x; t < chunk_len; t = t + BS) {\n\
                 \x20       var da = 0.0;\n\
                 \x20       var db = 0.0;\n\
                 \x20       let row4 = (kbase + (t_start + t) * D2) >> 2u;\n\
                 \x20       var kp = KC4[row4];\n\
                 \x20       var qap = Q4[qa4];\n\
                 \x20       var qbp = Q4[qb4];\n\
                 \x20       for (var j4 = 0u; j4 < D4; j4 = j4 + 1u) {\n\
                 \x20           let kv = kp;\n\
                 \x20           let qa = qap;\n\
                 \x20           let qb = qbp;\n\
                 \x20           let jn = min(j4 + 1u, D4 - 1u);\n\
                 \x20           kp = KC4[row4 + jn];\n\
                 \x20           qap = Q4[qa4 + jn];\n\
                 \x20           qbp = Q4[qb4 + jn];\n",
                );
            } else {
                t.push_str(
                    "        for (var t = lid.x; t < chunk_len; t = t + BS) {\n\
                 \x20       var da = 0.0;\n\
                 \x20       var db = 0.0;\n\
                 \x20       let row4 = (kbase + (t_start + t) * D2) >> 2u;\n\
                 \x20       for (var j4 = 0u; j4 < D4; j4 = j4 + 1u) {\n",
                );
                t.push_str(&format!("            let kv = KC4[{kv_index}];\n"));
                t.push_str("            let qa = Q4[qa4 + j4];\n");
                t.push_str("            let qb = Q4[qb4 + j4];\n");
            }
            for j in 0..4 {
                t.push_str(&format!(
                    "            let k{j} = unpack2x16float(kv.{});\n",
                    ['x', 'y', 'z', 'w'][j]
                ));
            }
            for j in 0..4 {
                t.push_str(&format!(
                    "            let a{j} = unpack2x16float(qa.{});\n",
                    ['x', 'y', 'z', 'w'][j]
                ));
            }
            for j in 0..4 {
                t.push_str(&format!(
                    "            da = da + (a{j}.x * k{j}.x + a{j}.y * k{j}.y);\n"
                ));
            }
            for j in 0..4 {
                t.push_str(&format!(
                    "            let b{j} = unpack2x16float(qb.{});\n",
                    ['x', 'y', 'z', 'w'][j]
                ));
            }
            for j in 0..4 {
                t.push_str(&format!(
                    "            db = db + (b{j}.x * k{j}.x + b{j}.y * k{j}.y);\n"
                ));
            }
            t.push_str(
                "        }\n\
             \x20       sc_a[t] = da * cfg.scale;\n\
             \x20       sc_b[t] = db * cfg.scale;\n\
             \x20       }\n",
            );
        }
        t
    };
    format!(
        "{HALF_AT}
{EXP_BT}
struct Cfg {{ cur_len: u32, max_seq: u32, scale: f32, n_chunks: u32 }};

@group(0) @binding(0) var<storage, read>       Q4:   array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       KC4:  array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       VC:   array<u32>;
@group(0) @binding(3) var<storage, read_write> POut: array<f32>;
@group(0) @binding(4) var<storage, read_write> PMax: array<f32>;
@group(0) @binding(5) var<storage, read_write> PSum: array<f32>;
@group(0) @binding(6) var<uniform>             cfg:  Cfg;

const D: u32 = {d}u;
const D2: u32 = {d2}u;
const D4: u32 = {d4}u;
const CHUNK: u32 = {chunk}u;
const BS: u32 = 256u;
const T_SPLIT: u32 = {t_split}u;

var<workgroup> sc_a:    array<f32, {chunk}u>;
var<workgroup> sc_b:    array<f32, {chunk}u>;
var<workgroup> pa:      array<f32, {d} * {t_split}u>;
var<workgroup> pb:      array<f32, {d} * {t_split}u>;
var<workgroup> red_max: array<f32, 512u>;
var<workgroup> red_sum: array<f32, 512u>;

@compute @workgroup_size(256)
fn gqa_split_p1_pair(@builtin(workgroup_id) wgid: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>) {{
    let kh = wgid.x;
    let by = wgid.y;
    let qa4 = (kh * 2u) * D2 >> 2u;
    let qb4 = (kh * 2u + 1u) * D2 >> 2u;
    let kbase = kh * cfg.max_seq * D2;
    let t_start = by * CHUNK;
    let mi_a = (kh * 2u) * cfg.n_chunks + by;
    let mi_b = (kh * 2u + 1u) * cfg.n_chunks + by;

    if (t_start >= cfg.cur_len) {{
        if (lid.x == 0u) {{
            PMax[mi_a] = bitcast<f32>(0xFF800000u);
            PSum[mi_a] = 0.0;
            PMax[mi_b] = bitcast<f32>(0xFF800000u);
            PSum[mi_b] = 0.0;
        }}
        if (lid.x < D) {{
            POut[mi_a * D + lid.x] = 0.0;
            POut[mi_b * D + lid.x] = 0.0;
        }}
        return;
    }}
    let chunk_len = min(CHUNK, cfg.cur_len - t_start);

    // Stage 1 — both heads' scores from one K row read per key.  The body is
    // generated: the serial form walks one key per lane, the cooperative form
    // walks one key per 8-lane group (see `s1` above).
{s1}    workgroupBarrier();

    // Stage 2 — both chunk maxima, two independent 256-wide trees in one pass.
    var lmax_a = bitcast<f32>(0xFF800000u);
    var lmax_b = bitcast<f32>(0xFF800000u);
    for (var t = lid.x; t < chunk_len; t = t + BS) {{
        if (sc_a[t] > lmax_a) {{ lmax_a = sc_a[t]; }}
        if (sc_b[t] > lmax_b) {{ lmax_b = sc_b[t]; }}
    }}
    red_max[lid.x] = lmax_a;
    red_max[256u + lid.x] = lmax_b;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{
            red_max[lid.x] = max(red_max[lid.x], red_max[lid.x + s]);
            red_max[256u + lid.x] = max(red_max[256u + lid.x], red_max[256u + lid.x + s]);
        }}
        workgroupBarrier();
    }}
    let max_a = red_max[0];
    let max_b = red_max[256u];
    workgroupBarrier();

    // Stage 3 — exp + sum, same per-head tree shape.
    var lsum_a = 0.0;
    var lsum_b = 0.0;
    for (var t = lid.x; t < chunk_len; t = t + BS) {{
        let ea = expf_bt(sc_a[t] - max_a);
        let eb = expf_bt(sc_b[t] - max_b);
        sc_a[t] = ea;
        sc_b[t] = eb;
        lsum_a = lsum_a + ea;
        lsum_b = lsum_b + eb;
    }}
    red_sum[lid.x] = lsum_a;
    red_sum[256u + lid.x] = lsum_b;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{
            red_sum[lid.x] = red_sum[lid.x] + red_sum[lid.x + s];
            red_sum[256u + lid.x] = red_sum[256u + lid.x] + red_sum[256u + lid.x + s];
        }}
        workgroupBarrier();
    }}
    let sum_a = red_sum[0];
    let sum_b = red_sum[256u];
    workgroupBarrier();

    // Stage 4 — both numerators from one V element read per (key, dim pair).
    let jp = lid.x % D2;
    let t_idx = lid.x / D2;
    if (t_idx < T_SPLIT) {{
        var a0 = 0.0;
        var a1 = 0.0;
        var b0 = 0.0;
        var b1 = 0.0;
        for (var t = t_idx; t < chunk_len; t = t + T_SPLIT) {{
            let row = kbase + (t_start + t) * D2;
            let v = unpack2x16float(VC[row + jp]);
            a0 = a0 + sc_a[t] * v.x;
            a1 = a1 + sc_a[t] * v.y;
            b0 = b0 + sc_b[t] * v.x;
            b1 = b1 + sc_b[t] * v.y;
        }}
        pa[t_idx * D + jp * 2u] = a0;
        pa[t_idx * D + jp * 2u + 1u] = a1;
        pb[t_idx * D + jp * 2u] = b0;
        pb[t_idx * D + jp * 2u + 1u] = b1;
    }}
    workgroupBarrier();

    if (lid.x < D) {{
        var acc_a = 0.0;
        var acc_b = 0.0;
        for (var ti = 0u; ti < T_SPLIT; ti = ti + 1u) {{
            acc_a = acc_a + pa[ti * D + lid.x];
            acc_b = acc_b + pb[ti * D + lid.x];
        }}
        POut[mi_a * D + lid.x] = acc_a;
        POut[mi_b * D + lid.x] = acc_b;
    }}
    if (lid.x == 0u) {{
        PMax[mi_a] = max_a;
        PSum[mi_a] = sum_a;
        PMax[mi_b] = max_b;
        PSum[mi_b] = sum_b;
    }}
}}
",
        d2 = d / 2,
        d4 = d / 8,
        t_split = t_split,
        s1 = s1,
    )
}

/// `fused_gqa_decode_split_p2_f16` — merge phase: online-softmax correction
/// across chunks.  One workgroup per q_head, one thread per f16 word.
pub fn gqa_split_merge(d: usize) -> String {
    format!(
        "struct Cfg {{ n_chunks: u32, _a: u32, _b: u32, _c: u32 }};

{EXP_BT}
@group(0) @binding(0) var<storage, read_write> Out:  array<u32>;
@group(0) @binding(1) var<storage, read>       POut: array<f32>;
@group(0) @binding(2) var<storage, read>       PMax: array<f32>;
@group(0) @binding(3) var<storage, read>       PSum: array<f32>;
@group(0) @binding(4) var<uniform>             cfg:  Cfg;

const D: u32 = {d}u;
const D2: u32 = {d2}u;

@compute @workgroup_size({d})
fn gqa_merge(@builtin(workgroup_id) wgid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {{
    let qh = wgid.x;
    let n = cfg.n_chunks;
    let maxbase = qh * n;
    let neg_inf = bitcast<f32>(0xFF800000u);

    var g_max = neg_inf;
    for (var c = 0u; c < n; c = c + 1u) {{
        if (PMax[maxbase + c] > g_max) {{ g_max = PMax[maxbase + c]; }}
    }}
    var g_sum = 0.0;
    for (var c = 0u; c < n; c = c + 1u) {{
        if (PMax[maxbase + c] > neg_inf) {{
            g_sum = g_sum + PSum[maxbase + c] * expf_bt(PMax[maxbase + c] - g_max);
        }}
    }}
    let inv = 1.0 / g_sum;

    if (lid.x < D2) {{
        var a0 = 0.0;
        var a1 = 0.0;
        for (var c = 0u; c < n; c = c + 1u) {{
            if (PMax[maxbase + c] > neg_inf) {{
                let w = expf_bt(PMax[maxbase + c] - g_max) * inv;
                a0 = a0 + w * POut[(maxbase + c) * D + lid.x * 2u];
                a1 = a1 + w * POut[(maxbase + c) * D + lid.x * 2u + 1u];
            }}
        }}
        Out[qh * D2 + lid.x] = pack2x16float(vec2<f32>(a0, a1));
    }}
}}
",
        d2 = d / 2,
    )
}

/// `argmax_into_slot_f16` — single block of 1024, strict `>` so the lowest index
/// wins a tie (which matters for reproducibility).
pub fn argmax_into_slot() -> String {
    format!(
        "{HALF_AT}
struct Cfg {{ n: u32, slot: u32, _a: u32, _b: u32 }};

@group(0) @binding(0) var<storage, read>       X:   array<u32>;
@group(0) @binding(1) var<storage, read_write> Out: array<i32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;

const BS: u32 = 1024u;

var<workgroup> smax: array<f32, 1024>;
var<workgroup> sidx: array<i32, 1024>;

@compute @workgroup_size(1024)
fn argmax(@builtin(local_invocation_id) lid: vec3<u32>) {{
    let n2 = cfg.n >> 1u;
    var lmax = bitcast<f32>(0xFF800000u);
    var lidx = 0;
    for (var i = lid.x; i < n2; i = i + BS) {{
        let v = unpack2x16float(X[i]);
        let ix = i * 2u;
        if (v.x > lmax) {{ lmax = v.x; lidx = i32(ix); }}
        if (v.y > lmax) {{ lmax = v.y; lidx = i32(ix + 1u); }}
    }}
    if ((cfg.n & 1u) == 1u && lid.x == 0u) {{
        let e = half_at(unpack2x16float(X[cfg.n >> 1u]), cfg.n - 1u);
        if (e > lmax) {{ lmax = e; lidx = i32(cfg.n - 1u); }}
    }}
    smax[lid.x] = lmax;
    sidx[lid.x] = lidx;
    workgroupBarrier();
    for (var s = BS >> 1u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{
            if (smax[lid.x + s] > smax[lid.x]) {{
                smax[lid.x] = smax[lid.x + s];
                sidx[lid.x] = sidx[lid.x + s];
            }}
        }}
        workgroupBarrier();
    }}
    if (lid.x == 0u) {{ Out[cfg.slot] = sidx[0]; }}
}}
"
    )
}

/// `embed_lookup_single_i32_f16` — gather one embedding row using a token id that
/// never left the GPU.
pub fn embed_lookup_single() -> String {
    "struct Cfg { slot: u32, d2: u32, _a: u32, _b: u32 };

@group(0) @binding(0) var<storage, read>       Table: array<u32>;
@group(0) @binding(1) var<storage, read>       Ids:   array<i32>;
@group(0) @binding(2) var<storage, read_write> Out:   array<u32>;
@group(0) @binding(3) var<uniform>             cfg:   Cfg;

const BS: u32 = 1024u;

@compute @workgroup_size(1024)
fn embed(@builtin(local_invocation_id) lid: vec3<u32>) {
    let id = u32(Ids[cfg.slot]);
    let base = id * cfg.d2;
    for (var w = lid.x; w < cfg.d2; w = w + BS) {
        Out[w] = Table[base + w];
    }
}
"
    .to_string()
}

/// Format an f32 so WGSL parses back the exact same value (hex float, exact).
pub fn format_f32(v: f32) -> String {
    let bits = v.to_bits();
    let sign = if bits >> 31 == 1 { "-" } else { "" };
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x007F_FFFF;
    if exp == 0 && man == 0 {
        return format!("{sign}0.0");
    }
    let man = man << 1;
    let e = exp - 127;
    format!("{sign}0x1.{man:06x}p{e}")
}

/// Default 128×128 tile micro-geometry and K step, shared with callers.
pub const PREFILL_GEMM_TM: usize = 8;
pub const PREFILL_GEMM_TN: usize = 8;
pub const PREFILL_GEMM_BK: usize = 16;
pub const PREFILL_GEMM_BM: usize = 16 * PREFILL_GEMM_TM;
pub const PREFILL_GEMM_BN: usize = 16 * PREFILL_GEMM_TN;

/// Prefill GEMM family: `C[m,n] f16 = A[m,k] f16 × W[n,k]ᵀ` (or `× W[n,k]`
/// when `transb`), f32 accumulate, one f16 rounding.
///
/// `tile64` selects a 64×64 tile (16×16 threads, 4×4 micro-tile) for short
/// prefills on high-SM devices; the default 128×128 tile (8×8 micro-tile) is
/// unchanged.  `use_f16` keeps the shared operand tiles as `vec4<f16>` and
/// widens each read to `vec4<f32>` before the f32 FMAs.  Both forms have the
/// same accumulation order and epilogue, so the output is bit-identical.
///
/// `beta=1` folds the residual add into the epilogue; `bias=1` adds the
/// per-column `Bias` vector (binding 4); `causal_skip` / `causal_k` are the
/// two attention tile skips; `w8` reads an int8-resident W operand with f32
/// per-row scales (binding 4) and dequantizes to f16 exactly as the host does.
///
/// The kernel's own `BM`/`BN`/`BK` constants and every caller's padding and
/// dispatch grids must agree.  The 128×128 shape is exported as
/// [`PREFILL_GEMM_TM`]/[`PREFILL_GEMM_TN`]/[`PREFILL_GEMM_BM`]/[`PREFILL_GEMM_BN`].
pub fn prefill_gemm_cfg(
    transb: bool,
    beta: bool,
    bias: bool,
    causal_skip: bool,
    causal_k: bool,
    w8: bool,
    tile64: bool,
    use_f16: bool,
) -> String {
    prefill_gemm_impl(
        transb,
        beta,
        bias,
        causal_skip,
        causal_k,
        w8,
        tile64,
        use_f16,
    )
}

/// The shipped 128×128, f32-shared-tile form (backward-compatible entry point).
pub fn prefill_gemm(transb: bool, beta: bool) -> String {
    prefill_gemm_cfg(transb, beta, false, false, false, false, false, false)
}

/// [`prefill_gemm`] with an int8-resident W operand and per-row f32 scales.
pub fn prefill_gemm_w8(beta: bool) -> String {
    prefill_gemm_cfg(false, beta, false, false, false, true, false, false)
}

/// [`prefill_gemm`] with the causal score-tile skip.
pub fn prefill_gemm_causal() -> String {
    prefill_gemm_cfg(false, false, false, true, false, false, false, false)
}

/// [`prefill_gemm`] with the causal AV k bound.
pub fn prefill_gemm_causal_av() -> String {
    prefill_gemm_cfg(true, false, false, false, true, false, false, false)
}

/// [`prefill_gemm`] plus a per-column bias (binding 4).
pub fn prefill_gemm_bias(transb: bool, beta: bool) -> String {
    prefill_gemm_cfg(transb, beta, true, false, false, false, false, false)
}

/// 64×64 f32-shared-tile variant of [`prefill_gemm`].
pub fn prefill_gemm_64(transb: bool, beta: bool, use_f16: bool) -> String {
    prefill_gemm_cfg(transb, beta, false, false, false, false, true, use_f16)
}

/// 64×64 int8-weight variant of [`prefill_gemm_w8`].
pub fn prefill_gemm_w8_64(beta: bool, use_f16: bool) -> String {
    prefill_gemm_cfg(false, beta, false, false, false, true, true, use_f16)
}

/// 64×64 f16/f32-shared variant of [`prefill_gemm_causal`].
pub fn prefill_gemm_causal_64(use_f16: bool) -> String {
    prefill_gemm_cfg(false, false, false, true, false, false, true, use_f16)
}

/// 64×64 f16/f32-shared variant of [`prefill_gemm_causal_av`].
pub fn prefill_gemm_causal_av_64(use_f16: bool) -> String {
    prefill_gemm_cfg(true, false, false, false, true, false, true, use_f16)
}

/// 64×64 f16/f32-shared variant of [`prefill_gemm_bias`].
pub fn prefill_gemm_bias_64(transb: bool, beta: bool, use_f16: bool) -> String {
    prefill_gemm_cfg(transb, beta, true, false, false, false, true, use_f16)
}

// Keep the 128×128 constants above as the exported defaults; the 64×64 tile's
// geometry is derived in the generator (tm = tn = 4).

/// `QASR_GEMM_V4` -- the prefill/encoder GEMM's shared-memory tile: the default
/// (`1`) is the k-major 16-byte chunk tile, `0` restores the scalar per-(row, k)
/// tile with its row permutation.  Generator-side, read once per pipeline, so it
/// exists to re-run the A/B rather than to be set in production.  See the note in
/// [`prefill_gemm_impl`] for the measurements.
fn gemm_v4() -> bool {
    std::env::var("QASR_GEMM_V4")
        .map(|s| s != "0")
        .unwrap_or(true)
}

/// `QASR_GEMM_UNROLL` -- the k-unroll factor for the prefill/encoder GEMM.
/// `1` is the runtime q-loop (everything before `8x8DU*`), `4` is shipped;
/// anything not dividing `BK` falls back to `1`.  A generator-side switch, read
/// once per pipeline creation, so it exists to re-run the A/B, not to be set in
/// production.  See the unrolling note in [`prefill_gemm_impl`].
fn gemm_unroll() -> usize {
    std::env::var("QASR_GEMM_UNROLL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
}

fn prefill_gemm_impl(
    transb: bool,
    beta: bool,
    bias: bool,
    causal_skip: bool,
    causal_k: bool,
    w8: bool,
    tile64: bool,
    use_f16: bool,
) -> String {
    let tm = if tile64 { 4 } else { PREFILL_GEMM_TM };
    let tn = if tile64 { 4 } else { PREFILL_GEMM_TN };
    let bk = PREFILL_GEMM_BK;
    let bm = 16 * tm;
    let bn = 16 * tn;
    let pad = bk + 1;
    let n_as = bm * bk / 256;
    let n_bs = bn * bk / 256;
    let v4 = gemm_v4();
    let (aq, bq) = (bm / 4, bn / 4);
    let cstep = bm * bk / 4 / 2;
    let f16_tile = v4 && use_f16;
    if w8 {
        assert!(
            !transb && !bias && !causal_skip && !causal_k,
            "prefill_gemm_w8 supports only the plain weight operand (the attention GEMMs stay f16)"
        );
        assert!(
            v4,
            "prefill_gemm_w8 requires the v4 chunk tile (QASR_GEMM_V4 != 0)"
        );
    }
    assert!(tm == 4 || tm == 8, "unsupported GEMM micro-tile M");
    assert!(tn == 4 || tn == 8, "unsupported GEMM micro-tile N");
    assert_eq!(
        bk % 16,
        0,
        "the vec4 chunk store assumes 8 k per chunk half"
    );
    assert!(
        n_as == (if tile64 { 4 } else { 8 }),
        "unexpected v4 A tile shape"
    );
    assert!(
        n_bs == (if tile64 { 4 } else { 8 }),
        "unexpected v4 B tile shape"
    );

    let mut s = String::new();
    if f16_tile {
        s.push_str("enable f16;\n");
    }
    s.push_str(
        "struct GDims { m: u32, n: u32, k: u32, ldc: u32, bsa: u32, bsb: u32, bsc: u32, beta: u32, row0: u32, lda: u32, q_off: u32, gqa_group: u32 };\n\
         @group(0) @binding(0) var<storage, read>       A: array<u32>;\n\
         @group(0) @binding(1) var<storage, read>       W: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> C: array<u32>;\n\
         @group(0) @binding(3) var<uniform>             gd: GDims;\n",
    );
    if bias {
        s.push_str("@group(0) @binding(4) var<storage, read> Bias: array<u32>;\n");
    }
    if w8 {
        s.push_str("@group(0) @binding(4) var<storage, read> S8: array<f32>;\n");
    }
    s.push_str(&format!(
        "const BM: u32 = {bm}u;\nconst BN: u32 = {bn}u;\nconst BK: u32 = {bk}u;\n\
         const PAD: u32 = {pad}u;\nconst TM: u32 = {tm}u;\nconst TN: u32 = {tn}u;\n\
         const AQ: u32 = {aq}u;\nconst BQ: u32 = {bq}u;\nconst CSTEP: u32 = {cstep}u;\n\
         const TRANSB: u32 = {}u;\nconst BETA: u32 = {}u;\nconst BIAS: u32 = {}u;\n\
         const CAUSAL: u32 = {}u;\nconst CAUSAL_K: u32 = {}u;\n",
        u32::from(transb),
        u32::from(beta),
        u32::from(bias),
        u32::from(causal_skip),
        u32::from(causal_k),
    ));
    if v4 {
        let ty = if f16_tile { "f16" } else { "f32" };
        s.push_str(&format!(
            "var<workgroup> A4: array<vec4<{ty}>, {}>;\n",
            bm * bk / 4
        ));
        s.push_str(&format!(
            "var<workgroup> B4: array<vec4<{ty}>, {}>;\n",
            bn * bk / 4
        ));
    } else {
        s.push_str(&format!("var<workgroup> As: array<f32, {}>;\n", bm * pad));
        s.push_str(&format!("var<workgroup> Bs: array<f32, {}>;\n", bn * pad));
    }
    if f16_tile {
        s.push_str(
            "fn halve_f16(w: u32, odd: bool) -> f16 {\n\
             \x20 let p = bitcast<vec2<f16>>(w);\n\
             \x20 return select(p.x, p.y, odd);\n\
             }\n",
        );
    } else {
        s.push_str(
            "fn halve(w: u32, odd: bool) -> f32 {\n\
             \x20 let p = unpack2x16float(w);\n\
             \x20 return select(p.x, p.y, odd);\n\
             }\n",
        );
    }
    if w8 {
        if f16_tile {
            s.push_str(
                "fn q8(w: u32, sh: u32) -> f32 {\n\
                 \x20 return f32((bitcast<i32>(w << sh)) >> 24u);\n\
                 }\n\
                 fn wq_f16(w: u32, by: u32, sc: f32) -> f16 {\n\
                 \x20 return f16(q8(w, 24u - 8u * by) * sc);\n\
                 }\n",
            );
        } else {
            s.push_str(
                "fn q8(w: u32, sh: u32) -> f32 {\n\
                 \x20 return f32((bitcast<i32>(w << sh)) >> 24u);\n\
                 }\n\
                 fn wq(w: u32, by: u32, sc: f32) -> f32 {\n\
                 \x20 return unpack2x16float(pack2x16float(vec2<f32>(q8(w, 24u - 8u * by) * sc, 0.0))).x;\n\
                 }\n",
            );
        }
    }

    s.push_str(&format!(
        "@compute @workgroup_size(16, 16)\n\
         fn gemm(@builtin(workgroup_id) wid: vec3<u32>,\n\
                 @builtin(local_invocation_id) lid: vec3<u32>) {{\n\
         let tx = lid.x;\n let ty = lid.y;\n\
         let m0 = wid.y * BM;\n let n0 = wid.x * BN;\n let kk = gd.k / {kkd}u;\n\
         let alda = gd.lda / 2u;\n\
         let abase = wid.z * gd.bsa;\n\
         let wb = (wid.z / max(gd.gqa_group, 1u)) * gd.bsb;\n\
         let cbase = wid.z * gd.bsc;\n\
         if (CAUSAL == 1u && n0 + gd.row0 > gd.q_off + m0 + BM - 1u) {{ return; }}\n\
         let klim = select(gd.k, min(gd.k, max(gd.q_off + m0 + BM, gd.row0) - gd.row0), CAUSAL_K == 1u);\n",
        kkd = if w8 { 4 } else { 2 },
    ));
    if v4 {
        s.push_str(" let c0 = ty * 16u + tx;\n");
        if tile64 {
            // One chunk per thread: c0's 16 k-columns and 16 four-row groups
            // cover the whole BK×BN tile with no second half.
            s.push_str(
                " let crg = c0 % AQ;\n let ck = c0 / AQ;\n\
                 \x20 let cvodd = ((ck & 1u) == 1u);\n\
                 \x20 let bop = c0 % BQ;\n",
            );
            if w8 {
                s.push_str(
                    " let brow = gd.row0 + n0 + 4u * bop;\n\
                     \x20 let s0 = S8[brow]; let s1 = S8[brow + 1u]; let s2 = S8[brow + 2u]; let s3 = S8[brow + 3u];\n\
                     \x20 let by = ck % 4u;\n",
                );
            }
        } else {
            s.push_str(
                " let crg = c0 % AQ;\n let ck = c0 / AQ;\n\
                 \x20 let cvodd = ((ck & 1u) == 1u);\n\
                 \x20 let bop = c0 % BQ;\n\
                 \x20 let bog = ((bop >> 4u) << 3u) | (bop & 7u);\n\
                 \x20 let bos = (bop >> 3u) & 1u;\n",
            );
            if w8 {
                s.push_str(
                    " let brow = gd.row0 + n0 + 8u * bog + 4u * bos;\n\
                     \x20 let s0 = S8[brow]; let s1 = S8[brow + 1u]; let s2 = S8[brow + 2u]; let s3 = S8[brow + 3u];\n\
                     \x20 let by = ck % 4u;\n",
                );
            }
        }
    }

    for i in 0..tm {
        for j in 0..tn {
            s.push_str(&format!("var c{i}{j}: f32 = 0.0;\n"));
        }
    }

    let vec = |parts: &[String]| -> String {
        format!(
            "vec4<{}>({})",
            if f16_tile { "f16" } else { "f32" },
            parts.join(",")
        )
    };
    let halve_call = |word: &str, odd: String| {
        if f16_tile {
            format!("halve_f16({word}, {odd})")
        } else {
            format!("halve({word}, {odd})")
        }
    };
    let wq_call = |word: &str, e: usize| {
        if f16_tile {
            format!("wq_f16({word}, by, s{e})")
        } else {
            format!("wq({word}, by, s{e})")
        }
    };
    let a4_call = |suffix: &str, comps: &[String]| format!("A4[c0{suffix}] = {};\n", vec(comps));
    let b4_call = |suffix: &str, comps: &[String]| format!("B4[c0{suffix}] = {};\n", vec(comps));

    // Store expressions for A and B, for both the initial load and the
    // software-pipelined prefetch (the latter reads `pfa*`/`pfb*` registers).
    let a_group = |e: usize| -> String { format!("4u * crg + {e}u") };
    let b_rows = |e: usize| -> String {
        if tile64 {
            format!("4u * bop + {e}u")
        } else {
            format!("8u * bog + 4u * bos + {e}u")
        }
    };
    let transb_wo = if tile64 {
        "(n0 + 4u * bop) / 2u".to_string()
    } else {
        "(n0 + 8u * bog + 4u * bos) / 2u".to_string()
    };

    let load_as = |kx: &str, pf: bool| -> String {
        let mut t = String::new();
        if v4 {
            let halves: Vec<(String, String)> = if tile64 {
                vec![(String::new(), "0u".to_string())]
            } else {
                vec![
                    (String::new(), "0u".to_string()),
                    (" + CSTEP".into(), "8u".to_string()),
                ]
            };
            for (idx, (suffix, kadd)) in halves.iter().enumerate() {
                let mut comps = Vec::new();
                for e in 0..4 {
                    let (word, odd) = if pf {
                        (format!("pfa{}", idx * 4 + e), "cvodd".to_string())
                    } else {
                        (
                            format!(
                                "A[abase + (m0 + {}) * alda + ({kx} + ck + {kadd}) / 2u]",
                                a_group(e)
                            ),
                            "cvodd".to_string(),
                        )
                    };
                    comps.push(halve_call(&word, odd));
                }
                t.push_str(&a4_call(suffix, &comps));
            }
            return t;
        }
        for e in 0..n_as {
            let (word, odd) = if pf {
                (format!("pfa{e}"), "(tx & 1u) == 1u".to_string())
            } else {
                (
                    format!(
                        "A[abase + (m0 + ty + {}u) * alda + ({kx} + tx) / 2u]",
                        e * 16
                    ),
                    "((tx & 1u) == 1u)".to_string(),
                )
            };
            t.push_str(&format!(
                "  As[(ty + {}u) * PAD + tx] = {};\n",
                e * 16,
                halve_call(&word, odd)
            ));
        }
        t
    };
    let load_bs = |kx: &str, pf: bool| -> String {
        let mut t = String::new();
        if v4 {
            let halves: Vec<(String, String)> = if tile64 {
                vec![(String::new(), "0u".to_string())]
            } else {
                vec![
                    (String::new(), "0u".to_string()),
                    (" + CSTEP".into(), "8u".to_string()),
                ]
            };
            for (idx, (suffix, kadd)) in halves.iter().enumerate() {
                if transb {
                    let base = format!("wb + (gd.row0 + {kx} + ck + {kadd}) * (gd.n / 2u)");
                    let w0 = if pf {
                        format!("pfb{}", idx * 2)
                    } else {
                        format!("W[{base} + {transb_wo}]")
                    };
                    let w1 = if pf {
                        format!("pfb{}", idx * 2 + 1)
                    } else {
                        format!("W[{base} + {transb_wo} + 1u]")
                    };
                    if f16_tile {
                        let scope = if pf { "p" } else { "i" };
                        t.push_str(&format!(
                            "let tp_{scope}_{idx}_0 = bitcast<vec2<f16>>({w0}); let tp_{scope}_{idx}_1 = bitcast<vec2<f16>>({w1}); B4[c0{suffix}] = vec4<f16>(tp_{scope}_{idx}_0.x, tp_{scope}_{idx}_0.y, tp_{scope}_{idx}_1.x, tp_{scope}_{idx}_1.y);\n"
                        ));
                        continue;
                    }
                    let comps = vec![
                        halve_call(&w0, "false".to_string()),
                        halve_call(&w0, "true".to_string()),
                        halve_call(&w1, "false".to_string()),
                        halve_call(&w1, "true".to_string()),
                    ];
                    t.push_str(&b4_call(suffix, &comps));
                    continue;
                }
                if w8 {
                    let mut comps = Vec::new();
                    for e in 0..4 {
                        let word = if pf {
                            format!("pfb{}", idx * 4 + e)
                        } else {
                            format!(
                                "W[wb + (gd.row0 + n0 + {}) * kk + ({kx} + ck + {kadd}) / 4u]",
                                b_rows(e)
                            )
                        };
                        comps.push(wq_call(&word, e));
                    }
                    t.push_str(&b4_call(suffix, &comps));
                    continue;
                }
                let mut comps = Vec::new();
                for e in 0..4 {
                    let (word, odd) = if pf {
                        (format!("pfb{}", idx * 4 + e), "cvodd".to_string())
                    } else {
                        (
                            format!(
                                "W[wb + (gd.row0 + n0 + {}) * kk + ({kx} + ck + {kadd}) / 2u]",
                                b_rows(e)
                            ),
                            "cvodd".to_string(),
                        )
                    };
                    comps.push(halve_call(&word, odd));
                }
                t.push_str(&b4_call(suffix, &comps));
            }
            return t;
        }
        for e in 0..n_bs {
            let slot = format!(
                "(16u * ((ty + {}u) % {tn}u) + (ty + {}u) / {tn}u)",
                e * 16,
                e * 16
            );
            let (word, odd) = if pf {
                (
                    format!("pfb{e}"),
                    if transb {
                        format!("((n0 + ty + {}u) & 1u) == 1u", e * 16)
                    } else {
                        "(tx & 1u) == 1u".to_string()
                    },
                )
            } else if transb {
                (
                    format!(
                        "W[wb + (gd.row0 + {kx} + tx) * (gd.n / 2u) + (n0 + ty + {}u) / 2u]",
                        e * 16
                    ),
                    format!("((n0 + ty + {}u) & 1u) == 1u", e * 16),
                )
            } else {
                (
                    format!(
                        "W[wb + (gd.row0 + n0 + ty + {}u) * kk + ({kx} + tx) / 2u]",
                        e * 16
                    ),
                    "(tx & 1u) == 1u".to_string(),
                )
            };
            t.push_str(&format!(
                "  Bs[{slot} * PAD + tx] = {};\n",
                halve_call(&word, odd)
            ));
        }
        t
    };
    let pf_decl = || {
        let mut t = String::new();
        for e in 0..n_as {
            t.push_str(&format!("  var pfa{e}: u32 = 0u;\n"));
        }
        for e in 0..n_bs {
            t.push_str(&format!("  var pfb{e}: u32 = 0u;\n"));
        }
        t
    };
    let pf_as = |kx: &str| -> String {
        let mut t = String::new();
        if v4 {
            let halves: Vec<u32> = if tile64 { vec![0] } else { vec![0, 8] };
            for (idx, hadd) in halves.iter().enumerate() {
                for e in 0..4 {
                    t.push_str(&format!(
                        "   pfa{} = A[abase + (m0 + {}) * alda + ({kx} + ck + {hadd}u) / 2u];\n",
                        idx * 4 + e,
                        a_group(e)
                    ));
                }
            }
            return t;
        }
        for e in 0..n_as {
            t.push_str(&format!(
                "   pfa{e} = A[abase + (m0 + ty + {}u) * alda + ({kx} + tx) / 2u];\n",
                e * 16
            ));
        }
        t
    };
    let pf_bs = |kx: &str| -> String {
        let mut t = String::new();
        if v4 {
            let halves: Vec<u32> = if tile64 { vec![0] } else { vec![0, 8] };
            for (idx, hadd) in halves.iter().enumerate() {
                if transb {
                    let base = format!("wb + (gd.row0 + {kx} + ck + {hadd}u) * (gd.n / 2u)");
                    for w in 0..2 {
                        t.push_str(&format!(
                            "   pfb{} = W[{base} + {transb_wo}{}];\n",
                            idx * 2 + w,
                            if w == 0 {
                                String::new()
                            } else {
                                " + 1u".into()
                            }
                        ));
                    }
                } else if w8 {
                    for e in 0..4 {
                        t.push_str(&format!(
                            "   pfb{} = W[wb + (gd.row0 + n0 + {}) * kk + ({kx} + ck + {hadd}u) / 4u];\n",
                            idx * 4 + e,
                            b_rows(e)
                        ));
                    }
                } else {
                    for e in 0..4 {
                        t.push_str(&format!(
                            "   pfb{} = W[wb + (gd.row0 + n0 + {}) * kk + ({kx} + ck + {hadd}u) / 2u];\n",
                            idx * 4 + e,
                            b_rows(e)
                        ));
                    }
                }
            }
            return t;
        }
        for e in 0..n_bs {
            if transb {
                t.push_str(&format!(
                    "   pfb{e} = W[wb + (gd.row0 + {kx} + tx) * (gd.n / 2u) + (n0 + ty + {}u) / 2u];\n", e * 16
                ));
            } else {
                t.push_str(&format!(
                    "   pfb{e} = W[wb + (gd.row0 + n0 + ty + {}u) * kk + ({kx} + tx) / 2u];\n",
                    e * 16
                ));
            }
        }
        t
    };

    // ---- k-step unrolling -------------------------------------------------
    let mut unroll_q = gemm_unroll();
    if unroll_q < 2 {
        unroll_q = 1;
    } else if bk % unroll_q != 0 {
        eprintln!("QASR_GEMM_UNROLL={unroll_q} does not divide BK={bk}; using 1");
        unroll_q = 1;
    }
    let compute = || {
        let mut t = String::new();
        let v4_read = |t: &mut String, idx: &str, tag: &str| {
            if tile64 {
                t.push_str(&format!(
                    "   let av{tag} = vec4<f32>(A4[{idx} * AQ + ty]);\n"
                ));
                t.push_str(&format!(
                    "   let bv{tag} = vec4<f32>(B4[{idx} * BQ + tx]);\n"
                ));
            } else {
                t.push_str(&format!(
                    "   let av{tag} = vec4<f32>(A4[{idx} * AQ + ty * 2u]);\n"
                ));
                t.push_str(&format!(
                    "   let aw{tag} = vec4<f32>(A4[{idx} * AQ + ty * 2u + 1u]);\n"
                ));
                t.push_str(&format!(
                    "   let bv{tag} = vec4<f32>(B4[{idx} * BQ + (tx & 7u) + 16u * (tx >> 3u)]);\n"
                ));
                t.push_str(&format!(
                    "   let bw{tag} = vec4<f32>(B4[{idx} * BQ + (tx & 7u) + 8u + 16u * (tx >> 3u)]);\n"
                ));
            }
        };
        let v4_fma = |t: &mut String, tag: &str| {
            for i in 0..tm {
                let a = if i < 4 || tile64 {
                    format!("av{tag}[{i}]")
                } else {
                    format!("aw{tag}[{}]", i - 4)
                };
                for j in 0..tn {
                    let b = if j < 4 || tile64 {
                        format!("bv{tag}[{j}]")
                    } else {
                        format!("bw{tag}[{}]", j - 4)
                    };
                    t.push_str(&format!("   c{i}{j} = c{i}{j} + {a} * {b};\n"));
                }
            }
        };
        if unroll_q == 1 {
            t.push_str("  var q: u32 = 0u;\n  loop {\n   if (q >= BK) { break; }\n");
            if v4 {
                v4_read(&mut t, "q", "");
                v4_fma(&mut t, "");
            } else {
                for i in 0..tm {
                    t.push_str(&format!(
                        "   let a{i} = As[(ty * {tm}u + {i}u) * PAD + q];\n"
                    ));
                }
                for j in 0..tn {
                    t.push_str(&format!("   let b{j} = Bs[(16u * {j}u + tx) * PAD + q];\n"));
                }
                for i in 0..tm {
                    for j in 0..tn {
                        t.push_str(&format!("   c{i}{j} = c{i}{j} + a{i} * b{j};\n"));
                    }
                }
            }
            t.push_str("   q = q + 1u;\n  }\n");
            return t;
        }
        t.push_str("  var q0: u32 = 0u;\n  loop {\n   if (q0 >= BK) { break; }\n");
        for u in 0..unroll_q {
            if v4 {
                v4_read(&mut t, &format!("(q0 + {u}u)"), &format!("_{u}"));
                v4_fma(&mut t, &format!("_{u}"));
                continue;
            }
            for i in 0..tm {
                t.push_str(&format!(
                    "   let a{i}_{u} = As[(ty * {tm}u + {i}u) * PAD + (q0 + {u}u)];\n"
                ));
            }
            for j in 0..tn {
                t.push_str(&format!(
                    "   let b{j}_{u} = Bs[(16u * {j}u + tx) * PAD + (q0 + {u}u)];\n"
                ));
            }
            for i in 0..tm {
                for j in 0..tn {
                    t.push_str(&format!("   c{i}{j} = c{i}{j} + a{i}_{u} * b{j}_{u};\n"));
                }
            }
        }
        t.push_str(&format!("   q0 = q0 + {unroll_q}u;\n  }}\n"));
        t
    };

    s.push_str(" var k0: u32 = 0u;\n");
    s.push_str(&load_as("k0", false));
    s.push_str(&load_bs("k0", false));
    s.push_str(" workgroupBarrier();\n");
    s.push_str(" loop {\n  if (k0 >= klim) { break; }\n");
    s.push_str("  let kn = k0 + BK;\n");
    s.push_str(&pf_decl());
    s.push_str("  if (kn < klim) {\n");
    s.push_str(&pf_as("kn"));
    s.push_str(&pf_bs("kn"));
    s.push_str("  }\n");
    s.push_str(&compute());
    s.push_str("  workgroupBarrier();\n");
    s.push_str("  if (kn < klim) {\n");
    s.push_str(&load_as("kn", true));
    s.push_str(&load_bs("kn", true));
    s.push_str("  }\n  workgroupBarrier();\n");
    s.push_str("  k0 = kn;\n }\n");

    for i in 0..tm {
        s.push_str(&format!("let row{i} = m0 + ty * {tm}u + {i}u;\n"));
    }
    for i in 0..tm {
        for e in 0..tn / 2 {
            let je = 2 * e;
            let jo = je + 1;
            s.push_str(&format!(
                "let we{i}_{e} = (cbase + row{i} * gd.ldc + n0 + tx * {tn}u + {je}u) / 2u;\n"
            ));
            s.push_str(&format!("var v{i}{e} = vec2<f32>(c{i}{je}, c{i}{jo});\n"));
            if bias {
                s.push_str(&format!(
                    "v{i}{e} = v{i}{e} + unpack2x16float(Bias[(n0 + tx * {tn}u + {je}u) / 2u]);\n"
                ));
            }
            s.push_str(&format!(
                "if (BETA == 1u) {{\n  let old = unpack2x16float(C[we{i}_{e}]);\n  v{i}{e} = v{i}{e} + old;\n}}\n"
            ));
            s.push_str(&format!("C[we{i}_{e}] = pack2x16float(v{i}{e});\n"));
        }
    }
    s.push_str("}\n");
    s
}

/// Causal scaled softmax over prefill scores — port of
/// `softmax_scaled_causal_f16`.  One workgroup per score row; block size `bs`
/// matches `block_for_reduction(n)` so the reduction trees line up.
/// Row `p` (of the head) attends `min(p + 1 - row0, valid)` positions; columns
/// `valid..n_w` are written zero so downstream GEMMs read zeros.
///
/// `row0` is the column offset of the score block this dispatch covers (0 = the
/// whole row, the flat path): the row index is still absolute, so the causal
/// bound is `p + 1 - row0` clamped at zero.  With `row0 = 0` the bound collapses
/// to `min(p + 1, valid)` — the flat path is untouched.
/// The flat causal softmax at a **physical** workgroup width of `bp` computing a
/// **logical** reduction tree of width `bt` (`bt >= bp`, `bt % bp == 0`).
///
/// `bt` is the width the arithmetic is defined at -- it sets both the
/// element-to-lane map (`lane l takes elements l, l+bt, ...`) and the tree's
/// operand pairing.  `bp` only decides how many threads share the work: each
/// thread emulates `r = bt/bp` lanes and folds the tree's first `log2(r)` levels
/// *inside itself*, in the tree's own operand order, so the value that reaches
/// `red_max[t]` / `red_sum[t]` is the same f32 the wider tree would have built
/// from those lanes.
///
/// The reason to want that: at `bt = 1024` a workgroup is one per SM
/// (`1024 x ~40 registers`), so the row's 16-barrier tree plus its two cold row
/// reads run only 15 rows at a time.  That was 80 ms of a 90 s_en prefill when it
/// was priced, 56 of which the `QASR_SM_BS` physical-width change collected.
/// Capping the block *changes the tree* and the gate caught it
/// (`1.7B / 180s_zh` MISMATCH at r22); this changes only the thread count, so the
/// two are not the same experiment and only one of them is safe.
///
/// The max fold is order-free (max is exact); the sum fold must place the
/// parentheses exactly where the tree would: level 1 pairs lane `k` with lane
/// `k + r/2`, level 2 with `k + r/4`, and so on, each as `left + right`.
/// Compatibility wrapper: the legacy 32-lane shuffle / shared-memory tree.
pub fn softmax_causal_tree(bp: usize, bt: usize, subgroup: bool) -> String {
    softmax_causal_tree_legacy(bp, bt, subgroup)
}

/// Select the causal-softmax reduction path from device capabilities.
pub fn softmax_causal_tree_mode(bp: usize, bt: usize, mode: SubgroupMode) -> String {
    match mode {
        SubgroupMode::Warp32 => softmax_causal_tree_legacy(bp, bt, true),
        SubgroupMode::None => softmax_causal_tree_legacy(bp, bt, false),
        SubgroupMode::Warp64 | SubgroupMode::Generic => softmax_causal_tree_generic(bp, bt),
    }
}

/// Portable `subgroupMax` / `subgroupAdd` reduction for non-32-lane adapters.
///
/// Every subgroup builtin and barrier sits at the top level of the entry point
/// (or in a workgroup-uniform branch), so Naga's uniformity analysis accepts it.
/// The arithmetic that feeds the reductions — the lane loop, the `r = bt / bp`
/// fold and the causal bound — is the same as the legacy tree's.
fn softmax_causal_tree_generic(bp: usize, bt: usize) -> String {
    assert!(
        bt >= bp && bt % bp == 0,
        "softmax_causal_tree: bt must be a multiple of bp"
    );
    let r = bt / bp;
    assert!(
        r.is_power_of_two(),
        "softmax_causal_tree: the emulation factor must be a power of two"
    );
    assert!(
        bp >= 32,
        "softmax_causal_tree: the shuffle tail needs 32 lanes"
    );
    let n_sg = "(BP + SG - 1u) / SG";

    let mut max_accs = String::new();
    let mut max_body = String::new();
    let mut sum_accs = String::new();
    let mut sum_body = String::new();
    if r == 1 {
        max_body.push_str("    var lmax = bitcast<f32>(0xFF800000u);\n    for (var j = lid.x; j < valid; j = j + BT) {\n        let v = half_at(unpack2x16float(X[base_x + (j >> 1u)]), j);\n        let sc = v * scale;\n        if (sc > lmax) { lmax = sc; }\n    }\n");
        sum_body.push_str("    var lsum = 0.0;\n    for (var j = lid.x; j < valid; j = j + BT) {\n        let v = half_at(unpack2x16float(X[base_x + (j >> 1u)]), j);\n        lsum = lsum + exp(v * scale - row_max);\n    }\n");
    } else {
        for k in 0..r {
            max_accs.push_str(&format!("    var m{k} = bitcast<f32>(0xFF800000u);\n"));
            sum_accs.push_str(&format!("    var s{k} = 0.0;\n"));
            max_body.push_str(&format!(
                "        let j{k} = base + lid.x + {}u;\n        if (j{k} < valid) {{\n            let v{k} = half_at(unpack2x16float(X[base_x + (j{k} >> 1u)]), j{k});\n            m{k} = max(m{k}, v{k} * scale);\n        }}\n",
                k * bp
            ));
            sum_body.push_str(&format!(
                "        let j{k} = base + lid.x + {}u;\n        if (j{k} < valid) {{\n            let v{k} = half_at(unpack2x16float(X[base_x + (j{k} >> 1u)]), j{k});\n            s{k} = s{k} + exp(v{k} * scale - row_max);\n        }}\n",
                k * bp
            ));
        }
        let fold = |pre: &str, op: &str| -> String {
            let mut cur: Vec<String> = (0..r).map(|k| format!("{pre}{k}")).collect();
            while cur.len() > 1 {
                let half = cur.len() / 2;
                let mut next = Vec::with_capacity(half);
                for i in 0..half {
                    next.push(if op == "max" {
                        format!("max({}, {})", cur[i], cur[i + half])
                    } else {
                        format!("({} + {})", cur[i], cur[i + half])
                    });
                }
                cur = next;
            }
            cur.pop().unwrap()
        };
        max_body.push_str(&format!("    }}\n    let lmax = {};\n", fold("m", "max")));
        sum_body.push_str(&format!("    }}\n    let lsum = {};\n", fold("s", "sum")));
        // The `for` opening is emitted here so the closing brace above lands
        // in the right scope; the r==1 path has no fold loop.
        max_body =
            format!("    for (var base = 0u; base < valid; base = base + BT) {{\n{max_body}");
        sum_body =
            format!("    for (var base = 0u; base < valid; base = base + BT) {{\n{sum_body}");
    }

    format!(
        "struct Cfg {{ n_w: u32, n_x: u32, valid: u32, m: u32, mp: u32, scale: f32, gx: u32, row0: u32, q_off: u32 }};

@group(0) @binding(0) var<storage, read>       X:   array<u32>;
@group(0) @binding(1) var<storage, read_write> Out: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;

const BP: u32 = {bp}u;
const BT: u32 = {bt}u;

{HALF_AT}
var<workgroup> sg_red: array<f32, {bp}>;
var<workgroup> pub_red: array<f32, 1>;

@compute @workgroup_size({bp})
fn softmax(@builtin(workgroup_id) wgid: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(subgroup_invocation_id) sg_lid: u32,
           @builtin(subgroup_size) sg_size: u32) {{
    let row = wgid.x + wgid.y * cfg.gx;
    let head = row / cfg.m;
    let pos = row % cfg.m;
    let base_x = (head * cfg.mp + pos) * cfg.n_x;
    let base_o = (head * cfg.mp + pos) * cfg.n_w;
    let row_in_head = pos;
    let valid = min(cfg.valid, max(row_in_head + 1u + cfg.q_off, cfg.row0) - cfg.row0);
    let scale = cfg.scale;

    if (valid == 0u) {{
        for (var w = lid.x; w < cfg.n_w; w = w + BP) {{
            Out[base_o + w] = 0u;
        }}
        return;
    }}

    let SG: u32 = sg_size;
    let sg_id = lid.x / sg_size;
{max_accs}{max_body}    let sg_max = subgroupMax(lmax);
    if (sg_lid == 0u) {{ sg_red[sg_id] = sg_max; }}
    workgroupBarrier();
    if (lid.x == 0u) {{
        var gm = sg_red[0];
        for (var k = 1u; k < {n_sg}; k = k + 1u) {{ gm = max(gm, sg_red[k]); }}
        pub_red[0] = gm;
    }}
    workgroupBarrier();
    let row_max = pub_red[0];
    workgroupBarrier();

{sum_accs}{sum_body}    let sg_sum = subgroupAdd(lsum);
    if (sg_lid == 0u) {{ sg_red[sg_id] = sg_sum; }}
    workgroupBarrier();
    if (lid.x == 0u) {{
        var gs = 0.0;
        for (var k = 0u; k < {n_sg}; k = k + 1u) {{ gs = gs + sg_red[k]; }}
        pub_red[0] = 1.0 / gs;
    }}
    workgroupBarrier();
    let inv_sum = pub_red[0];
    workgroupBarrier();

    for (var w = lid.x; w < cfg.n_w; w = w + BP) {{
        let j0 = w * 2u;
        let j1 = j0 + 1u;
        var e0 = 0.0;
        var e1 = 0.0;
        if (j0 < valid) {{
            let v0 = half_at(unpack2x16float(X[base_x + w]), j0);
            e0 = exp(v0 * scale - row_max) * inv_sum;
        }}
        if (j1 < valid) {{
            let v1 = half_at(unpack2x16float(X[base_x + w]), j1);
            e1 = exp(v1 * scale - row_max) * inv_sum;
        }}
        Out[base_o + w] = pack2x16float(vec2<f32>(e0, e1));
    }}
}}
",
        bp = bp,
        bt = bt,
        HALF_AT = HALF_AT,
        n_sg = n_sg,
        max_accs = max_accs,
        max_body = max_body,
        sum_accs = sum_accs,
        sum_body = sum_body,
    )
}

fn softmax_causal_tree_legacy(bp: usize, bt: usize, subgroup: bool) -> String {
    assert!(
        bt >= bp && bt % bp == 0,
        "softmax_causal_tree: bt must be a multiple of bp"
    );
    let r = bt / bp;
    assert!(
        r.is_power_of_two(),
        "softmax_causal_tree: the emulation factor must be a power of two"
    );
    assert!(
        bp >= 32,
        "softmax_causal_tree: the shuffle tail needs 32 lanes"
    );
    let (max_tail, sum_tail) = if subgroup {
        (
            "    if (lid.x < 32u) {\n\
             \x20       var t = red_max[lid.x];\n\
             \x20       t = max(t, subgroupShuffleXor(t, 16u));\n\
             \x20       t = max(t, subgroupShuffleXor(t, 8u));\n\
             \x20       t = max(t, subgroupShuffleXor(t, 4u));\n\
             \x20       t = max(t, subgroupShuffleXor(t, 2u));\n\
             \x20       t = max(t, subgroupShuffleXor(t, 1u));\n\
             \x20       red_max[lid.x] = t;\n\
             \x20   }\n"
                .to_string(),
            "    if (lid.x < 32u) {\n\
             \x20       var t = red_sum[lid.x];\n\
             \x20       t = t + subgroupShuffleXor(t, 16u);\n\
             \x20       t = t + subgroupShuffleXor(t, 8u);\n\
             \x20       t = t + subgroupShuffleXor(t, 4u);\n\
             \x20       t = t + subgroupShuffleXor(t, 2u);\n\
             \x20       t = t + subgroupShuffleXor(t, 1u);\n\
             \x20       red_sum[lid.x] = t;\n\
             \x20   }\n"
                .to_string(),
        )
    } else {
        let keep = "    for (var sh = 16u; sh > 0u; sh = sh >> 1u) {\n\
                    \x20   if (lid.x < sh) {{ red_{r}[lid.x] = {op}; }}\n\
                    \x20   workgroupBarrier();\n\
                    \x20   }\n";
        (
            keep.replace("{r}", "max")
                .replace("{op}", "max(red_max[lid.x], red_max[lid.x + sh])"),
            keep.replace("{r}", "sum")
                .replace("{op}", "red_sum[lid.x] + red_sum[lid.x + sh]"),
        )
    };
    // The two row passes.  At `r == 1` this is the original one element per
    // thread per iteration, emitted unchanged; above it each thread walks its `r`
    // lanes together, one element each per iteration, so a lane still sees its
    // elements in ascending order and the fold below is the tree's own first
    // levels.  Nothing about the values or their order depends on `bp`.
    let lanes = |op: &str| -> String {
        if r == 1 {
            return if op == "max" {
                "    var lmax = bitcast<f32>(0xFF800000u);\n\
                 \x20   for (var j = lid.x; j < valid; j = j + {bt}u) {\n\
                 \x20       let v = half_at(unpack2x16float(X[base_x + (j >> 1u)]), j);\n\
                 \x20       let sc = v * scale;\n\
                 \x20       if (sc > lmax) { lmax = sc; }\n\
                 \x20   }\n\
                 \x20   red_max[lid.x] = lmax;\n"
                    .replace("{bt}", &bt.to_string())
            } else {
                "    var lsum = 0.0;\n\
                 \x20   for (var j = lid.x; j < valid; j = j + {bt}u) {\n\
                 \x20       let v = half_at(unpack2x16float(X[base_x + (j >> 1u)]), j);\n\
                 \x20       lsum = lsum + exp(v * scale - row_max);\n\
                 \x20   }\n\
                 \x20   red_sum[lid.x] = lsum;\n"
                    .replace("{bt}", &bt.to_string())
            };
        }
        let init = if op == "max" {
            "bitcast<f32>(0xFF800000u)"
        } else {
            "0.0"
        };
        // The two passes share a function scope, so the accumulators need
        // different names (m* and s*, not both *).
        let pre = if op == "max" { "m" } else { "s" };
        let mut accl = String::new();
        for k in 0..r {
            accl.push_str(&format!("    var {pre}{k} = {init};\n"));
        }
        let mut body = String::new();
        for k in 0..r {
            body.push_str(&format!("        let j{k} = base + lid.x + {}u;\n", k * bp));
            body.push_str(&format!("        if (j{k} < valid) {{\n"));
            body.push_str(&format!(
                "            let v{k} = half_at(unpack2x16float(X[base_x + (j{k} >> 1u)]), j{k});\n"
            ));
            if op == "max" {
                body.push_str(&format!(
                    "            {pre}{k} = max({pre}{k}, v{k} * scale);\n"
                ));
            } else {
                body.push_str(&format!(
                    "            {pre}{k} = {pre}{k} + exp(v{k} * scale - row_max);\n"
                ));
            }
            body.push_str("        }\n");
        }
        // Fold the tree's first `log2(r)` levels here, pairing `k` with
        // `k + half` exactly as the wider tree does.
        let mut cur: Vec<String> = (0..r).map(|k| format!("{pre}{k}")).collect();
        while cur.len() > 1 {
            let half = cur.len() / 2;
            let mut next = Vec::with_capacity(half);
            for i in 0..half {
                next.push(if op == "max" {
                    format!("max({}, {})", cur[i], cur[i + half])
                } else {
                    format!("({} + {})", cur[i], cur[i + half])
                });
            }
            cur = next;
        }
        let folded = cur.pop().unwrap();
        let lname = if op == "max" { "lmax" } else { "lsum" };
        let arr = if op == "max" { "red_max" } else { "red_sum" };
        format!(
            "{accl}    for (var base = 0u; base < valid; base = base + {bt}u) {{\n{body}\
             \x20   }}\n\
             \x20   let {lname} = {folded};\n\
             \x20   {arr}[lid.x] = {lname};\n",
        )
    };
    format!(
        "{enable}struct Cfg {{ n_w: u32, n_x: u32, valid: u32, m: u32, mp: u32, scale: f32, gx: u32, row0: u32, q_off: u32 }};

@group(0) @binding(0) var<storage, read>       X:   array<u32>;
@group(0) @binding(1) var<storage, read_write> Out: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;

{HALF_AT}
var<workgroup> red_max: array<f32, {bp}>;
var<workgroup> red_sum: array<f32, {bp}>;

@compute @workgroup_size({bp})
fn softmax(@builtin(workgroup_id) wgid: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>) {{
    // X rows are n_x-strided (tile-padded scores, one row per head*pos);
    // Out uses the BATCHED [head][mp][cur16] layout the AV GEMM expects.
    // Two grid axes: `nqh · s` rows overflow the 65535-per-dimension limit for
    // prefills beyond ~4.2 minutes (x is capped, y continues the row index).
    let row = wgid.x + wgid.y * cfg.gx;
    let head = row / cfg.m;
    let pos = row % cfg.m;
    let base_x = (head * cfg.mp + pos) * cfg.n_x;
    let base_o = (head * cfg.mp + pos) * cfg.n_w;
    let row_in_head = pos;
    // causal bound inside this dispatch's column window; row0 == 0 (flat path)
    // leaves `min(row_in_head + 1, valid)` exactly as it was
    let valid = min(cfg.valid, max(row_in_head + 1u + cfg.q_off, cfg.row0) - cfg.row0);
    let scale = cfg.scale;

    // A row left of the whole slab has no live column there, but the AV GEMM of
    // a straddling row block still reads its columns — they have to be zeros.
    // Only the reductions are skipped; the branch is workgroup-uniform (it
    // depends on the row alone), so it may skip the barriers.
    if (valid == 0u) {{
        for (var w = lid.x; w < cfg.n_w; w = w + {bp}u) {{
            Out[base_o + w] = 0u;
        }}
        return;
    }}

{load_max}    workgroupBarrier();
    for (var sh = {bp}u >> 1u; sh > 16u; sh = sh >> 1u) {{
        if (lid.x < sh) {{ red_max[lid.x] = max(red_max[lid.x], red_max[lid.x + sh]); }}
        workgroupBarrier();
    }}
{max_tail}    workgroupBarrier();
    let row_max = red_max[0];
    workgroupBarrier();

{load_sum}    workgroupBarrier();
    for (var sh = {bp}u >> 1u; sh > 16u; sh = sh >> 1u) {{
        if (lid.x < sh) {{ red_sum[lid.x] = red_sum[lid.x] + red_sum[lid.x + sh]; }}
        workgroupBarrier();
    }}
{sum_tail}    workgroupBarrier();
    let inv_sum = 1.0 / red_sum[0];
    workgroupBarrier();

    // one thread per f16 word — no read-modify-write races, and the per-element
    // arithmetic is the same as one element per thread
    for (var w = lid.x; w < cfg.n_w; w = w + {bp}u) {{
        let j0 = w * 2u;
        let j1 = j0 + 1u;
        var e0 = 0.0;
        var e1 = 0.0;
        if (j0 < valid) {{
            let v0 = half_at(unpack2x16float(X[base_x + w]), j0);
            e0 = exp(v0 * scale - row_max) * inv_sum;
        }}
        if (j1 < valid) {{
            let v1 = half_at(unpack2x16float(X[base_x + w]), j1);
            e1 = exp(v1 * scale - row_max) * inv_sum;
        }}
        Out[base_o + w] = pack2x16float(vec2<f32>(e0, e1));
    }}
}}
",
        bp = bp,
        enable = "",
        load_max = lanes("max"),
        load_sum = lanes("sum"),
        max_tail = max_tail,
        sum_tail = sum_tail,
    )
}

/// The shipped flat causal softmax: `bt == bp`, i.e. no lane emulation.
pub fn softmax_causal(bs: usize, subgroup: bool) -> String {
    softmax_causal_mode(
        bs,
        if subgroup {
            SubgroupMode::Warp32
        } else {
            SubgroupMode::None
        },
    )
}

pub fn softmax_causal_mode(bs: usize, mode: SubgroupMode) -> String {
    softmax_causal_tree_mode(bs, bs, mode)
}

/// Per-*slab* softmax statistics for the tiled prefill attention: one
/// `(max, Σexp)` pair per score row per key slab, written to
/// `Stats[slab][row]` as `(max, sum)` f32 pairs.
///
/// The slab width is the workgroup size, so an element is read once and kept in
/// a register across both reductions; the reduction trees and the exp are the
/// ones [`softmax_causal`] runs internally, and the causal bound is the same
/// `min(valid, row + 1 - row0)`.  That is what makes the pair recorded here
/// exactly the pair the (normed) slab it describes was divided by — the merge
/// weights are only meaningful under that agreement.
///
/// `row0` doubles as the slab index (`row0 / T`), which is how the dispatch
/// knows its slot without a second uniform field.
pub fn slab_stats(bs: usize, t: usize) -> String {
    assert!(
        t.is_power_of_two(),
        "slab_stats: the slab width must be a power of two"
    );
    assert!(bs.is_power_of_two() && bs <= t, "slab_stats: block size");
    format!(
        "struct StatsCfg {{ n_x: u32, valid: u32, m: u32, mp: u32, scale: f32, row0: u32, gx: u32, rows: u32, q_off: u32 }};

@group(0) @binding(0) var<storage, read>       X:     array<u32>;
@group(0) @binding(1) var<storage, read_write> Stats: array<f32>;
@group(0) @binding(2) var<uniform>             cfg:   StatsCfg;

const BS: u32 = {bs}u;
const T: u32 = {t}u;

{HALF_AT}
var<workgroup> red_max: array<f32, BS>;
var<workgroup> red_sum: array<f32, BS>;

@compute @workgroup_size(BS)
fn slab_stats(@builtin(workgroup_id) wgid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {{
    // Two grid axes: `nqh · mp` rows pass the 65535-per-dimension limit at a
    // ~5-minute prefill (the softmax carries the same pair).
    let row = wgid.x + wgid.y * cfg.gx;
    let head = row / cfg.m;
    let pos = row % cfg.m;
    let base_x = (head * cfg.mp + pos) * cfg.n_x;
    let valid = min(cfg.valid, max(pos + 1u + cfg.q_off, cfg.row0) - cfg.row0);
    // rows are the head-major `head·mp + pos` of the score buffer, not the
    // dispatch's flat row index (those differ: the grid spans `nqh · s`)
    let r = head * cfg.mp + pos;

    // Most rows of a given slab are entirely left of its first column — the
    // dispatch covers the full `nqh · s` either way, and running the two
    // reductions for them was two thirds of a long prefill's attention time.
    // The branch is workgroup-uniform (`valid` depends on the row only), so it
    // may skip the barriers.
    if (valid == 0u) {{
        if (lid.x == 0u) {{
            Stats[(cfg.row0 / T * cfg.rows + r) * 2u] = bitcast<f32>(0xFF800000u);
            Stats[(cfg.row0 / T * cfg.rows + r) * 2u + 1u] = 0.0;
        }}
        return;
    }}

    var lmax = bitcast<f32>(0xFF800000u);
    for (var j = lid.x; j < valid; j = j + BS) {{
        let sc = half_at(unpack2x16float(X[base_x + (j >> 1u)]), j) * cfg.scale;
        if (sc > lmax) {{ lmax = sc; }}
    }}
    red_max[lid.x] = lmax;
    workgroupBarrier();
    for (var sh = BS >> 1u; sh > 0u; sh = sh >> 1u) {{
        if (lid.x < sh) {{ red_max[lid.x] = max(red_max[lid.x], red_max[lid.x + sh]); }}
        workgroupBarrier();
    }}
    let row_max = red_max[0];
    workgroupBarrier();

    var lsum = 0.0;
    for (var j = lid.x; j < valid; j = j + BS) {{
        let sc = half_at(unpack2x16float(X[base_x + (j >> 1u)]), j) * cfg.scale;
        lsum = lsum + exp(sc - row_max);
    }}
    red_sum[lid.x] = lsum;
    workgroupBarrier();
    for (var sh = BS >> 1u; sh > 0u; sh = sh >> 1u) {{
        if (lid.x < sh) {{ red_sum[lid.x] = red_sum[lid.x] + red_sum[lid.x + sh]; }}
        workgroupBarrier();
    }}
    if (lid.x == 0u) {{
        Stats[(cfg.row0 / T * cfg.rows + r) * 2u] = row_max;
        Stats[(cfg.row0 / T * cfg.rows + r) * 2u + 1u] = red_sum[0];
    }}
}}
",
    )
}

/// Merge weights from the per-slab statistics: `w_t = exp(m_t − M) · Σexp_t`
/// normalised by the row's total, where `M` is the row's global max over slabs.
/// One thread per row — its own `n_slab` slots, read and written by nobody else.
///
/// A slab with no valid column for a row has `m_t = -inf, Σexp = 0`, so
/// `w_t = 0` with no NaN: `exp(-inf − M)` is 0 and `0 · 0` is 0.  At least one
/// slab (the first, `row0 = 0`) always has a valid column, so `M` is finite and
/// the total cannot be zero.
pub fn slab_weights(bs: usize) -> String {
    format!(
        "struct WCfg {{ rows: u32, n_slab: u32, gx: u32, _p: u32 }};

@group(0) @binding(0) var<storage, read>       Stats: array<f32>;
@group(0) @binding(1) var<storage, read_write> Wt:    array<f32>;
@group(0) @binding(2) var<uniform>             cfg:   WCfg;

@compute @workgroup_size({bs})
fn slab_weights(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x + gid.y * (cfg.gx * {bs}u);
    if (row >= cfg.rows) {{ return; }}

    var m = bitcast<f32>(0xFF800000u);
    for (var t = 0u; t < cfg.n_slab; t = t + 1u) {{
        m = max(m, Stats[(t * cfg.rows + row) * 2u]);
    }}
    var total = 0.0;
    for (var t = 0u; t < cfg.n_slab; t = t + 1u) {{
        total = total + exp(Stats[(t * cfg.rows + row) * 2u] - m) * Stats[(t * cfg.rows + row) * 2u + 1u];
    }}
    // rows past the last position have no statistics at all (the stats dispatch
    // only covers real positions); `total == 0` there, and 0 · (1/0) would be a
    // NaN the merge would then hand to the padded output rows
    let inv = select(0.0, 1.0 / total, total > 0.0);
    for (var t = 0u; t < cfg.n_slab; t = t + 1u) {{
        Wt[t * cfg.rows + row] =
            exp(Stats[(t * cfg.rows + row) * 2u] - m) * Stats[(t * cfg.rows + row) * 2u + 1u] * inv;
    }}
}}
",
        bs = bs,
    )
}

/// Weighted merge of the per-slab AV outputs into the attention output:
/// `out[head][row] = Σ_t w_t[head][row] · part[t][head][row]`, the weights
/// already normalised by [`slab_weights`].  Elementwise — `n_slab` f16 words
/// per output word, accumulated in f32 and rounded once.
///
/// One grid axis per (row, head-dim word) band with the head on `wgid.y`, so
/// neither index needs a division.  The two operands are laid out differently —
/// the statistics/weights are head-major (`head · mp + pos`, one entry per row),
/// the AV output is position-major (`pos · nqh·hd + head·hd + …`, the AV GEMM's
/// own C layout) — hence the two independent indices.
pub fn slab_merge(nqh: usize, hd: usize) -> String {
    let hd2 = hd / 2;
    format!(
        "struct MCfg {{ rows: u32, n_slab: u32, gx: u32, _p: u32 }};

@group(0) @binding(0) var<storage, read>       Part: array<u32>;
@group(0) @binding(1) var<storage, read>       Wt:   array<f32>;
@group(0) @binding(2) var<storage, read_write> Out:  array<u32>;
@group(0) @binding(3) var<uniform>             cfg:  MCfg;

const NQH: u32 = {nqh}u;
const HD2: u32 = {hd2}u;

@compute @workgroup_size(256)
fn slab_merge(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(workgroup_id) wgid: vec3<u32>) {{
    // `rows` = mp (padded positions); the stats/weights rows are `nqh · mp` with
    // the head-major index `head · rows + row`.
    let head = wgid.y;
    let i = gid.x + gid.z * (cfg.gx * 256u);
    if (i >= cfg.rows * HD2) {{ return; }}
    let row = i / HD2;
    let w = i - row * HD2;
    // weights/stats: [t][head · mp + pos]; AV output: [t][pos][head · hd + col]
    let wi = head * cfg.rows + row;
    let pi = row * (NQH * HD2) + head * HD2 + w;

    var acc = vec2<f32>(0.0, 0.0);
    for (var t = 0u; t < cfg.n_slab; t = t + 1u) {{
        acc = acc + Wt[t * (cfg.rows * NQH) + wi] * unpack2x16float(Part[t * (cfg.rows * NQH * HD2) + pi]);
    }}
    Out[pi] = pack2x16float(acc);
}}
",
        nqh = nqh,
        hd2 = hd2,
    )
}

/// repeat_kv:  K cache `[nkvh, max_seq, hd]` rows `0..cur` duplicated per GQA
/// group into `[nqh, cur, hd]`.
pub fn repeat_kv(nrep: usize) -> String {
    format!(
        "struct Cfg {{ nkvh: u32, max_seq: u32, cur: u32, hd: u32, npw: u32, gx: u32 }};

@group(0) @binding(0) var<storage, read>       Cache: array<u32>;
@group(0) @binding(1) var<storage, read_write> Out:   array<u32>;
@group(0) @binding(2) var<uniform>             cfg:   Cfg;

const NREP: u32 = {nrep}u;

@compute @workgroup_size(256)
fn repeat_kv(@builtin(global_invocation_id) gid: vec3<u32>) {{
    // Two grid axes: `nkvh · nrep · cur · hd/2` reaches 65535 words at a
    // ~19-minute prefill, so x is capped and y continues the index space.
    let i = gid.x + gid.y * (cfg.gx * 256u);
    let words_per_head_pos = cfg.hd / 2u;
    let per_head = cfg.cur * words_per_head_pos;
    let total = cfg.nkvh * NREP * per_head;
    if (i >= total) {{ return; }}
    let qh = i / per_head;
    let rem = i - qh * per_head;
    let p = rem / words_per_head_pos;
    let w = rem - p * words_per_head_pos;
    let kh = qh / NREP;
    // out rows are np-padded per head (the GEMM's n tiles read the padding)
    Out[qh * cfg.npw + p * words_per_head_pos + w] =
        Cache[(kh * cfg.max_seq + p) * words_per_head_pos + w];
}}
",
        nrep = nrep,
    )
}

/// Out-of-plane tap sentinel.  **One definition for both sides** — it is
/// interpolated into the WGSL below *and* written into the tap table by the
/// Rust caller, because the two drifting apart is exactly how the gather
/// silently read every out-of-bounds tap as a valid address (the shader
/// compared against `0xFFFF` while the table held `0xFFFF_FFFF`).
pub const TAP_OOB: u32 = 0xFFFF_FFFF;

/// im2col for `conv2d(3×3, stride 2, pad 1)`: a gather driven by a precomputed
/// tap table, so the inner loop has no divisions and no boundary tests.
///
/// **Layout contract** (the one place it is stated; `audio_encoder_gpu.rs`
/// derives every buffer size and dispatch from the same quantities):
///
/// * The operand is `[k_pad][n]` f16 — `n` the position axis — with the row
///   stride (in words) `n/2`.  That is what `prefill_gemm(transb = true)` reads
///   as its `B` operand (`W[(k)*(gd.n/2) + n/2]`, half by `n & 1`), and the
///   GEMM's `n` is the *whole tile*: every chunk's positions laid end to end,
///   chunk `c` occupying `[c*plane_pad, c*plane_pad + plane)`.
/// * A thread owns **one `k` and two adjacent positions**, writing one packed
///   word: `Cols[k*(n/2) + col/2]`.  Consecutive `tx` therefore write
///   consecutive words of one row — coalesced, with no chance of the
///   position/k axes aliasing (which is what corrupted this kernel twice).
/// * `k = ic*9 + kh*3 + kw`, the flattening `weight[c_out, c_in, 3, 3]` uses;
///   `Taps[p*9 + tap]` is the source offset of that tap inside one input
///   channel plane, `TAP_OOB` outside it.  The source is
///   `in_chunk0*in_chunk + ic*in_ic + tap`, which covers both inputs:
///   the mel (`ic == 0`, `in_chunk` = one chunk's whole image) and a previous
///   activation (`in_chunk` = one chunk's positions, `in_ic` = the channel
///   stride, i.e. `n_all` of that level).
/// * Every element of the operand is **written**: `k >= k_real` (the pad rows
///   the GEMM's 16-wide k-tile reads), `chunk >= n_chunks` (the chunks a short
///   final round does not have) and `p >= plane` get an explicit zero, so the
///   GEMM never accumulates bytes this kernel did not define.
///
/// Bindings: 0 = input, 1 = taps, 2 = operand, 3 = `Im2Cfg`.
pub fn audio_im2col() -> String {
    format!(
        "struct Im2Cfg {{ taps: u32, k: u32, k_pad: u32, plane: u32, plane_pad: u32,
                     n_chunks: u32, n_all: u32, in_chunk: u32, in_ic: u32,
                     chunk0: u32, bpc: u32, _a: u32 }};

@group(0) @binding(0) var<storage, read>       Input: array<u32>;
@group(0) @binding(1) var<storage, read>       Taps:  array<u32>;
@group(0) @binding(2) var<storage, read_write> Cols:  array<u32>;
@group(0) @binding(3) var<uniform>             cfg:   Im2Cfg;

/// `TAP_OOB` injected from the Rust table builder — see `TAP_OOB`.
const OOB: u32 = {oob}u;

fn scalar(addr: u32) -> f32 {{
    let w = unpack2x16float(Input[addr / 2u]);
    return select(w.x, w.y, (addr & 1u) == 1u);
}}

@compute @workgroup_size(16, 16)
fn im2col(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>) {{
    // `bpc` = position-blocks per chunk, so `wid.x` splits into (chunk, block)
    // with no division inside the element loop; the blocks past `chunks*bpc`
    // are the tile's tail padding and write zeros.
    // `wid.x` splits into (chunk, position block) — two *independent* axes.
    // Treating the position as a single flat counter (`chunk` implicit in the
    // column) is wrong whenever `plane_pad > plane`: the last block of the last
    // chunk then looks like padding and gets zeroed.
    let chunk = wid.x / cfg.bpc;
    let p = (wid.x % cfg.bpc) * 32u + 2u * lid.x;
    let col = chunk * cfg.plane_pad + p;
    let k = wid.y * 16u + lid.y;
    if (k >= cfg.k_pad || col >= cfg.n_all) {{ return; }}
    let word = k * (cfg.n_all / 2u) + col / 2u;
    if (k >= cfg.k || chunk >= cfg.n_chunks || p >= cfg.plane) {{
        Cols[word] = 0u;
        return;
    }}
    let ic = k / cfg.taps;
    let base = p * cfg.taps + (k % cfg.taps);
    let src = (cfg.chunk0 + chunk) * cfg.in_chunk + ic * cfg.in_ic;
    let a = Taps[base];
    let b = Taps[base + cfg.taps];
    var x = vec2<f32>(0.0, 0.0);
    if (a != OOB) {{ x.x = scalar(src + a); }}
    if (b != OOB && p + 1u < cfg.plane) {{ x.y = scalar(src + b); }}
    Cols[word] = pack2x16float(x);
}}
",
        oob = TAP_OOB
    )
}

/// `dst[i] = gelu(src[i] + bias[c])` over packed f16, two elements per thread.
///
/// `bias` is the per-*channel* vector, padded with zeros to the activation's
/// channel count and packed as plain f16 (two channels per word) — **not** a
/// broadcast image, and not one bias per word: reading `Bias[c]` for channel
/// `c` doubles the channel, which is what made every conv activation wrong.
/// `cfg.words` is the number of words per channel or per row, and `cfg.mode`
/// selects `i / words` (channel-major conv activation: one channel per `n_all`
/// positions, both halves the same channel) or `i % words` (token-major GEMM
/// output: the halves are adjacent columns of one row).
/// Broadcasting to an image instead would cost `rows · n_pad` per tensor, which
/// at `MAX_TOKENS` rows is 117 MB *per FFN layer*.
///
/// WGSL has no `erf`, so GELU is the A&S 7.1.26 `tanh`-style rational
/// approximation with `|ε| ≤ 1.5e-7` — two f32 ulp at the extremes of the
/// argument range this tower produces, and far below the f16 rounding of the
/// result.  (The *reference* uses erf-based GELU; matching it to the last f32
/// bit is pointless when the operand itself is already an f16 GEMM output.)
///
/// `n` is the element count; it must be even so a thread's pair never straddles
/// the channel vector's real/padding boundary.
///
/// Bindings: 0 = src, 1 = bias, 2 = dst, 3 = `ScaleCfg { n, words, mode }`.
pub fn audio_bias_gelu() -> String {
    "struct ScaleCfg { n: u32, words: u32, mode: u32, gx: u32 };

@group(0) @binding(0) var<storage, read>       Src:  array<u32>;
@group(0) @binding(1) var<storage, read>       Bias: array<u32>;
@group(0) @binding(2) var<storage, read_write> Dst:  array<u32>;
@group(0) @binding(3) var<uniform>             cfg:  ScaleCfg;

const FRAC_1_SQRT_2: f32 = 0.70710678;
// A&S 7.1.26's `p` = 1/(1 + p|x|).  It was `0.147`, which is not this
// approximation's constant: the resulting erf was off by ~5 %, i.e. an order of
// magnitude more than the f16 rounding of the operand it feeds.
const C_A: f32 = 0.3275911;
const C_2_SQRT_PI: f32 = 1.128379167;

/// Abramowitz & Stegun 7.1.26: max abs error 1.5e-7.
fn erf_approx(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + C_A * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                   - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return select(-y, y, x >= 0.0);
}

fn gelu(x: f32) -> f32 {
    return 0.5 * x * (1.0 + erf_approx(x * FRAC_1_SQRT_2));
}

@compute @workgroup_size(256)
fn bias_gelu(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Two grid axes: a long clip's FFN activation is `tokens · ffn/2` words,
    // which passes 65535 workgroups well before the 22-minute token cap.
    let i = gid.x + gid.y * (cfg.gx * 256u);
    if (i * 2u >= cfg.n) { return; }
    let s = unpack2x16float(Src[i]);
    var b = vec2<f32>(0.0, 0.0);
    if (cfg.mode == 1u) {
        // Channel-major: both halves of the word are the same channel, one
        // channel per `words` words, and `Bias` is packed two channels per word
        // — so the channel's element is one *half* of `Bias[c/2]`, not `Bias[c]`.
        let c = i / cfg.words;
        let w = unpack2x16float(Bias[c / 2u]);
        let b0 = select(w.x, w.y, (c & 1u) == 1u);
        b = vec2<f32>(b0, b0);
    } else {
        // Token-major: the halves are adjacent columns of one row, and the
        // word's own two halves are exactly those two biases.
        b = unpack2x16float(Bias[i % cfg.words]);
    }
    Dst[i] = pack2x16float(vec2<f32>(gelu(s.x + b.x), gelu(s.y + b.y)));
}
"
    .to_string()
}

/// `LayerNorm` over the last dim, one workgroup per row.  Serial two-pass
/// reduction in a single lane — bit-identical to the CPU reference's
/// `mean`, then `var`, then `(x - mean) * inv_std * w + b`, which matters
/// because the encoder feeds the decoder f16 embeddings that later layers
/// amplify.
///
/// Bindings: 0 = src, 1 = weight, 2 = bias, 3 = `LnCfg { d, eps, ... }`,
/// 4 = dst.  (Uniform before storage: wgpu requires the uniform last, so `Dst`
/// takes binding 4 and the uniform stays at 3.)
pub fn audio_layernorm() -> String {
    "struct LnCfg { d: u32, eps: f32, _a: u32, _b: u32 };

@group(0) @binding(0) var<storage, read>       Src: array<u32>;
@group(0) @binding(1) var<storage, read>       Wgt: array<u32>;
@group(0) @binding(2) var<storage, read>       Bia: array<u32>;
@group(0) @binding(3) var<uniform>             cfg: LnCfg;
@group(0) @binding(4) var<storage, read_write> Dst: array<u32>;

@compute @workgroup_size(32)
fn layernorm(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    // One row per lane, 32 rows per warp.
    //
    // This used to be `@workgroup_size(1)` -- one thread walking a whole 896-wide
    // row through three serial passes, so a warp instruction drove one lane and
    // an SM was pinned at its 32-workgroup ceiling, i.e. 32 one-lane warps.  The
    // per-row arithmetic and its order are *untouched*: each lane still walks its
    // own row, element by element, in the same sequence.  Only the lane mapping
    // changes.  Measured +45.1 ms of a 302 ms transformer, 1.3 GB/s.
    let d = cfg.d;
    let base = (wid.x * 32u + lid.x) * (d / 2u);
    var mean = 0.0;
    // Word-at-a-time instead of element-at-a-time: the old loop re-read and
    // re-unpacked the same `Src` word for the even and the odd element it holds.
    // `x` then `y` is the same accumulation order the element loop had.
    for (var w: u32 = 0u; w < d / 2u; w = w + 1u) {
        let s = unpack2x16float(Src[base + w]);
        mean = mean + s.x;
        mean = mean + s.y;
    }
    mean = mean / f32(d);
    var var_ = 0.0;
    for (var w: u32 = 0u; w < d / 2u; w = w + 1u) {
        let s = unpack2x16float(Src[base + w]);
        let x = s.x - mean;
        let y = s.y - mean;
        var_ = var_ + x * x;
        var_ = var_ + y * y;
    }
    var_ = var_ / f32(d);
    let inv = 1.0 / sqrt(var_ + cfg.eps);
    for (var w: u32 = 0u; w < d / 2u; w = w + 1u) {
        let s = unpack2x16float(Src[base + w]);
        let g = unpack2x16float(Wgt[w]);
        let b = unpack2x16float(Bia[w]);
        Dst[base + w] = pack2x16float(vec2<f32>(
            (s.x - mean) * inv * g.x + b.x,
            (s.y - mean) * inv * g.y + b.y,
        ));
    }
}
"
    .to_string()
}

/// Split the fused QKV projection into the attention layouts and add the
/// sinusoidal positional embedding to Q.
///
/// `Qkv` is `[tok, 3·d_model]`; Q/K/V are written as `[tok, nh, hd]` (row
/// stride `attn_cols`, head-major inside the row), which is exactly the
/// `[head][tok][hd]` view the attention GEMMs index with a per-head batch
/// stride.  PE row index is `tok % tpc` (`tpc = feo(n_window*2)`), matching the
/// CPU reference's `it % t2` broadcast.
///
/// Bindings: 0 = qkv, 1 = Q, 2 = K, 3 = V, 5 = `ExCfg`.
pub fn audio_extract_qkv() -> String {
    "struct ExCfg { n_tokens: u32, nh: u32, hd: u32, tpc: u32,
                row_stride: u32, attn_cols: u32, pe_stride: u32, dm: u32 };

@group(0) @binding(0) var<storage, read>       Qkv: array<u32>;
@group(0) @binding(1) var<storage, read_write> Q:   array<u32>;
@group(0) @binding(2) var<storage, read_write> K:   array<u32>;
@group(0) @binding(3) var<storage, read_write> V:   array<u32>;
@group(0) @binding(5) var<uniform>             cfg: ExCfg;

@compute @workgroup_size(256)
fn extract(@builtin(global_invocation_id) gid: vec3<u32>) {
    // gid = (word of a token's q/k/v slice, token).  The token is on `y` and the
    // word on `x` so adjacent lanes read adjacent words; with the token on `x`
    // every warp load was 32 lanes at a `3*dm/2`-word stride -- 32 cache lines
    // for 128 B of use, each line re-fetched once per word that fell in it.
    // Measured +30.1 ms of a 302 ms transformer, 11.7 GB/s against a 289 GB/s
    // roof.  `s_pad` alone cannot exceed 65535, so the `s_pad * nh` limit this
    // used to work around no longer exists.
    let acols2 = cfg.attn_cols / 2u;
    let col = gid.x;
    if (col >= acols2) { return; }
    let tok = gid.y;
    if (tok >= cfg.n_tokens) { return; }

    let dm2 = cfg.dm / 2u;
    let row = tok * (dm2 * 3u);
    let dst = tok * acols2 + col;

    // No positional embedding here: the reference adds it to the *conv_out
    // output* (`audio_add_pe`), and adding it twice is not the reference's math.
    Q[dst] = Qkv[row + col];
    K[dst] = Qkv[row + dm2 + col];
    V[dst] = Qkv[row + dm2 * 2u + col];
}
"
    .to_string()
}

/// Re-pack the projected Q/K/V into the per-`(head, window)` blocks the two
/// attention GEMMs read.
///
/// The projections come out token-major (`[tok][nh·hd]`), but `prefill_gemm`
/// fixes both operand row strides from `k`: the `A` row stride is `k/2` words
/// and a `transb` `B` row stride is `n/2`.  A head's slice of a token row is
/// `nh·hd` halves wide, so neither GEMM can read it in place — this kernel is
/// what makes the strides line up:
///
/// ```text
///   qp[z][row][j]   row stride hd/2 words   (scores A, m = token)
///   kt[z][j][row]   row stride wpad/2 words (scores B, transb, n = token)
///   vp[z][row][j]   row stride 64 words     (AV B, transb, n = 128 = hd_pad)
/// ```
///
/// `z = head·n_win + win` and rows are `win·wlen + row` in the global token
/// axis; rows/keys past a short last window are **zeroed**, which is what lets
/// the softmax mask them by a loop bound.  The `j` axis is padded to `hd_pad`
/// (= `GEMM_BN`) because the AV GEMM's `n` must be a whole 128-wide tile.
///
/// Bindings: 0 = Q, 1 = K, 2 = V, 3 = qp, 4 = kt, 5 = vp, 6 = `WinCfg`.
pub fn audio_win_pack() -> String {
    "struct WinCfg { wlen: u32, wpad: u32, hd: u32, n_win: u32, s: u32, acols: u32,
                pad_n: u32, _a: u32 };

@group(0) @binding(0) var<storage, read>       Q:  array<u32>;
@group(0) @binding(1) var<storage, read>       K:  array<u32>;
@group(0) @binding(2) var<storage, read>       V:  array<u32>;
@group(0) @binding(3) var<storage, read_write> Qp: array<u32>;
@group(0) @binding(4) var<storage, read_write> Kt: array<u32>;
@group(0) @binding(5) var<storage, read_write> Vp: array<u32>;
@group(0) @binding(6) var<uniform>             cfg: WinCfg;

@compute @workgroup_size(16, 16)
fn win_pack(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {
    let z = wid.z;
    let head = z / cfg.n_win;
    let win = z - head * cfg.n_win;
    let rp = wid.x * 16u + lid.x;          // token pair inside the window
    let jw = wid.y * 16u + lid.y;          // head-dim word (two j values)
    let hd2 = cfg.hd / 2u;
    let wpad2 = cfg.wpad / 2u;
    if (2u * rp >= cfg.wpad) { return; }
    let valid = min(cfg.wlen, cfg.s - win * cfg.wlen);
    let j0 = 2u * jw;
    let in_range = j0 < cfg.hd;            // the head-dim padding stays zero

    var q0 = vec2<f32>(0.0, 0.0);
    var k0 = q0; var v0 = q0; var q1 = q0; var k1 = q0; var v1 = q0;
    if (in_range) {
        let t0 = win * cfg.wlen + 2u * rp;
        let i0 = t0 * (cfg.acols / 2u) + head * hd2 + jw;
        if (2u * rp < valid) {
            q0 = unpack2x16float(Q[i0]);
            k0 = unpack2x16float(K[i0]);
            v0 = unpack2x16float(V[i0]);
        }
        if (2u * rp + 1u < valid) {
            let i1 = i0 + cfg.acols / 2u;
            q1 = unpack2x16float(Q[i1]);
            k1 = unpack2x16float(K[i1]);
            v1 = unpack2x16float(V[i1]);
        }
    }

    // vp: one row per token, `pad_n/2` words wide, so the head-dim padding is
    // *inside* the row and these threads write the zeros for it.
    let vp = z * (cfg.wpad * (cfg.pad_n / 2u)) + (2u * rp) * (cfg.pad_n / 2u) + jw;
    Vp[vp] = pack2x16float(v0);
    Vp[vp + cfg.pad_n / 2u] = pack2x16float(v1);

    // qp: the row is exactly `hd2` words with no padding, so a `jw` past the
    // row must not write at all — `Qp[qp]` would land in the *next* token's row
    // and this thread's zeros would race the writes of that row's own thread
    // (whichever workgroup the hardware ran last decided whether a Q row was
    // real or zero).
    if (in_range) {
        let qp = z * (cfg.wpad * hd2) + (2u * rp) * hd2 + jw;
        Qp[qp] = pack2x16float(q0);
        Qp[qp + hd2] = pack2x16float(q1);
    }

    // kt: the transpose — row `j`, and the token pair in one word.  Only the
    // real `hd` rows exist; the rest would run into the next block.
    if (in_range) {
        let kt = z * (cfg.hd * wpad2) + j0 * wpad2 + rp;
        Kt[kt] = pack2x16float(vec2<f32>(k0.x, k1.x));
        Kt[kt + wpad2] = pack2x16float(vec2<f32>(k0.y, k1.y));
    }
}
"
    .to_string()
}

/// Windowed attention softmax — one thread per `(head, window, token)` row,
/// serial over the window's valid keys.
///
/// Blocks are **dense**: `[z][wpad][wpad]` with `z = head·n_win + win`, so the
/// batched GEMMs' block strides are all `wpad²/2` words.  (The old layout mixed
/// a per-head `s_pad` stride with a per-window one, which is why its `A`
/// operand addressing never lined up.)
///
/// The reference runs each window over `min(wlen, s - win·wlen)` tokens, so a
/// short last window must **not** see the padded keys: the loops run to `valid`
/// and everything past it is written zero.
///
/// Bindings: 0 = scores, 1 = probs, 2 = `SmCfg { s, wlen, wpad, n_win, scale }`.
pub fn audio_window_softmax() -> String {
    "struct SmCfg { s: u32, wlen: u32, wpad: u32, n_win: u32, scale: f32,
                _a: u32, _b: u32, _c: u32 };

@group(0) @binding(0) var<storage, read>       Sc:  array<u32>;
@group(0) @binding(1) var<storage, read_write> Out: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: SmCfg;

fn half_at(v: vec2<f32>, i: u32) -> f32 { return select(v.x, v.y, (i & 1u) == 1u); }

@compute @workgroup_size(128)
fn softmax(@builtin(global_invocation_id) gid: vec3<u32>) {
    // gid.x = token inside the window; gid.y = head·n_win + window
    let row = gid.x;
    if (row >= cfg.wpad) { return; }
    let z = gid.y;
    let win = z % cfg.n_win;
    let valid = min(cfg.wlen, cfg.s - win * cfg.wlen);
    let base = z * (cfg.wpad * (cfg.wpad / 2u)) + row * (cfg.wpad / 2u);
    // Keys past `valid` contribute nothing even for a live row, and a row past
    // `valid` has no consumer at all — write both as zero.
    if (row >= valid) {
        for (var w: u32 = 0u; w < cfg.wpad / 2u; w = w + 1u) { Out[base + w] = 0u; }
        return;
    }

    // Element-at-a-time, and deliberately so: the word-at-a-time rewrite that
    // paid off in the audio layer norm (half the loads and unpacks) measured
    // *slower* here -- +3 ms on a 243 ms transformer, 3 reps each -- so it was
    // reverted.  The two passes below are only ~15 ms of the transformer between
    // them, which is inside this probe's resolution.
    var mx = -3.0e38;
    for (var j: u32 = 0u; j < valid; j = j + 1u) {
        mx = max(mx, half_at(unpack2x16float(Sc[base + j / 2u]), j) * cfg.scale);
    }
    var sum = 0.0;
    for (var j: u32 = 0u; j < valid; j = j + 1u) {
        sum = sum + exp(half_at(unpack2x16float(Sc[base + j / 2u]), j) * cfg.scale - mx);
    }
    let inv = 1.0 / sum;
    for (var w: u32 = 0u; w < cfg.wpad / 2u; w = w + 1u) {
        let j0 = w * 2u;
        let j1 = j0 + 1u;
        let v = unpack2x16float(Sc[base + w]);
        let p0 = select(0.0, exp(v.x * cfg.scale - mx) * inv, j0 < valid);
        let p1 = select(0.0, exp(v.y * cfg.scale - mx) * inv, j1 < valid);
        Out[base + w] = pack2x16float(vec2<f32>(p0, p1));
    }
}
"
    .to_string()
}

/// Conv-stem epilogue: `h = conv_out + PE`.  `h` is `[tokens, d]` with token
/// stride `s_pad` (the transformer's operand layout); `dst` is the packed
/// `[tokens, d]` conv-stem output.  Both get the same value so the embedding
/// dump and the transformer consume identical bytes.
///
/// Bindings: 0 = h, 1 = PE `[tpc, d]`, 2 = dst, 3 = `PeCfg { d, tpc, s_pad }`.
pub fn audio_add_pe() -> String {
    "struct PeCfg { d: u32, tpc: u32, s_pad: u32, rows: u32, dst_row: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       H:   array<u32>;
@group(0) @binding(1) var<storage, read>       Pe:  array<u32>;
@group(0) @binding(2) var<storage, read_write> Dst: array<u32>;
@group(0) @binding(3) var<uniform>             cfg: PeCfg;

@compute @workgroup_size(16, 16)
fn add_pe(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>) {
    // 256 words per workgroup, so the y grid has to walk the row: `d_model/2`
    // is 448 words and the old single-workgroup row left 192 of them unwritten.
    let w = wid.y * 256u + lid.x + 16u * lid.y;
    let d2 = cfg.d / 2u;
    if (w >= d2) { return; }
    let tok = wid.x;
    // Rows past `rows` are the caller's (e.g. the next utterance's block).
    if (tok >= cfg.rows) { return; }
    let v = unpack2x16float(H[tok * (cfg.s_pad / 2u) + w])
          + unpack2x16float(Pe[(tok % cfg.tpc) * d2 + w]);
    let packed = pack2x16float(v);
    Dst[(cfg.dst_row + tok) * (cfg.s_pad / 2u) + w] = packed;
}
"
    .to_string()
}

/// Gather the conv3 activation `[c][pos]` into the `conv_out` operand
/// `[tok, c·f]`.
///
/// `audio_conv`'s GEMM writes its `C` as `[m][n]` with `m` the **channel** and
/// `n` the flattened position, so a chunk's plane for channel `c` starts at
/// `c*n_all` and holds `plane_pad` positions per chunk.  Position
/// `p = f*t3 + ti` (`ti` fastest — the GEMM's tile wrote `(h_out, w_out)` in
/// row-major order), and the chunk's token `ti` is the `t3`-th part of the
/// token index.  A packed row element `j = c*f_dim + f` therefore reads
///
/// ```text
///   src = c*n_all + chunk*plane_pad + f*t3 + ti
/// ```
///
/// The positional embedding is **not** added here: the reference adds it to the
/// `conv_out` *output* (`d_model` wide), not to this `c·f` operand — see
/// [`audio_add_pe`].
///
/// Bindings: 0 = c3, 1 = packed, 2 = `PmCfg`.
pub fn audio_permute_pe() -> String {
    "struct PmCfg { c: u32, f: u32, t3: u32, s_pad: u32, n_all: u32, plane_pad: u32,
                tok0: u32, n_tokens: u32 };

@group(0) @binding(0) var<storage, read>       C3:  array<u32>;
@group(0) @binding(1) var<storage, read_write> Dst: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: PmCfg;

fn half_at(addr: u32) -> f32 {
    let v = unpack2x16float(C3[addr / 2u]);
    return select(v.x, v.y, (addr & 1u) == 1u);
}

@compute @workgroup_size(16, 16)
fn permute_pe(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    let cf = cfg.c * cfg.f;
    // 256 words per workgroup, so `wid.y` walks the row: `c·f/2` is 3840 words
    // and a single block would leave all but the first 256 unwritten.
    let w = wid.y * 256u + lid.x + 16u * lid.y;
    if (w * 2u >= cf) { return; }
    // `wid.x` is the token inside this tile, spanning `plane_pad / t3` chunks;
    // the grid is padded to a GEMM `m` tile so the rows past the clip — which
    // the `conv_out` GEMM still reads — are written zero rather than left as
    // whatever the allocator handed us.
    let tok = wid.x;
    if (cfg.tok0 + tok >= cfg.n_tokens) {
        // The destination row is the *global* token: using the tile-local `tok`
        // zeroed the wrong rows whenever a round did not start at token 0 (it
        // wiped the previous round's tail and its own head).
        Dst[(cfg.tok0 + tok) * (cfg.s_pad / 2u) + w] = 0u;
        return;
    }
    let chunk = (cfg.tok0 + tok) / cfg.t3 - cfg.tok0 / cfg.t3;
    let ti = (cfg.tok0 + tok) % cfg.t3;
    var x = vec2<f32>(0.0, 0.0);
    for (var e: u32 = 0u; e < 2u; e = e + 1u) {
        let j = w * 2u + e;
        let c = j / cfg.f;
        let f = j - c * cfg.f;
        let src = c * cfg.n_all + chunk * cfg.plane_pad + f * cfg.t3 + ti;
        let v = half_at(src);
        if (e == 0u) { x.x = v; } else { x.y = v; }
    }
    Dst[(cfg.tok0 + tok) * (cfg.s_pad / 2u) + w] = pack2x16float(x);
}
"
    .to_string()
}

/// Flatten the per-`(head, window)` attention output into the token-major
/// `[tok, nh·hd]` operand `out_proj` wants.
///
/// The AV GEMM writes `[z][row][hd_pad]` blocks (`z = head·n_win + win`); a
/// destination word covers two adjacent `j`, which stay inside one head, so the
/// copy is word-for-word — only the two index maps (row → global token, head →
/// block) change.
///
/// Bindings: 0 = src blocks, 1 = dst, 2 = `CpCfg { cols, wlen, wpad, hd, n_win,
/// rows, ... }`.
pub fn audio_attn_flat() -> String {
    "struct CpCfg { cols: u32, wlen: u32, wpad: u32, hd: u32, n_win: u32, rows: u32,
                _a: u32, _b: u32 };

@group(0) @binding(0) var<storage, read>       Src: array<u32>;
@group(0) @binding(1) var<storage, read_write> Dst: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: CpCfg;

@compute @workgroup_size(256)
fn attn_flat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;                       // word index into the destination
    let cols2 = cfg.cols / 2u;
    if (i >= cfg.rows * cols2) { return; }
    let tok = i / cols2;
    let w = i - tok * cols2;
    let win = tok / cfg.wlen;
    let row = tok - win * cfg.wlen;
    let hd2 = cfg.hd / 2u;
    let head = w / hd2;
    let z = head * cfg.n_win + win;
    let src = z * (cfg.wpad * 64u) + row * 64u + (w - head * hd2);
    Dst[i] = Src[src];
}
"
    .to_string()
}

#[cfg(test)]
mod stage2_subgroup_tests {
    use super::*;

    fn gemv_cases(mode: SubgroupMode) -> Vec<(&'static str, String)> {
        vec![
            ("gemv_split", gemv_split_mode(64, 1024, mode, 2)),
            ("gemv", gemv_mode(64, 1024, true, mode, 8)),
            ("gemv_rpwr", gemv_rpwr_mode(64, 1024, true, mode, 8, 2)),
            ("gemv_w8", gemv_w8_mode(64, 1024, true, mode, 8)),
            (
                "gemv_w8_rpwr",
                gemv_w8_rpwr_mode(64, 1024, true, mode, 8, 2),
            ),
            (
                "gemv_norm",
                gemv_norm_mode(64, 1024, true, mode, 1024, 1024, 1e-5),
            ),
            (
                "gemv_norm_w8",
                gemv_norm_w8_mode(64, 1024, true, mode, 1024, 1024, 1e-5),
            ),
            (
                "gemv_batch_slots",
                gemv_batch_slots_mode(64, 1024, 4, true, mode),
            ),
            (
                "gemv_norm_batch_slots",
                gemv_norm_batch_slots_mode(64, 1024, 4, true, 1024, 1024, 1e-5, mode),
            ),
        ]
    }

    #[test]
    fn legacy_gemv_wrappers_are_byte_identical() {
        assert_eq!(
            gemv_split(64, 1024, true, 2),
            gemv_split_mode(64, 1024, SubgroupMode::Warp32, 2)
        );
        assert_eq!(
            gemv_split(64, 1024, false, 2),
            gemv_split_mode(64, 1024, SubgroupMode::None, 2)
        );
        assert_eq!(
            gemv(64, 1024, true, true, 8),
            gemv_mode(64, 1024, true, SubgroupMode::Warp32, 8)
        );
        assert_eq!(
            gemv(64, 1024, true, false, 8),
            gemv_mode(64, 1024, true, SubgroupMode::None, 8)
        );
        assert_eq!(
            gemv_rpwr(64, 1024, true, true, 8, 2),
            gemv_rpwr_mode(64, 1024, true, SubgroupMode::Warp32, 8, 2)
        );
        assert_eq!(
            gemv_w8(64, 1024, true, true, 8),
            gemv_w8_mode(64, 1024, true, SubgroupMode::Warp32, 8)
        );
        assert_eq!(
            gemv_w8_rpwr(64, 1024, true, false, 8, 2),
            gemv_w8_rpwr_mode(64, 1024, true, SubgroupMode::None, 8, 2)
        );
        assert_eq!(
            gemv_norm(64, 1024, true, true, 1024, 1024, 1e-5),
            gemv_norm_mode(64, 1024, true, SubgroupMode::Warp32, 1024, 1024, 1e-5)
        );
        assert_eq!(
            gemv_norm_w8(64, 1024, true, false, 1024, 1024, 1e-5),
            gemv_norm_w8_mode(64, 1024, true, SubgroupMode::None, 1024, 1024, 1e-5)
        );
        assert_eq!(
            gemv_batch_slots(64, 1024, 4, true),
            gemv_batch_slots_mode(64, 1024, 4, true, SubgroupMode::None)
        );
        assert_eq!(
            gemv_norm_batch_slots(64, 1024, 4, true, 1024, 1024, 1e-5),
            gemv_norm_batch_slots_mode(64, 1024, 4, true, 1024, 1024, 1e-5, SubgroupMode::None)
        );
    }

    #[test]
    fn legacy_softmax_wrappers_are_byte_identical() {
        assert_eq!(
            softmax_causal(256, true),
            softmax_causal_mode(256, SubgroupMode::Warp32)
        );
        assert_eq!(
            softmax_causal(256, false),
            softmax_causal_mode(256, SubgroupMode::None)
        );
        assert_eq!(
            softmax_causal_tree(64, 256, true),
            softmax_causal_tree_mode(64, 256, SubgroupMode::Warp32)
        );
        assert_eq!(
            softmax_causal_tree(64, 256, false),
            softmax_causal_tree_mode(64, 256, SubgroupMode::None)
        );
    }

    #[test]
    fn no_mode_emits_enable_subgroups() {
        let modes = [
            SubgroupMode::Warp32,
            SubgroupMode::Warp64,
            SubgroupMode::Generic,
            SubgroupMode::None,
        ];
        for mode in modes {
            for (name, wgsl) in gemv_cases(mode) {
                assert!(!wgsl.contains("enable subgroups;"), "{name} {mode:?}");
            }
            for bs in [32usize, 256] {
                let sm = softmax_causal_mode(bs, mode);
                assert!(!sm.contains("enable subgroups;"), "softmax {bs} {mode:?}");
                let tree = softmax_causal_tree_mode(64, 256, mode);
                assert!(!tree.contains("enable subgroups;"), "tree {mode:?}");
            }
        }
    }

    #[test]
    fn warp64_uses_xor_shuffles_for_gemv() {
        for mode in [SubgroupMode::Warp32, SubgroupMode::Warp64] {
            for (name, wgsl) in gemv_cases(mode) {
                assert!(wgsl.contains("subgroupShuffleXor"), "{name} {mode:?}");
                assert!(!wgsl.contains("var<workgroup> bt0"), "{name} {mode:?}");
                assert!(!wgsl.contains("@builtin(subgroup_size)"), "{name} {mode:?}");
            }
        }
    }

    #[test]
    fn generic_uses_subgroup_add_and_sg_red_for_gemv() {
        for (name, src) in gemv_cases(SubgroupMode::Generic) {
            assert!(src.contains("subgroupAdd(v)"), "{name}");
            assert!(src.contains("var<workgroup> sg_red"), "{name}");
            assert!(src.contains("@builtin(subgroup_size) sg_size"), "{name}");
            assert!(!src.contains("var<workgroup> bt0"), "{name}");
        }
    }

    #[test]
    fn portable_softmax_uses_subgroup_max_and_add() {
        for mode in [SubgroupMode::Warp64, SubgroupMode::Generic] {
            for bs in [32usize, 64, 128, 256, 512, 1024] {
                let src = softmax_causal_mode(bs, mode);
                assert!(src.contains("subgroupMax(lmax)"), "bs={bs} {mode:?}");
                assert!(src.contains("subgroupAdd(lsum)"), "bs={bs} {mode:?}");
                assert!(src.contains("row_in_head + 1u + cfg.q_off"), "bs={bs}");
                assert!(src.contains("q_off: u32"), "bs={bs}");
                assert!(!src.contains("subgroupShuffleXor"), "bs={bs}");
            }
            let tree = softmax_causal_tree_mode(64, 256, mode);
            assert!(tree.contains("subgroupMax(lmax)"));
            assert!(tree.contains("subgroupAdd(lsum)"));
        }
    }

    #[test]
    fn vulkan_validates_all_stage2_modes() {
        let gpu = match pollster::block_on(crate::gpu::Gpu::new_with(
            crate::gpu::DeviceSelector::parse("vulkan").unwrap(),
        )) {
            Ok(gpu) => gpu,
            Err(e) => {
                eprintln!("skip Vulkan WGSL validation test: {e}");
                return;
            }
        };
        if !gpu.has_subgroups() {
            eprintln!("skip Vulkan WGSL validation test: adapter lacks SUBGROUP");
            return;
        }
        let modes = [
            SubgroupMode::Warp32,
            SubgroupMode::Warp64,
            SubgroupMode::Generic,
            SubgroupMode::None,
        ];
        for (i, mode) in modes.iter().enumerate() {
            let label = |name: &str| format!("stage2_{i}_{name}");
            let jobs: Vec<(&str, String, &str)> = vec![
                ("gemv", gemv_mode(64, 1024, true, *mode, 8), "gemv"),
                ("gemv_w8", gemv_w8_mode(64, 1024, true, *mode, 8), "gemv"),
                ("gemv_split", gemv_split_mode(64, 1024, *mode, 2), "gemv"),
                (
                    "gemv_norm",
                    gemv_norm_mode(64, 1024, true, *mode, 1024, 1024, 1e-5),
                    "gemv",
                ),
                (
                    "gemv_norm_w8",
                    gemv_norm_w8_mode(64, 1024, true, *mode, 1024, 1024, 1e-5),
                    "gemv",
                ),
            ];
            for (name, src, entry) in jobs {
                gpu.pipeline(&label(name), &src, entry, None)
                    .unwrap_or_else(|e| panic!("{name} {mode:?} failed: {e}"));
            }
            if gpu.supports_binding_array() {
                let batch = [
                    (
                        "gemv_batch_slots",
                        gemv_batch_slots_mode(64, 1024, 4, true, *mode),
                        "gemv_batch_slots",
                    ),
                    (
                        "gemv_norm_batch_slots",
                        gemv_norm_batch_slots_mode(64, 1024, 4, true, 1024, 1024, 1e-5, *mode),
                        "gemv_norm_batch_slots",
                    ),
                ];
                for (name, src, entry) in batch {
                    gpu.pipeline(&label(name), &src, entry, None)
                        .unwrap_or_else(|e| panic!("{name} {mode:?} failed: {e}"));
                }
            }
            for bs in [32usize, 64, 128, 256, 512, 1024] {
                let src = softmax_causal_mode(bs, *mode);
                gpu.pipeline(&label(&format!("softmax{bs}")), &src, "softmax", None)
                    .unwrap_or_else(|e| panic!("softmax bs={bs} {mode:?} failed: {e}"));
            }
            for (bp, bt) in [(64usize, 256usize), (128, 512), (256, 1024)] {
                let src = softmax_causal_tree_mode(bp, bt, *mode);
                gpu.pipeline(&label(&format!("tree{bp}x{bt}")), &src, "softmax", None)
                    .unwrap_or_else(|e| panic!("tree {bp}x{bt} {mode:?} failed: {e}"));
            }
        }
    }
}

#[cfg(test)]
mod stage3_gemm_tests {
    use super::*;

    #[test]
    fn prefill_gemm_variants_are_wgsl_type_safe_and_gqa_aware() {
        let variants = [
            ("plain", false, false, false, false, false, false),
            ("beta", false, true, false, false, false, false),
            ("transb", true, false, false, false, false, false),
            ("causal", false, false, false, true, false, false),
            ("causal_av", true, false, false, false, true, false),
            ("bias", false, false, true, false, false, false),
            ("beta_bias", false, true, true, false, false, false),
            ("w8", false, false, false, false, false, true),
            ("w8_beta", false, true, false, false, false, true),
        ];
        for (name, transb, beta, bias, causal, causal_k, w8) in variants {
            for (tile64, use_f16) in [(false, false), (true, false), (false, true), (true, true)] {
                let src =
                    prefill_gemm_cfg(transb, beta, bias, causal, causal_k, w8, tile64, use_f16);
                let tag = format!(
                    "{name}/{}/{}",
                    if tile64 { "64" } else { "128" },
                    if use_f16 { "f16" } else { "f32" }
                );
                assert_eq!(src.starts_with("enable f16;\n"), use_f16, "{tag}");
                assert!(!src.contains("enable subgroups;"), "{tag}");
                assert!(src.contains("gqa_group: u32 }"), "{tag}");
                assert!(src.contains("wid.z / max(gd.gqa_group, 1u)"), "{tag}");
                if use_f16 {
                    assert!(src.contains("array<vec4<f16>"), "{tag}");
                    assert!(src.contains("vec4<f32>(A4["), "{tag}");
                    assert!(src.contains("vec4<f32>(B4["), "{tag}");
                    assert!(src.contains("var c00: f32 = 0.0;"), "{tag}");
                    assert!(src.contains("fn halve_f16"), "{tag}");
                    assert!(
                        !src.contains("unpack2x16float(pack2x16float(vec2<f32>("),
                        "{tag}"
                    );
                }
                if tile64 {
                    assert!(
                        src.contains(
                            "const BM: u32 = 64u;\nconst BN: u32 = 64u;\nconst BK: u32 = 16u;"
                        ),
                        "{tag}"
                    );
                    assert!(
                        src.contains("const TM: u32 = 4u;\nconst TN: u32 = 4u;"),
                        "{tag}"
                    );
                    assert!(
                        src.contains("const AQ: u32 = 16u;\nconst BQ: u32 = 16u;"),
                        "{tag}"
                    );
                    assert!(!src.contains("A4[c0 + CSTEP]"), "{tag}");
                    assert!(!src.contains("B4[c0 + CSTEP]"), "{tag}");
                } else {
                    assert!(
                        src.contains("const BM: u32 = 128u;\nconst BN: u32 = 128u;"),
                        "{tag}"
                    );
                    assert!(
                        src.contains("const TM: u32 = 8u;\nconst TN: u32 = 8u;"),
                        "{tag}"
                    );
                    assert!(src.contains("A4[c0 + CSTEP]"), "{tag}");
                    assert!(src.contains("B4[c0 + CSTEP]"), "{tag}");
                }
            }
        }
    }

    #[test]
    fn prefill_gemm_legacy_builders_are_bitwise_128_f32() {
        assert_eq!(
            prefill_gemm(false, false),
            prefill_gemm_cfg(false, false, false, false, false, false, false, false)
        );
        assert_eq!(
            prefill_gemm(true, true),
            prefill_gemm_cfg(true, true, false, false, false, false, false, false)
        );
        assert_eq!(
            prefill_gemm_w8(true),
            prefill_gemm_cfg(false, true, false, false, false, true, false, false)
        );
        assert_eq!(
            prefill_gemm_causal(),
            prefill_gemm_cfg(false, false, false, true, false, false, false, false)
        );
        assert_eq!(
            prefill_gemm_causal_av(),
            prefill_gemm_cfg(true, false, false, false, true, false, false, false)
        );
        assert_eq!(
            prefill_gemm_bias(true, false),
            prefill_gemm_cfg(true, false, true, false, false, false, false, false)
        );
    }

    #[test]
    fn vulkan_validates_all_prefill_gemm_variants() {
        let gpu = match pollster::block_on(crate::gpu::Gpu::new_with(
            crate::gpu::DeviceSelector::parse("vulkan").unwrap(),
        )) {
            Ok(gpu) => gpu,
            Err(e) => {
                eprintln!("skip Vulkan GEMM validation test: {e}");
                return;
            }
        };
        let variants = [
            ("plain", false, false, false, false, false, false),
            ("beta", false, true, false, false, false, false),
            ("transb", true, false, false, false, false, false),
            ("causal", false, false, false, true, false, false),
            ("causal_av", true, false, false, false, true, false),
            ("bias", false, false, true, false, false, false),
            ("beta_bias", false, true, true, false, false, false),
            ("w8", false, false, false, false, false, true),
            ("w8_beta", false, true, false, false, false, true),
        ];
        let mut f16_variants = vec![false];
        if gpu.has_f16() {
            f16_variants.push(true);
        }
        for (name, transb, beta, bias, causal, causal_k, w8) in variants {
            for tile64 in [false, true] {
                for use_f16 in &f16_variants {
                    let src = prefill_gemm_cfg(
                        transb, beta, bias, causal, causal_k, w8, tile64, *use_f16,
                    );
                    let label = format!(
                        "stage3_{name}_{}_{}",
                        if tile64 { "64" } else { "128" },
                        if *use_f16 { "f16" } else { "f32" }
                    );
                    gpu.pipeline(&label, &src, "gemm", None)
                        .unwrap_or_else(|e| panic!("{label} failed: {e}"));
                }
            }
        }
    }
}

#[cfg(test)]
mod stage3_gemm_numeric_tests {
    use super::*;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Gd {
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

    struct Lcg(u32);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((self.0 >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
        }
        fn i8(&mut self) -> i8 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((self.0 >> 16) as i32 % 255 - 127) as i8
        }
    }

    fn vulkan() -> Option<crate::gpu::Gpu> {
        match pollster::block_on(crate::gpu::Gpu::new_with(
            crate::gpu::DeviceSelector::parse("vulkan").unwrap(),
        )) {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("skip numeric GEMM test: {e}");
                None
            }
        }
    }

    /// One dispatch of a generated GEMM against a CPU reference.
    ///
    /// `w8` reads `W` as int8 bytes with `scales`, `transb` as `[k][n]`, and
    /// `heads` stacks that many independent `A`/`C` slices; `gqa` is the
    /// shader's `gqa_group` (2 makes every quality head read KV head 0).
    #[allow(clippy::too_many_arguments)]
    fn check(
        gpu: &crate::gpu::Gpu,
        src: &str,
        tile64: bool,
        m: usize,
        n: usize,
        k: usize,
        transb: bool,
        beta: bool,
        bias: bool,
        w8: bool,
        heads: usize,
        gqa: usize,
        label: &str,
    ) {
        let mut rng = Lcg(0x5151_2323);
        let a: Vec<half::f16> = (0..heads * m * k)
            .map(|_| half::f16::from_f32(rng.f() * 0.5))
            .collect();
        let scales: Vec<f32> = (0..n)
            .map(|_| 0.004 + 0.026 * ((rng.f() + 1.0) * 0.5))
            .collect();
        // The `w8` arm's weights are int8 bytes with a per-row scale; the f16
        // arms are plain f16.  Keeping `q` is what lets the reference read the
        // exact operand the kernel dequantizes.
        let q: Vec<i8> = if w8 {
            (0..n * k).map(|_| rng.i8()).collect()
        } else {
            Vec::new()
        };
        let wf: Vec<half::f16> = if w8 {
            q.iter()
                .enumerate()
                .map(|(i, &b)| half::f16::from_f32(b as f32 * scales[i / k]))
                .collect()
        } else {
            (0..n * k)
                .map(|_| half::f16::from_f32(rng.f() * 0.5))
                .collect()
        };
        let bias_v: Vec<half::f16> = (0..n).map(|_| half::f16::from_f32(rng.f() * 0.1)).collect();
        let c_init: Vec<half::f16> = (0..heads * m * n)
            .map(|_| half::f16::from_f32(rng.f() * 0.25))
            .collect();

        // Operand layout: row-major `[n][k]`, or `[k][n]` for `transb`.
        let mut wbuf: Vec<u8> = Vec::new();
        if w8 {
            wbuf = q.iter().map(|&b| b as u8).collect();
        } else if transb {
            for c in 0..k {
                for r in 0..n {
                    wbuf.extend_from_slice(&wf[r * k + c].to_le_bytes());
                }
            }
        } else {
            wbuf = crate::weights::words_bytes(&wf);
        }
        let a_buf = gpu.storage("n.a", (a.len() * 2) as u64);
        let w_buf = gpu.storage("n.w", wbuf.len() as u64);
        let c_buf = gpu.storage("n.c", (c_init.len() * 2) as u64);
        let s_buf = gpu.storage("n.s", (n * 4) as u64);
        let b_buf = gpu.storage("n.b", (n * 2) as u64);
        gpu.upload(&a_buf, &crate::weights::words_bytes(&a));
        gpu.upload(&w_buf, &wbuf);
        gpu.upload(&c_buf, &crate::weights::words_bytes(&c_init));
        gpu.upload(&b_buf, &crate::weights::words_bytes(&bias_v));
        let mut sb = Vec::with_capacity(n * 4);
        for s in &scales {
            sb.extend_from_slice(&s.to_le_bytes());
        }
        gpu.upload(&s_buf, &sb);

        let gd = Gd {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            ldc: n as u32,
            bsa: (m * k / 2) as u32,
            bsb: (n * k / 2) as u32,
            bsc: (m * n) as u32,
            beta: u32::from(beta),
            row0: 0,
            lda: k as u32,
            q_off: 0,
            gqa_group: gqa as u32,
        };
        let u_gd = gpu.uniform("n.gd", 256);
        gpu.upload(&u_gd, bytemuck::bytes_of(&gd));
        let pipe = gpu
            .pipeline(label, src, "gemm", None)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: w_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: c_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &u_gd,
                    offset: 0,
                    size: std::num::NonZeroU64::new(64),
                }),
            },
        ];
        if bias {
            entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: b_buf.as_entire_binding(),
            });
        } else if w8 {
            entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: s_buf.as_entire_binding(),
            });
        }
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("n.bg"),
            layout: &pipe.get_bind_group_layout(0),
            entries: &entries,
        });
        let tile = if tile64 { 64 } else { 128 };
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&pipe);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups((n / tile) as u32, (m / tile) as u32, heads as u32);
        }
        gpu.queue.submit([enc.finish()]);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let got = gpu
            .readback(&c_buf, (heads * m * n * 2) as u64)
            .expect("readback");
        let got: Vec<f32> = got
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();

        // Reference: the reading head's slice of A against KV head
        // `z / gqa` of W (so `gqa = 2` proves the divisor picks head 0 twice).
        let mut worst = 0.0f32;
        for z in 0..heads {
            let kv = z / gqa.max(1);
            let _ = kv;
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for c in 0..k {
                        let av = a[(z * m + i) * k + c].to_f32();
                        // Every `heads` slice shares KV head 0 when `gqa == 2`.
                        let wv = if w8 {
                            q[j * k + c] as f32 * scales[j]
                        } else {
                            wf[j * k + c].to_f32()
                        };
                        acc += (av * wv) as f64;
                    }
                    let mut want = acc as f32;
                    if bias {
                        want += bias_v[j].to_f32();
                    }
                    if beta {
                        want += c_init[(z * m + i) * n + j].to_f32();
                    }
                    let want = half::f16::from_f32(want).to_f32();
                    let d = (got[(z * m + i) * n + j] - want).abs();
                    if d > worst {
                        worst = d;
                    }
                }
            }
        }
        assert!(
            worst <= 0.05,
            "{label}: max |GPU - CPU| = {worst:.4} (m={m} n={n} k={k} transb={transb} beta={beta} bias={bias} w8={w8} heads={heads})"
        );
    }

    #[test]
    fn prefill_gemm_matches_cpu_on_both_tiles() {
        let Some(gpu) = vulkan() else { return };
        let mut f16s = vec![false];
        if gpu.has_f16() {
            f16s.push(true);
        }
        for &use_f16 in &f16s {
            for &tile64 in &[false, true] {
                let tile = if tile64 { 64 } else { 128 };
                for (transb, beta, bias) in [
                    (false, false, false),
                    (true, false, false),
                    (false, true, false),
                    (false, false, true),
                    (false, true, true),
                ] {
                    let src =
                        prefill_gemm_cfg(transb, beta, bias, false, false, false, tile64, use_f16);
                    let label = format!("num_{tile}_{transb}_{beta}_{bias}_{use_f16}");
                    check(
                        &gpu, &src, tile64, tile, tile, 32, transb, beta, bias, false, 1, 1, &label,
                    );
                }
            }
        }
    }

    #[test]
    fn prefill_gemm_w8_matches_cpu_on_both_tiles() {
        let Some(gpu) = vulkan() else { return };
        let mut f16s = vec![false];
        if gpu.has_f16() {
            f16s.push(true);
        }
        for &use_f16 in &f16s {
            for &tile64 in &[false, true] {
                let tile = if tile64 { 64 } else { 128 };
                for beta in [false, true] {
                    let src =
                        prefill_gemm_cfg(false, beta, false, false, false, true, tile64, use_f16);
                    let label = format!("numw8_{tile}_{beta}_{use_f16}");
                    check(
                        &gpu, &src, tile64, tile, tile, 32, false, beta, false, true, 1, 1, &label,
                    );
                }
            }
        }
    }

    /// Two quality heads over one KV head: both batch slices must read the
    /// *same* operand, which is exactly what `gqa_group` makes `wid.z` do.
    #[test]
    fn prefill_gemm_gqa_group_reads_one_kv_head() {
        let Some(gpu) = vulkan() else { return };
        for &tile64 in &[false, true] {
            let tile = if tile64 { 64 } else { 128 };
            let src = prefill_gemm_cfg(false, false, false, false, false, false, tile64, false);
            let label = format!("numgqa_{tile}");
            check(
                &gpu, &src, tile64, tile, tile, 32, false, false, false, false, 2, 2, &label,
            );
        }
    }
}
