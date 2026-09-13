//! WGSL sources for the decode chain.
//!
//! Every kernel here is a structural port of the matching CUDA kernel in
//! `src/kernels/kernels.cu`, and the *arithmetic order is deliberately mirrored*:
//! lane-strided accumulation, the same reduction tree, the same operand grouping.
//! f32 addition is not associative, so a different order would give a different
//! last-ulp result, and the CUDA backend has already shown that last-ulp
//! differences get amplified by 600+ autoregressive steps into different text
//! (see `ROADMAP.md` §1.4.2).  Mirroring the order is what makes a token-id-level
//! comparison against the CUDA golden meaningful.
//!
//! Two layout conventions are shared with the CUDA backend so values are
//! bit-comparable:
//!
//! * **activations** are f16 stored as `array<u32>`, word `j` holding elements
//!   `2j` / `2j+1` — byte-identical to CUDA's `__half2` view;
//! * **weights** are f16 stored as `array<vec4<u32>>` (8 halves = 16 B per
//!   element) — byte-identical to CUDA's `uint4` view, and the reason a plain
//!   `array<f16>` binding is never needed (Pascal exposes 16-bit storage but not
//!   `shaderFloat16`).

#![allow(dead_code)]

pub const LOG2_E: f32 = std::f32::consts::LOG2_E;

/// Read element `i` out of a packed f16 word (`v.x` = even index, `v.y` = odd).
const HALF_AT: &str = "\
fn half_at(v: vec2<f32>, i: u32) -> f32 { return select(v.y, v.x, (i & 1u) == 0u); }
";

/// `rms_norm_f16` — one workgroup per row, block tree reduction over `LAST`.
///
/// Bits that matter: `local += vx*vx + vy*vy` per `__half2` word (CUDA's odd/even
/// tail is irrelevant because every model dim is even), the `s`-halving tree, and
/// the two passes reading `x` twice exactly as CUDA does.
pub fn rms_norm(last: usize, bs: usize) -> String {
    assert!(last % 2 == 0 && last >= bs, "rms_norm: last must be even and >= bs");
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

/// `gemv_f16` -- warp-per-row, `uint4` (8-half) lane-strided loads on both the
/// weight row and the activation vector, four independent f32 accumulators,
/// 5-round xor butterfly over 32-lane warps, residual add folded into the epilogue.
///
/// `n` must be a multiple of 8 (all model shapes are) so no partial workgroup
/// exists; `k/8` must be a multiple of 32 so the granule loop divides evenly.
pub fn gemv(n: usize, k: usize, accum: bool) -> String {
    assert_eq!(n % 8, 0, "gemv: rows must be a multiple of 8");
    let kg = k / 8;
    assert_eq!(kg % 32, 0, "gemv: k/8 must be a multiple of 32");
    assert_eq!(k % 8, 0, "gemv: k must be a multiple of 8");
    let tiles = kg / 32;
    assert_eq!(tiles % 4, 0, "gemv: k/256 must be a multiple of 4 (unrolled x4)");
    let accum_lit = if accum { 1u32 } else { 0u32 };
    format!(
        "@group(0) @binding(0) var<storage, read>       Wt: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       X:  array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> Y:  array<u32>;

const KG: u32 = {kg}u;
const TILES: u32 = {tiles}u;
const ACCUM: u32 = {accum_lit}u;

var<workgroup> bt0: array<f32, 256>;
var<workgroup> bt1: array<f32, 256>;
var<workgroup> rows_out: array<f32, 8>;

/// 5-round xor butterfly over the 32 lanes of one warp -- the exact tree
/// `__shfl_xor_sync(acc, [16,8,4,2,1])` produces, done through shared memory
/// because WGSL has no portable warp shuffle on this wgpu version.
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
    bt1[lid] = t;
    workgroupBarrier();
    return t;
}}

@compute @workgroup_size(256)
fn gemv(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let lane = lid.x & 31u;
    let warp = lid.x >> 5u;
    let row = wgid.x * 8u + warp;

    let wbase = row * KG;
    var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
    // Tiles unrolled x4 with prefetched loads: each warp keeps 4 (Wt, X) pairs
    // in flight to hide DRAM latency (rows == warps leaves the machine
    // underfilled on the small projections).  The fma chain still consumes
    // tiles in ascending order, so the reduction bits are unchanged.
    var i = lane;
    for (var g = 0u; g < TILES; g = g + 4u) {{
        let wv0 = Wt[wbase + i];
        let xv0 = X[i];
        let wv1 = Wt[wbase + i + 32u];
        let xv1 = X[i + 32u];
        let wv2 = Wt[wbase + i + 64u];
        let xv2 = X[i + 64u];
        let wv3 = Wt[wbase + i + 96u];
        let xv3 = X[i + 96u];
        let w00 = unpack2x16float(wv0.x); let x00 = unpack2x16float(xv0.x);
        let w01 = unpack2x16float(wv0.y); let x01 = unpack2x16float(xv0.y);
        let w02 = unpack2x16float(wv0.z); let x02 = unpack2x16float(xv0.z);
        let w03 = unpack2x16float(wv0.w); let x03 = unpack2x16float(xv0.w);
        a0 = fma(w00.x, x00.x, fma(w00.y, x00.y, a0));
        a1 = fma(w01.x, x01.x, fma(w01.y, x01.y, a1));
        a2 = fma(w02.x, x02.x, fma(w02.y, x02.y, a2));
        a3 = fma(w03.x, x03.x, fma(w03.y, x03.y, a3));
        let w10 = unpack2x16float(wv1.x); let x10 = unpack2x16float(xv1.x);
        let w11 = unpack2x16float(wv1.y); let x11 = unpack2x16float(xv1.y);
        let w12 = unpack2x16float(wv1.z); let x12 = unpack2x16float(xv1.z);
        let w13 = unpack2x16float(wv1.w); let x13 = unpack2x16float(xv1.w);
        a0 = fma(w10.x, x10.x, fma(w10.y, x10.y, a0));
        a1 = fma(w11.x, x11.x, fma(w11.y, x11.y, a1));
        a2 = fma(w12.x, x12.x, fma(w12.y, x12.y, a2));
        a3 = fma(w13.x, x13.x, fma(w13.y, x13.y, a3));
        let w20 = unpack2x16float(wv2.x); let x20 = unpack2x16float(xv2.x);
        let w21 = unpack2x16float(wv2.y); let x21 = unpack2x16float(xv2.y);
        let w22 = unpack2x16float(wv2.z); let x22 = unpack2x16float(xv2.z);
        let w23 = unpack2x16float(wv2.w); let x23 = unpack2x16float(xv2.w);
        a0 = fma(w20.x, x20.x, fma(w20.y, x20.y, a0));
        a1 = fma(w21.x, x21.x, fma(w21.y, x21.y, a1));
        a2 = fma(w22.x, x22.x, fma(w22.y, x22.y, a2));
        a3 = fma(w23.x, x23.x, fma(w23.y, x23.y, a3));
        let w30 = unpack2x16float(wv3.x); let x30 = unpack2x16float(xv3.x);
        let w31 = unpack2x16float(wv3.y); let x31 = unpack2x16float(xv3.y);
        let w32 = unpack2x16float(wv3.z); let x32 = unpack2x16float(xv3.z);
        let w33 = unpack2x16float(wv3.w); let x33 = unpack2x16float(xv3.w);
        a0 = fma(w30.x, x30.x, fma(w30.y, x30.y, a0));
        a1 = fma(w31.x, x31.x, fma(w31.y, x31.y, a1));
        a2 = fma(w32.x, x32.x, fma(w32.y, x32.y, a2));
        a3 = fma(w33.x, x33.x, fma(w33.y, x33.y, a3));
        i = i + 128u;
    }}
    let acc = (a0 + a1) + (a2 + a3);
    let r = bfly(acc, lid.x, lane);
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

/// Sum of squares + tree reduction + inverse RMS, exactly as the CUDA block does it
/// (one element per thread, `s` halving from `bs/2`).
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

/// `fused_gqa_decode_f16` — single-block-per-(b,q_head) flash-style attention.
/// `bs` mirrors CUDA's adaptive choice (256 for `cur_len <= 512`, else 512);
/// CUDA also has a 1024 case that only the split path ever uses.
// Bit-exact port of CUDA's expf (sm_61, CUDA 12.8 — PTX probed) for the decode
// attention softmax.  WGSL's builtin exp() differs from CUDA expf by 1..58 ulp
// on ~92% of inputs, which the decode chain amplifies into token flips at
// near-tie steps (q06_180s_en @ 363).  This port reproduces CUDA expf bit for
// bit over the whole finite domain (probe: wgpu/exp_probe, 0/2^20 mismatch):
//   fma.rn(x,c1,0.5) -> sat -> fma.rm(f5,252,12582913) -> f8 grid index
//   f12 = fma.rn(x,LOG2E,126-Q); f14 = fma.rn(x,c2,f12)
//   result = ex2.approx.ftz(f14) * 2^(Q-126)
// The three correctly-rounded fmas and the final power-of-two scaling are done
// in PURE u32 integer arithmetic (integer ops are exact and associative, so no
// driver transform — FFMA contraction, reassociation, CSE — can perturb them;
// empirically the driver *does* fold float-domain two_sum residuals to zero).
// WGSL exp2 == ex2.approx.ftz bit-for-bit (measured), incl. the subnormal
// output flush; the remaining exactness lives in scale_pow2's integer GRS.
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
// subnormal outputs; CUDA's mul.rn does not).
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

@group(0) @binding(0) var<storage, read>       Q:   array<u32>;
@group(0) @binding(1) var<storage, read>       KC:  array<u32>;
@group(0) @binding(2) var<storage, read>       VC:  array<u32>;
@group(0) @binding(3) var<storage, read_write> Out: array<u32>;
@group(0) @binding(4) var<uniform>             cfg: Cfg;

const D: u32 = {d}u;
const D2: u32 = {d2}u;
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

    // Stage 1 — scores[t] = (Q . K[t]) * scale
    for (var t = lid.x; t < cfg.cur_len; t = t + BS) {{
        var dot = 0.0;
        let row = kbase + t * D2;
        for (var j2 = 0u; j2 < D2; j2 = j2 + 1u) {{
            let qv = unpack2x16float(Q[qbase + j2]);
            let kv = unpack2x16float(KC[row + j2]);
            dot = dot + (qv.x * kv.x + qv.y * kv.y);
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

    // Stage 4 — partial[t_idx][j] then cross-t_chunk merge
    let j_idx = lid.x % D;
    let t_idx = lid.x / D;
    var acc = 0.0;
    for (var t = t_idx; t < cfg.cur_len; t = t + TCH) {{
        let row = kbase + t * D2;
        acc = acc + sc[t] * half_at(unpack2x16float(VC[row + (j_idx >> 1u)]), j_idx);
    }}
    partial[t_idx * D + j_idx] = acc;
    workgroupBarrier();

    // CUDA writes one element per thread; writing whole f16 words here keeps the
    // per-element arithmetic (and its order) identical.
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
        rep = nqh / nkvh,
    )
}

/// `silu_mul_split_f16` — `gu` holds `[gate | up]` per row; CUDA uses `__expf`,
/// which is `ex2.approx(x * log2(e))`, so `exp2` is used here rather than `exp`.
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

struct Cfg {{ inter2: u32, total2: u32, _a: u32, _b: u32 }};

const INTER2: u32 = {inter2}u;
const LOG2E: f32 = {log2e};
const THREADS: u32 = {threads}u;

@compute @workgroup_size({threads})
fn silu_mul_split(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
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

/// `fused_gqa_decode_split_p1_f16` — long-context (cur_len > 1024) split-K
/// attention, phase 1: one workgroup per (q_head, chunk).  Chunk-local scores →
/// chunk max/sum via the shared block-reduction tree → unnormalized partial
/// numerator.  Block size is fixed at 256 (CUDA's choice) regardless of chunk
/// width; `t_split = 256/d = 2` threads cooperate per output element.
///
/// Empty chunks (t_start >= cur_len) write max=-inf, sum=0, partial=0 exactly as
/// CUDA does — the merge kernel's `> -inf` guard depends on it.
pub fn gqa_decode_split_p1(nqh: usize, nkvh: usize, d: usize, chunk: usize) -> String {
    assert_eq!(d % 2, 0);
    let t_split = 256 / d;
    format!(
        "{HALF_AT}
{EXP_BT}
struct Cfg {{ cur_len: u32, max_seq: u32, scale: f32, n_chunks: u32 }};

@group(0) @binding(0) var<storage, read>       Q:    array<u32>;
@group(0) @binding(1) var<storage, read>       KC:   array<u32>;
@group(0) @binding(2) var<storage, read>       VC:   array<u32>;
@group(0) @binding(3) var<storage, read_write> POut: array<f32>;
@group(0) @binding(4) var<storage, read_write> PMax: array<f32>;
@group(0) @binding(5) var<storage, read_write> PSum: array<f32>;
@group(0) @binding(6) var<uniform>             cfg:  Cfg;

const D: u32 = {d}u;
const D2: u32 = {d2}u;
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
    for (var t = lid.x; t < chunk_len; t = t + BS) {{
        var dot = 0.0;
        let row = kbase + (t_start + t) * D2;
        for (var j2 = 0u; j2 < D2; j2 = j2 + 1u) {{
            let qv = unpack2x16float(Q[qbase + j2]);
            let kv = unpack2x16float(KC[row + j2]);
            dot = dot + (qv.x * kv.x + qv.y * kv.y);
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

    // Stage 4 — partial numerator, unnormalized; V read at the global position
    let j_idx = lid.x % D;
    let t_idx = lid.x / D;
    var acc = 0.0;
    for (var t = t_idx; t < chunk_len; t = t + T_SPLIT) {{
        let row = kbase + (t_start + t) * D2;
        acc = acc + sc[t] * half_at(unpack2x16float(VC[row + (j_idx >> 1u)]), j_idx);
    }}
    partial[t_idx * D + j_idx] = acc;
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
        t_split = t_split,
        rep = nqh / nkvh,
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
/// wins a tie (identical tie-breaking to CUDA, which matters for reproducibility).
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
    format!(
        "struct Cfg {{ slot: u32, d2: u32, _a: u32, _b: u32 }};

@group(0) @binding(0) var<storage, read>       Table: array<u32>;
@group(0) @binding(1) var<storage, read>       Ids:   array<i32>;
@group(0) @binding(2) var<storage, read_write> Out:   array<u32>;
@group(0) @binding(3) var<uniform>             cfg:   Cfg;

const BS: u32 = 1024u;

@compute @workgroup_size(1024)
fn embed(@builtin(local_invocation_id) lid: vec3<u32>) {{
    let id = u32(Ids[cfg.slot]);
    let base = id * cfg.d2;
    for (var w = lid.x; w < cfg.d2; w = w + BS) {{
        Out[w] = Table[base + w];
    }}
}}
"
    )
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
    // WGSL hex float literal: 0x1.<mantissa>p<exponent>.  Six hex digits are 24
    // fractional bits but the f32 mantissa is 23, so emit `man << 1` (always <
    // 2^24) — otherwise every non-dyadic constant comes out half-ish wrong
    // (1.4427 -> 1.2213, which silently bent every SiLU in the network).
    let man = man << 1;
    let e = exp - 127;
    format!("{sign}0x1.{man:06x}p{e}")
}


/// Prefill GEMM:  C[m,n] f16 = A[m,k] f16 × W[n,k]ᵀ (or × W[n,k] when
/// `transb`), f32 accumulate, one f16 rounding — the wgpu stand-in for the
/// CUDA engine's cuBLAS calls.  Tile 128×128 per workgroup (16×16 threads,
/// 8×8 micro-tile), BK=16, software-pipelined (next tile prefetched into
/// registers during compute).  `beta=1` folds the residual add into the
/// epilogue (mirrors cuBLAS beta=1: acc + f16(C_in), one rounding).
/// `ldc` = C row stride in elements; `bsa/bsb/bsc` = per-batch strides with
/// batch index wgid.z (batch of 1 for plain GEMMs).  `transb=1` reads W as a
/// [k, n] f16 matrix instead of [n, k] (attention AV: V is [cur, d]).
pub fn prefill_gemm(transb: bool, beta: bool) -> String {
    let tm = 8usize;
    let tn = 8usize;
    let bk = 16usize;
    let bm = 16 * tm;
    let bn = 16 * tn;
    let pad = bk + 1;
    let n_as = bm * bk / 256;
    let n_bs = bn * bk / 256;

    let mut s = String::new();
    s.push_str(
        "struct GDims { m: u32, n: u32, k: u32, ldc: u32, bsa: u32, bsb: u32, bsc: u32, beta: u32 };\n\
         @group(0) @binding(0) var<storage, read>       A: array<u32>;\n\
         @group(0) @binding(1) var<storage, read>       W: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> C: array<u32>;\n\
         @group(0) @binding(3) var<uniform>             gd: GDims;\n",
    );
    s.push_str(&format!(
        "const BM: u32 = {bm}u;\nconst BN: u32 = {bn}u;\nconst BK: u32 = {bk}u;\n\
         const PAD: u32 = {pad}u;\nconst TM: u32 = {tm}u;\nconst TN: u32 = {tn}u;\n\
         const TRANSB: u32 = {}u;\nconst BETA: u32 = {}u;\n",
        u32::from(transb),
        u32::from(beta),
    ));
    s.push_str(&format!("var<workgroup> As: array<f32, {}>;\n", bm * pad));
    s.push_str(&format!("var<workgroup> Bs: array<f32, {}>;\n", bn * pad));
    s.push_str(
        "fn halve(w: u32, odd: bool) -> f32 {\n\
         \x20 let p = unpack2x16float(w);\n\
         \x20 return select(p.x, p.y, odd);\n\
         }\n",
    );

    // A element (m-row r, k-col c): word abase + r*kk + c/2, half c&1.
    // W element: transb=0 -> [n, k] f16: word wb + r*kk + c/2, half c&1;
    //   transb=1 -> [k, n] f16: element (r, c) at (k0+c)*(n/2) + (n0+r)/2,
    //   half (n0+r)&1 (n padded even, n0 multiple of BN).
    s.push_str(
        "@compute @workgroup_size(16, 16)\n\
         fn gemm(@builtin(workgroup_id) wid: vec3<u32>,\n\
                 @builtin(local_invocation_id) lid: vec3<u32>) {\n\
         let tx = lid.x;\n let ty = lid.y;\n\
         let m0 = wid.y * BM;\n let n0 = wid.x * BN;\n let kk = gd.k / 2u;\n\
         let abase = wid.z * gd.bsa;\n let wb = wid.z * gd.bsb;\n\
         let cbase = wid.z * gd.bsc;\n",
    );

    for i in 0..tm {
        for j in 0..tn {
            s.push_str(&format!("var c{i}{j} = 0.0;\n"));
        }
    }

    let load_as = |kx: &str| -> String {
        let mut t = String::new();
        for e in 0..n_as {
            t.push_str(&format!(
                "  As[(ty + {}u) * PAD + tx] = halve(A[abase + (m0 + ty + {}u) * kk + ({kx} + tx) / 2u], (tx & 1u) == 1u);\n",
                e * 16, e * 16
            ));
        }
        t
    };
    let load_bs = |kx: &str| -> String {
        let mut t = String::new();
        if !transb {
            for e in 0..n_bs {
                t.push_str(&format!(
                    "  Bs[(ty + {}u) * PAD + tx] = halve(W[wb + (n0 + ty + {}u) * kk + ({kx} + tx) / 2u], (tx & 1u) == 1u);\n",
                    e * 16, e * 16
                ));
            }
        } else {
            for e in 0..n_bs {
                t.push_str(&format!(
                    "  Bs[(ty + {}u) * PAD + tx] = halve(W[wb + ({kx} + tx) * (gd.n / 2u) + (n0 + ty + {}u) / 2u], ((n0 + ty + {}u) & 1u) == 1u);\n",
                    e * 16, e * 16, e * 16
                ));
            }
        }
        t
    };
    let pf_as = |kx: &str| -> String {
        let mut t = String::new();
        for e in 0..n_as {
            t.push_str(&format!(
                "   pfa{e} = A[abase + (m0 + ty + {}u) * kk + ({kx} + tx) / 2u];\n", e * 16
            ));
        }
        t
    };
    let pf_bs = |kx: &str| -> String {
        let mut t = String::new();
        if !transb {
            for e in 0..n_bs {
                t.push_str(&format!(
                    "   pfb{e} = W[wb + (n0 + ty + {}u) * kk + ({kx} + tx) / 2u];\n", e * 16
                ));
            }
        } else {
            for e in 0..n_bs {
                t.push_str(&format!(
                    "   pfb{e} = W[wb + ({kx} + tx) * (gd.n / 2u) + (n0 + ty + {}u) / 2u];\n", e * 16
                ));
            }
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
    let store_as = || {
        let mut t = String::new();
        for e in 0..n_as {
            t.push_str(&format!(
                "   As[(ty + {}u) * PAD + tx] = halve(pfa{e}, (tx & 1u) == 1u);\n", e * 16
            ));
        }
        t
    };
    let store_bs = || {
        let mut t = String::new();
        for e in 0..n_bs {
            // the prefetched word's halves follow the SAME layout as load_bs:
            // transb=0 -> k parity (= tx parity); transb=1 -> n parity of the
            // n-element this word was fetched for.  Using tx parity for
            // transb=1 swapped every odd-n B value in every prefetched tile
            // (tile 0 loads directly and is fine — AV output rows 0-15 exact,
            // rows 16+ garbage).
            let sel = if transb {
                format!("((n0 + ty + {}u) & 1u) == 1u", e * 16)
            } else {
                "(tx & 1u) == 1u".to_string()
            };
            t.push_str(&format!(
                "   Bs[(ty + {}u) * PAD + tx] = halve(pfb{e}, {sel});\n",
                e * 16
            ));
        }
        t
    };
    let compute = || {
        let mut t = String::new();
        t.push_str("  var q: u32 = 0u;\n  loop {\n   if (q >= BK) { break; }\n");
        for i in 0..tm {
            t.push_str(&format!("   let a{i} = As[(ty * {tm}u + {i}u) * PAD + q];\n"));
        }
        for j in 0..tn {
            t.push_str(&format!("   let b{j} = Bs[(tx * {tn}u + {j}u) * PAD + q];\n"));
        }
        for i in 0..tm {
            for j in 0..tn {
                t.push_str(&format!("   c{i}{j} = c{i}{j} + a{i} * b{j};\n"));
            }
        }
        t.push_str("   q = q + 1u;\n  }\n");
        t
    };

    s.push_str(" var k0: u32 = 0u;\n");
    s.push_str(&load_as("k0"));
    s.push_str(&load_bs("k0"));
    s.push_str(" workgroupBarrier();\n");
    s.push_str(" loop {\n  if (k0 >= gd.k) { break; }\n");
    s.push_str("  let kn = k0 + BK;\n");
    s.push_str(&pf_decl());
    s.push_str("  if (kn < gd.k) {\n");
    s.push_str(&pf_as("kn"));
    s.push_str(&pf_bs("kn"));
    s.push_str("  }\n");
    s.push_str(&compute());
    s.push_str("  workgroupBarrier();\n");
    s.push_str("  if (kn < gd.k) {\n");
    s.push_str(&store_as());
    s.push_str(&store_bs());
    s.push_str("  }\n  workgroupBarrier();\n");
    s.push_str("  k0 = kn;\n }\n");

    // epilogue: f16 words along n; a thread owns TN consecutive columns so
    // pairs stay in-thread.  beta=1 adds the f16 C_in before the rounding.
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
            s.push_str(&format!(
                "var v{i}{e} = vec2<f32>(c{i}{je}, c{i}{jo});\n"
            ));
            s.push_str(&format!(
                "if (BETA == 1u) {{\n  let old = unpack2x16float(C[we{i}_{e}]);\n  v{i}{e} = v{i}{e} + old;\n}}\n"
            ));
            s.push_str(&format!(
                "C[we{i}_{e}] = pack2x16float(v{i}{e});\n"
            ));
        }
    }
    s.push_str("}\n");
    s
}

/// Causal scaled softmax over prefill scores — port of
/// `softmax_scaled_causal_f16`.  One workgroup per score row; block size `bs`
/// matches `block_for_reduction(n)` so the reduction trees line up.
/// Row `p` (of the head) attends `min(p + 1, valid)` positions; columns
/// `valid..n_pad` are written zero so downstream GEMMs read zeros.
pub fn softmax_causal(bs: usize) -> String {
    format!(
        "struct Cfg {{ n_w: u32, n_x: u32, valid: u32, m: u32, mp: u32, scale: f32 }};

@group(0) @binding(0) var<storage, read>       X:   array<u32>;
@group(0) @binding(1) var<storage, read_write> Out: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;

const BS: u32 = {bs}u;

{HALF_AT}
var<workgroup> red_max: array<f32, {bs}>;
var<workgroup> red_sum: array<f32, {bs}>;

@compute @workgroup_size({bs})
fn softmax(@builtin(workgroup_id) wgid: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>) {{
    // X rows are n_x-strided (tile-padded scores, one row per head*pos);
    // Out uses the BATCHED [head][mp][cur16] layout the AV GEMM expects.
    let head = wgid.x / cfg.m;
    let pos = wgid.x % cfg.m;
    let base_x = (head * cfg.mp + pos) * cfg.n_x;
    let base_o = (head * cfg.mp + pos) * cfg.n_w;
    let row_in_head = pos;
    let valid = min(cfg.valid, row_in_head + 1u);
    let scale = cfg.scale;

    var lmax = bitcast<f32>(0xFF800000u);
    for (var j = lid.x; j < valid; j = j + BS) {{
        let v = half_at(unpack2x16float(X[base_x + (j >> 1u)]), j);
        let sc = v * scale;
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
        let v = half_at(unpack2x16float(X[base_x + (j >> 1u)]), j);
        lsum = lsum + exp(v * scale - row_max);
    }}
    red_sum[lid.x] = lsum;
    workgroupBarrier();
    for (var sh = BS >> 1u; sh > 0u; sh = sh >> 1u) {{
        if (lid.x < sh) {{ red_sum[lid.x] = red_sum[lid.x] + red_sum[lid.x + sh]; }}
        workgroupBarrier();
    }}
    let inv_sum = 1.0 / red_sum[0];
    workgroupBarrier();

    // one thread per f16 word — no read-modify-write races; the per-element
    // arithmetic is identical to CUDA's one-element-per-thread form
    for (var w = lid.x; w < cfg.n_w; w = w + BS) {{
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
        bs = bs,
    )
}

/// repeat_kv:  K cache `[nkvh, max_seq, hd]` rows `0..cur` duplicated per GQA
/// group into `[nqh, cur, hd]` (CUDA `repeat_kv_from_cache`).
pub fn repeat_kv(nrep: usize) -> String {
    format!(
        "struct Cfg {{ nkvh: u32, max_seq: u32, cur: u32, hd: u32, npw: u32, _a: u32 }};

@group(0) @binding(0) var<storage, read>       Cache: array<u32>;
@group(0) @binding(1) var<storage, read_write> Out:   array<u32>;
@group(0) @binding(2) var<uniform>             cfg:   Cfg;

const NREP: u32 = {nrep}u;

@compute @workgroup_size(256)
fn repeat_kv(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
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
