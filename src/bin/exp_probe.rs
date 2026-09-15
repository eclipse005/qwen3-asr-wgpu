use anyhow::Result;
use qwen3_asr_wgpu::gpu::Gpu;

const SHADER: &str = r#"

struct P { n: u32, c1: f32, log2e: f32, c2: f32 };

@group(0) @binding(0) var<storage, read>       X:   array<f32>;
@group(0) @binding(1) var<storage, read>       Y:   array<f32>;
@group(0) @binding(2) var<storage, read>       Z:   array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<u32>;
@group(0) @binding(4) var<uniform>             p:   P;

// index of the highest set bit (v != 0)
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

// 64-bit right shift; .z = OR of every shifted-out bit (sticky)
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

// 64-bit left shift (caller guarantees no significant bit is lost)
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

// OR of all bits at positions < pos0
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

// correctly-rounded f32 fma: rn(a*b + c), PURE u32/i32 integer arithmetic in
// the exactness path — integer ops are exact and associative, so no driver
// transform (contraction / reassociation / CSE) can change the result.
// Finite inputs (inf/nan take an unvalidated mul+add fallback).
fn fma_int(a: f32, b: f32, c: f32) -> f32 {
    let ba = bitcast<u32>(a);
    let bb = bitcast<u32>(b);
    let bc = bitcast<u32>(c);
    let ea0 = i32((ba >> 23u) & 0xFFu);
    let eb0 = i32((bb >> 23u) & 0xFFu);
    let ec0 = i32((bc >> 23u) & 0xFFu);
    if (ea0 == 255 || eb0 == 255 || ec0 == 255) {
        return a * b + c;
    }
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
    if (ma == 0u || mb == 0u) { return c; }   // a*b == 0: rn(c) == c

    // exact product mp = ma*mb as 48 bits via 12-bit limbs
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
    var ep = ea + eb - 300;            // value_P = mp * 2^ep, mp in [2^46, 2^48)
    if (pHi < 0x8000u) {               // normalize mp to [2^47, 2^48)
        pHi = (pHi << 1u) | (pLo >> 31u);
        pLo = pLo << 1u;
        ep = ep - 1;
    }
    let EP = ep + 47;                  // leading-bit exponent of P

    // frame: value = M * 2^(E-63); headroom 4 => any signed add fits in 64 bits
    var E = EP + 4;
    var hasC = (mc != 0u);
    if (hasC) { E = max(E, ec - 127 + 4); }
    let shP = E - EP;                  // >= 4
    var sticky = 0u;
    var Mp = shl64(pHi, pLo, 16 - shP);
    if (shP > 16) {
        let r = shr64(pHi, pLo, shP - 16);
        Mp = vec2<u32>(r.x, r.y);
        sticky = r.z;
    }
    var McHi = 0u; var McLo = 0u;
    if (hasC) {
        let shC = E - (ec - 127);      // >= 4
        if (shC <= 40) {
            let r = shl64(0u, mc, 40 - shC);
            McHi = r.x; McLo = r.y;
        } else {
            let r = shr64(0u, mc, shC - 40);
            McHi = r.x; McLo = r.y;
            sticky = sticky | r.z;
        }
    }

    // signed 64-bit add of the aligned terms
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
        if (eq) { return bitcast<f32>(0u); }   // exact cancellation -> +0 (RN)
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
    // subnormal: k = round(m * 2^(F+149))
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

// Knuth two_sum (float domain — reference path, breaks under driver CSE)
fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = bitcast<f32>(bitcast<u32>(a + b));
    let ap = bitcast<f32>(bitcast<u32>(s - b));
    let bp = bitcast<f32>(bitcast<u32>(s - ap));
    let e = bitcast<f32>(bitcast<u32>((a - ap) + (b - bp)));
    return vec2<f32>(s, e);
}

fn fma_soft(a: f32, b: f32, c: f32) -> f32 {
    let p = bitcast<f32>(bitcast<u32>(a * b));
    let bxa = bitcast<u32>(a);
    let ahi = bitcast<f32>(bxa & 0xFFFFF000u);
    let alo = bitcast<f32>(bitcast<u32>(a - ahi));
    let bxb = bitcast<u32>(b);
    let bhi = bitcast<f32>(bxb & 0xFFFFF000u);
    let blo = bitcast<f32>(bitcast<u32>(b - bhi));
    let t1 = bitcast<f32>(bitcast<u32>(ahi * bhi - p));
    let t2 = bitcast<f32>(bitcast<u32>(t1 + ahi * blo));
    let t3 = bitcast<f32>(bitcast<u32>(t2 + alo * bhi));
    let e = bitcast<f32>(bitcast<u32>(t3 + alo * blo));
    let s12 = two_sum(p, c);
    let l12 = two_sum(e, s12.y);
    let r12 = two_sum(s12.x, l12.x);
    let v12 = two_sum(r12.y, l12.y);
    let v1 = v12.x;
    let v2 = v12.y;
    let sb = bitcast<u32>(r12.x);
    let exp_r = i32((sb >> 23u) & 0xFFu);
    if (exp_r == 0) { return r12.x + v1; }
    let half = bitcast<f32>((u32(exp_r - 24)) << 23u);
    let av = abs(v1);
    var rr = r12.x;
    if (av > half || (av == half && (v2 != 0.0 || (sb & 1u) == 1u))) {
        rr = bitcast<f32>(select(sb - 1u, sb + 1u, v1 > 0.0));
    }
    return rr;
}

// rn(A * 2^(q-126)) for normal A > 0, q in [0,252]: f15 is a power of two, so
// the exact product only re-biases the exponent; assemble bits directly and
// do integer GRS rounding when the result lands in the subnormal grid.
// Replaces OpFMul, which the driver compiles with FTZ (a correctly-rounded
// multiply does not flush), making the final multiply bit-exact as well.
fn scale_pow2(Abits: u32, q: u32) -> u32 {
    let sgn = Abits & 0x80000000u;
    if ((Abits & 0x7FFFFFFFu) == 0u) { return sgn; }   // A == 0 (exp2 flushed)
    let mA = (Abits & 0x7FFFFFu) | 0x800000u;
    let eA = i32((Abits >> 23u) & 0xFFu) - 127;
    let eRes = eA + i32(q) - 126;
    if (eRes >= 128) { return sgn | 0x7F800000u; }
    if (eRes >= -126) {
        return sgn | (u32(eRes + 127) << 23u) | (Abits & 0x7FFFFFu);
    }
    let s = -126 - eRes;
    if (s >= 25) { return sgn; }       // mA/2^s < 0.5 strictly: rounds to 0
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
    if (kR >= 0x800000u) { return sgn | 0x800000u; }   // up to smallest normal
    return sgn | kR;
}

// expf — fma source selectable
fn expf_body(x: f32, useInt: bool) -> f32 {
    let f5 = clamp(select(fma_soft(x, p.c1, 0.5), fma_int(x, p.c1, 0.5), useInt), 0.0, 1.0);
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
    let f9 = f8f + bitcast<f32>(0xCB40007Fu);
    let f10 = -f9;
    let f12 = select(fma_soft(x, p.log2e, f10), fma_int(x, p.log2e, f10), useInt);
    let f14 = select(fma_soft(x, p.c2, f12), fma_int(x, p.c2, f12), useInt);
    let f15 = bitcast<f32>(f8 << 23u);
    return exp2(f14) * f15;
}

fn expf_soft(x: f32) -> f32 { return expf_body(x, false); }
fn expf_int2(x: f32) -> f32 {
    let f5 = clamp(fma_int(x, p.c1, 0.5), 0.0, 1.0);
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
    let f12 = fma_int(x, p.log2e, f10);
    let f14 = fma_int(x, p.c2, f12);
    let q = f8 - 12582913u;
    return bitcast<f32>(scale_pow2(bitcast<u32>(exp2(f14)), q));
}

@compute @workgroup_size(256)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let x = X[i];
    let y = Y[i];
    let z = Z[i];
    Out[i*24u+0u] = bitcast<u32>(x * y + z);
    Out[i*24u+1u] = bitcast<u32>(exp(x));
    Out[i*24u+2u] = bitcast<u32>(exp2(x));
    Out[i*24u+3u] = bitcast<u32>(expf_soft(x));
    Out[i*24u+4u] = bitcast<u32>(exp2(x * p.log2e));
    Out[i*24u+5u] = bitcast<u32>(fma_soft(x, y, z));
    // expf internals (soft path) + fold detectors
    let f5 = clamp(fma_soft(x, p.c1, 0.5), 0.0, 1.0);
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
    let f9 = f8f + bitcast<f32>(0xCB40007Fu);
    let f10 = -f9;
    let f12 = fma_soft(x, p.log2e, f10);
    let f14 = fma_soft(x, p.c2, f12);
    Out[i*24u+6u] = f8;
    Out[i*24u+7u] = bitcast<u32>(f12);
    Out[i*24u+8u] = bitcast<u32>(f14);
    Out[i*24u+9u] = bitcast<u32>(f5);
    let d_ahi = bitcast<f32>(bitcast<u32>(x) & 0xFFFFF000u);
    Out[i*24u+10u] = bitcast<u32>(x - d_ahi);
    let d_p = bitcast<f32>(bitcast<u32>(x * p.log2e));
    let d_ahi2 = bitcast<f32>(bitcast<u32>(x) & 0xFFFFF000u);
    let d_alo = bitcast<f32>(bitcast<u32>(x - d_ahi2));
    let d_bhi = bitcast<f32>(bitcast<u32>(p.log2e) & 0xFFFFF000u);
    let d_blo = bitcast<f32>(bitcast<u32>(p.log2e - d_bhi));
    let d_t1 = bitcast<f32>(bitcast<u32>(d_ahi2 * d_bhi - d_p));
    let d_t2 = bitcast<f32>(bitcast<u32>(d_t1 + d_ahi2 * d_blo));
    let d_t3 = bitcast<f32>(bitcast<u32>(d_t2 + d_alo * d_bhi));
    Out[i*24u+11u] = bitcast<u32>(d_t3 + d_alo * d_blo);
    // raw mul+add contraction probes
    Out[i*24u+12u] = bitcast<u32>(x * p.c1 + 0.5);
    Out[i*24u+13u] = bitcast<u32>(x * p.log2e + f10);
    Out[i*24u+14u] = bitcast<u32>(x * p.c2 + f12);
    Out[i*24u+15u] = bitcast<u32>(x * p.log2e + 75.0);
    // pure-integer path
    Out[i*24u+16u] = bitcast<u32>(fma_int(x, y, z));
    Out[i*24u+17u] = bitcast<u32>(expf_int2(x));
    let f5i = clamp(fma_int(x, p.c1, 0.5), 0.0, 1.0);
    let b5i = bitcast<u32>(f5i);
    var f8i = 12582913u;
    if ((b5i & 0x7FFFFFFFu) != 0u) {
        let raw_e = i32((b5i >> 23u) & 0xFFu) - 127;
        let m = select(b5i & 0x7FFFFFu, (b5i & 0x7FFFFFu) | 0x800000u, raw_e != -127);
        let ee = select(-126, raw_e, raw_e != -127);
        let pp = (m << 8u) - (m << 2u);
        let shift = u32(23 - ee);
        let q = select(0u, pp >> shift, shift < 32u);
        f8i = 12582913u + q;
    }
    let f8fi = bitcast<f32>(0x4B000000u | (f8i - 0x800000u));
    let f10i = -(f8fi + bitcast<f32>(0xCB40007Fu));
    let f12i = fma_int(x, p.log2e, f10i);
    let f14i = fma_int(x, p.c2, f12i);
    Out[i*24u+18u] = bitcast<u32>(f5i);
    Out[i*24u+19u] = bitcast<u32>(f12i);
    Out[i*24u+20u] = bitcast<u32>(f14i);
}
"#;

fn read_f32(path: &str) -> Result<Vec<f32>> {
    let raw = std::fs::read(path)?;
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

async fn run() -> Result<()> {
    let dir = "wgpu/exp_probe";
    let x = read_f32(&format!("{dir}/input_x.bin"))?;
    let y = read_f32(&format!("{dir}/input_y.bin"))?;
    let z = read_f32(&format!("{dir}/input_z.bin"))?;
    let n = x.len();
    println!("inputs: {n}");

    let gpu = Gpu::new(Some("nvidia")).await?;
    let device = &gpu.device;
    let queue = &gpu.queue;

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("exp_probe"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("exp_probe"),
        layout: None,
        module: &module,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mk = |v: &[f32]| -> wgpu::Buffer {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (v.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(v));
        buf
    };
    let bx = mk(&x);
    let by = mk(&y);
    let bz = mk(&z);
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: (n * 24 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let uniform = {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let c1 = f32::from_bits(0x3BBB989Du32);
        let log2e = f32::from_bits(0x3FB8AA3Bu32);
        let c2 = f32::from_bits(0x32A57060u32);
        queue.write_buffer(
            &buf,
            0,
            bytemuck::bytes_of(&[n as u32, c1.to_bits(), log2e.to_bits(), c2.to_bits()]),
        );
        buf
    };
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: bx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: by.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bz.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: uniform.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&pipe);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(n.div_ceil(256) as u32, 1, 1);
    }
    queue.submit([enc.finish()]);

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (n * 24 * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut renc = device.create_command_encoder(&Default::default());
    renc.copy_buffer_to_buffer(&out, 0, &staging, 0, (n * 24 * 4) as u64);
    queue.submit([renc.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let raw = slice.get_mapped_range()?.to_vec();
    staging.unmap();
    std::fs::write(format!("{dir}/wgpu_out.bin"), &raw)?;
    println!("wgpu_out.bin written ({} u32)", n * 24);
    Ok(())
}

fn main() -> Result<()> {
    pollster::block_on(run())
}
