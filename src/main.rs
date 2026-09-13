//! wgpu feasibility spike for qwen3-asr-rs.
//!
//! Answers, with measurements rather than guesses:
//!   1. Which wgpu features does each adapter actually expose (notably f16)?
//!   2. Can a hand-written WGSL f16 GEMV reach cuBLAS-class bandwidth on the
//!      0.6B decode shapes?
//!   3. What does one dispatch cost in wgpu — the launch tax on a ~255-dispatch step?
//!   4. Does a u32-packed-f16 kernel (no f16 extension needed) hold up?

use std::time::Instant;

// ─────────────────────────────────────────────────────────────────────────────
// Kernels
// ─────────────────────────────────────────────────────────────────────────────

/// Needs Features::SHADER_F16.  Weight is `array<vec4<f16>>` (8 bytes/element).
const SHADER_F16: &str = r#"
enable f16;

struct Dims { n: u32, k: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<storage, read>       W: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       X: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> Y: array<f16>;
@group(0) @binding(3) var<uniform>           dims: Dims;

const LANES: u32 = 32u;
const ROWS_PER_WG: u32 = 8u;
var<workgroup> smem: array<f32, 256>;

@compute @workgroup_size(256)
fn gemv(@builtin(workgroup_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x % LANES;
    let rsub = lid.x / LANES;
    let row  = gid.x * ROWS_PER_WG + rsub;
    let valid = row < dims.n;
    let kv = dims.k / 4u;

    var acc: f32 = 0.0;
    if (valid) {
        var i: u32 = lane;
        loop {
            if (i >= kv) { break; }
            let wv = W[row * kv + i];
            let xv = X[i];
            acc = acc + f32(wv.x) * f32(xv.x) + f32(wv.y) * f32(xv.y)
                      + f32(wv.z) * f32(xv.z) + f32(wv.w) * f32(xv.w);
            i = i + LANES;
        }
    }
    smem[lid.x] = acc;
    workgroupBarrier();
    if (lane < 16u) { smem[lid.x] = smem[lid.x] + smem[lid.x + 16u]; }
    workgroupBarrier();
    if (lane < 8u)  { smem[lid.x] = smem[lid.x] + smem[lid.x + 8u]; }
    workgroupBarrier();
    if (lane < 4u)  { smem[lid.x] = smem[lid.x] + smem[lid.x + 4u]; }
    workgroupBarrier();
    if (lane < 2u)  { smem[lid.x] = smem[lid.x] + smem[lid.x + 2u]; }
    workgroupBarrier();
    if (lane < 1u)  { smem[lid.x] = smem[lid.x] + smem[lid.x + 1u]; }
    workgroupBarrier();
    if (lane == 0u && valid) { Y[row] = f16(smem[lid.x]); }
}
"#;

/// Control: plain f32 weights.  Same kernel shape, but 2x the weight traffic.
const SHADER_F32: &str = r#"
struct Dims { n: u32, k: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<storage, read>       W: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform>           dims: Dims;

const LANES: u32 = 32u;
const ROWS_PER_WG: u32 = 8u;
var<workgroup> smem: array<f32, 256>;

@compute @workgroup_size(256)
fn gemv(@builtin(workgroup_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x % LANES;
    let rsub = lid.x / LANES;
    let row  = gid.x * ROWS_PER_WG + rsub;
    let valid = row < dims.n;
    let kv = dims.k / 4u;

    var acc: f32 = 0.0;
    if (valid) {
        var i: u32 = lane;
        loop {
            if (i >= kv) { break; }
            acc = acc + dot(W[row * kv + i], X[i]);
            i = i + LANES;
        }
    }
    smem[lid.x] = acc;
    workgroupBarrier();
    if (lane < 16u) { smem[lid.x] = smem[lid.x] + smem[lid.x + 16u]; }
    workgroupBarrier();
    if (lane < 8u)  { smem[lid.x] = smem[lid.x] + smem[lid.x + 8u]; }
    workgroupBarrier();
    if (lane < 4u)  { smem[lid.x] = smem[lid.x] + smem[lid.x + 4u]; }
    workgroupBarrier();
    if (lane < 2u)  { smem[lid.x] = smem[lid.x] + smem[lid.x + 2u]; }
    workgroupBarrier();
    if (lane < 1u)  { smem[lid.x] = smem[lid.x] + smem[lid.x + 1u]; }
    workgroupBarrier();
    if (lane == 0u && valid) { Y[row] = smem[lid.x]; }
}
"#;

/// Portable path.  f16 weights packed as `u32` (2 halves each — the byte layout is
/// identical to `array<f16>`), unpacked with the core builtin `unpack2x16float`.
/// No f16 extension and no f16 arithmetic: works on every wgpu backend, including
/// Pascal, where Vulkan exposes 16-bit storage but not shaderFloat16.
const SHADER_P16: &str = r#"
struct Dims { n: u32, k: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<storage, read>       W: array<vec4<u32>>;   // 8 f16 per element
@group(0) @binding(1) var<storage, read>       X: array<vec4<f32>>;   // f32 activations
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform>           dims: Dims;
const LANES: u32 = 32u;
const ROWS_PER_WG: u32 = 8u;
var<workgroup> smem: array<f32, 256>;

@compute @workgroup_size(256)
fn gemv(@builtin(workgroup_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x % LANES;
    let rsub = lid.x / LANES;
    let row  = gid.x * ROWS_PER_WG + rsub;
    let valid = row < dims.n;
    let kv = dims.k / 8u;

    var acc: f32 = 0.0;
    if (valid) {
        var i: u32 = lane;
        loop {
            if (i >= kv) { break; }
            let wv = W[row * kv + i];
            let x0 = X[i * 2u];
            let x1 = X[i * 2u + 1u];
            let a = unpack2x16float(wv.x);
            let b = unpack2x16float(wv.y);
            let c = unpack2x16float(wv.z);
            let d = unpack2x16float(wv.w);
            acc = acc + dot(a, x0.xy) + dot(b, x0.zw) + dot(c, x1.xy) + dot(d, x1.zw);
            i = i + LANES;
        }
    }
    smem[lid.x] = acc;
    workgroupBarrier();
    if (lane < 16u) { smem[lid.x] = smem[lid.x] + smem[lid.x + 16u]; }
    workgroupBarrier();
    if (lane < 8u)  { smem[lid.x] = smem[lid.x] + smem[lid.x + 8u]; }
    workgroupBarrier();
    if (lane < 4u)  { smem[lid.x] = smem[lid.x] + smem[lid.x + 4u]; }
    workgroupBarrier();
    if (lane < 2u)  { smem[lid.x] = smem[lid.x] + smem[lid.x + 2u]; }
    workgroupBarrier();
    if (lane < 1u)  { smem[lid.x] = smem[lid.x] + smem[lid.x + 1u]; }
    workgroupBarrier();
    if (lane == 0u && valid) { Y[row] = smem[lid.x]; }
}
"#;

/// Kept in its own module: WGSL resource variables must not share (group, binding)
/// with an unrelated variable reachable from another entry point.
const SHADER_PROBE: &str = r#"
// Pure streaming read of the same bytes, for the bandwidth ceiling.
@group(0) @binding(0) var<storage, read>       R: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> O: array<f32>;

const CHUNK: u32 = 8192u;
var<workgroup> rsmem: array<f32, 256>;

@compute @workgroup_size(256)
fn readprobe(@builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(workgroup_id) wid: vec3<u32>) {
    let base = wid.x * CHUNK;
    var acc: f32 = 0.0;
    var i: u32 = lid.x;
    loop {
        if (i >= CHUNK) { break; }
        let v = R[base + i];
        let p0 = unpack2x16float(v.x);
        let p1 = unpack2x16float(v.y);
        let p2 = unpack2x16float(v.z);
        let p3 = unpack2x16float(v.w);
        acc = acc + p0.x + p0.y + p1.x + p1.y + p2.x + p2.y + p3.x + p3.y;
        i = i + 256u;
    }
    rsmem[lid.x] = acc;
    workgroupBarrier();
    if (lid.x < 16u) { rsmem[lid.x] = rsmem[lid.x] + rsmem[lid.x + 16u]; }
    workgroupBarrier();
    if (lid.x < 8u)  { rsmem[lid.x] = rsmem[lid.x] + rsmem[lid.x + 8u]; }
    workgroupBarrier();
    if (lid.x < 4u)  { rsmem[lid.x] = rsmem[lid.x] + rsmem[lid.x + 4u]; }
    workgroupBarrier();
    if (lid.x < 2u)  { rsmem[lid.x] = rsmem[lid.x] + rsmem[lid.x + 2u]; }
    workgroupBarrier();
    if (lid.x < 1u)  { rsmem[lid.x] = rsmem[lid.x] + rsmem[lid.x + 1u]; }
    workgroupBarrier();
    if (lid.x == 0u) { O[wid.x] = rsmem[0u]; }
}

@compute @workgroup_size(1)
fn noop() {}
"#;

/// Prefill GEMM: C[m,n] = A[m,k] * W[n,k]^T, f32 accumulate, packed-f16 weights.
/// 64x64 output tile per workgroup, 256 threads, 4x4 micro-tile per thread.
/// Deliberately straightforward — the point is to bound how far a hand-written WGSL
/// GEMM falls behind cuBLAS on the prefill shapes, not to ship a tuned kernel.
/// Generates a prefill GEMM with a configurable `tm x tn` micro-tile per thread.
/// Workgroup is 16x16 = 256 threads; the output tile is `16*tm x 16*tn`.
/// Scalar shared-memory loads, f32 accumulate, packed-f16 weights on the right operand.
/// This exists so the micro-tile can be swept without hand-writing each variant.
fn gemm_shader(tm: usize, tn: usize, unroll_q: bool) -> String {
    let bm = 16 * tm;
    let bn = 16 * tn;
    let bk = 16usize;
    let pad = bk + 1;
    let n_as = bm * bk / 256;
    let n_bs = bn * bk / 256;
    assert_eq!(bm * bk % 256, 0);
    assert_eq!(bn * bk % 256, 0);

    let mut s = String::new();
    s.push_str(
        "struct GDims { m: u32, n: u32, k: u32, _p: u32 };\n\
         @group(0) @binding(0) var<storage, read>       A: array<f32>;\n\
         @group(0) @binding(1) var<storage, read>       W: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> C: array<f32>;\n\
         @group(0) @binding(3) var<uniform>             gd: GDims;\n",
    );
    s.push_str(&format!(
        "const BM: u32 = {bm}u;\nconst BN: u32 = {bn}u;\nconst BK: u32 = {bk}u;\nconst PAD: u32 = {pad}u;\n"
    ));
    s.push_str(&format!("var<workgroup> As: array<f32, {}>;\n", bm * pad));
    s.push_str(&format!("var<workgroup> Bs: array<f32, {}>;\n", bn * pad));
    s.push_str(
        "@compute @workgroup_size(16, 16)\n\
         fn gemm(@builtin(workgroup_id) wid: vec3<u32>,\n\
                 @builtin(local_invocation_id) lid: vec3<u32>) {\n\
         let tx = lid.x;\n let ty = lid.y;\n\
         let m0 = wid.y * BM;\n let n0 = wid.x * BN;\n let kk = gd.k / 2u;\n",
    );

    for i in 0..tm {
        for j in 0..tn {
            s.push_str(&format!("var c{i}{j} = 0.0;\n"));
        }
    }

    s.push_str("var k0: u32 = 0u;\nloop {\n if (k0 >= gd.k) { break; }\n");

    s.push_str(&format!(
        " var t: u32 = 0u;\n loop {{\n  if (t >= {n_as}u) {{ break; }}\n  \
         let idx = lid.y * 16u + lid.x + t * 256u;\n  let r = idx / BK;\n  let c = idx % BK;\n  \
         As[r * PAD + c] = A[(m0 + r) * gd.k + k0 + c];\n  t = t + 1u;\n }}\n"
    ));
    s.push_str(&format!(
        " var u: u32 = 0u;\n loop {{\n  if (u >= {n_bs}u) {{ break; }}\n  \
         let idx = lid.y * 16u + lid.x + u * 256u;\n  let r = idx / BK;\n  let c = idx % BK;\n  \
         let pr = unpack2x16float(W[(n0 + r) * kk + (k0 + c) / 2u]);\n  \
         Bs[r * PAD + c] = select(pr.x, pr.y, (c & 1u) == 1u);\n  u = u + 1u;\n }}\n"
    ));
    s.push_str(" workgroupBarrier();\n");

    if unroll_q {
        for q in 0..bk {
            for i in 0..tm {
                s.push_str(&format!(" let a{i}_{q} = As[(ty * {tm}u + {i}u) * PAD + {q}u];\n"));
            }
            for j in 0..tn {
                s.push_str(&format!(" let b{j}_{q} = Bs[(tx * {tn}u + {j}u) * PAD + {q}u];\n"));
            }
            for i in 0..tm {
                for j in 0..tn {
                    s.push_str(&format!(" c{i}{j} = c{i}{j} + a{i}_{q} * b{j}_{q};\n"));
                }
            }
        }
    } else {
        s.push_str(" var q: u32 = 0u;\n loop {\n  if (q >= BK) { break; }\n");
        for i in 0..tm {
            s.push_str(&format!("  let a{i} = As[(ty * {tm}u + {i}u) * PAD + q];\n"));
        }
        for j in 0..tn {
            s.push_str(&format!("  let b{j} = Bs[(tx * {tn}u + {j}u) * PAD + q];\n"));
        }
        for i in 0..tm {
            for j in 0..tn {
                s.push_str(&format!("  c{i}{j} = c{i}{j} + a{i} * b{j};\n"));
            }
        }
        s.push_str("  q = q + 1u;\n }\n");
    }
    s.push_str(" workgroupBarrier();\n k0 = k0 + BK;\n}\n");

    for i in 0..tm {
        s.push_str(&format!(
            "let r{i} = (m0 + ty * {tm}u + {i}u) * gd.n + n0 + tx * {tn}u;\n"
        ));
    }
    for i in 0..tm {
        for j in 0..tn {
            s.push_str(&format!("C[r{i} + {j}u] = c{i}{j};\n"));
        }
    }
    s.push_str("}\n");
    s
}

#[allow(dead_code)]
const SHADER_GEMM: &str = r#"
struct GDims { m: u32, n: u32, k: u32, _p: u32 };

@group(0) @binding(0) var<storage, read>       A: array<f32>;   // [m, k]
@group(0) @binding(1) var<storage, read>       W: array<u32>;   // [n, k/2] packed f16
@group(0) @binding(2) var<storage, read_write> C: array<f32>;   // [m, n]
@group(0) @binding(3) var<uniform>             gd: GDims;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
const PAD: u32 = 17u;

var<workgroup> As: array<f32, 1088>;
var<workgroup> Bs: array<f32, 1088>;

@compute @workgroup_size(16, 16)
fn gemm(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let m0 = wid.y * BM;
    let n0 = wid.x * BN;
    let kk = gd.k / 2u;

    var c00 = 0.0; var c01 = 0.0; var c02 = 0.0; var c03 = 0.0;
    var c10 = 0.0; var c11 = 0.0; var c12 = 0.0; var c13 = 0.0;
    var c20 = 0.0; var c21 = 0.0; var c22 = 0.0; var c23 = 0.0;
    var c30 = 0.0; var c31 = 0.0; var c32 = 0.0; var c33 = 0.0;

    var k0: u32 = 0u;
    loop {
        if (k0 >= gd.k) { break; }

        var t: u32 = 0u;
        loop {
            if (t >= 4u) { break; }
            let idx = lid.y * 16u + lid.x + t * 256u;
            let r = idx / BK;
            let c = idx % BK;
            As[r * PAD + c] = A[(m0 + r) * gd.k + k0 + c];
            let pr = unpack2x16float(W[(n0 + r) * kk + (k0 + c) / 2u]);
            Bs[r * PAD + c] = select(pr.x, pr.y, (c & 1u) == 1u);
            t = t + 1u;
        }
        workgroupBarrier();

        var q: u32 = 0u;
        loop {
            if (q >= BK) { break; }
            let a0 = As[(ty * 4u + 0u) * PAD + q];
            let a1 = As[(ty * 4u + 1u) * PAD + q];
            let a2 = As[(ty * 4u + 2u) * PAD + q];
            let a3 = As[(ty * 4u + 3u) * PAD + q];
            let b0 = Bs[(tx * 4u + 0u) * PAD + q];
            let b1 = Bs[(tx * 4u + 1u) * PAD + q];
            let b2 = Bs[(tx * 4u + 2u) * PAD + q];
            let b3 = Bs[(tx * 4u + 3u) * PAD + q];
            c00 = c00 + a0 * b0; c01 = c01 + a0 * b1; c02 = c02 + a0 * b2; c03 = c03 + a0 * b3;
            c10 = c10 + a1 * b0; c11 = c11 + a1 * b1; c12 = c12 + a1 * b2; c13 = c13 + a1 * b3;
            c20 = c20 + a2 * b0; c21 = c21 + a2 * b1; c22 = c22 + a2 * b2; c23 = c23 + a2 * b3;
            c30 = c30 + a3 * b0; c31 = c31 + a3 * b1; c32 = c32 + a3 * b2; c33 = c33 + a3 * b3;
            q = q + 1u;
        }
        workgroupBarrier();
        k0 = k0 + BK;
    }

    let r0 = (m0 + ty * 4u + 0u) * gd.n + n0 + tx * 4u;
    let r1 = (m0 + ty * 4u + 1u) * gd.n + n0 + tx * 4u;
    let r2 = (m0 + ty * 4u + 2u) * gd.n + n0 + tx * 4u;
    let r3 = (m0 + ty * 4u + 3u) * gd.n + n0 + tx * 4u;
    C[r0 + 0u] = c00; C[r0 + 1u] = c01; C[r0 + 2u] = c02; C[r0 + 3u] = c03;
    C[r1 + 0u] = c10; C[r1 + 1u] = c11; C[r1 + 2u] = c12; C[r1 + 3u] = c13;
    C[r2 + 0u] = c20; C[r2 + 1u] = c21; C[r2 + 2u] = c22; C[r2 + 3u] = c23;
    C[r3 + 0u] = c30; C[r3 + 1u] = c31; C[r3 + 2u] = c32; C[r3 + 3u] = c33;
}
"#;

// ─────────────────────────────────────────────────────────────────────────────
// Qwen3-ASR-0.6B text decoder shapes
// (mirrors examples/cublas_gemv_bench.rs so numbers are directly comparable)
// ─────────────────────────────────────────────────────────────────────────────
const HS: usize = 1024;
const Q_DIM: usize = 16 * 128;                      // 2048
const KV_DIM: usize = 8 * 128;                      // 1024
const FUSED_QKV_COLS: usize = Q_DIM + 2 * KV_DIM;   // 4096
const INTER: usize = 3072;
const FUSED_GU_COLS: usize = 2 * INTER;             // 6144
const VOCAB: usize = 151936;
const LAYERS: usize = 28;

#[derive(Clone, Copy)]
struct Shaped { name: &'static str, rows: usize, cols: usize }

const SHAPES: [Shaped; 5] = [
    Shaped { name: "qkv",       rows: FUSED_QKV_COLS, cols: HS },
    Shaped { name: "o_proj",    rows: HS,             cols: Q_DIM },
    Shaped { name: "gate_up",   rows: FUSED_GU_COLS,  cols: HS },
    Shaped { name: "down_proj", rows: HS,             cols: INTER },
    Shaped { name: "lm_head",   rows: VOCAB,          cols: HS },
];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct D { n: u32, k: u32, _p0: u32, _p1: u32 }

fn dims_buf(device: &wgpu::Device, queue: &wgpu::Queue, n: u32, k: u32) -> wgpu::Buffer {
    let b = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&b, 0, bytemuck::bytes_of(&D { n, k, _p0: 0, _p1: 0 }));
    b
}

/// f16 bit patterns inside [0.5, 1.0): no denormals, no inf/nan.
fn f16_bytes(seed: u32, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 2);
    let mut s = seed | 1;
    for _ in 0..len {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let bits: u16 = 0x3800 | ((s >> 16) as u16 & 0x03FF);
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn f32_bytes(seed: u32, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 4);
    let mut s = seed | 1;
    for _ in 0..len {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let v = 0.5f32 + ((s >> 8) as f32 / 16777216.0) * 0.5;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let want = std::env::args().nth(1).map(|s| s.to_lowercase());

    let instance = wgpu::Instance::default();

    println!("=== wgpu adapter inventory ===");
    let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
    for (idx, a) in adapters.iter().enumerate() {
        let i = a.get_info();
        println!(
            "  [{}] {:?} | {} | {:?} | {} {} | SHADER_F16={}",
            idx, i.backend, i.name, i.device_type, i.driver, i.driver_info,
            a.features().contains(wgpu::Features::SHADER_F16)
        );
    }

    let adapter = adapters
        .into_iter()
        .find(|a| match &want {
            Some(w) => a.get_info().name.to_lowercase().contains(w.as_str()),
            None => a.get_info().device_type == wgpu::DeviceType::DiscreteGpu,
        })
        .expect("no matching adapter");

    let info = adapter.get_info();
    println!(
        "\n=== selected: {} ({:?}, {:?}) | {} {} ===",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    );

    let has_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
    println!("SHADER_F16: {}", has_f16);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("spike"),
            required_features: if has_f16 { wgpu::Features::SHADER_F16 } else { wgpu::Features::empty() },
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .expect("request_device");

    println!(
        "limits: maxStorageBufferBindingSize={} MiB, maxComputeInvocationsPerWorkgroup={}, \
         maxComputeWorkgroupsPerDimension={}",
        device.limits().max_storage_buffer_binding_size / (1024 * 1024),
        device.limits().max_compute_invocations_per_workgroup,
        device.limits().max_compute_workgroups_per_dimension,
    );

    // ── Kernels ───────────────────────────────────────────────────────────────
    let build = |src: &str| device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let mk = |m: &wgpu::ShaderModule, e: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(e),
        layout: None,
        module: m,
        entry_point: Some(e),
        compilation_options: Default::default(),
        cache: None,
    });

    let m_p16 = build(SHADER_P16);
    let m_probe = build(SHADER_PROBE);
    let m_f32 = build(SHADER_F32);
    let m_f16 = if has_f16 { Some(build(SHADER_F16)) } else { None };

    let p16 = mk(&m_p16, "gemv");
    let probe = mk(&m_probe, "readprobe");
    let noop = mk(&m_probe, "noop");
    let f32p = mk(&m_f32, "gemv");
    let f16p = m_f16.as_ref().map(|m| mk(m, "gemv"));

    // ── Weight buffers (shared; identical bytes for every packing) ────────────
    let mut w_bufs: Vec<wgpu::Buffer> = Vec::new();
    let mut total_f16_bytes = 0usize;
    for sh in &SHAPES {
        let bytes = sh.rows * sh.cols * 2;
        let b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(sh.name),
            size: bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&b, 0, &f16_bytes(1, sh.rows * sh.cols));
        total_f16_bytes += bytes;
        w_bufs.push(b);
    }

    println!("\n=== WGSL f16 GEMV (m=1) bandwidth, Qwen3-ASR-0.6B decode shapes ===");
    println!("{:<11} {:>7} {:>7} {:>11} {:>10} {:>10}", "gemv", "rows", "cols", "weight(MB)", "ms", "GB/s");
    println!("{}", "-".repeat(62));

    let mut totals: Vec<(String, f64, usize)> = Vec::new();

    if let Some(p) = &f16p {
        println!("* array<vec4<f16>> + f32 accumulate   [needs SHADER_F16]");
        let (sum, bytes, rows) = bench_p16_style(&device, &queue, p, &w_bufs, &SHAPES, true);
        for (sh, ms) in SHAPES.iter().zip(rows.iter()) {
            println!("{:<11} {:>7} {:>7} {:>11.1} {:>10.3} {:>10.1}",
                sh.name, sh.rows, sh.cols, (sh.rows * sh.cols * 2) as f64 / 1048576.0,
                ms, (sh.rows * sh.cols * 2) as f64 / 1e9 / (ms / 1000.0));
        }
        println!("  -> decode-step total {:.3} ms | {:.1} GB/s\n", sum, bytes as f64 / 1e9 / (sum / 1000.0));
        totals.push(("f16-ext".into(), sum, bytes));
    }

    {
        println!("* array<vec4<u32>> + unpack2x16float  [portable, no f16 ext]");
        let (sum, bytes, rows) = bench_p16_style(&device, &queue, &p16, &w_bufs, &SHAPES, false);
        for (sh, ms) in SHAPES.iter().zip(rows.iter()) {
            println!("{:<11} {:>7} {:>7} {:>11.1} {:>10.3} {:>10.1}",
                sh.name, sh.rows, sh.cols, (sh.rows * sh.cols * 2) as f64 / 1048576.0,
                ms, (sh.rows * sh.cols * 2) as f64 / 1e9 / (ms / 1000.0));
        }
        println!("  -> decode-step total {:.3} ms | {:.1} GB/s\n", sum, bytes as f64 / 1e9 / (sum / 1000.0));
        totals.push(("packed-f16".into(), sum, bytes));
    }

    {
        println!("* array<vec4<f32>>                    [control: 2x weight traffic]");
        let mut w32: Vec<wgpu::Buffer> = Vec::new();
        for sh in &SHAPES {
            let b = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (sh.rows * sh.cols * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&b, 0, &f32_bytes(1, sh.rows * sh.cols));
            w32.push(b);
        }
        let mut sum = 0.0;
        let mut bytes = 0usize;
        for (sh, w) in SHAPES.iter().zip(w32.iter()) {
            let x = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: (sh.cols * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let y = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: (sh.rows * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let dims = dims_buf(&device, &queue, sh.rows as u32, sh.cols as u32);
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &f32p.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: w.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
                ],
            });
            let gx = (sh.rows as u32).div_ceil(8);
            for _ in 0..3 { submit_n(&device, &queue, &f32p, &bg, gx, 1); }
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let iters = 30u32;
            let t0 = Instant::now();
            submit_n(&device, &queue, &f32p, &bg, gx, iters);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            println!("{:<11} {:>7} {:>7} {:>11.1} {:>10.3} {:>10.1}",
                sh.name, sh.rows, sh.cols, (sh.rows * sh.cols * 4) as f64 / 1048576.0, ms,
                (sh.rows * sh.cols * 4) as f64 / 1e9 / (ms / 1000.0));
            sum += ms;
            bytes += sh.rows * sh.cols * 4;
        }
        println!("  -> decode-step total {:.3} ms | {:.1} GB/s\n", sum, bytes as f64 / 1e9 / (sum / 1000.0));
        totals.push(("f32".into(), sum, bytes));
    }

    correctness_probe(&device, &queue, &p16);
    verify_full_k(&device, &queue, &p16, &w_bufs[0], SHAPES[0].rows, SHAPES[0].cols);

    // ── Pure streaming-read ceiling ───────────────────────────────────────────
    {
        let slots = (total_f16_bytes / 16).next_power_of_two();
        let slots = slots - (slots % 8192);
        let probe_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe"),
            size: (slots * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&probe_buf, 0, &f16_bytes(5, slots * 8));
        let nwg = slots / 8192;
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (nwg * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &probe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: probe_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            ],
        });
        let run = || {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&probe);
                cp.set_bind_group(0, &bg, &[]);
                cp.dispatch_workgroups(nwg as u32, 1, 1);
            }
            queue.submit([enc.finish()]);
        };
        run();
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let t0 = Instant::now();
        run();
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("=== pure streaming read: {:.0} MB in {:.3} ms -> {:.1} GB/s ===",
            (slots * 16) as f64 / 1048576.0, ms, (slots * 16) as f64 / 1e9 / (ms / 1000.0));
    }

    // ── Dispatch overhead ─────────────────────────────────────────────────────
    println!("\n=== dispatch overhead ===");
    let bg_noop = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &noop.get_bind_group_layout(0), entries: &[],
    });
    let n = 4096u32;

    let t0 = Instant::now();
    let enc = {
        let mut e = device.create_command_encoder(&Default::default());
        {
            let mut cp = e.begin_compute_pass(&Default::default());
            cp.set_pipeline(&noop);
            cp.set_bind_group(0, &bg_noop, &[]);
            for _ in 0..n { cp.dispatch_workgroups(1, 1, 1); }
        }
        e
    };
    println!("  CPU encode only, batched      : {:>7.2} us / dispatch",
        t0.elapsed().as_secs_f64() * 1e6 / n as f64);
    drop(enc);

    let t0 = Instant::now();
    let mut e = device.create_command_encoder(&Default::default());
    {
        let mut cp = e.begin_compute_pass(&Default::default());
        cp.set_pipeline(&noop);
        cp.set_bind_group(0, &bg_noop, &[]);
        for _ in 0..n { cp.dispatch_workgroups(1, 1, 1); }
    }
    queue.submit([e.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    println!("  encode + 1 submit, batched    : {:>7.2} us / dispatch",
        t0.elapsed().as_secs_f64() * 1e6 / n as f64);

    let m = 512u32;
    let t0 = Instant::now();
    for _ in 0..m {
        let mut e = device.create_command_encoder(&Default::default());
        {
            let mut cp = e.begin_compute_pass(&Default::default());
            cp.set_pipeline(&noop);
            cp.set_bind_group(0, &bg_noop, &[]);
            cp.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([e.finish()]);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    println!("  encode + submit each          : {:>7.2} us / dispatch",
        t0.elapsed().as_secs_f64() * 1e6 / m as f64);

    // ── Simulated decode step ─────────────────────────────────────────────────
    println!("\n=== simulated decode step: 28 layers x 4 GEMVs + lm_head ===");
    let idx = [0usize, 1, 2, 3];
    let gx: Vec<u32> = idx.iter().map(|&i| (SHAPES[i].rows as u32).div_ceil(8)).collect();
    let bgs: Vec<wgpu::BindGroup> = idx.iter().map(|&i| {
        let x = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (SHAPES[i].cols * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let y = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (SHAPES[i].rows * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dims = dims_buf(&device, &queue, SHAPES[i].rows as u32, SHAPES[i].cols as u32);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &p16.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_bufs[i].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
            ],
        })
    }).collect();

    let decode = |extra_noops: u32| -> f64 {
        for _ in 0..2 {
            let mut e = device.create_command_encoder(&Default::default());
            {
                let mut cp = e.begin_compute_pass(&Default::default());
                cp.set_pipeline(&p16);
                for _ in 0..LAYERS {
                    for (k, bg) in bgs.iter().enumerate() {
                        cp.set_bind_group(0, bg, &[]);
                        cp.dispatch_workgroups(gx[k], 1, 1);
                    }
                }
            }
            queue.submit([e.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        }
        let t0 = Instant::now();
        let mut e = device.create_command_encoder(&Default::default());
        {
            let mut cp = e.begin_compute_pass(&Default::default());
            cp.set_pipeline(&p16);
            for _ in 0..LAYERS {
                for (k, bg) in bgs.iter().enumerate() {
                    cp.set_bind_group(0, bg, &[]);
                    cp.dispatch_workgroups(gx[k], 1, 1);
                }
            }
            if extra_noops > 0 {
                cp.set_pipeline(&noop);
                cp.set_bind_group(0, &bg_noop, &[]);
                for _ in 0..extra_noops { cp.dispatch_workgroups(1, 1, 1); }
            }
        }
        queue.submit([e.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        t0.elapsed().as_secs_f64() * 1000.0
    };

    let layer_bytes = LAYERS * (FUSED_QKV_COLS * HS + HS * Q_DIM + FUSED_GU_COLS * HS + HS * INTER) * 2
        + VOCAB * HS * 2;
    let a = decode(0);
    println!("  113 GEMV dispatches             : {:>7.3} ms/step -> {:.0} tok/s, {:.1} GB/s",
        a, 1000.0 / a, layer_bytes as f64 / 1e9 / (a / 1000.0));
    let b = decode(142);
    println!("  +142 no-ops (255 dispatches)    : {:>7.3} ms/step -> {:.0} tok/s (tax {:+.3} ms)",
        b, 1000.0 / b, b - a);

    // ── Prefill GEMM ─────────────────────────────────────────────────────────
    let layer_specs = [("qkv", 0usize), ("o_proj", 1), ("gate_up", 2), ("down", 3)];
    for &(tm, tn, un) in &[(4usize, 4usize, false), (4, 4, true), (8, 4, true)] {
        let src = gemm_shader(tm, tn, un);
        let mg = build(&src);
        let gemm = mk(&mg, "gemm");
        let (bm, bn) = (16 * tm, 16 * tn);

        let ok = verify_gemm(&device, &queue, &gemm, &w_bufs[0], bm, bn, 64);

        println!("\n=== prefill GEMM, micro-tile {tm}x{tn}, k-unroll {un}, output tile {bm}x{bn}  (correct: {ok}) ===");
        println!("{:<10} {:>6} {:>7} {:>6} {:>10} {:>11} {:>9}",
            "shape", "m", "n", "k", "ms", "GFLOP/s", "GB/s(W)");
        println!("{}", "-".repeat(63));

        for &m in &[384usize, 1536usize] {
            let mut layer_total = 0.0;
            for (name, si) in layer_specs.iter() {
                let sh = SHAPES[*si];
                let (n, k) = (sh.rows, sh.cols);
                let ms = run_gemm(&device, &queue, &gemm, &w_bufs[*si], m, n, k, 5, 30);
                let flops = 2.0 * m as f64 * n as f64 * k as f64;
                println!("{:<10} {:>6} {:>7} {:>6} {:>10.3} {:>11.1} {:>9.1}",
                    name, m, n, k, ms, flops / (ms / 1000.0) / 1e9,
                    (n * k * 2) as f64 / 1e9 / (ms / 1000.0));
                layer_total += ms;
            }
            println!("  -> 28 layers at m={}: {:.1} ms   (cuBLAS: {:.1} ms)",
                m, layer_total * LAYERS as f64, if m == 384 { 75.3 } else { 191.1 });
        }
    }

    println!("\n=== summary: one decode step of weight GEMVs ===");
    for (k, ms, by) in &totals {
        println!("  {:<12} {:>7.3} ms | {:>6.1} MB | {:>6.1} GB/s", k, ms, *by as f64 / 1048576.0,
            *by as f64 / 1e9 / (ms / 1000.0));
    }
}

/// GEMV over the shared weight buffers.  `as_f16` picks the vec4<f16> activation buffer
/// (f16-extension pipeline) over the vec4<f32> one (packed-u32 pipeline).
fn bench_p16_style(
    device: &wgpu::Device, queue: &wgpu::Queue, pipe: &wgpu::ComputePipeline,
    w: &[wgpu::Buffer], shapes: &[Shaped], as_f16: bool,
) -> (f64, usize, Vec<f64>) {
    let mut rows = Vec::new();
    let mut sum = 0.0;
    let mut bytes = 0usize;
    for (sh, wb) in shapes.iter().zip(w.iter()) {
        let xsize = if as_f16 { sh.cols * 2 } else { sh.cols * 4 };
        let ysize = if as_f16 { sh.rows * 2 } else { sh.rows * 4 };
        let x = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: xsize as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let y = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: ysize as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dims = dims_buf(device, queue, sh.rows as u32, sh.cols as u32);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
            ],
        });
        let gx = (sh.rows as u32).div_ceil(8);
        for _ in 0..3 { submit_n(device, queue, pipe, &bg, gx, 1); }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let iters = 40u32;
        let t0 = Instant::now();
        submit_n(device, queue, pipe, &bg, gx, iters);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        rows.push(ms);
        sum += ms;
        bytes += sh.rows * sh.cols * 2;
    }
    (sum, bytes, rows)
}

fn submit_n(
    device: &wgpu::Device, queue: &wgpu::Queue, pipe: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup, gx: u32, reps: u32,
) {
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(pipe);
        cp.set_bind_group(0, bg, &[]);
        for _ in 0..reps { cp.dispatch_workgroups(gx, 1, 1); }
    }
    queue.submit([enc.finish()]);
}

/// One small packed-u32 GEMV vs a CPU f32 reference: proves `unpack2x16float` and the
/// f16 storage bytes round-trip correctly on this backend — the exact area where the
/// earlier Intel-Vulkan attempt hit device-lost.
fn correctness_probe(device: &wgpu::Device, queue: &wgpu::Queue, pipe: &wgpu::ComputePipeline) {
    const N: usize = 64;
    const K: usize = 128;

    let w_host = f16_bytes(3, N * K);
    let x_host = f32_bytes(11, K);

    let st = |size: u64, src: bool| device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size,
        usage: if src {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
        } else {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
        },
        mapped_at_creation: false,
    });

    let w = st((N * K * 2) as u64, false);
    let x = st((K * 4) as u64, false);
    let y = st((N * 4) as u64, true);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (N * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let dims = dims_buf(device, queue, N as u32, K as u32);
    queue.write_buffer(&w, 0, &w_host);
    queue.write_buffer(&x, 0, &x_host);

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(pipe);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups((N as u32).div_ceil(8), 1, 1);
    }
    enc.copy_buffer_to_buffer(&y, 0, &staging, 0, (N * 4) as u64);
    queue.submit([enc.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();

    let mut max_err = 0.0f32;
    let mut finite = true;
    {
        let data = slice.get_mapped_range().unwrap();
        let got: Vec<f32> = data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let wf: Vec<f32> = w_host.chunks_exact(2)
            .map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
        let xf: Vec<f32> = x_host.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        for i in 0..N {
            let mut acc = 0.0f32;
            for j in 0..K { acc += wf[i * K + j] * xf[j]; }
            if !got[i].is_finite() { finite = false; }
            max_err = max_err.max((acc - got[i]).abs());
        }
    }
    drop(slice);
    staging.unmap();

    println!("=== numerical sanity: 64x128 packed-f16 GEMV vs CPU f32 reference ===");
    println!("  all outputs finite : {}", finite);
    println!("  max abs error      : {:.5}  (dot products are O(40): rounding-level)", max_err);
}

/// Full-K verification on a real decode shape: runs the packed kernel over the 4096x1024
/// qkv weight and diffs the first and last output rows against a CPU f32 reference.
/// This is the check that catches a kernel which silently reads less than K.
fn verify_full_k(
    device: &wgpu::Device, queue: &wgpu::Queue, pipe: &wgpu::ComputePipeline,
    w: &wgpu::Buffer, rows: usize, cols: usize,
) {
    let x_host = f32_bytes(9, cols);
    let x = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (cols * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let y = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (rows * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (rows * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&x, 0, &x_host);
    let dims = dims_buf(device, queue, rows as u32, cols as u32);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(pipe);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups((rows as u32).div_ceil(8), 1, 1);
    }
    enc.copy_buffer_to_buffer(&y, 0, &staging, 0, (rows * 4) as u64);
    queue.submit([enc.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();

    let xf: Vec<f32> = x_host.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let whost = f16_bytes(1, rows * cols);
    let wf: Vec<f32> = whost.chunks_exact(2)
        .map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();

    let mut report = Vec::new();
    for &r in &[0usize, 1, rows / 2, rows - 1] {
        let got = {
            let d = slice.get_mapped_range().unwrap();
            let off = r * 4;
            f32::from_le_bytes([d[off], d[off + 1], d[off + 2], d[off + 3]])
        };
        let mut acc = 0.0f32;
        for j in 0..cols { acc += wf[r * cols + j] * xf[j]; }
        report.push((r, acc, got, (acc - got).abs()));
    }
    drop(slice);
    staging.unmap();

    println!("=== full-K check on qkv {rows}x{cols} (packed kernel) ===");
    for (r, cpu, gpu, e) in &report {
        println!("  row {:>6}: cpu {:.4} | gpu {:.4} | err {:.5}", r, cpu, gpu, e);
    }
    let worst = report.iter().map(|x| x.3).fold(0.0f32, f32::max);
    println!("  -> {}", if worst < 0.05 { "kernel reads all K correctly" } else { "MISMATCH: kernel is not doing the full reduction" });
}

/// Time one prefill-shaped GEMM.  `w` must hold at least n*(k/2) u32 (packed f16 rows).
fn run_gemm(
    device: &wgpu::Device, queue: &wgpu::Queue, pipe: &wgpu::ComputePipeline,
    w: &wgpu::Buffer, m: usize, n: usize, k: usize, warm: u32, iters: u32,
) -> f64 {
    let a = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (m * k * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&a, 0, &f32_bytes(3, m * k));
    let c = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (m * n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let dims = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&dims, 0, bytemuck::bytes_of(&[m as u32, n as u32, k as u32, 0u32]));

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: c.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
        ],
    });
    let (gx, gy) = ((n / 64) as u32, (m / 64) as u32);

    let run = |reps: u32| {
        let mut e = device.create_command_encoder(&Default::default());
        {
            let mut cp = e.begin_compute_pass(&Default::default());
            cp.set_pipeline(pipe);
            cp.set_bind_group(0, &bg, &[]);
            for _ in 0..reps { cp.dispatch_workgroups(gx, gy, 1); }
        }
        queue.submit([e.finish()]);
    };
    run(warm);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let t0 = Instant::now();
    run(iters);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

/// Small-GEMM correctness: GPU vs CPU f32 reference on a 64x64x64 case, reusing the
/// packed weight buffer reinterpreted with a k=64 row stride.
fn verify_gemm(
    device: &wgpu::Device, queue: &wgpu::Queue, pipe: &wgpu::ComputePipeline,
    w: &wgpu::Buffer, m: usize, n: usize, k: usize,
) -> bool {
    let a_host = f32_bytes(3, m * k);
    let a = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (m * k * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&a, 0, &a_host);
    let c = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (m * n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (m * n * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let dims = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&dims, 0, bytemuck::bytes_of(&[m as u32, n as u32, k as u32, 0u32]));

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: c.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
        ],
    });
    let mut e = device.create_command_encoder(&Default::default());
    {
        let mut cp = e.begin_compute_pass(&Default::default());
        cp.set_pipeline(pipe);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups((n / 64) as u32, (m / 64) as u32, 1);
    }
    e.copy_buffer_to_buffer(&c, 0, &staging, 0, (m * n * 4) as u64);
    queue.submit([e.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();

    // CPU reference over the same raw bytes.
    let wraw = f16_bytes(1, 4096 * 1024);
    let words: Vec<u32> = wraw.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let af: Vec<f32> = a_host.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let mut worst = 0.0f32;
    {
        let data = slice.get_mapped_range().unwrap();
        for &(i, j) in &[(0usize, 0usize), (1, 3), (17, 29), (63, 63)] {
            let got = {
                let off = (i * n + j) * 4;
                f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
            };
            let mut acc = 0.0f32;
            for q in 0..k {
                let word = words[j * (k / 2) + q / 2];
                let wv = if q % 2 == 0 { word & 0xFFFF } else { word >> 16 };
                acc += half_to_f32(wv as u16) * af[i * k + q];
            }
            worst = worst.max((acc - got).abs());
        }
    }
    drop(slice);
    staging.unmap();

    println!("\n=== GEMM numerical check 64x64x64: max abs err {:.5} ===", worst);
    worst < 0.05
}

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if frac == 0 { sign << 31 } else {
            let mut e: i32 = 127 - 15 + 1;
            let mut m = frac;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            (sign << 31) | ((e as u32) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 31 {
        (sign << 31) | 0x7F80_0000 | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}
