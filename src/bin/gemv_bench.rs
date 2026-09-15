use std::time::Instant;

use anyhow::Result;

use qwen3_asr_wgpu::gpu::Gpu;
use qwen3_asr_wgpu::shaders;

const UNROLL4_WGSL: &str = "\
@group(0) @binding(0) var<storage, read>       Wt: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       X:  array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> Y:  array<u32>;

const KG: u32 = {kg}u;
const TILES: u32 = {tiles}u;

var<workgroup> bt0: array<f32, 256>;
var<workgroup> bt1: array<f32, 256>;
var<workgroup> rows_out: array<f32, 8>;

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
            Y[wordbase + w] = pack2x16float(vec2<f32>(rows_out[2u * w], rows_out[2u * w + 1u]));
        }}
    }}
}}
";

const XSMEM_WGSL: &str = "\
@group(0) @binding(0) var<storage, read>       Wt: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       X:  array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> Y:  array<u32>;

const KG: u32 = {kg}u;
const TILES: u32 = {tiles}u;
const XW: u32 = {xw}u;

var<workgroup> xs:     array<vec4<f32>, {xw2}u>;
var<workgroup> rows_out: array<f32, 8>;
var<workgroup> bt0: array<f32, 256>;
var<workgroup> bt1: array<f32, 256>;

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

    for (var i = lid.x; i < XW; i = i + 256u) {{
        let v = X[i];
        let lo = unpack2x16float(v.x);
        let hi = unpack2x16float(v.y);
        let lo2 = unpack2x16float(v.z);
        let hi2 = unpack2x16float(v.w);
        xs[2u * i] = vec4<f32>(lo.x, lo.y, hi.x, hi.y);
        xs[2u * i + 1u] = vec4<f32>(lo2.x, lo2.y, hi2.x, hi2.y);
    }}
    workgroupBarrier();

    let wbase = row * KG;
    var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
    var i = lane;
    for (var g = 0u; g < TILES; g = g + 4u) {{
        let wv0 = Wt[wbase + i];
        let x0 = xs[2u * i];
        let x1 = xs[2u * i + 1u];
        let wv1 = Wt[wbase + i + 32u];
        let x2 = xs[2u * (i + 32u)];
        let x3 = xs[2u * (i + 32u) + 1u];
        let wv2 = Wt[wbase + i + 64u];
        let x4 = xs[2u * (i + 64u)];
        let x5 = xs[2u * (i + 64u) + 1u];
        let wv3 = Wt[wbase + i + 96u];
        let x6 = xs[2u * (i + 96u)];
        let x7 = xs[2u * (i + 96u) + 1u];
        let w00 = unpack2x16float(wv0.x);
        let w01 = unpack2x16float(wv0.y);
        let w02 = unpack2x16float(wv0.z);
        let w03 = unpack2x16float(wv0.w);
        a0 = fma(w00.x, x0.x, fma(w00.y, x0.y, a0));
        a1 = fma(w01.x, x0.z, fma(w01.y, x0.w, a1));
        a2 = fma(w02.x, x1.x, fma(w02.y, x1.y, a2));
        a3 = fma(w03.x, x1.z, fma(w03.y, x1.w, a3));
        let w10 = unpack2x16float(wv1.x);
        let w11 = unpack2x16float(wv1.y);
        let w12 = unpack2x16float(wv1.z);
        let w13 = unpack2x16float(wv1.w);
        a0 = fma(w10.x, x2.x, fma(w10.y, x2.y, a0));
        a1 = fma(w11.x, x2.z, fma(w11.y, x2.w, a1));
        a2 = fma(w12.x, x3.x, fma(w12.y, x3.y, a2));
        a3 = fma(w13.x, x3.z, fma(w13.y, x3.w, a3));
        let w20 = unpack2x16float(wv2.x);
        let w21 = unpack2x16float(wv2.y);
        let w22 = unpack2x16float(wv2.z);
        let w23 = unpack2x16float(wv2.w);
        a0 = fma(w20.x, x4.x, fma(w20.y, x4.y, a0));
        a1 = fma(w21.x, x4.z, fma(w21.y, x4.w, a1));
        a2 = fma(w22.x, x5.x, fma(w22.y, x5.y, a2));
        a3 = fma(w23.x, x5.z, fma(w23.y, x5.w, a3));
        let w30 = unpack2x16float(wv3.x);
        let w31 = unpack2x16float(wv3.y);
        let w32 = unpack2x16float(wv3.z);
        let w33 = unpack2x16float(wv3.w);
        a0 = fma(w30.x, x6.x, fma(w30.y, x6.y, a0));
        a1 = fma(w31.x, x6.z, fma(w31.y, x6.w, a1));
        a2 = fma(w32.x, x7.x, fma(w32.y, x7.y, a2));
        a3 = fma(w33.x, x7.z, fma(w33.y, x7.w, a3));
        i = i + 128u;
    }}
    let acc = (a0 + a1) + (a2 + a3);
    let r = bfly(acc, lid.x, lane);
    if (lane == 0u) {{ rows_out[warp] = r; }}
    workgroupBarrier();
    if (lid.x == 0u) {{
        let wordbase = (wgid.x * 8u) >> 1u;
        for (var w = 0u; w < 4u; w = w + 1u) {{
            Y[wordbase + w] = pack2x16float(vec2<f32>(rows_out[2u * w], rows_out[2u * w + 1u]));
        }}
    }}
}}
";

const ROWS2_WGSL: &str = "\
@group(0) @binding(0) var<storage, read>       Wt: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       X:  array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> Y:  array<u32>;

const KG: u32 = {kg}u;
const TILES: u32 = {tiles}u;

var<workgroup> bt0: array<vec2<f32>, 256>;
var<workgroup> bt1: array<vec2<f32>, 256>;
var<workgroup> rows_out: array<vec2<f32>, 8>;

fn bfly2(v: vec2<f32>, lid: u32, lane: u32) -> vec2<f32> {{
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

fn acc4(w0: vec4<u32>, w1: vec4<u32>, w2: vec4<u32>, w3: vec4<u32>,
        x0: vec4<u32>, x1: vec4<u32>, x2: vec4<u32>, x3: vec4<u32>,
        carry: vec4<f32>) -> vec4<f32> {{
    var a = carry;
    let s0 = unpack2x16float(w0.x); let t0 = unpack2x16float(x0.x);
    let s1 = unpack2x16float(w0.y); let t1 = unpack2x16float(x0.y);
    let s2 = unpack2x16float(w0.z); let t2 = unpack2x16float(x0.z);
    let s3 = unpack2x16float(w0.w); let t3 = unpack2x16float(x0.w);
    a.x = fma(s0.x, t0.x, fma(s0.y, t0.y, a.x));
    a.y = fma(s1.x, t1.x, fma(s1.y, t1.y, a.y));
    a.z = fma(s2.x, t2.x, fma(s2.y, t2.y, a.z));
    a.w = fma(s3.x, t3.x, fma(s3.y, t3.y, a.w));
    let u0 = unpack2x16float(w1.x); let v0 = unpack2x16float(x1.x);
    let u1 = unpack2x16float(w1.y); let v1 = unpack2x16float(x1.y);
    let u2 = unpack2x16float(w1.z); let v2 = unpack2x16float(x1.z);
    let u3 = unpack2x16float(w1.w); let v3 = unpack2x16float(x1.w);
    a.x = fma(u0.x, v0.x, fma(u0.y, v0.y, a.x));
    a.y = fma(u1.x, v1.x, fma(u1.y, v1.y, a.y));
    a.z = fma(u2.x, v2.x, fma(u2.y, v2.y, a.z));
    a.w = fma(u3.x, v3.x, fma(u3.y, v3.y, a.w));
    let p0 = unpack2x16float(w2.x); let q0 = unpack2x16float(x2.x);
    let p1 = unpack2x16float(w2.y); let q1 = unpack2x16float(x2.y);
    let p2 = unpack2x16float(w2.z); let q2 = unpack2x16float(x2.z);
    let p3 = unpack2x16float(w2.w); let q3 = unpack2x16float(x2.w);
    a.x = fma(p0.x, q0.x, fma(p0.y, q0.y, a.x));
    a.y = fma(p1.x, q1.x, fma(p1.y, q1.y, a.y));
    a.z = fma(p2.x, q2.x, fma(p2.y, q2.y, a.z));
    a.w = fma(p3.x, q3.x, fma(p3.y, q3.y, a.w));
    let r0 = unpack2x16float(w3.x); let z0 = unpack2x16float(x3.x);
    let r1 = unpack2x16float(w3.y); let z1 = unpack2x16float(x3.y);
    let r2 = unpack2x16float(w3.z); let z2 = unpack2x16float(x3.z);
    let r3 = unpack2x16float(w3.w); let z3 = unpack2x16float(x3.w);
    a.x = fma(r0.x, z0.x, fma(r0.y, z0.y, a.x));
    a.y = fma(r1.x, z1.x, fma(r1.y, z1.y, a.y));
    a.z = fma(r2.x, z2.x, fma(r2.y, z2.y, a.z));
    a.w = fma(r3.x, z3.x, fma(r3.y, z3.y, a.w));
    return a;
}}

@compute @workgroup_size(256)
fn gemv(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let lane = lid.x & 31u;
    let warp = lid.x >> 5u;
    let row0 = wgid.x * 16u + warp * 2u;
    let wb0 = row0 * KG;
    let wb1 = (row0 + 1u) * KG;

    var a = vec4<f32>(0.0);
    var b = vec4<f32>(0.0);
    var i = lane;
    for (var g = 0u; g < TILES; g = g + 4u) {{
        let x0 = X[i];
        let x1 = X[i + 32u];
        let x2 = X[i + 64u];
        let x3 = X[i + 96u];
        let wa0 = Wt[wb0 + i];
        let wa1 = Wt[wb0 + i + 32u];
        let wa2 = Wt[wb0 + i + 64u];
        let wa3 = Wt[wb0 + i + 96u];
        let wc0 = Wt[wb1 + i];
        let wc1 = Wt[wb1 + i + 32u];
        let wc2 = Wt[wb1 + i + 64u];
        let wc3 = Wt[wb1 + i + 96u];
        a = acc4(wa0, wa1, wa2, wa3, x0, x1, x2, x3, a);
        b = acc4(wc0, wc1, wc2, wc3, x0, x1, x2, x3, b);
        i = i + 128u;
    }}
    // row0: (a.x + a.y) + (a.z + a.w); row1 likewise -- same tree as prod.
    let r = bfly2(vec2<f32>((a.x + a.y) + (a.z + a.w), (b.x + b.y) + (b.z + b.w)), lid.x, lane);
    if (lane == 0u) {{ rows_out[warp] = r; }}
    workgroupBarrier();
    if (lid.x == 0u) {{
        let wordbase = (wgid.x * 16u) >> 1u;
        for (var w = 0u; w < 8u; w = w + 1u) {{
            Y[wordbase + w] = pack2x16float(rows_out[w]);
        }}
    }}
}}
";

const SPLIT_MERGE_WGSL: &str = "\
struct Cfg { words: u32, splits: u32 };

@group(0) @binding(0) var<storage, read>       P: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;

@compute @workgroup_size(256)
fn gemv_merge(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.words) { return; }
    var va = 0.0;
    var vb = 0.0;
    for (var s = 0u; s < cfg.splits; s = s + 1u) {
        va = va + P[(2u * i) * cfg.splits + s];
        vb = vb + P[(2u * i + 1u) * cfg.splits + s];
    }
    Y[i] = pack2x16float(vec2<f32>(va, vb));
}
";

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn make_weight(rows: usize, cols: usize) -> Vec<u8> {
    let mut s: u32 = 12345;
    let mut out = Vec::with_capacity(rows * cols * 2);
    for _ in 0..rows * cols {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let v = half::f16::from_f32(0.5 + ((s >> 8) as f32 / 16777216.0) * 0.5);
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let adapter = arg(&args, "--adapter");
    let gpu = pollster::block_on(Gpu::new(adapter.as_deref()))?;
    println!("adapter: {}", gpu.describe());

    let shapes: [(&str, usize, usize); 5] = [
        ("qkv", 4096, 1024),
        ("o_proj", 1024, 2048),
        ("gate_up", 6144, 1024),
        ("down_proj", 1024, 3072),
        ("lm_head", 151936, 1024),
    ];
    let iters = 50usize;

    println!(
        "{:<11} {:>10} {:>14} {:>12} {:>12} {:>12} {:>12} {:>9}",
        "gemv", "weight(MB)", "prod GB/s", "unroll4", "xsmem", "rows2", "splitK", "p/sp"
    );
    println!("{}", "-".repeat(70));

    let mut totals = [0.0f64; 5];
    let mut bytes_total = 0usize;
    for (name, rows, cols) in shapes {
        let bytes = rows * cols * 2;
        let mut up = gpu.uploader();
        let w = up.storage("w", bytes as u64);
        up.upload(&w, &make_weight(rows, cols))?;
        let x = up.storage("x", (cols * 2) as u64);
        let xv: Vec<u8> = vec![0.75f32; 1]
            .iter()
            .flat_map(|_| half::f16::from_f32(0.75).to_bits().to_le_bytes())
            .cycle()
            .take(cols * 2)
            .collect();
        up.upload(&x, &xv)?;
        let y = up.storage("y", (rows * 2) as u64);
        up.finish()?;

        let kg = cols / 8;
        let tiles = kg / 32;
        assert!(tiles % 4 == 0, "bench shapes need TILES % 4 == 0");
        let grid = (rows / 8) as u32;

        let run = |pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup| -> Result<f64> {
            for _ in 0..3 {
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(pipe);
                cp.set_bind_group(0, bg, &[]);
                cp.dispatch_workgroups(grid, 1, 1);
                drop(cp);
                gpu.queue.submit([enc.finish()]);
            }
            gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
            let t0 = Instant::now();
            for _ in 0..iters {
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(pipe);
                cp.set_bind_group(0, bg, &[]);
                cp.dispatch_workgroups(grid, 1, 1);
                drop(cp);
                gpu.queue.submit([enc.finish()]);
            }
            gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
            Ok(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64)
        };

        let src = shaders::gemv(rows, cols, false, false);
        let pipe_p = gpu.pipeline("prod", &src, "gemv", None)?;
        let bg_p = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prod"),
            layout: &pipe_p.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
            ],
        });
        let ms_p = run(&pipe_p, &bg_p)?;

        let src4 = UNROLL4_WGSL
            .replace("{kg}u", &format!("{kg}u"))
            .replace("{tiles}u", &format!("{tiles}u"));
        let pipe_u = gpu.pipeline("unroll4", &src4, "gemv", None)?;
        let bg_u = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unroll4"),
            layout: &pipe_u.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
            ],
        });
        let ms_u = run(&pipe_u, &bg_u)?;

        let srcs = XSMEM_WGSL
            .replace("{kg}u", &format!("{kg}u"))
            .replace("{tiles}u", &format!("{tiles}u"))
            .replace("{xw2}u", &format!("{}u", cols / 4))
            .replace("{xw}u", &format!("{}u", cols / 8));
        let pipe_s = gpu.pipeline("xsmem", &srcs, "gemv", None)?;
        let bg_s = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("xsmem"),
            layout: &pipe_s.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
            ],
        });
        let ms_s = run(&pipe_s, &bg_s)?;

        let src2 = ROWS2_WGSL
            .replace("{kg}u", &format!("{kg}u"))
            .replace("{tiles}u", &format!("{tiles}u"));
        let pipe_2 = gpu.pipeline("rows2", &src2, "gemv", None)?;
        let bg_2 = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rows2"),
            layout: &pipe_2.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
            ],
        });
        let ms_2 = run(&pipe_2, &bg_2)?;

        let splits = 2usize;
        let srcs = shaders::gemv_split(rows, cols, false, splits);
        let pipe_sp = gpu.pipeline("gemv_split", &srcs, "gemv", None)?;
        let ngran = cols / 8 / 32 / splits;
        let _ = ngran;
        let p_buf = gpu.storage("p_partial", (rows * splits * 4) as u64);
        let merge_src = SPLIT_MERGE_WGSL.replace("{words}u", &format!("{}u", rows / 2));
        let pipe_mg = gpu.pipeline("gemv_merge", &merge_src, "gemv_merge", None)?;
        let u_merge = gpu.uniform("u_merge", 32);
        let cfg: [u32; 2] = [(rows / 2) as u32, splits as u32];
        gpu.queue.write_buffer(&u_merge, 0, bytemuck::cast_slice(&cfg));
        let bg_sp = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("split"),
            layout: &pipe_sp.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: p_buf.as_entire_binding() },
            ],
        });
        let bg_mg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("merge"),
            layout: &pipe_mg.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: u_merge.as_entire_binding() },
            ],
        });
        let _ = ngran;
        let run_split = |pipe_sp: &wgpu::ComputePipeline, bg_sp: &wgpu::BindGroup,
                         pipe_mg: &wgpu::ComputePipeline, bg_mg: &wgpu::BindGroup|
         -> Result<f64> {
            for _ in 0..3 {
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                {
                    let mut cp = enc.begin_compute_pass(&Default::default());
                    cp.set_pipeline(pipe_sp);
                    cp.set_bind_group(0, bg_sp, &[]);
                    cp.dispatch_workgroups(grid, 1, splits as u32);
                    cp.set_pipeline(pipe_mg);
                    cp.set_bind_group(0, bg_mg, &[]);
                    cp.dispatch_workgroups(((rows / 2) as u32).div_ceil(256), 1, 1);
                }
                gpu.queue.submit([enc.finish()]);
            }
            gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
            let t0 = Instant::now();
            for _ in 0..iters {
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                {
                    let mut cp = enc.begin_compute_pass(&Default::default());
                    cp.set_pipeline(pipe_sp);
                    cp.set_bind_group(0, bg_sp, &[]);
                    cp.dispatch_workgroups(grid, 1, splits as u32);
                    cp.set_pipeline(pipe_mg);
                    cp.set_bind_group(0, bg_mg, &[]);
                    cp.dispatch_workgroups(((rows / 2) as u32).div_ceil(256), 1, 1);
                }
                gpu.queue.submit([enc.finish()]);
            }
            gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
            Ok(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64)
        };
        let ms_sp = run_split(&pipe_sp, &bg_sp, &pipe_mg, &bg_mg)?;

        let bw = |ms: f64| bytes as f64 / 1e9 / (ms / 1000.0);
        println!(
            "{:<11} {:>10.1} {:>14.1} {:>12.1} {:>12.1} {:>12.1} {:>12.1} {:>9.2}",
            name,
            bytes as f64 / 1048576.0,
            bw(ms_p),
            bw(ms_u),
            bw(ms_s),
            bw(ms_2),
            bw(ms_sp),
            ms_p / ms_sp
        );
        totals[0] += ms_p;
        totals[1] += ms_u;
        totals[2] += ms_s;
        totals[3] += ms_2;
        totals[4] += ms_sp;
        bytes_total += bytes;
    }
    println!("{}", "-".repeat(70));
    println!(
        "aggregate: prod {:.3} ms ({:.1} GB/s) | unroll4 {:.1} | xsmem {:.1} | rows2 {:.3} ms ({:.1} GB/s)",
        totals[0],
        bytes_total as f64 / 1e9 / (totals[0] / 1000.0),
        bytes_total as f64 / 1e9 / (totals[1] / 1000.0),
        bytes_total as f64 / 1e9 / (totals[2] / 1000.0),
        totals[3],
        bytes_total as f64 / 1e9 / (totals[3] / 1000.0),
    );
    Ok(())
}
