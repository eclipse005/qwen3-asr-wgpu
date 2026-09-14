//! Isolated GEMV benchmark mirroring `cudarc_engine::gemv_vs_cublas_bandwidth`
//! (same five decode projections, same deterministic weights, 50 timed iters)
//! so the numbers are directly comparable to the CUDA baseline.
//!
//! Variants:
//! * `prod`   — the production `shaders::gemv` (subgroup-shuffle butterfly)
//! * `unroll4`— experimental: tile loop unrolled ×4 with prefetched loads
//!              (accumulation order unchanged — bit-identical results)
//!
//! ```text
//! cargo run --release --manifest-path wgpu/Cargo.toml --bin gemv_bench -- --adapter nvidia
//! ```

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

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

/// Deterministic f16 weights in [0.5, 1.0) — identical LCG to the CUDA bench.
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
        "{:<11} {:>10} {:>16} {:>16} {:>9}",
        "gemv", "weight(MB)", "prod GB/s", "unroll4 GB/s", "u4/prod"
    );
    println!("{}", "-".repeat(70));

    let mut totals = [0.0f64; 2];
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

        // production kernel
        // production kernel, in BOTH reduction flavours: the subgroup form must
        // be bit-identical and is the one the decoder now uses when the adapter
        // exposes `Features::SUBGROUP` (see `subgroup_bfly_bench`).
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

        // unroll4 kernel
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

        let bw = |ms: f64| bytes as f64 / 1e9 / (ms / 1000.0);
        println!(
            "{:<11} {:>10.1} {:>16.1} {:>16.1} {:>9.2}",
            name,
            bytes as f64 / 1048576.0,
            bw(ms_p),
            bw(ms_u),
            ms_p / ms_u
        );
        totals[0] += ms_p;
        totals[1] += ms_u;
        bytes_total += bytes;
    }
    println!("{}", "-".repeat(70));
    println!(
        "aggregate: prod {:.3} ms ({:.1} GB/s) | unroll4 {:.3} ms ({:.1} GB/s) | CUDA kernel ref 289 GB/s",
        totals[0],
        bytes_total as f64 / 1e9 / (totals[0] / 1000.0),
        totals[1],
        bytes_total as f64 / 1e9 / (totals[1] / 1000.0),
    );
    Ok(())
}

