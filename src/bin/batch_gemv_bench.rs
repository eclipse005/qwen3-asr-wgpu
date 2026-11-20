//! Batched GEMV microbenchmark: one weight read, `batch` activation vectors.
//!
//! Prices the kernel that decode batching stands on (`shaders::gemv_batch`)
//! at the shapes the 1.7B decoder actually issues, and checks it against a
//! f32 reference so a faster-but-wrong kernel cannot walk out of here.
//!
//! Run from `target/release`:
//!   `cargo run --release --bin batch_gemv_bench`

use std::time::Instant;

use anyhow::Result;

use qwen3_asr_wgpu::gpu::Gpu;
use qwen3_asr_wgpu::shaders;

const RPW: usize = 8;
const REPS: u32 = 16;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn f16_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

/// Time `REPS` chained dispatches (one submit) and return ms per dispatch.
fn chain(gpu: &Gpu, pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, grid: u32) -> Result<f64> {
    for _ in 0..2 {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(pipe);
        cp.set_bind_group(0, bg, &[]);
        for _ in 0..REPS {
            cp.dispatch_workgroups(grid, 1, 1);
        }
        drop(cp);
        gpu.queue.submit([enc.finish()]);
    }
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;

    let t0 = Instant::now();
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    let mut cp = enc.begin_compute_pass(&Default::default());
    cp.set_pipeline(pipe);
    cp.set_bind_group(0, bg, &[]);
    for _ in 0..REPS {
        cp.dispatch_workgroups(grid, 1, 1);
    }
    drop(cp);
    gpu.queue.submit([enc.finish()]);
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
    Ok(t0.elapsed().as_secs_f64() * 1e3 / REPS as f64)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let adapter = arg(&args, "--adapter");
    let gpu = pollster::block_on(Gpu::new(adapter.as_deref()))?;
    println!("adapter: {}", gpu.describe());

    // The 1.7B text tower's four weight shapes plus lm_head (hidden 2048,
    // intermediate 6144, vocab 151936).
    let shapes: [(&str, usize, usize); 4] = [
        ("o_proj/qkv-lm", 2048, 2048),
        ("gate_up", 6144, 2048),
        ("down_proj", 2048, 6144),
        ("lm_head", 151936, 2048),
    ];
    let batches = [1usize, 2, 4, 8];

    println!(
        "\n{:<14} {:>6} {:>6} {:>5} {:>9} {:>9} {:>9}",
        "shape", "rows", "k", "b", "ms/disp", "GB/s", "tok/s@step"
    );
    println!("{}", "-".repeat(66));

    for (name, rows, k) in shapes {
        let w_bytes = (rows * k * 2) as u64;
        // Deterministic f16 weights: no RNG dependency, and every value is exactly
        // representable so the reference check below has a tight tolerance.
        let mut w = vec![0u16; rows * k];
        for i in 0..w.len() {
            w[i] = f16_bits(((i % 7) as f32) * 0.25);
        }
        let wbuf = gpu.storage("w", w_bytes);
        gpu.upload(&wbuf, bytemuck::cast_slice(&w));

        for b in batches {
            let x: Vec<u16> = (0..b * k).map(|i| f16_bits(((i % 5) as f32) * 0.5)).collect();
            let y_bytes = (b * rows * 2) as u64;
            let xbuf = gpu.storage("x", (b * k * 2) as u64);
            let ybuf = gpu.storage("y", y_bytes);
            gpu.upload(&xbuf, bytemuck::cast_slice(&x));

            let src = shaders::gemv_batch(rows, k, b);
            let pipe = gpu.pipeline(&format!("gb{b}"), &src, "gemv_batch", None)?;
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gb"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wbuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: xbuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ybuf.as_entire_binding() },
                ],
            });
            let grid = (rows / RPW) as u32;
            let ms = chain(&gpu, &pipe, &bg, grid)?;
            let gbs = w_bytes as f64 / (ms / 1e3) / 1e9;
            println!(
                "{name:<14} {rows:>6} {k:>6} {b:>5} {ms:>9.2} {gbs:>9.1} {:>9.0}",
                b as f64 / (ms / 1e3)
            );

            // ── correctness: first four rows of every batch vector vs f32 ──
            if rows <= 6144 && (b == 1 || b == 8) {
                let got: Vec<u16> = gpu
                    .readback(&ybuf, y_bytes)?
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let mut worst = 0.0f64;
                for bb in 0..b {
                    for r in 0..4 {
                        let mut want = 0.0f32;
                        for c in 0..k {
                            want += half::f16::from_bits(w[r * k + c]).to_f32()
                                * half::f16::from_bits(x[bb * k + c]).to_f32();
                        }
                        let gotv = half::f16::from_bits(got[bb * rows + r]).to_f32();
                        let denom = want.abs().max(1.0) as f64;
                        worst = worst.max((gotv - want).abs() as f64 / denom);
                    }
                }
                let ok = if worst < 3e-3 { "OK" } else { "FAIL" };
                println!("    correctness b={b}: max rel err {worst:.2e}  {ok}");
                if ok == "FAIL" {
                    anyhow::bail!("gemv_batch b={b} {name}: rel err {worst:.2e}");
                }
            }
        }
    }

    // ── cold sweep: one full step's weight stream, b=1 vs b=8 ──────────
    // The shapes above are small enough to sit in Infinity Cache, which flatters
    // b=1.  A real step streams 3.44 GB -- 28 layers x (qkv+o+gu+dp) plus
    // lm_head -- so this sweeps the same dispatches over a 1 GiB pool at rotating
    // offsets (always cold).  One sweep is one step, and one step yields `b`
    // tokens.  The pool is left zeroed: only the traffic is being priced.
    if args.iter().any(|a| a == "--cold") {
        let pool_bytes: u64 = 1 << 30;
        let pool = gpu.storage("cold_pool", pool_bytes);
        println!("
--- cold step sweep (3.44 GB of weights per step) ---");
        println!("{:>3} {:>10} {:>10} {:>12} {:>9}", "b", "step ms", "GB/s", "tok/s", "vs b=1");
        let per_layer: [(&str, usize, usize); 4] = [
            ("qkv", 4096, 2048),
            ("o_proj", 2048, 2048),
            ("gate_up", 6144, 2048),
            ("down_proj", 2048, 6144),
        ];
        let layers = 28usize;
        let (lm_rows, lm_k): (usize, usize) = (151936, 2048);
        let step_bytes: u64 = per_layer
            .iter()
            .map(|&(_, r, k)| (r * k * 2) as u64 * layers as u64)
            .sum::<u64>()
            + (lm_rows * lm_k * 2) as u64;
        let mut base_tok = 0.0f64;
        for b in [1usize, 8] {
            let x = gpu.storage("cold_x", (b * 6144 * 2) as u64);
            let y = gpu.storage("cold_y", (b * lm_rows * 2) as u64);
            let mut pipes: Vec<wgpu::ComputePipeline> = Vec::new();
            let mut jobs: Vec<(usize, wgpu::BindGroup, u32)> = Vec::new();
            let mut cursor: u64 = 0;
            let mut slice_bg = |pipe: &wgpu::ComputePipeline, off: u64, bytes: u64| {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("cold"),
                    layout: &pipe.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &pool,
                                offset: off & !255,
                                size: std::num::NonZeroU64::new(bytes),
                            }),
                        },
                        wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                    ],
                })
            };
            for &(_, rows, k) in per_layer.iter() {
                let src = shaders::gemv_batch(rows, k, b);
                let pipe = gpu.pipeline(&format!("cold{rows}x{k}b{b}"), &src, "gemv_batch", None)?;
                let bytes = (rows * k * 2) as u64;
                let idx = pipes.len();
                let grid = (rows / RPW) as u32;
                for _ in 0..layers {
                    let off = cursor % (pool_bytes - bytes);
                    cursor += bytes;
                    let bg = slice_bg(&pipe, off, bytes);
                    jobs.push((idx, bg, grid));
                }
                pipes.push(pipe);
            }
            {
                let src = shaders::gemv_batch(lm_rows, lm_k, b);
                let pipe = gpu.pipeline(&format!("coldlm{b}"), &src, "gemv_batch", None)?;
                let bytes = (lm_rows * lm_k * 2) as u64;
                let idx = pipes.len();
                let bg = slice_bg(&pipe, 0, bytes);
                jobs.push((idx, bg, (lm_rows / RPW) as u32));
                pipes.push(pipe);
            }
            let run = || -> Result<f64> {
                for _ in 0..1 {
                    let mut enc = gpu.device.create_command_encoder(&Default::default());
                    {
                        let mut cp = enc.begin_compute_pass(&Default::default());
                        for (i, bg, g) in jobs.iter() {
                            cp.set_pipeline(&pipes[*i]);
                            cp.set_bind_group(0, bg, &[]);
                            cp.dispatch_workgroups(*g, 1, 1);
                        }
                    }
                    gpu.queue.submit([enc.finish()]);
                }
                gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
                let t = Instant::now();
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                {
                    let mut cp = enc.begin_compute_pass(&Default::default());
                    for (i, bg, g) in jobs.iter() {
                        cp.set_pipeline(&pipes[*i]);
                        cp.set_bind_group(0, bg, &[]);
                        cp.dispatch_workgroups(*g, 1, 1);
                    }
                }
                gpu.queue.submit([enc.finish()]);
                gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
                Ok(t.elapsed().as_secs_f64() * 1e3)
            };
            let ms = run()?;
            let tok = b as f64 / (ms / 1e3);
            if b == 1 {
                base_tok = tok;
            }
            println!(
                "{b:>3} {ms:>10.2} {:>10.1} {tok:>12.0} {:>8.2}x",
                step_bytes as f64 / (ms / 1e3) / 1e9,
                tok / base_tok
            );
        }
    }

    // b=1 at a batch of 8 must not regress the b=1 path: report the ratio once.
    println!("\n(GPU: {}, wg per dispatch = rows/8)", gpu.describe());
    Ok(())
}
