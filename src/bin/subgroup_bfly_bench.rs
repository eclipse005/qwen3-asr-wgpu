//! A/B: shared-memory butterfly vs `subgroupShuffleXor` for the decode GEMV's
//! 32-lane reduction.
//!
//! `shaders::gemv` reduces each output row with a 5-round xor butterfly
//! (`lane ^ 16, ^8, ^4, ^2, ^1`) done through `var<workgroup>` plus **5
//! workgroupBarrier()s** — the comment says it is shared-memory because "WGSL
//! has no portable warp shuffle on this wgpu version".  `feature_probe` just
//! showed this stack *does* support `Features::SUBGROUP` (P104-100 / Vulkan /
//! 572.75), so the shuffle form is available.
//!
//! Both kernels below compute the same thing and must agree **exactly**: the
//! xor order is preserved, and f32 addition is performed in the same sequence,
//! so the reduction tree — and therefore every bit — is unchanged.  That is the
//! property that matters, because the decode chain's alignment gate depends on
//! the current tree.
//!
//! ```text
//! cargo run --release --bin subgroup_bfly_bench -- --adapter nvidia
//! ```

use anyhow::Result;
use wgpu::util::DeviceExt;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

/// Shared-memory butterfly — a copy of `shaders::gemv`'s `bfly`.
const WGSL_SMEM: &str = r#"
struct P { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;

var<workgroup> bt0: array<f32, 256>;
var<workgroup> bt1: array<f32, 256>;

fn bfly(v: f32, lid: u32, lane: u32) -> f32 {
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
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x & 31u;
    let warp = lid.x >> 5u;
    let row = wgid.x * 8u + warp;
    if (row >= p.n) { return; }
    var acc = 0.0;
    var i = lane;
    loop {
        if (i >= 512u) { break; }
        acc = acc + src[(row % 16u) * 512u + i];
        i = i + 32u;
    }
    let t = bfly(acc, lid.x, lane);
    if (lane == 0u) { dst[row] = t; }
}
"#;

/// Subgroup form: same xor order, no barrier, no shared memory.
const WGSL_SUB: &str = r#"
struct P { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x & 31u;
    let warp = lid.x >> 5u;
    let row = wgid.x * 8u + warp;
    if (row >= p.n) { return; }
    var acc = 0.0;
    var i = lane;
    loop {
        if (i >= 512u) { break; }
        acc = acc + src[(row % 16u) * 512u + i];
        i = i + 32u;
    }
    // identical tree to bfly(): 16, 8, 4, 2, 1
    var t = acc;
    t = t + subgroupShuffleXor(t, 16u);
    t = t + subgroupShuffleXor(t, 8u);
    t = t + subgroupShuffleXor(t, 4u);
    t = t + subgroupShuffleXor(t, 2u);
    t = t + subgroupShuffleXor(t, 1u);
    if (lane == 0u) { dst[row] = t; }
}
"#;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    pollster::block_on(run(arg(&args, "--adapter").as_deref()))
}

async fn run(prefer: Option<&str>) -> Result<()> {
    let instance = wgpu::Instance::default();
    let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
    let adapter = match prefer {
        Some(w) => {
            let w = w.to_lowercase();
            adapters
                .iter()
                .find(|a| a.get_info().name.to_lowercase().contains(&w))
                .ok_or_else(|| anyhow::anyhow!("no adapter matching {w:?}"))?
        }
        None => adapters
            .iter()
            .find(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
            .unwrap_or(&adapters[0]),
    };
    let info = adapter.get_info();
    let feats = adapter.features();
    anyhow::ensure!(feats.contains(wgpu::Features::SUBGROUP), "SUBGROUP unavailable on {}", info.name);
    println!("adapter: {} ({:?})", info.name, info.backend);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("bfly_bench"),
            required_features: wgpu::Features::SUBGROUP,
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await?;
    device.on_uncaptured_error(std::sync::Arc::new(|e| eprintln!("[wgpu error] {e}")));

    // rows = 28 * 6 GEMVs per decode step would be thousands; use the shape the
    // decoder actually dispatches: one warp per output row.
    for &(rows, iters) in &[(512usize, 200usize), (18992usize, 50usize)] {
        let n = rows * 8; // 8 warps per workgroup
        let gx = (rows as u32).div_ceil(8);

        let src: Vec<f32> = (0..16 * 512)
            .map(|i| ((i % 29) as f32) * 0.031 - 0.4)
            .collect();
        let src_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&src),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params = [rows as u32, 0, 0, 0];
        let pbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let mk_out = || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (n * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let mut results = Vec::new();
        for (label, wgsl) in [("shared-mem bfly", WGSL_SMEM), ("subgroup xor", WGSL_SUB)] {
            let guard = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
            let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            if let Some(e) = pollster::block_on(guard.pop()) {
                anyhow::bail!("{label} pipeline: {e}");
            }
            let out = mk_out();
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: src_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: pbuf.as_entire_binding() },
                ],
            });

            // warm up
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&pipe);
                cp.set_bind_group(0, &bg, &[]);
                cp.dispatch_workgroups(gx, 1, 1);
            }
            queue.submit([enc.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely())?;

            // all iterations in ONE encoder + ONE submit, so the measurement is
            // the dispatch loop's GPU cost and not per-submit overhead
            let t = std::time::Instant::now();
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&pipe);
                cp.set_bind_group(0, &bg, &[]);
                for _ in 0..iters {
                    cp.dispatch_workgroups(gx, 1, 1);
                }
            }
            queue.submit([enc.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely())?;
            let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            // read back for the bit-exactness check
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (n * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&out, 0, &staging, 0, (n * 4) as u64);
            queue.submit([enc.finish()]);
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            device.poll(wgpu::PollType::wait_indefinitely())?;
            rx.recv()??;
            let data = slice.get_mapped_range()?;
            let words: Vec<u32> = data
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            results.push((label, ms, words));
        }

        let (l0, ms0, w0) = &results[0];
        let (l1, ms1, w1) = &results[1];
        let exact = w0[..rows] == w1[..rows];
        println!("\n-- rows={rows}, 8 warps/workgroup, {iters} dispatches, 1 submit --");
        println!("  {l0:<16} {ms0:8.4} ms/dispatch");
        println!("  {l1:<16} {ms1:8.4} ms/dispatch   -> {:.2}x", ms0 / ms1);
        println!(
            "  bit-exact ({} rows): {}",
            rows,
            if exact { "IDENTICAL" } else { "*** DIFFERS ***" }
        );
        if !exact {
            for i in 0..rows.min(18992) {
                if w0[i] != w1[i] {
                    println!(
                        "    first diff row {i}: smem {:.9} ({:#010x}) vs sub {:.9} ({:#010x})",
                        f32::from_bits(w0[i]),
                        w0[i],
                        f32::from_bits(w1[i]),
                        w1[i]
                    );
                    break;
                }
            }
        }
    }
    println!("\ndone.");
    Ok(())
}
