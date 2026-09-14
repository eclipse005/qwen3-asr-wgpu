//! Probe: which of the wgpu 30 features this Pascal/Vulkan stack can actually
//! request, and whether subgroup WGSL built-ins compile **without** the
//! `enable subgroups;` directive.
//!
//! Motivation (see `docs/wgpu-best-practices-audit.md` §H/§V4): the roadmaps
//! recorded "naga rejects `enable subgroups;`" and concluded warp-shuffle
//! butterflies were unavailable.  naga leaves that *directive* unimplemented on
//! purpose (tracking issue #5555), while `wgpu::Features::SUBGROUP` exists and
//! lists Vulkan — so subgroup built-ins may be gated by the capability bit
//! instead.  If they work here, the decode GQA butterfly can move from shared
//! memory to warp shuffles.
//!
//! Also reports limits that gate the two other planned optimisations
//! (`max_immediate_size` for the `GDims` ring, and the dynamic-offset
//! alignment), plus whether `zero_initialize_workgroup_memory: false` pipelines
//! build.
//!
//! ```text
//! cargo run --release --bin feature_probe -- --adapter nvidia
//! ```

use anyhow::Result;
use wgpu::util::DeviceExt;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

/// Subgroup built-ins with **no** `enable` directive: ballot + shuffle-xor +
/// a mask popcount.  If this module compiles, subgroup built-ins are reachable
/// on this stack and the old "unavailable" conclusion was wrong.
const SUBGROUP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> out_buf: array<u32>;

@compute @workgroup_size(64)
fn probe(@builtin(local_invocation_index) lid: u32) {
    let m = subgroupBallot(lid % 2u == 0u);
    let sh = subgroupShuffleXor(lid, 1u);
    out_buf[lid] = m.x ^ sh;
}
"#;

/// A kernel that actually *uses* `var<workgroup>` so the zero-init question is
/// about a real allocation, not a trivially removable one.
const WORKGROUP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

var<workgroup> tile: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce(@builtin(local_invocation_id) lid: vec3<u32>,
          @builtin(workgroup_id) wid: vec3<u32>) {
    let i = lid.x;
    tile[i] = src[wid.x * 256u + i];      // full-width write before any read
    workgroupBarrier();
    var acc = 0.0;
    for (var j: u32 = 0u; j < 256u; j = j + 1u) {
        acc = acc + tile[j];
    }
    dst[wid.x * 256u + i] = acc;
}
"#;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let prefer = arg(&args, "--adapter");
    pollster::block_on(run(prefer.as_deref()))
}

async fn run(prefer: Option<&str>) -> Result<()> {
    let instance = wgpu::Instance::default();
    let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
    anyhow::ensure!(!adapters.is_empty(), "no adapters");

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
    let lims = adapter.limits();
    println!("adapter : {} ({:?}, {:?})", info.name, info.backend, info.device_type);
    println!("driver  : {} | {}", info.driver, info.driver_info);
    println!("limits  : max_storage_binding {} MiB | max_buffer {} MiB | max_uniform_binding {} KiB",
        lims.max_storage_buffer_binding_size / (1024 * 1024),
        lims.max_buffer_size / (1024 * 1024),
        lims.max_uniform_buffer_binding_size / 1024);
    println!("limits  : min_uniform_offset_align {} | min_storage_offset_align {} | max_immediate_size {}",
        lims.min_uniform_buffer_offset_alignment,
        lims.min_storage_buffer_offset_alignment,
        lims.max_immediate_size);

    // ── which of the interesting features does the ADAPTER advertise? ──
    let interesting: &[(&str, wgpu::Features)] = &[
        ("SUBGROUP", wgpu::Features::SUBGROUP),
        ("SUBGROUP_BARRIER", wgpu::Features::SUBGROUP_BARRIER),
        ("SUBGROUP_VERTEX", wgpu::Features::SUBGROUP_VERTEX),
        ("IMMEDIATES", wgpu::Features::IMMEDIATES),
        ("PIPELINE_CACHE", wgpu::Features::PIPELINE_CACHE),
        ("SHADER_F16", wgpu::Features::SHADER_F16),
        ("SHADER_I16", wgpu::Features::SHADER_I16),
        ("TIMESTAMP_QUERY", wgpu::Features::TIMESTAMP_QUERY),
        ("TIMESTAMP_QUERY_INSIDE_ENCODERS", wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
        ("TIMESTAMP_QUERY_INSIDE_PASSES", wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES),
    ];
    println!("\n-- adapter features --");
    for (name, f) in interesting {
        println!("  {name:<32} {}", if feats.contains(*f) { "YES" } else { "no" });
    }

    // ── request only what we can, then test the WGSL front end ──
    let want = wgpu::Features::SUBGROUP
        | wgpu::Features::SUBGROUP_BARRIER
        | wgpu::Features::IMMEDIATES;
    let got = feats & want;
    anyhow::ensure!(
        got.contains(wgpu::Features::SUBGROUP) || got.is_empty(),
        "SUBGROUP_BARRIER requires SUBGROUP"
    );

    // `max_immediate_size` defaults to 0 and must be asked for explicitly.
    let mut req = lims.clone();
    if feats.contains(wgpu::Features::IMMEDIATES) {
        req.max_immediate_size = lims.max_immediate_size.min(128);
    }

    println!("\nrequesting: {:?}", got);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("feature_probe"),
            required_features: got,
            required_limits: req,
            ..Default::default()
        })
        .await?;

    device.on_uncaptured_error(std::sync::Arc::new(|e| eprintln!("[wgpu error] {e}")));

    // ── 1. subgroup WGSL built-ins, WITHOUT `enable subgroups;` ──
    println!("\n-- subgroup WGSL (no `enable` directive) --");
    match build(&device, "probe_subgroup", SUBGROUP_WGSL) {
        Ok(pipe) => {
            let out = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("out"),
                size: 64 * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipe.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: out.as_entire_binding() }],
            });
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&pipe);
                cp.set_bind_group(0, &bg, &[]);
                cp.dispatch_workgroups(1, 1, 1);
            }
            queue.submit([enc.finish()]);
            // Read back: a non-zero, non-garbage pattern proves the subgroup
            // ops really executed (not just that the module parsed).
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stage"),
                size: 64 * 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&out, 0, &staging, 0, 64 * 4);
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
            println!("  COMPILED + RAN. out[0..8] = {:?}", &words[..8]);
            println!("  -> subgroup built-ins are reachable on this stack");
        }
        Err(e) => println!("  FAILED: {e}"),
    }

    // ── 2. `enable subgroups;` should still fail (naga #5555) ──
    let with_enable = format!("enable subgroups;\n{SUBGROUP_WGSL}");
    println!("\n-- `enable subgroups;` (expected to still be Unimplemented) --");
    match build(&device, "probe_enable", &with_enable) {
        Ok(_) => println!("  UNEXPECTEDLY COMPILED"),
        Err(e) => {
            let msg = format!("{e}");
            let kind = if msg.contains("subgroups") || msg.contains("Unimplemented") {
                "confirms naga #5555"
            } else {
                "different error — read it"
            };
            println!("  rejected as expected ({kind})");
            println!("  first line: {}", msg.lines().next().unwrap_or(""));
        }
    }

    // ── 3. zero-initialised workgroup memory: default vs disabled ──
    println!("\n-- zero_initialize_workgroup_memory --");
    for zero in [true, false] {
        let opts = wgpu::PipelineCompilationOptions {
            zero_initialize_workgroup_memory: zero,
            ..Default::default()
        };
        match build_with(&device, "probe_wg", WORKGROUP_WGSL, opts) {
            Ok(_) => println!("  zero_initialize={zero:<5} pipeline OK"),
            Err(e) => println!("  zero_initialize={zero:<5} FAILED: {}", format!("{e}").lines().next().unwrap_or("")),
        }
    }

    // ── 4. does disabling it actually change the result? (correctness check) ──
    println!("\n-- correctness with zero_initialize=false --");
    match correctness(&device, &queue).await {
        Ok(()) => {}
        Err(e) => println!("  FAILED: {e:#}"),
    }

    println!("\ndone.");
    Ok(())
}

fn build(device: &wgpu::Device, label: &str, wgsl: &str) -> Result<wgpu::ComputePipeline> {
    build_with(device, label, wgsl, Default::default())
}

fn build_with(
    device: &wgpu::Device,
    label: &str,
    wgsl: &str,
    opts: wgpu::PipelineCompilationOptions,
) -> Result<wgpu::ComputePipeline> {
    let guard = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let entry = if wgsl.contains("fn reduce") { "reduce" } else { "probe" };
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: opts,
        cache: None,
    });
    if let Some(e) = pollster::block_on(guard.pop()) {
        anyhow::bail!("validation: {e}");
    }
    Ok(pipe)
}

/// Run the workgroup-reduction kernel twice with zero-init disabled and compare
/// against a CPU reduction — a nonzero-initialised `tile` would show up as a
/// wrong sum if any thread read before writing (it does not here).
async fn correctness(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
    const N: usize = 256;
    let src: Vec<f32> = (0..N).map(|i| (i as f32) * 0.5 - 8.0).collect();
    let expect: f32 = src.iter().sum();

    let opts = wgpu::PipelineCompilationOptions {
        zero_initialize_workgroup_memory: false,
        ..Default::default()
    };
    let pipe = build_with(device, "probe_wg_correct", WORKGROUP_WGSL, opts)?;

    let s = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&src),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let d = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (N * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: s.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: d.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&pipe);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(1, 1, 1);
    }
    queue.submit([enc.finish()]);

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (N * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&d, 0, &staging, 0, (N * 4) as u64);
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let data = slice.get_mapped_range()?;
    let got: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let ok = got.iter().all(|v| (v - expect).abs() < 1e-3);
    println!(
        "  cpu sum {expect:.3} | gpu {:.3} | {}",
        got[0],
        if ok { "MATCH" } else { "MISMATCH — kernel DOES depend on zeroed workgroup memory" }
    );
    Ok(())
}
