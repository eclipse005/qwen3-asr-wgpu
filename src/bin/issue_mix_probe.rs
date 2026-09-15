use std::time::Instant;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let prefer = arg(&args, "--adapter");
    pollster::block_on(run(prefer.as_deref()))
}

async fn run(prefer: Option<&str>) -> anyhow::Result<()> {
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
    println!("adapter : {} ({:?})", info.name, info.backend);
    println!("SHADER_F16: {}", if feats.contains(wgpu::Features::SHADER_F16) { "YES" } else { "no" });

    let mut want = wgpu::Features::empty();
    if feats.contains(wgpu::Features::SHADER_F16) {
        want |= wgpu::Features::SHADER_F16;
    }
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("f16_arith_probe"),
            required_features: want,
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await?;
    device.on_uncaptured_error(std::sync::Arc::new(|e| eprintln!("[wgpu error] {e}")));

    let wgsl = r#"
const PAD: u32 = 260u;
const ITERS: u32 = 256u;

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

var<workgroup> tile: array<f32, 256 * 260>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let tx = lid.x;
  var f = tx;
  loop {
    if (f >= 256u * 256u) { break; }
    // varying values, so no constant folding
    tile[(f / 256u) * PAD + (f % 256u)] = inp[f % 4096u] + f32(f % 7u) * 0.001;
    f = f + 256u;
  }
  workgroupBarrier();

  var c0 = 0.0; var c1 = 0.0; var c2 = 0.0; var c3 = 0.0;
  var c4 = 0.0; var c5 = 0.0; var c6 = 0.0; var c7 = 0.0;
  var idx = 0u;
  let a = 1.0000001;
  // 8 loads + 64 FMAs per iteration (prefill_gemm's per-k-step shape)
  loop {
    if (idx >= ITERS) { break; }
    let base = ((idx * 3u + (bitcast<u32>(c0) & 15u)) & 255u) * PAD + tx;
    let b0 = tile[base];
    let b1 = tile[(base + 17u) & 0xFFFFFFu];
    let b2 = tile[(base + 33u) & 0xFFFFFFu];
    let b3 = tile[(base + 51u) & 0xFFFFFFu];
    let b4 = tile[(base + 67u) & 0xFFFFFFu];
    let b5 = tile[(base + 83u) & 0xFFFFFFu];
    let b6 = tile[(base + 99u) & 0xFFFFFFu];
    let b7 = tile[(base + 113u) & 0xFFFFFFu];
    c0 = fma(a, b0, c0); c1 = fma(a, b1, c1); c2 = fma(a, b2, c2); c3 = fma(a, b3, c3);
    c4 = fma(a, b4, c4); c5 = fma(a, b5, c5); c6 = fma(a, b6, c6); c7 = fma(a, b7, c7);
    c0 = fma(a, b1, c0); c1 = fma(a, b2, c1); c2 = fma(a, b3, c2); c3 = fma(a, b4, c3);
    c4 = fma(a, b5, c4); c5 = fma(a, b6, c5); c6 = fma(a, b7, c6); c7 = fma(a, b0, c7);
    c0 = fma(a, b2, c0); c1 = fma(a, b3, c1); c2 = fma(a, b4, c2); c3 = fma(a, b5, c3);
    c4 = fma(a, b6, c4); c5 = fma(a, b7, c5); c6 = fma(a, b0, c6); c7 = fma(a, b1, c7);
    c0 = fma(a, b3, c0); c1 = fma(a, b4, c1); c2 = fma(a, b5, c2); c3 = fma(a, b6, c3);
    c4 = fma(a, b7, c4); c5 = fma(a, b0, c5); c6 = fma(a, b1, c6); c7 = fma(a, b2, c7);
    c0 = fma(a, b4, c0); c1 = fma(a, b5, c1); c2 = fma(a, b6, c2); c3 = fma(a, b7, c3);
    c4 = fma(a, b0, c4); c5 = fma(a, b1, c5); c6 = fma(a, b2, c6); c7 = fma(a, b3, c7);
    c0 = fma(a, b5, c0); c1 = fma(a, b6, c1); c2 = fma(a, b7, c2); c3 = fma(a, b0, c3);
    c4 = fma(a, b1, c4); c5 = fma(a, b2, c5); c6 = fma(a, b3, c6); c7 = fma(a, b4, c7);
    c0 = fma(a, b6, c0); c1 = fma(a, b7, c1); c2 = fma(a, b0, c2); c3 = fma(a, b1, c3);
    c4 = fma(a, b2, c4); c5 = fma(a, b3, c5); c6 = fma(a, b4, c6); c7 = fma(a, b5, c7);
    c0 = fma(a, b7, c0); c1 = fma(a, b0, c1); c2 = fma(a, b1, c2); c3 = fma(a, b2, c3);
    c4 = fma(a, b3, c4); c5 = fma(a, b4, c5); c6 = fma(a, b5, c6); c7 = fma(a, b6, c7);
    idx = idx + 1u;
  }
  out[wid.x * 256u + tx] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
}
"#;

    let guard = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("probe"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("probe"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    if let Some(e) = pollster::block_on(guard.pop()) {
        anyhow::bail!("validation: {e}");
    }

    let inp = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4096 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&inp, 0, bytemuck::cast_slice(&vec![0.37f32; 4096]));
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4096 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: inp.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
        ],
    });

    let gx = 144u32;
    let run = |reps: usize| -> anyhow::Result<f64> {
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&pipe);
            cp.set_bind_group(0, &bg, &[]);
            for _ in 0..reps {
                cp.dispatch_workgroups(gx, 1, 1);
            }
        }
        let t = Instant::now();
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely())?;
        Ok(t.elapsed().as_secs_f64() * 1e3 / reps as f64)
    };
    run(3)?;
    let ms = run(20)?;
    let fmas = gx as f64 * 256.0 * 256.0 * 64.0;
    let loads = gx as f64 * 256.0 * 256.0 * 8.0;
    println!(
        "\nshape: 8 shared loads + 64 FMA per iteration\n  {ms:.4} ms/dispatch  ->  {:.0} GFLOP/s  ({:.2} TFLOP/s)",
        fmas * 2.0 / (ms / 1e3) / 1e9,
        fmas * 2.0 / (ms / 1e3) / 1e12
    );
    println!(
        "  shared loads/s = {:.2e} ({:.1} per SM per cycle at 1.6 GHz, 20 SM)",
        loads / (ms / 1e3),
        loads / (ms / 1e3) / 20.0 / 1.6e9
    );
    println!("\n(compare: real prefill_gemm = 2.07 TFLOP/s)");
    Ok(())
}
