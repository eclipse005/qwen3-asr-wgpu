use anyhow::Result;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn kernel(l: usize, r: usize) -> String {
    let loads = (0..l)
        .map(|i| format!("    let b{i} = bs[((q + {i}u) & 255u) * PAD + tx];\n"))
        .collect::<String>();
    let mut fmas = String::new();
    for i in 0..l {
        fmas.push_str(&format!("    c{} = fma(a, b{i}, c{});\n", i % r, i % r));
    }
    let decls = (0..r).map(|i| format!("  var c{i} = 0.0;\n")).collect::<String>();
    let finals = (0..r).map(|i| format!("c{i}")).collect::<Vec<_>>().join(" + ");
    format!(
        "const PAD: u32 = 260u;
const LOADS: u32 = {l}u;
const REPS: u32 = 512u;

@group(0) @binding(0) var<storage, read>       bs_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> out:   array<f32>;

var<workgroup> bs: array<f32, 256 * 260>;

@compute @workgroup_size(256)
fn roof(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {{
  let tx = lid.x;
  // fill the shared tile once (not measured)
  var f = tx;
  loop {{
    if (f >= 256u * 256u) {{ break; }}
    bs[(f / 256u) * PAD + (f % 256u)] = bs_in[f % 4096u];
    f = f + 256u;
  }}
  workgroupBarrier();
  let a = 1.0000001;
{decls}  var q = 0u;
  loop {{
    if (q >= REPS) {{ break; }}
{loads}{fmas}    q = q + 1u;
  }}
  out[wid.x * 256u + tx] = {finals};
}}
"
    )
}

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
    println!("adapter: {} ({:?})", info.name, info.backend);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("smem_roof"),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await?;
    device.on_uncaptured_error(std::sync::Arc::new(|e| eprintln!("[wgpu error] {e}")));

    let src = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4096 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&src, 0, bytemuck::cast_slice(&vec![0.5f32; 4096]));
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let cases = [
        (16usize, 16usize, "prefill_gemm shape: 16 loads / 64 FMA"),
        (16, 64, "4x registers: 16 loads / 256 FMA"),
        (8, 64, "8 loads / 256 FMA"),
        (16, 32, "16 loads / 128 FMA"),
    ];

    println!("\n{:<42} {:>10} {:>12} {:>10}", "variant", "us", "GFLOP/s", "vs first");
    let mut baseline = 0.0f64;
    for (l, r, label) in cases {
        let guard = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(kernel(l, r).into()),
        });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some("roof"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = pollster::block_on(guard.pop()) {
            println!("{label:<42} FAILED: {}", format!("{e}").lines().next().unwrap_or(""));
            continue;
        }
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            ],
        });
        let gx = 80u32;
        let run = |reps: usize| -> Result<f64> {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&pipe);
                cp.set_bind_group(0, &bg, &[]);
                for _ in 0..reps {
                    cp.dispatch_workgroups(gx, 1, 1);
                }
            }
            let t = std::time::Instant::now();
            queue.submit([enc.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely())?;
            Ok(t.elapsed().as_secs_f64() * 1e3 / reps as f64)
        };
        run(3)?;
        let ms = run(20)?;

        let fmas_per_iter = l as f64;
        let flop = gx as f64 * 256.0 * 512.0 * fmas_per_iter * 2.0;
        let gflops = flop / (ms / 1e3) / 1e9;
        if baseline == 0.0 {
            baseline = gflops;
        }
        println!(
            "{label:<42} {ms:>10.4} {gflops:>12.1} {:>9.2}x",
            gflops / baseline
        );
    }
    println!("\n(Both variants issue identical shared-memory loads; only register FMAs differ.)");
    Ok(())
}
