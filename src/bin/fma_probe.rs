//! What can this card actually issue, and what does a shared-memory read cost?
//!
//! `docs/perf.md` spent several rounds asserting that the GEMMs were "at the
//! machine's own rate" without ever measuring the machine.  `nvidia-smi` gives
//! 1911 MHz max SM clock and the P104-100 has 1920 FP32 lanes, so *by spec* the
//! card is 7.34 TFLOP/s -- but spec clocks assume the part never throttles, and
//! a mining card at a 180 W cap with no display output is exactly the part that
//! might.  This probe measures the ceiling instead of arguing about it, and
//! prints the number as a function of grid size so the plateau is visible.
//!
//! Run `nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv
//! -l 1` alongside it to see what the SM clock does under load: an FP32 rate
//! well under spec is either throttling or an SM count lower than assumed, and
//! the two are told apart by the clock.
//!
//! Deliberately dumb: no memory at all, sixteen *static* accumulators (a
//! dynamically indexed `var<array>` would be laid out in local memory and the
//! probe would measure that instead), no loop-carried value feeding an address,
//! and a store guarded by a comparison that is never true so nothing is dead
//! code.  The inner body is 16 chains x 4 rounds = 64 FMAs per loop iteration,
//! against ~4 instructions of loop overhead.
//!
//! ## The three mix arms
//!
//! The real GEMM's inner loop is 16 shared loads per 64 FMAs (8 A + 8 B), and it
//! runs at 2.2-3.0 TFLOP/s against this probe's 6.5.  `--mix16` and `--mix4`
//! move the *same 512 words per warp per iteration*, conflict-free in both
//! cases, as 16 scalar loads or as 4 `vec4` loads:
//!
//! * `--mix4` answers "is a shared load expensive per *instruction* or per
//!   *byte*?"  If vec4 closes most of the gap, the GEMM wants a k-major tile
//!   where each thread's eight values are contiguous -- 16 scalar reads per
//!   k-step become 4 `vec4` reads, same bytes.
//! * `--mix16` is the control that should reproduce the real GEMM's rate.  If it
//!   does, this probe has earned the right to predict the GEMM's response to a
//!   layout change; if it does not, the GEMM's cost is not the load count either.
//!
//! Both mix arms charge the FMAs they actually issue (the loaded values have to
//! be accumulated or the loads are dead code), which is why the reported rate is
//! computed from `fma_per_iter` rather than from the pure arm's 64.

use std::time::Instant;

use qwen3_asr_wgpu::gpu::Gpu;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let prefer = arg(&args, "--adapter");
    let mix = if args.iter().any(|a| a == "--mix16") {
        16
    } else if args.iter().any(|a| a == "--mix4") {
        4
    } else if args.iter().any(|a| a == "--mixA") {
        // 8 scalar B loads + 2 vec4 A loads per 64 FMAs: the GEMM's ratio with
        // *only* its broadcast operand vectorised.
        8
    } else if args.iter().any(|a| a == "--mix") {
        // The original arm, kept so its recorded number stays reproducible.
        99
    } else {
        0
    };
    pollster::block_on(run(prefer.as_deref(), mix))
}

const ITERS: u32 = 4096;

/// `mix`: 0 = pure FFMA, 4 = 4 vec4 shared loads per iteration, 16 = 16 scalar
/// shared loads per iteration, 99 = the original 8-load arm.
fn shader(mix: usize) -> String {
    let decl: String = (0..16).map(|i| format!("  var c{i} = 0.0;\n")).collect();
    let init: String = (0..16)
        .map(|i| format!("  c{i} = f32(lid.x + {i}u) * 1e-9;\n"))
        .collect();
    let sink: String = (0..16).map(|i| format!("  s = s + c{i};\n")).collect();

    let mut body = String::new();
    for r in 0..4 {
        for i in 0..16 {
            // `fma(c, a, b)`, not `fma(a, b, c)`: the latter is an affine
            // recurrence in `c` and the compiler folds the whole loop
            // (c_n = c_0 + n*a*b), which is how the first version of this probe
            // "measured" 19.19 TFLOP/s on a 7.34 TFLOP/s card.  A multiplicative
            // recurrence cannot be closed-form'd, and the values stay bounded.
            body.push_str(&format!("    c{i} = fma(c{i}, a, b);\n"));
        }
        match mix {
            4 => {
                // Four `vec4` loads per iteration = 16 words per lane, the same
                // 512 words per warp per iteration the scalar arm moves.  Lane L
                // takes chunk `base + L`, so a phase's eight chunks are
                // contiguous words and every bank is used once.
                for i in 0..4 {
                    body.push_str(&format!(
                        "    let v{r}_{i} = tile[((idx * 4u + {o}u + lid.x) & 511u)];\n",
                        o = i * 128
                    ));
                }
                for i in 0..4 {
                    for e in 0..4 {
                        let k = i * 4 + e;
                        body.push_str(&format!(
                            "    c{k} = fma(c{k}, v{r}_{i}[{e}], b);\n"
                        ));
                    }
                }
            }
            16 => {
                // Sixteen scalar loads, addressed so that a warp instruction's
                // thirty-two lanes hit the thirty-two banks: word
                // `(idx*16 + i)*32 + L`, i.e. lane L always takes bank L.
                for i in 0..16 {
                    body.push_str(&format!(
                        "    let t{r}_{i} = tile[((idx * 16u + {u}u) * 32u + lid.x) & 1023u];\n",
                        u = i
                    ));
                }
                for i in 0..16 {
                    body.push_str(&format!(
                        "    c{i} = fma(c{i}, t{r}_{i}, b);\n"
                    ));
                }
            }
            8 => {
                // The A side of the real GEMM: one `vec4` per four rows at a
                // fixed k, and the address depends only on which half of the
                // warp the lane is in -- so a phase's eight lanes all want the
                // same 16 bytes and the read is a broadcast, no permutation
                // needed.  The B side stays eight scalar loads (16 words per
                // lane per iteration either way).
                for i in 0..2 {
                    body.push_str(&format!(
                        "    let w{r}_{i} = tileA[((idx * 2u + (lid.x >> 4u) + {o}u) & 511u)];\n",
                        o = i * 128
                    ));
                }
                for i in 0..8 {
                    body.push_str(&format!(
                        "    let t{r}_{i} = tileB[((idx * 8u + {u}u) * 32u + (lid.x & 15u)) & 1023u];\n",
                        u = i
                    ));
                }
                for i in 0..4 {
                    body.push_str(&format!(
                        "    c{i} = fma(c{i}, w{r}_0[{i}], b);\n"
                    ));
                }
                for i in 0..4 {
                    body.push_str(&format!(
                        "    c{ii} = fma(c{ii}, w{r}_1[{i}], b);\n",
                        ii = 4 + i
                    ));
                }
                for i in 0..8 {
                    body.push_str(&format!("    c{i} = fma(c{i}, t{r}_{i}, b);\n"));
                }
            }
            99 => {
                // The original 8-load arm: two loads per round at a loop-counter
                // address, 16 words per lane per iteration.
                for i in 0..2 {
                    body.push_str(&format!(
                        "    let t{r}{i} = tile[((idx + {o}u) & 1023u)];\n",
                        o = r * 17 + i * 3
                    ));
                }
                body.push_str(&format!(
                    "    c0 = fma(c0, t{r}0, b);\n    c1 = fma(c1, t{r}1, b);\n"
                ));
            }
            _ => {}
        }
    }

    let tile_decl = match mix {
        0 => String::new(),
        4 => "var<workgroup> tile: array<vec4<f32>, 512>;\n".to_string(),
        8 => "var<workgroup> tileA: array<vec4<f32>, 512>;\nvar<workgroup> tileB: array<f32, 1024>;\n"
            .to_string(),
        _ => "var<workgroup> tile: array<f32, 1024>;\n".to_string(),
    };
    let fill_vec = |name: &str| -> String {
        (0..2)
            .map(|i| {
                format!(
                    "  {name}[lid.x + {o}u] = vec4<f32>(f32(lid.x) * 1e-6);\n",
                    o = i * 256
                )
            })
            .collect()
    };
    let fill = match mix {
        0 => String::new(),
        4 => fill_vec("tile"),
        8 => {
            fill_vec("tileA")
                + &(0..4)
                    .map(|i| {
                        format!("  tileB[lid.x + {o}u] = f32(lid.x) * 1e-6;\n", o = i * 256)
                    })
                    .collect::<String>()
        }
        _ => (0..4)
            .map(|i| format!("  tile[lid.x + {o}u] = f32(lid.x) * 1e-6;\n", o = i * 256))
            .collect(),
    };

    format!(
        "@group(0) @binding(0) var<storage, read_write> out: array<f32>;
{tile_decl}
const ITERS: u32 = {ITERS}u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {{
{fill}{decl}{init}
  let a = 1.0000001 + f32(wid.x) * 1e-12;
  let b = 1.0000002 + f32(lid.x) * 1e-12;
  var idx = 0u;
  loop {{
    if (idx >= ITERS) {{ break; }}
{body}    idx = idx + 1u;
  }}
  var s = 0.0;
{sink}  if (s == 1.2345e30) {{ out[wid.x * 256u + lid.x] = s; }}
}}
"
    )
}

async fn run(prefer: Option<&str>, mix: usize) -> anyhow::Result<()> {
    let gpu = pollster::block_on(Gpu::new(prefer))?;
    println!("adapter : {}", gpu.describe());
    let device = &gpu.device;
    let queue = &gpu.queue;

    let src = shader(mix);
    let pipe = gpu.pipeline("fma", &src, "main", None)?;
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 65536 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: out.as_entire_binding() }],
    });

    // The inner body issues 64 FMAs plus one more per loaded value accumulated
    // (the loads have to feed something).
    let fma_per_iter = match mix {
        4 | 8 => 64 + 16,
        16 => 64 + 16,
        99 => 64 + 8,
        _ => 64,
    };
    let fmas_per_lane = ITERS as f64 * fma_per_iter as f64;
    let arm = match mix {
        4 => "4 vec4 shared loads + 80 FMA per iteration",
        8 => "2 vec4 (A) + 8 scalar (B) shared loads + 80 FMA per iteration",
        16 => "16 scalar shared loads + 80 FMA per iteration",
        99 => "the original 8-load arm (address depends on c0 -- see below)",
        _ => "pure FFMA, no memory, 16 chains x 4 rounds",
    };
    println!("\narm: {arm}");
    println!("{:>7}  {:>9}  {:>10}", "grid", "warps/SM", "TFLOP/s");
    let mut best = 0f64;
    for gx in [15u32, 30, 60, 120, 240, 480, 960, 1920, 3840] {
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
            queue.submit([enc.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely())?;
            let t0 = Instant::now();
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&pipe);
                cp.set_bind_group(0, &bg, &[]);
                for _ in 0..reps {
                    cp.dispatch_workgroups(gx, 1, 1);
                }
            }
            queue.submit([enc.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely())?;
            Ok(t0.elapsed().as_secs_f64() * 1e3 / reps as f64)
        };
        run(2)?;
        let ms = run(10)?;
        let flop = gx as f64 * 256.0 * fmas_per_lane * 2.0;
        let tflops = flop / (ms / 1e3) / 1e12;
        best = best.max(tflops);
        println!("{gx:>7}  {:>9.1}  {tflops:>10.2}", (gx as f64 * 8.0) / 15.0);
    }
    println!("\npeak measured: {best:.2} TFLOP/s  (spec: 1920 lanes x 2 x 1.911 GHz = 7.34)");
    Ok(())
}
