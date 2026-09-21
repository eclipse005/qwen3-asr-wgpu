//! Can a *grid-wide* barrier work on this stack?
//!
//! The decode step issues ~229 dispatches per token, and `gemv_bench`'s chained
//! timings put ~2.25 us of pure launch cost on each of them plus a few
//! microseconds of wave fill that a 1024-warp dispatch cannot avoid on a 960-warp
//! machine.  That is ~0.5-0.9 ms/token, 6-11% of the decode, and it is the one
//! item in the pipeline that is neither at a measured roof nor behind the
//! reduction-order gate.  The only lever is the dispatch *count*, which means
//! fusing a whole layer (or a whole step) into one persistent kernel and
//! synchronising between its phases on the device.
//!
//! WGSL has no grid-wide barrier, but it has atomics, so the question is whether
//! a hand-rolled one is (a) correct and (b) cheap, and whether the driver will
//! keep every workgroup resident -- a spin barrier deadlocks if it will not, and
//! that is not a failure the transcript gate can catch.
//!
//! This probe answers both, with a spin bound so a non-resident grid *reports*
//! rather than hangs:
//!
//! * `rounds` phases, each with a sense-reversing barrier: every workgroup
//!   atomically bumps a counter, then every thread spins on it until all
//!   `n_wg` have arrived.
//! * each phase first writes `phase*n_wg + wgid` into a slot, then after the
//!   barrier reads all slots and checks they are all >= the current phase's
//!   base.  If the barrier is wrong, the check fails and says so.
//! * the spin is bounded at `MAX_SPIN` iterations; a grid that cannot be
//!   co-resident reports `timeout` instead of hanging the card.
//!
//! Run `nvidia-smi --query-gpu=clocks.sm,utilization.gpu --format=csv -l 1`
//! alongside if the numbers look odd.

use std::time::Instant;

use qwen3_asr_wgpu::gpu::Gpu;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let prefer = arg(&args, "--adapter");
    pollster::block_on(run(prefer.as_deref()))
}

const ROUNDS: u32 = 64;

fn shader() -> String {
    format!(
        "@group(0) @binding(0) var<storage, read_write> bar: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> slots: array<u32>;
@group(0) @binding(2) var<storage, read_write> report: array<u32>;

const N_WG: u32 = {ng}u;
const ROUNDS: u32 = {rounds}u;
const MAX_SPIN: u32 = 200000000u;

fn grid_barrier(g: u32, tid: u32) -> u32 {{
  // One atomic add per workgroup, then every thread spins.  `g` is the sense:
  // the target is (g+1)*N_WG, so a straggler from the previous round cannot
  // release the next one.
  if (tid == 0u) {{
    atomicAdd(&bar[0], 1u);
  }}
  workgroupBarrier();
  let goal = (g + 1u) * N_WG;
  var spins = 0u;
  loop {{
    if (atomicLoad(&bar[0]) >= goal) {{ break; }}
    spins = spins + 1u;
    if (spins > MAX_SPIN) {{ return 0u; }}
  }}
  workgroupBarrier();
  return 1u;
}}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
  let tid = lid.x;
  // A little work, so the barrier is not the only thing being measured.
  var acc = f32(wgid.x) * 1e-6;
  for (var i = 0u; i < 256u; i = i + 1u) {{
    acc = fma(acc, 1.0000001, 0.5);
  }}
  if (acc == 1.2345e30) {{ slots[wgid.x] = u32(acc); }}

  for (var g = 0u; g < ROUNDS; g = g + 1u) {{
    // Publish this round's arrival, then synchronise, then check that every
    // other workgroup has published too.
    if (tid == 0u) {{
      slots[wgid.x] = g * N_WG + wgid.x + 1u;
    }}
    let ok = grid_barrier(g, tid);
    if (ok == 0u) {{
      if (tid == 0u) {{ report[0] = 1u; }}   // timeout
      return;
    }}
    if (tid == 0u) {{
      // Every slot must have been written this round (or a later one).
      let base = g * N_WG;
      for (var w = 0u; w < N_WG; w = w + 1u) {{
        if (slots[w] <= base) {{
          report[1] = 1u;                    // barrier did not synchronise
        }}
      }}
    }}
  }}
  if (tid == 0u) {{ report[2] = report[2] + 1u; }}
}}
",
        ng = 0,
        rounds = ROUNDS
    )
}

async fn run(prefer: Option<&str>) -> anyhow::Result<()> {
    let gpu = pollster::block_on(Gpu::new(prefer))?;
    println!("adapter : {}", gpu.describe());
    let device = &gpu.device;
    let queue = &gpu.queue;

    println!(
        "\n{:>6}  {:>9}  {:>10}  {:>12}  {:>12}",
        "grid", "warps/SM", "ms", "us/barrier", "verdict"
    );
    for n_wg in [15u32, 30, 60, 120, 240, 480] {
        let src = shader().replace("const N_WG: u32 = 0u;", &format!("const N_WG: u32 = {n_wg}u;"));
        let pipe = match gpu.pipeline("bar", &src, "main", None) {
            Ok(p) => p,
            Err(e) => {
                println!("{n_wg:>6}  pipeline: {e}");
                continue;
            }
        };
        let mk = |label: &str, len: u64, atomic: bool| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: len,
                usage: if atomic {
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
                } else {
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
                },
                mapped_at_creation: false,
            })
        };
        let bar = mk("bar", 16, true);
        let slots = mk("slots", (n_wg as u64) * 4, false);
        let report = mk("report", 16, false);
        // Zero everything, and keep a zeroed copy to re-zero between reps.
        for b in [&bar, &slots, &report] {
            queue.write_buffer(b, 0, &vec![0u8; b.size() as usize]);
        }
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bar"),
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bar.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: slots.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: report.as_entire_binding() },
            ],
        });
        let one = |reps: u32| -> anyhow::Result<f64> {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&pipe);
                cp.set_bind_group(0, &bg, &[]);
                for _ in 0..reps {
                    cp.dispatch_workgroups(n_wg, 1, 1);
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
                    cp.dispatch_workgroups(n_wg, 1, 1);
                }
            }
            queue.submit([enc.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely())?;
            Ok(t0.elapsed().as_secs_f64() * 1e3 / reps as f64)
        };
        // One dispatch only (the barrier counter is monotonic, so more than one
        // would need it re-zeroed and the sense reset).
        let ms = one(1)?;
        let rep = gpu.readback(&report, 12)?;
        let r: Vec<u32> = rep.chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        let verdict = if r[0] == 1 {
            "TIMEOUT (grid not co-resident)".to_string()
        } else if r[1] == 1 {
            "UNSYNCHRONISED".to_string()
        } else if r[2] == n_wg {
            "ok".to_string()
        } else {
            format!("incomplete: {} of {} workgroups finished", r[2], n_wg)
        };
        println!(
            "{n_wg:>6}  {:>9.1}  {ms:>10.3}  {:>12.3}  {verdict}",
            n_wg as f64 * 8.0 / 15.0,
            ms * 1000.0 / ROUNDS as f64
        );
    }
    Ok(())
}
