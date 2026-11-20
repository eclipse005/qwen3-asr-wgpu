//! Does one 128-workgroup dispatch overlap where eight 16-workgroup ones do not?
//!
//! The decode step's attention launches `nqh` workgroups per layer (16 at the
//! 1.7B shape).  At b=8 that is eight dispatches of 16 inside one compute pass,
//! and the measured slope (2.87 ms per extra slot) says the driver runs them one
//! after another.  If the same work is issued as a single dispatch with
//! `grid = (nqh, 1, 8)` the scheduler can hold all 128 workgroups at once.
//!
//! This probe only prices the scheduling question: every workgroup here computes
//! the *same* slot (the shader ignores `wgid.z`), so the numbers are garbage and
//! the timing is what is being read.  It exists to decide whether a batched
//! attention kernel is worth writing, before writing it.

use std::time::Instant;

use anyhow::Result;

use qwen3_asr_wgpu::decoder::{TextConfig, WgpuTextDecoder};
use qwen3_asr_wgpu::gpu::Gpu;

const SEQ: usize = 780;
const SLOTS: usize = 8;
const REPS: usize = 8;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = arg(&args, "--model").unwrap_or_else(|| "/workspace/Qwen3-ASR-1.7B-hf".into());
    let gpu = pollster::block_on(Gpu::new(arg(&args, "--adapter").as_deref()))?;
    println!("adapter: {}", gpu.describe());
    let cfg = TextConfig::from_model_dir(std::path::Path::new(&model))?;
    let mut d = WgpuTextDecoder::load(
        gpu.clone(),
        std::path::Path::new(&model),
        "thinker.model",
        cfg.clone(),
        16_384,
        16_384,
    )?;
    d.ensure_capacity(SEQ + 8);
    // KV must be allocated at `SEQ` slots; prefill a synthetic prompt to fill it.
    let hidden = vec![0u8; SEQ * cfg.hidden_size * 2];
    d.prefill(&hidden, SEQ, 0)?;
    d.pos = SEQ;
    d.debug_write_step_uniforms(SEQ);

    let nl = cfg.num_hidden_layers;
    let nqh = cfg.num_attention_heads as u32;
    let pipe = d.gqa512_pipe();

    let time = |grid: (u32, u32, u32), dispatches: usize| -> Result<f64> {
        let run = || -> Result<()> {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(pipe);
                for l in 0..dispatches {
                    cp.set_bind_group(0, d.gqa_bind(l % nl), &[]);
                    cp.dispatch_workgroups(grid.0, grid.1, grid.2);
                }
            }
            gpu.queue.submit([enc.finish()]);
            gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
            Ok(())
        };
        run()?;
        let t = Instant::now();
        for _ in 0..REPS {
            run()?;
        }
        Ok(t.elapsed().as_secs_f64() * 1e3 / REPS as f64)
    };

    // A: what the step does today -- SLOTS dispatches of 16 workgroups.
    let serial = time((nqh, 1, 1), nl * SLOTS)?;
    // A2: the same SLOTS dispatches but of one slot only (per-slot cost).
    let one = time((nqh, 1, 1), nl)?;
    // B: one dispatch per layer, SLOTS x the workgroups.
    let wide = time((nqh, 1, SLOTS as u32), nl)?;

    println!("\n{SEQ} positions, {nl} layers, {nqh} q heads, {SLOTS} slots, {REPS} reps");
    println!("  {SLOTS} separate dispatches/layer : {serial:7.2} ms  ({:.2} ms/slot)", serial / SLOTS as f64);
    println!("  1 dispatch/layer, grid.z={SLOTS}     : {wide:7.2} ms  ({:.2} ms/slot)", wide / SLOTS as f64);
    println!("  1 slot, 1 dispatch/layer         : {one:7.2} ms");
    println!("  overlap gain of the wide form    : {:.2}x", serial / wide);
    Ok(())
}
