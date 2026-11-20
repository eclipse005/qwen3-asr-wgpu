//! End-to-end decode batch probe: N slots over one shared copy of the weights.
//!
//! Three arms, all seeded to the same state (a 700-token prefill of zero hidden
//! states), so a difference in the token streams can only come from the path:
//!
//! * `serial`  -- the shipped b=1 `step()`, one slot at a time.
//! * `interleaved` -- [`WgpuTextDecoder::step_batch`]: all slots' dispatches in
//!   one submit, weight read once per slot.
//! * `fused` -- [`BatchProjections`]: one dispatch per projection reads the
//!   weight once for every slot.
//!
//! Every arm must produce the identical token stream; the bench bails if not.
//!
//!   `cargo run --release --bin batch_step_bench -- --model /workspace/Qwen3-ASR-1.7B-hf`

use std::time::Instant;

use anyhow::Result;

use qwen3_asr_wgpu::decoder::{BatchProjections, TextConfig, WgpuTextDecoder};
use qwen3_asr_wgpu::gpu::Gpu;

const SEED_TOKENS: usize = 700;
const STEPS: usize = 24;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = arg(&args, "--model").unwrap_or_else(|| "/workspace/Qwen3-ASR-1.7B-hf".into());
    let steps: usize = arg(&args, "--steps").and_then(|s| s.parse().ok()).unwrap_or(STEPS);
    let batches: Vec<usize> = arg(&args, "--batch")
        .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8]);

    let gpu = pollster::block_on(Gpu::new(arg(&args, "--adapter").as_deref()))?;
    println!("adapter: {}", gpu.describe());
    let cfg = TextConfig::from_model_dir(std::path::Path::new(&model))?;
    let max_seq = 16_384;
    let t = Instant::now();
    let base = WgpuTextDecoder::load(
        gpu.clone(),
        std::path::Path::new(&model),
        "thinker.model",
        cfg.clone(),
        max_seq,
        max_seq,
    )?;
    println!(
        "decoder loaded in {:.2}s ({} layers, hidden {}, {} kv heads, vocab {})",
        t.elapsed().as_secs_f64(),
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_key_value_heads,
        cfg.vocab_size
    );

    let hidden = vec![0u8; SEED_TOKENS * cfg.hidden_size * 2];
    println!(
        "\n{:<4} {:>9} {:>9} {:>9} {:>8} {:>8} {:>8} {:>7}  {}",
        "b", "serial", "interlv", "fused", "s tok/s", "i tok/s", "f tok/s", "f speedup", "tokens"
    );
    println!("{}", "-".repeat(92));

    for n in batches {
        let mut serial: Vec<WgpuTextDecoder> = (0..n).map(|_| base.fork()).collect::<Result<_>>()?;
        let mut inter: Vec<WgpuTextDecoder> = (0..n).map(|_| base.fork()).collect::<Result<_>>()?;
        let mut fused: Vec<WgpuTextDecoder> = (0..n)
            .map(|_| base.fork_capped(SEED_TOKENS + steps + 8))
            .collect::<Result<_>>()?;
        let mut first = Vec::with_capacity(n);
        for d in serial.iter_mut().chain(inter.iter_mut()).chain(fused.iter_mut()) {
            d.ensure_capacity(SEED_TOKENS + steps + 8);
            first.push(d.prefill(&hidden, SEED_TOKENS, 0)?);
            d.pos = SEED_TOKENS;
        }

        // ── arm A: the shipped path, one slot at a time ──
        let mut tok_s: Vec<Vec<i32>> = vec![Vec::with_capacity(steps + 1); n];
        for i in 0..n {
            serial[i].set_input_token(first[i]);
            tok_s[i].push(first[i]);
        }
        let t0 = Instant::now();
        for _ in 0..steps {
            for i in 0..n {
                let tk = serial[i].step()?;
                tok_s[i].push(tk);
                serial[i].set_input_token(tk);
            }
        }
        let serial_s = t0.elapsed().as_secs_f64();

        // ── arm B: interleaved independent steps ──
        let mut tok_i: Vec<Vec<i32>> = vec![Vec::with_capacity(steps + 1); n];
        for i in 0..n {
            inter[i].set_input_token(first[i]);
            tok_i[i].push(first[i]);
        }
        let t1 = Instant::now();
        if n == 1 {
            for _ in 0..steps {
                let out = WgpuTextDecoder::step_batch(&mut inter)?;
                inter[0].set_input_token(out[0]);
                tok_i[0].push(out[0]);
            }
        } else {
            for _ in 0..steps {
                let out = WgpuTextDecoder::step_batch(&mut inter)?;
                for i in 0..n {
                    tok_i[i].push(out[i]);
                    inter[i].set_input_token(out[i]);
                }
            }
        }
        let inter_s = t1.elapsed().as_secs_f64();

        // ── arm C: fused projections (b >= 2 only) ──
        let mut tok_f: Vec<Vec<i32>> = vec![Vec::with_capacity(steps + 1); n];
        for i in 0..n {
            fused[i].set_input_token(first[i]);
            tok_f[i].push(first[i]);
        }
        let fused_s = if n >= 2 {
            let proj = BatchProjections::new(&fused)?;
            let t2 = Instant::now();
            for _ in 0..steps {
                let out = proj.step(&mut fused)?;
                for i in 0..n {
                    tok_f[i].push(out[i]);
                    fused[i].set_input_token(out[i]);
                }
            }
            Some(t2.elapsed().as_secs_f64())
        } else {
            None
        };

        let mut bad = Vec::new();
        if tok_i != tok_s {
            bad.push("interleaved");
        }
        if let Some(_) = fused_s {
            if tok_f != tok_s {
                bad.push("fused");
            }
        }
        let tokens = if bad.is_empty() { "IDENTICAL".to_string() } else { format!("DIFF: {}", bad.join(",")) };
        let fms = fused_s.map(|s| s * 1e3);
        println!(
            "{n:<4} {:>9.1} {:>9.1} {:>9} {:>8.1} {:>8.1} {:>8} {:>6}  {tokens}",
            serial_s * 1e3,
            inter_s * 1e3,
            fms.map(|m| format!("{m:.1}")).unwrap_or_else(|| "-".into()),
            steps as f64 * n as f64 / serial_s,
            steps as f64 * n as f64 / inter_s,
            fused_s
                .map(|s| format!("{:.1}", steps as f64 * n as f64 / s))
                .unwrap_or_else(|| "-".into()),
            fused_s
                .map(|s| format!("{:.2}x", serial_s / s))
                .unwrap_or_else(|| "-".into()),
        );
        if !bad.is_empty() {
            anyhow::bail!("b={n}: token stream differs ({tokens})");
        }
    }
    Ok(())
}
