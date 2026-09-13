//! Decode-path validation + benchmark harness.
//!
//! Two jobs:
//!
//! 1. **Alignment** — start the wgpu decoder from exactly the KV-cache state the
//!    CUDA backend had after prefill, run the same number of decode steps, and
//!    compare the token ids id-by-id.  Per-op intermediates dumped for step 0 and
//!    the logits of the first few steps localise any divergence.
//! 2. **Speed** — measure ms/token and effective weight bandwidth, to compare
//!    against the CUDA baseline (6.58 ms/token, 289 GB/s on P104-100).
//!
//! ```text
//! cargo run --release --bin decode_check -- --tag q06_15s_en [--adapter nvidia]
//! ```
//!
//! Everything is read from paths given on the command line, so the crate stays
//! portable: `--golden <dir>` defaults to `golden/<tag>`, and the model directory
//! comes from the golden's `meta.json`.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};

use qwen3_asr_wgpu::decoder::{TextConfig, WgpuTextDecoder};
use qwen3_asr_wgpu::golden::{diff_f16, Golden};
use qwen3_asr_wgpu::gpu::Gpu;
use qwen3_asr_wgpu::mrope;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn f16_bytes(v: &[half::f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let tag = arg(&args, "--tag").unwrap_or_else(|| "q06_15s_en".to_string());
    let golden_dir = arg(&args, "--golden")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("golden").join(&tag));
    let adapter = arg(&args, "--adapter");
    let bench_steps: usize = arg(&args, "--bench")
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let no_bench = args.iter().any(|a| a == "--no-bench");

    println!("=== qwen3-asr wgpu decode check ===");
    let g = Golden::load(&golden_dir)?;
    g.check_integrity()?;
    let m = &g.meta;
    println!(
        "golden {}: model={} wav={} seq_len={} decode={} vocab={} layers={}",
        m.tag, m.model_dir, m.wav, m.seq_len, m.n_decode, m.vocab_size, m.num_hidden_layers
    );

    // ── device ───────────────────────────────────────────────────────────
    let gpu = pollster::block_on(Gpu::new(adapter.as_deref()))?;
    println!("adapter: {}", gpu.describe());

    // ── MRoPE parity (computed here, not copied from the golden) ─────────
    let (cv, sv) = mrope::compute_mrope_cos_sin(
        &mrope::text_positions(m.max_seq),
        m.head_dim,
        m.rope_theta,
        &m.mrope_section,
        m.mrope_interleaved,
    );
    let cos_mine: Vec<half::f16> = cv.iter().map(|&x| half::f16::from_f32(x)).collect();
    let sin_mine: Vec<half::f16> = sv.iter().map(|&x| half::f16::from_f32(x)).collect();
    println!("\n-- MRoPE tables vs golden --");
    println!("{}", diff_f16(&g.cos, &cos_mine).describe("mrope cos"));
    println!("{}", diff_f16(&g.sin, &sin_mine).describe("mrope sin"));

    // ── decoder ──────────────────────────────────────────────────────────
    let cfg = TextConfig {
        vocab_size: m.vocab_size,
        hidden_size: m.hidden_size,
        intermediate_size: m.intermediate_size,
        num_hidden_layers: m.num_hidden_layers,
        num_attention_heads: m.num_attention_heads,
        num_key_value_heads: m.num_key_value_heads,
        head_dim: m.head_dim,
        rms_norm_eps: m.rms_norm_eps as f32,
    };
    let t_load = Instant::now();
    let model_dir = PathBuf::from(&m.model_dir);
    let mut dec = WgpuTextDecoder::load(
        gpu,
        &model_dir,
        "thinker.model",
        cfg.clone(),
        m.max_seq,
        m.max_seq,
    )
    .context("load wgpu decoder")?;
    println!(
        "\ndecoder loaded in {:.1}s ({} layers, {} dispatches/step, {:.0} MB weights, {:.0} MB KV)",
        t_load.elapsed().as_secs_f64(),
        m.num_hidden_layers,
        dec.dispatches_per_step(m.seq_len + 1),
        dec.step_weight_bytes() as f64 / 1e6,
        (m.num_hidden_layers * m.num_key_value_heads * m.max_seq * m.head_dim * 2 * 2) as f64 / 1e6,
    );

    // Use the golden tables so a table mismatch cannot masquerade as a kernel bug.
    dec.set_rope_tables(&g.cos, &g.sin);

    // ── seed the KV cache with the post-prefill prefix ───────────────────
    let (nkvh, hd, seq_len, max_seq) = (m.num_key_value_heads, m.head_dim, m.seq_len, m.max_seq);
    let per_layer = nkvh * seq_len * hd;
    for l in 0..m.num_hidden_layers {
        for (cache, all) in [
            (dec.k_cache(l), &g.k_cache),
            (dec.v_cache(l), &g.v_cache),
        ] {
            let layer = g.layer_slice(all, l);
            for h in 0..nkvh {
                let chunk = &layer[h * seq_len * hd..(h + 1) * seq_len * hd];
                let off = (h * max_seq * hd * 2) as u64;
                dec.gpu.queue.write_buffer(cache, off, &f16_bytes(chunk));
            }
        }
    }
    let _ = per_layer;
    dec.pos = seq_len;

    // ── decode loop ──────────────────────────────────────────────────────
    let expected = g.expected_sequence();
    if expected[0] != m.first_token {
        bail!(
            "golden sequence starts with {} but meta.first_token is {}",
            expected[0],
            m.first_token
        );
    }
    dec.set_input_token(expected[0]);

    println!("\n-- decode alignment ({} steps + EOS) --", m.n_decode);
    let mut got: Vec<i32> = vec![expected[0]];
    let mut per_step_ms: Vec<f64> = Vec::new();
    let mut op_diffs: Vec<String> = Vec::new();
    let mut logit_diffs: Vec<String> = Vec::new();
    let mut first_mismatch: Option<usize> = None;
    let mut flip_logits: Option<Vec<half::f16>> = None;

    for k in 0..m.n_decode {
        let t0 = Instant::now();
        let tok = dec.step().context("decode step")?;
        per_step_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        got.push(tok);
        if *got.last().unwrap() != expected[k + 1] && first_mismatch.is_none() {
            first_mismatch = Some(k + 1);
            // Capture the logits at the flip step before the next step overwrites
            // them — the near-tie diagnostic needs both engines' values there.
            flip_logits = Some(dec.read_f16(&dec.scratch.logits, m.vocab_size)?);
        }

        if k == 0 {
            let hs = m.hidden_size;
            let checks: [(&str, &wgpu::Buffer, usize); 9] = [
                ("norm1", &dec.scratch.norm1, hs),
                ("qkv", &dec.scratch.qkv, m.num_attention_heads * hd + 2 * nkvh * hd),
                ("q_out", &dec.scratch.q_out, m.num_attention_heads * hd),
                ("attn_out", &dec.scratch.attn_out, m.num_attention_heads * hd),
                ("norm2", &dec.scratch.norm2, hs),
                ("gate_up", &dec.scratch.gate_up, 2 * m.intermediate_size),
                ("activated", &dec.scratch.activated, m.intermediate_size),
                ("h_out", &dec.scratch.h, hs),
                ("final_norm", &dec.scratch.final_norm, hs),
            ];
            for (name, buf, n) in checks {
                if let Some(refv) = g.step0.get(name) {
                    let ours = dec.read_f16(buf, n)?;
                    op_diffs.push(diff_f16(refv, &ours).describe(&format!("step0 {name}")));
                }
            }
        }
        if k < m.n_dumped_step_logits {
            let ours = dec.read_f16(&dec.scratch.logits, m.vocab_size)?;
            let refv = &g.step_logits[k * m.vocab_size..(k + 1) * m.vocab_size];
            logit_diffs.push(diff_f16(refv, &ours).describe(&format!("step{k} logits")));
        }
    }

    println!("\n-- step-0 intermediates (CUDA vs wgpu) --");
    for d in &op_diffs {
        println!("  {d}");
    }
    println!("\n-- logits parity for the first {} steps --", logit_diffs.len());
    for d in &logit_diffs {
        println!("  {d}");
    }

    println!("\n-- token ids --");
    let mismatches = (0..expected.len()).filter(|&i| got[i] != expected[i]).count();
    println!(
        "  sequence: {} ids, {} mismatches{}",
        expected.len(),
        mismatches,
        match first_mismatch {
            Some(i) => format!(" (first at index {i}: ref={} got={})", expected[i], got[i]),
            None => " — IDENTICAL".to_string(),
        }
    );
    if mismatches > 0 {
        let lo = first_mismatch.unwrap().saturating_sub(2);
        let hi = (lo + 8).min(expected.len());
        println!("  ref [{lo}..{hi}): {:?}", &expected[lo..hi]);
        println!("  got [{lo}..{hi}): {:?}", &got[lo..hi]);
        // Near-tie diagnostic: if the flipped token's logits are within noise of
        // the reference winner, the divergence is amplified last-ulp drift, not
        // a discrete port bug.  Requires the golden to have dumped the failing
        // step's logits (QASR_DUMP_STEPS).
        let k = first_mismatch.unwrap() - 1;
        if let (Some(ours), true) = (&flip_logits, k < m.n_dumped_step_logits) {
            let mut idx: Vec<usize> = (0..m.vocab_size).collect();
            idx.sort_by(|&a, &b| ours[b].to_f32().total_cmp(&ours[a].to_f32()));
            println!("  step {k} top-4 (by our logits):");
            for t in idx.iter().take(4) {
                let r = g.step_logits[k * m.vocab_size + *t].to_f32();
                let o = ours[*t].to_f32();
                println!("    tok {:>7}  ref={r:.4}  got={o:.4}", t);
            }
        }
    }
    println!(
        "  first 12 ref {:?}\n  first 12 got {:?}",
        &expected[..12.min(expected.len())],
        &got[..12.min(got.len())]
    );

    let mean_ms = per_step_ms.iter().sum::<f64>() / per_step_ms.len() as f64;
    println!(
        "\n  host-synced step: {:.2} ms/token (mean of {}, incl. D2H + poll)",
        mean_ms,
        per_step_ms.len()
    );

    // ── pure GPU throughput ──────────────────────────────────────────────
    if !no_bench {
        let ms = dec.bench_steps(bench_steps, 3)?;
        let bytes = dec.step_weight_bytes() as f64;
        println!(
            "  GPU-bound step  : {:.2} ms/token -> {:.1} GB/s effective weight bandwidth ({} steps, no per-step sync)",
            ms,
            bytes / 1e9 / (ms / 1000.0),
            bench_steps
        );
        println!(
            "  CUDA reference  : 6.58 ms/token, 289 GB/s  -> wgpu is {:.2}x on the GEMV-bound step",
            6.58 / ms
        );
    }

    dec.gpu
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("final poll")?;

    if mismatches == 0 {
        println!("\nRESULT: decode chain matches the CUDA token sequence exactly.");
    } else {
        println!("\nRESULT: {mismatches} token mismatches.");
    }
    Ok(())
}
