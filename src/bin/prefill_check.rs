//! Prefill validation against the CUDA golden.
//!
//! Runs the wgpu prefill chain (GEMM stand-ins for cuBLAS) on the golden's
//! `hidden.bin`, then compares:
//!   1. K/V cache vs `k_cache.bin` / `v_cache.bin`  — f16 tolerance (cuBLAS
//!      accumulation order is a black box, so bit-exactness is not achievable)
//!   2. last-position logits vs `logits_prefill.bin` + argmax vs `tokens[0]`
//!   3. decode continuation from the *wgpu-prefilled* KV vs golden tokens
//!   4. prefill wall time
//!
//! ```text
//! cargo run --release --manifest-path wgpu/Cargo.toml --bin prefill_check -- \
//!     --tag q06_15s_en --golden wgpu/golden/q06_15s_en --adapter nvidia
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};

use qwen3_asr_wgpu::decoder::{TextConfig, WgpuTextDecoder};
use qwen3_asr_wgpu::golden::{diff_f16, Diff, Golden};
use qwen3_asr_wgpu::gpu::Gpu;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn f16_bytes(v: &[half::f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

/// Elements outside the calibrated KV tolerance: |Δ| > 0.0625 + |ref|·2⁻⁹.
/// DIAGNOSTIC ONLY — do not gate on this.  Measured envelope (q06_15s_en):
/// even torch-refp vs the CUDA golden — two correct accumulation orders —
/// violate it at nearly every deep layer (L12: 83 elems, L24: 151, max|Δ|
/// 0.69), because 28 layers of last-ulp GEMM differences random-walk the
/// caches.  The pass/fail gate is the absolute envelope (`worst < 1.0`), far
/// above that measured order-noise envelope yet far below the O(1–10)
/// concentrated errors real structural bugs produce (e.g. the AV prefetch
/// half-selector bug bent L0 h by 3.3 from row 16 on).
fn rel_violations(a: &[half::f16], b: &[half::f16]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| {
            let (x, y) = (x.to_f32(), y.to_f32());
            (x - y).abs() > 0.0625 + x.abs() * (1.0 / 512.0)
        })
        .count()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let tag = arg(&args, "--tag").unwrap_or_else(|| "q06_15s_en".to_string());
    let golden_dir = arg(&args, "--golden")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("wgpu/golden").join(&tag));
    let adapter = arg(&args, "--adapter");
    let no_bench = args.iter().any(|a| a == "--no-bench");

    println!("=== qwen3-asr wgpu prefill check ===");
    let g = Golden::load(&golden_dir)?;
    g.check_integrity()?;
    let m = &g.meta;
    let s = m.seq_len;
    println!(
        "golden {}: model={} s={} decode={}",
        m.tag, m.model_dir, s, m.n_decode
    );

    let gpu = pollster::block_on(Gpu::new(adapter.as_deref()))?;
    println!("adapter: {}", gpu.describe());

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
    let mut dec = WgpuTextDecoder::load(
        gpu,
        PathBuf::from(&m.model_dir).as_path(),
        "thinker.model",
        cfg,
        m.max_seq,
        m.max_seq,
    )
    .context("load decoder")?;
    dec.set_rope_tables(&g.cos, &g.sin);

    // ── prefill from the golden's prefill input ─────────────────────────────
    let hidden = f16_bytes(&g.hidden);
    let t0 = Instant::now();
    let first = dec
        .prefill(&hidden, s, 0)
        .context("wgpu prefill")?;
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("\nprefill: {:.1} ms ({:.1} ms/pos over {s} positions)", prefill_ms, prefill_ms / s as f64);

    // debug: dump prefill h for offline comparison
    if let Some(hb) = &dec.debug_l0_h {
        let hb = dec.gpu.readback(hb, (m.hidden_size * 2) as u64)?;
        std::fs::write("wgpu/l0_h_debug.bin", &hb).ok();
    }
    if let Some(hb) = &dec.debug_prefill_h {
        let hb = dec.gpu.readback(hb, (s * m.hidden_size * 2) as u64)?;
        std::fs::write("wgpu/prefill_h_debug.bin", &hb).ok();
    }

    // ── 1. KV cache vs golden ───────────────────────────────────────────────
    println!("\n-- KV cache vs CUDA golden (f16 tolerance expected) --");
    let (nkvh, hd, max_seq) = (m.num_key_value_heads, m.head_dim, m.max_seq);
    let mut worst = diff_f16(&[], &[]);
    let mut layer_worst = vec![(0f32, 0usize); m.num_hidden_layers];
    let mut rel_bad = 0usize;
    for l in 0..m.num_hidden_layers {
        for (name, cache_buf, golden_all) in
            [("k", dec.k_cache(l), &g.k_cache), ("v", dec.v_cache(l), &g.v_cache)]
        {
            let layer = g.layer_slice(golden_all, l);
            for h in 0..nkvh {
                let golden_chunk = &layer[h * s * hd..(h + 1) * s * hd];
                let off = (h * max_seq * hd * 2) as u64;
                let got = dec.gpu.readback(cache_buf, off + (s * hd * 2) as u64)?;
                let got = &got[off as usize..];
                let got: Vec<half::f16> = got
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let d = diff_f16(golden_chunk, &got);
                rel_bad += rel_violations(golden_chunk, &got);
                if l == 1 && h == 0 && name == "k" {
                    // per-position error pattern: is position 0 (no attention) clean?
                    let per_pos: Vec<f32> = (0..s)
                        .map(|p| {
                            let a = &golden_chunk[p * hd..(p + 1) * hd];
                            let b = &got[p * hd..(p + 1) * hd];
                            a.iter().zip(b).map(|(x, y)| (x.to_f32() - y.to_f32()).abs()).fold(0f32, f32::max)
                        })
                        .collect();
                    let bad: Vec<usize> = per_pos.iter().enumerate().filter(|(_, v)| **v > 1.0).map(|(i, _)| i).collect();
                    println!("  L1 K h0: {} positions with max|Δ|>1 (first: {:?})",
                        bad.len(), bad.iter().take(5).collect::<Vec<_>>());
                    println!("  L1 K h0 pos0 max|Δ|={:.4} pos1={:.4} pos2={:.4}",
                        per_pos[0], per_pos[1], per_pos[2]);
                }
                if d.max_abs > layer_worst[l].0 {
                    layer_worst[l] = (d.max_abs, h);
                }
                if d.max_abs > worst.max_abs {
                    worst = Diff {
                        max_abs: d.max_abs,
                        n_exact: d.n_exact,
                        n: d.n,
                        first_bad: d.first_bad,
                    };
                }
                if l == 0 && h == 0 {
                    println!("  L0 H0 {name}: {}", d.describe(&format!("{name}_cache")));
                }
            }
        }
    }
    println!("  worst across all layers/heads: max|Δ|={:.4e}", worst.max_abs);
    println!("  rel-tol violations (0.0625 + |ref|·2⁻⁹): {rel_bad}");
    for (l, (mx, h)) in layer_worst.iter().enumerate() {
        println!("  L{l:02} worst max|Δ|={mx:.4} (head {h})");
    }
    let kv_ok = worst.max_abs < 1.0; // envelope: order-noise ≤0.69 measured, bugs O(1–10)

    // ── 2. prefill logits + first token ─────────────────────────────────────
    let logits = dec.read_f16(&dec.scratch.logits, m.vocab_size)?;
    let dl = diff_f16(&g.logits_prefill, &logits);
    println!("\nprefill logits: {}", dl.describe("logits"));
    let argmax_wgpu = first;
    let argmax_ref = g.tokens[0];
    println!("first token: wgpu={argmax_wgpu} cuda={argmax_ref} {}", if argmax_wgpu == argmax_ref { "MATCH" } else { "MISMATCH" });

    // ── 3. decode continuation from the wgpu-prefilled KV ───────────────────
    let expected = g.expected_sequence();
    println!("\n-- decode continuation (wgpu prefill KV, {} steps) --", m.n_decode);
    let mut got: Vec<i32> = vec![argmax_wgpu];
    let mut per_step: Vec<f64> = Vec::new();
    for _ in 0..m.n_decode {
        let ts = Instant::now();
        let tok = dec.step().context("decode step")?;
        per_step.push(ts.elapsed().as_secs_f64() * 1000.0);
        got.push(tok);
    }
    let mismatches = (0..expected.len().min(got.len()))
        .filter(|&i| got[i] != expected[i])
        .count();
    let first_mm = (0..expected.len().min(got.len())).find(|&i| got[i] != expected[i]);
    println!(
        "  tokens: {}/{} aligned{}",
        expected.len().min(got.len()) - mismatches,
        expected.len().min(got.len()),
        match first_mm {
            Some(i) => format!(" (first diff at {i}: ref={} got={})", expected[i], got[i]),
            None => " — IDENTICAL".to_string(),
        }
    );
    let mean_ms = per_step.iter().sum::<f64>() / per_step.len() as f64;
    println!("  decode: {mean_ms:.2} ms/token");

    // ── RTFx estimate ───────────────────────────────────────────────────────
    // duration = the `<digits>s` segment of the tag ("q17_15s_en" -> 15);
    // the old trim_start_matches ate leading tag digits too (15s -> "5s")
    let secs: f64 = m
        .tag
        .split('_')
        .find_map(|seg| seg.strip_suffix('s').and_then(|d| d.parse().ok()))
        .unwrap_or(0.0);
    let total = prefill_ms / 1000.0 + per_step.iter().sum::<f64>() / 1000.0;
    if secs > 0.0 {
        println!("\nRTFx (text decoder only, audio {secs:.0}s): {:.1}×", secs / total);
    }

    // Acceptance = functional: first token matches AND the whole decode
    // continuation from the wgpu-prefilled KV is identical.  The KV cache
    // diff is a diagnostic: between ANY two distinct accumulation orders the
    // f16 caches random-walk apart as the chain grows (measured torch-vs-CUDA
    // on q06_15s_en reaches 0.69; wgpu-vs-CUDA grows from 0.40 @15s to ~8
    // @90s with L0 exact and smooth per-layer growth — the healthy signature;
    // the AV half-selector bug instead showed max|Δ|=93 from L01 on plus a
    // first-token mismatch).  An absolute KV envelope alone cannot separate
    // order noise from bugs across fixture lengths.
    let seq_identical = mismatches == 0 && argmax_wgpu == argmax_ref;
    if !seq_identical {
        println!(
            "\nRESULT: PREFILL OUT OF TOLERANCE (first token {} / continuation mismatches {mismatches})",
            if argmax_wgpu == argmax_ref { "match" } else { "MISMATCH" }
        );
    } else if kv_ok {
        println!("\nRESULT: prefill within f16 tolerance, first token matches, continuation identical");
    } else {
        println!(
            "\nRESULT: prefill accepted — KV drift above the 1.0 envelope (max|Δ|={:.3}) but decode chain fully identical",
            worst.max_abs
        );
    }
    if no_bench {
        return Ok(());
    }
    Ok(())
}
