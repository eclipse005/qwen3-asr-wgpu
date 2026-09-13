//! Per-op decode-step profile via GPU timestamp queries.
//!
//! Runs the real decode step op-by-op (one compute pass per op) with a
//! timestamp around every dispatch group, so the 8.x ms/token step can be
//! attributed to kernels — separating small-GEMV ramp cost from bandwidth.
//!
//! ```text
//! cargo run --release --manifest-path wgpu/Cargo.toml --bin step_profile -- \
//!     --tag q06_15s_en --golden wgpu/golden/q06_15s_en --adapter nvidia
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;

use qwen3_asr_wgpu::decoder::TextConfig;
use qwen3_asr_wgpu::decoder::WgpuTextDecoder;
use qwen3_asr_wgpu::golden::Golden;
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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let tag = arg(&args, "--tag").unwrap_or_else(|| "q06_15s_en".to_string());
    let golden_dir = arg(&args, "--golden")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("wgpu/golden").join(&tag));
    let adapter = arg(&args, "--adapter");

    let g = Golden::load(&golden_dir)?;
    let m = &g.meta;
    let gpu = pollster::block_on(Gpu::new(adapter.as_deref()))?;
    if !gpu.features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        anyhow::bail!("device lacks TIMESTAMP_QUERY; per-op profiling unavailable");
    }
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
    )?;
    dec.set_rope_tables(&g.cos, &g.sin);

    // seed KV prefix + token, same as decode_check
    let (nkvh, hd, seq_len, max_seq) = (m.num_key_value_heads, m.head_dim, m.seq_len, m.max_seq);
    for l in 0..m.num_hidden_layers {
        for (cache, all) in [(dec.k_cache(l), &g.k_cache), (dec.v_cache(l), &g.v_cache)] {
            let layer = g.layer_slice(all, l);
            for h in 0..nkvh {
                let chunk = &layer[h * seq_len * hd..(h + 1) * seq_len * hd];
                let off = (h * max_seq * hd * 2) as u64;
                dec.gpu.queue.write_buffer(cache, off, &f16_bytes(chunk));
            }
        }
    }
    dec.pos = seq_len;
    let expected = g.expected_sequence();
    dec.set_input_token(expected[0]);

    // timestamp scaffolding: 2 + 28*9 + 3 marks
    const OPS: [&str; 9] = [
        "rms1", "gemv_qkv", "extract", "gqa", "gemv_o", "rms2", "gemv_gu", "silu", "gemv_dp",
    ];
    let n_marks = 2 + m.num_hidden_layers * 9 + 1;
    let qs = dec.gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("step_profile"),
        ty: wgpu::QueryType::Timestamp,
        count: n_marks as u32,
    });
    let ts_bytes = (n_marks as u64) * 8;
    let ts_buf = {
        let mut b = dec.gpu.storage("ts_buf", ts_bytes);
        // resolve needs QUERY_RESOLVE on the target
        b = dec.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts_buf"),
            size: ts_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        });
        b
    };
    let ts_staging = dec.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ts_staging"),
        size: ts_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let pos = dec.pos;
    dec.debug_write_step_uniforms(pos);
    let mut names: Vec<&'static str> = Vec::with_capacity(n_marks);
    // --skip-micro: ablation (timing only, outputs are garbage) — drops
    // rms1/rms2/extract/silu to expose their real cost share.
    let skip_micro = args.iter().any(|a| a == "--skip-micro");
    let mut enc = dec.gpu.device.create_command_encoder(&Default::default());
    let mut idx = 0u32;
    // ONE compute pass — timestamps between dispatch groups measure the true
    // in-step cost with no pass-boundary overhead.
    let mut cp = enc.begin_compute_pass(&Default::default());
    cp.write_timestamp(&qs, idx);
    names.push("start");
    idx += 1;
    dec.debug_dispatch_embed(&mut cp);
    cp.write_timestamp(&qs, idx);
    names.push("embed");
    idx += 1;
    for l in 0..m.num_hidden_layers {
        for (op, name) in OPS.iter().enumerate() {
            if skip_micro && (op == 0 || op == 2 || op == 5 || op == 7) {
                // keep the mark aligned; zero-width delta
                cp.write_timestamp(&qs, idx);
                names.push("skipped");
                idx += 1;
                continue;
            }
            dec.debug_dispatch_op(&mut cp, l, pos, op);
            cp.write_timestamp(&qs, idx);
            names.push(if l == 0 || l == m.num_hidden_layers - 1 {
                Box::leak(format!("L{l}:{name}").into_boxed_str()) as &'static str
            } else {
                name
            });
            idx += 1;
        }
    }
    dec.debug_dispatch_tail(&mut cp);
    cp.write_timestamp(&qs, idx);
    names.push("tail_rms+lm+argmax");
    idx += 1;
    drop(cp);
    enc.resolve_query_set(&qs, 0..n_marks as u32, &ts_buf, 0);
    enc.copy_buffer_to_buffer(&ts_buf, 0, &ts_staging, 0, ts_bytes);
    dec.gpu.queue.submit([enc.finish()]);

    let mut data: Vec<u8> = Vec::new();
    let t0 = Instant::now();
    {
        let slice = ts_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        dec.gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        data = slice.get_mapped_range()?.to_vec();
        drop(slice);
        ts_staging.unmap();
    }
    let wall = t0.elapsed().as_secs_f64() * 1000.0;
    let marks: Vec<u64> = data
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();

    // aggregate per op name (GPU deltas between consecutive marks)
    use std::collections::BTreeMap;
    let mut agg: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
    let mut total = 0.0;
    for w in marks.windows(2) {
        let dt = (w[1] - w[0]) as f64 / 1e6; // ns -> ms (NVIDIA period = 1 ns)
        total += dt;
    }
    for (i, w) in marks.windows(2).enumerate() {
        let dt = (w[1] - w[0]) as f64 / 1e6;
        let e = agg.entry(names[i + 1]).or_insert((0.0, 0));
        e.0 += dt;
        e.1 += 1;
    }
    println!("step GPU time (op-by-op passes): {total:.3} ms (wall incl. readback {wall:.1} ms)");
    println!("{:<22} {:>10} {:>10} {:>12}", "op", "count", "ms total", "µs each");
    for (name, (ms, n)) in &agg {
        println!("{:<22} {:>10} {:>10.3} {:>12.2}", name, n, ms, ms * 1000.0 / *n as f64);
    }
    Ok(())
}
