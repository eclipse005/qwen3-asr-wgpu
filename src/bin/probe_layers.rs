//! Layer-by-layer probe of one decode step, dumped to disk.
//!
//! Port-time scaffolding for localising the first divergent op against the
//! torch reference (`tools/ref_decode_step.py`) and the CUDA golden.  Runs the
//! decode step one layer per submit, reading every scratch buffer back in
//! between, and writes raw f16 bins named like the golden dumps:
//!
//! ```text
//! <out>/L{ll}_norm1.bin ... L{ll}_h.bin, plus embed_h.bin, final_norm.bin,
//! logits.bin, token.txt
//! ```
//!
//! ```text
//! cargo run --release --manifest-path wgpu/Cargo.toml --bin probe_layers -- \
//!     --tag q06_15s_en --golden wgpu/golden/q06_15s_en --adapter nvidia
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result};

use qwen3_asr_wgpu::decoder::TextConfig;
use qwen3_asr_wgpu::decoder::WgpuTextDecoder;
use qwen3_asr_wgpu::golden::Golden;
use qwen3_asr_wgpu::gpu::Gpu;

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
        .unwrap_or_else(|| PathBuf::from("wgpu/golden").join(&tag));
    let adapter = arg(&args, "--adapter");
    let out_dir = arg(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| golden_dir.with_file_name(format!("{tag}_wgpu_probe")));
    std::fs::create_dir_all(&out_dir)?;

    let g = Golden::load(&golden_dir)?;
    let m = &g.meta;
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
    let model_dir = PathBuf::from(&m.model_dir);
    let mut dec = WgpuTextDecoder::load(gpu, &model_dir, "thinker.model", cfg, m.max_seq, m.max_seq)
        .context("load decoder")?;
    dec.set_rope_tables(&g.cos, &g.sin);

    // seed KV prefix (same as decode_check)
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

    let save = |name: &str, buf: &wgpu::Buffer, elems: usize| -> Result<()> {
        let v = dec.read_f16(buf, elems)?;
        std::fs::write(out_dir.join(format!("{name}.bin")), f16_bytes(&v))?;
        Ok(())
    };
    let hs = m.hidden_size;
    let qd = m.num_attention_heads * hd;
    let kvd = nkvh * hd;

    // --check-table: verify the embed table actually landed in VRAM.
    if args.iter().any(|a| a == "--check-table") {
        let bytes = dec.debug_embed_table_bytes(4096)?;
        println!("table[0..32] = {:02x?}", &bytes[..32]);
        return Ok(());
    }

    // --check-table content reference: expected bytes from row 0 of safetensors.
    // (compared offline with tools)

    // embed
    let pos = dec.pos;
    if !args.iter().any(|a| a == "--skip-embed") {
        let mut enc = dec.gpu.device.create_command_encoder(&Default::default());
        dec.debug_encode_embed(&mut enc);
        dec.gpu.queue.submit([enc.finish()]);
        save("embed_h", &dec.scratch.h, hs)?;
        println!("embed done");
    }

    // --op-by-op: layer 0's nine ops, one submit + readback each, to bisect a
    // device loss down to a single dispatch.  `--only N` runs just op N.
    if args.iter().any(|a| a == "--op-by-op") {
        const OPS: [&str; 10] = [
            "rms1", "gemv_qkv", "extract", "gqa", "gemv_o", "rms2", "gemv_gu", "silu", "gemv_dp",
            "embed-again",
        ];
        let only: Option<usize> = arg(&args, "--only").and_then(|s| s.parse().ok());
        for (op, name) in OPS.iter().enumerate() {
            if let Some(n) = only {
                if op != n {
                    continue;
                }
            }
            let mut enc = dec.gpu.device.create_command_encoder(&Default::default());
            if op == 9 {
                dec.debug_encode_embed(&mut enc);
            } else {
                dec.debug_encode_op(&mut enc, 0, pos, op);
            }
            dec.gpu.queue.submit([enc.finish()]);
            save("probe_last", &dec.scratch.h, hs)?;
            println!("op {op} ({name}) ok");
        }
        return Ok(());
    }

    // one layer per submit, full readback
    for l in 0..m.num_hidden_layers {
        let mut enc = dec.gpu.device.create_command_encoder(&Default::default());
        dec.debug_encode_layer(&mut enc, l, pos);
        dec.gpu.queue.submit([enc.finish()]);
        save(&format!("L{l:02}_norm1"), &dec.scratch.norm1, hs)?;
        save(&format!("L{l:02}_qkv"), &dec.scratch.qkv, qd + 2 * kvd)?;
        save(&format!("L{l:02}_q_out"), &dec.scratch.q_out, qd)?;
        save(&format!("L{l:02}_attn_out"), &dec.scratch.attn_out, qd)?;
        save(&format!("L{l:02}_norm2"), &dec.scratch.norm2, hs)?;
        save(&format!("L{l:02}_gate_up"), &dec.scratch.gate_up, 2 * m.intermediate_size)?;
        save(&format!("L{l:02}_activated"), &dec.scratch.activated, m.intermediate_size)?;
        save(&format!("L{l:02}_h"), &dec.scratch.h, hs)?;
    }
    println!("{} layers done", m.num_hidden_layers);

    // tail
    let mut enc = dec.gpu.device.create_command_encoder(&Default::default());
    dec.debug_encode_tail(&mut enc);
    dec.gpu.queue.submit([enc.finish()]);
    save("final_norm", &dec.scratch.final_norm, hs)?;
    save("logits", &dec.scratch.logits, m.vocab_size)?;
    let tok = {
        let v = dec.gpu.readback(&dec.scratch.token, 4)?;
        i32::from_le_bytes([v[0], v[1], v[2], v[3]])
    };
    std::fs::write(out_dir.join("token.txt"), tok.to_string())?;
    println!("tail done, argmax token = {tok}");
    Ok(())
}
