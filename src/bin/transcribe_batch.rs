//! Transcribe a list of segment WAVs, `--batch` decoded together.
//!
//! `--batch 1` is the shipped path.  `--batch N` decodes N segments at once
//! segments and decodes them in lockstep: every projection reads its weights
//! once for the whole batch.  `--compare` runs the same files twice (b=1, then
//! b=N) and diffs the transcripts, which is the only check that matters -- a
//! batch that changes a character is a bug, not a trade-off.
//!
//!   `cargo run --release --bin transcribe_batch -- --model DIR --batch 8 \
//!        --wav segs/a.wav segs/b.wav ... --out /tmp/batch8.txt`

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;

use qwen3_asr_wgpu::{AsrInference, DeviceSelector, TranscribeOptions};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = PathBuf::from(
        arg(&args, "--model").unwrap_or_else(|| "/workspace/Qwen3-ASR-1.7B-hf".into()),
    );
    let batch: usize = arg(&args, "--batch").and_then(|v| v.parse().ok()).unwrap_or(1);
    // Same default as `transcribe` and `TranscribeOptions::default()`: the cap
    // must not be smaller than the shipped path's, or a batched run would
    // truncate transcripts the single-file path keeps.  It also sizes each
    // slot's KV (`cap = prompt + max_new + 8`), so a caller running many slots
    // on a small card lowers it explicitly.
    let max_new: usize = arg(&args, "--max-new")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| TranscribeOptions::default().max_new_tokens);
    let compare = args.iter().any(|a| a == "--compare");
    let timings = args.iter().any(|a| a == "--timings");
    let out_path = arg(&args, "--out").map(PathBuf::from);

    let wavs: Vec<PathBuf> = match args.iter().position(|a| a == "--wav") {
        Some(i) => args[i + 1..]
            .iter()
            .take_while(|a| !a.starts_with("--"))
            .map(PathBuf::from)
            .collect(),
        None => anyhow::bail!("--wav <file...> is required"),
    };
    anyhow::ensure!(!wavs.is_empty(), "no wavs");

    let t = Instant::now();
    let asr = AsrInference::load_on(&model, DeviceSelector::parse("vulkan")?)?;
    println!(
        "loaded in {:.2}s on {}",
        t.elapsed().as_secs_f64(),
        asr.device_description()
    );

    let opts = TranscribeOptions::default().with_max_new_tokens(max_new);
    let run = |b: usize| -> Result<(Vec<String>, f64, f64, Option<qwen3_asr_wgpu::inference::BatchTimings>)> {
        // Audio seconds, from the WAV headers, so RTFx is over real audio.
        let mut audio = 0.0f64;
        for w in &wavs {
            let r = hound::WavReader::open(w)?;
            audio += r.duration() as f64 / r.spec().sample_rate as f64;
        }
        let t = Instant::now();
        let (rs, ph) = asr.transcribe_wavs_batch_timed(&wavs, &opts, b)?;
        let secs = t.elapsed().as_secs_f64();
        Ok((
            rs.into_iter().map(|r| r.text).collect(),
            audio,
            secs,
            timings.then_some(ph),
        ))
    };

    let (texts, audio, secs, phases) = run(batch)?;
    if let Some(p) = phases {
        // The four phases are sequential: mel and the encoder are still per
        // segment, the decode half is what `--batch` batches.
        println!(
            "phases  mel={:.2}s encoder={:.2}s prefill={:.2}s decode={:.2}s",
            p.mel.as_secs_f64(),
            p.encoder.as_secs_f64(),
            p.prefill.as_secs_f64(),
            p.decode.as_secs_f64()
        );
    }
    println!(
        "batch={batch} files={} audio={audio:.1}s elapsed={secs:.2}s RTFx={:.2}",
        wavs.len(),
        audio / secs
    );
    for (w, t) in wavs.iter().zip(texts.iter()) {
        println!(
            "  {} : {}",
            w.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
            t.chars().take(40).collect::<String>()
        );
    }
    if let Some(p) = &out_path {
        std::fs::write(p, texts.join("\n"))?;
    }

    if compare {
        let (base, _, base_secs, _) = run(1)?;
        println!(
            "batch=1 files={} audio={audio:.1}s elapsed={base_secs:.2}s RTFx={:.2}",
            wavs.len(),
            audio / base_secs
        );
        let diffs = texts.iter().zip(base.iter()).filter(|(a, b)| a != b).count();
        let mut chars = 0usize;
        let mut same = 0usize;
        for (a, b) in texts.iter().zip(base.iter()) {
            chars += b.chars().count();
            for (x, y) in a.chars().zip(b.chars()) {
                if x == y {
                    same += 1;
                }
            }
        }
        let agree = if chars > 0 { same as f64 / chars as f64 } else { 1.0 };
        println!(
            "vs b=1: {} of {} segments differ, char agreement {:.4}",
            diffs,
            texts.len(),
            agree
        );
        if let Some(p) = &out_path {
            std::fs::write(p.with_extension("b1.txt"), base.join("\n"))?;
        }
        if diffs != 0 {
            anyhow::bail!("batch={batch} changed {diffs} transcript(s)");
        }
    }
    Ok(())
}
