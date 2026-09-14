//! End-to-end transcribe on wgpu (CPU audio encoder + wgpu text decoder).
//!
//! Alignment target: Python Transformers-native `-hf` greedy dumps.
//!
//! ```text
//! cargo run --release --bin transcribe -- --model D:\Qwen3-ASR\models\Qwen3-ASR-0.6B-hf --wav D:\qwen3-asr-rs\tests\fixtures\15s_en.wav --adapter nvidia
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use qwen3_asr_wgpu::inference::{EncoderBackend, TranscribeOptions};
use qwen3_asr_wgpu::WgpuAsr;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = PathBuf::from(arg(&args, "--model").unwrap_or_else(|| {
        r"D:\Qwen3-ASR\models\Qwen3-ASR-0.6B-hf".to_string()
    }));
    let wav = PathBuf::from(arg(&args, "--wav").unwrap_or_else(|| {
        r"D:\qwen3-asr-rs\tests\fixtures\15s_en.wav".to_string()
    }));
    let adapter = arg(&args, "--adapter");
    let max_new: usize = arg(&args, "--max-new")
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    // The GPU audio tower is the default: it is bit-aligned with the CPU
    // reference on all 12 fixture/model pairs and ~2.7× faster on the encoder
    // phase.  `--cpu-enc` forces the host path back (A/B and regression work);
    // `--gpu-enc` is still accepted and is the default.  A device that cannot
    // build the tower falls back to the CPU one on its own, with a reason.
    let backend = if flag(&args, "--cpu-enc") {
        EncoderBackend::Cpu
    } else {
        EncoderBackend::Gpu
    };

    println!("model: {}", model.display());
    println!("wav:   {}", wav.display());
    let t_load = Instant::now();
    let mut asr = WgpuAsr::load_with(&model, adapter.as_deref(), backend)?;
    println!(
        "loaded in {:.1}s (audio tower: {})",
        t_load.elapsed().as_secs_f64(),
        if asr.gpu_encoder_active() { "gpu" } else { "cpu" }
    );

    // Upstream parity knobs: `--context` (hotword/bias text, goes into the chat
    // template's system message) and `--lang` (force the output language).
    let opts = TranscribeOptions {
        context: arg(&args, "--context").unwrap_or_default(),
        language: arg(&args, "--lang"),
    };
    let dump = arg(&args, "--dump").map(PathBuf::from);
    let compare_enc = flag(&args, "--compare-enc");
    if flag(&args, "--languages") {
        println!("{}", qwen3_asr_wgpu::inference::supported_languages().join(", "));
        return Ok(());
    }
    if flag(&args, "--diag-enc") {
        if let Some(mel_path) = arg(&args, "--mel") {
            let raw = std::fs::read(&mel_path)?;
            let mel: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let n_mels = 128usize;
            let n_frames = mel.len() / n_mels;
            println!("diag from mel: {n_mels}x{n_frames}");
            asr.diagnose_encoder_mel(&mel, n_mels, n_frames)?;
        } else {
            asr.diagnose_encoder(&wav)?;
        }
        return Ok(());
    }
    let t0 = Instant::now();
    let r = if let Some(mel_path) = arg(&args, "--mel") {
        let raw = std::fs::read(&mel_path)?;
        anyhow::ensure!(raw.len() % 4 == 0, "mel not f32");
        let mel: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let n_mels = 128usize;
        anyhow::ensure!(mel.len() % n_mels == 0, "mel length");
        let n_frames = mel.len() / n_mels;
        println!("mel: {n_mels}x{n_frames} from {mel_path}");
        asr.transcribe_from_mel_cmp_opts(
            &mel, n_mels, n_frames, max_new, dump.as_deref(), compare_enc, &opts,
        )?
    } else if let Some(embeds_path) = arg(&args, "--embeds") {
        let raw = std::fs::read(&embeds_path)?;
        anyhow::ensure!(raw.len() % 4 == 0, "embeds not f32");
        let embeds: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        println!("embeds: {} from {embeds_path}", embeds.len());
        asr.transcribe_from_embeds(&embeds, max_new, dump.as_deref())?
    } else if flag(&args, "--session") {
        // Feed the wav through the incremental session in 1 s chunks (what a
        // live source would do), then flush — the text must equal the
        // whole-clip run.
        let samples = qwen3_asr_wgpu::mel::load_audio_wav(&wav, 16_000)?;
        let mut sess = asr.create_streaming_session(opts.clone(), max_new)?;
        let chunk = 16_000;
        for part in samples.chunks(chunk) {
            sess.push_samples(part)?;
        }
        println!("session encoded {} tokens", sess.encoded_tokens());
        sess.flush()?
    } else if flag(&args, "--stream") {
        // Per-token callback (reference port's `transcribe_streaming`): the
        // incremental text goes to stderr, so stdout still carries the final
        // text for `--baseline`.
        let mut shown = 0usize;
        asr.transcribe_file_streaming(&wav, max_new, dump.as_deref(), compare_enc, &opts, |t| {
            if t.text_so_far.len() > shown {
                eprint!("{}", &t.text_so_far[shown..]);
                shown = t.text_so_far.len();
            }
        })?
    } else {
        asr.transcribe_file_opts(&wav, max_new, dump.as_deref(), compare_enc, &opts)?
    };
    let elapsed = t0.elapsed().as_secs_f64();
    let audio_s = if arg(&args, "--mel").is_none() && arg(&args, "--embeds").is_none() {
        hound::WavReader::open(&wav).ok().map(|rdr| {
            let sr = rdr.spec().sample_rate as f64;
            rdr.duration() as f64 / sr
        })
    } else {
        None
    };
    match audio_s {
        Some(audio_s) => println!(
            "elapsed={elapsed:.3}s audio={audio_s:.3}s RTFx={:.3} lang={}",
            audio_s / elapsed,
            r.language
        ),
        None => println!("elapsed={elapsed:.3}s lang={}", r.language),
    }
    println!("{}", r.text);

    if let Some(base) = arg(&args, "--baseline") {
        let raw = std::fs::read_to_string(&base)?.replace("\r\n", "\n");
        let want = raw.split("\n\n").nth(1).unwrap_or("").trim();
        let ok = r.text.trim() == want;
        println!("vs {}: {}", base, if ok { "MATCH" } else { "MISMATCH" });
        if !ok {
            println!("WANT: {want}");
            println!("GOT:  {}", r.text);
            std::process::exit(1);
        }
    }
    Ok(())
}
