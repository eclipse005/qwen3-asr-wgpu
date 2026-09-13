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
use qwen3_asr_wgpu::WgpuAsr;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
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

    println!("model: {}", model.display());
    println!("wav:   {}", wav.display());
    let t_load = Instant::now();
    let asr = WgpuAsr::load(&model, adapter.as_deref())?;
    println!("loaded in {:.1}s", t_load.elapsed().as_secs_f64());

    let dump = arg(&args, "--dump").map(PathBuf::from);
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
        asr.transcribe_from_mel(&mel, n_mels, n_frames, max_new, dump.as_deref())?
    } else if let Some(embeds_path) = arg(&args, "--embeds") {
        let raw = std::fs::read(&embeds_path)?;
        anyhow::ensure!(raw.len() % 4 == 0, "embeds not f32");
        let embeds: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        println!("embeds: {} from {embeds_path}", embeds.len());
        asr.transcribe_from_embeds(&embeds, max_new, dump.as_deref())?
    } else {
        asr.transcribe_with_dump(&wav, max_new, dump.as_deref())?
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
