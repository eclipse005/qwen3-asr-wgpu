use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use qwen3_asr_wgpu::{diagnostics, AsrInference, DeviceSelector, EncoderBackend, TranscribeOptions};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn required(args: &[String], name: &str, env: &str) -> Result<String> {
    arg(args, name)
        .or_else(|| std::env::var(env).ok().filter(|s| !s.is_empty()))
        .ok_or_else(|| anyhow::anyhow!("{name} is required (or set {env})"))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    // These two answer without a model, so they run before the paths are
    // required — `transcribe --list-devices` on a fresh machine is the first
    // thing anyone types.
    if flag(&args, "--languages") {
        println!("{}", qwen3_asr_wgpu::supported_languages().join(", "));
        return Ok(());
    }
    if flag(&args, "--list-devices") || flag(&args, "--devices") {
        println!("runtimes are the axis; the vendor is information (a card is reachable");
        println!("through several runtimes and those are different code paths).\n");
        for t in AsrInference::device_targets() {
            println!("{}", t.describe());
        }
        println!("* {:<10} host backend: CPU audio tower + CPU text decoder (--cpu-dec)", "cpu");
        return Ok(());
    }
    let model = PathBuf::from(required(&args, "--model", "QASR_MODEL")?);
    let wav = PathBuf::from(required(&args, "--wav", "QASR_WAV")?);
    let adapter = if flag(&args, "--cpu-dec") {
        Some("cpu".to_string())
    } else {
        arg(&args, "--device")
            .or_else(|| arg(&args, "--adapter"))
            .or_else(|| std::env::var("QASR_DEVICE").ok().filter(|s| !s.is_empty()))
    };
    let max_new: usize = arg(&args, "--max-new")
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let backend = if flag(&args, "--cpu-enc") {
        EncoderBackend::Cpu
    } else {
        EncoderBackend::Gpu
    };
    let selector = match adapter.as_deref() {
        Some(a) => DeviceSelector::parse(a)?,
        None => DeviceSelector::Auto,
    };

    println!("model: {}", model.display());
    println!("wav:   {}", wav.display());
    let t_load = Instant::now();
    let asr = AsrInference::load_with(&model, selector, backend)?;
    println!(
        "loaded in {:.1}s (audio tower: {})",
        t_load.elapsed().as_secs_f64(),
        if asr.gpu_encoder_active() { "gpu" } else { "cpu" }
    );

    let mut opts = TranscribeOptions::default().with_max_new_tokens(max_new);
    opts.context = arg(&args, "--context")
        .or_else(|| arg(&args, "--prompt"))
        .unwrap_or_default();
    opts.language = arg(&args, "--lang");
    let dump = arg(&args, "--dump").map(PathBuf::from);
    let compare_enc = flag(&args, "--compare-enc");
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
            diagnostics::diagnose_encoder_mel(&asr, &mel, n_mels, n_frames)?;
        } else {
            diagnostics::diagnose_encoder(&asr, &wav)?;
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
        diagnostics::transcribe_from_mel(
            &asr, &mel, n_mels, n_frames, &opts, dump.as_deref(), compare_enc,
        )?
    } else if let Some(embeds_path) = arg(&args, "--embeds") {
        let raw = std::fs::read(&embeds_path)?;
        anyhow::ensure!(raw.len() % 4 == 0, "embeds not f32");
        let embeds: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        println!("embeds: {} from {embeds_path}", embeds.len());
        diagnostics::transcribe_from_embeds(&asr, &embeds, &opts, dump.as_deref())?
    } else if flag(&args, "--session") {
        let samples = qwen3_asr_wgpu::load_audio_wav(&wav, 16_000)?;
        let mut sess = asr.create_streaming_session(opts.clone())?;
        let chunk = 16_000;
        for part in samples.chunks(chunk) {
            sess.push_samples(part)?;
        }
        println!("session encoded {} tokens", sess.encoded_tokens());
        sess.flush()?
    } else if flag(&args, "--stream") {
        let mut shown = 0usize;
        diagnostics::transcribe_with_dump_streaming(
            &asr,
            &wav,
            &opts,
            dump.as_deref(),
            compare_enc,
            |t| {
                if t.text_so_far.len() > shown {
                    eprint!("{}", &t.text_so_far[shown..]);
                    shown = t.text_so_far.len();
                }
            },
        )?
    } else {
        diagnostics::transcribe_with_dump(&asr, &wav, &opts, dump.as_deref(), compare_enc)?
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
