use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use qwen3_asr_wgpu::{
    diagnostics, AsrInference, AudioSpec, Backend, DeviceSelector, EncoderBackend,
    TranscribeOptions,
};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

/// `auto` may fall back. `cpu` is the host. `gpu` and any other spec require
/// that device. `--cpu-enc` keeps the host audio tower and still obeys the
/// same device policy for the text decoder.
fn load_asr(model: &std::path::Path, spec: &str, cpu_enc: bool) -> Result<AsrInference> {
    if spec.eq_ignore_ascii_case("cpu") {
        return Ok(AsrInference::load(model, Backend::Cpu)?);
    }
    if cpu_enc {
        let (selector, allow_fallback) = if spec.eq_ignore_ascii_case("auto") {
            (DeviceSelector::Auto, true)
        } else if spec.eq_ignore_ascii_case("gpu") {
            (DeviceSelector::Auto, false)
        } else {
            (DeviceSelector::parse(spec)?, false)
        };
        return Ok(AsrInference::load_with_fallback(
            model,
            selector,
            EncoderBackend::Cpu,
            allow_fallback,
        )?);
    }
    if spec.eq_ignore_ascii_case("auto") {
        return Ok(AsrInference::load(model, Backend::Auto)?);
    }
    if spec.eq_ignore_ascii_case("gpu") {
        return Ok(AsrInference::load(model, Backend::Gpu)?);
    }
    Ok(AsrInference::load_on(model, DeviceSelector::parse(spec)?)?)
}

fn required(args: &[String], name: &str, env: &str) -> Result<String> {
    arg(args, name)
        .or_else(|| std::env::var(env).ok().filter(|s| !s.is_empty()))
        .ok_or_else(|| anyhow::anyhow!("{name} is required (or set {env})"))
}


/// Every `--wav` value: variadic (a list) and repeatable, so both
/// `--wav a.wav b.wav` and `--wav a.wav --wav b.wav` work.
fn collect_wavs(args: &[String]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--wav" {
            let mut j = i + 1;
            while j < args.len() && !args[j].starts_with("--") {
                out.push(PathBuf::from(&args[j]));
                j += 1;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// One manifest line.  `audio` is required; the rest fall back to the command
/// line, so `--lang`/`--prompt`/`--max-new` still mean "for every item".
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestLine {
    audio: PathBuf,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    lang: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    context: Option<String>,
    #[serde(default)]
    max_new: Option<usize>,
}

/// Read a JSONL manifest into specs, resolving relative paths against the
/// manifest's own directory and rejecting unknown fields so a typo is an error
/// rather than a silently ignored setting.
fn load_manifest(
    path: &std::path::Path,
    default_lang: Option<&str>,
    default_ctx: &str,
    default_max_new: usize,
) -> Result<Vec<AudioSpec>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let mut out = Vec::new();
    for (no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut de = serde_json::Deserializer::from_str(line);
        let item: ManifestLine = serde::Deserialize::deserialize(&mut de)
            .map_err(|e| anyhow::anyhow!("{}:{}: {e}", path.display(), no + 1))?;
        de.end()
            .map_err(|e| anyhow::anyhow!("{}:{}: {e}", path.display(), no + 1))?;
        let audio = if item.audio.is_absolute() { item.audio } else { base.join(&item.audio) };
        anyhow::ensure!(
            audio.is_file(),
            "{}:{}: audio not found: {}",
            path.display(),
            no + 1,
            audio.display()
        );
        out.push(AudioSpec {
            audio,
            language: item.language.or(item.lang).or_else(|| default_lang.map(str::to_string)),
            context: item
                .prompt
                .or(item.context)
                .unwrap_or_else(|| default_ctx.to_string()),
            max_new_tokens: item.max_new.unwrap_or(default_max_new),
        });
    }
    anyhow::ensure!(!out.is_empty(), "{}: no items", path.display());
    Ok(out)
}

/// The list half: one call, one transcript per item, in input order.
fn run_list(
    asr: &AsrInference,
    opts: &TranscribeOptions,
    specs: &[AudioSpec],
    batch: usize,
    compare: bool,
    timings: bool,
    out_path: Option<&std::path::Path>,
    jsonl: bool,
) -> Result<()> {
    let mut audio = 0.0f64;
    for sp in specs {
        let rd = hound::WavReader::open(&sp.audio)
            .map_err(|e| anyhow::anyhow!("{}: {e}", sp.audio.display()))?;
        audio += rd.duration() as f64 / rd.spec().sample_rate as f64;
    }
    let t = Instant::now();
    let (rs, ph) = asr.transcribe_specs_timed(specs, opts, batch)?;
    let secs = t.elapsed().as_secs_f64();
    if timings {
        println!(
            "phases  mel={:.2}s encoder={:.2}s prefill={:.2}s decode={:.2}s",
            ph.mel.as_secs_f64(),
            ph.encoder.as_secs_f64(),
            ph.prefill.as_secs_f64(),
            ph.decode.as_secs_f64()
        );
    }
    println!(
        "batch={batch} files={} audio={audio:.1}s elapsed={secs:.2}s RTFx={:.2}",
        specs.len(),
        audio / secs
    );
    if !jsonl {
        for (sp, r) in specs.iter().zip(rs.iter()) {
            let name = sp.audio.display().to_string();
            println!("  {name} : {}", r.text.chars().take(40).collect::<String>());
        }
    }
    if let Some(p) = out_path {
        let body = if jsonl {
            let mut v = String::new();
            for (sp, r) in specs.iter().zip(rs.iter()) {
                let stop = match r.stop_reason {
                    qwen3_asr_wgpu::StopReason::Eos => "eos",
                    qwen3_asr_wgpu::StopReason::MaxTokens => "max-tokens",
                };
                v.push_str(&serde_json::json!({
                    "audio": sp.audio.display().to_string(),
                    "language": r.language,
                    "stop": stop,
                    "text": r.text,
                })
                .to_string());
                v.push(NL);
            }
            v
        } else {
            let mut v: String =
                rs.iter().map(|r| r.text.as_str()).collect::<Vec<_>>().join(&NL.to_string());
            v.push(NL);
            v
        };
        std::fs::write(p, body).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
    }
    if compare {
        // The batched path's own gate: the same list at `--batch 1` has to match
        // segment by segment, or a "faster" run changed a transcript.
        let (one, _) = asr.transcribe_specs_timed(specs, opts, 1)?;
        let mut bad = 0usize;
        for (a, b) in one.iter().zip(rs.iter()) {
            if a.text != b.text {
                bad += 1;
                println!("  DIFF {}", a.text.chars().take(40).collect::<String>());
                println!("   vs {}", b.text.chars().take(40).collect::<String>());
            }
        }
        println!(
            "vs b=1: {bad} of {} segments differ{}",
            rs.len(),
            if bad == 0 { ", transcripts identical" } else { "" }
        );
        if bad > 0 {
            std::process::exit(1);
        }
    }
    Ok(())
}

const NL: char = '\n';

#[allow(dead_code)] // also compiled as a module by the `transcribe_batch` shim
fn main() -> Result<()> {
    run(std::env::args().collect())
}

/// The whole CLI as a function, so `transcribe_batch` can forward to it.
pub fn run(args: Vec<String>) -> Result<()> {
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
    // Inputs: `--manifest items.jsonl`, or `--wav a.wav [b.wav ...]`.  One wav
    // keeps the single-file behaviour byte for byte; more than one (or a
    // manifest) goes through the list path.
    let manifest = arg(&args, "--manifest");
    let wavs = collect_wavs(&args);
    anyhow::ensure!(
        manifest.is_some() || !wavs.is_empty(),
        "--wav <file...> or --manifest <jsonl> is required"
    );
    anyhow::ensure!(
        !(manifest.is_some() && !wavs.is_empty()),
        "give either --manifest or --wav, not both"
    );
    let list_mode = manifest.is_some() || wavs.len() > 1;
    let wav = wavs
        .first()
        .cloned()
        .or_else(|| std::env::var("QASR_WAV").ok().filter(|s| !s.is_empty()).map(PathBuf::from))
        .unwrap_or_default();
    // `--cpu-dec` is the whole host path. Otherwise `auto` may fall back,
    // `gpu` must open a GPU, and a runtime or name is that adapter or an error.
    let spec = if flag(&args, "--cpu-dec") {
        "cpu".to_string()
    } else {
        arg(&args, "--device")
            .or_else(|| arg(&args, "--adapter"))
            .or_else(|| std::env::var("QASR_DEVICE").ok().filter(|s| !s.is_empty()))
            .unwrap_or_else(|| "auto".to_string())
    };
    let max_new: usize = arg(&args, "--max-new")
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let cpu_enc = flag(&args, "--cpu-enc");

    println!("model: {}", model.display());
    if let Some(m) = &manifest {
        println!("manifest: {m}");
    } else if list_mode {
        println!("inputs: {} file(s)", wavs.len());
    }
    if !list_mode {
        println!("wav:   {}", wav.display());
    }
    let t_load = Instant::now();
    let asr = load_asr(&model, &spec, cpu_enc)?;
    println!(
        "loaded in {:.1}s (audio tower: {})",
        t_load.elapsed().as_secs_f64(),
        if asr.gpu_encoder_active() { "gpu" } else { "cpu" }
    );

    let batch: usize = arg(&args, "--batch").and_then(|v| v.parse().ok()).unwrap_or(1);
    let out_path = arg(&args, "--out").map(PathBuf::from);
    let jsonl = arg(&args, "--out-format").map(|f| f.eq_ignore_ascii_case("jsonl")).unwrap_or(false);
    let want_compare = flag(&args, "--compare");
    let want_timings = flag(&args, "--timings");
    if list_mode {
        for f in ["--stream", "--session", "--tokens", "--mel", "--embeds", "--diag-enc",
                  "--baseline", "--compare-enc"] {
            anyhow::ensure!(
                !flag(&args, f),
                "{f} is a single-file mode; give it one --wav"
            );
        }
        let default_lang = arg(&args, "--language").or_else(|| arg(&args, "--lang"));
        let default_ctx = arg(&args, "--prompt")
            .or_else(|| arg(&args, "--context"))
            .unwrap_or_default();
        let specs = match &manifest {
            Some(m) => load_manifest(std::path::Path::new(m), default_lang.as_deref(),
                                     &default_ctx, max_new)?,
            None => wavs
                .iter()
                .map(|w| AudioSpec {
                    audio: w.clone(),
                    language: default_lang.clone(),
                    context: default_ctx.clone(),
                    max_new_tokens: max_new,
                })
                .collect(),
        };
        // The list's shared defaults: the per-item specs already carry the
        // resolved language/context, so only the knobs that stay global (here:
        // nothing yet, but the options are what `--compare` reruns with) ride on
        // this struct.
        let mut opts = TranscribeOptions::default().with_max_new_tokens(max_new);
        if let Some(l) = default_lang.as_deref() {
            opts = opts.with_language(l);
        }
        opts = opts.with_context(&default_ctx);
        return run_list(
            &asr, &opts, &specs, batch.max(1), want_compare, want_timings,
            out_path.as_deref(), jsonl,
        );
    }
    if batch > 1 || want_compare {
        eprintln!(
            "{}",
            "note: --batch/--compare need a list (two or more --wav, or --manifest); running the single-file path"
        );
    }
    let mut opts = TranscribeOptions::default().with_max_new_tokens(max_new);
    // Official name first: transformers `apply_transcription_request(prompt=…)`
    // and the chat-template system message; `--context` is the qwen-asr
    // package's `transcribe(context=…)` spelling for the same slot.
    opts.context = arg(&args, "--prompt")
        .or_else(|| arg(&args, "--context"))
        .unwrap_or_default();
    opts.language = arg(&args, "--language").or_else(|| arg(&args, "--lang"));
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
        let mut sess = asr.create_buffered_session(opts.clone())?;
        let chunk = 16_000;
        for part in samples.chunks(chunk) {
            sess.push_samples(part)?;
        }
        println!("session encoded {} tokens", sess.encoded_tokens());
        sess.flush()?
    } else if flag(&args, "--tokens") {
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
    } else if flag(&args, "--stream") {
        let samples = qwen3_asr_wgpu::load_audio_wav(&wav, 16_000)?;
        let mut shown = 0usize;
        let mut sess = asr.create_stream_session(opts.clone(), |u| {
            if u.stable.len() > shown {
                eprint!("{}", &u.stable[shown..]);
                shown = u.stable.len();
            }
        })?;
        for part in samples.chunks(32_000) {
            sess.push_samples(part)?;
        }
        let r = sess.flush()?;
        eprintln!();
        r
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
    let stop = match r.stop_reason {
        qwen3_asr_wgpu::StopReason::Eos => "eos",
        qwen3_asr_wgpu::StopReason::MaxTokens => "max-tokens",
    };
    match audio_s {
        Some(audio_s) => println!(
            "elapsed={elapsed:.3}s audio={audio_s:.3}s RTFx={:.3} lang={} stop={stop}",
            audio_s / elapsed,
            r.language
        ),
        None => println!("elapsed={elapsed:.3}s lang={} stop={stop}", r.language),
    }
    println!("{}", r.text);
    if let Some(p) = &out_path {
        std::fs::write(p, format!("{}{NL}", r.text))
            .map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
    }

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
