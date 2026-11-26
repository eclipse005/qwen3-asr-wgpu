//! Batch driver for [`TranscribeOptions::dump_states`]: one process, one
//! weight load, one output directory per utterance.
//!
//! `transcribe --dump-states` refuses to run in list mode (it is a single-file
//! mode), and reloading the checkpoint per utterance would pay the ~10 s load
//! 471 times, so this example owns the loop: load once, then dump each manifest
//! line into `<out-root>/<utt_id>`.
//!
//! ```text
//! cargo build --release --example dump_states_batch
//! dump_states_batch --model <dir> --manifest items.jsonl --out-root <dir> \
//!                   --device vulkan --max-new 512
//! ```
//!
//! Manifest lines are `{"utt_id": "...", "wav": "..."}`.  Each utterance goes
//! through `diagnostics::transcribe_with_dump`, i.e. the same
//! `transcribe_file_opts` call `transcribe --wav one.wav --dump D --dump-states D`
//! makes, with `D = <out-root>/<utt_id>`: the states dump, and with it the
//! audio tower's output (`audio_embeds.f32`) that a later stage would otherwise
//! have to re-encode every clip to obtain.
//!
//! A directory that already holds `meta.json` is skipped, so an interrupted run
//! continues where it stopped — the dump writes `meta.json` last, which is what
//! makes "has meta.json" the same question as "this utterance's states are
//! complete".

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use qwen3_asr_wgpu::{diagnostics, AsrInference, Backend, DeviceSelector, TranscribeOptions};

/// One manifest line.  Unknown fields are rejected so a typo is an error rather
/// than a silently ignored key.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestLine {
    utt_id: String,
    wav: PathBuf,
}

struct Item {
    utt_id: String,
    wav: PathBuf,
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn required(args: &[String], name: &str) -> Result<String> {
    arg(args, name).ok_or_else(|| anyhow::anyhow!("{name} is required"))
}

/// `--device` with the meaning it has in `transcribe`: `auto` may fall back,
/// `cpu` is the host, and any other spec is that adapter or an error — so
/// `vulkan` opens Vulkan or fails instead of landing on another runtime.
fn load_asr(model: &Path, spec: &str) -> Result<AsrInference> {
    if spec.eq_ignore_ascii_case("cpu") {
        return Ok(AsrInference::load(model, Backend::Cpu)?);
    }
    if spec.eq_ignore_ascii_case("auto") {
        return Ok(AsrInference::load(model, Backend::Auto)?);
    }
    if spec.eq_ignore_ascii_case("gpu") {
        return Ok(AsrInference::load(model, Backend::Gpu)?);
    }
    Ok(AsrInference::load_on(model, DeviceSelector::parse(spec)?)?)
}

/// Manifest text → items.  A relative `wav` resolves against the manifest's own
/// directory, and a missing file is an error here rather than a failed decode
/// 200 utterances later.
fn load_manifest(path: &Path) -> Result<Vec<Item>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut items: Vec<Item> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let item: ManifestLine = serde_json::from_str(line)
            .map_err(|e| anyhow::anyhow!("{}:{}: {e}", path.display(), no + 1))?;
        let wav = if item.wav.is_absolute() { item.wav } else { base.join(&item.wav) };
        anyhow::ensure!(
            wav.is_file(),
            "{}:{}: audio not found: {}",
            path.display(),
            no + 1,
            wav.display()
        );
        anyhow::ensure!(
            seen.insert(item.utt_id.clone()),
            "{}:{}: duplicate utt_id {}",
            path.display(),
            no + 1,
            item.utt_id
        );
        items.push(Item { utt_id: item.utt_id, wav });
    }
    anyhow::ensure!(!items.is_empty(), "{}: no items", path.display());
    Ok(items)
}

/// A finished dump's decode-step count, or `None` when the directory has no
/// `meta.json`.  Rows, not generated tokens: they differ by at most one (the
/// step that predicts the EOS is a row without being a generated token) and the
/// rows are what the states files — and the packed npz — hold.
fn finished_rows(dir: &Path) -> Option<usize> {
    let meta = std::fs::read_to_string(dir.join("meta.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&meta).ok()?;
    v.get("rows")?.as_u64().map(|n| n as usize)
}

/// Audio duration from the wav header.
fn audio_seconds(path: &Path) -> f64 {
    match hound::WavReader::open(path) {
        Ok(r) => r.duration() as f64 / r.spec().sample_rate as f64,
        Err(_) => 0.0,
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = PathBuf::from(required(&args, "--model")?);
    let manifest = PathBuf::from(required(&args, "--manifest")?);
    let out_root = PathBuf::from(required(&args, "--out-root")?);
    let device = arg(&args, "--device").unwrap_or_else(|| "auto".to_string());
    let max_new: usize = arg(&args, "--max-new")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| TranscribeOptions::default().max_new_tokens);

    let items = load_manifest(&manifest)?;
    std::fs::create_dir_all(&out_root).with_context(|| format!("{}", out_root.display()))?;
    let total_audio: f64 = items.iter().map(|i| audio_seconds(&i.wav)).sum();
    eprintln!(
        "manifest: {} item(s), {:.1}s audio; out-root: {}",
        items.len(),
        total_audio,
        out_root.display()
    );

    let t_load = Instant::now();
    let asr = load_asr(&model, &device)?;
    let load_s = t_load.elapsed().as_secs_f64();
    eprintln!(
        "model: {} loaded in {:.1}s (device: {}, audio tower: {})",
        model.display(),
        load_s,
        device,
        if asr.gpu_encoder_active() { "gpu" } else { "cpu" }
    );

    let t_run = Instant::now();
    let mut rows_total = 0usize;
    let mut audio_done = 0.0f64;
    let mut done = 0usize;
    let mut skipped = 0usize;
    let mut failed: Vec<(String, String)> = Vec::new();

    for item in &items {
        let dir = out_root.join(&item.utt_id);
        let audio_s = audio_seconds(&item.wav);
        if let Some(rows) = finished_rows(&dir) {
            skipped += 1;
            done += 1;
            rows_total += rows;
            audio_done += audio_s;
            if done % 50 == 0 {
                eprintln!(
                    "progress: {}/{} rows={} audio={:.1}s wall={:.1}s",
                    done,
                    items.len(),
                    rows_total,
                    audio_done,
                    t_run.elapsed().as_secs_f64()
                );
            }
            continue;
        }
        // One directory holds every artifact of this utterance: the states dump
        // chosen by `dump_states` (`states.f16`, `states_norm.f32`,
        // `gen_ids.json`, `meta.json`) plus, through the `dump_dir` argument, the
        // audio tower's output the same call already has in hand
        // (`audio_embeds.f32`, with `mel.f32` / `wave16k.f32` / `shapes.txt`).
        let opts = TranscribeOptions::default()
            .with_max_new_tokens(max_new)
            .with_dump_states(&dir);
        match diagnostics::transcribe_with_dump(&asr, &item.wav, &opts, Some(&dir), false) {
            Ok(_) => {
                let rows = match finished_rows(&dir) {
                    Some(r) => r,
                    None => {
                        failed.push((item.utt_id.clone(), "no meta.json after the run".to_string()));
                        done += 1;
                        continue;
                    }
                };
                rows_total += rows;
                audio_done += audio_s;
                done += 1;
                if done % 50 == 0 {
                    eprintln!(
                        "progress: {}/{} rows={} audio={:.1}s wall={:.1}s",
                        done,
                        items.len(),
                        rows_total,
                        audio_done,
                        t_run.elapsed().as_secs_f64()
                    );
                }
            }
            Err(e) => {
                done += 1;
                failed.push((item.utt_id.clone(), format!("{e:#}")));
                eprintln!("FAILED {}: {e:#}", item.utt_id);
            }
        }
    }

    let run_s = t_run.elapsed().as_secs_f64();
    println!(
        "summary: items={} ok={} skipped={} failed={} rows={} audio={:.1}s load={:.1}s run={:.1}s wall={:.1}s",
        items.len(),
        done - skipped - failed.len(),
        skipped,
        failed.len(),
        rows_total,
        audio_done,
        load_s,
        run_s,
        load_s + run_s
    );
    if !failed.is_empty() {
        println!("failed utt_ids ({}):", failed.len());
        for (id, why) in &failed {
            println!("  {id}: {why}");
        }
    }
    // A partial dump is not a success: downstream totals would silently cover
    // fewer sentences than the manifest names.
    anyhow::ensure!(failed.is_empty(), "{} item(s) failed", failed.len());
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
