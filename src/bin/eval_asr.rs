//! Corpus WER/CER against a **human-labelled** reference — the quality metric,
//! as opposed to the "does it match python-hf byte for byte" regression check.
//!
//! Input is a FLEURS `test.tsv` (`id, filename, raw, transcription, …`, the
//! 4th column is the normalized reference) or a plain two-column
//! `filename<TAB>reference` manifest for your own audio.
//!
//! ```text
//! cargo run --release --bin eval_asr -- --model D:\Qwen3-ASR\models\Qwen3-ASR-0.6B-hf ^
//!   --tsv eval_data\fleurs\tsv\en_us.test.tsv --audio-dir eval_data\fleurs\audio\en_us ^
//!   --mode wer --limit 200
//! ```
//!
//! Both sides are normalized the same way before scoring: lowercase, and every
//! non-alphanumeric character folded to a space (FLEURS keeps hyphens and the
//! occasional period, the model writes its own punctuation — folding both sides
//! is what makes the comparison fair).  `--mode cer` drops whitespace as well,
//! which is the standard metric for Chinese and Japanese.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use qwen3_asr_wgpu::inference::{EncoderBackend, TranscribeOptions};
use qwen3_asr_wgpu::mel::load_audio_wav;
use qwen3_asr_wgpu::WgpuAsr;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Word error rate — whitespace-separated units.
    Wer,
    /// Character error rate — whitespace is dropped first.
    Cer,
}

struct Row {
    file: String,
    reference: String,
}

/// FLEURS rows carry 7 columns (id, filename, raw, normalized, …); anything
/// with two columns is taken as `filename<TAB>reference`.
fn read_rows(path: &Path) -> Result<Vec<Row>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut rows = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        let row = if cols.len() >= 4 {
            Row { file: cols[1].to_string(), reference: cols[3].to_string() }
        } else if cols.len() == 2 {
            Row { file: cols[0].to_string(), reference: cols[1].to_string() }
        } else {
            bail!("{}:{}: {} columns, expected 2 or >=4", path.display(), n + 1, cols.len());
        };
        rows.push(row);
    }
    Ok(rows)
}

/// Lowercase; every non-alphanumeric character becomes a space (CJK ideographs
/// are alphanumeric, so they survive), runs collapsed and trimmed.
fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        } else {
            out.push(' ');
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn units(s: &str, mode: Mode) -> Vec<String> {
    match mode {
        Mode::Wer => s.split_whitespace().map(str::to_string).collect(),
        Mode::Cer => s.chars().filter(|c| !c.is_whitespace()).map(|c| c.to_string()).collect(),
    }
}

#[derive(Default, Clone, Copy)]
struct Edits {
    sub: usize,
    del: usize,
    ins: usize,
}

impl Edits {
    fn total(&self) -> usize {
        self.sub + self.del + self.ins
    }
}

/// Levenshtein distance with the substitution/deletion/insertion breakdown
/// (backtraced from the DP matrix; the sequences are tens of units long).
fn edit_counts(reference: &[String], hypothesis: &[String]) -> Edits {
    let (n, m) = (reference.len(), hypothesis.len());
    let idx = |i: usize, j: usize| i * (m + 1) + j;
    let mut d = vec![0usize; (n + 1) * (m + 1)];
    for i in 0..=n {
        d[idx(i, 0)] = i;
    }
    for j in 0..=m {
        d[idx(0, j)] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = usize::from(reference[i - 1] != hypothesis[j - 1]);
            d[idx(i, j)] =
                (d[idx(i - 1, j - 1)] + cost).min(d[idx(i - 1, j)] + 1).min(d[idx(i, j - 1)] + 1);
        }
    }
    let (mut i, mut j) = (n, m);
    let mut e = Edits::default();
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && reference[i - 1] == hypothesis[j - 1] {
            i -= 1;
            j -= 1;
        } else if i > 0 && j > 0 && d[idx(i, j)] == d[idx(i - 1, j - 1)] + 1 {
            e.sub += 1;
            i -= 1;
            j -= 1;
        } else if i > 0 && d[idx(i, j)] == d[idx(i - 1, j)] + 1 {
            e.del += 1;
            i -= 1;
        } else {
            e.ins += 1;
            j -= 1;
        }
    }
    e
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = PathBuf::from(
        arg(&args, "--model").unwrap_or_else(|| r"D:\Qwen3-ASR\models\Qwen3-ASR-0.6B-hf".into()),
    );
    let tsv = PathBuf::from(
        arg(&args, "--tsv").unwrap_or_else(|| r"eval_data\fleurs\tsv\en_us.test.tsv".into()),
    );
    let audio_dir = arg(&args, "--audio-dir").map(PathBuf::from);
    let mode = match arg(&args, "--mode").as_deref() {
        Some("cer") => Mode::Cer,
        Some("wer") | None => Mode::Wer,
        Some(other) => bail!("--mode {other}: expected wer or cer"),
    };
    let limit: usize = arg(&args, "--limit").and_then(|s| s.parse().ok()).unwrap_or(100);
    let offset: usize = arg(&args, "--offset").and_then(|s| s.parse().ok()).unwrap_or(0);
    // `--even N` spreads the N clips across the whole test set instead of taking
    // the head, so short and long clips are both represented.
    let even: Option<usize> = arg(&args, "--even").and_then(|s| s.parse().ok());
    let max_new: usize = arg(&args, "--max-new").and_then(|s| s.parse().ok()).unwrap_or(256);
    let adapter = arg(&args, "--adapter");
    let backend =
        if flag(&args, "--cpu-enc") { EncoderBackend::Cpu } else { EncoderBackend::Gpu };
    let opts = TranscribeOptions { context: String::new(), language: arg(&args, "--lang") };
    let quiet = flag(&args, "--quiet");
    // Hypothesis dump for cross-checking the numbers with a standards-compliant
    // scorer (tools/score_asr.py) — one `filename<TAB>text` line per clip.
    let hyps_out = arg(&args, "--hyps-out").map(PathBuf::from);
    let mut hyps_file = match &hyps_out {
        Some(p) => {
            let f = std::fs::File::create(p).with_context(|| format!("create {}", p.display()))?;
            Some(std::io::BufWriter::new(f))
        }
        None => None,
    };
    // Optional second dump for the language-identification score.
    let lang_out = arg(&args, "--lang-out").map(PathBuf::from);
    let mut lang_file = match &lang_out {
        Some(p) => {
            let f = std::fs::File::create(p).with_context(|| format!("create {}", p.display()))?;
            Some(std::io::BufWriter::new(f))
        }
        None => None,
    };
    // Machine-readable run summary (same columns as the Python runner writes).
    let summary_out = arg(&args, "--summary-out").map(PathBuf::from);

    let all = read_rows(&tsv)?;
    let end = (offset + limit).min(all.len());
    if offset >= all.len() {
        bail!("--offset {offset} past the {} rows of {}", all.len(), tsv.display());
    }
    let audio_dir = audio_dir.unwrap_or_else(|| tsv.parent().unwrap_or(Path::new(".")).to_path_buf());
    // Which clips to run.  `--even N` spreads N across the test set sorted by
    // file size — every FLEURS clip is 16 kHz mono PCM16, so size order is
    // duration order and the sample covers short *and* long audio instead of
    // whatever happens to sit at the head of the tsv.  The other system must
    // pick the same clips, so the key is (size, name), never a random draw.
    let rows: Vec<&Row> = match even {
        Some(n) => {
            let mut sorted: Vec<&Row> = all.iter().collect();
            sorted.sort_by_key(|r| {
                let size = std::fs::metadata(audio_dir.join(&r.file)).map(|m| m.len()).unwrap_or(0);
                (size, r.file.clone())
            });
            let step = sorted.len() as f64 / n as f64;
            (0..n)
                .map(|i| (i as f64 * step) as usize)
                .filter(|&i| i < sorted.len())
                .map(|i| sorted[i])
                .collect()
        }
        None => all[offset..end].iter().collect(),
    };

    println!("model:  {}", model.display());
    println!(
        "tsv:    {} ({} of {} clips{})",
        tsv.display(),
        rows.len(),
        all.len(),
        if even.is_some() { ", evenly spaced by size" } else { "" }
    );
    println!("audio:  {}", audio_dir.display());
    println!("metric: {}  (lang: {})",
        if mode == Mode::Cer { "CER" } else { "WER" },
        opts.language.as_deref().unwrap_or("<detect>"));
    let t_load = Instant::now();
    let mut asr = WgpuAsr::load_with(&model, adapter.as_deref(), backend)?;
    let load_s = t_load.elapsed().as_secs_f64();
    println!(
        "loaded in {load_s:.1}s (audio tower: {})",
        if asr.gpu_encoder_active() { "gpu" } else { "cpu" }
    );

    let mut total_units = 0usize;
    let mut total = Edits::default();
    let mut per_utt = Vec::new();
    let mut audio_s = 0.0f64;
    let mut languages = std::collections::BTreeMap::<String, usize>::new();
    let t0 = Instant::now();

    for (i, row) in rows.iter().enumerate() {
        let path = audio_dir.join(&row.file);
        // The timer covers "read the wav -> text", which is what the Python
        // reference run measures too, so the two RTFx numbers are comparable.
        let t = Instant::now();
        let samples =
            load_audio_wav(&path, 16_000).with_context(|| format!("audio {}", path.display()))?;
        audio_s += samples.len() as f64 / 16_000.0;
        let out = asr.transcribe_samples(&samples, max_new)?;
        let secs = t.elapsed().as_secs_f64();

        let reference = units(&normalize(&row.reference), mode);
        let hypothesis = units(&normalize(&out.text), mode);
        let e = edit_counts(&reference, &hypothesis);
        if let Some(f) = hyps_file.as_mut() {
            use std::io::Write;
            writeln!(f, "{}\t{}", row.file, out.text.replace(['\r', '\n'], " "))?;
        }
        if let Some(f) = lang_file.as_mut() {
            use std::io::Write;
            writeln!(f, "{}\t{}", row.file, out.language)?;
        }
        let rate = if reference.is_empty() {
            0.0
        } else {
            e.total() as f64 / reference.len() as f64
        };
        *languages.entry(out.language.clone()).or_default() += 1;
        if !quiet {
            println!(
                "[{:>4}] {:>5.2}s {:>5.1}% lang={:<10} ref{:>4} err{:>3}  {}",
                i,
                secs,
                100.0 * rate,
                if out.language.is_empty() { "-" } else { &out.language },
                reference.len(),
                e.total(),
                truncate(&out.text, 90)
            );
        }
        total_units += reference.len();
        total.sub += e.sub;
        total.del += e.del;
        total.ins += e.ins;
        per_utt.push(rate);
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let corpus = if total_units == 0 { 0.0 } else { total.total() as f64 / total_units as f64 };
    let mean = per_utt.iter().sum::<f64>() / per_utt.len().max(1) as f64;
    let mut sorted = per_utt.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
    let rtfx = if elapsed > 0.0 { audio_s / elapsed } else { 0.0 };
    println!("\n=== {} on {} clips of {} ===",
        if mode == Mode::Cer { "CER" } else { "WER" }, rows.len(), audio_dir.display());
    println!(
        "corpus {:>6.2}%   mean/utt {:>6.2}%   median/utt {:>6.2}%",
        100.0 * corpus, 100.0 * mean, 100.0 * median
    );
    println!(
        "units {}  sub {} del {} ins {}  (total {})",
        total_units, total.sub, total.del, total.ins, total.total()
    );
    println!("elapsed {elapsed:.1}s  audio {audio_s:.1}s  RTFx {rtfx:.2}");
    println!("detected languages: {languages:?}");
    // One machine-readable line so the final table needs no scraping; the Python
    // runner writes the same file name with the same columns.
    if let Some(p) = summary_out.as_ref() {
        let langs: Vec<String> =
            languages.iter().map(|(k, v)| format!("{}={v}", if k.is_empty() { "-" } else { k })).collect();
        let line = format!(
            "system\tconfig\tclips\taudio_s\telapsed_s\trtfx\tload_s\tmetric\tcorpus_pct\tmean_pct\tunits\tsub\tdel\tins\tlanguages\n\
             wgpu\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.1}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{}\t{}\t{}\n",
            tsv.file_stem().and_then(|s| s.to_str()).unwrap_or("?"),
            rows.len(),
            audio_s,
            elapsed,
            rtfx,
            load_s,
            if mode == Mode::Cer { "cer" } else { "wer" },
            100.0 * corpus,
            100.0 * mean,
            total_units,
            total.sub,
            total.del,
            total.ins,
            langs.join(" ")
        );
        std::fs::write(p, line).with_context(|| format!("write {}", p.display()))?;
    }
    if let (Some(f), Some(p)) = (hyps_file.as_mut(), hyps_out.as_ref()) {
        use std::io::Write;
        f.flush().with_context(|| format!("write {}", p.display()))?;
    }
    if let (Some(f), Some(p)) = (lang_file.as_mut(), lang_out.as_ref()) {
        use std::io::Write;
        f.flush().with_context(|| format!("write {}", p.display()))?;
    }
    Ok(())
}

/// Byte-safe truncation for the per-utterance preview.
fn truncate(s: &str, max: usize) -> String {
    let t = s.trim();
    if t.chars().count() <= max {
        return t.to_string();
    }
    let mut out: String = t.chars().take(max).collect();
    out.push('…');
    out
}
