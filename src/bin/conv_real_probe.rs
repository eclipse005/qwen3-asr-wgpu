//! Real-data conv-stem probe: runs the *actual* `CpuAudioEncoder` on real
//! weights and mel, sweeping the conv batch tile size.  Synthetic probes reach
//! ~0.9 TFLOP/s on the same GEMM shape while the encoder's conv buckets show
//! far less, so this isolates real-data effects (denormals, cache pressure,
//! per-tile allocation) from the GEMM kernel itself.
//!
//! ```text
//! $env:QWEN3_CONV_TILES="8,16,32,72"
//! cargo run --release --bin conv_real_probe
//! ```
//!
//! No GPU work: `WgpuAsr::load` creates the Vulkan device and uploads the text
//! decoder, but this binary never encodes or submits a command.

use std::time::Instant;

use anyhow::Result;
use qwen3_asr_wgpu::audio_encoder::CONV_TILE;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| r"D:\Qwen3-ASR\models\Qwen3-ASR-0.6B-hf".to_string());
    let wav = args
        .iter()
        .position(|a| a == "--wav")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| r"D:\qwen3-asr-rs\tests\fixtures\180s_en.wav".to_string());

    println!("loading (no GPU submits)...");
    let mut asr = qwen3_asr_wgpu::WgpuAsr::load_with(
        std::path::Path::new(&model),
        None,
        qwen3_asr_wgpu::inference::EncoderBackend::Cpu,
    )?;
    let samples = qwen3_asr_wgpu::mel::load_audio_wav(std::path::Path::new(&wav), 16_000)?;
    let (mel, n_mels, n_frames) = asr.extract_mel(&samples)?;
    println!("wav {wav}: mel {n_mels}x{n_frames}");
    let cs = 100usize;
    println!(
        "chunk plan: cs={cs} feo(cs)={} feo(n_mels)={} feo^2(n_mels)={} feo^3(n_mels)={}",
        qwen3_asr_wgpu::audio_encoder::feo(cs),
        qwen3_asr_wgpu::audio_encoder::feo(n_mels),
        qwen3_asr_wgpu::audio_encoder::feo(qwen3_asr_wgpu::audio_encoder::feo(n_mels)),
        qwen3_asr_wgpu::audio_encoder::feo(qwen3_asr_wgpu::audio_encoder::feo(qwen3_asr_wgpu::audio_encoder::feo(n_mels))),
    );
    println!(
        "conv_out k = {} (expect c3_out * f3 * t3)",
        asr.conv_out_in_features()
    );
    println!("cpu conv geometry (c,h,w)x3 = {:?}", asr.conv_geometry(n_mels)?);

    // ── hand-check a few output values against the CPU reference ──
    // At `(ho=0, wo=0)` the c1 tap table leaves only taps 4 (`kh=1,kw=1`) and
    // 7 (`kh=2,kw=1`) in bounds, so that output is a 2-term dot product that can
    // be recomputed independently — which settles whether a GPU/CPU mismatch
    // lives in the GEMM or in the reference.
    hand_check(&asr, &mel, n_mels, n_frames)?;

    let tiles: Vec<usize> = match std::env::var("QWEN3_CONV_TILES") {
        Ok(v) => v.split(',').filter_map(|s| s.trim().parse().ok()).collect(),
        Err(_) => vec![8],
    };
    for tile in tiles {
        CONV_TILE.store(tile as u64, std::sync::atomic::Ordering::Relaxed);
        let mut times = Vec::new();
        let mut checksum = 0.0f64;
        for _ in 0..3 {
            let t = Instant::now();
            let out = asr.encode_mel(&mel, n_mels, n_frames)?;
            times.push(t.elapsed().as_secs_f64() * 1e3);
            checksum = out.iter().map(|v| *v as f64).sum();
        }
        println!(
            "TILE={tile:<4} passes: {}   (sum {checksum:.4})",
            times.iter().map(|v| format!("{v:>7.1} ms")).collect::<Vec<_>>().join(" ")
        );
    }
    Ok(())
}

/// Recompute some `c1` outputs from first principles and compare against the
/// crate's CPU reference.
///
/// At `(ho=0, wo=0)` the 3×3/s2/p1 tap table leaves only the taps with
/// `kh >= 1 && kw >= 1` in bounds, so that output is a short dot product that can
/// be recomputed independently — which settles whether a GPU/CPU mismatch lives
/// in the GEMM or in the reference.
fn hand_check(
    asr: &qwen3_asr_wgpu::WgpuAsr,
    mel: &[f32],
    n_mels: usize,
    n_frames: usize,
) -> anyhow::Result<()> {
    let (w, c_out, k) = asr.conv_weight_c1()?;
    anyhow::ensure!(k == 9, "expected 9 taps, got {k}");
    let bias = asr.conv_bias_c1()?;
    let (raw, _c_out, pos, _chunks) = asr.conv_reference_c1(&mel, n_mels, n_frames)?;

    // input plane is [n_mels][n_frames] for chunk 0; row ih = 2*0 + kh - 1
    let mut terms: Vec<(usize, f32)> = Vec::new();
    for kh in 0..3usize {
        for kw in 0..3usize {
            let ih = kh as isize - 1;
            let iw = kw as isize - 1;
            if ih < 0 || iw < 0 {
                continue;
            }
            terms.push((kh * 3 + kw, mel[ih as usize * n_frames + iw as usize]));
        }
    }
    println!(
        "hand-check (ho=0,wo=0): {} in-bounds taps {:?}",
        terms.len(),
        terms.iter().map(|(t, _)| *t).collect::<Vec<_>>()
    );
    for c in [0usize, 1, 5] {
        let mut acc = 0.0f32;
        for (t, v) in &terms {
            acc += w[c * k + t] * v;
        }
        let cu = acc + bias[c];
        let g = gelu_ref(cu);
        let r = raw[c * pos]; // [c_out][pos] layout, pos 0
        println!(
            "  c={c}: pre-gelu {cu:+.6} -> gelu {g:+.6} | cpu ref {r:+.6} | {}",
            if (g - r).abs() < 1e-4 { "AGREE" } else { "*** DIFFER ***" }
        );
    }
    Ok(())
}

/// `F.gelu` (erf form), matching the CPU reference.
fn gelu_ref(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}
