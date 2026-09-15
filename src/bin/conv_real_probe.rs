use std::time::Instant;

use anyhow::Result;
use qwen3_asr_wgpu::audio_encoder::CONV_TILE;
use qwen3_asr_wgpu::diagnostics;
use qwen3_asr_wgpu::{AsrInference, DeviceSelector, EncoderBackend};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |name: &str, env: &str| -> Result<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .or_else(|| std::env::var(env).ok().filter(|s| !s.is_empty()))
            .ok_or_else(|| anyhow::anyhow!("{name} is required (or set {env})"))
    };
    let model = get("--model", "QASR_MODEL")?;
    let wav = get("--wav", "QASR_WAV")?;

    println!("loading (no GPU submits)...");
    let asr = AsrInference::load_with(
        std::path::Path::new(&model),
        DeviceSelector::Auto,
        EncoderBackend::Cpu,
    )?;
    let samples = qwen3_asr_wgpu::mel::load_audio_wav(std::path::Path::new(&wav), 16_000)?;
    let (mel, n_mels, n_frames) = diagnostics::extract_mel(&asr, &samples)?;
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
        diagnostics::conv_out_in_features(&asr)
    );
    println!(
        "cpu conv geometry (c,h,w)x3 = {:?}",
        diagnostics::conv_geometry(&asr, n_mels)?
    );

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
            let out = diagnostics::encode_mel(&asr, &mel, n_mels, n_frames)?;
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

fn hand_check(
    asr: &AsrInference,
    mel: &[f32],
    n_mels: usize,
    n_frames: usize,
) -> anyhow::Result<()> {
    let (w, _c_out, k) = diagnostics::conv_weight_c1(asr)?;
    anyhow::ensure!(k == 9, "expected 9 taps, got {k}");
    let bias = diagnostics::conv_bias_c1(asr)?;
    let (raw, _c_out, pos, _chunks) = diagnostics::conv_reference_c1(asr, mel, n_mels, n_frames)?;

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
        let r = raw[c * pos];
        println!(
            "  c={c}: pre-gelu {cu:+.6} -> gelu {g:+.6} | cpu ref {r:+.6} | {}",
            if (g - r).abs() < 1e-4 { "AGREE" } else { "*** DIFFER ***" }
        );
    }
    Ok(())
}

fn gelu_ref(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}
