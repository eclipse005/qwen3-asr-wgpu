use std::path::{Path, PathBuf};

use qwen3_asr_wgpu::{
    diagnostics, AsrError, AsrInference, Backend, DeviceSelector, EncoderBackend,
    TranscribeOptions,
};

#[test]
fn inference_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AsrInference>();
}

#[test]
fn backend_variants_are_stable() {
    assert_eq!(Backend::default(), Backend::Auto);
    assert_eq!(Backend::best(), Backend::Auto);
    assert_eq!(Backend::best().tag(), "auto");
    assert_eq!(Backend::Cpu.tag(), "cpu");
    assert_eq!(Backend::Gpu.tag(), "gpu");
}

#[test]
fn supported_languages_is_the_upstream_list() {
    assert_eq!(qwen3_asr_wgpu::supported_languages().len(), 30);
    assert!(qwen3_asr_wgpu::supported_languages().contains(&"Chinese"));
}

#[test]
fn resolve_model_dir_prefers_the_hf_layout() {
    let dir = qwen3_asr_wgpu::resolve_model_dir(Path::new("C:/nonexistent"), "0.6B");
    assert!(dir.ends_with("Qwen3-ASR-0.6B"), "{}", dir.display());
}

fn baseline_body() -> Option<String> {
    let Ok(path) = std::env::var("QASR_TEST_BASELINE") else {
        return None;
    };
    let path = PathBuf::from(path);
    if !path.is_file() {
        eprintln!("baseline {} not found — skipping the frozen-text check", path.display());
        return None;
    }
    let raw = std::fs::read_to_string(&path).ok()?.replace("\r\n", "\n");
    Some(raw.split("\n\n").nth(1).unwrap_or("").trim().to_string())
}

fn test_model() -> PathBuf {
    PathBuf::from(std::env::var("QASR_TEST_MODEL").expect("QASR_TEST_MODEL"))
}

fn test_wav() -> PathBuf {
    PathBuf::from(std::env::var("QASR_TEST_WAV").expect("QASR_TEST_WAV"))
}

fn load() -> AsrInference {
    AsrInference::load_on(
        &test_model(),
        DeviceSelector::parse("nvidia").expect("selector"),
    )
    .expect("load on nvidia")
}

fn load_or_skip() -> Option<AsrInference> {
    if std::env::var("QASR_TEST_MODEL").is_err() || std::env::var("QASR_TEST_WAV").is_err() {
        eprintln!("QASR_TEST_MODEL / QASR_TEST_WAV not set — skipping");
        return None;
    }
    Some(load())
}

#[test]
#[ignore = "needs a GPU and the -hf weights"]
fn transcribe_matches_the_frozen_baseline() {
    let Some(asr) = load_or_skip() else { return };
    let wav = test_wav();
    let out = asr
        .transcribe(wav.to_str().expect("utf-8 path"), TranscribeOptions::default().with_max_new_tokens(512))
        .expect("transcribe");
    assert!(!out.text.is_empty());
    if let Some(want) = baseline_body() {
        assert_eq!(out.text.trim(), want, "transcript drifted from python-hf");
    }
}

#[test]
#[ignore = "needs a GPU and the -hf weights"]
fn forced_language_is_reported_and_validated() {
    let Some(asr) = load_or_skip() else { return };
    let wav = test_wav();
    let path = wav.to_str().expect("utf-8 path");

    let by_name = asr
        .transcribe(path, TranscribeOptions::default().with_language("Chinese"))
        .expect("forced by name");
    assert_eq!(by_name.language, "Chinese");

    let by_code = asr
        .transcribe(path, TranscribeOptions::default().with_language("zh"))
        .expect("forced by code");
    assert_eq!(by_code.language, "Chinese", "ISO code should resolve to the full name");
    assert_eq!(by_code.text, by_name.text, "code and name must prompt identically");

    let bad = asr.transcribe(path, TranscribeOptions::default().with_language("Klingon"));
    assert!(
        matches!(bad, Err(AsrError::InvalidOptions(_))),
        "expected InvalidOptions, got {bad:?}"
    );

    let zero = asr.transcribe(path, TranscribeOptions::default().with_max_new_tokens(0));
    assert!(
        matches!(zero, Err(AsrError::InvalidOptions(_))),
        "expected InvalidOptions, got {zero:?}"
    );
}

#[test]
#[ignore = "needs a GPU and the -hf weights"]
fn auto_language_is_the_models_own() {
    let Some(asr) = load_or_skip() else { return };
    let samples = qwen3_asr_wgpu::load_audio_wav(&test_wav(), 16_000).expect("wav");
    let out = asr
        .transcribe_samples(&samples, TranscribeOptions::default().with_max_new_tokens(512))
        .expect("transcribe_samples");
    assert!(!out.text.is_empty());
    assert!(!out.language.is_empty(), "auto-detect should name a language");
}

#[test]
#[ignore = "needs the -hf weights (no GPU required)"]
fn cpu_backend_transcribes() {
    if std::env::var("QASR_TEST_MODEL").is_err() || std::env::var("QASR_TEST_WAV").is_err() {
        eprintln!("QASR_TEST_MODEL / QASR_TEST_WAV not set — skipping");
        return;
    }
    let asr = AsrInference::load(&test_model(), Backend::Cpu).expect("load cpu");
    assert!(!asr.gpu_encoder_active());
    assert!(asr.device().is_none(), "the CPU backend owns no adapter");
    let out = asr
        .transcribe(
            test_wav().to_str().expect("utf-8 path"),
            TranscribeOptions::default().with_max_new_tokens(512),
        )
        .expect("transcribe");
    assert!(!out.text.is_empty());
}

#[test]
#[ignore = "needs a GPU and the -hf weights"]
fn diagnostics_and_streaming_agree_with_the_one_call_path() {
    let Some(asr) = load_or_skip() else { return };
    let wav = test_wav();
    let opts = TranscribeOptions::default().with_max_new_tokens(512);

    let one_call = asr
        .transcribe(wav.to_str().expect("utf-8 path"), opts.clone())
        .expect("transcribe");

    let samples = qwen3_asr_wgpu::load_audio_wav(&wav, 16_000).expect("wav");
    let (mel, n_mels, n_frames) = diagnostics::extract_mel(&asr, &samples).expect("mel");
    assert_eq!(n_mels, 128);
    let from_mel = diagnostics::transcribe_from_mel(
        &asr, &mel, n_mels, n_frames, &opts, None, false,
    )
    .expect("transcribe from mel");
    assert_eq!(from_mel.text, one_call.text, "mel-in path drifted");

    let mut session = asr.create_streaming_session(opts).expect("session");
    for part in samples.chunks(16_000) {
        session.push_samples(part).expect("push");
    }
    assert_eq!(session.sample_count(), samples.len());
    let streamed = session.flush().expect("flush");
    assert_eq!(streamed.text, one_call.text, "streamed text drifted");

    let host_tower = AsrInference::load_with(
        &test_model(),
        DeviceSelector::parse("nvidia").expect("selector"),
        EncoderBackend::Cpu,
    )
    .expect("load with the host tower");
    assert!(!host_tower.gpu_encoder_active());
    let host_out = host_tower
        .transcribe(
            wav.to_str().expect("utf-8 path"),
            TranscribeOptions::default().with_max_new_tokens(512),
        )
        .expect("transcribe on the host tower");
    assert_eq!(host_out.text, one_call.text, "host tower drifted from the GPU one");
}

#[test]
#[ignore = "needs a GPU and the -hf weights"]
fn one_instance_serves_two_threads() {
    let Some(asr) = load_or_skip() else { return };
    let asr = std::sync::Arc::new(asr);
    let wav = test_wav();
    let path = wav.to_str().expect("utf-8 path").to_string();

    let handles: Vec<_> = (0..2)
        .map(|_| {
            let asr = std::sync::Arc::clone(&asr);
            let path = path.clone();
            std::thread::spawn(move || {
                let mut r = None;
                for _ in 0..1 {
                    r = Some(
                        asr.transcribe(&path, TranscribeOptions::default().with_max_new_tokens(64))
                            .expect("transcribe"),
                    );
                }
                r.expect("result").text
            })
        })
        .collect();

    let texts: Vec<String> = handles.into_iter().map(|h| h.join().expect("join")).collect();
    assert_eq!(texts[0], texts[1], "concurrent calls disagreed");
}
