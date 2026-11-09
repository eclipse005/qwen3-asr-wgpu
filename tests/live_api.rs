//! Live streaming integration checks.  Need weights + a GPU, so they are
//! `#[ignore]`d and gated on `QASR_TEST_MODEL` / `QASR_TEST_WAV` like the rest of
//! the suite.  Everything runs in one test so the `QASR_LIVE_NO_ENC_CACHE` env
//! toggle is never mutated concurrently.

use std::cell::RefCell;
use std::path::PathBuf;

use qwen3_asr_wgpu::{AsrInference, Backend, LiveUpdate, TranscribeOptions, TranscribeResult};

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}

fn run_live(
    asr: &AsrInference,
    samples: &[f32],
    chunk: usize,
    opts: &TranscribeOptions,
) -> (TranscribeResult, Vec<LiveUpdate>, usize) {
    let updates: RefCell<Vec<LiveUpdate>> = RefCell::new(Vec::new());
    let mut sess = asr
        .create_live_session(opts.clone(), |u| updates.borrow_mut().push(u))
        .expect("create live session");
    let mut max_windows = 0;
    for part in samples.chunks(chunk) {
        sess.push_samples(part).expect("push");
        max_windows = max_windows.max(sess.cached_windows());
    }
    assert_eq!(sess.sample_count(), samples.len());
    let r = sess.flush().expect("flush");
    (r, updates.into_inner(), max_windows)
}

#[test]
#[ignore = "needs a GPU and weights (QASR_TEST_MODEL / QASR_TEST_WAV)"]
fn live_streaming_invariants() {
    let (Some(model), Some(wav)) = (env_path("QASR_TEST_MODEL"), env_path("QASR_TEST_WAV")) else {
        eprintln!("QASR_TEST_MODEL / QASR_TEST_WAV not set — skipping");
        return;
    };
    let asr = AsrInference::load(&model, Backend::Auto).expect("load");
    let samples = qwen3_asr_wgpu::load_audio_wav(&wav, 16_000).expect("wav");
    let opts = TranscribeOptions::default().with_max_new_tokens(512);

    // Baseline: encoder window cache on, 2 s pushes.
    std::env::remove_var("QASR_LIVE_NO_ENC_CACHE");
    let (base, updates, max_windows) = run_live(&asr, &samples, 32_000, &opts);
    assert!(!base.text.is_empty(), "live produced no text");
    assert!(
        max_windows <= 4,
        "encoder cache grew past the eviction cap: {max_windows}"
    );

    // Committed text never retracts.
    for pair in updates.windows(2) {
        assert!(
            pair[1].stable.starts_with(&pair[0].stable),
            "stable text retracted:\n  was {:?}\n  now {:?}",
            pair[0].stable,
            pair[1].stable
        );
    }
    if let Some(last) = updates.last() {
        assert!(last.partial.is_empty() || base.text.contains(&last.stable));
    }

    // Push granularity does not change the result (the round grid is fixed).
    let (fine, _, _) = run_live(&asr, &samples, 16_000, &opts);
    assert_eq!(fine.text, base.text, "1 s vs 2 s pushes drifted");
    let (whole, _, _) = run_live(&asr, &samples, samples.len().max(1), &opts);
    assert_eq!(
        whole.text, base.text,
        "whole-file push drifted from chunked"
    );

    // Encoder cache on/off is numerically equivalent while no window is evicted
    // (<= ~32 s): both encode the same windows with the same per-window mel
    // frontend.  Beyond that, eviction deliberately drops audio context, so the
    // transcripts legitimately diverge — that trade-off is checked separately.
    let short = &samples[..samples.len().min(30 * 16_000)];
    std::env::set_var("QASR_LIVE_NO_ENC_CACHE", "1");
    let (no_cache, _, _) = run_live(&asr, short, 32_000, &opts);
    std::env::remove_var("QASR_LIVE_NO_ENC_CACHE");
    let (cache_short, _, _) = run_live(&asr, short, 32_000, &opts);
    assert_eq!(
        no_cache.text, cache_short.text,
        "encoder cache changed the transcript (no eviction)"
    );

    // Informational: how close the live transcript is to the offline one.
    let offline = asr.transcribe_samples(&samples, opts).expect("offline");
    eprintln!("live    : {}", base.text);
    eprintln!("offline : {}", offline.text);
    eprintln!("live == offline: {}", base.text == offline.text);
}
