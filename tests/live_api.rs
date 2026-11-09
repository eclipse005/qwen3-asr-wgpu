//! Live streaming integration checks.  Need weights + a GPU, so they are
//! `#[ignore]`d and gated on `QASR_TEST_MODEL` / `QASR_TEST_WAV` like the rest of
//! the suite.  Everything runs in one test so the `QASR_LIVE_NO_ENC_CACHE` and
//! `QASR_LIVE_NO_KV_REUSE` toggles are never mutated concurrently.

use std::cell::RefCell;
use std::path::PathBuf;

use qwen3_asr_wgpu::{AsrInference, Backend, LiveUpdate, TranscribeOptions, TranscribeResult};

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}

struct LiveRun {
    result: TranscribeResult,
    updates: Vec<LiveUpdate>,
    max_windows: usize,
    /// Updates emitted strictly by `push_samples`, before `flush`.
    pre_flush_updates: usize,
    /// Committed-text length (bytes) after the last pre-flush update.
    pre_flush_stable_len: usize,
}

fn run_live(
    asr: &AsrInference,
    samples: &[f32],
    chunk: usize,
    opts: &TranscribeOptions,
) -> LiveRun {
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
    let pre_flush_updates = updates.borrow().len();
    let pre_flush_stable_len = updates.borrow().last().map_or(0, |u| u.stable.len());
    let result = sess.flush().expect("flush");
    LiveRun {
        result,
        updates: updates.into_inner(),
        max_windows,
        pre_flush_updates,
        pre_flush_stable_len,
    }
}

/// Character-level Levenshtein distance (for the live-vs-offline drift bound).
fn char_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
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

    // Baseline: encoder window cache on, KV prefix reuse on, 2 s pushes.
    std::env::remove_var("QASR_LIVE_NO_ENC_CACHE");
    std::env::remove_var("QASR_LIVE_NO_KV_REUSE");
    let base = run_live(&asr, &samples, 32_000, &opts);
    assert!(!base.result.text.is_empty(), "live produced no text");
    assert!(
        base.max_windows <= 4,
        "encoder cache grew past the eviction cap: {}",
        base.max_windows
    );

    // Text really streams: `push_samples` commits text on its own, not only the
    // final `flush`.  (The C port guards this with a "≥2 Commit:" stderr check.)
    assert!(
        base.pre_flush_updates >= 2,
        "only {} updates before flush — the streaming loop did not run incrementally",
        base.pre_flush_updates
    );
    assert!(
        base.pre_flush_stable_len > 0,
        "no committed text before flush — everything was emitted at the end"
    );
    assert!(
        base.updates.len() >= 3,
        "only {} total updates",
        base.updates.len()
    );

    // Committed text never retracts.
    for pair in base.updates.windows(2) {
        assert!(
            pair[1].stable.starts_with(&pair[0].stable),
            "stable text retracted:\n  was {:?}\n  now {:?}",
            pair[0].stable,
            pair[1].stable
        );
    }
    // The callback stream agrees with the returned result.
    if let Some(last) = base.updates.last() {
        assert_eq!(
            last.stable.trim(),
            base.result.text.trim(),
            "last committed text differs from the returned transcript"
        );
    }

    // Push granularity does not change the result (the round grid is fixed).
    let fine = run_live(&asr, &samples, 16_000, &opts);
    assert_eq!(
        fine.result.text, base.result.text,
        "1 s vs 2 s pushes drifted"
    );
    let whole = run_live(&asr, &samples, samples.len().max(1), &opts);
    assert_eq!(
        whole.result.text, base.result.text,
        "whole-file push drifted from chunked"
    );

    // Encoder cache on/off is numerically equivalent while no window is evicted
    // (<= ~32 s): both encode the same windows with the same per-window mel
    // frontend.  Beyond that, eviction deliberately drops audio context, so the
    // transcripts legitimately diverge — that trade-off is checked separately.
    let short = &samples[..samples.len().min(30 * 16_000)];
    std::env::set_var("QASR_LIVE_NO_ENC_CACHE", "1");
    let no_cache = run_live(&asr, short, 32_000, &opts);
    std::env::remove_var("QASR_LIVE_NO_ENC_CACHE");
    let cache_short = run_live(&asr, short, 32_000, &opts);
    assert_eq!(
        no_cache.result.text, cache_short.result.text,
        "encoder cache changed the transcript (no eviction)"
    );

    // With no eviction the live path must stay close to the one-shot decode:
    // this checks that the windowed encoder + streaming decoder stay aligned
    // with the offline path (frontend normalisation and window spans).  Exact
    // equality is not guaranteed — rollback can lock in a different phrasing at
    // a chunk boundary ("4-hour" vs "four-hour") — so bound the drift instead.
    let offline_short = asr
        .transcribe_samples(short, opts.clone())
        .expect("offline short");
    let dist = char_distance(&cache_short.result.text, &offline_short.text);
    let max_len = cache_short
        .result
        .text
        .chars()
        .count()
        .max(offline_short.text.chars().count())
        .max(1);
    assert!(
        dist * 100 <= max_len * 5,
        "live vs offline (no eviction) drifted {dist}/{max_len} chars (> 5%):\n  live    {:?}\n  offline {:?}",
        cache_short.result.text,
        offline_short.text
    );

    // KV-prefix reuse must be numerically equivalent to a full re-prefill.
    std::env::set_var("QASR_LIVE_NO_KV_REUSE", "1");
    let no_reuse = run_live(&asr, &samples, 32_000, &opts);
    std::env::remove_var("QASR_LIVE_NO_KV_REUSE");
    assert_eq!(
        no_reuse.result.text, base.result.text,
        "KV prefix reuse changed the transcript"
    );

    // Informational: full-length comparison.  Past ~32 s eviction drops audio
    // context, so a small divergence from the one-shot decode is expected.
    let offline = asr.transcribe_samples(&samples, opts).expect("offline");
    eprintln!("live    : {}", base.result.text);
    eprintln!("offline : {}", offline.text);
    eprintln!("live == offline: {}", base.result.text == offline.text);
}
