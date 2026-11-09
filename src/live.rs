//! True streaming decoding: audio goes in, committed text comes out while the
//! audio is still arriving.
//!
//! This is the live counterpart to [`crate::AsrStreamingSession`] (which only
//! encodes incrementally and decodes once at `flush`, reproducing the offline
//! result exactly).  Here every ~2 s of audio triggers a decode round whose
//! committed text is emitted through a callback; memory stays bounded on
//! arbitrarily long input.
//!
//! The round structure follows the C port's `stream_impl`, which in turn mirrors
//! the official vLLM streaming policy (`unfixed_chunk_num=2`,
//! `unfixed_token_num=5`):
//!
//! * completed encoder windows are encoded once and cached (sliding eviction
//!   keeps the last few); only the partial tail window is re-encoded per round;
//! * the decoder is re-prefilled each round with the previous raw tokens minus a
//!   short rollback, reusing the longest unchanged KV prefix (only the delta
//!   rows are prefilled; the prefill causal mask honours `kv_start`);
//! * text is committed token-wise against the previous candidate (longest common
//!   prefix + overlap de-duplication), so committed output never retracts;
//! * degenerate / stagnant streams re-anchor to a short committed tail.

use std::collections::VecDeque;
use std::sync::MutexGuard;

use half::f16;

use crate::error::{AsrError, Result};
use crate::inference::{with_forced_language, Inner, TranscribeOptions};
use crate::mel::{HOP_LENGTH, N_FFT};
use crate::prompt::{
    self, StopReason, TranscribeResult, ASR_TEXT_SEP_TOKEN_ID, ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID,
};

/// Audio per decode round (2 s at 16 kHz), matching the reference cadence.
const CHUNK_SAMPLES: usize = 32_000;
/// Tokens rolled back from the previous raw output before re-decoding.
const ROLLBACK_TOKENS: usize = 5;
/// Cold-start rounds that neither roll back nor commit text.
const UNFIXED_CHUNKS: usize = 2;
/// Generation budget per non-final round.
const CHUNK_MAX_NEW: usize = 32;
/// Completed encoder windows kept before the oldest is evicted (~32 s).
const MAX_ENC_WINDOWS: usize = 4;
/// Decoder text prefix cap (the tail of the rolled-back raw tokens).
const MAX_PREFIX_TOKENS: usize = 150;
/// Same-token run longer than this is suppressed within a round.
const MAX_REPEAT_RUN: usize = 12;
/// Overlap de-duplication window (tokens) when committing a delta.
const OVERLAP_MIN: usize = 4;
const OVERLAP_MAX: usize = 48;
/// Tail periodicity (period cap / minimum repeats) that signals degeneration.
const DEGEN_MAX_PERIOD: usize = 6;
const DEGEN_MIN_REPEATS: usize = 4;
/// Consecutive stagnant rounds before a recovery re-anchor.
const STALE_CHUNKS: usize = 4;
/// Periodic re-anchor cadence (rounds) and committed-tail carry.
const RESET_INTERVAL: usize = 45;
const RESET_CARRY: usize = 24;

/// One incremental streaming update, emitted after each decode round that
/// changes the visible text.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct LiveUpdate {
    /// Cumulative committed text.  Grows monotonically — once emitted it is
    /// never taken back.
    pub stable: String,
    /// The current unfixed tail (may be revised by the next round).  Empty on
    /// the final `flush`.
    pub partial: String,
}

/// A live streaming session: feed 16 kHz mono audio with [`Self::push_samples`]
/// and receive committed text through the callback given at creation; call
/// [`Self::flush`] once at end of audio for the final [`TranscribeResult`].
///
/// Holds the engine lock for its whole lifetime (like
/// [`crate::AsrStreamingSession`]), so it is `!Send` and stays on its thread.
pub struct AsrLiveSession<'a, F: FnMut(LiveUpdate)> {
    guard: MutexGuard<'a, Inner>,
    opts: TranscribeOptions,
    forced: Option<String>,
    on_update: F,
    state: LiveState,
    finished: bool,
}

impl<'a, F: FnMut(LiveUpdate)> AsrLiveSession<'a, F> {
    pub(crate) fn new(
        guard: MutexGuard<'a, Inner>,
        opts: TranscribeOptions,
        forced: Option<String>,
        on_update: F,
    ) -> Self {
        let state = LiveState::new(&guard);
        Self {
            guard,
            opts,
            forced,
            on_update,
            state,
            finished: false,
        }
    }

    /// Feed more audio; every complete 2 s grid point runs a decode round and
    /// may emit a [`LiveUpdate`].  Output depends only on the audio, not on how
    /// it is chunked across calls.
    pub fn push_samples(&mut self, samples: &[f32]) -> Result<()> {
        self.ensure_open()?;
        self.state
            .push(
                &mut self.guard,
                samples,
                &self.opts,
                self.forced.as_deref(),
                &mut self.on_update,
            )
            .map_err(AsrError::Inference)
    }

    /// Run the final round over all audio and return the complete result.
    pub fn flush(&mut self) -> Result<TranscribeResult> {
        self.ensure_open()?;
        self.finished = true;
        let r = self
            .state
            .flush(
                &mut self.guard,
                &self.opts,
                self.forced.as_deref(),
                &mut self.on_update,
            )
            .map_err(AsrError::Inference)?;
        Ok(with_forced_language(r, self.forced.clone()))
    }

    /// 16 kHz samples fed so far.
    pub fn sample_count(&self) -> usize {
        self.state.base + self.state.samples.len()
    }

    /// Completed encoder windows currently cached (bounded by design).
    pub fn cached_windows(&self) -> usize {
        self.state.windows.len()
    }

    fn ensure_open(&self) -> Result<()> {
        if self.finished {
            return Err(AsrError::Inference(anyhow::anyhow!(
                "live session already flushed"
            )));
        }
        Ok(())
    }
}

struct LiveState {
    samples: Vec<f32>,
    base: usize,
    frame: usize,
    windows: VecDeque<Vec<f32>>,
    win_frames: usize,
    n_mels: usize,
    no_cache: bool,

    raw: Vec<u32>,
    stable: Vec<u32>,
    emitted: Vec<u32>,
    prev_hidden: Vec<u8>,
    kv_reuse: bool,
    chunk_idx: usize,
    next_cursor: usize,
    stagnant: usize,
    stop_reason: StopReason,
    last_emitted_len: usize,
    last_partial: String,
    language: String,
}

impl LiveState {
    fn new(inner: &Inner) -> Self {
        let ac = &inner.config.thinker_config.audio_config;
        let cs = ac.n_window * 2;
        let win_frames = cs * (ac.n_window_infer / cs).max(1);
        let no_cache = std::env::var("QASR_LIVE_NO_ENC_CACHE")
            .map(|v| v != "0")
            .unwrap_or(false);
        let kv_reuse = std::env::var("QASR_LIVE_NO_KV_REUSE").map(|v| v != "1").unwrap_or(true);
        Self {
            samples: Vec::new(),
            base: 0,
            frame: 0,
            windows: VecDeque::new(),
            win_frames,
            n_mels: ac.num_mel_bins,
            no_cache,
            raw: Vec::new(),
            stable: Vec::new(),
            emitted: Vec::new(),
            prev_hidden: Vec::new(),
            kv_reuse,
            chunk_idx: 0,
            next_cursor: CHUNK_SAMPLES,
            stagnant: 0,
            stop_reason: StopReason::Eos,
            last_emitted_len: 0,
            last_partial: String::new(),
            language: String::new(),
        }
    }

    fn push<F: FnMut(LiveUpdate)>(
        &mut self,
        inner: &mut Inner,
        samples: &[f32],
        opts: &TranscribeOptions,
        forced: Option<&str>,
        on_update: &mut F,
    ) -> anyhow::Result<()> {
        self.samples.extend_from_slice(samples);
        let total = self.base + self.samples.len();
        while self.next_cursor <= total {
            let cursor = self.next_cursor;
            self.round(inner, opts, forced, cursor, false, on_update)?;
            self.next_cursor += CHUNK_SAMPLES;
        }
        Ok(())
    }

    fn flush<F: FnMut(LiveUpdate)>(
        &mut self,
        inner: &mut Inner,
        opts: &TranscribeOptions,
        forced: Option<&str>,
        on_update: &mut F,
    ) -> anyhow::Result<TranscribeResult> {
        let total = self.base + self.samples.len();
        if total == 0 {
            return Ok(TranscribeResult {
                text: String::new(),
                language: String::new(),
                raw_output: String::new(),
                stop_reason: StopReason::Eos,
            });
        }
        self.round(inner, opts, forced, total, true, on_update)?;
        let raw_output = decode_ids(inner, &self.raw)?;
        let (mut language, _) = prompt::parse_asr_output(&raw_output);
        if language.is_empty() {
            language = self.language.clone();
        }
        let text = decode_ids(inner, &self.emitted)?.trim().to_string();
        Ok(TranscribeResult {
            text,
            language,
            raw_output,
            stop_reason: self.stop_reason,
        })
    }

    fn round<F: FnMut(LiveUpdate)>(
        &mut self,
        inner: &mut Inner,
        opts: &TranscribeOptions,
        forced: Option<&str>,
        cursor: usize,
        is_final: bool,
        on_update: &mut F,
    ) -> anyhow::Result<()> {
        let enc = self.encode(inner, cursor)?;
        let hs = inner.config.thinker_config.text_config.hidden_size;
        let nat = enc.len() / hs;

        let prefix_full = if self.chunk_idx >= UNFIXED_CHUNKS && !self.raw.is_empty() {
            self.raw.len().saturating_sub(ROLLBACK_TOKENS)
        } else {
            0
        };
        let prefix_start = prefix_full.saturating_sub(MAX_PREFIX_TOKENS);
        let prefix_ids: Vec<i64> = self.raw[prefix_start..prefix_full]
            .iter()
            .map(|&t| t as i64)
            .collect();

        let tc = &inner.config.thinker_config;
        let (input_ids, asp) = prompt::build_prompt(
            &inner.tokenizer,
            tc.audio_start_token_id,
            tc.audio_token_id,
            tc.audio_end_token_id,
            nat,
            forced,
            &opts.context,
            Some(&prefix_ids),
        )?;
        let s = input_ids.len();
        let budget = if is_final {
            opts.max_new_tokens
        } else {
            CHUNK_MAX_NEW
        };
        anyhow::ensure!(
            s + budget + 8 <= inner.decoder.max_seq(),
            "live seq {s} + max_new {budget} exceeds decoder max_seq {}",
            inner.decoder.max_seq()
        );
        inner.decoder.ensure_capacity(s + budget + 8);

        let et = inner
            .tensors
            .get("thinker.model.embed_tokens.weight")
            .ok_or_else(|| anyhow::anyhow!("missing embed_tokens weight"))?;
        let mut hidden: Vec<u8> = Vec::with_capacity(s * hs * 2);
        for &id in &input_ids[..asp] {
            et.append_f16_row_le(id as usize, hs, &mut hidden)?;
        }
        for v in &enc {
            hidden.extend_from_slice(&f16::from_f32(*v).to_le_bytes());
        }
        for &id in &input_ids[asp + nat..] {
            et.append_f16_row_le(id as usize, hs, &mut hidden)?;
        }

        // Reuse the longest unchanged prefill prefix: the cached KV rows are
        // identical to what a full re-prefill would recompute, so only the delta
        // rows are prefilled (`kv_start > 0`).
        let row = hs * 2;
        let reuse = if self.kv_reuse {
            common_rows(&self.prev_hidden, &hidden, row).min(s - 1)
        } else {
            0
        };
        let first = inner.decoder.prefill(&hidden[reuse * row..], s - reuse, reuse)?;
        if self.kv_reuse {
            self.prev_hidden = hidden;
        }

        let eos = [ENDOFTEXT_TOKEN_ID as i32, IM_END_TOKEN_ID as i32];
        let mut token = first;
        let mut new_tokens: Vec<u32> = Vec::new();
        let mut hit_eos = false;
        loop {
            if eos.contains(&token) {
                hit_eos = true;
                break;
            }
            new_tokens.push(token as u32);
            if new_tokens.len() >= budget {
                break;
            }
            token = inner.decoder.step()?;
        }
        if is_final {
            self.stop_reason = if hit_eos {
                StopReason::Eos
            } else {
                StopReason::MaxTokens
            };
        }

        let (kept, dropped) = suppress_runs(&self.raw[..prefix_full], &new_tokens, MAX_REPEAT_RUN);
        self.raw.truncate(prefix_full);
        self.raw.extend_from_slice(&kept);

        let text_start = if forced.is_some() {
            0
        } else {
            self.raw
                .iter()
                .position(|&t| t == ASR_TEXT_SEP_TOKEN_ID)
                .map(|p| p + 1)
                .unwrap_or(0)
        };
        if forced.is_none() && self.language.is_empty() && text_start > 0 {
            let head = decode_ids(inner, &self.raw[..text_start])?;
            self.language = prompt::parse_asr_output(&head).0;
        }
        let n_text = self.raw.len() - text_start;
        let cand_len = candidate_len(n_text, self.chunk_idx, is_final);
        let candidate = &self.raw[text_start..text_start + cand_len];

        if std::env::var("QASR_LIVE_DEBUG").is_ok() {
            eprintln!(
                "[live] chunk={} final={} cursor={} nat={} reuse={} prefix={} new={} raw={} tstart={} cand={} emit={} eos={} lang={}",
                self.chunk_idx, is_final, cursor, nat, reuse, prefix_full,
                new_tokens.len(), self.raw.len(), text_start, cand_len, self.emitted.len(), hit_eos, self.language
            );
        }

        let (reps, period) = tail_repeat_blocks(candidate, DEGEN_MAX_PERIOD);
        let advance = cand_len as i64 - self.stable.len() as i64;
        if !is_final && new_tokens.len() >= budget && advance <= 1 {
            self.stagnant += 1;
        } else {
            self.stagnant = 0;
        }
        let recovery = (period > 0 && reps >= DEGEN_MIN_REPEATS)
            || self.stagnant >= STALE_CHUNKS
            || dropped >= 8;

        let mut did_reset = recovery;
        if recovery {
            self.reanchor(forced.is_some());
        } else {
            let l = lcp(&self.stable, candidate);
            self.stable.clear();
            self.stable.extend_from_slice(candidate);
            let emit_start = overlap_skip(&self.emitted, candidate, l);
            self.emitted.extend_from_slice(&candidate[emit_start..]);

            if !is_final
                && self.chunk_idx >= UNFIXED_CHUNKS
                && (self.chunk_idx + 1) % RESET_INTERVAL == 0
            {
                self.reanchor(forced.is_some());
                did_reset = true;
            }
        }

        if !did_reset {
            let stable_text = decode_ids(inner, &self.emitted)?;
            let partial = if is_final {
                String::new()
            } else {
                decode_ids(inner, &self.raw[text_start + cand_len..])?
            };
            if self.emitted.len() != self.last_emitted_len || partial != self.last_partial {
                on_update(LiveUpdate {
                    stable: stable_text,
                    partial: partial.clone(),
                });
                self.last_emitted_len = self.emitted.len();
                self.last_partial = partial;
            }
        }

        self.chunk_idx += 1;
        Ok(())
    }

    fn encode(&mut self, inner: &mut Inner, cursor: usize) -> anyhow::Result<Vec<f32>> {
        if self.no_cache {
            // Debug fallback: recompute the same windows + partial every round
            // with the identical per-window frontend (no caching, no eviction),
            // so cache on/off is numerically — not merely textually — equivalent.
            let mut enc = Vec::new();
            let mut f = 0;
            loop {
                let need_end = (f + self.win_frames - 1) * HOP_LENGTH + N_FFT / 2;
                if need_end > cursor {
                    break;
                }
                enc.extend_from_slice(&encode_span(
                    inner,
                    &self.samples,
                    0,
                    f,
                    need_end,
                    Some(self.win_frames),
                    self.n_mels,
                )?);
                f += self.win_frames;
            }
            if cursor > f * HOP_LENGTH {
                enc.extend_from_slice(&encode_span(
                    inner,
                    &self.samples,
                    0,
                    f,
                    cursor,
                    None,
                    self.n_mels,
                )?);
            }
            return Ok(enc);
        }
        loop {
            let f_last = self.frame + self.win_frames - 1;
            let need_end = f_last * HOP_LENGTH + N_FFT / 2;
            if need_end > cursor {
                break;
            }
            let e = encode_span(
                inner,
                &self.samples,
                self.base,
                self.frame,
                need_end,
                Some(self.win_frames),
                self.n_mels,
            )?;
            self.windows.push_back(e);
            self.frame += self.win_frames;
            while self.windows.len() > MAX_ENC_WINDOWS {
                self.windows.pop_front();
            }
        }
        let keep_from = (self.frame * HOP_LENGTH).saturating_sub(2 * HOP_LENGTH);
        if keep_from > self.base {
            self.samples.drain(..keep_from - self.base);
            self.base = keep_from;
        }
        let mut enc = Vec::new();
        for w in &self.windows {
            enc.extend_from_slice(w);
        }
        if cursor > self.frame * HOP_LENGTH {
            let p = encode_span(
                inner,
                &self.samples,
                self.base,
                self.frame,
                cursor,
                None,
                self.n_mels,
            )?;
            enc.extend_from_slice(&p);
        }
        Ok(enc)
    }

    fn reanchor(&mut self, forced: bool) {
        let carry = self.emitted.len().min(RESET_CARRY);
        let tail = self.emitted[self.emitted.len() - carry..].to_vec();
        self.raw.clear();
        if !forced {
            self.raw.push(ASR_TEXT_SEP_TOKEN_ID);
        }
        self.raw.extend_from_slice(&tail);
        self.stable.clear();
        self.stable.extend_from_slice(&tail);
        self.windows.clear();
        self.prev_hidden.clear();
        self.stagnant = 0;
    }
}

/// Encode `[f0, end_abs)` mel frames (with the 2-hop left context the conv stem
/// needs) into audio embeddings.  `n_frames = Some` pins a complete window;
/// `None` takes whatever the slice yields (the partial tail).
fn encode_span(
    inner: &mut Inner,
    samples: &[f32],
    base: usize,
    f0: usize,
    end_abs: usize,
    n_frames: Option<usize>,
    n_mels: usize,
) -> anyhow::Result<Vec<f32>> {
    let want_from = f0 * HOP_LENGTH;
    let slice_start = want_from.saturating_sub(2 * HOP_LENGTH);
    let drop = (want_from - slice_start) / HOP_LENGTH;
    let lo = slice_start - base;
    let hi = end_abs - base;
    let (mel, nm, frames) = inner.mel.extract(&samples[lo..hi])?;
    anyhow::ensure!(nm == n_mels, "mel bins {nm} != {n_mels}");
    let n = match n_frames {
        Some(n) => n,
        None => frames.saturating_sub(drop),
    };
    anyhow::ensure!(
        n > 0 && frames >= drop + n,
        "mel slice too short ({frames} frames)"
    );
    let mut win = vec![0.0f32; n_mels * n];
    for m in 0..n_mels {
        let src = m * frames + drop;
        win[m * n..(m + 1) * n].copy_from_slice(&mel[src..src + n]);
    }
    inner.run_encoder(&win, n_mels, n, false)
}

fn decode_ids(inner: &Inner, ids: &[u32]) -> anyhow::Result<String> {
    inner
        .tokenizer
        .decode(ids, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))
}

/// Longest common prefix of two equal-width byte buffers, counted in rows.
fn common_rows(a: &[u8], b: &[u8], row: usize) -> usize {
    if row == 0 {
        return 0;
    }
    let n = (a.len() / row).min(b.len() / row);
    let mut i = 0;
    while i < n && a[i * row..(i + 1) * row] == b[i * row..(i + 1) * row] {
        i += 1;
    }
    i
}

/// Longest common prefix of two token slices.
fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Skip a leading run of `candidate[emit_start..]` that already ends `emitted`,
/// so a re-generated continuation is not committed twice.
fn overlap_skip(emitted: &[u32], candidate: &[u32], emit_start: usize) -> usize {
    if emit_start >= candidate.len() || emitted.is_empty() {
        return emit_start;
    }
    let max_overlap = (candidate.len() - emit_start)
        .min(emitted.len())
        .min(OVERLAP_MAX);
    for k in (OVERLAP_MIN..=max_overlap).rev() {
        if emitted[emitted.len() - k..] == candidate[emit_start..emit_start + k] {
            return emit_start + k;
        }
    }
    emit_start
}

/// Suppress same-token runs longer than `max_run`, continuing the run counted
/// from the tail of `prev`.  Returns the kept tokens and how many were dropped.
fn suppress_runs(prev: &[u32], new: &[u32], max_run: usize) -> (Vec<u32>, usize) {
    let mut out = Vec::with_capacity(new.len());
    let mut dropped = 0;
    let (mut prev_tok, mut prev_run) = match prev.last() {
        Some(&last) => {
            let mut run = 1;
            for &t in prev.iter().rev().skip(1) {
                if t != last {
                    break;
                }
                run += 1;
                if run >= max_run {
                    break;
                }
            }
            (Some(last), run)
        }
        None => (None, 0),
    };
    for &tok in new {
        let suppress = if Some(tok) == prev_tok {
            prev_run += 1;
            prev_run > max_run
        } else {
            prev_tok = Some(tok);
            prev_run = 1;
            false
        };
        if suppress {
            dropped += 1;
        } else {
            out.push(tok);
        }
    }
    (out, dropped)
}

/// Best tail periodicity: the largest number of repeated blocks (period capped
/// at `max_period`) ending the token slice.  Mirrors the C degeneration probe.
fn tail_repeat_blocks(tokens: &[u32], max_period: usize) -> (usize, usize) {
    let n = tokens.len();
    if n < 2 {
        return (1, 0);
    }
    let mut best_reps = 1;
    let mut best_period = 0;
    let period_cap = (n / 2).min(max_period);
    for p in 1..=period_cap {
        let mut reps = 1;
        while (reps + 1) * p <= n {
            let a = &tokens[n - (reps + 1) * p..n - reps * p];
            let b = &tokens[n - reps * p..n - (reps - 1) * p];
            if a != b {
                break;
            }
            reps += 1;
        }
        if reps > best_reps {
            best_reps = reps;
            best_period = p;
        }
    }
    (best_reps, best_period)
}

fn candidate_len(n_text: usize, chunk_idx: usize, is_final: bool) -> usize {
    if is_final {
        return n_text;
    }
    if chunk_idx >= UNFIXED_CHUNKS {
        let mut c = n_text as i64 - ROLLBACK_TOKENS as i64;
        if c <= 0 && n_text > 0 {
            c = n_text as i64 - 1;
        }
        c.max(0) as usize
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcp_basic() {
        assert_eq!(lcp(&[1, 2, 3], &[1, 2, 4]), 2);
        assert_eq!(lcp(&[], &[1]), 0);
        assert_eq!(lcp(&[1, 2], &[1, 2, 3]), 2);
    }

    #[test]
    fn common_rows_counts_whole_rows() {
        let a = [1u8, 2, 3, 4, 5, 6];
        let b = [1u8, 2, 3, 4, 9, 9];
        assert_eq!(common_rows(&a, &b, 2), 2);
        assert_eq!(common_rows(&a, &a, 2), 3);
        assert_eq!(common_rows(&[], &a, 2), 0);
    }

    #[test]
    fn overlap_skips_regenerated_suffix() {
        let emitted = [1, 2, 3, 4, 5, 6];
        let candidate = [9, 3, 4, 5, 6, 7, 8];
        // emit_start 1; candidate[1..5] == emitted tail [3,4,5,6] -> skip 4.
        assert_eq!(overlap_skip(&emitted, &candidate, 1), 5);
        assert_eq!(overlap_skip(&[], &candidate, 0), 0);
        assert_eq!(overlap_skip(&emitted, &[1, 2], 2), 2);
    }

    #[test]
    fn overlap_respects_min_window() {
        let emitted = [1, 2, 3];
        let candidate = [2, 3, 9];
        // only 2 tokens overlap (< OVERLAP_MIN) -> no skip.
        assert_eq!(overlap_skip(&emitted, &candidate, 0), 0);
    }

    #[test]
    fn suppress_continues_prefix_run() {
        let prev = [7, 7, 7];
        let new = [7, 7, 8, 8];
        let (kept, dropped) = suppress_runs(&prev, &new, 4);
        // prev run 3; first new 7 reaches run 4 (== max, kept), second reaches
        // 5 (> max, dropped); the 8s start a fresh run.
        assert_eq!(kept, vec![7, 8, 8]);
        assert_eq!(dropped, 1);
    }

    #[test]
    fn suppress_no_prev() {
        let (kept, dropped) = suppress_runs(&[], &[1, 1, 1], 2);
        assert_eq!(kept, vec![1, 1]);
        assert_eq!(dropped, 1);
    }

    #[test]
    fn tail_repeat_finds_period() {
        let t = [1, 2, 3, 1, 2, 3, 1, 2, 3];
        let (reps, period) = tail_repeat_blocks(&t, 6);
        assert_eq!(period, 3);
        assert_eq!(reps, 3);
        assert_eq!(tail_repeat_blocks(&[5], 6), (1, 0));
    }

    #[test]
    fn candidate_len_cold_and_rollback() {
        assert_eq!(candidate_len(10, 0, false), 0);
        assert_eq!(candidate_len(10, 1, false), 0);
        assert_eq!(candidate_len(10, 2, false), 5);
        assert_eq!(candidate_len(3, 2, false), 2); // short text keeps 1 unfixed
        assert_eq!(candidate_len(10, 2, true), 10);
    }
}
