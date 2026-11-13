use std::sync::MutexGuard;

use crate::error::{AsrError, Result};
use crate::inference::{with_forced_language, Inner, StreamToken, TranscribeOptions};
use crate::mel::{HOP_LENGTH, N_FFT};
use crate::prompt::TranscribeResult;

pub(crate) struct StreamState {
    samples: Vec<f32>,
    base: usize,
    frame: usize,
    embeds: Vec<f32>,
    /// One entry per encoded window, in `embeds` order.
    windows: Vec<EncodedWindow>,
    /// Whole-clip log-mel maximum seen so far (grows monotonically).
    max_val: f32,
    n_mels: usize,
    out_dim: usize,
    win_frames: usize,
}

/// Raw log-mel for one encoder window plus what was needed to encode it.
///
/// The whole-clip normalisation maximum is only known at `flush`, so a window
/// encoded earlier may have used a smaller one; keeping the raw values lets
/// `flush` re-normalise and re-encode just those windows.
struct EncodedWindow {
    /// `n_mels * n`, bin-major, already stripped of the left-context frames.
    mel: Vec<f32>,
    n: usize,
    /// The maximum these embeddings were normalised with.
    used_max: f32,
    /// Range in `embeds`, in f32 elements.
    embed_start: usize,
    embed_len: usize,
}

impl StreamState {
    pub(crate) fn new(inner: &Inner) -> Self {
        let ac = &inner.config.thinker_config.audio_config;
        let cs = ac.n_window * 2;
        let win_frames = cs * (ac.n_window_infer / cs).max(1);
        Self {
            samples: Vec::new(),
            base: 0,
            frame: 0,
            embeds: Vec::new(),
            windows: Vec::new(),
            max_val: f32::NEG_INFINITY,
            n_mels: ac.num_mel_bins,
            out_dim: ac.output_dim,
            win_frames,
        }
    }

    pub(crate) fn push(&mut self, inner: &mut Inner, samples: &[f32]) -> anyhow::Result<()> {
        self.samples.extend_from_slice(samples);
        loop {
            let f_last = self.frame + self.win_frames - 1;
            let need_end = f_last * HOP_LENGTH + N_FFT / 2;
            if self.base + self.samples.len() < need_end {
                break;
            }
            self.encode_window(inner, Some(self.win_frames))?;
        }
        Ok(())
    }

    pub(crate) fn flush(
        &mut self,
        inner: &mut Inner,
        opts: &TranscribeOptions,
        forced: Option<String>,
        on_token: &mut dyn FnMut(StreamToken),
    ) -> Result<TranscribeResult> {
        let total_frames = (self.base + self.samples.len()) / HOP_LENGTH;
        if total_frames > self.frame {
            self.encode_window(inner, None).map_err(AsrError::Inference)?;
        }
        self.apply_global_normalisation(inner)
            .map_err(AsrError::Inference)?;
        let embeds = std::mem::take(&mut self.embeds);
        let r = inner
            .decode_from_audio_embeds(
                &embeds,
                opts.max_new_tokens,
                None,
                0.0,
                0.0,
                opts,
                Some(on_token),
            )
            .map_err(AsrError::Inference)?;
        Ok(with_forced_language(r, forced))
    }

    fn encode_window(&mut self, inner: &mut Inner, n_frames: Option<usize>) -> anyhow::Result<()> {
        let f0 = self.frame;
        let want_from = f0 * HOP_LENGTH;
        let slice_start = want_from.saturating_sub(2 * HOP_LENGTH);
        let drop = (want_from - slice_start) / HOP_LENGTH;
        let hi = match n_frames {
            Some(n) => {
                let end = (f0 + n - 1) * HOP_LENGTH + N_FFT / 2;
                anyhow::ensure!(end <= self.base + self.samples.len(), "stream slice past the buffer");
                end - self.base
            }
            None => self.samples.len(),
        };
        let lo = slice_start - self.base;
        let (mel, n_mels, frames) = inner.mel.extract_log_mel(&self.samples[lo..hi])?;
        anyhow::ensure!(n_mels == self.n_mels, "mel bins {n_mels} != {}", self.n_mels);
        let n = match n_frames {
            Some(n) => n,
            None => frames.saturating_sub(drop),
        };
        anyhow::ensure!(n > 0 && frames >= drop + n, "mel slice too short ({frames} frames, need {})", drop + n);

        let mut win = vec![0.0f32; self.n_mels * n];
        for m in 0..self.n_mels {
            let src = m * frames + drop;
            win[m * n..(m + 1) * n].copy_from_slice(&mel[src..src + n]);
        }
        // The offline path normalises against the whole clip's maximum.  Use
        // the largest value seen so far; `flush` re-encodes any window that
        // ended up with a smaller one.
        let chunk_max = win.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        self.max_val = self.max_val.max(chunk_max);
        let mut norm = win.clone();
        inner.mel.normalize_mel_in_place(&mut norm, self.max_val);

        let embeds = inner.run_encoder(&norm, self.n_mels, n, false)?;
        let tokens = embeds.len() / self.out_dim;
        anyhow::ensure!(embeds.len() == tokens * self.out_dim, "encoder output not a whole number of tokens");
        let embed_start = self.embeds.len();
        self.embeds.extend_from_slice(&embeds);
        self.windows.push(EncodedWindow {
            mel: win,
            n,
            used_max: self.max_val,
            embed_start,
            embed_len: embeds.len(),
        });
        self.frame += n;

        let keep_from = (self.frame * HOP_LENGTH).saturating_sub(2 * HOP_LENGTH);
        if keep_from > self.base {
            self.samples.drain(..(keep_from - self.base));
            self.base = keep_from;
        }
        Ok(())
    }

    /// Re-normalise every window against the final whole-clip maximum and
    /// re-encode the ones that were produced with a smaller one.  Windows whose
    /// `used_max` already equals it are left untouched, so a clip whose loudest
    /// moment is early pays nothing.
    fn apply_global_normalisation(&mut self, inner: &mut Inner) -> anyhow::Result<()> {
        let max_val = self.max_val;
        for w in self.windows.iter_mut() {
            if w.used_max == max_val {
                continue;
            }
            let mut mel = w.mel.clone();
            inner.mel.normalize_mel_in_place(&mut mel, max_val);
            let embeds = inner.run_encoder(&mel, self.n_mels, w.n, false)?;
            anyhow::ensure!(
                embeds.len() == w.embed_len,
                "re-encode changed the token count ({} != {})",
                embeds.len(),
                w.embed_len
            );
            self.embeds[w.embed_start..w.embed_start + w.embed_len].copy_from_slice(&embeds);
            w.used_max = max_val;
        }
        Ok(())
    }

    pub(crate) fn sample_count(&self) -> usize {
        self.base + self.samples.len()
    }

    pub(crate) fn encoded_tokens(&self) -> usize {
        self.embeds.len() / self.out_dim
    }
}

/// A streaming ASR session: audio in via [`Self::push_samples`], text at
/// [`Self::flush`].
///
/// Windows are encoded as they complete, and the mel frames are kept so that
/// `flush` can apply the whole clip's normalisation maximum (the one scalar the
/// one-shot path uses) and re-encode only the windows that saw a smaller one.
/// The transcript therefore matches `transcribe` exactly; memory grows with the
/// audio, like the accumulated embeddings.
///
/// Created by [`crate::AsrInference::create_buffered_session`].  It holds the
/// engine's lock for its whole lifetime — the parent [`crate::AsrInference`]
/// must not be used until the session is dropped — so it is `!Send` and stays
/// on the thread that made it.
pub struct AsrBufferedSession<'a> {
    guard: MutexGuard<'a, Inner>,
    opts: TranscribeOptions,
    forced: Option<String>,
    state: StreamState,
}

impl<'a> AsrBufferedSession<'a> {
    pub(crate) fn new(
        guard: MutexGuard<'a, Inner>,
        opts: TranscribeOptions,
        forced: Option<String>,
    ) -> Self {
        let state = StreamState::new(&guard);
        Self { guard, opts, forced, state }
    }

    /// Feed more 16 kHz mono audio; every complete window is encoded on the way in.
    pub fn push_samples(&mut self, samples: &[f32]) -> Result<()> {
        self.state
            .push(&mut self.guard, samples)
            .map_err(AsrError::Inference)
    }

    /// Encode the remaining audio and decode the text.
    pub fn flush(&mut self) -> Result<TranscribeResult> {
        self.flush_streaming(|_| {})
    }

    /// [`Self::flush`] with a per-token callback.
    pub fn flush_streaming<F>(&mut self, mut on_token: F) -> Result<TranscribeResult>
    where
        F: FnMut(StreamToken),
    {
        self.state
            .flush(&mut self.guard, &self.opts, self.forced.clone(), &mut on_token)
    }

    /// 16 kHz samples fed so far.
    pub fn sample_count(&self) -> usize {
        self.state.sample_count()
    }

    /// Audio tokens encoded so far (one per token the audio tower produced).
    pub fn encoded_tokens(&self) -> usize {
        self.state.encoded_tokens()
    }
}
