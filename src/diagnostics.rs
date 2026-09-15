use std::path::Path;

use crate::error::{AsrError, Result};
use crate::inference::{AsrInference, TranscribeOptions};
use crate::prompt::TranscribeResult;

/// [`AsrInference::transcribe`] plus the two harness hooks: a directory to dump
/// every intermediate tensor into, and a CPU-vs-GPU comparison of the audio
/// tower.
pub fn transcribe_with_dump(
    asr: &AsrInference,
    wav: &Path,
    opts: &TranscribeOptions,
    dump_dir: Option<&Path>,
    compare_enc: bool,
) -> Result<TranscribeResult> {
    let mut g = asr.lock()?;
    g.transcribe_file_opts(wav, opts.max_new_tokens, dump_dir, compare_enc, opts)
        .map_err(AsrError::Inference)
}

/// [`transcribe_with_dump`] with a per-token callback — the two hooks and the
/// streaming shape at once.
pub fn transcribe_with_dump_streaming<F>(
    asr: &AsrInference,
    wav: &Path,
    opts: &TranscribeOptions,
    dump_dir: Option<&Path>,
    compare_enc: bool,
    on_token: F,
) -> Result<TranscribeResult>
where
    F: FnMut(crate::inference::StreamToken),
{
    let mut g = asr.lock()?;
    g.transcribe_file_streaming(wav, opts.max_new_tokens, dump_dir, compare_enc, opts, on_token)
        .map_err(AsrError::Inference)
}

/// Transcribe precomputed mel features, with the same two hooks.
pub fn transcribe_from_mel(
    asr: &AsrInference,
    mel: &[f32],
    n_mels: usize,
    n_frames: usize,
    opts: &TranscribeOptions,
    dump_dir: Option<&Path>,
    compare_enc: bool,
) -> Result<TranscribeResult> {
    let mut g = asr.lock()?;
    g.transcribe_from_mel_cmp_opts(
        mel,
        n_mels,
        n_frames,
        opts.max_new_tokens,
        dump_dir,
        compare_enc,
        opts,
    )
    .map_err(AsrError::Inference)
}

/// Decode from already-encoded audio embeddings — skips the tower entirely,
/// which is what makes it useful for isolating a decoder bug.
pub fn transcribe_from_embeds(
    asr: &AsrInference,
    audio_embeds: &[f32],
    opts: &TranscribeOptions,
    dump_dir: Option<&Path>,
) -> Result<TranscribeResult> {
    let mut g = asr.lock()?;
    g.transcribe_from_embeds(audio_embeds, opts.max_new_tokens, dump_dir)
        .map_err(AsrError::Inference)
}

/// The mel frontend alone — 16 kHz samples to `(mel, n_mels, n_frames)`.
pub fn extract_mel(asr: &AsrInference, samples: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
    asr.lock()?.extract_mel(samples).map_err(AsrError::Inference)
}

/// The audio tower alone — mel to audio-token embeddings, as host `f32`.
///
/// This is the host tower; the GPU one is reached through
/// [`transcribe_with_dump`]'s `compare_enc` or [`diagnose_encoder_mel`].
pub fn encode_mel(
    asr: &AsrInference,
    mel: &[f32],
    n_mels: usize,
    n_frames: usize,
) -> Result<Vec<f32>> {
    asr.lock()?
        .encode_mel(mel, n_mels, n_frames)
        .map_err(AsrError::Inference)
}

/// Audio-tower `(c, h, w)` per conv layer for this mel width.
pub fn conv_geometry(asr: &AsrInference, n_mels: usize) -> Result<[usize; 9]> {
    asr.lock()?
        .conv_geometry(n_mels)
        .map_err(AsrError::Inference)
}

/// `conv_out`'s input width (see `audio_encoder.rs`'s permute).
pub fn conv_out_in_features(asr: &AsrInference) -> usize {
    asr.lock().map(|g| g.conv_out_in_features()).unwrap_or(0)
}

/// `conv2d1` bias.
pub fn conv_bias_c1(asr: &AsrInference) -> Result<Vec<f32>> {
    asr.lock()?.conv_bias_c1().map_err(AsrError::Inference)
}

/// `conv2d1` weight as `(data, c_out, taps)`, row-major `[c_out, kh*3+kw]`.
pub fn conv_weight_c1(asr: &AsrInference) -> Result<(Vec<f32>, usize, usize)> {
    asr.lock()?.conv_weight_c1().map_err(AsrError::Inference)
}

/// Raw `conv2d1` output, `[chunk][c_out][pos]`.
pub fn conv_reference_c1(
    asr: &AsrInference,
    mel: &[f32],
    n_mels: usize,
    n_frames: usize,
) -> Result<(Vec<f32>, usize, usize, usize)> {
    asr.lock()?
        .conv_reference_c1(mel, n_mels, n_frames)
        .map_err(AsrError::Inference)
}

/// Stage-by-stage GPU-vs-CPU comparison of the audio tower for one wav.
///
/// Prints to stderr; needs a GPU text decoder and the GPU tower (both are the
/// default).
pub fn diagnose_encoder(asr: &AsrInference, wav: &Path) -> Result<()> {
    asr.lock()?.diagnose_encoder(wav).map_err(AsrError::Inference)
}

/// [`diagnose_encoder`] from precomputed mel features.
pub fn diagnose_encoder_mel(
    asr: &AsrInference,
    mel: &[f32],
    n_mels: usize,
    n_frames: usize,
) -> Result<()> {
    asr.lock()?
        .diagnose_encoder_mel(mel, n_mels, n_frames)
        .map_err(AsrError::Inference)
}
