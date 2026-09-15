use std::path::Path;

use crate::error::{AsrError, Result};
use crate::inference::{supported_languages, AsrInference, TranscribeOptions};
use crate::mel::{load_audio_wav, HOP_LENGTH, MEL_SAMPLE_RATE};
use crate::prompt;

/// `return_format` of the reference's `processor.decode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnFormat {
    Raw,
    Parsed,
    TranscriptionOnly,
}

/// The parsed dict (`_parse_single_output`): `language` and
/// `transcription`.  `language` is empty when the language was forced — the
/// metadata then lives in the *prompt*, exactly like the reference.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AsrTranscription {
    pub language: String,
    pub transcription: String,
}

/// What [`AsrInference::decode`] returns, one variant per `return_format`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decoded {
    Raw(String),
    Parsed(AsrTranscription),
    TranscriptionOnly(String),
}

impl Decoded {
    /// The transcribed text, whichever format was asked for (`Raw` is the raw
    /// string, which still carries the `language …<asr_text>` prefix).
    pub fn transcription_text(&self) -> &str {
        match self {
            Decoded::Raw(s) => s,
            Decoded::Parsed(t) => &t.transcription,
            Decoded::TranscriptionOnly(s) => s,
        }
    }

    /// The detected language, when the format carries one.
    pub fn language(&self) -> Option<&str> {
        match self {
            Decoded::Parsed(t) if !t.language.is_empty() => Some(&t.language),
            _ => None,
        }
    }
}

/// What `processor.apply_transcription_request` hands to `model.generate`: the
/// log-mel features plus the request's options.
///
/// The reference puts `input_ids` in here as well; ours are built inside
/// [`AsrInference::generate`] because the number of audio placeholder tokens is
/// the audio tower's *output* length, and deriving it a second time (from the mel
/// length) is exactly the kind of duplicated geometry this design avoids.
#[derive(Debug, Clone)]
pub struct TranscriptionRequest {
    /// `[n_mels, n_frames]` f32, the same layout the tower's im2col reads.
    pub mel: Vec<f32>,
    pub n_mels: usize,
    pub n_frames: usize,
    /// `language` + `prompt`/context, i.e. [`TranscribeOptions`] — including the
    /// `max_new_tokens` [`AsrInference::generate`] will use.
    pub options: TranscribeOptions,
}

impl TranscriptionRequest {
    /// The sample rate the features were computed at (`MEL_SAMPLE_RATE`).
    pub fn sample_rate(&self) -> u32 {
        MEL_SAMPLE_RATE
    }

    /// Duration of the audio the features cover, in seconds — the reference's
    /// `audio_s`, which the gold harness uses to size `max_new_tokens`.
    pub fn audio_seconds(&self) -> f64 {
        (self.n_frames as f64) * (HOP_LENGTH as f64) / (MEL_SAMPLE_RATE as f64)
    }
}

impl AsrInference {
    /// `processor.apply_transcription_request(audio=…, language=…, prompt=…)`
    /// for in-memory samples at 16 kHz.
    pub fn apply_transcription_request(
        &self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<TranscriptionRequest> {
        let (mel, n_mels, n_frames) = crate::diagnostics::extract_mel(self, samples)?;
        Ok(TranscriptionRequest { mel, n_mels, n_frames, options: options.clone() })
    }

    /// The same, reading a wav file (any sample rate; resampled exactly like
    /// every other entry point).
    pub fn apply_transcription_request_file(
        &self,
        wav: &Path,
        options: &TranscribeOptions,
    ) -> Result<TranscriptionRequest> {
        let samples = load_audio_wav(wav, MEL_SAMPLE_RATE)?;
        self.apply_transcription_request(&samples, options)
    }

    /// `model.generate(**inputs, do_sample=False)`.
    ///
    /// Runs the audio tower, builds the prompt and the mixed hidden states, then
    /// prefill + one greedy step per token (stopping at the EOS set, like
    /// `generate`).  Returns **only the generated ids** — the
    /// reference slices `output_ids[:, input_ids.shape[1]:]` before `decode`.
    ///
    /// `max_new_tokens` is read from [`TranscriptionRequest::options`], so the
    /// ceiling is fixed where the request is built, exactly as
    /// [`AsrInference::transcribe`] fixes it from its own options.
    pub fn generate(&self, req: &TranscriptionRequest) -> Result<Vec<u32>> {
        let mut g = self.lock()?;
        let embeds = g
            .encode_mel(&req.mel, req.n_mels, req.n_frames)
            .map_err(AsrError::Inference)?;
        Ok(g
            .generate_from_embeds(&embeds, req.options.max_new_tokens, &req.options, None)
            .map_err(AsrError::Inference)?
            .ids)
    }

    /// `processor.decode(ids, return_format=…)`.
    pub fn decode(&self, generated_ids: &[u32], format: ReturnFormat) -> Result<Decoded> {
        let g = self.lock()?;
        match format {
            ReturnFormat::Raw => Ok(Decoded::Raw(
                g.decode_ids(generated_ids, false).map_err(AsrError::Inference)?,
            )),
            ReturnFormat::Parsed => {
                let text = g.decode_ids(generated_ids, true).map_err(AsrError::Inference)?;
                let (language, transcription) = prompt::parse_asr_output(&text);
                Ok(Decoded::Parsed(AsrTranscription { language, transcription }))
            }
            ReturnFormat::TranscriptionOnly => {
                let text = g.decode_ids(generated_ids, true).map_err(AsrError::Inference)?;
                let (_, transcription) = prompt::parse_asr_output(&text);
                Ok(Decoded::TranscriptionOnly(transcription))
            }
        }
    }

    /// `processor.get_supported_languages()` — the 30 canonical names, i.e. the
    /// values `language=` accepts (codes work too).
    pub fn get_supported_languages(&self) -> &'static [&'static str] {
        supported_languages()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, DeviceSelector};

    #[test]
    #[ignore = "needs a GPU and the -hf weights (see the doc comment)"]
    fn processor_api_matches_the_one_call_wrapper() {
        let (Ok(model), Ok(wav)) = (
            std::env::var("QASR_TEST_MODEL"),
            std::env::var("QASR_TEST_WAV"),
        ) else {
            eprintln!("QASR_TEST_MODEL / QASR_TEST_WAV not set — skipping");
            return;
        };
        let wav = std::path::PathBuf::from(wav);
        let opts = TranscribeOptions::default().with_max_new_tokens(512);
        let asr = AsrInference::load_on(
            std::path::Path::new(&model),
            DeviceSelector::parse("nvidia").expect("selector"),
        )
        .expect("load decoder");

        let req = asr
            .apply_transcription_request_file(&wav, &opts)
            .expect("apply_transcription_request");
        assert_eq!(req.n_mels, 128);
        assert!(req.audio_seconds() > 0.0);
        let ids = asr.generate(&req).expect("generate");
        let parsed = asr.decode(&ids, ReturnFormat::Parsed).expect("decode parsed");
        let only = asr
            .decode(&ids, ReturnFormat::TranscriptionOnly)
            .expect("decode transcription_only");
        let raw = asr.decode(&ids, ReturnFormat::Raw).expect("decode raw");

        assert_eq!(parsed.transcription_text(), only.transcription_text());
        assert!(!parsed.transcription_text().is_empty());
        assert!(parsed.language().is_some(), "auto-detected language should be set");
        assert!(raw.transcription_text().contains("<asr_text>"));

        let one_call = asr
            .transcribe(wav.to_str().expect("utf-8 path"), opts.clone())
            .expect("transcribe");
        assert_eq!(one_call.text, parsed.transcription_text());
        assert_eq!(Some(one_call.language.as_str()), parsed.language());
        assert_eq!(asr.get_supported_languages().len(), 30);
        assert_eq!(Backend::best().tag(), "auto");
    }
}
