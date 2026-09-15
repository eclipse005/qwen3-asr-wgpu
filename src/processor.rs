//! The reference pipeline's API, mirrored — `transformers`' `Qwen3ASRProcessor`
//! plus `Qwen3ASRForConditionalGeneration.generate`.
//!
//! Upstream is three steps, and a caller coming from the Python side should find
//! the same three here:
//!
//! ```python
//! processor = AutoProcessor.from_pretrained(model_dir)
//! inputs    = processor.apply_transcription_request(audio=wav, language=None, prompt="hotwords")
//! out_ids   = model.generate(**inputs, max_new_tokens=512, do_sample=False)
//! gen_ids   = out_ids[:, inputs["input_ids"].shape[1]:]
//! parsed    = processor.decode(gen_ids, return_format="parsed")[0]   # {"language", "transcription"}
//! raw       = processor.decode(gen_ids, return_format="raw")[0]
//! ```
//!
//! ```no_run
//! # use qwen3_asr_wgpu::{WgpuAsr, TranscribeOptions, ReturnFormat};
//! # fn main() -> anyhow::Result<()> {
//! let mut asr = WgpuAsr::load(std::path::Path::new("model-dir"), Some("nvidia"))?;
//! let opts = TranscribeOptions::default();
//!
//! // processor.apply_transcription_request(audio=…)
//! let req = asr.apply_transcription_request_file(std::path::Path::new("clip.wav"), &opts)?;
//! // model.generate(**inputs, max_new_tokens=…)
//! let ids = asr.generate(&req, 512)?;
//! // processor.decode(ids, return_format="parsed")
//! let parsed = asr.decode(&ids, ReturnFormat::Parsed)?;
//! assert!(!parsed.transcription_text().is_empty());
//! # Ok(())
//! # }
//! ```
//!
//! | reference (Python) | here | notes |
//! |---|---|---|
//! | `Qwen3ASRProcessor.apply_transcription_request(audio, language, prompt)` | [`WgpuAsr::apply_transcription_request`] / [`WgpuAsr::apply_transcription_request_file`] | `prompt` is the hotword/system context — [`TranscribeOptions::context`] |
//! | `Qwen3ASRForConditionalGeneration.generate(**inputs, max_new_tokens, do_sample=False)` | [`WgpuAsr::generate`] | always greedy; the ids are the *generated* tail, like `out_ids[:, prompt_len:]` |
//! | `Qwen3ASRProcessor.decode(ids, return_format=…)` | [`WgpuAsr::decode`] + [`Decoded`] | all three formats, same semantics |
//! | `Qwen3ASRProcessor.get_supported_languages()` | [`WgpuAsr::get_supported_languages`] | the same 30 canonical names |
//! | `return_time_stamps=True` (needs `Qwen3-ForcedAligner`) | — | a second model, not ported; [`AsrTranscription`] has no time stamps |
//!
//! The audio tower runs inside [`WgpuAsr::generate`] (upstream encodes the audio
//! inside the model too); the request carries the mel features the processor
//! produced, not yet-encoded embeddings.

use anyhow::Result;

use crate::inference::{supported_languages, TranscribeOptions, WgpuAsr};
use crate::mel::{load_audio_wav, HOP_LENGTH, MEL_SAMPLE_RATE};
use crate::prompt;

/// `return_format` of the reference's `processor.decode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnFormat {
    /// The tokenizer's raw decode, special tokens included.
    Raw,
    /// `{"language": …, "transcription": …}`, exactly upstream's dict.
    Parsed,
    /// Just the text after `<asr_text>`.
    TranscriptionOnly,
}

/// Upstream's parsed dict (`_parse_single_output`): `language` and
/// `transcription`.  `language` is empty when the language was forced — the
/// metadata then lives in the *prompt*, exactly like the reference.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AsrTranscription {
    pub language: String,
    pub transcription: String,
}

/// What [`WgpuAsr::decode`] returns, one variant per `return_format`.
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
/// Upstream puts `input_ids` in here as well; ours are built inside
/// [`WgpuAsr::generate`] because the number of audio placeholder tokens is the
/// audio tower's *output* length, and deriving it a second time (from the mel
/// length) is exactly the kind of duplicated geometry this port avoids.
#[derive(Debug, Clone)]
pub struct TranscriptionRequest {
    /// `[n_mels, n_frames]` f32, the same layout the tower's im2col reads.
    pub mel: Vec<f32>,
    pub n_mels: usize,
    pub n_frames: usize,
    /// `language` + `prompt`/context, i.e. [`TranscribeOptions`].
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

impl WgpuAsr {
    /// `processor.apply_transcription_request(audio=…, language=…, prompt=…)`
    /// for in-memory samples at 16 kHz.
    pub fn apply_transcription_request(
        &self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<TranscriptionRequest> {
        let (mel, n_mels, n_frames) = self.extract_mel(samples)?;
        Ok(TranscriptionRequest { mel, n_mels, n_frames, options: options.clone() })
    }

    /// The same, reading a wav file (any sample rate; resampled exactly like
    /// every other entry point, vendored soxr HQ).
    pub fn apply_transcription_request_file(
        &self,
        wav: &std::path::Path,
        options: &TranscribeOptions,
    ) -> Result<TranscriptionRequest> {
        let samples = load_audio_wav(wav, MEL_SAMPLE_RATE)?;
        self.apply_transcription_request(&samples, options)
    }

    /// `model.generate(**inputs, max_new_tokens=…, do_sample=False)`.
    ///
    /// Runs the audio tower, builds the prompt and the mixed hidden states, then
    /// prefill + one greedy step per token (stopping at the EOS set, like
    /// upstream's `generate`).  Returns **only the generated ids** — the
    /// reference slices `output_ids[:, input_ids.shape[1]:]` before `decode`.
    pub fn generate(&mut self, req: &TranscriptionRequest, max_new_tokens: usize) -> Result<Vec<u32>> {
        let embeds = self.encode_mel(&req.mel, req.n_mels, req.n_frames)?;
        Ok(self
            .generate_from_embeds(&embeds, max_new_tokens, &req.options, None)?
            .ids)
    }

    /// `processor.decode(ids, return_format=…)`.
    pub fn decode(&self, generated_ids: &[u32], format: ReturnFormat) -> Result<Decoded> {
        match format {
            // Upstream leaves `skip_special_tokens` at its default here, which is
            // why the raw string still shows `<|im_start|>assistant …`.
            ReturnFormat::Raw => Ok(Decoded::Raw(self.decode_ids(generated_ids, false)?)),
            ReturnFormat::Parsed => {
                let (language, transcription) = prompt::parse_asr_output(&self.decode_ids(generated_ids, true)?);
                Ok(Decoded::Parsed(AsrTranscription { language, transcription }))
            }
            ReturnFormat::TranscriptionOnly => {
                let (_, transcription) = prompt::parse_asr_output(&self.decode_ids(generated_ids, true)?);
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

    /// The three-step reference API and the one-call wrapper share every line
    /// after the processor, so they must agree exactly — this is the test that
    /// says so.  It needs a GPU and the `-hf` weights, hence `#[ignore]`:
    ///
    /// ```text
    /// QASR_TEST_MODEL=D:\Qwen3-ASR\models\Qwen3-ASR-0.6B-hf
    /// QASR_TEST_WAV=D:\qwen3-asr-rs\tests\fixtures\15s_en.wav
    /// cargo test --release -- --ignored upstream_api
    /// ```
    #[test]
    #[ignore = "needs a GPU and the -hf weights (see the doc comment)"]
    fn upstream_api_matches_the_one_call_wrapper() {
        let (Ok(model), Ok(wav)) = (
            std::env::var("QASR_TEST_MODEL"),
            std::env::var("QASR_TEST_WAV"),
        ) else {
            eprintln!("QASR_TEST_MODEL / QASR_TEST_WAV not set — skipping");
            return;
        };
        let wav = std::path::PathBuf::from(wav);
        let opts = TranscribeOptions::default();
        let mut asr = WgpuAsr::load(std::path::Path::new(&model), Some("nvidia")).expect("load decoder");

        // processor.apply_transcription_request → model.generate → processor.decode
        let req = asr
            .apply_transcription_request_file(&wav, &opts)
            .expect("apply_transcription_request");
        assert_eq!(req.n_mels, 128);
        assert!(req.audio_seconds() > 0.0);
        let ids = asr.generate(&req, 512).expect("generate");
        let parsed = asr.decode(&ids, ReturnFormat::Parsed).expect("decode parsed");
        let only = asr
            .decode(&ids, ReturnFormat::TranscriptionOnly)
            .expect("decode transcription_only");
        let raw = asr.decode(&ids, ReturnFormat::Raw).expect("decode raw");

        assert_eq!(parsed.transcription_text(), only.transcription_text());
        assert!(!parsed.transcription_text().is_empty());
        assert!(parsed.language().is_some(), "auto-detected language should be set");
        // `raw` is the tokenizer's decode with special tokens: it still carries
        // the metadata the parsed form strips.
        assert!(raw.transcription_text().contains("<asr_text>"));

        let one_call = asr
            .transcribe_file_opts(&wav, 512, None, false, &opts)
            .expect("transcribe_file_opts");
        assert_eq!(one_call.text, parsed.transcription_text());
        assert_eq!(Some(one_call.language.as_str()), parsed.language());
        assert_eq!(asr.get_supported_languages().len(), 30);
    }
}
