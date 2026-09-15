use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use half::f16;
use tokenizers::Tokenizer;

use crate::audio_encoder::CpuAudioEncoder;
use crate::audio_encoder_gpu::GpuAudioEncoder;
use crate::backend::Backend;
use crate::config::AsrConfig;
use crate::decoder::{TextConfig, WgpuTextDecoder};
use crate::gpu::{DeviceSelector, Gpu};
use crate::mel::{load_audio_wav, MelExtractor, HOP_LENGTH, MEL_SAMPLE_RATE, N_FFT};
use crate::mrope::{compute_mrope_cos_sin, text_positions};
use crate::prompt::{self, TranscribeResult, ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID};
use crate::weights;

const DECODER_MAX_SEQ: usize = 16384;

/// Which audio-tower implementation a [`WgpuAsr`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderBackend {
    Cpu,
    Gpu,
}

/// One incremental decode event: the token id the decoder just produced and the
/// raw text so far.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct StreamToken {
    /// The token id the decoder just produced.
    pub token_id: u32,
    /// Raw decoding of every token generated so far — not parsed (no
    /// `language X<asr_text>` split, no repetition fix), so a caller sees the
    /// text grow exactly as the reference's callback does.
    pub text_so_far: String,
}

/// The languages the forced-language suffix accepts, in the reference's order
/// (`qwen_asr.inference.utils.SUPPORTED_LANGUAGES` / the processor's
/// `LANGUAGE_CODE_TO_NAME` values — the two lists are the same 30 names).
pub fn supported_languages() -> &'static [&'static str] {
    &prompt::SUPPORTED_LANGUAGES
}

/// Pick a local checkpoint directory for `"0.6B"` / `"1.7B"`.
///
/// Order: the `QWEN3_ASR_MODEL_06_DIR` / `QWEN3_ASR_MODEL_17_DIR` environment
/// override, then `root/models/Qwen3-ASR-{size}-hf`, then
/// `root/models/Qwen3-ASR-{size}`.  The `-hf` suffix marks the
/// transformers-native checkpoint the frozen baselines were made with.
#[must_use]
pub fn resolve_model_dir(root: &Path, size: &str) -> std::path::PathBuf {
    let env_key = match size {
        "0.6B" => "QWEN3_ASR_MODEL_06_DIR",
        "1.7B" => "QWEN3_ASR_MODEL_17_DIR",
        _ => "",
    };
    if !env_key.is_empty() {
        if let Ok(p) = std::env::var(env_key) {
            return std::path::PathBuf::from(p);
        }
    }
    let hf = root.join(format!("models/Qwen3-ASR-{size}-hf"));
    if hf.join("config.json").is_file() {
        return hf;
    }
    root.join(format!("models/Qwen3-ASR-{size}"))
}

fn emit_token(
    cb: &mut dyn FnMut(StreamToken),
    tokenizer: &tokenizers::Tokenizer,
    generated: &[u32],
) -> Result<()> {
    let text_so_far = tokenizer
        .decode(generated, true)
        .map_err(|e| anyhow::anyhow!("decode: {}", e))?;
    cb(StreamToken { token_id: *generated.last().unwrap_or(&0), text_so_far });
    Ok(())
}

fn mel_pad_even(v: usize) -> usize {
    if v % 2 == 1 { v + 1 } else { v }
}

/// Per-request knobs, mirroring `Qwen3ASRModel.transcribe(audio,
/// context=…, language=…)`.  `context` is the hotword/bias text and goes into
/// the chat template's **system** message; `language` forces text-only output by
/// prefilling `language {Language}<asr_text>` after the assistant header.
///
/// Every entry point that takes these reads `max_new_tokens` from here — there
/// is no second, positional copy of it to fall out of sync with.
///
/// `#[non_exhaustive]` on purpose: fields get added (this one did), and a
/// harmless struct-literal at a call site would then be a breaking change.
/// Build one with [`TranscribeOptions::default`] plus the `with_*` setters.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    /// Hotword / bias text — the content of the chat template's `system` turn.
    /// Empty means "no context", which is what the frozen baselines were made
    /// with.
    pub context: String,
    /// `None` = let the model detect it.  Normalised and validated against
    /// [`prompt::SUPPORTED_LANGUAGES`] on use: a code (`"en"`) or
    /// a full name (`"English"`), either case.
    pub language: Option<String>,
    /// Upper bound on generated tokens (the stop is EOS, as in HF `generate`).
    /// The default matches common product / transformers use (2048).
    pub max_new_tokens: usize,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            context: String::new(),
            language: None,
            max_new_tokens: 2048,
        }
    }
}

impl TranscribeOptions {
    /// Bias the transcription towards a domain — the chat template's `system`
    /// message.  See the hotword notes in `README.md` for what this does and
    /// does not buy.
    #[must_use]
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = context.into();
        self
    }

    /// Force the output language, text-only (no language metadata turn).
    /// Accepts a code or a full name; an unsupported one fails at the call, not
    /// here, so the setter stays infallible like the other two.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Replace the generated-token ceiling.
    #[must_use]
    pub fn with_max_new_tokens(mut self, n: usize) -> Self {
        self.max_new_tokens = n;
        self
    }

    pub(crate) fn forced_language(&self) -> anyhow::Result<Option<String>> {
        let Some(raw) = self.language.as_deref() else {
            return Ok(None);
        };
        if raw.trim().is_empty() {
            return Ok(None);
        }
        Ok(Some(prompt::resolve_language(raw.trim())?))
    }
}

fn ch0_of(cap: &crate::audio_encoder_gpu::Capture, r: &crate::audio_encoder_gpu::ConvRound) -> usize {
    let _ = cap;
    r.chunk0
}

#[derive(Default)]
struct Diff {
    max: f32,
    sum2: f64,
    n: usize,
    bad: usize,
    first: Option<(usize, usize, f32, f32)>,
}

impl Diff {
    fn add(&mut self, i: usize, j: usize, got: f32, want: f32, tol: f32) {
        let d = (got - want).abs();
        if d > self.max {
            self.max = d;
        }
        self.sum2 += (d as f64) * (d as f64);
        self.n += 1;
        if d > tol {
            self.bad += 1;
            if self.first.is_none() {
                self.first = Some((i, j, want, got));
            }
        }
    }

    fn report(&self, what: &str, tag: &str) {
        let rms = if self.n > 0 { (self.sum2 / self.n as f64).sqrt() as f32 } else { 0.0 };
        let mut line = format!(
            "    {what:<20} max|d| {:.6}  rms {:.6}  bad {}/{}",
            self.max, rms, self.bad, self.n
        );
        if let Some((i, j, want, got)) = self.first {
            line += &format!("  first [{i},{j}] want {want:+.6} got {got:+.6}");
        }
        eprintln!("{line} ({tag})");
    }
}

pub(crate) enum TextBackend {
    Gpu(WgpuTextDecoder),
    Cpu(crate::cpu_decoder::CpuTextDecoder),
}

impl TextBackend {
    fn prefill(&mut self, hidden_words: &[u8], s: usize, kv_start: usize) -> Result<i32> {
        match self {
            Self::Gpu(d) => d.prefill(hidden_words, s, kv_start),
            Self::Cpu(d) => d.prefill(hidden_words, s, kv_start),
        }
    }

    fn step(&mut self) -> Result<i32> {
        match self {
            Self::Gpu(d) => d.step(),
            Self::Cpu(d) => d.step(),
        }
    }

    fn max_seq(&self) -> usize {
        match self {
            Self::Gpu(d) => d.max_seq,
            Self::Cpu(d) => d.max_seq,
        }
    }

    fn gpu(&self) -> Option<&Gpu> {
        match self {
            Self::Gpu(d) => Some(&d.gpu),
            Self::Cpu(_) => None,
        }
    }

    fn host_ms(&self) -> (f64, f64) {
        match self {
            Self::Gpu(d) => (d.host_submit_ms, d.host_read_ms),
            Self::Cpu(_) => (0.0, 0.0),
        }
    }

    fn set_rope_tables(&mut self, cos: &[f16], sin: &[f16]) {
        match self {
            Self::Gpu(d) => d.set_rope_tables(cos, sin),
            Self::Cpu(d) => d.set_rope_tables(cos, sin),
        }
    }
}

pub(crate) struct Inner {
    pub(crate) config: AsrConfig,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) encoder: CpuAudioEncoder,
    pub(crate) gpu_encoder: Option<GpuAudioEncoder>,
    pub(crate) mel: MelExtractor,
    pub(crate) tensors: std::collections::HashMap<String, weights::RawTensor>,
    pub(crate) decoder: TextBackend,
}

/// A loaded Qwen3-ASR model — the crate's entry point.
///
/// One `AsrInference` owns every device object, weight mapping and scratch
/// buffer the model needs, and serializes access behind a mutex.  That is what
/// lets one instance be shared rather than duplicated (the weights are ~1.2 GiB
/// at 0.6B, and the KV cache scales with the sequence cap):
///
/// ```no_run
/// use qwen3_asr_wgpu::{AsrInference, Backend, TranscribeOptions};
/// # fn main() -> qwen3_asr_wgpu::Result<()> {
/// let asr = std::sync::Arc::new(AsrInference::load(
///     std::path::Path::new("model-dir"),
///     Backend::best(),
/// )?);
///
/// let worker = std::sync::Arc::clone(&asr);
/// std::thread::spawn(move || worker.transcribe("clip.wav", TranscribeOptions::default()));
/// # Ok(())
/// # }
/// ```
///
/// Every method takes `&self`, so `AsrInference` is `Send + Sync`.  Concurrent
/// calls queue on the mutex — a transcription is not split across threads, it is
/// serialized.  Nothing leaks between calls: each generation prefills its own
/// prompt and resets the KV position to zero.
pub struct AsrInference {
    pub(crate) inner: std::sync::Mutex<Inner>,
}

impl AsrInference {
    pub(crate) fn lock(&self) -> crate::Result<std::sync::MutexGuard<'_, Inner>> {
        self.inner
            .lock()
            .map_err(|_| crate::AsrError::Inference(anyhow::anyhow!("mutex poisoned")))
    }

    /// Load a checkpoint directory with a one-word backend choice — the common
    /// path.  See [`Backend`] for what each variant does and
    /// [`Self::load_on`] / [`Self::load_with`] for the finer knobs.
    ///
    /// `model_dir` must hold `config.json`, `tokenizer.json` and the
    /// safetensors weights — i.e. one of the HuggingFace `Qwen/Qwen3-ASR-*`
    /// repositories, as downloaded or via [`Self::from_pretrained`].
    pub fn load(model_dir: &Path, backend: Backend) -> crate::Result<Self> {
        let (selector, encoder) = backend.resolve()?;
        Self::load_with(model_dir, selector, encoder)
    }

    /// [`Self::load`] with [`Backend::Auto`].
    pub fn new(model_dir: &Path) -> crate::Result<Self> {
        Self::load(model_dir, Backend::Auto)
    }

    /// Load on a specific device, named by runtime and index (`vulkan:0`,
    /// `dx12`, `metal`), by raw enumeration index, or by a substring of the
    /// adapter name.  See [`crate::AsrInference::device_targets`] —
    /// or `transcribe --list-devices` — for the names this machine accepts.
    ///
    /// The GPU audio tower is used; a device that cannot build it falls back to
    /// the host tower with a printed reason.
    pub fn load_on(model_dir: &Path, selector: DeviceSelector) -> crate::Result<Self> {
        Self::load_with(model_dir, selector, EncoderBackend::Gpu)
    }

    /// The full form: pick the device *and* which audio tower to build.
    ///
    /// [`EncoderBackend::Cpu`] is the host f32 reference the GPU tower is
    /// verified against — ~2.7× slower on the encoder phase, and the reason the
    /// two can be diffed at all.
    pub fn load_with(
        model_dir: &Path,
        selector: DeviceSelector,
        encoder: EncoderBackend,
    ) -> crate::Result<Self> {
        let inner = Inner::load_with_selector(model_dir, selector, encoder)
            .map_err(crate::AsrError::ModelLoad)?;
        Ok(Self {
            inner: std::sync::Mutex::new(inner),
        })
    }

    /// Download `model_id` from HuggingFace into `cache_dir` (if it is not
    /// already there) and load it.
    ///
    /// Requires the `hub` feature.  The download is resumed by presence, not by
    /// range: an interrupted one leaves a directory without the `.complete`
    /// marker and is fetched again from scratch.
    #[cfg(feature = "hub")]
    pub fn from_pretrained(
        model_id: &str,
        cache_dir: &Path,
        backend: Backend,
    ) -> crate::Result<Self> {
        let model_dir =
            crate::hub::ensure_model_cached(model_id, cache_dir).map_err(crate::AsrError::ModelLoad)?;
        Self::load(&model_dir, backend)
    }

    /// Every adapter wgpu can see on this machine, in the order
    /// [`DeviceSelector::Index`] indexes into.  Cheap: no device is created.
    pub fn devices() -> Vec<crate::gpu::DeviceInfo> {
        Inner::devices()
    }

    /// The user-facing device list: each target named `<runtime>:<index>`, the
    /// form [`Self::load_on`] accepts, with the default marked.
    pub fn device_targets() -> Vec<crate::gpu::DeviceTarget> {
        Inner::device_targets()
    }

    /// The device this instance actually runs on, or `None` on the CPU backend.
    pub fn device(&self) -> Option<wgpu::AdapterInfo> {
        self.lock().ok().and_then(|g| g.device().cloned())
    }

    /// One-line description of the running backend (device name, runtime,
    /// driver and limits — or that this is the CPU path).
    pub fn device_description(&self) -> String {
        match self.lock() {
            Ok(g) => g.device_description(),
            Err(e) => format!("unavailable: {e}"),
        }
    }

    /// True when the GPU audio tower is loaded rather than the host one.
    pub fn gpu_encoder_active(&self) -> bool {
        self.lock().map(|g| g.gpu_encoder_active()).unwrap_or(false)
    }

    /// Transcribe a wav file (any sample rate; it is resampled to 16 kHz).
    ///
    /// Returns the parsed transcript plus the language — either the one the
    /// model named in its output, or, when the caller forced one, the caller's
    /// (see [`TranscribeOptions::language`]).
    pub fn transcribe(
        &self,
        audio_path: &str,
        opts: TranscribeOptions,
    ) -> crate::Result<TranscribeResult> {
        let forced = check_options(&opts)?;
        let samples = load_audio_wav(audio_path, MEL_SAMPLE_RATE)?;
        let mut guard = self.lock()?;
        let r = guard
            .transcribe_samples(&samples, opts.max_new_tokens)
            .map_err(crate::AsrError::Inference)?;
        Ok(with_forced_language(r, forced))
    }

    /// [`Self::transcribe`] from 16 kHz mono samples already in memory.
    pub fn transcribe_samples(
        &self,
        samples: &[f32],
        opts: TranscribeOptions,
    ) -> crate::Result<TranscribeResult> {
        let forced = check_options(&opts)?;
        let mut guard = self.lock()?;
        let r = guard
            .transcribe_samples(samples, opts.max_new_tokens)
            .map_err(crate::AsrError::Inference)?;
        Ok(with_forced_language(r, forced))
    }

    /// [`Self::transcribe`] with `on_token` called once per generated token.
    ///
    /// The callback is host-side only: the token stream, and therefore the final
    /// text, is identical to the non-streaming call.  `text_so_far` is the raw
    /// decode — no metadata split, no repetition fix.
    pub fn transcribe_streaming<F>(
        &self,
        audio_path: &str,
        opts: TranscribeOptions,
        on_token: F,
    ) -> crate::Result<TranscribeResult>
    where
        F: FnMut(StreamToken),
    {
        let forced = check_options(&opts)?;
        let samples = load_audio_wav(audio_path, MEL_SAMPLE_RATE)?;
        let mut guard = self.lock()?;
        let r = guard
            .transcribe_samples_streaming(
                &samples,
                opts.max_new_tokens,
                None,
                false,
                &opts,
                on_token,
            )
            .map_err(crate::AsrError::Inference)?;
        Ok(with_forced_language(r, forced))
    }

    /// [`Self::transcribe_samples`] with `on_token` called once per token.
    pub fn transcribe_samples_streaming<F>(
        &self,
        samples: &[f32],
        opts: TranscribeOptions,
        on_token: F,
    ) -> crate::Result<TranscribeResult>
    where
        F: FnMut(StreamToken),
    {
        let forced = check_options(&opts)?;
        let mut guard = self.lock()?;
        let r = guard
            .transcribe_samples_streaming(
                samples,
                opts.max_new_tokens,
                None,
                false,
                &opts,
                on_token,
            )
            .map_err(crate::AsrError::Inference)?;
        Ok(with_forced_language(r, forced))
    }

    /// Start a session that takes audio incrementally and returns text at
    /// [`crate::AsrStreamingSession::flush`].
    ///
    /// The session holds this instance's lock until it is dropped, so the two
    /// cannot be used at once.
    pub fn create_streaming_session(
        &self,
        opts: TranscribeOptions,
    ) -> crate::Result<crate::AsrStreamingSession<'_>> {
        check_options(&opts)?;
        let forced = opts.forced_language().ok().flatten();
        let guard = self.lock()?;
        Ok(crate::AsrStreamingSession::new(guard, opts, forced))
    }
}

fn check_options(opts: &TranscribeOptions) -> crate::Result<Option<String>> {
    if opts.max_new_tokens == 0 {
        return Err(crate::AsrError::InvalidOptions(
            "max_new_tokens is 0 — nothing would be generated".to_string(),
        ));
    }
    opts.forced_language()
        .map_err(|e| crate::AsrError::InvalidOptions(e.to_string()))
}

pub(crate) fn with_forced_language(
    mut r: TranscribeResult,
    forced: Option<String>,
) -> TranscribeResult {
    if r.language.is_empty() {
        if let Some(name) = forced {
            r.language = name;
        }
    }
    r
}

impl Inner {
    pub fn load_with_selector(
        model_dir: &Path,
        selector: DeviceSelector,
        backend: EncoderBackend,
    ) -> Result<Self> {
        let config = AsrConfig::from_file(&model_dir.join("config.json"))
            .with_context(|| format!("config {}", model_dir.display()))?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let tensors = weights::load_tensors(model_dir)?;
        let encoder = CpuAudioEncoder::load(
            &tensors,
            "thinker.audio_tower",
            &config.thinker_config.audio_config,
        )?;
        let n_mels = config.thinker_config.audio_config.num_mel_bins;
        let want_cpu = selector == DeviceSelector::Cpu;
        let gpu = if want_cpu {
            None
        } else {
            match pollster::block_on(Gpu::new_with(selector.clone())) {
                Ok(g) => Some(g),
                Err(e) if selector == DeviceSelector::Auto => {
                    eprintln!("[device] no usable GPU ({e:#}); falling back to the CPU backend");
                    None
                }
                Err(e) => return Err(e),
            }
        };
        let gpu_encoder = match (&gpu, backend) {
            (None, _) | (_, EncoderBackend::Cpu) => None,
            (Some(gpu), EncoderBackend::Gpu) => {
                let ac = &config.thinker_config.audio_config;
                let window_infer = ac.n_window_infer.max(ac.n_window * 2);
                match GpuAudioEncoder::load(gpu, &tensors, "thinker.audio_tower", ac, window_infer) {
                    Ok(e) => Some(e),
                    Err(e) => {
                        eprintln!("[gpu encoder] unavailable, falling back to CPU: {e:#}");
                        None
                    }
                }
            }
        };
        let text_cfg = TextConfig::from_model_dir(model_dir)?;
        crate::cpu_decoder::check_config(&text_cfg)?;
        let mut decoder = match gpu {
            Some(gpu) => TextBackend::Gpu(WgpuTextDecoder::load(
                gpu,
                model_dir,
                "thinker.model",
                text_cfg,
                DECODER_MAX_SEQ,
                DECODER_MAX_SEQ,
            )?),
            None => {
                eprintln!("[decoder] CPU backend (no GPU involved)");
                TextBackend::Cpu(crate::cpu_decoder::CpuTextDecoder::load(
                    model_dir,
                    "thinker.model",
                    text_cfg,
                    DECODER_MAX_SEQ,
                    DECODER_MAX_SEQ,
                )?)
            }
        };
        let text = &config.thinker_config.text_config;
        let (cos, sin) = compute_mrope_cos_sin(
            &text_positions(DECODER_MAX_SEQ),
            text.head_dim,
            text.rope_theta,
            &text.mrope_section(),
            text.mrope_interleaved(),
        );
        let cos_f16: Vec<f16> = cos.iter().copied().map(f16::from_f32).collect();
        let sin_f16: Vec<f16> = sin.iter().copied().map(f16::from_f32).collect();
        decoder.set_rope_tables(&cos_f16, &sin_f16);
        if let Some(gpu) = decoder.gpu() {
            if let Err(e) = gpu.save_pipeline_cache() {
                eprintln!("[pipeline cache] not saved: {e:#}");
            }
        }
        Ok(Self {
            config,
            tokenizer,
            encoder,
            gpu_encoder,
            mel: MelExtractor::new(N_FFT, HOP_LENGTH, n_mels, MEL_SAMPLE_RATE),
            tensors,
            decoder,
        })
    }

    /// Every adapter wgpu can see on this machine, in the order
    /// [`DeviceSelector::Index`] indexes into.  Cheap: no device is created.
    pub fn devices() -> Vec<crate::gpu::DeviceInfo> {
        pollster::block_on(crate::gpu::list_devices())
    }

    /// The user-facing device list: each target named `<runtime>:<index>`, the
    /// form `--device` accepts, with the default marked.
    pub fn device_targets() -> Vec<crate::gpu::DeviceTarget> {
        pollster::block_on(crate::gpu::list_targets())
    }

    /// The device this instance actually runs on, or `None` on the CPU backend.
    pub fn device(&self) -> Option<&wgpu::AdapterInfo> {
        self.decoder.gpu().map(|g| &g.info)
    }

    /// One-line description of the running backend (device name, runtime,
    /// driver and limits — or that this is the CPU path).
    pub fn device_description(&self) -> String {
        match self.decoder.gpu() {
            Some(g) => g.describe(),
            None => format!(
                "cpu ({} threads, f16 weights, rayon)",
                rayon::current_num_threads()
            ),
        }
    }

    /// True when the GPU audio tower is loaded (i.e. `--gpu-enc` took effect).
    pub fn gpu_encoder_active(&self) -> bool {
        self.gpu_encoder.is_some()
    }

    /// [`Inner::transcribe_file_opts`] with `context` / `language` knobs.
    pub fn transcribe_file_opts(
        &mut self,
        wav: &Path,
        max_new_tokens: usize,
        dump_dir: Option<&Path>,
        compare_enc: bool,
        opts: &TranscribeOptions,
    ) -> Result<TranscribeResult> {
        let samples = load_audio_wav(wav, MEL_SAMPLE_RATE)?;
        if let Some(dir) = dump_dir {
            std::fs::create_dir_all(dir)?;
            let mut bytes = Vec::with_capacity(samples.len() * 4);
            for v in &samples {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(dir.join("wave16k.f32"), bytes)?;
        }
        self.transcribe_samples_impl(&samples, max_new_tokens, dump_dir, compare_enc, opts)
    }

    /// [`Self::transcribe_file_opts`] with a per-token callback.
    ///
    /// `on_token` fires once per generated token with the token id and the raw
    /// text so far.  The callback is host-side only: the token stream
    /// (and therefore the final text) is identical to the non-streaming call.
    pub fn transcribe_file_streaming<F>(
        &mut self,
        wav: &Path,
        max_new_tokens: usize,
        dump_dir: Option<&Path>,
        compare_enc: bool,
        opts: &TranscribeOptions,
        mut on_token: F,
    ) -> Result<TranscribeResult>
    where
        F: FnMut(StreamToken),
    {
        let samples = load_audio_wav(wav, MEL_SAMPLE_RATE)?;
        if let Some(dir) = dump_dir {
            std::fs::create_dir_all(dir)?;
            let mut bytes = Vec::with_capacity(samples.len() * 4);
            for v in &samples {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(dir.join("wave16k.f32"), bytes)?;
        }
        self.transcribe_samples_streaming(&samples, max_new_tokens, dump_dir, compare_enc, opts, &mut on_token)
    }

    /// [`Self::transcribe_samples`] with a per-token callback.
    pub fn transcribe_samples_streaming<F>(
        &mut self,
        samples: &[f32],
        max_new_tokens: usize,
        dump_dir: Option<&Path>,
        compare_enc: bool,
        opts: &TranscribeOptions,
        mut on_token: F,
    ) -> Result<TranscribeResult>
    where
        F: FnMut(StreamToken),
    {
        self.transcribe_samples_impl_stream(
            samples,
            max_new_tokens,
            dump_dir,
            compare_enc,
            opts,
            Some(&mut on_token),
        )
    }

    /// Mel-in entry point with an optional CPU-vs-GPU encoder comparison,
    /// with `context` / `language` knobs.
    #[allow(clippy::too_many_arguments)]
    pub fn transcribe_from_mel_cmp_opts(
        &mut self,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
        max_new_tokens: usize,
        dump_dir: Option<&Path>,
        compare_enc: bool,
        opts: &TranscribeOptions,
    ) -> Result<TranscribeResult> {
        let t1 = Instant::now();
        let audio_embeds = self.run_encoder(mel, n_mels, n_frames, compare_enc)?;
        let t_enc = t1.elapsed();
        if let Some(dir) = dump_dir {
            std::fs::create_dir_all(dir)?;
            write_f32_dump(dir, "mel.f32", mel)?;
            write_f32_dump(dir, "audio_embeds.f32", &audio_embeds)?;
            std::fs::write(
                dir.join("shapes.txt"),
                format!(
                    "mel {n_mels} {n_frames}\nembeds {} {}\n",
                    audio_embeds.len() / self.config.thinker_config.text_config.hidden_size,
                    self.config.thinker_config.text_config.hidden_size
                ),
            )?;
        }
        self.decode_from_audio_embeds(
            &audio_embeds,
            max_new_tokens,
            dump_dir,
            0.0,
            t_enc.as_secs_f64() * 1000.0,
            opts,
            None,
        )
    }

    /// Mel frontend only (probe hook — no GPU work).
    pub fn extract_mel(&self, samples: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        self.mel.extract(samples)
    }

    pub(crate) fn decode_ids(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))
    }

    /// Audio encoder only (probe hook — no GPU work).
    pub fn encode_mel(&mut self, mel: &[f32], n_mels: usize, n_frames: usize) -> Result<Vec<f32>> {
        self.encoder.forward(mel, n_mels, n_frames)
    }

    /// CPU conv-stem geometry `(c,h,w)` per conv layer (probe hook).
    pub fn conv_geometry(&self, n_mels: usize) -> Result<[usize; 9]> {
        self.encoder.conv_geometry(n_mels)
    }

    /// `conv_out` input width (probe hook).
    pub fn conv_out_in_features(&self) -> usize {
        self.encoder.conv_out_in_features()
    }

    /// `conv2d1` bias (probe hook).
    pub fn conv_bias_c1(&self) -> Result<Vec<f32>> {
        self.encoder.conv_bias_c1()
    }

    /// `conv2d1` weight as `(data, c_out, taps)`, row-major `[c_out, kh*3+kw]`
    /// (probe hook for hand-checking the GEMM).
    pub fn conv_weight_c1(&self) -> Result<(Vec<f32>, usize, usize)> {
        self.encoder.conv_weight_c1()
    }

    /// Raw c1 conv output `[chunk][c_out][pos]` (probe hook).
    pub fn conv_reference_c1(
        &self,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
    ) -> Result<(Vec<f32>, usize, usize, usize)> {
        self.encoder.conv_reference_c1(mel, n_mels, n_frames)
    }

    /// Stage-by-stage GPU-vs-CPU comparison of the audio tower (diagnostic).
    pub fn diagnose_encoder(&mut self, wav: &Path) -> Result<()> {
        let samples = load_audio_wav(wav, MEL_SAMPLE_RATE)?;
        let (mel, n_mels, n_frames) = self.mel.extract(&samples)?;
        self.diagnose_encoder_mel(&mel, n_mels, n_frames)
    }

    /// As [`Self::diagnose_encoder`] but from a precomputed mel.
    pub fn diagnose_encoder_mel(
        &mut self,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
    ) -> Result<()> {
        let cs = self.encoder.config().n_window * 2;
        let TextBackend::Gpu(dec) = &self.decoder else {
            anyhow::bail!("--diag-enc needs the GPU text decoder and audio tower");
        };
        let gpu = &dec.gpu;
        let Some(enc) = self.gpu_encoder.as_mut() else {
            anyhow::bail!("--diag-enc needs the GPU audio tower (it is the default; drop --cpu-enc)");
        };
        let cap = enc.encode_capture(gpu, mel, n_mels, n_frames, true)?;
        anyhow::ensure!(!cap.rounds.is_empty(), "no conv rounds captured");
        let n_chunks: usize = cap.rounds.iter().map(|r| r.n_chunks).sum();
        eprintln!(
            "-- conv stem: {} chunks in {} round(s) of {} | mel {}x{} | cs {cs}",
            n_chunks,
            cap.rounds.len(),
            crate::audio_encoder_gpu::conv_tile(),
            n_mels,
            n_frames
        );

        let w0 = {
            let v = enc.level(0);
            let _ = v;
            mel_pad_even(cs)
        };
        let mut input: Vec<f32> = vec![0.0; n_chunks * n_mels * w0];
        for i in 0..n_chunks {
            for m in 0..n_mels {
                for f in 0..cs.min(n_frames.saturating_sub(i * cs).max(0)) {
                    let src = if i * cs + f < n_frames { mel[m * n_frames + i * cs + f] } else { 0.0 };
                    input[(i * n_mels + m) * w0 + f] = half::f16::from_f32(src).to_f32();
                }
            }
        }
        let (mut c_in, mut h, mut w) = (1usize, n_mels, cs);

        for level in 0..3 {
            let v = enc.level(level);
            let cpu = self.encoder.conv_stages_from(level, &input, n_chunks, c_in, h, w)?;
            anyhow::ensure!(
                (cpu.plane, cpu.c_out, cpu.k) == (v.plane, v.c_out, v.k),
                "conv{}: CPU geometry {}x{}x{} != GPU {}x{}x{}",
                level + 1,
                cpu.plane,
                cpu.c_out,
                cpu.k,
                v.plane,
                v.c_out,
                v.k
            );
            eprintln!(
                "  conv{}: {}x{} -> {}x{} ({} pos/chunk, padded {}), c_out {} /*pad {}*/, k {} /*pad {}*/, n_all {}",
                level + 1,
                v.h_in, v.w_in, v.h_out, v.w_out, v.plane, v.plane_pad, v.c_out, v.m_pad, v.k, v.k_pad, v.n_all
            );

            let mut dcol = Diff::default();
            let mut draw = Diff::default();
            let mut dact = Diff::default();
            for r in &cap.rounds {
                for cc in 0..r.n_chunks {
                    let chunk = ch0_of(&cap, r) + cc;
                    let lcol = chunk * v.plane;
                    for p in 0..v.plane {
                        let gcol = cc * v.plane_pad + p;
                        for kk in 0..v.k {
                            dcol.add(
                                chunk,
                                kk,
                                r.col[level][kk * v.n_all + gcol].to_f32(),
                                cpu.cols[(lcol + p) * v.k + kk],
                                2e-3,
                            );
                        }
                        for c in 0..v.c_out {
                            let cri = c * (n_chunks * v.plane) + lcol + p;
                            let cai = (chunk * v.c_out + c) * v.plane + p;
                            draw.add(chunk, c, r.raw[level][c * v.n_all + gcol].to_f32(), cpu.raw[cri], 2e-3);
                            dact.add(chunk, c, r.act[level][c * v.n_all + gcol].to_f32(), cpu.act[cai], 2e-3);
                        }
                    }
                }
            }
            dcol.report(&format!("conv{} operand", level + 1), "[k][pos]");
            draw.report(&format!("conv{} GEMM", level + 1), "raw (pre-GELU)");
            dact.report(&format!("conv{} +bias/GELU", level + 1), "act");

            let (c_out, plane, plane_pad, n_all) = (v.c_out, v.plane, v.plane_pad, v.n_all);
            let mut next = vec![0.0f32; n_chunks * c_out * plane];
            for r in &cap.rounds {
                for cc in 0..r.n_chunks {
                    let chunk = ch0_of(&cap, r) + cc;
                    for c in 0..c_out {
                        for p in 0..plane {
                            next[(chunk * c_out + c) * plane + p] =
                                r.act[level][c * n_all + cc * plane_pad + p].to_f32();
                        }
                    }
                }
            }
            input = next;
            c_in = c_out;
            h = v.h_out;
            w = v.w_out;
        }

        let tower = self.encoder.conv_tower(mel, n_mels, n_frames)?;
        let cf = tower.packed.len().checked_div(tower.n_total).unwrap_or(0);
        let dm = self.encoder.config().d_model;
        if let Some(a) = cap.attn.as_ref() {
            let (hd, wlen, wpad) = (a.hd, a.wlen, a.wpad);
            let hd_pad = a.attn_out.len() / (a.nh * a.n_win * a.wpad);
            let scale = 1.0f32 / (hd as f32).sqrt();
            let norm = |src: &[f16]| {
                let n = (a.s * a.acols).min(src.len());
                src[..n].iter().fold(0.0f32, |m, v| m.max(v.to_f32().abs()))
            };
            let inv_scale = (hd as f32).sqrt();
            let (mut dsc, mut dpr, mut dav) = (Diff::default(), Diff::default(), Diff::default());
            for (head, win) in [(0usize, 0usize), (1, 0), (0, a.n_win - 1), (1, a.n_win - 1)] {
                let z = head * a.n_win + win;
                let valid = wlen.min(a.s - win * wlen);
                let rows = valid.min(8);
                let sc0 = dsc.bad;
                let (mut pbad, mut abad, mut first_bad) = (0usize, 0usize, usize::MAX);
                let (mut dq, mut dk, mut dv) = (0.0f32, 0.0f32, 0.0f32);
                let mut qat = (0usize, 0usize);
                for r in 0..valid {
                    for jj in 0..hd {
                        let g = a.qp[z * wpad * hd + r * hd + jj].to_f32();
                        let w = a.q[(win * wlen + r) * a.acols + head * hd + jj].to_f32();
                        if (g - w).abs() > dq {
                            dq = (g - w).abs();
                            qat = (r, jj);
                        }
                        let g = a.vp[z * wpad * a.hd_pad + r * a.hd_pad + jj].to_f32();
                        let w = a.v[(win * wlen + r) * a.acols + head * hd + jj].to_f32();
                        dv = dv.max((g - w).abs());
                    }
                }
                for j in 0..hd {
                    for t in 0..valid {
                        let g = a.kt[z * hd * wpad + j * wpad + t].to_f32();
                        let w = a.k[(win * wlen + t) * a.acols + head * hd + j].to_f32();
                        dk = dk.max((g - w).abs());
                    }
                }
                for row in 0..rows {
                    let tok = win * wlen + row;
                    let mut sc = vec![0.0f32; valid];
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..valid {
                        let mut dot = 0.0f32;
                        for d in 0..hd {
                            let qi = a.q[tok * a.acols + head * hd + d].to_f32();
                            let ki = a.k[(win * wlen + j) * a.acols + head * hd + d].to_f32();
                            dot += qi * ki;
                        }
                        sc[j] = dot * scale;
                        mx = mx.max(dot * scale);
                    }
                    let mut sum = 0.0f32;
                    for j in 0..valid {
                        sum += (sc[j] - mx).exp();
                    }
                    let pr_row = dpr.bad;
                    let av_row = dav.bad;
                    for j in 0..valid {
                        let row_g = z * wpad + row;
                        let gs = a.scores[z * wpad * wpad + row * wpad + j].to_f32();
                        let raw = sc[j] * inv_scale;
                        dsc.add(row_g, j, gs, raw, 2e-2 + 2e-3 * raw.abs());
                        let gp = a.probs[z * wpad * wpad + row * wpad + j].to_f32();
                        dpr.add(row_g, j, gp, (sc[j] - mx).exp() / sum, 2e-2);
                    }
                    for d in 0..hd {
                        let mut acc = 0.0f32;
                        for j in 0..valid {
                            acc += (sc[j] - mx).exp() / sum * a.v[(win * wlen + j) * a.acols + head * hd + d].to_f32();
                        }
                        let got = a.attn_out[(z * wpad + row) * hd_pad + d].to_f32();
                        dav.add(z * wpad + row, d, got, acc, 2e-2);
                    }
                    if dpr.bad > pr_row {
                        pbad += 1;
                    }
                    if dav.bad > av_row {
                        abad += 1;
                        if first_bad == usize::MAX {
                            first_bad = row;
                        }
                    }
                }
                eprintln!(
                    "    head {head} win {win} z {z} (valid {valid}): operands qp {dq:.4}@{qat:?} kt {dk:.4} vp {dv:.4} | scores bad {} | probs bad rows {pbad} | av bad rows {abad}{}",
                    dsc.bad - sc0,
                    if first_bad == usize::MAX {
                        String::new()
                    } else {
                        format!(", first bad row {first_bad}")
                    },
                );
            }
            eprintln!(
                "    max|q| {:.4}  max|k| {:.4}  max|v| {:.4}  (must be non-zero)",
                norm(&a.q),
                norm(&a.k),
                norm(&a.v)
            );
            if !a.normed.is_empty() && !a.qkv.is_empty() && !a.attn_flat.is_empty() {
                let h0: Vec<f32> = cap.h[..tower.n_total * dm].iter().map(|v| v.to_f32()).collect();
                let n3 = tower.n_total * 3 * dm;
                let mut dn = Diff::default();
                let mut dq = Diff::default();
                let mut da = Diff::default();
                let want_normed = self.encoder.dbg_layer_norm(0, &h0, tower.n_total)?;
                for t in 0..tower.n_total {
                    for j in 0..dm {
                        dn.add(t, j, a.normed[t * dm + j].to_f32(), want_normed[t * dm + j], 3e-3);
                    }
                }
                let want_qkv = self.encoder.dbg_qkv(0, &want_normed, tower.n_total)?;
                for t in 0..tower.n_total {
                    for j in 0..3 * dm {
                        let gi = t * (n3 / tower.n_total) + j;
                        if gi < a.qkv.len() {
                            dq.add(t, j, a.qkv[gi].to_f32(), want_qkv[t * 3 * dm + j], 3e-3);
                        }
                    }
                }
                let want_attn = self.encoder.dbg_attn_flat(0, &want_normed, tower.n_total)?;
                for t in 0..tower.n_total {
                    for j in 0..dm {
                        da.add(t, j, a.attn_flat[t * a.acols + j].to_f32(), want_attn[t * dm + j], 3e-2);
                    }
                }
                if !a.mid.is_empty() {
                    let attn_out = self.encoder.dbg_attn(0, &want_normed, tower.n_total)?;
                    let mut dmid = Diff::default();
                    for t in 0..tower.n_total {
                        for j in 0..dm {
                            let want = h0[t * dm + j] + attn_out[t * dm + j];
                            dmid.add(t, j, a.mid[t * dm + j].to_f32(), want, 5e-3);
                        }
                    }
                    dmid.report("attn residual", "input + CPU attention");
                }
                eprintln!("-- layer 0 stages (CPU oracle on the same tokens) --");
                dn.report("LN1 (normed)", "vs CPU");
                dq.report("fused QKV", "vs CPU");
                da.report("attention block", "vs CPU attention+out_proj");
            }
            eprintln!("-- attention oracle (layer 0, heads 0/1, first/last window) --");
            dsc.report("scores", "host dot(q,k)");
            dpr.report("probs", "host softmax");
            dav.report("av out", "host prob*v");
        }

        if !cap.layers.is_empty() {
            let mut prev: Vec<f32> = cap.h.iter().take(tower.n_total * dm).map(|v| v.to_f32()).collect();
            for (li, got) in cap.layers.iter().enumerate() {
                let want = self.encoder.layer_forward(li, &prev, tower.n_total)?;
                let mut d = Diff::default();
                for t in 0..tower.n_total {
                    for j in 0..dm {
                        d.add(t, j, got[t * dm + j].to_f32(), want[t * dm + j], 5e-3);
                    }
                }
                d.report(&format!("layer {li}"), "vs CPU layer on the same input");
                prev = want;
            }
        }

        let mut dpack = Diff::default();
        if cf > 0 && cap.packed.len() >= tower.n_total * cf {
            for t in 0..tower.n_total {
                for j in 0..cf {
                    dpack.add(t, j, cap.packed[t * cf + j].to_f32(), tower.packed[t * cf + j], 5e-3);
                }
            }
        }
        for level in 0..3 {
            let v = enc.level(level);
            let mut d = Diff::default();
            for r in &cap.rounds {
                for cc in 0..r.n_chunks {
                    let chunk = ch0_of(&cap, r) + cc;
                    for c in 0..v.c_out {
                        for p in 0..v.plane {
                            let cai = (chunk * v.c_out + c) * v.plane + p;
                            d.add(
                                chunk,
                                c,
                                r.act[level][c * v.n_all + cc * v.plane_pad + p].to_f32(),
                                tower.stages[level].act[cai],
                                5e-3,
                            );
                        }
                    }
                }
            }
            d.report(&format!("conv{} act", level + 1), "vs CPU chain (loose, f16 drift included)");
        }
        dpack.report("packed [tok][c·f]", "vs CPU permute");
        let mut dh = Diff::default();
        for t in 0..tower.n_total {
            for j in 0..dm {
                if (t * dm + j) < cap.h.len() && (t * dm + j) < tower.h.len() {
                    dh.add(t, j, cap.h[t * dm + j].to_f32(), tower.h[t * dm + j], 5e-3);
                }
            }
        }
        dh.report("h [tok][d_model]", "vs CPU conv_out+PE");
        if cf > 0 {
            let gpu_packed: Vec<f32> = cap.packed[..tower.n_total * cf]
                .iter()
                .map(|v| v.to_f32())
                .collect();
            let tight = self.encoder.conv_out_from(&gpu_packed, tower.n_total)?;
            let mut dt = Diff::default();
            for t in 0..tower.n_total {
                for j in 0..dm {
                    dt.add(t, j, cap.h[t * dm + j].to_f32(), tight[t * dm + j], 5e-3);
                }
            }
            dt.report("h [tok][d_model]", "vs CPU conv_out(GPU operand)+PE");
            eprintln!(
                "    h[0..6] gpu {:?}",
                cap.h[..6].iter().map(|v| v.to_f32()).collect::<Vec<_>>()
            );
            eprintln!(
                "    h[0..6] cpu {:?}",
                tower.h[..6].iter().map(|v| *v).collect::<Vec<_>>()
            );
            eprintln!(
                "    pe[0..6]    {:?}",
                self.encoder.pe_row(0).into_iter().take(6).collect::<Vec<_>>()
            );
        }
        eprintln!(
            "  (the whole-chain rows carry {} chunks of f16 activation drift; the per-level",
            n_chunks
        );
        eprintln!("   tables above are the tight ones)");
        eprintln!(
            "-- gpu encode wall {:.1} ms (pack {:.1} ms)",
            crate::audio_encoder_gpu::last_encode_ms(),
            crate::audio_encoder_gpu::last_pack_ms()
        );
        Ok(())
    }

    pub(crate) fn run_encoder(
        &mut self,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
        compare: bool,
    ) -> Result<Vec<f32>> {
        let TextBackend::Gpu(dec) = &self.decoder else {
            return self.encoder.forward(mel, n_mels, n_frames);
        };
        let Some(enc) = self.gpu_encoder.as_mut() else {
            return self.encoder.forward(mel, n_mels, n_frames);
        };
        let t = Instant::now();
        let out16 = enc.encode(&dec.gpu, mel, n_mels, n_frames)?;
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;
        let embeds: Vec<f32> = out16.iter().map(|v| v.to_f32()).collect();
        if compare {
            let t = Instant::now();
            let cpu = self.encoder.forward(mel, n_mels, n_frames)?;
            let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;
            print_embed_diff(&cpu, &embeds, n_frames);
            eprintln!(
                "encoder compare: cpu {cpu_ms:.1} ms vs gpu {gpu_ms:.1} ms (gpu pack {:.1} ms)",
                crate::audio_encoder_gpu::last_pack_ms()
            );
        }
        Ok(embeds)
    }

    pub fn transcribe_samples(&mut self, samples: &[f32], max_new_tokens: usize) -> Result<TranscribeResult> {
        self.transcribe_samples_with_dump(samples, max_new_tokens, None)
    }

    pub fn transcribe_from_embeds(
        &mut self,
        audio_embeds: &[f32],
        max_new_tokens: usize,
        dump_dir: Option<&Path>,
    ) -> Result<TranscribeResult> {
        self.decode_from_audio_embeds(
            audio_embeds,
            max_new_tokens,
            dump_dir,
            0.0,
            0.0,
            &TranscribeOptions::default(),
            None,
        )
    }

    pub fn transcribe_samples_with_dump(
        &mut self,
        samples: &[f32],
        max_new_tokens: usize,
        dump_dir: Option<&Path>,
    ) -> Result<TranscribeResult> {
        self.transcribe_samples_impl(samples, max_new_tokens, dump_dir, false, &TranscribeOptions::default())
    }

    fn transcribe_samples_impl(
        &mut self,
        samples: &[f32],
        max_new_tokens: usize,
        dump_dir: Option<&Path>,
        compare_enc: bool,
        opts: &TranscribeOptions,
    ) -> Result<TranscribeResult> {
        self.transcribe_samples_impl_stream(samples, max_new_tokens, dump_dir, compare_enc, opts, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn transcribe_samples_impl_stream(
        &mut self,
        samples: &[f32],
        max_new_tokens: usize,
        dump_dir: Option<&Path>,
        compare_enc: bool,
        opts: &TranscribeOptions,
        stream: Option<&mut dyn FnMut(StreamToken)>,
    ) -> Result<TranscribeResult> {
        let t0 = Instant::now();
        let (mel, n_mels, n_frames) = self.mel.extract(samples)?;
        let t_mel = t0.elapsed();

        let t1 = Instant::now();
        let audio_embeds = self.run_encoder(&mel, n_mels, n_frames, compare_enc)?;
        let t_enc = t1.elapsed();
        if let Some(dir) = dump_dir {
            std::fs::create_dir_all(dir)?;
            write_f32_dump(dir, "mel.f32", &mel)?;
            write_f32_dump(dir, "audio_embeds.f32", &audio_embeds)?;
            std::fs::write(
                dir.join("shapes.txt"),
                format!(
                    "mel {n_mels} {n_frames}\nembeds {} {}\n",
                    audio_embeds.len() / self.config.thinker_config.text_config.hidden_size,
                    self.config.thinker_config.text_config.hidden_size
                ),
            )?;
        }
        self.decode_from_audio_embeds(
            &audio_embeds,
            max_new_tokens,
            dump_dir,
            t_mel.as_secs_f64() * 1000.0,
            t_enc.as_secs_f64() * 1000.0,
            opts,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_from_audio_embeds(
        &mut self,
        audio_embeds: &[f32],
        max_new_tokens: usize,
        dump_dir: Option<&Path>,
        t_mel_ms: f64,
        t_enc_ms: f64,
        opts: &TranscribeOptions,
        stream: Option<&mut dyn FnMut(StreamToken)>,
    ) -> Result<TranscribeResult> {
        if let Some(dir) = dump_dir {
            std::fs::create_dir_all(dir)?;
            let mut bytes = Vec::with_capacity(audio_embeds.len() * 4);
            for v in audio_embeds {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(dir.join("audio_embeds.f32"), bytes)?;
        }
        let gen = self.generate_from_embeds(audio_embeds, max_new_tokens, opts, stream)?;
        let host = self.decoder.host_ms();
        eprintln!(
            "mel={:.0}ms enc={:.0}ms prefill={:.0}ms decode={:.0}ms [host submit {:.0} / read {:.0}] tokens={} seq={}",
            t_mel_ms,
            t_enc_ms,
            gen.prefill_ms,
            gen.decode_ms,
            host.0,
            host.1,
            gen.ids.len(),
            gen.seq_len,
        );
        if let Some(dir) = dump_dir {
            let ids: String = gen.ids.iter().map(|id| format!("{id}\n")).collect();
            std::fs::write(dir.join("gen_ids.txt"), ids)?;
        }
        prompt::decode_result(&self.tokenizer, &gen.ids)
    }

    pub(crate) fn generate_from_embeds(
        &mut self,
        audio_embeds: &[f32],
        max_new_tokens: usize,
        opts: &TranscribeOptions,
        mut stream: Option<&mut dyn FnMut(StreamToken)>,
    ) -> Result<Generation> {
        let hs = self.config.thinker_config.text_config.hidden_size;
        let nat = audio_embeds.len() / hs;
        let language = opts.forced_language()?;
        let (input_ids, asp) = prompt::build_prompt(
            &self.tokenizer,
            self.config.thinker_config.audio_start_token_id,
            self.config.thinker_config.audio_token_id,
            self.config.thinker_config.audio_end_token_id,
            nat,
            language.as_deref(),
            &opts.context,
            None,
        )?;
        let seq_len = input_ids.len();
        anyhow::ensure!(
            seq_len + max_new_tokens + 8 <= self.decoder.max_seq(),
            "seq {seq_len} + max_new {max_new_tokens} exceeds decoder max_seq {}",
            self.decoder.max_seq()
        );

        let name = "thinker.model.embed_tokens.weight";
        let et = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing {name}"))?;
        anyhow::ensure!(et.shape.len() == 2 && et.shape[1] == hs, "embed shape {:?}", et.shape);
        let mut hidden_bytes: Vec<u8> = Vec::with_capacity(seq_len * hs * 2);
        for &id in &input_ids[..asp] {
            et.append_f16_row_le(id as usize, hs, &mut hidden_bytes)?;
        }
        for v in audio_embeds {
            hidden_bytes.extend_from_slice(&f16::from_f32(*v).to_le_bytes());
        }
        for &id in &input_ids[asp + nat..] {
            et.append_f16_row_le(id as usize, hs, &mut hidden_bytes)?;
        }
        anyhow::ensure!(hidden_bytes.len() == seq_len * hs * 2, "hidden {} bytes", hidden_bytes.len());

        let t2 = Instant::now();
        let first = self.decoder.prefill(&hidden_bytes, seq_len, 0)?;
        let t_prefill = t2.elapsed();

        let eos = [ENDOFTEXT_TOKEN_ID as i32, IM_END_TOKEN_ID as i32];
        let mut generated: Vec<u32> = Vec::new();
        let t3 = Instant::now();
        if !eos.contains(&first) {
            generated.push(first as u32);
            if let Some(cb) = stream.as_deref_mut() {
                emit_token(cb, &self.tokenizer, &generated)?;
            }
            for _ in 1..max_new_tokens {
                let tok = self.decoder.step()?;
                if eos.contains(&tok) {
                    break;
                }
                generated.push(tok as u32);
                if let Some(cb) = stream.as_deref_mut() {
                    emit_token(cb, &self.tokenizer, &generated)?;
                }
            }
        }
        let t_decode = t3.elapsed();
        Ok(Generation {
            ids: generated,
            seq_len,
            prefill_ms: t_prefill.as_secs_f64() * 1000.0,
            decode_ms: t_decode.as_secs_f64() * 1000.0,
        })
    }
}

pub(crate) struct Generation {
    pub ids: Vec<u32>,
    pub seq_len: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

fn write_f32_dump(dir: &Path, name: &str, data: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join(name), bytes)?;
    Ok(())
}

fn print_embed_diff(cpu: &[f32], gpu: &[f32], mel_frames: usize) {
    if cpu.len() != gpu.len() {
        eprintln!(
            "encoder compare: LENGTH MISMATCH cpu {} vs gpu {} (mel {mel_frames} frames)",
            cpu.len(),
            gpu.len()
        );
        return;
    }
    let n = cpu.len();
    let mut max_abs = 0.0f32;
    let mut rms = 0.0f64;
    let mut max_rel = 0.0f32;
    let mut over = 0usize;
    for i in 0..n {
        let d = (cpu[i] - gpu[i]).abs();
        max_abs = max_abs.max(d);
        rms += (d as f64) * (d as f64);
        let tol = 0.0625 + cpu[i].abs() * (1.0 / 512.0);
        if d > tol {
            over += 1;
        }
        max_rel = max_rel.max(d / (cpu[i].abs() + 1e-6));
    }
    rms = (rms / n as f64).sqrt();
    eprintln!(
        "encoder compare: {n} elements, max|d| {max_abs:.5}, rms {rms:.6}, max_rel {max_rel:.4}, \
         outside 0.0625+|x|*2^-9: {over}"
    );
}
