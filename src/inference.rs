//! End-to-end transcribe: CPU audio encoder + wgpu text decoder.
//! Alignment target is the Python Transformers-native `-hf` greedy baseline.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use half::f16;
use tokenizers::Tokenizer;

use crate::audio_encoder::CpuAudioEncoder;
use crate::audio_encoder_gpu::GpuAudioEncoder;
use crate::config::AsrConfig;
use crate::decoder::{TextConfig, WgpuTextDecoder};
use crate::gpu::Gpu;
use crate::mel::{load_audio_wav, MelExtractor, HOP_LENGTH, MEL_SAMPLE_RATE, N_FFT};
use crate::mrope::{compute_mrope_cos_sin, text_positions};
use crate::prompt::{self, TranscribeResult, ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID};
use crate::weights;

/// KV + MRoPE size.  The binding limit, not this, is what really caps audio
/// length: the prefill's causal-attention scratch is O(s²) per head
/// (`scores`/`attn` in `decoder::prefill`), i.e. 16·s²·2 bytes each, which
/// passes `max_storage_buffer_binding_size` (2047 MiB here) at s ≈ 8 200.
/// `prefill` guards that explicitly; this constant just has to stay above it.
const DECODER_MAX_SEQ: usize = 9216;

/// Which audio-tower implementation a [`WgpuAsr`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderBackend {
    /// Host f32 reference (`gemm` crate + rayon).  Kept for A/B and regression
    /// work — every fixture is bit-matched against it.
    Cpu,
    /// wgpu audio tower — the **default** (see [`WgpuAsr::load`]); falls back to
    /// [`EncoderBackend::Cpu`] if the GPU tower cannot be built (with a printed
    /// reason).  No GPU device at all is not a case this has to handle: the text
    /// decoder needs one before the encoder is even built.
    Gpu,
}

/// Round-to-even pad of a width — mirrors `audio_encoder_gpu`'s mel stride.
fn mel_pad_even(v: usize) -> usize {
    if v % 2 == 1 { v + 1 } else { v }
}

/// Per-request knobs, mirroring upstream `Qwen3ASRModel.transcribe(audio,
/// context=…, language=…)`.  `context` is the hotword/bias text and goes into
/// the chat template's **system** message; `language` forces text-only output by
/// prefilling `language {Language}<asr_text>` after the assistant header.
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    pub context: String,
    /// `None` = let the model detect it.  Normalised and validated against
    /// [`prompt::SUPPORTED_LANGUAGES`] on use, like upstream.
    pub language: Option<String>,
}

impl TranscribeOptions {
    /// The canonical language name to prompt with, or `None` when the caller did
    /// not force one.  Mirrors the reference processor: a code (`"en"`) or a full
    /// name (`"English"`), either case, resolved to the canonical name; anything
    /// else is rejected.
    fn forced_language(&self) -> Result<Option<String>> {
        let Some(raw) = self.language.as_deref() else {
            return Ok(None);
        };
        if raw.trim().is_empty() {
            return Ok(None);
        }
        Ok(Some(prompt::resolve_language(raw.trim())?))
    }
}

/// The global chunk index a captured round starts at.
fn ch0_of(cap: &crate::audio_encoder_gpu::Capture, r: &crate::audio_encoder_gpu::ConvRound) -> usize {
    let _ = cap;
    r.chunk0
}

/// Per-stage difference accumulator: max, rms, and how many elements exceed the
/// tolerance, with the first offender for a legible failure.
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

pub struct WgpuAsr {
    config: AsrConfig,
    tokenizer: Tokenizer,
    encoder: CpuAudioEncoder,
    /// Present only when [`EncoderBackend::Gpu`] was requested *and* built.
    gpu_encoder: Option<GpuAudioEncoder>,
    mel: MelExtractor,
    tensors: std::collections::HashMap<String, weights::RawTensor>,
    decoder: WgpuTextDecoder,
}

impl WgpuAsr {
    /// Loads on the GPU audio tower (the default; it falls back to the CPU tower
    /// by itself if the device cannot build it).
    pub fn load(model_dir: &Path, adapter: Option<&str>) -> Result<Self> {
        Self::load_with(model_dir, adapter, EncoderBackend::Gpu)
    }

    pub fn load_with(
        model_dir: &Path,
        adapter: Option<&str>,
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
        let gpu = pollster::block_on(Gpu::new(adapter))?;
        let gpu_encoder = match backend {
            EncoderBackend::Cpu => None,
            EncoderBackend::Gpu => {
                let ac = &config.thinker_config.audio_config;
                // `n_window_infer` is the attention window in *mel frames*
                // (800 shipped = 8 chunks of 100).
                let window_infer = ac.n_window_infer.max(ac.n_window * 2);
                match GpuAudioEncoder::load(&gpu, &tensors, "thinker.audio_tower", ac, window_infer) {
                    Ok(e) => Some(e),
                    Err(e) => {
                        eprintln!("[gpu encoder] unavailable, falling back to CPU: {e:#}");
                        None
                    }
                }
            }
        };
        let text_cfg = TextConfig::from_model_dir(model_dir)?;
        let decoder = WgpuTextDecoder::load(
            gpu,
            model_dir,
            "thinker.model",
            text_cfg,
            DECODER_MAX_SEQ,
            DECODER_MAX_SEQ,
        )?;
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

    /// True when the GPU audio tower is loaded (i.e. `--gpu-enc` took effect).
    pub fn gpu_encoder_active(&self) -> bool {
        self.gpu_encoder.is_some()
    }


    pub fn transcribe(&mut self, wav: &Path, max_new_tokens: usize) -> Result<TranscribeResult> {
        self.transcribe_with_dump(wav, max_new_tokens, None)
    }

    /// File entry point with an optional CPU-vs-GPU encoder comparison.
    pub fn transcribe_file(
        &mut self,
        wav: &Path,
        max_new_tokens: usize,
        dump_dir: Option<&Path>,
        compare_enc: bool,
    ) -> Result<TranscribeResult> {
        self.transcribe_file_opts(wav, max_new_tokens, dump_dir, compare_enc, &TranscribeOptions::default())
    }

    /// [`Self::transcribe_file`] with upstream's `context` / `language` knobs.
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

    /// Mel-in entry point with an optional CPU-vs-GPU encoder comparison.
    pub fn transcribe_from_mel_cmp(
        &mut self,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
        max_new_tokens: usize,
        dump_dir: Option<&Path>,
        compare_enc: bool,
    ) -> Result<TranscribeResult> {
        self.transcribe_from_mel_cmp_opts(
            mel, n_mels, n_frames, max_new_tokens, dump_dir, compare_enc, &TranscribeOptions::default(),
        )
    }

    /// [`Self::transcribe_from_mel_cmp`] with upstream's options.
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
        )
    }

    /// Mel frontend only (probe hook — no GPU work).
    pub fn extract_mel(&self, samples: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        self.mel.extract(samples)
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

    /// The GPU audio tower, when one was built.
    pub fn gpu_encoder(&self) -> Option<&GpuAudioEncoder> {
        self.gpu_encoder.as_ref()
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
        let gpu = self.decoder.gpu();
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

        // The GPU's inputs are f16, so the CPU is fed *exactly* what the GPU
        // read: the f16-rounded mel for level 1, and the GPU's own activation
        // for the deeper levels.  Anything the comparison then shows is a bug in
        // the gather/GEMM/GELU, not the operand rounding.
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
                            // The two CPU tensors have *different* layouts: `raw`
                            // is the GEMM's own `[c_out][all positions]` (what the
                            // GPU's buffer holds), `act` is `[chunk][c_out][plane]`
                            // (the shape `conv_block` returns).  Using the raw
                            // stride for both is correct exactly when there is one
                            // chunk — which is how it went unnoticed.
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

            // Next level's input: the GPU's activation, de-padded into the
            // CPU's `[chunk][c][h][w]` layout.
            let (c_out, plane, plane_pad, n_all) = (v.c_out, v.plane, v.plane_pad, v.n_all);
            // `conv_block_stages`/`im2col_3x3_s2p1` take `[chunk][c_in][h][w]`,
            // i.e. one channel plane per chunk — *not* the GEMM's channel-major
            // `[c][all positions]`.  The two agree only at one chunk, which is
            // exactly where this was first tested.
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

        // The whole-chain (CPU f32) comparison: coarser, but it covers the
        // permute and the projection, where a layout slip is O(1).
        let tower = self.encoder.conv_tower(mel, n_mels, n_frames)?;
        let cf = tower.packed.len().checked_div(tower.n_total).unwrap_or(0);
        let dm = self.encoder.config().d_model;
        // Self-contained attention oracle: rebuild a (head, window) block's
        // scores, softmax and AV from the GPU's own Q/K/V.  Every block is the
        // *same* arithmetic — the head slice and the window bound are the parts
        // that can differ — so it runs head 0 *and* head 1, first and last
        // window, and prints the Q/K/V magnitudes (an all-zero actor block makes
        // every one of these comparisons agree for free).
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
            // `z = head·n_win + win`; the last window is the short one.
            for (head, win) in [(0usize, 0usize), (1, 0), (0, a.n_win - 1), (1, a.n_win - 1)] {
                let z = head * a.n_win + win;
                let valid = wlen.min(a.s - win * wlen);
                let rows = valid.min(8);
                let sc0 = dsc.bad;
                let (mut pbad, mut abad, mut first_bad) = (0usize, 0usize, usize::MAX);
                // The packed operands against the host's Q/K/V: a wrong `qp` is a
                // prep bug (the re-pack), a right `qp` with wrong scores is the
                // GEMM's.
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
                        // `scores` holds the raw dots: the softmax kernel applies
                        // `scale` itself, and comparing against a pre-scaled host
                        // value is off by 1/scale (8 at hd 64).
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
                        // f16 storage: the ulp at |s| 300 is 0.25, so the score
                        // tolerance has to follow the magnitude.
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
            // The CPU's LN1 / QKV / attention block on the same tokens.
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
                // `attn_flat` is the flattened attention *before* `out_proj`, so
                // the CPU side has to stop at the same stage (comparing it with
                // `dbg_attn` compared a pre-projection tensor with a projected
                // one — a guaranteed mismatch that says nothing about the GPU).
                let want_attn = self.encoder.dbg_attn_flat(0, &want_normed, tower.n_total)?;
                for t in 0..tower.n_total {
                    for j in 0..dm {
                        da.add(t, j, a.attn_flat[t * a.acols + j].to_f32(), want_attn[t * dm + j], 3e-2);
                    }
                }
                // Split layer 0: the attention residual (before the FFN) and
                // the FFN residual (after it).
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

        // Per-layer: the CPU runs layer `li` on exactly the tokens the GPU fed
        // it, so any mismatch is that layer's — attention, residual or FFN.
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
        // Loose per-level check: the GPU's activations against the CPU's *own*
        // chain, so the f16 rounding each level feeds the next one shows up.
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
        // Same projection, but from the GPU's own operand: separates a
        // conv_out/PE bug from the f16 drift upstream of it.
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

    /// Run a mel through the configured tower, returning f32 rows.
    /// `compare` additionally runs the CPU reference and prints the envelope.
       fn run_encoder(
        &mut self,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
        compare: bool,
    ) -> Result<Vec<f32>> {
        let Some(enc) = self.gpu_encoder.as_mut() else {
            return self.encoder.forward(mel, n_mels, n_frames);
        };
        let t = Instant::now();
        let out16 = enc.encode(self.decoder.gpu(), mel, n_mels, n_frames)?;
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

    pub fn transcribe_with_dump(
        &mut self,
        wav: &Path,
        max_new_tokens: usize,
        dump_dir: Option<&Path>,
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
        self.transcribe_samples_with_dump(&samples, max_new_tokens, dump_dir)
    }

    pub fn transcribe_samples(&mut self, samples: &[f32], max_new_tokens: usize) -> Result<TranscribeResult> {
        self.transcribe_samples_with_dump(samples, max_new_tokens, None)
    }

    pub fn transcribe_from_mel(
        &mut self,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
        max_new_tokens: usize,
        dump_dir: Option<&Path>,
    ) -> Result<TranscribeResult> {
        let t1 = Instant::now();
        let audio_embeds = self.run_encoder(mel, n_mels, n_frames, false)?;
        let t_enc = t1.elapsed();
        self.decode_from_audio_embeds(
            &audio_embeds,
            max_new_tokens,
            dump_dir,
            0.0,
            t_enc.as_secs_f64() * 1000.0,
            &TranscribeOptions::default(),
        )
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
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_from_audio_embeds(
        &mut self,
        audio_embeds: &[f32],
        max_new_tokens: usize,
        dump_dir: Option<&Path>,
        t_mel_ms: f64,
        t_enc_ms: f64,
        opts: &TranscribeOptions,
    ) -> Result<TranscribeResult> {
        let hs = self.config.thinker_config.text_config.hidden_size;
        let nat = audio_embeds.len() / hs;
        if let Some(dir) = dump_dir {
            std::fs::create_dir_all(dir)?;
            let mut bytes = Vec::with_capacity(audio_embeds.len() * 4);
            for v in audio_embeds {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(dir.join("audio_embeds.f32"), bytes)?;
        }
        // Upstream normalises and validates a forced language before prompting,
        // and only then appends `language X<asr_text>` to the assistant turn.
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
            seq_len + max_new_tokens + 8 <= self.decoder.max_seq,
            "seq {seq_len} + max_new {max_new_tokens} exceeds decoder max_seq {}",
            self.decoder.max_seq
        );

        let name = "thinker.model.embed_tokens.weight";
        let et = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing {name}"))?;
        anyhow::ensure!(et.shape.len() == 2 && et.shape[1] == hs, "embed shape {:?}", et.shape);
        // Rows come straight out of the mapped file.  Going through
        // `weights::get_f16` re-materialised the whole table (155 M elements at
        // 0.6 B) as a fresh `Vec<f16>` on every call — ~0.3 s of the 12 s run
        // (twice that at 1.7 B).
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
            for _ in 1..max_new_tokens {
                let tok = self.decoder.step()?;
                if eos.contains(&tok) {
                    break;
                }
                generated.push(tok as u32);
            }
        }
        let t_decode = t3.elapsed();
        eprintln!(
            "mel={:.0}ms enc={:.0}ms prefill={:.0}ms decode={:.0}ms tokens={} seq={}",
            t_mel_ms,
            t_enc_ms,
            t_prefill.as_secs_f64() * 1000.0,
            t_decode.as_secs_f64() * 1000.0,
            generated.len(),
            seq_len,
        );
        if let Some(dir) = dump_dir {
            let ids: String = generated.iter().map(|id| format!("{id}\n")).collect();
            std::fs::write(dir.join("gen_ids.txt"), ids)?;
        }
        prompt::decode_result(&self.tokenizer, &generated)
    }
}

/// Diagnostic envelope of one stage against the CPU reference.
fn stage_diff(name: &str, reference: &[f32], got: &[f32], shape: &[usize]) {
    let n = reference.len().min(got.len());
    if reference.len() != got.len() {
        eprintln!(
            "  {name:<24} SIZE MISMATCH ref {} vs gpu {} (shape {shape:?})",
            reference.len(),
            got.len()
        );
    }
    let mut max_abs = 0.0f32;
    let mut rms = 0.0f64;
    let mut ref_rms = 0.0f64;
    let mut worst = 0usize;
    for i in 0..n {
        let d = (reference[i] - got[i]).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
        rms += (d as f64) * (d as f64);
        ref_rms += (reference[i] as f64) * (reference[i] as f64);
    }
    let rms = (rms / n.max(1) as f64).sqrt();
    let ref_rms = (ref_rms / n.max(1) as f64).sqrt();
    eprintln!(
        "  {name:<24} n={n} max|d| {max_abs:.6} (idx {worst}: ref {:.5} gpu {:.5}) \
         rms {rms:.6} vs ref-rms {ref_rms:.6}",
        reference[worst],
        got[worst]
    );
}

/// Write a flat f32 tensor as little-endian bytes.
fn write_f32_dump(dir: &Path, name: &str, data: &[f32]) -> Result<()> {    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join(name), bytes)?;
    Ok(())
}

/// Diagnostic envelope of the GPU tower's f16 embeddings against the CPU
/// reference's f32 ones — the alignment gate for the audio path.
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



