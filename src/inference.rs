//! End-to-end transcribe: CPU audio encoder + wgpu text decoder.
//! Alignment target is the Python Transformers-native `-hf` greedy baseline.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use half::f16;
use tokenizers::Tokenizer;

use crate::audio_encoder::CpuAudioEncoder;
use crate::config::AsrConfig;
use crate::decoder::{TextConfig, WgpuTextDecoder};
use crate::gpu::Gpu;
use crate::mel::{load_audio_wav, MelExtractor, HOP_LENGTH, MEL_SAMPLE_RATE, N_FFT};
use crate::mrope::{compute_mrope_cos_sin, text_positions};
use crate::prompt::{self, TranscribeResult, ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID};
use crate::weights;

/// KV + MRoPE size; 180s prompt (~2355) + 1024 new tokens fits with margin.
const DECODER_MAX_SEQ: usize = 4096;

pub struct WgpuAsr {
    config: AsrConfig,
    tokenizer: Tokenizer,
    encoder: CpuAudioEncoder,
    mel: MelExtractor,
    tensors: std::collections::HashMap<String, weights::RawTensor>,
    decoder: WgpuTextDecoder,
}

impl WgpuAsr {
    pub fn load(model_dir: &Path, adapter: Option<&str>) -> Result<Self> {
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
            mel: MelExtractor::new(N_FFT, HOP_LENGTH, n_mels, MEL_SAMPLE_RATE),
            tensors,
            decoder,
        })
    }

    pub fn transcribe(&mut self, wav: &Path, max_new_tokens: usize) -> Result<TranscribeResult> {
        self.transcribe_with_dump(wav, max_new_tokens, None)
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
        let audio_embeds = self.encoder.forward(mel, n_mels, n_frames)?;
        let t_enc = t1.elapsed();
        self.decode_from_audio_embeds(
            &audio_embeds,
            max_new_tokens,
            dump_dir,
            0.0,
            t_enc.as_secs_f64() * 1000.0,
        )
    }

    pub fn transcribe_from_embeds(
        &mut self,
        audio_embeds: &[f32],
        max_new_tokens: usize,
        dump_dir: Option<&Path>,
    ) -> Result<TranscribeResult> {
        self.decode_from_audio_embeds(audio_embeds, max_new_tokens, dump_dir, 0.0, 0.0)
    }

    pub fn transcribe_samples_with_dump(
        &mut self,
        samples: &[f32],
        max_new_tokens: usize,
        dump_dir: Option<&Path>,
    ) -> Result<TranscribeResult> {
        let t0 = Instant::now();
        let (mel, n_mels, n_frames) = self.mel.extract(samples)?;
        let t_mel = t0.elapsed();

        let t1 = Instant::now();
        let audio_embeds = self.encoder.forward(&mel, n_mels, n_frames)?;
        let t_enc = t1.elapsed();
        if let Some(dir) = dump_dir {
            std::fs::create_dir_all(dir)?;
            let mut bytes = Vec::with_capacity(mel.len() * 4);
            for v in &mel {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(dir.join("mel.f32"), &bytes)?;
            let mut bytes = Vec::with_capacity(audio_embeds.len() * 4);
            for v in &audio_embeds {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(dir.join("audio_embeds.f32"), &bytes)?;
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
        )
    }

    fn decode_from_audio_embeds(
        &mut self,
        audio_embeds: &[f32],
        max_new_tokens: usize,
        dump_dir: Option<&Path>,
        t_mel_ms: f64,
        t_enc_ms: f64,
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
        let (input_ids, asp) = prompt::build_prompt(
            &self.tokenizer,
            self.config.thinker_config.audio_start_token_id,
            self.config.thinker_config.audio_token_id,
            self.config.thinker_config.audio_end_token_id,
            nat,
            None,
            None,
        )?;
        let seq_len = input_ids.len();
        anyhow::ensure!(
            seq_len + max_new_tokens + 8 <= self.decoder.max_seq,
            "seq {seq_len} + max_new {max_new_tokens} exceeds decoder max_seq {}",
            self.decoder.max_seq
        );

        let (embed, eshape) = weights::get_f16(&self.tensors, "thinker.model.embed_tokens.weight")?;
        anyhow::ensure!(eshape.len() == 2 && eshape[1] == hs, "embed shape {eshape:?}");
        let mut hidden: Vec<f16> = Vec::with_capacity(seq_len * hs);
        let lookup = |id: i64, out: &mut Vec<f16>| {
            let row = id as usize;
            out.extend_from_slice(&embed[row * hs..(row + 1) * hs]);
        };
        for &id in &input_ids[..asp] {
            lookup(id, &mut hidden);
        }
        for v in audio_embeds {
            hidden.push(f16::from_f32(*v));
        }
        for &id in &input_ids[asp + nat..] {
            lookup(id, &mut hidden);
        }
        anyhow::ensure!(hidden.len() == seq_len * hs);

        let mut hidden_bytes = Vec::with_capacity(hidden.len() * 2);
        for h in &hidden {
            hidden_bytes.extend_from_slice(&h.to_bits().to_le_bytes());
        }

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
        prompt::decode_result(&self.tokenizer, &generated, None)
    }
}
