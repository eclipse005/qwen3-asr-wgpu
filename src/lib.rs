//! qwen3-asr-wgpu — a **standalone** wgpu inference backend for Qwen3-ASR.
//!
//! This crate is intentionally self-contained: it depends only on crates.io
//! packages, never on the CUDA crate it was developed alongside, and it reads
//! model weights from a directory the caller passes in.  Nothing here references
//! the parent repository, so the whole `wgpu/` tree can be renamed and moved
//! elsewhere without edits.
//!
//! Both towers run on wgpu (the CPU audio tower is kept as `--cpu-enc`, a
//! fallback and a reference).  Everything is verified against the Python
//! `-hf` reference: see the README's parity gates, and
//! `docs/design-tiled-prefill.md` for the long-context attention.
//!
//! The entry points are the reference pipeline's own three steps — see
//! [`processor`] — plus convenience wrappers that do all three in one call
//! ([`WgpuAsr::transcribe_file_opts`] and friends).

pub mod audio_encoder;
pub mod audio_encoder_gpu;
pub mod config;
pub mod cpu_tensor;
pub mod decoder;
pub mod golden;
pub mod gpu;
pub mod inference;
pub mod mel;
pub mod mrope;
pub mod processor;
pub mod prompt;
pub mod shaders;
pub mod weights;

pub use decoder::{TextConfig, WgpuTextDecoder};
pub use gpu::Gpu;
pub use inference::{supported_languages, StreamToken, TranscribeOptions, WgpuAsr};
pub use processor::{AsrTranscription, Decoded, ReturnFormat, TranscriptionRequest};
pub use prompt::TranscribeResult;
