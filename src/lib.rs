//! qwen3-asr-wgpu — a **standalone** wgpu inference backend for Qwen3-ASR.
//!
//! This crate is intentionally self-contained: it depends only on crates.io
//! packages, never on the CUDA crate it was developed alongside, and it reads
//! model weights from a directory the caller passes in.  Nothing here references
//! the parent repository, so the whole `wgpu/` tree can be renamed and moved
//! elsewhere without edits.
//!
//! Text decoder (prefill + decode) runs on wgpu. Audio encoder currently uses
//! the CPU reference (same math as qwen3-asr-rs) so end-to-end transcripts can
//! be checked against the Python `-hf` baseline while the GPU audio tower is
//! still being ported. See `ROADMAP-wgpu.md`.

pub mod audio_encoder;
pub mod config;
pub mod cpu_tensor;
pub mod decoder;
pub mod golden;
pub mod gpu;
pub mod inference;
pub mod mel;
pub mod mrope;
pub mod prompt;
pub mod shaders;
pub mod weights;

pub use decoder::{TextConfig, WgpuTextDecoder};
pub use gpu::Gpu;
pub use inference::WgpuAsr;
pub use prompt::TranscribeResult;
