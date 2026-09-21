//! A wgpu inference engine for Qwen3-ASR.
//!
//! Both towers run on wgpu — the host audio tower is kept as an
//! [`EncoderBackend::Cpu`] fallback — so one code path drives Vulkan, D3D12 and
//! Metal.  Weights are read from a checkpoint directory the caller passes in.
//!
//! # Using it
//!
//! ```no_run
//! use qwen3_asr_wgpu::{AsrInference, Backend, TranscribeOptions};
//!
//! # fn main() -> qwen3_asr_wgpu::Result<()> {
//! let asr = AsrInference::load(std::path::Path::new("Qwen3-ASR-0.6B-hf"), Backend::best())?;
//! let out = asr.transcribe("clip.wav", TranscribeOptions::default())?;
//! println!("[{}] {}", out.language, out.text);
//! # Ok(())
//! # }
//! ```
//!
//! [`AsrInference`] is the entry point: it owns the model and takes `&self`, so
//! one instance can be wrapped in an `Arc` and shared.  Three knobs narrow the
//! choice of hardware, from coarse to fine — [`Backend`] (one word),
//! [`DeviceSelector`] (runtime and index) and [`EncoderBackend`] (which audio
//! tower).
//!
//! # Layout
//!
//! * [`AsrInference`] plus its types ([`TranscribeOptions`],
//!   [`TranscribeResult`], [`StreamToken`], [`AsrStreamingSession`],
//!   [`AsrError`]) — the library API.
//! * [`processor`] — the reference processor's own three steps
//!   (`apply_transcription_request` → `generate` → `decode`).
//! * [`diagnostics`] — probe and dump hooks.  Not part of the library API;
//!   they exist so the `--mel` / `--diag-enc` / `--dump` paths in
//!   `src/bin/transcribe.rs` can reach the same engine.
//! * The remaining public modules are engine internals (`decoder`, `shaders`,
//!   `audio_encoder*`, `gpu`, `weights`, …).  They are public because the probe
//!   binaries in `src/bin/` are separate crates that drive them directly; a
//!   library user does not need them.

mod backend;
mod error;
mod streaming;
#[cfg(feature = "hub")]
mod hub;

pub mod diagnostics;

pub mod processor;

pub mod audio_encoder;
pub mod audio_encoder_gpu;
pub mod config;
pub mod cpu_decoder;
pub mod cpu_tensor;
pub mod decoder;
pub mod gpu;
pub mod inference;
pub mod load_trace;
pub mod mel;
pub mod mrope;
pub mod prompt;
pub mod shaders;
pub mod weights;

pub use backend::Backend;
pub use error::{AsrError, Result};
pub use gpu::{list_devices, DeviceInfo, DeviceSelector, DeviceTarget, Gpu};
pub use inference::{
    resolve_model_dir, supported_languages, AsrInference, EncoderBackend, StreamToken,
    TranscribeOptions,
};
pub use mel::load_audio_wav;
pub use processor::{AsrTranscription, Decoded, ReturnFormat, TranscriptionRequest};
pub use prompt::TranscribeResult;
pub use streaming::AsrStreamingSession;
