# Qwen3-ASR wgpu

**Qwen3-ASR speech recognition in Rust with wgpu.**

[![GitHub stars](https://img.shields.io/github/stars/eclipse005/qwen3-asr-wgpu?style=flat-square&logo=github&color=f59e0b)](https://github.com/eclipse005/qwen3-asr-wgpu/stargazers) [![GitHub forks](https://img.shields.io/github/forks/eclipse005/qwen3-asr-wgpu?style=flat-square&logo=github)](https://github.com/eclipse005/qwen3-asr-wgpu/network/members) [![Build](https://img.shields.io/github/actions/workflow/status/eclipse005/qwen3-asr-wgpu/build.yml?branch=master&style=flat-square&label=build)](https://github.com/eclipse005/qwen3-asr-wgpu/actions/workflows/build.yml) [![License](https://img.shields.io/github/license/eclipse005/qwen3-asr-wgpu?style=flat-square&color=16a34a)](LICENSE)

[![Windows](https://img.shields.io/badge/Windows-x64-0078D6?style=flat-square&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iI2ZmZiI%2BPHBhdGggZD0iTTAsMEgxMS4zNzdWMTEuMzcySDBaTTEyLjYyMywwSDI0VjExLjM3MkgxMi42MjNaTTAsMTIuNjIzSDExLjM3N1YyNEgwWm0xMi42MjMsMEgyNFYyNEgxMi42MjMiLz48L3N2Zz4%3D)](#install) [![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-111111?style=flat-square&logo=apple&logoColor=white)](#install) [![Linux](https://img.shields.io/badge/Linux-x86__64-FCC624?style=flat-square&logo=linux&logoColor=111111)](#install)

**English** · [简体中文](README.zh-CN.md)

A lightweight, cross-platform Rust implementation of [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), using [wgpu](https://github.com/gfx-rs/wgpu) for GPU acceleration.

The goal is simple: run Qwen3-ASR **locally and natively** without Python or vendor-specific GPU runtimes.

### Features

* 🦀 Pure Rust
* 🎮 GPU acceleration with wgpu
* 🌍 Cross-platform GPU support
* 🖥️ Windows / macOS / Linux
* ⚡ CPU fallback
* 📦 Offline local inference
* 🎙️ Qwen3-ASR 0.6B / 1.7B
* 🔢 INT8 support
* 📡 Streaming inference
* 🧩 CLI + Rust library

### Install

As a Cargo dependency:

```toml
[dependencies]
qwen3-asr-wgpu = { git = "https://github.com/eclipse005/qwen3-asr-wgpu.git" }
```

Or build the CLI from source:

```bash
git clone https://github.com/eclipse005/qwen3-asr-wgpu.git
cd qwen3-asr-wgpu
cargo build --release        # target/release/transcribe
```

| Feature | Description |
|---------|-------------|
| `hub` | Lets the program download models from Hugging Face itself (`AsrInference::from_pretrained`). Off by default |

### Model download

Weights are **not** included in this repository. Download the `-hf` checkpoints from Hugging Face (rights remain with the original authors) and pass the downloaded directory to `--model` unchanged:

- [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

INT8 quantized checkpoints load exactly like the fp16 ones (auto-detected at load, with lower VRAM use and faster decoding):

- [eclipse005/Qwen3-ASR-0.6B-int8](https://modelscope.cn/models/eclipse005/Qwen3-ASR-0.6B-int8) (ModelScope)
- [eclipse005/Qwen3-ASR-1.7B-int8](https://modelscope.cn/models/eclipse005/Qwen3-ASR-1.7B-int8) (ModelScope)

### Quick Start

```bash
transcribe --model ./Qwen3-ASR-0.6B-hf --wav ./audio.wav
```

The run prints the device in use (GPU or CPU), timings and the recognized language, followed by the transcript.

| Option | Description |
|--------|-------------|
| `--model <dir>` | Model directory (or the `QASR_MODEL` environment variable) |
| `--wav <file>` | Audio to transcribe; any sample rate, resampled to 16 kHz internally |
| `--lang <name>` | Language such as `zh` or `English`; omit for automatic detection |
| `--prompt <text>` | Context / hot words (domain terms, names) that bias transcription through the system message |
| `--max-new <n>` | Generation cap, default 2048 |
| `--device <name>` | `auto` (default: discrete GPU, then integrated, else CPU), `cpu`, `gpu` (discrete then integrated, error if none opens), or `vulkan[:i]` / `dx12` / a name substring |
| `--list-devices` | List the devices usable on this machine |

### Library

```rust
use qwen3_asr_wgpu::{AsrInference, Backend, TranscribeOptions};

let asr = AsrInference::load("Qwen3-ASR-0.6B-hf".as_ref(), Backend::best())?;
let out = asr.transcribe("audio.wav", TranscribeOptions::default())?;
println!("[{}] {}", out.language, out.text);
```

A model is loaded once and reused across calls; one instance can be shared by several threads (calls are queued internally). Also available:

* Streaming — `transcribe_streaming(path, opts, cb)` and `create_streaming_session()` for feed-as-you-go input
* Context / hot words — `TranscribeOptions::default().with_language("English").with_context("Vocabulary: Quilter, apostle, gospel.")`
* Device introspection — `AsrInference::devices()`, `device_description()`, `supported_languages()`
* The official three-step flow — `apply_transcription_request` → `generate` → `decode`, with `ReturnFormat::Raw | Parsed | TranscriptionOnly`

See `cargo doc` for the full API.

### Why wgpu?

Instead of relying on CUDA, ROCm, or other vendor-specific runtimes, this project uses **wgpu** as a unified GPU abstraction.

This makes it possible to build a single Rust-based ASR runtime for different platforms and GPU vendors.

### Project Status

🚧 **Active development**

Performance and hardware compatibility are still being actively optimized and tested across different GPUs.

### Related

* [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) — the official model project
* [wgpu](https://github.com/gfx-rs/wgpu)
* [qwen3-aligner-wgpu](https://github.com/eclipse005/qwen3-aligner-wgpu) — word-level timestamps (forced alignment)

### License

Apache-2.0, matching the upstream project.

This repository is an **independent Rust inference implementation** for loading and running the officially released Qwen3-ASR weights — not an official Alibaba / Qwen release, and not affiliated with the original authors. Model weights remain under the terms of their respective owners.
