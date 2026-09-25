# Qwen3-ASR wgpu

**基于 wgpu 的 Qwen3-ASR 语音识别（Rust 实现）。**

[English](README.md) · **简体中文**

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 的轻量级跨平台 Rust 实现，使用 [wgpu](https://github.com/gfx-rs/wgpu) 进行 GPU 加速。

目标很简单：让 Qwen3-ASR **在本地原生运行**，不依赖 Python，也不依赖特定厂商的 GPU 运行时。

### 特性

* 🦀 纯 Rust
* 🎮 wgpu GPU 加速
* 🌍 跨平台 GPU 支持
* 🖥️ Windows / macOS / Linux
* ⚡ CPU 回退
* 📦 离线本地推理
* 🎙️ Qwen3-ASR 0.6B / 1.7B
* 🔢 INT8 量化支持
* 📡 流式推理
* 🧩 CLI + Rust 库

### 安装

作为 Cargo 依赖：

```toml
[dependencies]
qwen3-asr-wgpu = { git = "https://github.com/eclipse005/qwen3-asr-wgpu.git" }
```

或从源码构建 CLI：

```bash
git clone https://github.com/eclipse005/qwen3-asr-wgpu.git
cd qwen3-asr-wgpu
cargo build --release        # target/release/transcribe
```

| Feature | 说明 |
|---------|------|
| `hub` | 让程序自行从 Hugging Face 下载模型（`AsrInference::from_pretrained`）。默认关闭 |

### 模型下载

权重**不在**本仓库内。请从 Hugging Face 下载 `-hf` 版本的检查点（版权归原作者），下载得到的目录直接作为 `--model` 传入：

- [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

INT8 量化版权重与 fp16 加载方式完全相同（加载时自动识别，显存占用更低、解码更快）：

- [eclipse005/Qwen3-ASR-0.6B-int8](https://modelscope.cn/models/eclipse005/Qwen3-ASR-0.6B-int8)（ModelScope）
- [eclipse005/Qwen3-ASR-1.7B-int8](https://modelscope.cn/models/eclipse005/Qwen3-ASR-1.7B-int8)（ModelScope）

### 快速上手

```bash
transcribe --model ./Qwen3-ASR-0.6B-hf --wav ./audio.wav
```

运行时会打印所用设备（GPU 或 CPU）、耗时与识别出的语言，随后输出转写文本。

| 参数 | 说明 |
|------|------|
| `--model <dir>` | 模型目录（或环境变量 `QASR_MODEL`） |
| `--wav <file>` | 要转写的音频；任意采样率，内部自动重采样到 16 kHz |
| `--lang <name>` | 语言，如 `zh`、`English`；不填则自动识别 |
| `--prompt <text>` | 上下文 / 热词（领域词、人名），经 system 消息偏置转写结果 |
| `--max-new <n>` | 生成上限，默认 2048 |
| `--device <name>` | 强制指定设备；`cpu` 表示强制 CPU。默认自动挑选最佳设备 |
| `--list-devices` | 列出本机可用设备 |

### 作为库使用

```rust
use qwen3_asr_wgpu::{AsrInference, Backend, TranscribeOptions};

let asr = AsrInference::load("Qwen3-ASR-0.6B-hf".as_ref(), Backend::best())?;
let out = asr.transcribe("audio.wav", TranscribeOptions::default())?;
println!("[{}] {}", out.language, out.text);
```

模型只需加载一次，之后可反复调用；一个实例可被多个线程共享（内部排队执行）。此外还提供：

* 流式推理 —— `transcribe_streaming(path, opts, cb)` 与 `create_streaming_session()`，支持边喂音频边出结果
* 上下文 / 热词 —— `TranscribeOptions::default().with_language("English").with_context("Vocabulary: Quilter, apostle, gospel.")`
* 设备查询 —— `AsrInference::devices()`、`device_description()`、`supported_languages()`
* 官方三段式调用 —— `apply_transcription_request` → `generate` → `decode`，配合 `ReturnFormat::Raw | Parsed | TranscriptionOnly`

完整 API 见 `cargo doc`。

### 为什么选择 wgpu？

本项目不依赖 CUDA、ROCm 或其他特定厂商的运行时，而是以 **wgpu** 作为统一的 GPU 抽象层。

这使得同一套 Rust 语音识别运行时可以覆盖不同平台与不同显卡厂商。

### 项目状态

🚧 **积极开发中**

性能与硬件兼容性仍在不同显卡上持续优化与测试。

### 相关项目

* [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) —— 官方模型项目
* [wgpu](https://github.com/gfx-rs/wgpu)
* [qwen3-aligner-wgpu](https://github.com/eclipse005/qwen3-aligner-wgpu) —— 词级时间戳（强制对齐）

### 许可证

Apache-2.0，与上游一致。

本仓库是**独立的 Rust 推理实现**，用于加载并运行官方发布的 Qwen3-ASR 权重，并非 Alibaba / Qwen 官方发行版，与原作者无隶属关系。模型权重版权归原作者所有。
