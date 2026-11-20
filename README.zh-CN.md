# Qwen3-ASR wgpu

**基于 wgpu 的 Qwen3-ASR 语音识别（Rust 实现）。**

[![GitHub stars](https://img.shields.io/github/stars/eclipse005/qwen3-asr-wgpu?style=flat-square&logo=github&color=f59e0b)](https://github.com/eclipse005/qwen3-asr-wgpu/stargazers) [![GitHub forks](https://img.shields.io/github/forks/eclipse005/qwen3-asr-wgpu?style=flat-square&logo=github)](https://github.com/eclipse005/qwen3-asr-wgpu/network/members) [![Build](https://img.shields.io/github/actions/workflow/status/eclipse005/qwen3-asr-wgpu/build.yml?branch=master&style=flat-square&label=build)](https://github.com/eclipse005/qwen3-asr-wgpu/actions/workflows/build.yml) [![License](https://img.shields.io/github/license/eclipse005/qwen3-asr-wgpu?style=flat-square&color=16a34a)](LICENSE)

[![Windows](https://img.shields.io/badge/Windows-x64-0078D6?style=flat-square&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iI2ZmZiI%2BPHBhdGggZD0iTTAsMEgxMS4zNzdWMTEuMzcySDBaTTEyLjYyMywwSDI0VjExLjM3MkgxMi42MjNaTTAsMTIuNjIzSDExLjM3N1YyNEgwWm0xMi42MjMsMEgyNFYyNEgxMi42MjMiLz48L3N2Zz4%3D)](#安装) [![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-111111?style=flat-square&logo=apple&logoColor=white)](#安装) [![Linux](https://img.shields.io/badge/Linux-x86__64-FCC624?style=flat-square&logo=linux&logoColor=111111)](#安装)

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
| `--max-new <n>` | 生成上限（token 数），默认 2048 |
| `--device <name>` | `auto`（默认：先独显再集显，都打不开才用 CPU）、`cpu`、`gpu`（先独显再集显，没有或打不开就报错），或 `vulkan[:i]` / `dx12` / 适配器名子串 |
| `--list-devices` | 列出本机可用设备 |

### 批量转写（多段并发解码）

上游已经用 VAD 分好段（比如 FireRedVAD：60 s 目标块、在静音中点切、单段上限 3 分钟）时，
用 `transcribe_batch` 一次喂一个列表，返回**同序**的文本：

```bash
transcribe_batch --model ./Qwen3-ASR-1.7B-hf --batch 16 \
                 --wav segs/part000.wav segs/part001.wav ... --out texts.txt
```

| 参数 | 说明 |
|------|------|
| `--wav <file...>` | 音频列表，顺序即输出顺序 |
| `--batch <n>` | 同时解码几段，默认 `1`（逐段路径） |
| `--max-new <n>` | 每段生成的 **token** 上限（不是秒），默认 256；60 s 段约需 150–260，跑完看 `stop=max-tokens` 判断是否够用 |
| `--compare` | 再跑一遍 `--batch 1` 并逐段 diff，有差异即非零退出 |
| `--timings` | 打印 `mel / encoder / prefill / decode` 四段耗时 |
| `--out <file>` | 把文本按行写入文件（与输入同序） |
| `--model` / `--device` | 同 `transcribe` |

`--batch N` 指同时解码几段：默认 1 走逐段路径；`N>1` 时每一步的权重只读一次、喂给 N 段，
显存随 N 线性增长（每段一份 KV），按自己显卡显存选即可。

**输出与 `--batch 1` 逐字节相同** —— 不是近似，是构造上保证：每个 slot 读的是它自己那块
buffer（`binding_array` 按 `wgpu.z` 索引），batch 只改变"一次 dispatch 覆盖几段"，不改变
任何一段的累加顺序。加 `--compare` 会自己再跑一遍 `--batch 1` 并逐段 diff，任何一段不同
即非零退出。

| 限制 | 值 |
|---|---|
| 每批槽位上限 | 32（`binding_array` 宽度）；再大不会更快，只多占显存 |
| 批量路径的单段长度 | `prompt + max_new ≤ 4096`（受每槽 KV 显存限制）；超出的段自动退回逐段路径，不会失败 |
| 显存 | 约 `模型 4.8 GiB + 0.2 GiB × N`（60 s 段的典型值） |

库里的等价入口是 `AsrInference::transcribe_wavs_batch(wavs, opts, batch)`（以及带阶段
计时的 `transcribe_wavs_batch_timed`）。

### 作为库使用

```rust
use qwen3_asr_wgpu::{AsrInference, Backend, TranscribeOptions};

let asr = AsrInference::load("Qwen3-ASR-0.6B-hf".as_ref(), Backend::best())?;
let out = asr.transcribe("audio.wav", TranscribeOptions::default())?;
println!("[{}] {}", out.language, out.text);
```

模型只需加载一次，之后可反复调用；一个实例可被多个线程共享（内部排队执行）。此外还提供：

* 流式推理 —— `transcribe_streaming(path, opts, cb)` 逐 token 回调；`create_buffered_session()` 边收边编码、`flush` 时给出与整段离线完全一致的转写；`create_stream_session()`（CLI `--stream`）在喂音频的同时增量输出已稳定文本，长音频内存有界
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
