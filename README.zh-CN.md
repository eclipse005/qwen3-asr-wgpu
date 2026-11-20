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

### 单个 / 列表 / manifest

`transcribe` 一个入口吃三种输入，`--batch` 只是其中一个参数：

```bash
transcribe --model ./Qwen3-ASR-1.7B-hf --wav speech.wav                    # 单文件（原来的行为）
transcribe --model ./Qwen3-ASR-1.7B-hf --batch 16            --wav segs/part000.wav segs/part001.wav ... --out texts.txt     # 列表
transcribe --model ./Qwen3-ASR-1.7B-hf --manifest items.jsonl --batch 16 \
           --out texts.jsonl --out-format jsonl                            # manifest
```

上游已经用 VAD 分好段（FireRedVAD：60 s 目标块、静音中点切、单段上限 3 分钟）时用后两种。
**`--batch N` 指同时解码几段**：默认 1 走逐段路径；`N>1` 时每一步的权重只读一次、喂给 N 段，
显存随 N 线性增长（每段一份 KV）。

| 参数 | 说明 |
|------|------|
| `--wav <file...>` | 音频列表（可出现多次），顺序即输出顺序 |
| `--manifest <jsonl>` | 每行一个音频及其可选参数，见下 |
| `--batch <n>` | 同时解码几段，默认 `1` |
| `--max-new <n>` | 每段生成的 token 上限，默认 2048（与库/单文件一致）；跑完看 `stop=max-tokens` 判断是否截断 |
| `--out <file>` | 按输入顺序写文本，一行一段 |
| `--out-format <text\|jsonl>` | `jsonl` 时每行输出 `{"audio","language","stop","text"}` |
| `--compare` | 同一列表再跑一遍 `--batch 1` 并逐段对比，有差异即非零退出 |
| `--timings` | 打印 `mel / encoder / prefill / decode` 四段耗时 |

#### manifest 格式

```jsonl
{"audio":"segs/p000.wav"}
{"audio":"segs/p001.wav","language":"English"}
{"audio":"segs/p002.wav","prompt":"热词：舒心 天海酒吧","max_new":768}
```

| 字段 | 必需 | 说明 |
|------|------|------|
| `audio` | ✓ | 路径；相对路径按 manifest 所在目录解析 |
| `language` / `lang` | | 该段强制语言，覆盖 `--lang` |
| `prompt` / `context` | | 该段热词/上下文，覆盖 `--prompt` |
| `max_new` | | 该段生成上限，覆盖 `--max-new`；同批按最大值定 KV |

字段名写错、缺 `audio`、路径不存在都会报出**带行号**的错误；未知字段不会静默忽略。
命令行的 `--lang` / `--prompt` / `--max-new` 仍然对列表里的所有项生效，manifest 里的字段是逐项覆盖。

**输出与 `--batch 1` 逐字节相同**，不是近似：每个 slot 读的是它自己那块 buffer
（`binding_array` 按 `wgpu.z` 索引），batch 只改变"一次 dispatch 覆盖几段"，不改变任何一段的
累加顺序。`--compare` 就是这条保证的自检。

**一条坏掉只赔一条。** 文件解不开、`language` 不支持、prompt 拼不出来、解码报错，都只记在那一条上；
列表里其余段照常跑，`--out` 照常写出成功的那几条（按输入顺序，失败的不写），失败逐条打到 stderr，
进程退出码非零。如果真的失败在槽位本身（开不出这么多槽、设备丢了），会先把宽度减半、再退到
逐段重跑，跑不出来的那条才记错——所以 `--batch 32` 在不支持 32 槽的卡上是降级，不是整批报废。

| 限制 | 值 |
|---|---|
| 每批槽位上限 | 32（`binding_array` 宽度）；再大不会更快，只多占显存 |
| 批量路径的单段长度 | `prompt + max_new ≤ 4096`（每槽 KV 显存）；超出的段自动退回逐段路径，不会失败。`--max-new 2048` 时相当于单段约 163 s |
| 显存 | 约 `模型 + 0.5 GiB × N`（60 s 段、`--max-new 2048` 的典型值；每槽 KV 由 `prompt + max_new` 决定，想省显存就调小 `--max-new`） |

`--stream` / `--session` / `--mel` / `--embeds` 等诊断模式只对单文件有意义，给列表会直接报错。
旧的 `transcribe_batch` 仍可用，会打一行弃用提示然后转发到同一份实现。

库里的入口是 `AsrInference::transcribe_wavs_batch(wavs, opts, batch)`；
需要逐项参数时用 `transcribe_specs(&[AudioSpec], opts, batch)`（两者都有 `_timed` 变体）。
两者都返回 `Vec<ItemOutcome>`：每条一个、顺序同输入、各带自己的 `Result`——一条失败不会遮住其它条。
要进度与取消用 `transcribe_specs_with(specs, opts, batch, &mut events)`：
`BatchEvents::on_item` 在每条定稿时回调（管线可以按段落盘，不用等整批返回），
`BatchEvents::should_stop` 返回 `true` 就放弃没跑完的，它们以 `AsrError::Cancelled` 回来。

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
