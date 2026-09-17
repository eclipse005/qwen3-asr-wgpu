# qwen3-asr-wgpu

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 的 Rust 推理实现，基于 wgpu。音频前端、音频塔、文本解码整条流水线都跑在 GPU 上，没有可用 GPU 时自动改用 CPU；不需要 Python，也不需要任何深度学习框架，Windows / macOS / Linux 都能构建。

Qwen3-ASR 是阿里通义千问开源的语音识别模型，支持语言识别与 52 种语言和方言（30 种语言 + 22 种中文方言），提供 0.6B 和 1.7B 两个规格。

## 安装

作为依赖加入 `Cargo.toml`：

```toml
[dependencies]
qwen3-asr-wgpu = { git = "https://github.com/eclipse005/qwen3-asr-wgpu.git" }
```

构建命令行工具：

```bash
cargo build --release        # 得到 target/release/transcribe
```

| Feature | 说明 |
|---------|------|
| `hub` | 让程序自己从 HuggingFace 下载模型（`AsrInference::from_pretrained`）。默认关闭 |

## 模型下载

从 HuggingFace 下载 `-hf` 版本的权重（版权归原作者）：

- [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

下载得到的目录直接作为 `--model` 传入即可。

## 使用

### 命令行

```bash
transcribe --model ./Qwen3-ASR-0.6B-hf --wav audio.wav
```

运行时会打印用的哪块设备（GPU 还是 CPU）、耗时和识别出的语言，最后输出转写文本。

| 参数 | 说明 |
|------|------|
| `--model <dir>` | 模型目录（也可用环境变量 `QASR_MODEL`） |
| `--wav <file>` | 要转写的音频，任意采样率，内部自动转成 16 kHz（也可用 `QASR_WAV`） |
| `--lang <name>` | 指定语言，如 `zh`、`English`；不填则自动识别 |
| `--context <text>` | 热词/提示文本，帮助模型认准专有名词（`--prompt` 同义） |
| `--max-new <n>` | 最多生成多少 token，默认 2048 |
| `--device <name>` | 指定设备，默认自动；`cpu` 表示强制用 CPU |
| `--list-devices` | 列出这台机器上可用的设备 |
| `--languages` | 打印支持的语言列表 |

默认自动挑选一块 GPU；GPU 建不起来时会退回 CPU，并打印原因。

### 作为库

```rust
use qwen3_asr_wgpu::{AsrInference, Backend, TranscribeOptions};

let asr = AsrInference::load("Qwen3-ASR-0.6B-hf".as_ref(), Backend::best())?;
let out = asr.transcribe("audio.wav", TranscribeOptions::default())?;
println!("[{}] {}", out.language, out.text);
```

模型只需加载一次，之后可以反复转写；一个实例也能被多个线程共享（内部排队执行）：

```rust
let asr = std::sync::Arc::new(AsrInference::load(dir, Backend::best())?);
let worker = std::sync::Arc::clone(&asr);
std::thread::spawn(move || worker.transcribe("audio.wav", TranscribeOptions::default()));
```

## API

### 加载模型

| | |
|---|---|
| `AsrInference::load(dir, Backend::best())` | 最常用：自动挑一块设备 |
| `AsrInference::load(dir, Backend::Cpu)` | 只用 CPU |
| `AsrInference::from_pretrained(id, cache_dir, backend)` | 自动下载并加载（需要 `hub` feature） |

### 转写

| | |
|---|---|
| `transcribe(path, opts)` | 转写一个 wav 文件 |
| `transcribe_samples(&samples, opts)` | 转写内存里的 16 kHz 单声道采样 |
| `transcribe_streaming(path, opts, cb)` | 边生成边回调，回调收 `StreamToken { token_id, text_so_far }` |
| `create_streaming_session(opts)` | 开一个会话，用 `push_samples()` 分段喂音频，`flush()` 取结果 |

### 请求参数 `TranscribeOptions`

| 字段 | 说明 |
|------|------|
| `language` | 指定语言，`None` 表示自动识别 |
| `context` | 热词/提示文本 |
| `max_new_tokens` | 生成上限，默认 2048 |

链式构造：

```rust
let opts = TranscribeOptions::default()
    .with_language("zh")
    .with_context("订单号、专有名词……")
    .with_max_new_tokens(1024);
```

### 结果 `TranscribeResult`

| 字段 | 说明 |
|------|------|
| `text` | 转写文本 |
| `language` | 语言：自动识别得到的，或调用方指定的 |
| `raw_output` | 未解析的原始输出 |

### 其他

- `AsrInference::devices()` / `device_targets()`：查看可用设备
- `AsrInference::device_description()`：查看当前实际跑在哪
- `supported_languages()`：`--lang` 接受的语言
- `AsrError` / `Result<T>`：统一错误类型（模型加载 / 音频解码 / 推理 / 参数错误）

### 与官方 Python 对应的三段式调用

官方 Python 是 `apply_transcription_request(...)` → `generate(...)` → `decode(...)`，这里提供同样的三步：

```rust
use qwen3_asr_wgpu::{AsrInference, Backend, ReturnFormat, TranscribeOptions};

let asr = AsrInference::load("Qwen3-ASR-0.6B-hf".as_ref(), Backend::best())?;
let opts = TranscribeOptions::default().with_max_new_tokens(512);

let req = asr.apply_transcription_request_file("audio.wav".as_ref(), &opts)?;
let ids = asr.generate(&req)?;                      // 生成的 token id
let out = asr.decode(&ids, ReturnFormat::Parsed)?;  // 拆成语言 + 文本
println!("{}: {}", out.language().unwrap_or("?"), out.transcription_text());
```

`ReturnFormat` 有三种：`Raw` 保留特殊 token，`Parsed` 拆成语言和文本，`TranscriptionOnly` 只取文本。

## License

Apache-2.0，与上游一致。

## 致谢

本仓库是**独立的 Rust 推理实现**，用于加载并运行官方发布的 Qwen3-ASR 权重，**不是** Alibaba / Qwen 官方发行版，与原作者无隶属关系。使用模型权重时请遵守原作者的许可证。时间戳（强制对齐）见同系列的 [qwen-aligner-wgpu](https://github.com/eclipse005/qwen-aligner-wgpu)。

- 官方项目：[QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- 模型集合：[Qwen3-ASR on Hugging Face](https://huggingface.co/collections/Qwen/qwen3-asr)
