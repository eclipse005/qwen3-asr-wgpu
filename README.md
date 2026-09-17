# qwen3-asr-wgpu

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 的 Rust + wgpu 推理实现。整条流水线——log-mel 前端、音频塔、文本解码器——都跑在 GPU 上：同一份 WGSL 内核、同一套算术，由 **Vulkan / Metal / D3D12 / OpenGL(ES)** 任意一条运行时驱动，另有自带的 CPU 后端，零深度学习框架依赖。

Qwen3-ASR 是阿里通义千问开源的语音识别模型，官方称支持 52 种语言和方言，提供 0.6B 与 1.7B 两种规格。本实现的验收门禁是官方 python-hf 的冻结转写文本：0.6B / 1.7B × 6 条音频（15 s ~ 180 s）12/12 逐字节一致，每次改动重跑。

```text
wav ──► mel (STFT, 128 bins) ──► audio tower (conv stem + 18 transformer layers)
                                   │
                                   ▼
        prompt (chat template) ──► text decoder (prefill + greedy decode, KV cache)
                                   │
                                   ▼
                     language <NAME><asr_text>transcript
```

## 多后端支持

选择轴是**运行时**，不是硬件厂商：同一张卡可以通过多条运行时走到，而那是不同的代码路径（数值也可能不同），所以厂商与设备类型只出现在设备列表里，不出现在选择器里。带索引的写法用于同一运行时的多块设备，形状同 llama.cpp 的 `Vulkan0` / `CPU`。

| 运行时 | 说明 |
|--------|------|
| `vulkan[:N]` | Vulkan |
| `metal[:N]` | Metal（macOS / iOS） |
| `dx12[:N]` | D3D12；wgpu 在 Windows 上走 D3D12 compute，没有 DirectML（那是另一套 API） |
| `gl[:N]` | OpenGL / GLES，wgpu 的兼容运行时，能力最弱，老机器只剩它 |
| `cpu` | 自带的 CPU 后端：CPU 音频塔 + CPU 文本解码，不创建 adapter |
| `auto` | 默认：独显 → 集显 → 兜底 |

```bash
cargo run --release --bin transcribe -- --list-devices   # 只列设备，不创建 device
# * vulkan:0   NVIDIA P104-100 (NVIDIA, dGPU, driver …) binding 2047 MiB, subgroup 32
#   dx12:0     NVIDIA GeForce GTX 1070 (NVIDIA, dGPU, …) binding 2047 MiB, no subgroup
#   vulkan:1   Intel(R) Graphics (Intel, iGPU, …) binding 1023 MiB, subgroup 8..32
#   gl:0       Intel(R) Graphics (Intel, iGPU, …) binding 1024 MiB, no subgroup
#   cpu        the host backend (CPU audio tower + CPU text decoder)

cargo run --release --bin transcribe -- --device vulkan:1 …   # 集显走 Vulkan
cargo run --release --bin transcribe -- --device dx12:0 …     # 独显走 D3D12
cargo run --release --bin transcribe -- --device nvidia …     # 适配器名子串（兜底写法）
```

同一份门禁（0.6B、180 s 英文、对冻结 python-hf 文本逐词比对）在这台机器上每条运行时的读数：

| 后端 | 设备 | enc | prefill | decode | elapsed | RTFx | 结果 |
|------|------|-----|---------|--------|---------|------|------|
| `vulkan:0` | NVIDIA P104-100 | 1012 | 1534 | 5788 | 10.5 s | **17.8×** | MATCH |
| `dx12:0` | NVIDIA GTX 1070 | 1007 | 1644 | 8934 | 13.7 s | 12.9× | MATCH |
| `cpu` | 主机（20 线程，`gemm` prefill） | — | — | — | 48.8 s | 3.61× | MATCH |
| `vulkan:1` | Intel 集显 | 9176 | 14816 | 25062 | 49 s | 3.6× | MATCH |
| `gl:0` | Intel 集显（OpenGL） | 7083 | 12131 | 29992 | 51 s | 3.46× | MATCH |

两点说明：

* **D3D12 的 pipeline 构建约 5.5 分钟**（Intel 344 s、NVIDIA 328 s；同一台机器上各自的 Vulkan adapter 只要 4.6 / 6.6 s），上表是构建之后的读数。wgpu 30 的 D3D12 后端不暴露 `Features::PIPELINE_CACHE`，这笔开销缓存不掉，所以 D3D12 只作 Vulkan 不可用时的兜底，日常用 `vulkan:N`。
* **`Features::SUBGROUP` 是能力，不是宽度**：有的驱动会给出 8..32 的区间，此时 32 lane 的 xor butterfly 会归约到错误的 lane，模型流畅地输出垃圾而全程零报错。所有 shuffle 路径都以「adapter 明确承诺正好 32 lane」为前提，否则退回 shared-memory 路径（结果逐位相同）。`--list-devices` 把 subgroup 宽度列出来，就是为了这个。

`cpu` 是自带的主机后端（`--cpu-dec`）：不用 adapter，加载时把 f16 权重展宽成 f32，用 rayon 并行输出行与 attention 的 `(row, head)`。它六条 fixture 的 RTFx：4.25 / 3.77 / 3.95 / 4.17 / 3.61 / 3.47。

Metal 的代码路径在（同一批 WGSL 内核 + 运行时协商的 limits），但这台机器上没有跑过：只标「未验证」，不替它下结论——没跑过就是没跑过，未验证不等于不可用。

## 安装

```toml
[dependencies]
qwen3-asr-wgpu = { git = "https://github.com/eclipse005/qwen3-asr-wgpu.git" }
```

| Feature | 说明 |
|---------|------|
| `hub` | 从 HuggingFace 下载模型（`AsrInference::from_pretrained`）。默认关闭：它会把 reqwest 和 TLS 一起拉进来，而本地 checkpoint 并不需要 |

命令行工具：`cargo build --release`，产物是 `target/release/transcribe`。

## 使用

### 命令行

```bash
cargo run --release --bin transcribe -- \
    --model /path/to/Qwen3-ASR-0.6B-hf \
    --wav   clip.wav --device vulkan:0 --max-new 512
```

`--model` 也可以走 `QASR_MODEL`，`--wav` 走 `QASR_WAV`。

常用参数：`--lang en`（强制语言，ISO 码或全名，按参考实现的 30 个语言名校验）、`--context` / `--prompt "hotwords"`（chat template 的 system message）、`--languages`（打印支持的语言）、`--cpu-enc`（音频塔走 CPU，用于 A/B）、`--baseline file.txt`（与冻结参考文本比对，打印 `MATCH` / `MISMATCH`）、`--dump dir/`（导出 mel / embeddings / ids）。

### 作为库

```rust
use qwen3_asr_wgpu::{AsrInference, Backend, TranscribeOptions};

let asr = AsrInference::load("Qwen3-ASR-0.6B-hf".as_ref(), Backend::best())?;
let out = asr.transcribe("clip.wav", TranscribeOptions::default())?;
println!("[{}] {}", out.language, out.text);
```

`AsrInference` 持有模型且所有方法都是 `&self`，所以一个实例可以 `Arc` 出去共享——0.6B 的权重约 1.2 GiB，每份拷贝还会各带一套 KV cache：

```rust
let asr = std::sync::Arc::new(AsrInference::load(dir, Backend::best())?);
let worker = std::sync::Arc::clone(&asr);
std::thread::spawn(move || worker.transcribe("clip.wav", TranscribeOptions::default()));
```

并发调用在内部的互斥锁上串行；一次转写不会被拆到多个线程上。其余都是每次调用各自的状态——每次生成都从 KV 位置 0 重新 prefill，调用之间不残留。

`TranscribeOptions` 承载整个请求（`language`、`context`、`max_new_tokens`，默认 2048），是 `#[non_exhaustive]` 的，配 `with_*` setter：

```rust
let opts = TranscribeOptions::default()
    .with_language("zh")
    .with_context("Swing trading course. Terms: order block, time frame, …")
    .with_max_new_tokens(700);
```

强制语言时结果会把语言回填到 `TranscribeResult::language`；自动识别时这个字段是模型自己命名的那个。流式是同一套：`transcribe_streaming` 接每 token 的回调，`create_streaming_session` 接增量音频（`push_samples` → `flush`）。

错误是一个枚举 `AsrError`（`ModelLoad` / `AudioDecode` / `Inference` / `InvalidOptions`），各入口统一返回 `Result<T>`。

### 设备选择（库内）

```rust
use qwen3_asr_wgpu::{AsrInference, DeviceSelector};

for t in AsrInference::device_targets() { println!("{}", t.describe()); }   // 不创建 device
let asr = AsrInference::load_on("model-dir".as_ref(), DeviceSelector::parse("vulkan:1")?)?;
println!("running on {}", asr.device_description());
```

`Backend` 是粗粒度的一词选择（`Auto` / `Gpu` / `Cpu`），`DeviceSelector` 是细粒度的运行时 + 索引；`--adapter <name>` 是更早的写法，现在退化成「适配器名子串」这一种。目标给出的能力（binding 上限、workgroup storage、subgroup 宽度）一律按协商后的 limits 读取，跑不动就明确拒绝，而不是产出垃圾。

### 与参考实现相同的三段式 API

Python 侧是 `processor.apply_transcription_request(...)` → `model.generate(...)` → `processor.decode(ids, return_format=...)` 三步，本 crate 也是（`src/processor.rs`）：

```rust
use qwen3_asr_wgpu::{AsrInference, Backend, ReturnFormat, TranscribeOptions};

let asr = AsrInference::load("Qwen3-ASR-0.6B-hf".as_ref(), Backend::best())?;
let opts = TranscribeOptions::default().with_max_new_tokens(512);

let req = asr.apply_transcription_request_file("clip.wav".as_ref(), &opts)?;
let ids = asr.generate(&req)?;                        // 贪心，遇到 EOS 停
let out = asr.decode(&ids, ReturnFormat::Parsed)?;    // {"language", "transcription"}
println!("{}: {}", out.language().unwrap_or("?"), out.transcription_text());
```

`max_new_tokens` 固定在请求的 options 里，不再传给 `generate`——上限只有一个地方定义。

`decode` 实现了上游全部三个格式（`Raw` / `Parsed` / `TranscriptionOnly`），语义照搬：`Raw` 保留特殊 token，`Parsed` 去掉。注意 `decode` 报的是**模型自己说出**的语言（参考实现的 `_parse_single_output`）：强制语言时元信息在 prompt 里，这个字段是空的；`transcribe` 会从请求里把它补上——这是两者唯一不一致的地方，`#[ignore]` 的测试 `processor_api_matches_the_one_call_wrapper` 把它钉住。

## 模型下载

权重用 `-hf` checkpoint（版权归原作者）：

- [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

加载器直接按 `thinker.model.*` 从 safetensors 里取张量。打开 `hub` feature 后还可以用 `AsrInference::from_pretrained("Qwen/Qwen3-ASR-0.6B", cache_dir, backend)` 直接下载。

官方项目与文档：

- 代码与说明：[QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- 模型集合：[Qwen3-ASR on Hugging Face](https://huggingface.co/collections/Qwen/qwen3-asr)

## 性能

同一台机器（Windows、NVIDIA P104-100 8 GB、Vulkan、一次只跑一个 GPU 任务），0.6B：

| 音频 | mel | 音频塔 | prefill | decode | 总计 | RTFx |
|------|-----|--------|---------|--------|------|------|
| 15 s | 2 ms | 107 | 151 | 346 | 0.64 s | 23.6 |
| 180 s | 25 ms | 1012 | 1534 | 5788 | 8.4 s + 1.45 s 前端 | 17.8 |
| 15 min | 121 ms | 5236 | 16 259 | 76 556 | 98.3 s | **9.4** |

六条 fixture 的 RTFx：24.4 / 21.3 / 19.7 / 24.0 / 17.8 / 20.1。

对同机器同 GPU 上的 Python 实现：0.6B 1.95×、1.7B 1.52×（29 种语言 × 20 条 FLEURS 抽样，WER/CER 差在 ±0.7 pp 内）。

长音频是最有意思的一档：15 分钟音频要是把 `s²` attention 矩阵材料化就是 2 × 4.7 GB，8 GB 卡放不下。本实现把 prefill attention 切成 key slab，于是「直接拒绝」变成「能跑」——15 分钟音频（12 065 token prompt、3 343 token 转写）98 s 跑完。

显存（峰值 `memory.used`，每 ~100 ms 采样，六条 fixture、两个规格）：

| 音频 | 0.6B 峰值 | 1.7B 峰值 |
|------|-----------|-----------|
| 15 s en | 2.05 GiB | 4.34 GiB |
| 30 s zh | 2.05 GiB | 4.34 GiB |
| 90 s en | 2.30 GiB | 4.59 GiB |
| 90 s ja | 2.30 GiB | 4.59 GiB |
| 180 s en | 2.80 GiB | 5.34 GiB |
| 180 s zh | 2.80 GiB | 5.34 GiB |

KV cache 每 slot 112 KiB（28 层 × 8 KV head × 128，f16 的 K 与 V；两个规格几何相同），并且是**按需分配**的：`max_seq`（16 384 token）只是 rope 表的天花板，不是分配量——15 秒音频只占 1 024 slot（112 MiB），长音频按需增长。容量只增不减、按 256 slot 步进，所以连着跑同一形状的音频只分配一次（一次 resize 0.4 ms @1 792 slot 到 37 ms @12 032 slot），15 分钟请求需要 1.53 GiB，而不是固定预分配那 1.75 GiB。

15 分钟时的时间去向：decode 78 %（KV 扫描是指令受限而非带宽受限：有效 ~86 GB/s，对着 ~250 GB/s 的可用带宽）、prefill 17 %（其中 60 % 是两个 attention GEMM，已经贴着这张卡 ~2.1 TFLOP/s 的天花板）、音频塔 5 %（贴着权重带宽 roofline）。

## 一致性 / 验证

* 转写门禁：`cargo run --release --bin transcribe -- --model <dir> --wav clip.wav --device vulkan:0 --max-new 512 --baseline frozen.txt` —— 与冻结参考文本逐词比对，打印 `MATCH` / `MISMATCH`。上面那张跨后端表就是这么跑出来的，每个后端过的是同一份门禁。
* `cargo test --release` —— 纯 CPU 的一半：tokenizer / prompt / mel 几何、config 解析。
* `cargo test --release --test public_api -- --ignored` —— 真实模型上的公开 API：冻结文本比对、`Backend::Cpu`、增量 session、一个实例服务两个线程。需要 `QASR_TEST_MODEL` / `QASR_TEST_WAV`，逐字比对还要 `QASR_TEST_BASELINE`。
* `cargo run --release --bin transcribe -- --diag-enc` —— 音频塔自己的 oracle：逐级对主机参考，包括主机侧重算 attention 的算子。

**覆盖范围**：上表里每条运行时都在这台机器上跑过门禁；Metal 没有。没有批处理，也没有 vLLM 后端——转写本身是单条音频一次调用。时间戳/强制对齐不在本仓库，见同系列的 [qwen-aligner-wgpu](https://github.com/eclipse005/qwen-aligner-wgpu)。

## 致谢 / 原版出处

本仓库是**独立的 Rust 推理实现**，用于加载并运行官方发布的 Qwen3-ASR 权重；**不是** Alibaba / Qwen 官方发行版，与原作者无隶属关系。

| 组件 | 原版 | 链接 | 协议（以官方页面为准） |
|------|------|------|------------------------|
| 模型权重 | Qwen3-ASR 0.6B / 1.7B | [HF 0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) · [HF 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Apache-2.0 |
| 官方推理与文档 | Qwen3-ASR | [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) | Apache-2.0 |

使用模型权重时请遵守原作者许可证；本仓库的 Rust 推理代码以本仓库 License 为准。

## License

Apache-2.0，与上游 Qwen3-ASR 一致。内置的 soxr 重采样器在 `third_party/soxr` 下，协议见该目录。
