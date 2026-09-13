# 交接：qwen3-asr-wgpu

**新窗口请把工作区开在 `D:\qwen3-asr-wgpu`，把本文件全文交给 AI。**
改完对齐或 RTFx 后同步更新本文。不要再写第二份启动提示词。

---

## 机器上三个相关项目（不要混）

| 目录 | 角色 | 怎么跑 |
|---|---|---|
| **`D:\Qwen3-ASR`** | 官方 Python 原项目（QwenLM/Qwen3-ASR）。文本 **gold** 从这里出。 | conda 环境 **`asr`**：`C:\Users\ADMIN\miniconda3\envs\asr`。`conda activate asr`。实测：transformers **5.17.0**、librosa **0.11.0**、soxr **1.0.0**、torch **2.8.0+cu128**。模型：`D:\Qwen3-ASR\models\Qwen3-ASR-0.6B-hf` 和 `...\Qwen3-ASR-1.7B-hf`。 |
| **`D:\qwen3-asr-rs`** | Rust **CUDA + CPU** 移植。CUDA 手写是 RTFx **对照**，不是文本 oracle。 | 独立 git（origin `eclipse005/qwen3-asr-rs`）。**不要改 CUDA 后端。** fixture：`D:\qwen3-asr-rs\tests\fixtures\*.wav`。python-hf 冻结文本：`D:\qwen3-asr-rs\docs\baseline\texts\python-hf_{0.6B,1.7B}_*.txt`。CUDA RTFx 同目录 `cuda_*.txt`（0.6B 15s ~0.49s / 31×，180s_en ~9.87s / 18×）。 |
| **`D:\qwen3-asr-wgpu`** | **本仓库。** wgpu 文本解码 + CPU 音频编码器。从 rs 的 `wgpu/` 拆出，自有 git。 | 见下文。新对话只开这个目录。 |

硬件：Windows 11，NVIDIA **P104-100** Pascal 8GB，Vulkan，wgpu 30.0.1，驱动 572.75。**同一时间只能有一个 GPU 作业**（不要并行 Python dump 和 wgpu transcribe）。

---

## 本仓库当前 git

```
785ad13 docs: record dead-end RTFx probes in HANDOFF
f25f56c docs: replace stale handoff; retire port kickoff prompt
c833a84 perf: speed up CPU audio encoder conv/transformer
acd3ac4 perf: load wgpu text decoder once in WgpuAsr::load
75627d6 feat: align wgpu e2e transcribe with Python -hf greedy
```

对齐快照是 `75627d6`。其后是 RTFx + 文档。

---

## 验收（已完成）

**12/12 MATCH** vs python-hf greedy（0.6B + 1.7B × 15s_en / 30s_zh / 90s_en / 90s_ja / 180s_en / 180s_zh）。

对齐要点（不要回退）：

- 重采样：vendored `third_party/soxr`，**HQ**（`SOXR_20_BITQ=4`）。`soxr_oneshot` 默认 LQ，必须显式 HQ。
- `180s_en.wav` 是 **44.1 kHz mono**，不是 48 kHz。15s/90s_en 是 48 kHz stereo；中日 16 kHz 多数不用重采样。
- STFT：`torch.stft(center=True)` 反射 pad，**不要**先把波形补到 hop 整数倍。
- Encoder GELU 是 **erf**（`F.gelu` / `ACT2FN["gelu"]`），不是 tanh。
- Encoder 尾块仍用 `feo(tail)` 个 token。
- Decoder 在 `WgpuAsr::load` 里只上 GPU 一次，`max_seq=4096`。`transcribe` 要 `&mut self`。

跑：

```text
cd D:\qwen3-asr-wgpu
cargo run --release --bin transcribe -- ^
  --model D:\Qwen3-ASR\models\Qwen3-ASR-0.6B-hf ^
  --wav D:\qwen3-asr-rs\tests\fixtures\15s_en.wav ^
  --adapter nvidia --max-new 512 ^
  --baseline D:\qwen3-asr-rs\docs\baseline\texts\python-hf_0.6B_15s_en.txt
```

隔离：`tools/dump_python_hf.py`（conda `asr`）+ `transcribe --mel` / `--embeds`。

---

## RTFx（进行中，目标超过 CUDA 手写）

口径：`elapsed` 是 load 之后的 inference。RTFx = audio_s / elapsed。

| 0.6B clip | CUDA 手写（qwen3-asr-rs） | wgpu 当前 |
|---|---|---|
| 15s_en | 0.49s / **31×** | ~1.31s / **~11.4×** |
| 180s_en | 9.87s / **18×** | ~17.1s / **~10.3×** |

180s_en 构成（约）：**enc 2.7s CPU** + **prefill 3.0s** + **decode 9.9s**。decode 单独已经接近 CUDA 整条管线。

下一刀：

1. **Audio encoder 上 GPU，必须 im2col+GEMM（或等价 tiling），不要逐输出点扫 cin×9。**
   已试朴素 WGSL conv：15s **MATCH** 但 enc 293ms→1257ms，已撤回。c2 的 cin=480。
2. **Decode split-K**（长上下文 ~16.6 ms/token；短上下文 GPU-bound ~8.3 vs CUDA ~6.6）。
   greedy 不能把 step i+1 排进 GPU 再等 token i：`embed` 读同一个 `token_buf`。
3. **Prefill GEMM**：`gemm_bench` m=2304 k=1024 n=4096 上 8×8 ~1.96 TFLOP/s，软件流水 2.06。换 TM/TN 几乎没增益。

Pascal 无 `shaderFloat16`：f16 走 `unpack2x16float` / `pack2x16float`。不要改 decode kernel 加法顺序。

RTFx 改完至少复跑 0.6B `15s_en` + `180s_en` MATCH。

---

## 本仓库文档

| 文件 | 用途 |
|---|---|
| **HANDOFF.md**（本文件） | 唯一给下一任 AI 的实时入口 |
| `ROADMAP-wgpu.md` | 阶段测量记录。状态以本文为准 |
| `FEASIBILITY.md` | 2026-09-12 硬件实测（Pascal f16、GEMV 带宽）。不是任务清单 |

已删：`PROMPT-wgpu-port.md`。

不要提交：`align_dump/`、`golden/`、`target/`、`*.bin`。

---

## 硬约束

- 文本对齐 = **python-hf**（conda `asr` + `-hf` 权重），不是 Rust CUDA 文本。
- 不要改 `D:\qwen3-asr-rs` 的 CUDA 后端。
- 顺序用 GPU。
