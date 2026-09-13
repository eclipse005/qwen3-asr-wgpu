# 交接：qwen3-asr-wgpu

把本文件全文交给接手的 AI。它应反映仓库**当前**状态；改完对齐或 RTFx 后请同步更新本文，不要再追加过时的启动提示词。

---

## 这是什么

独立 crate：`D:\qwen3-asr-wgpu`（已从 `qwen3-asr-rs` 拆出，自有 git）。
wgpu 文本解码（prefill + decode）+ **CPU** 音频编码器。对齐目标是 **Python Transformers `-hf` greedy**，不是 Rust CUDA 文本。

硬件：Windows，NVIDIA **P104-100** Pascal 8GB，Vulkan，wgpu 30.0.1。**同一时间只能有一个 GPU 作业**（不要并行 Python dump 和 wgpu transcribe）。

## 当前 git

```
c833a84 perf: speed up CPU audio encoder conv/transformer
acd3ac4 perf: load wgpu text decoder once in WgpuAsr::load
75627d6 feat: align wgpu e2e transcribe with Python -hf greedy
```

## 验收（已完成）

**12/12 MATCH** vs `D:\qwen3-asr-rs\docs\baseline\texts\python-hf_{0.6B,1.7B}_*.txt`。

对齐要点（不要回退）：

- 重采样：vendored `third_party/soxr`，**HQ**（`SOXR_20_BITQ=4`）。`soxr_oneshot` 默认是 LQ，必须显式 HQ。
- `180s_en.wav` 是 **44.1 kHz mono**，不是 48 kHz。
- STFT：`torch.stft(center=True)` 反射 pad，**不要**先把波形补到 hop 整数倍（`180s_zh` 会多一帧）。
- Encoder GELU 是 **erf**（`F.gelu` / `ACT2FN["gelu"]`），不是 tanh。
- Encoder 尾块仍用 `feo(tail)` 个 token（Python 用 mask；token 数一致）。
- Decoder 在 `WgpuAsr::load` 里只上 GPU 一次，`max_seq=4096`。

跑：

```text
cd D:\qwen3-asr-wgpu
cargo run --release --bin transcribe -- ^
  --model D:\Qwen3-ASR\models\Qwen3-ASR-0.6B-hf ^
  --wav D:\qwen3-asr-rs\tests\fixtures\15s_en.wav ^
  --adapter nvidia --max-new 512 ^
  --baseline D:\qwen3-asr-rs\docs\baseline\texts\python-hf_0.6B_15s_en.txt
```

conda 环境：`asr`（transformers 5.17、librosa 0.11、soxr 1.0）。隔离用 `tools/dump_python_hf.py` 和 `transcribe --mel` / `--embeds`。

## RTFx（进行中，要超过 CUDA 手写）

计时口径：模型已 load 后的 inference（`elapsed`，不含 `loaded in`）。RTFx = audio_s / elapsed。

| 0.6B clip | CUDA 手写 | wgpu 当前 |
|---|---|---|
| 15s_en | 0.49s / **31×** | ~1.31s / **~11.4×** |
| 180s_en | 9.87s / **18×** | ~17.1s / **~10.3×** |

180s_en 构成（约）：**enc 2.7s CPU** + **prefill 3.0s** + **decode 9.9s**。decode 单独已经接近 CUDA 整条管线。

下一刀（按杠杆）：

1. **Audio encoder 上 GPU，必须用 im2col+GEMM（或等价 tiling），不要逐输出点扫 cin×9。**
   已试过朴素 WGSL conv（每线程扫 cin×3×3）：0.6B 15s **MATCH** 但 enc 293ms→1257ms，已撤回。
   c2 的 cin=480，朴素循环在 GPU 上也比 CPU `gemm` 慢。
2. **Decode split-K**（长上下文 ~16.6 ms/token；短上下文 GPU-bound ~8.3 ms vs CUDA ~6.6 ms）。
   greedy 不能把 step i+1 排进 GPU 再等 token i：`embed` 读的是同一个 `token_buf`。
3. **Prefill GEMM**：`gemm_bench` 在 m=2304 k=1024 n=4096 上 8×8 约 1.96 TFLOP/s，8×8 软件流水 2.06，换 tile 几乎没增益。要涨只能改微内核，不是换 TM/TN。

Pascal 无 `shaderFloat16`：f16 一律 `unpack2x16float` / `pack2x16float`。不要改 decode kernel 的加法顺序（会打文本）。

## 文档

| 文件 | 用途 |
|---|---|
| **HANDOFF.md**（本文件） | 唯一给下一任 AI 的实时入口 |
| `ROADMAP-wgpu.md` | 阶段测量记录（decode/prefill 历史数字）。状态节可能略滞后，以本文为准 |
| `FEASIBILITY.md` | 2026-09-12 硬件实测（Pascal f16 绕法、GEMV 带宽）。不要当当前任务清单 |

已删：`PROMPT-wgpu-port.md`（过时启动词，路径还写着 `D:\qwen3-asr-rs\wgpu`）。

不要提交：`align_dump/`、`golden/`、`target/`、`*.bin`。

## 硬约束

- 文本对齐以 **python-hf** 为准；优化 RTFx 后至少复跑 `15s_en` + `180s_en` 0.6B MATCH。
- 不要动 `D:\qwen3-asr-rs` 的 CUDA 后端。
- 顺序用 GPU。fixture：`D:\qwen3-asr-rs\tests\fixtures`。模型：`D:\Qwen3-ASR\models\Qwen3-ASR-*-hf`。
