# 交接：qwen3-asr-wgpu

**新窗口请把工作区开在 `D:\qwen3-asr-wgpu`，把本文件全文交给 AI。**
改完对齐或 RTFx 后同步更新本文。不要再写第二份启动提示词。

> **本窗口（2026-09-14 第二轮）做完的事一句话**：**GPU 音频塔的 transformer 对齐了**。
> conv stem 本来就是对的（上一窗口的结论没错，是诊断的 CPU 参考坏了，见 bug 23）；
> 真正挡住 transformer 的是四个 bug：`enc.ex` uniform **从未写入**（extract 第一句就 return，
> Q/K/V 全是 0 —— 而 oracle 拿 GPU 自己的 0 重算，于是「零对零」三行全绿，见 bug 20）、
> attention GEMM 的 `bsc` 传成**字**而不是**元素**（batch=1 时看不出来；28 个 block 时
> 相邻块互相覆盖、顺序不定，见 bug 21）、`win_pack` 里 `qp` 的**越行写入竞争**（bug 22）、
> 以及诊断侧的 tile 偏移（bug 23）和一处比错阶段（bug 24）。
> 修完后 **12/12 MATCH**（0.6B/1.7B × 6 fixture，与 python-hf 基线逐字相同），
> 180s_en 的 RTFx **10.1 → 14.9**，enc 相位 **2720 → 1019 ms**；
> GPU 塔成为**默认**（`--cpu-enc` 兜底）。
> 随后同一窗口的 **RTFx 专项**又推到 **16.8**（15s_en 23.6、90s_ja 23.6）：
> attention 的 K/V 改 `vec4` 读取（decode 11.66 → 9.86 ms/step）、**rms_norm 折进消费它的 GEMV**
> （虚拟线程复刻 1024 线程的 reduction 树，逐位一致）、`embed_tokens` 不再整表物化、
> soxr 补回被 cmake crate 吞掉的 `/O2`。12/12 全程复验。

---

## 机器上三个相关项目（不要混）

| 目录 | 角色 | 怎么跑 |
|---|---|---|
| **`D:\Qwen3-ASR`** | 官方 Python 原项目（QwenLM/Qwen3-ASR）。文本 **gold** 从这里出。 | conda 环境 **`asr`**：`C:\Users\ADMIN\miniconda3\envs\asr`。实测：transformers **5.17.0**、librosa 0.11.0、soxr 1.0.0、torch 2.8.0+cu128。模型：`D:\Qwen3-ASR\models\Qwen3-ASR-0.6B-hf` / `...-1.7B-hf`。 |
| **`D:\qwen3-asr-rs`** | Rust **CUDA + CPU** 移植。CUDA 手写是 RTFx **对照**，不是文本 oracle。 | 独立 git。**不要改 CUDA 后端。** fixture：`tests\fixtures\*.wav`。python-hf 冻结文本：`docs\baseline\texts\python-hf_*.txt`。CUDA RTFx：0.6B 15s ~0.49s/31×、180s_en ~9.87s/18×。 |
| **`D:\qwen3-asr-wgpu`** | **本仓库。** wgpu 文本解码 + 音频编码器（GPU 默认 / `--cpu-enc` 可选）。 | 见下文。新对话只开这个目录。 |

硬件：Windows 11，NVIDIA **P104-100** Pascal 8GB，Vulkan，wgpu 30.0.1，驱动 572.75。
**同一时间只能有一个 GPU 作业。** 测试完确认 `nvidia-smi --query-gpu=memory.used` 回落。

---

## 硬约束

- 文本对齐 = **python-hf**（conda `asr` + `-hf` 权重），不是 Rust CUDA 文本。
- 不要改 `D:\qwen3-asr-rs` 的 CUDA 后端。
- 顺序用 GPU，一次一个作业。
- `180s_en` 必须 `--max-new 700`（585 token 才到 EOS；512 会截断并报假 MISMATCH）。
- decode kernel 的加法顺序不能改（bit-exactness 是对齐的命根子）。
- 工作区有未提交改动，**这是正常的**；不要 revert。
- **默认走 GPU 音频塔**（12/12 复验过，连跑两次结果一致）；`--cpu-enc` 强制回 CPU
  （A/B 与回归用），`--gpu-enc` 仍被接受（它已经是默认）。塔建不起来会打印原因并自动回落 CPU。

---

## 验收（本窗口复验过）

```text
cd D:\qwen3-asr-wgpu
cargo run --release --bin transcribe -- --model D:\Qwen3-ASR\models\Qwen3-ASR-0.6B-hf ^
  --wav D:\qwen3-asr-rs\tests\fixtures\15s_en.wav --adapter nvidia --max-new 512 ^
  --baseline D:\qwen3-asr-rs\docs\baseline\texts\python-hf_0.6B_15s_en.txt
```

| 跑法 | 结果 |
|---|---|
| **默认（GPU 塔）** 0.6B `15s_en` | **MATCH**，RTFx **23.6**（elapsed 0.64s；mel 2 / enc 111 / prefill 147 / decode 349 ms） |
| **默认（GPU 塔）** 0.6B `180s_en`（`--max-new 700`） | **MATCH**，RTFx **16.8**（elapsed 10.50s；mel 23 / enc 1026 / prefill 1790 / decode 6024 ms） |
| `--cpu-enc` 0.6B `15s_en` / `180s_zh` | **MATCH**（CPU 参考路径，未回退） |

**12/12（默认配置，与 `python-hf_*.txt` 逐字比对全 MATCH）**：

| clip | 0.6B RTFx | 1.7B RTFx |
|---|---|---|
| 15s_en | 23.6 | 11.2 |
| 30s_zh | 20.8 | 10.3 |
| 90s_en | 19.2 | 10.8 |
| 90s_ja | 23.6 | 12.9 |
| 180s_en | 16.8 | 10.2 |
| 180s_zh | 19.0 | 10.8 |

`--cpu-enc` 抽查 `15s_en` / `180s_zh` 也 MATCH。
`90s_ja` 连跑两次结果一致（bug 22 修掉之前这里是非确定性的）；
（有一次连续跑 8 个 GPU 进程的批次在**收尾阶段**报了 `0xC0000005`，结果都已 MATCH 并打印；
单独复跑不复现，怀疑是 Pascal/Vulkan 的进程销毁抖动。）

对齐要点（不要回退）：重采样 vendored soxr **HQ**；`180s_en` 是 44.1 kHz mono；
STFT `center=True` 反射 pad；Encoder GELU 是 **erf**；`WgpuAsr::load` 只上一次 GPU；
decode 的 GEMV 归约树与 attention 的加法顺序（见「RTFx 优化」一节）。

---

## RTFx 现状

| 相位（180s_en, 0.6B） | 最早（CPU 塔） | 上一轮 | 现在 | CUDA 手写 |
|---|---|---|---|---|
| **wav 读入 + soxr HQ 重采样** | ~1450 ms | ~1450 ms | ~1450 ms | ? |
| mel (STFT) | 27 ms | 33 ms | 23 ms | 36 ms |
| audio enc | 2720 ms | 1050 ms | 1026 ms | 803 ms |
| prefill | 3009 ms | 1788 ms | 1790 ms | 1414 ms |
| decode | 9609 ms | 6226 ms | **6024 ms** | 7316 ms |
| 合计 | 17.46 s / 10.1× | 11.87 s / 14.9× | **10.50 s / 16.8×** | 9.87 s / 18.3× |

1.7B `180s_en`：enc 1393 / prefill 3679 / decode 10534 ms，RTFx **10.2**（原 9.6）。
**分相位之和现在与 elapsed 完全对得上**（`load_audio_wav` 以前是隐形的 1.4 s）。
decode 已比 CUDA 手写快 **18%**，音频塔和 prefill 各还差 ~25%。

### 对 CUDA 手写版（同机同时代口径）

CUDA 的数字取自 `D:\qwen3-asr-rs\docs\baseline\texts\cuda_*.txt` 头部记录（其
`baseline_snapshot` harness 的 `elapsed_s` 同样**包含 wav 读入 + 重采样**，且**无 warm-up**，
与我们同口径）：

| 模型 | 音频 | CUDA 手写 | 本仓库 wgpu | elapsed 比值 |
|---|---|---|---|---|
| 0.6B | 15s_en | 0.485 s / 30.9× | 0.635 s / 23.6× | **1.31** |
| 0.6B | 30s_zh | 1.380 s / 21.7× | 1.446 s / 20.7× | 1.05 |
| 0.6B | 90s_en | 3.896 s / 23.1× | 4.691 s / 19.2× | 1.20 |
| 0.6B | 90s_ja | 4.101 s / 21.9× | 3.783 s / 23.8× | **0.92** |
| 0.6B | 180s_en | 9.873 s / 18.2× | 10.503 s / 17.1× | 1.06 |
| 0.6B | 180s_zh | 10.735 s / 16.8× | 9.481 s / 19.0× | **0.88** |
| 1.7B | 15s_en | 1.212 s / 12.4× | 1.336 s / 11.2× | 1.10 |
| 1.7B | 30s_zh | 2.727 s / 11.0× | 2.909 s / 10.3× | 1.07 |
| 1.7B | 90s_en | 7.866 s / 11.4× | 8.338 s / 10.8× | 1.06 |
| 1.7B | 90s_ja | 6.818 s / 13.2× | 6.899 s / 13.0× | 1.01 |
| 1.7B | 180s_en | 16.416 s / 11.0× | 17.289 s / 10.4× | 1.05 |
| 1.7B | 180s_zh | 17.503 s / 10.3× | 16.613 s / 10.8× | **0.95** |

**12 组里赢 3 组（0.6B 90s_ja / 0.6B 180s_zh / 1.7B 180s_zh），其余落后 1%–31%。**
口径提醒：他们 180s_en 记的 audio_s 是标称 180.000，实际文件 176.309（我们的 RTFx 按实际值算），
所以上表按 **elapsed** 比才是公平的；按他们记的 RTFx 直接比会多算我们 2% 的亏。

**差距在哪（有实测支撑）**：以 15s_en 0.6B 为例，我们 0.635 s = 前端 33 + enc 111 + prefill 147 +
decode 349；他们 0.485 s 里 decode ≈ 0.336 s（其 ROADMAP 记的 6.58 ms/tok × 51）——
**decode 我们反而快 4%**，差距全在 **enc + prefill（258 ms vs ~120–150 ms，1.7–2×）**，
也就是那个共用的手写 `prefill_gemm`。这正是 `FEASIBILITY.md` 里写死的 go/no-go：
cuBLAS 在同形状上是 4–5 TFLOP/s，我们 `gemm_bench` 的**最好变体**也只有 2.18（m=384）/ 2.44（m=2304）
TFLOP/s，且已在 TM/TN/双缓冲/k 展开四个维度扫过（生产 kernel = 最优的「8x8 + 预取」变体）。
瓶颈是**指令发射**：每 k-step 104 条指令里只有 64 条 FMA，LDS/ST 与 FMA 抢同一个发射槽
（`issue_mix_probe.rs` 专门复现了这个配比）；WGSL 侧还缺 `cp.async`、寄存器控制与
`__launch_bounds__`，所以「逼近 cuBLAS」的收益上限大约是 +15–20%（向量化 LDS/换 smem 布局）。

**长音频那边还有一笔**：他们 180s_en 的分相位之和是 9569 ms，而其记录总时长 9873 ms ——
**残差 ~300 ms 才是他们的音频前端**；我们的 `load_audio_wav`（hound + **soxr HQ**）要 **1450 ms**。
即 **前端上我们白亏 ~1.15 s**，正好就是 180s_en 的总差距（10.503 vs 9.873）。
原因是两边选了不同的重采样器：我们用 soxr HQ（为了与 librosa **逐位一致**，HANDOFF 的
「不要回退」清单里有它），他们用 **`rubato::SincFixedIn`**（sinc_len 256 / oversampling 256）
—— 更快，但不是逐位一致；他们的 12 组文本同样对得上 python-hf。
⇒ 换重采样器能拿下 ~1.1 s（180s_en RTFx 16.8 → ~19.2，直接超过 CUDA 的 18.2），
代价是放弃「mel 与 librosa 逐位一致」这条属性。**这是策略决定，没有用户点头不要动。**

---

## RTFx 优化（本窗口第二轮，全部 12/12 复验过）

1. **`embed_tokens` 不再整表物化**（`inference.rs`）。`weights::get_f16` 每次调用都把整张表
   重建成 `Vec<f16>`（0.6B 155 M 元素 / 1.7B 622 M），约 0.3–0.6 s + 几百 MB 抖动。
   现在 `RawTensor::append_f16_row_le` 直接从 mmap 的字节里拷行（f16 是 memcpy，
   其它 dtype 逐元素转换）。
2. **soxr 的 Release 优化丢过一次**（`build.rs`）。`cmake` crate 是「替换」而不是「追加」
   `CMAKE_C_FLAGS_RELEASE`，于是 CMake 默认的 `/O2` 没了，vendored soxr 一直是无优化编译。
   补进 `CMAKE_C_FLAGS`（不会被同样的方式覆盖）。输出**逐位不变**（`wave16k.f32` MD5 相同）。
   *试过但没用*：`WITH_OPENMP=ON` + `soxr_runtime_spec(N)` —— soxr 的 OpenMP 只在
   `num_channels > 1` 时并行（`soxr.c`），我们是单声道，1 vs 8 线程同样 1.3 s 且逐位一致。
3. **decode attention 的 K/V 读取改成 `vec4`**（`shaders.rs` 的 `gqa_decode_single` /
   `gqa_decode_split_p1`）。原来每个 lane 走自己那一行 256 B，warp 内 32 个地址相隔 256 B，
   一条 load 打散成 32 个 cache line（实测 ~30 GB/s，而同样大小的 GEMV 形状有 289 GB/s）。
   改成一次读 4 个字，**每个字的解包与累加顺序一字不改** ⇒ 逐位一致。
   180s 的 decode 从 **11.66 → 9.86 ms/step**。
4. **rms_norm 折进消费它的 GEMV**（`shaders::gemv_norm`，用于 qkv / gu / lm_head 三处）。
   那些 norm 是「1 个 workgroup」的 dispatch，每个在临界路径上值 ~30 µs（消融实测：两层
   norm 全去掉省 1.1 ms/step）。两个必须一字不差的地方：
   * **归约树**：`rms_norm` 用 `bs = block_for_reduction(hs) = 1024` 个线程，每个「虚拟线程」
     的部分和是 `j = t, t+bs, …` 那些字的 `x²+y²`，然后 `red[t] += red[t+s]`（s 从 bs/2 减半）。
     GEMV 只有 256 线程，于是每线程算 4 个虚拟部分和，并把前两轮**按同样的配对**并进本地加法。
   * **被消费的值**：`rms_norm` 把归一化结果写成 f16，所以 GEMV 必须读
     `pack2x16float(x * inv_rms * w)`（同样的乘法顺序），不能读 f32 中间量。共享内存里的
     暂存就是这个表达式。
   180s 的 decode 再降 **~0.4 ms/step**（6010 ms），12/12 复验仍然逐位一致。

**decode 现在的时间分布**（180s_en，约 9.4 ms/step，用临时消融开关量的，已移除）：

| 部分 | ms/step | 说明 |
|---|---|---|
| attention p1（scores+AV 扫描） | **4.2** | 指令吞吐受限：scores 每 key 约 350 条指令（unpack+2 FMA+add），AV 每 (key,dim) 约 6 条。Pascal 的 f16 是 1/64 速率，只能 f32 标量 |
| mlp（gu+silu+dp） | 2.2 | 形状小 ⇒ 只有 105–156 GB/s（大形状 LM head 有 289 GB/s） |
| qkv | 1.3 | 同上 |
| extract | 1.0 | 24 个 workgroup；每层 57 µs 基本是前一个 kernel 的收尾 + 下一个的启动 |
| merge | 0.9 | 16 个 workgroup，每层 48 µs，同上 |
| lm_head | 0.7 | 297 MB，288 GB/s ✓ 已达带宽 |
| o_proj | 0.1–0.6 | |
| ~~rms_norm ×2/层~~ | ~~1.1~~ | **已折进 GEMV** |

⇒ 剩下的两个方向：(a) AV 扫描的指令数——每 (key,dim) 6 条里只有 1 条是 FMA，但
`T_SPLIT=2` 的分块方式限制了「一线程多 dim」的向量化（改了就不逐位一致）；
(b) `extract`/`merge` 的启动气泡，需要把它们折进邻居 kernel（merge 折进 o_proj 的
prologue 可行但会多读 40 KB/workgroup）。silu 折进 dp **不可行**：需要 7 KB smem，
而且 exp 会按 workgroup 数（128）重复计算。

---

## 下一步（RTFx；对齐已无欠账）

1. prefill 1788 ms（CUDA 1414 ms）：`gemm_bench` 已扫空 TM/TN，要动数据流而不是 tile。
2. 音频塔 enc 1050 ms：conv stem 只占 ~63 ms，其余是 18 层 transformer 的 GEMM。
3. decode 6024 ms：上面那张表就是清单（注意力 AV 的指令数 / extract+merge 的气泡）。
4. `--diag-enc` 里 `layer 17` 还有 917/174720 个点超 5e-3（max 0.128）—— f16 逐层累积，预期内。

---

## ★ 本窗口修掉的四个真 bug（都验证过，别再重复怀疑）

**20. `enc.ex` uniform 从未写入**（`audio_encoder_gpu.rs` 的 `layer()`）。
`u_ex` 只被创建、绑定，没有任何 `write_buffer`，所以 `ExCfg.n_tokens = 0`，
`audio_extract_qkv` 第一句 `if (tok >= cfg.n_tokens) { return; }` 让**每个线程立刻返回**，
Q/K/V 保持分配时的 0。后果：scores/probs/av 全 0，attention 退化成 `out_proj` 的 bias。
**最坑的是它看起来自洽**：oracle 是「拿 GPU 自己的 q/k/v 重算」，0·0=0、softmax(0) 也一致，
所以 `scores/probs/av out` 三行全绿。现在 oracle 会打印 `max|q|/|k|/|v|`（必须非零）。

**21. attention GEMM 的 `bsc` 传成「字」而不是「元素」**。
`prefill_gemm` 的 epilogue 是 `C[(cbase + row·ldc + col) / 2]` —— `cbase` 按**元素**解释，
所以 `bsc` 必须是**元素**（解码器一直是这么传的，`decoder.rs` 里有注释），
而音频塔传的是 `wpad*wpad/2`（字）。batch=1 时 `cbase = wid.z·bsc = 0`，看不出来；
attention 一批 28 个 block 时，除 block 0 外每个块都被写到**前半个块**的位置，
相邻块因此互相覆盖，谁赢取决于 workgroup 跑的顺序 → **同一份输入两次运行结果不同**。
修法：`bsc = wpad*wpad`（scores）/ `wpad*hd_pad`（AV）；`shaders::prefill_gemm` 的文档已写明单位。

**22. `audio_win_pack` 的 `qp` 越行写入**。
`jw`（head-dim 字下标）跑到 `hd2..pad_n/2` 时 `in_range == false`，读出来的是 0；
vp 的行宽是 `pad_n/2`，写 0 正好是「head 维 padding」，没问题；
但 `qp` 的行宽只有 `hd2`，同一个地址落到**下一个 token 的行**，
和那一行自己的线程**抢着写**（真实值 vs 0）—— 哪边赢看调度。
修法：`qp` 的两条写放进 `if (in_range)`。

**23. 诊断侧 `conv_block_stages` 的 epilogue 忘了 tile 偏移**（`audio_encoder.rs`）。
`want_raw` 路径下 `raw` 是整块 `[c_out][b·plane]`，epilogue 应读 `(b0+ib)·plane`，
原来读 `ib·plane` → **第 2 个 tile 起，act 全是第 1 个 tile 的复制品**。
单 chunk（一个 tile）时两者恰好重合，所以上一窗口的「单 chunk 复现」全绿，
而 15s（15 chunk / 2 round）的 `conv{1,2,3} +bias/GELU` 三行全红（bad 30–45%），把 GPU 冤枉了两轮。
修好后这三行是 0.001976 / 0.002727 / 0.001862 —— 与上一窗口记的数字逐位吻合。

**24. 诊断把 `attn_flat` 和 `dbg_attn` 比**（`inference.rs`）。
`enc.attn_flat` 是 **out_proj 之前**的张量，`dbg_attn` 是**之后**的（含 out_proj），两者永远对不上
（max 2.27、bad 86%）。现在新增 `CpuAudioAttention::flat()` 与 `dbg_attn_flat()`，同口径比 → max 0.003，0 bad。

---

## 诊断怎么读（`--diag-enc`，默认就是 GPU 塔）

15s_en 当前输出（全部干净，可作回归基线）：

```text
    conv1 operand        max|d| 0.000000  bad 0/432000        ([k][pos])
    conv1 GEMM           max|d| 0.000976  bad 0/23040000
    conv1 +bias/GELU     max|d| 0.001976  bad 0/23040000
    conv2 operand        max|d| 0.000000  bad 0/51840000
    conv2 GEMM           max|d| 0.003482  bad 24/5760000       <- 24 个点在 f16 边界上
    conv2 +bias/GELU     max|d| 0.002727  bad 11/5760000
    conv3 operand        max|d| 0.000000  bad 0/13478400
    conv3 GEMM           max|d| 0.001710  bad 0/1497600
    conv3 +bias/GELU     max|d| 0.001862  bad 0/1497600
    head 0 win 0 z 0 (valid 104): operands qp 0.0000 kt 0.0000 vp 0.0000 | scores bad 0 | probs bad rows 0 | av bad rows 0
    head 1 win 0 z 2 (valid 104): 同上（这一行以前是全红）
    head 0 win 1 z 1 (valid  91): 同上
    head 1 win 1 z 3 (valid  91): 同上
    max|q| 4.9141  max|k| 7.3594  max|v| 2.9629  (must be non-zero)
    attn residual        max|d| 0.002677  rms 0.000165  bad 0/174720
    LN1 (normed)         max|d| 0.000968  bad 0/174720
    fused QKV            max|d| 0.002124  bad 0/524160
    attention block      max|d| 0.002973  bad 0/174720
    scores / probs / av  max|d| 0.125 / 0.0006 / 0.0006  bad 0
    layer 0              max|d| 0.005963  bad 1/174720
    layer 17             max|d| 0.127735  bad 917/174720     <- f16 累积，预期内
    packed / h           max|d| 0.006 / 0.0035  bad 4 / 0
```

**注意力 oracle 的方法学（血的教训）**：它只用 GPU 自己写出的 `q/k/v` 重算 dot/softmax/AV，
所以只能证明「GPU 内部自洽」，**证明不了 q/k/v 本身是对的** —— bug 20 就是全 0 的 q/k/v 让三行全绿。
现在它：① 覆盖 **head 0/1 × 首个/最后一个窗口**（head 与 win 两个下标都动到）；
② 先比 **packed operand（qp/kt/vp）与主机的 q/k/v**（错在 repack 还是 GEMM 一眼可分）；
③ 打印 `max|q|/|k|/|v|`，全 0 时那行写着 `(must be non-zero)`；
④ `scores` 容差随幅值走（f16 在 |s|≈300 处的 ulp 就有 0.25）。

---

## 上游 API 对齐（2026-09-14，对照 transformers 的 `Qwen3ASRProcessor`）

**权威口径是 transformers 那条路**（`AutoProcessor.apply_transcription_request(audio, language=…,
prompt=…)` + `AutoModelForMultimodalLM.generate` + `processor.decode(…, return_format="parsed")`），
因为 12 组 gold 就是它生成的（`D:\Qwen3-ASR\run_hf_cuda.py`）。`qwen_asr` 那个包在本机 env 里
**import 就报错**（vendored modeling 与 transformers 5.17 的 `check_model_inputs` 签名不匹配），
而且它的 parser 与 transformers 的 parser 有几处细节不同 —— **以 transformers 为准**。

本窗口按它补齐的：

* **`context`（热词）= chat template 的 system 消息内容**（之前我们的 system 段是硬编码空）。
  `WgpuAsr::transcribe_file_opts(..., &TranscribeOptions { context, language })`；
  CLI 新增 **`--context`** / **`--lang`**。
* **`language`**：接受 ISO 码或全名（大小写不敏感）→ 规范全名，其它报错
  （= transformers 的 `resolve_language`）；随后按官方做法把 `language {NAME}<asr_text>`
  预填到 assistant 轮。
* **`parse_asr_output` 重写为 `_parse_single_output` 的逐行移植**：不再规范化语言、
  只看第一行非空、`prefix == "language none"` 而非 contains、无 tag 时语言为空、
  以及「强制语言时语言字段为空」（meta 在 prompt 里，生成的文本不带 meta）。
* `detect_and_fix_repetitions` 的 pattern 窗口从 96 改回上游的 **20**（原来会把上游不折叠的
  重复也折叠掉，属于行为差异）。

**逐字验证（6 个用例，官方 vs 我们，0.6B / 15s_en）**：

| 用例 | 官方 language | 官方文本 | 我们 |
|---|---|---|---|
| 无 context | English | MATCH | 文本逐字一致、language 一致 |
| 热词 context | English | MATCH | 一致（连把 `swing` 变成 `wind` 的偏置都一致） |
| `language="English"` | None | MATCH | 一致（language 也为空） |
| `language="Chinese"` | None | MATCH | 一致 |
| `language="en"`（ISO） | None | MATCH | 一致 |
| 热词 + `language="English"` | None | MATCH | 一致 |

补完这些之后 **12/12 基线复测仍全 MATCH**（prompt 空 context 时与原来逐 token 相同）。

**逐层对齐证据**（0.6B / 15s_en，官方 transformers 路径 vs 我们）：

| 层 | 结果 |
|---|---|
| **mel 前端** | 形状 `128×1500` 一致；**max\|d\| = 1.7e-5**、mean 1.7e-7（f32 最后一位量级，双方 FFT/算子顺序不同所致） |
| **prompt** | token 数一致（15s_en：210），空 context 时逐 token 相同；带 context / 强制语言时输出也逐字一致 |
| **生成 token 序列** | 我们 == 官方 **去掉末尾 `<\|im_end\|>`**（官方 `generate` 把终止符也算进 `generated_ids`，我们遇到 EOS 就停；两边 `decode(skip_special_tokens=True)` 都会把它去掉） |
| **文本** | 12/12（0.6B/1.7B × 6 音频）逐字 MATCH；6 个 context/language 用例逐字一致 |
| **特殊 token / 术语** | `<asr_text>` = 151704、`<\|im_start\|>`/`<\|im_end\|>` = 151644/151645、**EOS 集合 = 官方 `generation_config.json` 的 `[151643, 151645]`**、greedy（`do_sample=false`）、`max_new_tokens` 默认 512 —— 全部一致 |
| **meta 术语** | 输出 `language <NAME><asr_text><text>`、语言规范表（30 个全名 + ISO 码）与官方逐条相同 |

**API 面术语差异**（我们 vs 官方；功能等价，名字不同）：

| 官方 | 我们 | 说明 |
|---|---|---|
| `apply_transcription_request(prompt=…)` | `TranscribeOptions.context` / `--context` | 官方 `qwen_asr` 包叫 `context`，transformers 处理器叫 `prompt`，两者指同一个东西 |
| `ASRTranscription{language, text, time_stamps}` / parsed dict `{language, transcription}` | `TranscribeResult{text, language, raw_output}` | 字段名不同；`raw_output` 对应官方 `decode(return_format="raw")` |
| `decode(return_format="raw"｜"parsed"｜"transcription_only")` | 只做 parsed（`prompt::decode_result`） | 没有 raw/transcription-only 的开关 |
| `get_supported_languages()` | 无公开入口（`prompt::LANGUAGE_CODE_TO_NAME` 是 crate 内部） | 需要的话加一个 pub 访问器即可 |
| `return_time_stamps=True` + `Qwen3-ForcedAligner` | — | 第二个模型，未移植 |
| gold harness 的 `max_new` 规则 `max(256, min(2048, audio_s*8 + 64))` | CLI `--max-new`（默认 512） | 我们跑 180s 用 700（> 实际 585/641，不会截断） |

**上游还有、我们没有的**（都属于另外的产品面，不是推理链）：时间戳（`Qwen3-ForcedAligner`
是第二个模型）、流式（`ASRStreamingSession` / vLLM 后端）、批量（一次多音频）、
>20 分钟音频的**静音点切片**（`split_audio_into_chunks`，阈值 1200 s；我们的 6 个 fixture
都 ≤180 s，所以 gold 与我们的单段推理一致）、微调、vLLM serving / Gradio / DashScope。

### 热词实测：能不能矫正转录（0.6B，真实错误）

错误候选来自 **0.6B 与 1.7B baseline 的分歧**（同一段音频，两个尺寸不一致处通常有一边错），
再结合上下文判定：

| 片段 | 我们的错误 | 判定依据 | 给的热词 context | 结果 |
|---|---|---|---|---|
| 90s_en（交易课） | `order book` | 1.7B 说 `order block`；ICT 交易术语；上文在讲 swing trading | `Swing trading course. Terms: order block, time frame, four-hour chart, …, liquidity, entry model, stop loss.` | **修好** ✓ `order book → order block`；全文只有 6 处变化（含 `timeframe → time frame`），**零副作用** |
| 180s_en（动画） | `Robo headbox.` | 下一句 "Please stop hitting me with my own arm"；1.7B 说 `headbutt` | 人名 + 动作词（`Robo, Panther, Beastbot, Banana Blaster, Jimmy, the Shroud, pandemonium particles, headbutt, intruder, lockdown`） | 修好 `headbox → headbutt` ✓，但**也丢了 4 处短句**（"Help, Panther!" / "Hey." / "Uh oh." / "How? Come on."）并新增 1 处错（`dimwit's snarl → dim wit. It's`） ✗ |
| 同上 | 同上 | 同上 | **只给人名**（`Robo, Panther, Beastbot, Banana Blaster, Jimmy, the Shroud, pandemonium particles.`） | 没有删词/新错 ✓，把 `Banana blaster → Banana Blaster` 修正 ✓，但 `headbox` 没被修正（没点它） |

**结论/用法**：context 是**真有效**的（`order book → order block` 是实打实的矫正），
用法上：
1. **领域术语最划算**：给一串同领域词（不只是那一个词）——模型会锁到该领域，修正未见过的词，
   且这类 context 在长音频上表现稳定（90s_en 零副作用）。
2. **纯专有名词列表最安全**：只影响拼写/大小写（`Banana Blaster`），不删词、不引入新错。
3. **长 context + 小模型（0.6B）在长音频上有风险**：会掉短句/引入新错 —— 改完要 diff 一遍；
   短音频（15s/30s）没观察到这种副作用。
4. 上下文**只对当次请求生效**；加 context 后输出自然不再等于无 context 的 gold，
   `--baseline` 那种逐字比对只适用于默认（无 context）路径。

---

## 流式 API + WER/CER 评测（2026-09-14 第三轮）

### 流式（补齐「上游还有、我们没有的」第一条）

新增（形状对照 `qwen3-asr-rs` 的 `AsrStreamingSession` / `transcribe_streaming`）：

| 入口 | 说明 |
|---|---|
| `transcribe_file_streaming` / `transcribe_samples_streaming` | 每 token 回调 `StreamToken{token_id, text_so_far}`（原始解码，不做 language 切分/重复修复） |
| `supported_languages()` | 30 个语言名（= 官方模型卡清单，与 `prompt::SUPPORTED_LANGUAGES` 同源） |
| `create_streaming_session` → `push_samples` → `flush` / `flush_streaming` | 增量音频进、flush 出文本；每 104 token（8 conv chunk = 800 mel 帧）编码一个注意力窗口，内存 O(窗口) |
| CLI | `--stream`（增量打到 stderr，stdout 仍是最终文本）、`--session`（按 1 s 喂进 session）、`--languages` |

**验收**：6/6 fixture 的 session 文本与整段路径 MATCH（15s/30s/90s/180s 全语言）。

**一处必须记住的偏差（原设计「逐位等价」的说法不成立）**：mel 的 log-max 归一化
（`(max-8, +4)/4`）是**每次 `extract` 调用内**取 max。逐窗口编码时每个窗口用自己的 max，
所以窗口 mel 与整段 mel 的差 **恰好**是常数 `(M_whole − M_window)/4`（探针实测，见
`mel.rs::slice_extraction_matches_whole_clip_modulo_normalization`）。后果：文本 5/6 fixture
逐字相同，`180s_zh` 差 2 处标点（`，`→`。`）+ 4 字（丢 `嗯，`）。真流式**不可能**知道整段的
max，参考实现同样有这类漂移（它每次 push 用「已累积 buffer」重取 mel），因此选择保留真流式
（增量编码、内存有界）并在文档记录偏差，而不是缓存整段音频。

### WER/CER + RTFx 四组对比（真实音频 + 人工标注，parity 之外的第二个指标）

parity（与 python-hf 逐字相同）是**回归信号**，不是目标；目标是与**人工转写**的 WER/CER。
手头 6 个 fixture 没有人工真值（`docs/baseline/texts/` 全是 python-hf 输出），所以引入
**FLEURS test**（29/30 个受支持语言；该镜像缺粤语 `yue_hant_hk`，22 个汉语方言 FLEURS 无覆盖）。

工具链（都在本仓库，数据在 git-ignored 的 `eval_data/`）：

```text
tools/fetch_fleurs.ps1      从 modelscope.cn/pengzhendong/fleurs 拉数据（HF 不通）
tools/run_fleurs_sweep.ps1  本移植版（-Tag w06 -Even 20），导出 hyps/langs/summary
tools/run_python_hf.py      Python-HF 对照（--even 20 --tag py06，同一批 clip）
tools/run_eval_all.ps1      四个 run + 最终表，**严格串行**（8 GB 卡装不下两个 ASR 客户端）
tools/score_asr.py          归一化 + S/D/I 编辑距离（jiwer 交叉校验）+ 两系统逐句对比
tools/score_final.py        最终表：4 组 WER/CER + 4 组 RTFx + 2 组逐音频对齐度
src/bin/eval_asr.rs         评测入口（`--even N`：按文件大小≈时长排序后均匀取样）
```

**协议**：29 语言 × **20 条**（按文件大小排序后均匀取样，覆盖短/长音频），greedy、
`max_new 256`、语言自动识别；两边同一批 clip；计时口径同为「读 wav → 出文本」，
时序数字全部在**单作业**下测得。

**结果（`docs/eval-fleurs-final.md`，2026-09-14）**：

| 系统 | WER/CER 均值（29 语言） | RTFx 均值 |
|---|---|---|
| **wgpu 0.6B** | **21.50%** | **14.33×** |
| python 0.6B | 21.51% | 7.34× |
| **wgpu 1.7B** | **12.99%** | **7.01×** |
| python 1.7B | 12.98% | 4.60× |

- **逐音频对齐度**（与 Python 输出逐字相同）：0.6B **96.4%**、1.7B **97.4%**；
  多数语言 100%，少数 80~95%（差异都是个别 clip 的标点/单字）。
- **质量无退化**：每语言 WER/CER 与 Python 相差 −0.68 ~ +0.66 pp（多数 ±0.00），
  1.7B 明显强于 0.6B（均值 13.0% vs 21.5%）。
- **RTFx**：wgpu 分别是 Python 的 **1.95×**（0.6B）和 **1.52×**（1.7B）；慢语种
  （希腊/印地）两边都低，因为那批 clip 更长。

**度量口径（读表前必看）**：en 用 whisper `EnglishTextNormalizer`（数字/序数/缩略）；
zh 加 `cn2an`；ar/fa 去变音符/tatweel 并折叠 alef/ya/ta-marbuta（21.3% → 16.4%）；
zh/ja 两边去掉 FLEURS 参考里**没被朗读的拉丁人名**（`--strip-latin`）；其余用
whisper `BasicTextNormalizer`。已知偏悲观项：日文表记差异（子供/子ども）未归一、
中文时间表达（十一点三十五）cn2an 不转。

### 顺带修的真 bug

- `load_audio_wav`：FLEURS 镜像里 `ar_eg` 有一个 `data` 块声明 151492 字节、实际 147398 字节的
  **截断 wav**，`hound` 直接报错。改成**读多少用多少**（libsndfile/ffmpeg 行为）并打 warning。
- `prompt.rs` 的两个单测还是旧的 `parse_asr_output(raw, lang)` 签名，`cargo test` 编不过 —— 已修。

---

## RTFx 第三轮（2026-09-14）：实测账本 + 一次已落地的优化

**工具（都在仓库里）**：`tools/verify_all.ps1`（6 fixture × 2 模型，对齐 + 分相位 + 时钟，
每次改动先过它）；`QASR_DUP=<op>` 环境变量（`decoder.rs`）把某个 op **重复派发一次**——
这些 op 都是纯函数写同一缓冲，重跑结果不变 ⇒ token 流不变，时间差就是该 op 的真实成本
**含它的启动气泡**（op-by-op 时间戳分不出来）。解码侧支持：
`qkv/extract/gqa/o/gu/silu/dp/lm`；prefill 侧支持：`p_scores/p_softmax/p_av/p_repeat`。

**180s_en / 0.6B / 1911 MHz 实测（单相位之和 = elapsed）**

| 相位 | 时间 | 内部构成（消融实测） |
|---|---|---|
| 前端（44.1k→16k soxr HQ） | 1330 ms | **非编译问题、非 recipe**：cubic 与 HQ 同为 ~1.6 s；44.1k→16k 是非整数比（441/160）多相滤波本身贵；48k→16k（整数比）只要 ~46 ns/样本，快 4.5×。换算法=策略决定（会放弃与 librosa 逐位一致） |
| enc（音频塔） | 1000 ms | ≈ 塔权重 300 MB / DRAM 带宽 ⇒ **已在带宽上，无余量** |
| prefill | 1770 ms | scores GEMM 295 / softmax 243 / AV GEMM 283（**注意力 46% 全在搬 s² scratch ≈680 MB**）/ repeat_kv 仅 14 ms / 其余是权重 GEMM（~690 ms 带宽下限）+ elementwise |
| decode | 5730 ms | gqa 3.9 ms/步（**42%**）/ gu 1.3 / dp 1.1 / qkv 0.97 / lm 0.87（297 MB/步 ⇒ 带宽饱和）/ extract·silu ≈ 0（被掩盖）。四个投影合计 4.3 ms/步，而带宽下限 ~1.6 ms/步 |

**已落地的优化（每一步都过了 12/12 门禁）**：

1. **AV 扫描「一线程一对 dim」**（`gqa_decode_single` / `gqa_decode_split_p1`）：两个 dim 共用同一
   f16 字 ⇒ 1 次 LDS + 1 次 unpack 喂 2 条 FMA；每个 dim 的 key 步进不变 ⇒ 构造上逐位一致。
2. **因果 score tile 跳过**（`prefill_gemm_causal`，只给 scores GEMM 用）：`n0 > m0 + BM - 1`
   的 tile 整块在因果线上方，softmax 只读 `j < row+1` ⇒ 直接 return，逐位一致。
3. **AV GEMM 的因果 k 界**（`prefill_gemm_causal_av`，transb=1）：A 操作数是 softmax 输出，
   `row+1` 之后**恰好是 0**，加 0 不改和 ⇒ k 循环只扫 `k < m0 + BM`。

效果（0.6B，单作业实测）：prefill 180s_en **1772 → 1522 ms（−14%）**、180s_zh 1799 → 1528；
decode 每步 gqa 由 ~4.2 → ~3.9 ms。RTFx：

| | 15s_en | 30s_zh | 90s_en | 90s_ja | 180s_en | 180s_zh |
|---|---|---|---|---|---|---|
| 0.6B 起点 → 现在 | 23.59 → **24.38** | 21.03 → **21.27** | 19.40 → **19.74** | 23.44 → **23.95** | 17.03 → **17.77** | 19.22 → **20.14** |
| 1.7B 起点 → 现在 | 11.23 → **11.26** | 10.34 → **10.36** | 10.84 → **10.96** | 12.98 → **13.12** | 10.23 → **10.48** | 10.86 → **11.08** |

**decode 内部再拆一层**（消融，180s_en，40 步）：`gqa_p1`（分块扫描）= **3.45 ms/步**、
`gqa_merge`（每 head 一个 14-workgroup 的小 kernel）= **0.78 ms/步**；后者基本是 P1 结束后的
启动/同步气泡（handoff 早先评估：折叠进 o_proj prologue 可行，代价是 +40 KB/workgroup 读取）
⇒ 下一步候选。

**测过的死路（别再试）**：GEMV 的 `xsmem`（X 预展成 f32 smem）255.9 vs 255.2 GB/s；
GEMV 的 `rows2`（一 warp 两行）**200 GB/s，全形状退化 0.77–0.89×** ⇒ 小形状是**并行度受限**
（warp 少了更慢），而 split-K 会改求和顺序、破坏逐位一致 ⇒ 四个投影暂时无解；
`repeat_kv`（我按流量估 ≈0.45 s）实测只有 14 ms —— **先测再改**。

**下一步（按性价比）**：① `gqa_merge` 气泡（~0.78 ms/步 ≈ decode 8%）；
② decode 四个投影的并行度（受逐位一致约束，需新思路）；③ 前端重采样（策略决定，需用户点头）。

---

## 长音频（>4 分钟）：本窗口实测与遗留限制

`15m.wav`（16 kHz mono PCM16，926.93 s ≈ **12,065 token**）把这个仓库从没走过的路走了一遍，
暴露的都是**容量/网格上限**（不是数学错），而且 6 个 fixture 都 ≤180 s，所以以前从没碰到：

**已修的真 bug（都是同类：wgpu 每个 grid 维度上限 65535，超了整条命令缓冲被拒）**

| 位置 | 触发点 | 修法 |
|---|---|---|
| 音频塔 `extract` | `s_pad·nh` > 65535，**~6 分钟** | grid 改成 `(s_pad, nh, hd/2)` |
| prefill `silu` | `s·inter/2/256` > 65535，**~5.8 分钟** | 二维 grid + uniform 里的 `gx` |
| prefill causal `softmax` | `nqh·s` > 65535，**~4.2 分钟** | 同上 |
| `repeat_kv` | ~19 分钟（正好顶到 65536） | 同上 |
| 音频塔 `bias_gelu` | FFN/proj1 的 `n/512`（85k / 170k） | 同上 |

算术一律没动（只是把平坦下标换成 `x + y·gx·threads`）⇒ 短音频 12 组仍全 MATCH。

**真正的硬限制：prefill 的因果注意力 scratch 是 O(s²)**（`decoder::prefill` 里的 `scores`/`attn`
两块，各 `nqh·s²·2` 字节）。12,065 token 时 **≈4.7 GB × 2** ⇒ 既超 8 GB 显存，也超单绑定
**2047 MiB** 上限；wgpu 的失败方式是**静默出垃圾**（表现为「prefill 70 ms + 乱码」）。
现在 `prefill` 会**显式报错**（并把可支持的最大 token 数算给用户看）。
`DECODER_MAX_SEQ` 从 4096 提到 **8192**（与 scratch 上限自洽；KV cache +~450 MB）。
参考实现（transformers）走 **SDPA**，没有 O(s²) scratch，所以它 15 分钟能跑 —— **这是我们在长音频上唯一
与 Python 原版不一致的地方**，要补就得把 prefill 注意力分块（两遍式精确 softmax），是个正经的 kernel 工程。

**实测（0.6B）**

| 输入 | 结果 |
|---|---|
| 15m.wav（12,065 token） | **明确拒绝**：`seq 12065 + max_new 512 exceeds decoder max_seq 8192`（之前是静默垃圾） |
| 6m.wav（360 s / 4,695 token） | **跑通**：mel 50 / enc 1975 / prefill 4996 / decode 14397 ms，**峰值显存 4,600 MiB**；输出连贯 |
| 同一 6m.wav 用官方 Python 跑 | prompt token **4,695 = 我们的 4,695**（逐 token 相同）；Python 83.2 s vs 我们 21.4 s；**文本 4,519 vs 4,520 字符，全长只有 1 处不同**（`you're` / `you are`，属 f16 边界处的取整抖动） |

⇒ 结论：**结果是逐字对齐的（连没见过的 6 分钟长音频都只差一个缩写），长音频的容量是设计限制**。

**官方 Python 在同一台机器上的长音频表现（同 15m.wav，0.6B fp16）**

| 输入 | 官方 transformers（`attn_implementation: sdpa`） | 我们 |
|---|---|---|
| 6 min（4,695 token） | 跑通：83.2 s，torch 峰值分配 **5.25 GiB** | 跑通：21.4 s，nvidia-smi 峰值 **4,600 MiB** |
| 15 min（12,065 token） | **`CUDA error: out of memory`**（`generate` 第一次 forward 就挂；nvidia-smi 峰值 8,009/8,192 MiB） | **明确拒绝**（scratch 超 2047 MiB 单绑定上限），不给垃圾 |

⇒ 8 GB 卡上**两边都吃不下 15 分钟**（0.6B！）——它并不是「官方能跑我们不能」：
Pascal(sm61) 没有 flash-attention kernel，torch 的 SDPA 在这张卡上退化成会materialize
`s²` 分数矩阵的路径（16·12065²·2 ≈ 4.7 GB），所以它也是死在同一类内存上。
我们这边可用上限约 **10 分钟**（scratch 守卫 ≈ 8.2k token，`max_seq` 8,192），比它略高；
要真正支持任意长度，两边都得用分块/流式注意力（他们的 vLLM 后端即是）。

### 5 分钟三方实测（0.6B / 同一 `5m.wav` / max_new 2048）

| | 官方 transformers | CUDA 手写版（`qwen3-asr-rs`） | 我们（wgpu） |
|---|---|---|---|
| prompt token | **3,915** | — | **3,915** |
| 峰值显存（nvidia-smi） | 5,292 MiB | 5,802 MiB | **4,086 MiB** |
| 耗时（不含加载） | 83.5 s | **17.0 s** | 21.0 s |
| 输出 | 5,132 字符 | **不可复现**：四次运行 5,132 / 5,155 / 5,167 / 5,178，两次 language 字段是垃圾（`!language…` / `!!!! English`）✗ | 5,132 字符 |
| 逐字比对 | — | 无法比对 | **与官方 md5 相同（`457545e0c1bf`）、989 词、0 处差异** ✓；自身两次运行完全一致 ✓ |

CUDA 版在长上下文的表现（重复循环 + 语言字段垃圾）与它自己 ROADMAP 里记录的
「split-K 路径长音频不可复现 / 解码器陷入重复循环」是同一症状 ⇒ 那条路在 5 分钟上
**不能用于对齐**（不是我们的问题，但对比时要知道）。

### 时长 × 实现（8 GB 卡，0.6B）

| 输入 | 官方 transformers | CUDA 手写版 | 我们（wgpu） |
|---|---|---|---|
| 6 min | ✓ 5.25 GiB | 未测 | ✓ 4,600 MiB |
| 10 min | ✗ OOM | ✗ `CUDA_ERROR_OUT_OF_MEMORY`（音频塔阶段） | ✓ **7,203 MiB** |
| 15 min | ✗ OOM（峰值 8,009 MiB） | 未测（5 min 已不可复现） | ✗ 明确拒绝（单绑定 2047 MiB 上限） |

**换成 16 GB 卡会怎样**（`max_storage_buffer_binding_size` 是驱动的固定值 2047 MiB，
与显存大小无关）：

| | 16 GB 能否跑 15 min | 说明 |
|---|---|---|
| 官方 | **预计可以** ✓ | 分数矩阵 f16 ≈ 4.7 GB + 权重 1.25 + KV 1.1 + 激活 ≈ 7–9 GB |
| 我们 | **换卡没用** ✗ | 卡在单绑定 2047 MiB：scratch 是**两块整分配**（各 16·s²·2B）。必须先做**分块注意力**；做完后 15 min 只需 O(s·hd) ≈ 2–3 GB，8 GB 卡都够 |
| CUDA 版 | 内存上或许够（f32 scratch 12k token ≈ 11.6 GB，16 GB 很紧，24 GB 稳） | 但它 5 min 就已不可复现，内存不是它的首要问题 |

---

## 布局契约（单一来源，改代码前先读这里）

`src/audio_encoder_gpu.rs` 的 **`ConvLevel`** 是唯一的几何来源；`LevelView` / `enc.level(i)`
把它发布给诊断，所以**比较代码和 kernel 不可能各说各话**。`CONV_TILE = 8` 个 chunk 一轮。

```text
operand (im2col 输出 = transb GEMM 的 B)   [k_pad][n_all] f16
    一个字 (k, 位置对) 在 k*(n_all/2) + col/2，col = chunk*plane_pad + p
    通道寻址: 读 (chunk0+chunk)*in_chunk + ic*in_ic + tap，tap 来自 [plane*9] 的 tap 表
    k >= c_in*9 / chunk >= 本轮 chunk 数 / p >= plane 一律**显式写 0**

activation (GEMM 的 C)                     [c][pos] 通道优先（= CPU 参考的布局）
    元素 (c, chunk, p) 在 c*n_all + chunk*plane_pad + p
    m_pad = align(c_out, GEMM_BM) 行、n_all 列，ldc = n_all

mel 输入                                   [chunk][mel_bin][frame]，行距 w0 = pad_even(cs)

attention 的 per-(head,window) 分块          z = head*n_win + win
    qp[z][wpad][hd]      A(scores)，行距 hd/2 字
    kt[z][hd][wpad]      B(scores, transb)，行距 wpad/2 字
    vp[z][wpad][hd_pad]  B(AV, transb)，行距 hd_pad/2 字
    scores/probs[z][wpad][wpad]，attn_out[z][wpad][hd_pad]
```

* **哨兵只有一个**：`shaders::TAP_OOB`，由 Rust 注入 WGSL（`audio_im2col()` 里 `format!`）。
  两边曾经不一致（Rust `0xFFFF_FFFF` / WGSL `0xFFFF`），后果是每个越界 tap 都被当成合法地址。
* **k 必须按 `GEMM_BK`(=16) 对齐**，不只是 4：k 循环按整 16 宽 tile 走，`k=12` 时最后 4 个 k
  会读到 operand 行外并把乘积**累加进输出**。`pad_k_tile()` 现在按 BK 对齐。
* **bias 向量按元素打包**（两个字一组），GEMM 的 bias 变体一次读一对列正好是一个字；
  `bias_gelu` 的 channel-major 模式要 `select` 取半字（把它当字下标会让通道翻倍）。
* **每个"形状"一个 `ScaleCfg`**（`u_sc[5]`：c1/c2/c3/ffn/proj）。`queue.write_buffer`
  在下一次 submit 才生效，共享 uniform 会让同一条命令缓冲里所有 `bias_gelu` 读到最后一次写入的值。
* **每个 GEMM 一个 `GDims` slot**（`u_gd`，256 B 一槽 + 动态偏移），同上原因。
* **batch 步长的单位**：`bsa`/`bsb` 是**字**，`bsc` 是**元素**（epilogue 除以 2）。见 bug 21。
* 每个 round 单独 submit：`u_im` / `u_pm` 携带 per-round 值，同一 submit 里两个 round 会互相盖掉。

---

## 可用探针

| 命令 | 用途 |
|---|---|
| `--diag-enc [--mel <f32>]` | 逐级 CPU vs GPU 比对 + 主机 attention oracle（4 个 block，含 operand 比对）+ 逐层比对（需 GPU 塔；`--cpu-enc` 下会报错） |
| `--compare-enc` | 端到端 embedding 包络 + CPU/GPU 音频塔耗时 |
| `QWEN3_ENC_PROFILE=1 <transcribe>` | CPU 音频塔相位耗时 |
| `cargo run --bin conv_real_probe` | 真实权重 conv 几何 + 第一性原理重算 |
| `cargo run --bin subgroup_bfly_bench` | butterfly A/B（含位一致性） |
| `cargo run --bin feature_probe` | 本机 wgpu 能力 |
| `cargo run --bin cpu_gemm_probe` | `gemm` crate 在编码器真实形状上的吞吐 |

不要提交：`align_dump/`、`golden/`、`target/`、`*.bin`。

---

## 上一窗口的 bug 清单（保留备查）

**conv stem（1–10）**

1. `conv_taps` 传了**输出**维度当**输入**维度 → 三级 tap 表与位置数整体错一位
   （pos1/2/3 = 800/208/56，应为 3200/800/208）。这是「im2col 输出全是 0」的真因之一。
2. OOB 哨兵两边不一致（见上）。
3. `k` 未按 BK 对齐（c1 的 k=12）。
4. **一块缓冲两种布局**：GEMM 按 `[pos][c]` 写、im2col 按 `[c][pos]` 读、缓冲按第三种尺寸分配。
   现在 `col`（operand）/ `raw`（pre-GELU）/ `act`（激活）各一块，尺寸都由 `ConvLevel` 推。
5. `bias_gelu` 的覆盖数写成 `n/600`（每 workgroup 只做 512 个元素）→ 15% 的元素从没被处理。
6. **conv3 的 bias+GELU 整段缺失**（CPU 参考三级都有）。
7. `bias_gelu` 把偏置按"字"下标索引（通道翻倍）。
8. erf 近似常数写成 `0.147`（A&S 7.1.26 应为 `0.3275911`）→ GELU 差 ~5%。
9. `u_sc` 共享 uniform（见上）。
10. 多 chunk 时 im2col 每个 chunk 都写同一区域（没有 chunk stride），GEMM 也只跑 batch=1。

**permute / 投影（11–14）**

11. permute 的补零分支用 tile 内行号（忘了 `+tok0`）→ 第 2 个 round 把 `packed` 的 91..151 行清零。
12. PE 加在了 permute 的 `[c·f]` operand 上（宽 7680），而参考是加在 **conv_out 输出**（宽 896）上；
    同时 `extract` 又加了一次。现在是 `audio_add_pe` 在 conv_out 之后加一次。
13. `packed` 的行距写成了 `s_pad`（token 数）而不是 `conv_out.k`。
14. 行覆盖不足（一个 16×16 workgroup 覆盖 256 字，permute 行 3840 字 / add_pe 行 448 字）。

**transformer（15–17）**

15. 两个 attention GEMM 的 grid 用整除（`wlen/128 = 0`）⇒ 它们从来没跑过。
16. 分数块 stride 混用（每 head 的 `s_pad` vs 每窗口的 `wlen`），A operand 寻址永远对不上。
17. `out_proj` / `fc2` 用普通 GEMM（`beta=0`）⇒ 残差被丢弃；`qkv/o/fc2/proj2` 的 bias 从未加过。

**诊断自身（18–19，提醒：诊断的错会让 GPU 背锅）**

18. 用 `n_pad` 算读长度、却按 `n` 分配缓冲 → wgpu 报 copy 越界、staging 保持全 0，
    于是「c1_col 全是 0」是**读失败**，不是 kernel 没写。
19. 比对时把 CPU 的 `raw` 行距用在 `act` 上（两者布局不同：`raw` 是 `[c][pos]`、
    `act` 是 `[chunk][c][pos]`）；单 chunk 时两者恰好重合，所以只有多 chunk 才暴露。

---

## 已排除的死路（有数据，别再试）

1. **decode GEMV 用 split-K** —— 会改 butterfly 归约树，不可能 bit-exact。
2. **Pascal 的 f16 算术** —— GP104 (=P104-100) FP16 吞吐是 FP32 的 **1/64**（NVIDIA 官方）。
   实测 2 TFLOP/s 已高出 fp16 上限 15×，说明 kernel 本来就在跑 fp32 算术；`pack/unpack` 是运输不是瓶颈。
3. **prefill 换 TM/TN** —— `gemm_bench` 已扫空，全卡在 ~2.07 TFLOP/s。
4. **`enable subgroups;`** —— naga 故意不实现（issue #5555）。要用能力位 `Features::SUBGROUP`（本机 YES）。
5. **`smem_roof.rs` 式合成微基测** —— 会被强度削减，数不可信（已标 INCONCLUSIVE）。
6. **在本仓库里改 decode kernel 的加法顺序** —— 对齐命根子。
7. **把「用 GPU 自己的 operand 重算」当唯一 oracle** —— 见 bug 20：全 0 也能全绿。
   oracle 必须打印输入幅值，或者拿 CPU 的 operand 来比。

本窗口之前已验证并保留的优化：`shaders::gemv` 的 32-lane xor butterfly 用 `subgroupShuffleXor`
（xor 顺序不变 ⇒ 归约树逐位不变；A/B 1.17–1.18×，0 个输出不同）。
