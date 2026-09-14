# ROADMAP — wgpu backend

Decode-chain and prefill port record.  All numbers are **measured** on the
reference machine (Windows 11, NVIDIA P104-100 Pascal 8 GB, Vulkan, wgpu 30.0.1,
driver 572.75); repro commands regenerate every figure.

---

## Status: stage 3 — **12/12 MATCH** python-hf; RTFx vs CUDA 手写

Live handoff (paths, git, next RTFx cut): **`HANDOFF.md`**. This file keeps
measured stage records. Text decoder and **audio encoder** are both on wgpu now.

`WgpuAsr::load` uploads the text decoder once (`max_seq=4096`). Inference
RTFx on 0.6B `15s_en` is **23.6×** vs CUDA handwritten ~31×; `180s_en` **16.8×**
vs ~18× (encode 1.03 s + prefill 1.79 s + decode 6.02 s at 585 tokens — MATCH,
re-measured 2026-09-14 session 6; the frontend's ~1.45 s resample is counted
separately and now shows up in the accounting).  The wgpu audio tower is the
**default** (`--cpu-enc` forces the host reference back); its transformer was
aligned in session 6 — see the GPU audio encoder section below.

```text
cargo run --release --bin transcribe -- \
  --model D:\Qwen3-ASR\models\Qwen3-ASR-0.6B-hf \
  --wav D:\qwen3-asr-rs\tests\fixtures\15s_en.wav \
  --adapter nvidia --max-new 512 \
  --baseline D:\qwen3-asr-rs\docs\baseline\texts\python-hf_0.6B_15s_en.txt
```

Measured 2026-09-13 (P104-100, `-hf` weights). Audio frontend now matches
Transformers `load_audio` + `Qwen3ASRFeatureExtractor`:

* 48 kHz / 44.1 kHz → 16 kHz via **libsoxr HQ** (`soxr_oneshot` defaults to LQ;
  quality is passed explicitly). Rust 180s_en 44.1 kHz wave is bit-exact vs
  librosa/`soxr_hq`.
* Log-mel STFT runs on the **raw** waveform (torch.stft `center=True`). Do not
  hop-align with trailing zeros first — that extra frame created a whole
  encoder tail chunk (`180s_zh` 2880001 samples → 18001 vs 18000).
* Encoder still packs `feo(tail)` tokens on a partial last chunk (Python pads
  the mel tensor to `2 * n_window` then masks; token count matches).

| fixture | vs Python `-hf` | notes |
|---|---|---|
| `15s_en` | **MATCH** | 48 kHz stereo |
| `30s_zh` | **MATCH** | 16 kHz native |
| `90s_en` | **MATCH** | 48 kHz stereo |
| `90s_ja` | **MATCH** | 16 kHz native |
| `180s_en` | **MATCH** | 44.1 kHz mono; was resample + hop-pad |
| `180s_zh` | **MATCH** | 16 kHz native; was extra STFT frame |

A/B: `--embeds` and `--mel` from Python both MATCH on `180s_zh`. Native wav
load is bit-exact vs librosa on 16 kHz PCM.

1.7B vs Python `-hf` (same frontend). CPU encoder GELU was the
tanh approximation (`cudarc`); Transformers `activation_function: gelu`
and `F.gelu` on the conv stem are **erf**. Switching the CPU encoder to
erf dropped rust-vs-python-fp16 embed RMS 7.1e-5 → 2.7e-5 (same envelope
as Python fp16 vs fp32):

| fixture | vs Python `-hf` | notes |
|---|---|---|
| `15s_en` | **MATCH** | |
| `30s_zh` | **MATCH** | |
| `90s_en` | **MATCH** | was missing final `the` under tanh GELU |
| `90s_ja` | **MATCH** | |
| `180s_en` | **MATCH** | |
| `180s_zh` | **MATCH** | was `。` vs `！` under tanh GELU |

**12/12 MATCH** against Python `-hf` greedy.

---

## Stage 2.5 record — decode transcendentals bit-exact vs CUDA; 8/9 fixtures token-identical, 1 at a proven reference dead-tie

### Bit-exact expf (2026-09-13, session 3)

The decode softmax's `exp()` was the last *controllable* divergence source
between wgpu and CUDA: WGSL's builtin differs from CUDA `expf` by 1–58 ulp on
~92 % of inputs (1 M-sample bit-level probe).  `expf` is now reproduced **bit
for bit** in the decode attention kernels (`shaders.rs`, `EXP_BT` snippet, used
by `gqa_decode_single`, `gqa_decode_split_p1`, `gqa_split_merge`):

* CUDA 12.8's `expf` for sm_61 is an 8-instruction sequence (PTX-probed via
  `wgpu/exp_probe/probe*.cu`): `fma.rn → cvt.sat → fma.rm(f5,252,12582913) →
  add/neg → fma.rn×2 → shl → ex2.approx.ftz → mul`.
* The three correctly-rounded fmas are emulated in **pure u32 integer
  arithmetic** (`fma_int`): integer ops are exact and associative, so no driver
  transform can change them.  This is *required*: the float-domain software fma
  (Dekker two_prod + Knuth two_sum) is algebraically correct — a numpy float32
  step-simulation matches CUDA `fmaf` on 2^20/2^20 inputs — yet the GPU result
  diverges on 22 % of inputs because the NVIDIA Vulkan compiler CSE-folds the
  two_sum residual `(a-ap)+(b-bp)` to zero through `ap = s-b` substitution,
  degrading it to double rounding.  Bitcast round-trip "barriers" do not help
  (they are legal identities any optimizer removes).
* The final `× 2^(Q-126)` is a power-of-two scaling done by direct bit
  assembly + integer GRS rounding in the subnormal grid (`scale_pow2`), because
  the driver compiles OpFMul with output FTZ while CUDA's `mul.rn` keeps
  subnormals.
* WGSL `exp2` == `ex2.approx.ftz` bit-for-bit (measured, normal domain).
* Validation: `fma_int` vs CUDA `fmaf` 0/2^20; full `expf_bt` vs CUDA `expf`
  0/2^20 over the whole finite domain covered by the probe inputs.

Cost: the integer fma costs ~+6 % (single attention path, 8.25→8.78 ms/token)
to ~+18 % (split path, 14.77→17.25 ms/token on `q06_180s_en`).  Untouched
optimization headroom: share the Dekker split of `x` across the three fmas,
fast-path the common non-tie rounding.

### Outcome after the expf port (decode_check, CUDA-golden prefill KV)

| fixture | result | GPU-bound step |
|---|---|---|
| `q06_15s_en` / `30s_zh` / `75s_x` / `90s_en` / `90s_ja` / `180s_zh` | **IDENTICAL** | 8.78 / 9.2 / 10.4 / 12.64 / 12.6 / 17.4 ms |
| `q17_15s_en` / `q17_180s_en` | **IDENTICAL** (54 / 589 steps) | 17.9 / 23.6 ms |
| `q06_180s_en` | diverges at step 363 → 217/581 | 17.25 ms |

`q06_180s_en` flips **at the same step with the same top-4 numbers** as before
the port — the transcendentals are eliminated as a cause.  Hard data at the
flip (step 362, decode_check top-4):

```
tok 29850  ref=30.5312  got=30.5312
tok 27833  ref=30.5312  got=30.5000   ← wgpu 1 f16 ulp lower
```

The reference top-2 logits are **exactly tied at f16** (30.5312 == 30.5312).
The residual divergence driver is GEMM accumulation-order noise: wgpu's GEMV
order differs from cuBLAS's (a versioned black box), leaving ~1-ulp f16 noise
on 7–38 % of logit elements every step (decode_check step-0/parity dumps).
Any two correct implementations with different reduction orders flip such a
tie arbitrarily; cuBLAS does not even guarantee bit-reproducibility across its
own versions.  Eliminating this would require cloning cuBLAS's internal tiling
— out of scope by construction.  The transcript impact is one word at one
spot of a 180 s clip (both candidates are real words: `ĠJimmy` vs `ĠPP`), and
the sequences re-align afterwards.

### Prefill alignment matrix (KV vs CUDA golden, first token, full decode continuation from the wgpu-prefilled KV)

| fixture | s | KV max\|Δ\| | first token | continuation | verdict |
|---|---|---|---|---|---|
| `q06_15s_en` | 210 | 0.40 | MATCH | **52/52** | within tolerance |
| `q06_30s_zh` | 407 | 1.43 | MATCH | **126/126** | accepted (drift) |
| `q06_75s_x` | 990 | 5.05 | MATCH | **200/200** | accepted (drift) |
| `q06_90s_en` | 1185 | 8.09 | MATCH | **316/316** | accepted (drift) |
| `q06_90s_ja` | 1177 | 1.47 | MATCH | **296/296** | accepted (drift) |
| `q06_180s_zh` | 2356 | 0.84 | MATCH | **641/641** | within tolerance |
| `q06_180s_en` | 2307 | 2.41 | MATCH | 364/581, flip @363 | **known reference dead-tie**¹ |
| `q17_15s_en` | 210 | 1.91 | MATCH | **54/54** | accepted (drift) |
| `q17_180s_en` | 2307 | 16.03 | MATCH | **589/589** | accepted (drift) |

¹ Same boundary as stage 1: top-2 logits tie at f16 in the reference; the
identically-shaped `q06_180s_zh` aligns fully, so the prefill is not at fault.

**Prefill timing** (wall, incl. the TDR-guard polls): 15s 167 ms (0.8 ms/pos) ·
90s ≈1.26 s (1.1) · 180s_zh 3.12 s (1.3) · 1.7B/180s 4.99 s (2.2).  The GEMM is
the untuned first version — perf pass comes later (`gemm_bench`).

### KV drift is order noise, calibrated against a second reference pair

Absolute KV diffs vs the CUDA golden grow with chain length (0.4 → 16 across
the matrix) while the decode continuation stays token-identical.  Measured
envelope of **two known-good orders** (torch `ref_prefill.py` chain vs CUDA
golden, same fixture): violations of `0.0625 + |ref|·2⁻⁹` at nearly every deep
layer (L12: 83 elems, L24: 151), max\|Δ\| 0.69 — the same phenomenon.  L0 K
cache: exactly 5 elements above 0.0625, all on \|K\|≈217–386 outlier dims, all
single f16 ulp.  Acceptance is therefore **functional** (first token + full
continuation identical); the KV diff and rel-tol violations print as
diagnostics.  Structural bugs look completely different (see bugs below).

### Bugs fixed during stage 2

1. **AV GEMM prefetch half-selector** (`prefill_gemm(transb=1)`): `store_bs`
   picked the f16 half by `tx` parity (the transb=0 rule) instead of the
   n-element parity used by `load_bs`.  K-tile 0 loads directly (correct), so
   AV output rows 0–15 were exact and rows 16+ garbage — first visible at the
   second K-tile boundary.  Dump bisection (`tools/analyze_dump.py` +
   `tools/verify_hypothesis.py`) pinned it: with the hypothesized swap
   `V[k, (n&~1)\|(k&1)]` for k≥16 the dump matches to 1–2 f16 ulp (0.0001–9e-4)
   where the correct model misses by 0.43–0.60.  With the bug: L0 h error 3.3,
   deep-KV max\|Δ\| 93, first token mismatch, 0/52 continuation.  After: all
   rows clean (AV max\|Δ\| 0.0026).
2. **Deep-async-queue device loss on Pascal/WDDM**: queueing the whole
   multi-second prefill as one submit device-lost the driver (1.7B @ 2307
   positions) — surfaced only as a bare "async map a buffer" on the next
   readback.  Not VRAM (peak 5.1/8.2 GB, sampled via nvidia-smi), not a
   validation error (uncaptured-error hook silent).  Time-budgeted splits
   WITHOUT draining died at 100/500/1000 ms (3–25 submits); per-layer
   **submit + poll** survived repeatedly.  Guard: for s ≥ 512, submit+poll
   every layer (+~20% prefill wall; short prefills keep the single submit).
   Repro/verify: `cargo run ... --bin prefill_check -- --tag q17_180s_en ...`
   (old binary: fails at first readback; new: 589/589 ×3 runs).

### Stage 2 tooling

| file | content |
|---|---|
| `tools/ref_prefill.py` | full-prefill torch reference (all 28 layers, CUDA op semantics) → `golden/<tag>_refp/` |
| `tools/analyze_dump.py` | parses `prefill_l0_debug.bin` (11 length-prefixed chunks), per-tensor/per-row diff vs torch reference |
| `tools/verify_hypothesis.py` | numeric confirmation of the AV half-swap model |
| `tools/check_kv_outlier.py` | outlier-dim vs bug classification for KV diffs |
| `src/bin/prefill_check.rs` | gate reworked: functional acceptance, diagnostics printed |

Dump chunk order (`prefill_l0_debug.bin`): h(s·hs) / normed0 / qkv0 /
attn_flat(mp·nqh·hd) / gu0 / act0 / q_out(nqh·mp·hd) / v_rep(nqh·np·hd) /
k_rep(same) / scores(nqh·mp·np) / attn(nqh·mp·cur16).  Note GQA layout: k_rep
slot ih holds K head ih/2 — to compare K head kh use slot kh·(nqh/nkvh).


## Stage 1 record — decode chain, alignment matrix full, decode near bit-exact ceiling

### Alignment matrix (9 fixtures, all four attention paths)

| fixture | model | cur_len range | attention path | result | GPU-bound step | decode-only RTFx¹ |
|---|---|---|---|---|---|---|
| `q06_15s_en` | 0.6B | 211–262 | single wg-256 | **52/52 IDENTICAL** | 8.25 ms | **35.7×** |
| `q17_15s_en` | 0.6B→1.7B | 211–264 | single wg-256 (sharded) | **54/54 IDENTICAL** | 16.50 ms | 17.2× |
| `q06_30s_zh` | 0.6B | 408–533 | single 256→512 crossover | **126/126 IDENTICAL** | 8.73 ms | 27.5× |
| `q06_75s_x`² | 0.6B | 991–1190 | 513 + 1025 crossings | **200/200 IDENTICAL** | 9.97 ms | 37.8× |
| `q06_90s_en` | 0.6B | 1186–1501 | split chunk-256 | **316/316 IDENTICAL** | 10.59 ms | 27.0× |
| `q06_90s_ja` | 0.6B | 1178–1473 | split chunk-256 (ja) | **296/296 IDENTICAL** | 10.56 ms | 28.9× |
| `q06_180s_en` | 0.6B | 2308–2888 | split chunk-512 | 363 aligned, dead-tie flip³ | 14.77 ms | 21.0× |
| `q06_180s_zh` | 0.6B | 2357–2997 | split chunk-512 (zh) | **641/641 IDENTICAL** | 14.98 ms | 18.8× |
| `q17_180s_en` | 1.7B | 2308–2896 | split chunk-512 | **589/589 IDENTICAL** | 22.22 ms | 13.8× |

¹ decode-only: audio duration ÷ (n_decode × GPU-bound ms/token).  Excludes
prefill (stage 2) and the audio encoder — end-to-end RTFx lands lower.
² `target/tmp_75s.wav` = 15s_en ×5 concatenated; golden regenerable.
³ Flip at step 362 measured to be a **dead tie in the reference** (top-2 logits
both 30.5312 at f16; `decode_check` prints the top-4 at the flip step).  As of
the bit-exact expf port (see "stage 2.5" above) the transcendentals are ruled
out: the flip persists with identical numbers and is driven by GEMM
accumulation-order ulp noise against a tied reference — a measured boundary of
two correct implementations, not a port bug.  The same path aligns fully on
`q06_180s_zh` and `q17_180s_en`.

### Where the remaining 0.80× goes (per-op timestamp profile, `step_profile`)

True in-step cost per decode step (0.6B, cur_len 211, single pass, timestamps):

| stage | ms | share | effective bandwidth |
|---|---|---|---|
| 4 per-layer GEMVs | 4.36 | 54% | 123–209 GB/s (scales with row count) |
| LM head | ~1.1 | 14% | 264 GB/s |
| GQA attention | 1.66 | 21% | latency-bound (16 workgroups — CUDA-identical shape) |
| micro-ops (rms×2, extract, silu, embed) | 0.84 | 10% | ~9.5 µs **fixed** per kernel |

Findings that shaped the plan:

* **【2026-09-13 实测推翻 + 已修复】** 下面这条原结论是**误判**，保留原文以存档。
  正确结论：`enable subgroups;` 被 naga 拒绝是**故意行为**
  （跟踪 issue #5555），而 subgroup 内建由 **`wgpu::Features::SUBGROUP` 能力位**门控。
  **本机实测：`SUBGROUP` / `SUBGROUP_BARRIER` / `SUBGROUP_VERTEX` 全部 YES**
  （P104-100 / Vulkan / 572.75），不写 `enable` 指令的 subgroup WGSL 编译并正确运行。

  **已按此修复 `shaders::gemv`**：32-lane xor butterfly 从
  shared memory + 5×`workgroupBarrier()` 改为 `subgroupShuffleXor`，
  xor 顺序不变 ⇒ **归约树逐位不变**。A/B 实测 1.17–1.18×、18992 行输出
  **0 个不同**（`cargo run --release --bin subgroup_bfly_bench`）。
  端到端 0.6B `180s_en` decode **10019 → 9630 ms**（RTFx 9.88 → 10.50），
  `1.7B 180s_en` decode **15256 → 14851 ms**；**六项 fixtures + 1.7B 两项全部 MATCH**。

  注意：`gqa_decode_single` / `gqa_decode_split_p1` / `gqa_split_merge` 三处归约是
  `if (lid.x < sh) { red[i] op red[i+sh] }` 形状（跨多 warp、宽度 BS），
  **换 subgroup 会改变归约树、破坏 bit-exactness，故未改**。
  见 `docs/wgpu-best-practices-audit.md` §H。

  ~~wgpu 30's naga WGSL front **rejects `enable subgroups;`** (Unimplemented) —
  warp-shuffle butterflies are unavailable; the shared-memory butterfly stays.~~
* The per-kernel fixed cost on this stack is ~9.5 µs (ablation-verified:
  timestamp-adjacent marks cost 3.6 µs; isolated small-kernel numbers were
  polluted by per-submit cost and must not be trusted).
* Fusions that would remove launches are **blocked by bit-exactness**: silu→down
  re-runs 3.2 G exp2 (one per row); rms→gemv changes the 1024-lane reduction
  tree (last-ulp inv_rms drift).  The micro-kernel count is therefore frozen.
* `gemv` tile loop unrolled ×4 with prefetched loads (bit-safe: ascending fma
  order preserved): +2% end-to-end.  lm_head runs at 264 GB/s ≈ CUDA parity;
  small projections are wave-count-limited (rows == warps), which is not
  fixable without changing CUDA's one-warp-per-row reduction order.

**Decode floor at bit-exactness ≈ 8.0 ms/token vs CUDA 6.58 (0.82×).**  Official
best-practice audit (wgpu wiki Do's-and-Dont's, docs.rs): single submit per
step ✓, one compute pass ✓, static bind groups ✓, no per-frame resources ✓
(token readback uses a persistent staging buffer), MemoryHints::Performance is
already the default ✓.

## CUDA vs wgpu 分段实测（2026-09-13, session 4）— 差距最大的是音频编码器

同一台机器（P104-100）、同一 fixture（`180s_en`）、0.6B、`-hf` 权重、`max_new 1024`。

CUDA 侧复现命令（**没有改 CUDA 后端**，example 只装了 env_logger 让既有的
`info!` 计时真正输出）：

```text
cd D:\qwen3-asr-rs
RUST_LOG=info cargo run --release --example phase_timing_cuda -- 180s_en 1024
```

| 相位 | CUDA 手写 | wgpu | 绝对差 | 倍数 |
|---|---|---|---|---|
| mel | 36 ms | 30 ms | −6 ms | 1.2×（wgpu 更快） |
| **audio enc** | **803 ms** | **2730 ms** | **1.93 s** | **3.4×** |
| prefill | 1414 ms | 2930 ms | 1.52 s | 2.07× |
| decode | 7316 ms (580 tok × 12.61 ms) | ~10000 ms (585 tok) | ~2.7 s | ~1.39× |
| 合计 | **9.83 s / 18.3×** | **17.9 s / 9.9×** | | |

CUDA 的 9.83 s 与冻结基线 `cuda_0.6B_180s_en.txt`（9.873 s / 18.231×）在 0.5% 内
吻合，说明这份分段是可信的。CUDA decode 12.61 ms/token 也解释了为什么它整条
180s 只要 9.9 s。

**结论修正**：此前把优先级排成"attention → prefill → 编码器"是错的。按实测，
编码器差距最大（3.4×）且绝对差最大（1.93 s），decode 差距最小（1.39×）。

顺带记录（非结论，只是观察）：CUDA `cuda_audio_enc` 180s 用 0.8 s 做完约 100 GFLOP，
有效吞吐只有 ~1.5 GFLOP/s 量级 —— 说明**编码器是权重带宽受限的**（整塔 f16 权重
几百 MB，每个 pass 至少读一遍），不是算力受限。这解释了为什么它"只有" 0.8 s 而不是几十毫秒，
也说明想追上它必须同时优化 cuBLAS 等价物**和**访存，不是换个 tile 就够。

## GPU audio encoder 调查记录（2026-09-13 session 4 → 2026-09-14 session 5）

`src/audio_encoder_gpu.rs` + `shaders::audio_im2col/audio_bias_gelu/audio_layernorm/
audio_extract_qkv/audio_win_pack/audio_window_softmax/audio_permute_pe/audio_add_pe/audio_attn_flat`。
`transcribe` **默认就用它**（`--cpu-enc` 强制回到 CPU 参考；`--gpu-enc` 仍被接受、已是默认），
`--compare-enc` 打 embedding 包络，`--diag-enc` 做逐级比对。加载失败/几何不符会打印原因并回退，
不会静默出错。

### session 5 结论（一句话，已被 session 6 取代）

**conv stem 修好了，并且是逐值验证的；transformer/attention 还没对齐。**
`--gpu-enc` 音频塔 15s 从 **278 ms（CPU）→ 63 ms**，但 transformer 未对齐 ⇒ 暂时不能用。

### session 6 结论（一句话）

**transformer 对齐了，GPU 塔成为默认。** 12/12 MATCH（0.6B/1.7B × 6 fixture，
与 python-hf 基线逐字相同），180s_en enc **2720 → 1019 ms**，端到端 RTFx **10.1 → 14.9**。
挡住 transformer 的是四个 bug（前两个是"看起来自洽"的陷阱）：

* **⑳ `enc.ex` uniform 从未写入**。`u_ex` 只创建、绑定，没有 `write_buffer` ⇒
  `n_tokens = 0` ⇒ `audio_extract_qkv` 第一句 guard 直接 return ⇒ Q/K/V 全 0。
  宿主 oracle 拿 GPU 自己的 `q/k/v` 重算，`0·0 = 0`、`softmax(0)` 也一致，
  于是 `scores/probs/av out` **三行全绿** —— 一个没有任何信号的"自洽"。
  现在 oracle 会打印 `max|q|/|k|/|v|`。
* **㉑ attention GEMM 的 `bsc` 用错了单位**。`prefill_gemm` 的 epilogue 是
  `C[(cbase + …) / 2]`，所以 `bsc` 必须是**元素**（`decoder.rs` 一直这么传），
  而音频塔传的是 `wpad*wpad/2`（字）。batch=1 时 `cbase = 0`，看不出来；
  28 个 block 时除 block 0 外每个块都写到**前半个块**的位置 ⇒ 相邻块互相覆盖，
  谁赢看 workgroup 调度 ⇒ **同一份输入两次运行结果不同**（这类 bug 靠"重跑一次"抓）。
* **㉒ `audio_win_pack` 的 `qp` 越行写入**：`jw ≥ hd2` 的线程地址落到下一 token 的行，
  和那一行自己的写**竞争**（真实值 vs 0）。vp 没这个问题（它的行宽含 padding）。
* **㉓ 诊断侧 `conv_block_stages` 的 epilogue 忘了 `b0`**：`want_raw` 时读的是
  `ib·plane` 而不是 `(b0+ib)·plane` ⇒ 第 2 个 tile 起 act 全是第 1 个 tile 的复制品。
  单 tile（单 chunk）时重合，所以上一窗口的"单 chunk 复现"全绿，而 15s 的
  `conv{1,2,3} +bias/GELU` 三行全红 —— **诊断的错让 GPU 背了两轮锅**。
* **㉔ 诊断把 `attn_flat` 和 `dbg_attn` 比**（前者是 out_proj 之前，后者之后）。
  新增 `CpuAudioAttention::flat()` / `CpuAudioEncoder::dbg_attn_flat()` 后同口径。

修好后 15s_en 的诊断（`--diag-enc`）全部干净：conv 三级 act 0 坏值，
四个 `(head, win)` 块的 operand/scores/probs/av 全 0 bad，
`attn residual` 0/174720，`layer 0` 1 个点，`layer 17` 917 个点（f16 累积，预期内）。

### session 4 里那份结论是错的（已作废）

session 4 的「im2col 的输出全是 0 / 写入路径失效」**是诊断自己读越界造成的**：
诊断用 `pos1p * c1.n_pad` 算读长度、缓冲却按 `c1.n` 分配，wgpu 报
`Copy ... would end up overrunning the bounds of the Source buffer`，
`copy_buffer_to_buffer` 整条被拒 ⇒ staging 缓冲保持全 0 ⇒ 读回来全是 0。
`mel[0..8]` 与 `taps[0..16]` 之所以"对"，是因为它们读的是别的缓冲。

当时还有一条**被掩盖的真 bug**：`conv_taps` 被传了输出维度当输入维度，
于是三级 tap 表整体错一位（pos = 800/208/56 而非 3200/800/208）——
"每个位置都错一点点"正是它的表现。

### 现在的验证方法（取代 hand-check 地址）

`--gpu-enc --diag-enc` 让 **CPU 与 GPU 吃同一份输入**：第 1 级喂 f16 量化后的 mel，
第 2/3 级喂 **GPU 自己写出的激活**。这样每一级都是紧比对（只差 GEMM 的 f16 舍入），
而不用在链条上累积舍入误差。三级各比三样：im2col operand、GEMM raw（pre-GELU）、+bias/GELU。

诊断还内置两个主机 oracle：

* **attention oracle**：拿 GPU 自己写出的 `q/k/v`，在主机上按定义重算 `q·k`、softmax、`p·v`，
  与 GPU 的 `scores/probs/attn_out` 比 —— 于是 `win_pack` / 两个 attention GEMM / softmax
  的分块与 stride 可以**独立于 CPU 参考**被判对错。
* **逐层比对**：CPU 用 GPU 的 `h` 跑同一层（`CpuAudioEncoder::layer_forward`），
  所以 layer N 的差异只可能来自 layer N。

### 实测（0.6B，15s_en，15 chunks / 2 rounds）

```text
  conv1 operand        max|d| 0.000000  bad 0/28800          # 逐位一致
  conv1 GEMM           max|d| 0.000976  bad 0/1536000
  conv1 +bias/GELU     max|d| 0.000489  bad 0/1536000
  conv2 operand        max|d| 0.000000  bad 0/3456000
  conv2 GEMM           max|d| 0.003482  bad 24/5760000
  conv2 +bias/GELU     max|d| 0.002727  bad 11/5760000
  conv3 operand        max|d| 0.000000  bad 0/898560
  conv3 GEMM           max|d| 0.001710  bad 0/99840
  conv3 +bias/GELU     max|d| 0.001862  bad 0/99840
  packed [tok][c·f]    max|d| 0.006039  bad 4/1497600
  h [tok][d_model]     max|d| 0.003509  bad 0/174720
  -- layer 0 stages --
  LN1 (normed)         max|d| 0.000968  bad 0/174720        # 对齐
  fused QKV            max|d| 0.002124  bad 0/524160        # 对齐（含 bias）
  attn residual        max|d| 2.379767  rms 0.290804  bad 11419/11648   # ✗
  layer 0              max|d| 1.023661  rms 0.151112  bad 169456/174720 # ✗
  -- attention oracle (head 0, window 0, tokens 0..8) --
  scores               max|d| 0.000000  bad 0/832           # GPU 内部自洽
  probs                max|d| 0.000002  bad 0/832
  av out               max|d| 0.000000  bad 0/512
```

（conv stem 的残差是 f16 激活的固有舍入；`attn residual` 那几行才是真错。）

### 布局契约（单一来源）

`audio_encoder_gpu.rs::ConvLevel` 是唯一的几何来源，`LevelView`（`enc.level(i)`）
把它发布给诊断 ⇒ 比较代码与 kernel **不可能各说各话**。`CONV_TILE = 8` chunk 一轮。

```text
operand（im2col 输出 = transb GEMM 的 B）   [k_pad][n_all] f16，字 (k,位置对) 在 k*(n_all/2) + col/2
   读 (chunk0+chunk)*in_chunk + ic*in_ic + tap；k/chunk/p 越界一律显式写 0
activation（GEMM 的 C）                     [c][pos] 通道优先 = CPU 参考布局
   元素 (c, chunk, p) 在 c*n_all + chunk*plane_pad + p；ldc = n_all
mel 输入                                    [chunk][mel_bin][frame]，行距 w0 = pad_even(cs)
```

配套的三条"结构性"规则（都是本窗口踩出来的）：

* **哨兵只有一个**：`shaders::TAP_OOB`，由 Rust 注入 WGSL（`audio_im2col()` 里 `format!`）。
  两边曾经不一致（Rust `0xFFFF_FFFF` / WGSL `0xFFFF`）⇒ **每个越界 tap 都被当成合法地址**。
* **k 按 `GEMM_BK`(=16) 对齐**，不只是 4：k 循环走整 16 宽 tile，`k=12` 时最后 4 个 k
  读到 operand 行外并把乘积累加进输出（`pad_k_tile()`）。
* **bias 向量按元素打包**，每个形状一块独立的 `ScaleCfg`（`u_sc[5]`），
  每个 round 单独 submit —— 因为 `queue.write_buffer` 在下一次 submit 才生效。

### session 5 修掉的真 bug（19 个）

conv stem：① `conv_taps` 传错维度；② OOB 哨兵不一致；③ k 未按 BK 对齐；
④ 一块缓冲三种布局（operand/raw/act 现在各一块，尺寸全部由 `ConvLevel` 推）；
⑤ `bias_gelu` 覆盖数 `n/600`（15% 元素没处理）；⑥ conv3 的 bias+GELU 整段缺失；
⑦ bias 按"字"下标索引（通道翻倍）；⑧ erf 常数 `0.147`（应为 `0.3275911`）；
⑨ `u_sc` 共享 uniform；⑩ 多 chunk 时 im2col 每块写同一区域（无 chunk stride）。

permute/投影：⑪ 补零分支用 tile 内行号（第 2 个 round 清掉 `packed` 91..151 行）；
⑫ PE 加在 `[c·f]` operand 上（参考是加在 conv_out 输出）且 extract 又加一次；
⑬ `packed` 行距写成 `s_pad` 而非 `conv_out.k`；⑭ permute/add_pe 行覆盖不足。

transformer：⑮ 两个 attention GEMM 的 grid 整除得 0 ⇒ **从来没跑过**；
⑯ 分数块 stride 混用（每 head 的 `s_pad` vs 每窗口的 `wlen`）；
⑰ `out_proj`/`fc2` 用普通 GEMM ⇒ **残差被丢弃**，且 `qkv/o/fc2/proj2` 的 bias **从未加过**。

诊断自身：⑱ 读长度用 `n_pad` 而分配用 `n`（见上）；⑲ 比对时把 `raw` 的行距用在 `act` 上
（`raw` 是 `[c][pos]`、`act` 是 `[chunk][c][pos]`，单 chunk 恰好重合）。

### session 5 在 transformer 上改对的地方（保留，别再退回去）

* 新增 `audio_win_pack`：把 `[tok][nh·hd]` 的 Q/K/V 重排成 per-`(head,window)` 分块，
  使 `prefill_gemm` 固定的 operand 行距（A 行距 `k/2`、transb B 行距 `n/2`）能对上；
  块布局 `qp[z][row][hd]`、`kt[z][j][row]`、`vp[z][row][128]`，`z = head·n_win + win`。
* 分数块改为**稠密** `[z][wpad][wpad]`；两个 attention GEMM 的 grid 改成整 tile。
* softmax 增加短窗口 mask（`valid = min(wlen, s - win·wlen)`）；`audio_attn_flat` 按新块布局重写。
* PE 只在 conv_out 输出上加一次；`extract` 不再加 PE。
* `out_proj`/`fc2` 用 `prefill_gemm(false, true)`（beta 残差）；
  bias 通过新的 `prefill_gemm_bias()` 在 GEMM epilogue 加 —— 它是独立入口而不是
  `prefill_gemm` 的一个 flag，因为 binding 必须**只在使用它的变体里声明**
  （声明了但没绑定会让 pipeline 验证失败，即使读它的代码是死代码）。
* `audio_add_pe`/permute 的行覆盖修好。

### 下一步（未完成，交接给下一个 AI）

注意力输出相对 CPU 仍错（`attn residual` rms 0.29，layer 0 rms 0.15 并逐层放大）。
已知 LN1 ✓、fused QKV ✓、GPU attention **内部自洽** ✓ ⇒ 差值只可能来自 oracle 没覆盖的地方：

1. **head ≥ 1 的 head 切片**（oracle 只测了 head 0，那里偏移恰好是 0）：
   查 `audio_extract_qkv` 的 `col = head*hd2 + w` 与 `audio_win_pack` 的 `head*hd2 + jw`。
2. **最后一个窗口的 mask**（195 token、`wlen=104` ⇒ `n_win=2`，末窗 `valid=91`）。
3. `audio_attn_flat` 的 head/row 映射（`head = w/hd2`、`z = head*n_win + win`）。
4. `o_proj`/`fc2` 的 beta 残差是否真的生效（本窗口才加的）。

**建议第一步**：把主机 oracle 从 head 0/window 0 扩到 **head 1** 和**最后一个窗口**，
一次就能区分「head 切片错」还是「mask 错」。

## Design notes

**Bounded-staging bulk upload** (`gpu.rs::BulkUpload`, budget 256 MiB): wgpu
defers `write_buffer` to the next submit and holds staging until then; a
multi-GiB load staged into one submit overflows VRAM on WDDM drivers (copies
land as zeros, device lost).  The CUDA engine never holds staging (synchronous
htod) — `BulkUpload` is the wgpu counterpart: stage ≤ budget, submit+wait,
continue.  All load-time transfers route through it.

**Explicit shared pipeline layouts** for sibling kernel families (single-block
wg 256/512, split chunk 256/512): wgpu's implicit layouts are pipeline-exclusive,
so shared bind groups need one explicit `PipelineLayout` per family.

**Split-K attention** (`gqa_decode_split_p1` + `gqa_split_merge`): structural
port of CUDA's two-phase kernel (chunk scores → block-tree max/sum → f32 partial
numerators → online-softmax merge with `-inf` empty-chunk guard).  Chunk 256
below cur_len 2048, else 512; scratch sized `ceil(max_seq/256)` as CUDA does.

## File map

| file | content |
|---|---|
| `src/gpu.rs` | adapter pick, allocs, readback, pipelines (error scopes, explicit layouts, timestamp features), `flush`, `BulkUpload` |
| `src/weights.rs` | safetensors mmap (single + sharded), bf16/f32→f16 bit-matching CUDA, fused QKV / gate_up |
| `src/shaders.rs` | rms_norm, gemv (×4 unrolled), qkv_extract, gqa single 256/512, split-K P1 256/512 + merge, silu, argmax, embed |
| `src/decoder.rs` | step assembly, one encoder/submit per step, split dispatch, bench + debug hooks |
| `src/mrope.rs` | MRoPE tables (f64→f32, interleaved/blocked) |
| `src/golden.rs` | golden loader + diff (tolerates pre-`last_token` goldens) |
| `src/bin/decode_check.rs` | alignment + benchmark harness, near-tie diagnostic |
| `src/bin/prefill_check.rs` | prefill validation harness (KV diff, logits, continuation gate) |
| `src/bin/step_profile.rs` | per-op timestamp profiler (single pass, `--skip-micro` ablation) |
| `src/bin/gemv_bench.rs` | isolated GEMV A/B (methodology mirrors the CUDA bench) |
| `src/bin/probe_layers.rs` | per-layer/per-op dump probe |
| `src/bin/buffer_probe.rs` | big-buffer / big-SSBO driver diagnostic |
| `tools/ref_decode_step.py` | PyTorch CPU reference of one decode step |
| `tools/compare_probe.py` | probe-vs-reference table |

## Reproducing

```bash
# goldens (CUDA backend, repo root; deterministic)
QASR_TAG=<tag> QASR_WAV=<wav> [QASR_MODEL=models/Qwen3-ASR-1.7B] \
  cargo test --release --lib dump_decode_golden -- --ignored --nocapture
# q06_75s_x uses the concatenated wav at target/tmp_75s.wav (15s_en x5)

# wgpu alignment + benchmark (repo root)
cargo run --release --manifest-path wgpu/Cargo.toml --bin decode_check -- \
  --tag <tag> --golden wgpu/golden/<tag> --adapter nvidia

# per-op GPU profile
cargo run --release --manifest-path wgpu/Cargo.toml --bin step_profile -- \
  --tag q06_15s_en --golden wgpu/golden/q06_15s_en --adapter nvidia

# prefill validation (KV + first token + full decode continuation)
cargo run --release --manifest-path wgpu/Cargo.toml --bin prefill_check -- \
  --tag <tag> --golden wgpu/golden/<tag> --adapter nvidia

# torch reference for the full prefill chain (several minutes)
python wgpu/tools/ref_prefill.py --tag q06_15s_en
```

## Bugs fixed during bring-up

1. `qkv_extract` word-vs-element indexing (Q-head base, cos/sin row base).
2. `format_f32` mantissa shift (23-bit mantissa in 24 bits): LOG2_E 1.4427 →
   1.2213 bent every SiLU; fixed by emitting `man << 1`.
3. Bulk deferred uploads never land at 3.4 GiB on Pascal/WDDM → `BulkUpload`.
4. wgpu implicit layouts are pipeline-exclusive → explicit family layouts.
5. WGSL reserved word `meta`, const-context `bitcast` (caught by error scopes).

## Next: RTFx (see HANDOFF.md)

Measured 2026-09-13 (session 4), 0.6B, default CPU audio tower + wgpu decoder:

| clip | mel | enc | prefill | decode | total | RTFx |
|---|---|---|---|---|---|---|
| 15s_en | 3 ms | 288 ms | 236 ms | 447 ms | 1.33 s | **11.3×** |
| 180s_en | 30 ms | 2.73 s | 2.91 s | 8.88 s (512 tok, truncated) | 16.1 s | **10.9×** |
| 180s_en (full) | 30 ms | 3.0 s | 2.96 s | 10.0 s (585 tok) | 17.9 s | **9.9×** |

1.7B: 15s_en `411 + 605 + 906 ms` = 2.64 s / 5.69×; 180s_en `3575 + 5997 + 15256 ms`
= 27.4 s / 6.44×.

Where the time goes, and what is actually left (**session 6 RTFx pass**: the
audio tower moved to the GPU and the decode attention got vectorised loads):

0. **audio frontend** — the 44.1 kHz → 16 kHz **soxr HQ resample is ~1.45 s** for a
   3-minute clip, the single largest host cost, and it sits *outside* the phase
   timers (so the phases used to sum to ~2 s less than `elapsed`).  It cannot be
   parallelised through soxr's OpenMP build (that path only splits *channels* and
   this is mono) and the recipe is fixed by the bit-exactness of the reference.
   The vendored build *was* unoptimised (`/O2` dropped by the `cmake` crate) —
   fixed, bit-identical output.
1. **decode** — 6.02 s against CUDA's 7.32 s: **18 % faster than the CUDA
   handwritten kernel** (7.05 s before this pass).  Per step
   (180 s clip, ~9.4 ms): attention p1
   4.2 ms (instruction-bound: ~350 instructions per key for the scores and ~6 per
   (key, dim) for the AV scan; f32 scalar because Pascal's f16 is 1/64 rate),
   mlp 2.2 ms, qkv 1.3 ms, `extract` 1.0 ms, `merge` 0.9 ms, LM head 0.7 ms
   (288 GB/s ✓).  Two cuts landed: the vectorised K/V loads in `gqa_decode_*`
   (−1.8 ms/step) and folding the RMSNorms into the GEMVs that consume them
   (`shaders::gemv_norm`, −0.4 ms/step, bit-identical virtual-thread tree).
   The next candidates are the AV scan's instruction count and the
   `extract`/`merge` launch bubbles.
2. **prefill** — 1.79 s vs CUDA 1.41 s.  `gemm_bench` already swept the tile
   space (~2 TFLOP/s ceiling there, ~2.07 measured), so this needs a different
   blocking strategy, not a tile tweak.
3. **audio encoder** — 1.05 s vs CUDA 803 ms (**1.31×**).  It used to be 2730 ms
   on the CPU (3.4×); session 6 moved it to the wgpu tower and aligned it.
   Whatever is left is 18 layers of 128×128 GEMMs (the conv stem is ~63 ms of it).

CUDA's own `cuda_audio_enc` does ~100 GFLOP in 0.8 s ≈ 1.5 GFLOP/s effective —
the tower is weight-bandwidth bound (a few hundred MB of f16 weights per pass),
not compute bound, so matching it needs better memory behaviour as much as a
better GEMM.

The GPU audio tower (now the default) lives in `src/audio_encoder_gpu.rs`; its bug
list is in the "GPU audio encoder 调查记录" section above.  There is no open
correctness symptom left: all 12 fixture/model pairs are MATCH on it.


## In progress: bit-exact expf for the decode chain (kills the dead-tie flip)

q06_180s_en flips at step 363 because the reference itself has a dead tie
(top-2 logits equal at f16) and the decode chain is *not* fully bit-exact:
individual f16 rounding-boundary points flip randomly (≈1e-5 probability each,
~4M points per transcription → dozens of 1-ulp flips → autoregressive
amplification).  Root cause isolated to the attention-softmax `exp`:

* CUDA `expf` on sm_61 (probed via `wgpu/exp_probe/probe.cu` → PTX) is an
  8-instruction sequence: `fma.rn(x,c1,0.5)`, `cvt.sat`,
  **`fma.rm(f5,252.0,12582913.0)`** (constant is 252, not 251!), `add`,
  `neg`, `fma.rn`×2, `shl(bits(f8),23)`, `ex2.approx.ftz`, `mul`.
  Manually expanding the sequence reproduces `expf` bit-exactly (nvcc CSEs
  the expansion with its own `expf` — compiler-level identity proof).
* WGSL `exp()` (OpExp) differs from CUDA `expf` by **1–58 ulp on 92% of
  inputs** (probe: 1M samples) — the drift source, absorbed by f16 rounding
  except at rounding boundaries.
* Proven properties on the WGSL side (1M-sample bit-level probe,
  `wgpu/src/bin/exp_probe.rs` + `examples/exp_probe.rs`):
  - driver **does not reliably contract** `x*y+z` into FFMA: buffer-value
    triples contract 100%, but `x*bitcast_const + var` double-rounds
    (±1-2 ulp vs `math.fma`) and uniform-borne constants double-round too —
    so a software fma is required;
  - naga **miscompiles u32→f32 value conversion** (`f32(u32)` yields 0) —
    worked around via direct bit construction;
  - the `fma.rm` round-down is exactly `12582913 + floor(f5×252)`, and
    `floor(f5×252)` is computable in pure u32 (`M×252 = (M<<8)-(M<<2)`,
    `Q = P >> (23-e)`) — verified 0 mismatches;
  - `ex2.approx.ftz` ≡ WGSL `exp2()` on this stack for non-denormal outputs.
* Remaining: software fma (Dekker two_prod + Knuth two_sum + two-level
  rounding correction — exact, no hardware-FMA dependency) for `f12`/`f14`,
  then re-probe to 0 mismatches, swap into `gqa_decode*`/softmax kernels and
  re-run the full decode matrix expecting 9/9 identical (incl. q06_180s_en).

The prefill-side flip on q06_180s_en cannot be eliminated (prefill KV is
f16-tolerance vs the cuBLAS black box by design); the decode-side one can.
