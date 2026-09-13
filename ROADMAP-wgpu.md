# ROADMAP — wgpu backend

Decode-chain and prefill port record.  All numbers are **measured** on the
reference machine (Windows 11, NVIDIA P104-100 Pascal 8 GB, Vulkan, wgpu 30.0.1,
driver 572.75); repro commands regenerate every figure.

---

## Status: stage 3 — **12/12 MATCH** python-hf; RTFx vs CUDA 手写

Live handoff (paths, git, next RTFx cut): **`HANDOFF.md`**. This file keeps
measured stage records. Text decoder is on wgpu; audio encoder is still CPU.

`WgpuAsr::load` uploads the text decoder once (`max_seq=4096`). Inference
RTFx on 0.6B `15s_en` is ~11× vs CUDA handwritten ~31×; `180s_en` ~10× vs ~18×
(decode ~9.7s + CPU encode ~3.4s + prefill ~3s).

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

* wgpu 30's naga WGSL front **rejects `enable subgroups;`** (Unimplemented) —
  warp-shuffle butterflies are unavailable; the shared-memory butterfly stays.
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

- GPU audio encoder (180s CPU encode ~3.4s while the NVIDIA GPU is idle).
- Decode split-K on long context (~16.6 ms/token).
- Prefill GEMM tile (`gemm_bench` already swept; production kernel is still v1).

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
