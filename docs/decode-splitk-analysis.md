# Decode per-token latency for long contexts — where the ~16.6 ms goes, and what split-K can/cannot do

Scope: analysis only; no code changed, no GPU workload run. Identifiers/line numbers refer to
`D:\qwen3-asr-wgpu` at the current working tree; measured inputs come from `ROADMAP-wgpu.md` /
`HANDOFF.md` (P104-100, Vulkan, wgpu 30.0.1, 0.6B, `-hf` weights) and `FEASIBILITY.md` for
bandwidth ceilings.

**Headline.** The 4 per-layer GEMVs and the LM head are provably **independent of `cur_len`** —
they are GEMV over one token's activation with a fixed grid. The 8.3 → 16.6 ms growth is
therefore *entirely* attention, which is already split-K'd and is the single biggest remaining
cost. Bit-exactness **forbids a K-split of `shaders.rs::gemv`** (proof in §4) but does not forbid
touching attention, which is where the money is.

---

## 1. Exact per-step structure of `WgpuTextDecoder::step` (`src/decoder.rs`)

`step()` (L844-853): write the per-step uniforms, build one encoder, encode every dispatch, submit
once, read the token back, advance `pos`.

```rust
pub fn step(&mut self) -> Result<i32> {
    let pos = self.pos;
    self.write_step_uniforms(pos);                          // 5 queue.write_buffer
    let mut enc = self.gpu.device.create_command_encoder(&Default::default());
    self.encode_step(&mut enc, pos);                        // all dispatches
    self.gpu.queue.submit([enc.finish()]);                  // ONE submit
    let v = self.read_token()?;                             // D2H + poll
    self.pos += 1; Ok(v)
}
```

`encode_step` (L755-818) opens **one** compute pass (`enc.begin_compute_pass`, L761) and issues
every dispatch into it; the only non-dispatch work is one 4-byte `copy_buffer_to_buffer` (L817)
after `drop(cp)`. So: **1 encoder + 1 compute pass + 1 submit + 1 D2H copy per step**, matching
`FEASIBILITY.md` §4 (batched = 0.62 µs/dispatch vs 31 µs when submitted per-op).
`dispatches_per_step` (L673-676) = `1 + 28*per_layer + 3` = **256** above `cur_len 1024`, 255
below. Per-step dispatch list, in issue order (grid = `dispatch_workgroups` x/y/z,
`gemv_grid = rows/8` at L758):

| # | kernel (pipeline / WGSL entry) | grid | buffers | `cur_len`-dependent? |
|---|---|---|---|---|
| 1 | `pipes.embed` / `embed_lookup_single` | `(1,1,1)` @1024 | `embed_table`, `token` → `scratch.h` | no (2 KB copy) |
| *per layer ×28* | | | | |
| 2 | `pipes.rms_norm` / `rms_norm` | `(1,1,1)` @1024 | `h`,`iln_w`→`norm1` | no |
| 3 | `pipes.gemv_qkv` / `gemv` | `(512,1,1)` @256 | `qkv_w`,`norm1`→`qkv` (4096) | **no** |
| 4 | `pipes.extract` / `qkv_extract` | `(1,16,1)` @128 | `qkv`→`q_out`, `k_cache`,`v_cache` | no |
| 5 | attention (`encode_gqa`, L823-840) | see below | `q_out`,`k/v_cache`→`attn_out` | **YES** |
| 6 | `pipes.gemv_o` / `gemv` (ACCUM) | `(128,1,1)` @256 | `o_w`,`attn_out`→`h` | **no** |
| 7 | `pipes.rms_norm` / `rms_norm` | `(1,1,1)` | `h`,`pln_w`→`norm2` | no |
| 8 | `pipes.gemv_gu` / `gemv` | `(768,1,1)` @256 | `gu_w`,`norm2`→`gate_up` (6144) | **no** |
| 9 | `pipes.silu` / `silu_mul_split` | `(6,1,1)` @256 | `gate_up`→`activated` | no |
| 10 | `pipes.gemv_dp` / `gemv` (ACCUM) | `(128,1,1)` @256 | `dp_w`,`activated`→`h` | **no** |
| *tail* | | | | |
| 11-13 | `rms_norm` → `gemv_lm` → `argmax_into_slot` | `(1,1,1)`, `(18992,1,1)` @256, `(1,1,1)` @1024 | `h`,`norm_buf`→`final_norm`; `embed_table`→`logits`; `logits`→`token` | **no** |
| 14 | `copy_buffer_to_buffer` | — | `token`→`token_staging`, 4 B | no |

`gemv_grid` is `rows/8` (L758). Attention (row 5) is the only branch:
```rust
fn encode_gqa(&self, cp: &mut wgpu::ComputePass<'pass>, l: &Layer, cur_len: usize) {
    if cur_len <= GQA_SINGLE_CAP {                       // GQA_SINGLE_CAP = 1024 (L286)
        let gqa = if cur_len > 512 { &self.pipes.gqa512 } else { &self.pipes.gqa256 };
        cp.dispatch_workgroups(self.cfg.num_attention_heads as u32, 1, 1);   // (16,1,1)
        return;
    }
    let chunk = gqa_split_chunk(cur_len) as u32;         // 512 if cur_len >= 2048 else 256 (L291-297)
    let n_chunks = (cur_len as u32).div_ceil(chunk);
    cp.dispatch_workgroups(self.cfg.num_attention_heads as u32, n_chunks, 1); // (16, 5, 1)
    cp.dispatch_workgroups(self.cfg.num_attention_heads as u32, 1, 1);        // merge
}
```

### Why greedy cannot pipeline step i+1 behind step i — verified

The `HANDOFF.md` L78 claim is *structurally* true and is a single-buffer dependency, not a
queue-depth problem. `bg_embed` (built L638-647) takes `scratch.token` as its only data input —
binding 1 — and `argmax` writes exactly that slot (`Out[cfg.slot] = sidx[0]`, `shaders.rs` L974,
`slot: 0` from `ArgmaxCfg` at `decoder.rs` L475). Step *i+1*'s first dispatch therefore consumes
step *i*'s last dispatch's output. Step *i+1*'s `rms_norm`/`gemv_qkv` additionally overwrite
`h`, `norm1`, `qkv`, `q_out`, `attn_out`, `gate_up`, `activated`, `final_norm`, `logits` — the
same buffers step *i* is reading. Two steps can never be in flight: correctness would require
double-buffering all of those scratch buffers *and* moving the KV write cursor
(`QkvxCfg.start`, written in `write_step_uniforms`, L700-740) off the shared `u_qkvx` uniform, and
`step()`'s D2H readback (`read_token`, L966-984, `poll(wait_indefinitely)`) forces a host
round-trip anyway. The only place the pipelining is already exploited is `bench_steps`
(L997-1026), which keeps the token on the GPU across steps and skips the readback.

---

## 2. Where the ~16.6 ms at `cur_len ≈ 2300` goes

`src/bin/step_profile.rs` is the right instrument (read, not run): **one** compute pass with a
`TIMESTAMP_QUERY` mark after every op group, deltas aggregated per op name (L121-200). Marks sit
inside the pass so pass boundaries do not pollute the numbers (L123-125); `names[i+1]` charges
each delta to the op before it (L189-194), and `--skip-micro` (L135-141) emits a zero-width mark
to keep that alignment. It runs **one** step at `pos = golden.seq_len` and has only ever been used
on the 15s tag — extending it to `q06_180s_en` is step 0 of any optimization work.

The stage-1 profile at `cur_len 211` (`ROADMAP-wgpu.md` L235-240) plus the 0.6B shapes:

| stage | kernel (`shaders.rs`) | workgroups | ms @211 | scales with `cur_len`? | evidence |
|---|---|---|---|---|---|
| 4 per-layer GEMVs | `gemv` ×4/layer | 512+128+768+128 = **1536**/layer, 43008/step | 4.36 | **NO** | grid is `rows/8`; kernel binds only `Wt`,`X`,`Y`, and `X` is `norm1` (1024) or `activated` (3072) — one *token's* vector |
| LM head | `gemv` (accum=false) | 18992 | ~1.1 | **NO** | same kernel, `X = final_norm` (1024) |
| GQA attention | `gqa*` / `gqa_split_p1`+`gqa_split_merge` | 16 (single) or 16×`n_chunks` = 80 (split) | 1.66 | **YES** | the only loops bounded by `cfg.cur_len` (L636, 650, 664, 682) |
| micro-ops | `rms_norm`, `qkv_extract`, `silu_mul_split`, `embed`, `argmax` | 1/16/6/1/1 | 0.84 | **NO** | ~9.5 µs fixed per kernel (ablation-verified, `ROADMAP` L246) |
| **total** | | 256 dispatches | **~8.0** | | matches the 8.25-8.78 ms measured |

**Answer to the task's questions 2 and 3:** the GEMVs and the LM head are truly
`cur_len`-independent — code structure (a GEMV over one token's vector, fixed grid) and the
profile agree (4.36 ms at `cur_len 211` against an 842 MB weight stream at 193 GB/s, already the
measured ceiling; the step's 1.14 GB of weights is the 3.9 ms floor). The entire 8.3 → 16.6 ms
growth is **the attention pair plus the dispatch/barrier machinery it drags in**, whose share
rises from 21 % to ≈ 60 % of the step: slope `(16.6 - 8.25) ms / (2308 - 211) ≈` **4.1 µs per
position per step** (`4.1/28 ≈ 0.146 µs per position per layer`). The traffic arithmetic:

* per-layer K+V = `cur_len × 128 B × (nqh + nqh)` = `2308 × 128 × 32` = **9.5 MB/layer**,
  265 MB/step → **0.9 ms at 293 GB/s**. Measured attention ≈ 10 ms ⇒ **≈ 11× off the DRAM floor**,
  an effective **≈ 26 GB/s = 9 % of peak** (`FEASIBILITY.md` §1). Short context is off the same
  floor: 1.66 ms for 211 positions (25 MB/step → 0.086 ms at peak) ⇒ **19× off**. Attention is
  **not bandwidth-bound here; it is latency/occupancy-bound** — as `ROADMAP` L239 suspects.
* The constraint is 16 (or 80) workgroups × a serial chain of `D2 = 64` dependent `fma` per score
  (`gqa_decode_single` L639-643: `dot = dot + (qv.x*kv.x + qv.y*kv.y)`) with no prefetch — exactly
  the stall `gemv`'s ×4 unroll was added to hide (`shaders.rs` L146-149, "four (Wt,X) pairs in
  flight"). The bit-exact `expf_bt` (L578-597, three integer `fma_int`s + `scale_pow2`) runs
  `2 × cur_len × nqh` times per layer and was measured at **+18 %** on the split path
  (`ROADMAP` L99-100).

---

## 3. Why long context is slower than short — the mechanisms, ranked

1. **`n_chunks` grows linearly while the workgroup stays stalled, not saturated.** At
   `cur_len 2308`, `gqa_split_chunk` returns **512** → `n_chunks = 5` → only **80
   workgroups/layer**, and each workgroup's serial work doubles vs a 256-wide chunk. The 2048
   threshold *reduces* parallelism per head exactly when parallelism is the binding constraint.
2. **K and V are re-read once per q-head.** `REP = nqh/nkvh = 2` (`gqa_decode_split_p1` L774), so
   each KV head is streamed by 2 workgroups (`kh = qh / REP`, L786); the single path has all 16
   heads re-stream the same rows. 2× (split) / 16× (single) redundant DRAM traffic on the only
   growing tensor in the step.
3. **Q is re-read inside the score loop, in both paths.** `qbase = qh * D2` (single L632, split
   L787) and the inner loop re-loads `Q[qbase + j2]` for **every** `t` (single L640, split L809):
   `cur_len × D2` loads/thread instead of one 128 B register/L1-resident row. CUDA caches it in
   shared memory first (`kernels.cu` L1033 + L1047 `q_smem[j] = q_row[j];`, then
   `q2 = *(__half2*)(q_smem + j)` at L1054); **the wgpu port dropped that from both paths**.
4. **The merge kernel adds a dispatch per layer** (`+28`, `dispatches_per_step` L674) and re-runs
   `expf_bt` `n_chunks` times per head — small but non-zero.
5. **The micro-ops cannot be fused away** — silu→down would re-run `exp2` per row and rms→gemv
   would change the 1024-lane tree (`ROADMAP` L249-251), so their ~9.5 µs × 84 fixed cost stands.

---

## 4. Split-K for the decode GEMVs: the bit-exactness constraint, spelled out

### 4.1 The exact reduction order that must not move (`shaders.rs::gemv`, L91-217)

The current kernel is **warp-per-output-row with no cross-warp reduction at all**:

```wgsl
@compute @workgroup_size(256)                 // 8 warps
fn gemv(...) {
    let lane = lid.x & 31u;  let warp = lid.x >> 5u;
    let row  = wgid.x * 8u + warp;            // ← row is per-WARP, not per-workgroup
    var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
    var i = lane;
    for (var g = 0u; g < TILES; g = g + 4u) {         // TILES = (k/8)/32 = k/256
        a0 = fma(w00.x, x00.x, fma(w00.y, x00.y, a0)); // granule g+0 → a0
        a1 = ...;  a2 = ...;  a3 = ...;                // granules g+1..g+3
        i = i + 128u;                                  // 4 granules prefetched per pass
    }
    let acc = (a0 + a1) + (a2 + a3);
    let r = bfly(acc, lid.x, lane);           // 5 rounds: lane^16, ^8, ^4, ^2, ^1
    if (lane == 0u) { rows_out[warp] = r; }
    workgroupBarrier();
    if (lid.x == 0u) { ... pack rows_out → Y (ACCUM fuses the residual add) ... }
}
```

Consequences, all load-bearing for the 12/12 MATCH:

* Each row's sum is `((a0+a1)+(a2+a3))`, where `a_j` accumulated granules
  `{i ≡ lane + 32j (mod 128)}` in **ascending** order; the ×4 unroll (L151-192) is legal only
  because "the fma chain still consumes tiles in ascending order" (L148-149).
* The 32 lane values are folded by `bfly`: `+v[^16]`, `+v[^8]`, `+v[^4]`, `+v[^2]`, `+v[^1]` — 5
  sequential shared-memory rounds, a `workgroupBarrier` between each (L115-135), mirroring CUDA's
  `__shfl_xor_sync(acc,[16,8,4,2,1])` (`kernels.cu` L1354-1357). **A fixed 32-leaf binary tree;
  the order of its 31 additions is a global invariant.**
* Warps are fully independent: `bt0`/`bt1` are indexed `wb + …`, `wb = lid & 0xE0` (L116), and
  `rows_out` by `warp` — nothing crosses warps. Asserted: `n % 8 == 0`, `k % 8 == 0`,
  `(k/8) % 32 == 0`, `TILES % 4 == 0` (L92-97).

### 4.2 Why a K-dimension split cannot be bit-exact

The tree's leaves are the 32 lane partials; lane `ℓ`'s partial is the *ascending* sum of granules
`ℓ, ℓ+32, ℓ+64, …`. Every split shape changes that tree:

* **Split the 32 lanes into G groups, merge after.** The `lane^16` round must add `v[ℓ]+v[ℓ+16]`
  for all 32 lanes at once; with lanes 16-31 in another warp, round 1 becomes
  `(g0 chain)+(g1 chain)` where the original interleaves them. Different associativity ⇒ different
  last ulp. **Rejected.**
* **Split the granule range, run two butterflies, add the two scalars.** Gives `B0 + B1` instead
  of the interleaved tree. **Rejected.**
* **Split within a lane (same 32 lanes, two warps, alternating 128-granule tiles).** To keep
  `a0..a3` and the `i = i + 128` rotation, warp 0 takes granules `{i, i+256, …}` and warp 1
  `{i+128, …}`; then `a0` alternates owners, the ascending invariant breaks, and the two warps'
  results still need an order-preserving merge that does not exist. **Rejected.**
* **The only axis left is the output row** — which `warp = lid.x >> 5` already exploits: 8 rows per
  workgroup, `rows/8` workgroups (512/128/768/128 per layer, 18992 for the LM head). Fully
  parallel and bandwidth-saturated already.

**Bottom line: there is no bit-exact K-split of `gemv`**, and nothing to gain from one — the
kernel already streams each weight byte exactly once at 193-293 GB/s against a 293 GB/s ceiling.
**Corollary:** "Decode split-K" as written in `HANDOFF.md` L77 is not actionable on the GEMVs.
The same idea *is* actionable on attention, where it is *already implemented*
(`gqa_decode_split_p1` + `gqa_split_merge`, `shaders.rs` L753/L879) — so the remaining work is
tuning it, not writing it. The plan is §5.

### 4.3 What *would* be bit-safe in `gemv` (for completeness)

* More warps per workgroup (e.g. `@workgroup_size(512)`, 16 rows/workgroup): legal, same per-row
  order, but pointless — the row axis is not the bottleneck (43008 workgroups for 28 layers, 18992
  for lm_head, all bandwidth-saturated).
* Deeper prefetch / unroll: legal (ascending order preserved); `ROADMAP` L253 already bought +2 %,
  and more depth is the only remaining lever, worth < 3 %.

---

## 5. The actual highest-value change: fix attention, not the GEMVs

Proposals, in value order. Every one **keeps `expf_bt`, the two block-reduction trees, `T_SPLIT`'s
`t_idx` stride and every `fma`/`add` order intact** — they change *how much memory is fetched*,
never *in what order values are added*.

**P1 — chunk 256 instead of 512 for `cur_len >= 2048`** (`decoder.rs::gqa_split_chunk` L291-297;
`decoder.rs` L416 already sizes the scratch conservatively as `max_seq.div_ceil(256)`, so it
holds). At `cur_len 2308` that gives `n_chunks = 10` — **160 workgroups/layer instead of 80**: 2×
parallelism, half the serial chain per workgroup, so a latency-bound `cur_len` term halves
(4.1 → ~2.1 µs/position/step), attention ≈ 5.6 ms, step ≈ 11.1 ms. Only `SplitCfg` (L172-177) /
`MergeCfg` (L181-186) `n_chunks` (written at L725-739) moves, and the `> -inf` empty-chunk guard
is per-chunk independent. ⚠ Two numeric changes: each chunk's `chunk_max` now covers 256 instead
of 512 tokens, and the merge's `g_sum` has 10 terms instead of 5 — a re-association of the
online-softmax merge. Re-gate with `decode_check`.

**P2 — prefetch the K/V stream in the score loop** (`gqa_decode_split_p1` L805-814,
`gqa_decode_single` L636-645). Each thread issues `let kv = unpack2x16float(KC[row+j2])` and
immediately consumes it, inside a 64-deep dependent `add` chain. Unrolling ×4 with four in-flight
`(qv, kv)` pairs — the exact trick `gemv` uses at L152-159 — preserves the ascending fma order
and is **provably bit-safe**. The 9 %-of-peak effective bandwidth is a latency symptom; +30-50 %
on the attention stage (≈ 1-2 ms/token at 2308) is a reasonable target. No alignment risk.

**P3 — cache Q in shared memory** (`gqa_decode_single` L632/L640-641 and identically the split
path L787/L809; mirror `kernels.cu` L1033/L1047). `D2 = 64` words = 256 B/workgroup; removes
`cur_len × D2` redundant global loads per thread. Bit-safe (same value, same order); −0.2-0.5
ms/token in the 513-1024 single-path band (`ROADMAP`: 8.73-9.97 ms).

**P4 — share one K/V stream across the `REP = 2` q-heads** (`kh`-indexed tiles loaded once per
workgroup pair): halves the growing DRAM traffic, keeps `t_idx`/`j_idx` per head unchanged, so
bit-safe in principle — but a real rewrite of the thread mapping; only after P1/P2 show the floor.

**P5 (do first, zero risk) — profile at long context.** `src/bin/step_profile.rs` has no
`--cur-len`/`--pos` control and has only been run on the 15s tag; pointing it at the `q06_180s_en`
golden (`pos = 2307`) yields the per-op table at the exact regime of interest. Without it P1 vs P2
cannot be ranked by measurement, which is the roadmap's current state.

### Estimated speedup at `cur_len ≈ 2300` (0.6B, current 16.6 ms/token)

Cost model fitted to the two measured endpoints: `step_ms ≈ 5.5 fixed (4 GEMVs + lm_head +
micro-ops) + 4.1e-3 × cur_len`; at 211 that predicts ~6.0 ms against 8.25 ms measured (the extra
2.2 ms is the 16-workgroup latency floor), at 2308 it gives 5.5 + 11.1 = 16.6 ms.

| proposal | arithmetic | predicted step | `cur_len` term |
|---|---|---|---|
| none (today) | 5.5 fixed + ~11.1 attention | 16.6 ms | 4.1 µs/position/step |
| P1 chunk 256 | fixed + 11.1 × (256/512) ≈ 5.6 | **~11.1 ms** (1.50×) | ~2.1 |
| P2 prefetch ×4 | fixed + 11.1 / 1.4 ≈ 7.9 | **~13.4 ms** (1.24×) | ~2.9 |
| P3 Q in smem | single path only (`cur_len` ≤ 1024) | — | −0.2-0.5 ms at 513-1024 |
| **P1+P2** | 5.5 + 11.1 × 0.5 / 1.4 ≈ 9.5 | **~9.5 ms (1.75×)** | ~1.5 |
| absolute floor | fixed + K/V DRAM (265 MB @ 293 GB/s = 0.9 ms) | ~6.4 ms | 0.9 |

**Alignment risk.** All of P1-P4 leave the *arithmetic* order inside each stage untouched, but P1
changes the number of merge terms, and the chunk partition decides which tokens share a
`chunk_max` in `exp(x - chunk_max)`. The 12/12 MATCH is not mechanically guaranteed here —
`ROADMAP` L120-130 documents that alignment already sits one f16-ulp tie away from flipping at
`q06_180s_en` step 362. **Re-gate every patch with `decode_check` on all 6 × 0.6B fixtures plus
the 1.7B pair, and re-benchmark**; treat `q06_180s_en` as the canary it already is.

---

## 6. The single highest-value change

**Retune the split-attention chunking (P1), measurement-gated by P5 first.**

* Edit `src/decoder.rs::gqa_split_chunk` (L291-297): drop the `>= 2048 → 512` branch (or make the
  chunk sweepable). `decoder.rs` L416 already sizes the scratch as `max_seq.div_ceil(256)` — the
  conservative bound holds.
* Confirm first in `src/shaders.rs::gqa_decode_split_p1` (L753-875) and `gqa_split_merge`
  (L879-929) that `CHUNK` is purely a partition parameter, and in `SplitCfg` (L172-177) /
  `MergeCfg` (L181-186) that only `n_chunks` moves.
* Gate: `step_profile` on `q06_180s_en` → `decode_check --bench` → the full 12/12 matrix.
* Do **not** budget effort for a GEMV split-K: §4 proves it cannot be bit-exact, and §2 proves the
  GEMVs are already at the bandwidth ceiling and independent of `cur_len`.

**In one line:** the 4 per-layer GEMVs (grids 512/128/768/128) and the LM head (grid 18992) are
truly `cur_len`-independent; the 8.3 → 16.6 ms growth is 100 % attention (`gqa_decode_split_p1` +
`gqa_split_merge`, the only kernels whose loops are bounded by `cfg.cur_len`), running at ~9 % of
achievable bandwidth because it is latency/occupancy-bound — 80 workgroups on 20 SMs, a 64-deep
serial `fma` chain per score with no prefetch, and a bit-exact integer `expf_bt` on the critical
path.
