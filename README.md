# qwen3-asr-wgpu

A **wgpu** inference engine for [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR):
the whole pipeline — log-mel front end, audio tower, text decoder — runs on the
GPU through one WebGPU-style API, so the same code targets Vulkan, D3D12 and
Metal instead of one vendor's runtime.

It is a *port*, not a re-implementation: every kernel is written against the
reference's arithmetic (down to accumulation order, the bit-exact `exp`, and
where f16 rounding happens), and every claim below is measured on the machine
described under [Measured envelope](#measured-envelope).

**Verified targets** — the same verbatim-parity gate (0.6B, 180 s English clip,
against the frozen python-hf text) on every runtime this machine has:

| target | device | enc | prefill | decode | elapsed | RTFx | verdict |
|---|---|---|---|---|---|---|---|
| `vulkan:0` | NVIDIA P104-100 | 1012 | 1534 | 5788 | 10.5 s | **17.8×** | MATCH |
| `dx12:0` | NVIDIA GTX 1070 | 1007 | 1644 | 8934 | 13.7 s | 12.9× | MATCH |
| `cpu` | host (20 threads, f32 KV) | — | — | — | 65.1 s | 2.71× | MATCH |
| `vulkan:1` | Intel iGPU | 9176 | 14816 | 25062 | 49 s | 3.6× | MATCH |
| `gl:0` | Intel iGPU (OpenGL) | 7083 | 12131 | 29992 | 51 s | 3.46× | MATCH |

`cpu` is the host decoder (`--cpu-dec`): no adapter, f16 weights widened to f32
once at load, rayon over output rows and over `(row, head)` in the attention.
`src/cpu_decoder.rs` records the five measured pathologies that shaped it, each
with its number: 355 ms/token when the GEMV was sliced by batch (decode is
batch 1); 5.5 s of prefill when it was sliced by `(row, output)` and re-read the
weights per row instead of streaming them once; 5.4 s of prefill when the
`f16 → f32` conversion sat inside the inner loop; 18 s of a 34 s prefill when
the attention converted the f16 caches per key per query row; and 136 ms/token
when the attention was parallelised per row (a decode step *is* one row) instead
of per `(row, head)`.  Its six-fixture RTFx:
3.46 / 3.07 / 3.16 / 3.31 / 2.71 / 2.60 (15 s … 180 s).

One portability bug had to be fixed to get there, and it is the best argument for
this table existing: `Features::SUBGROUP` is a *capability*, not a width.  Intel's
Vulkan driver grants it with `subgroup_min_size..max_size = 8..32`, the 32-lane
xor butterfly then reduced across the wrong lanes, and the model produced fluent
garbage with no error anywhere — the transcript was the only symptom.  Shuffle
paths are now gated on the adapter promising exactly 32 lanes (the shared-memory
fallback is bit-identical); see `docs/PORTING.md` trap 10.

```
wav ──► mel (STFT, 128 bins) ──► audio tower (conv stem + 18 transformer layers)
                                   │
                                   ▼
        prompt (chat template) ──► text decoder (prefill + greedy decode, KV cache)
                                   │
                                   ▼
                     language <NAME><asr_text>transcript
```

## Status

| | |
|---|---|
| **Parity** | **12/12** byte-identical to the frozen Python reference (`0.6B` and `1.7B` × 6 clips, 15 s … 180 s), re-gated on every change |
| **RTFx** (0.6B, 15 s → 180 s) | 24.4 / 21.3 / 19.7 / 24.0 / 17.8 / 20.1 (English short → Chinese 180 s) |
| **vs Python on the same GPU** | 1.95× (0.6B) and 1.52× (1.7B) over 29 languages × 20 clips (FLEURS; WER/CER within ±0.7 pp) |
| **Long audio** | a **15-minute** clip transcribes end to end — 12 065-token prompt, 3 343-token transcript, 98 s, ~5.5 GiB peak, RTFx 9.4 |
| **Not ported** | time stamps (`Qwen3-ForcedAligner` is a second model), batching, the vLLM backend |

Long audio is the interesting case: the reference materialises the `s²`
attention matrix, which at 15 minutes is 2 × 4.7 GB and does not fit an 8 GB
card (the Python SDPA path OOMs there on Pascal). This port tiles the prefill
attention into key slabs (`docs/design-tiled-prefill.md`), which is what turns
"refused" into "runs".

## Quick start

```bash
cargo build --release
```

Weights: the `-hf` checkpoints (`Qwen/Qwen3-ASR-0.6B`, `Qwen/Qwen3-ASR-1.7B`).
The loader reads `thinker.model.*` tensors straight out of the safetensors file.

```bash
# whole pipeline, one call
cargo run --release --bin transcribe -- \
    --model  /path/to/Qwen3-ASR-0.6B-hf \
    --wav    clip.wav --adapter nvidia --max-new 512
```

Useful flags: `--lang en` (force a language; ISO code or full name), `--context`
/`--prompt "hotwords"` (the chat template's system message), `--languages`,
`--cpu-enc` (CPU audio tower, for A/B), `--baseline file.txt` (compare against a
frozen reference text), `--dump dir/` (mel/embeddings/ids).

### Choosing the runtime and device

The selectable axis is the **runtime**, with an optional index for machines that
have several devices on one runtime — the shape ONNX Runtime's execution
providers and llama.cpp's `CUDA0`/`Vulkan0`/`CPU` names use.  The **vendor is
not a selector**: a card of one vendor is reachable through several runtimes, and
those are different code paths (they even differ in numerics — see the parity
matrix above), so vendor and device class belong in the listing, not the choice.

```bash
cargo run --release --bin transcribe -- --list-devices
# * vulkan:0   NVIDIA P104-100 (NVIDIA, dGPU, driver 572.75) binding 2047 MiB, subgroup 32
#   dx12:0     NVIDIA GeForce GTX 1070 (NVIDIA, dGPU, ...) binding 2047 MiB, no subgroup
#   vulkan:1   Intel(R) Graphics (Intel, iGPU, ...) binding 1023 MiB, subgroup 8..32
#   dx12:1     Intel(R) Graphics (Intel, iGPU, ...) binding 2047 MiB, no subgroup
#   gl:0       Intel(R) Graphics (Intel, iGPU, ...) binding 1024 MiB, no subgroup
#   dx12:3     Microsoft Basic Render Driver (Microsoft, CPU, ...)
#   cpu        (not implemented yet — see the CPU section of HANDOFF.md)

cargo run --release --bin transcribe -- --device vulkan:1 …   # iGPU through Vulkan
cargo run --release --bin transcribe -- --device dx12:0 …     # dGPU through D3D12
```

The runtimes are `vulkan`, `metal`, `dx12`, `gl` (the four wgpu drives) and
`cpu` (our own implementation, still to come).  There is deliberately **no
`cuda` and no `dml`**: this crate has no CUDA backend, and wgpu drives Windows
through D3D12 *compute* — DirectML is a different API that would be a separate
integration.  `gl` is wgpu's compatibility runtime (OpenGL/GLES, weakest feature
set); it is kept in the list because old machines only have it, and it is
verified here.

```rust
use qwen3_asr_wgpu::{DeviceSelector, WgpuAsr};

for t in WgpuAsr::device_targets() { println!("{}", t.describe()); }  // no device created
let asr = WgpuAsr::load_on("model-dir".as_ref(),
                           DeviceSelector::Runtime { api: wgpu::Backend::Vulkan, index: 1 })?;
println!("running on {}", asr.device_description());
```

`--adapter <name>` (the older spelling) still works and is the last-resort
substring form.  Whatever a target grants — binding limit, workgroup storage,
subgroup width — the engine reads from the *negotiated* limits, so a target that
cannot run a given clip is refused explicitly rather than producing garbage
(that refusal is what the tiling in `docs/design-tiled-prefill.md` is about).

### The reference pipeline's API

The Python side is three steps — `processor.apply_transcription_request(...)`,
`model.generate(**inputs, ...)`, `processor.decode(ids, return_format=...)` —
and so is this crate ([`src/processor.rs`](src/processor.rs)):

```rust
use qwen3_asr_wgpu::{ReturnFormat, TranscribeOptions, WgpuAsr};

let mut asr = WgpuAsr::load("Qwen3-ASR-0.6B-hf".as_ref(), Some("nvidia"))?;
let opts = TranscribeOptions::default();                 // language, context/prompt

let req  = asr.apply_transcription_request_file("clip.wav".as_ref(), &opts)?;
let ids  = asr.generate(&req, 512)?;                     // greedy, stops at EOS
let out  = asr.decode(&ids, ReturnFormat::Parsed)?;      // {"language", "transcription"}
println!("{}: {}", out.language().unwrap_or("?"), out.transcription_text());
```

`decode` implements all three upstream formats (`Raw`, `Parsed`,
`TranscriptionOnly`) with the reference's exact semantics — including
`Raw` keeping the special tokens and `Parsed` dropping them. The convenience
wrappers (`transcribe_file_opts`, streaming, sessions) are thin layers over the
same internals; `#[ignore]`d test `upstream_api_matches_the_one_call_wrapper`
pins them together.

## Parity gates

Everything is verified against the *reference*, not against itself:

```powershell
powershell -File tools/verify_all.ps1          # 12 runs: 2 models × 6 clips, text vs frozen python-hf
QASR_SLAB=on powershell -File tools/verify_all.ps1 -SkipTiming   # same, forced through the slab attention
```

* `tools/verify_all.ps1` — the gate above; also prints per-phase timings and the
  SM clock (this machine's P104 boosts/drops between 1151/1607/1911 MHz, so
  numbers are only comparable at the same clock).
* `cargo run --release --bin prefill_check -- --tag q06_90s_en` — the layer-by-layer
  check against a CUDA golden: KV cache, prefill logits, first token, and a
  51-step decode continuation that must be *identical*.
* `cargo run --release --bin transcribe -- --diag-enc` — the audio tower's own
  oracle (stage-by-stage vs the CPU reference, including a host recomputation of
  attention operands).
* `cargo test` — the pure-CPU half (tokenizer/prompt/mel geometry, 24 tests).

`docs/PORTING.md` explains why these gates look the way they do and what a port
to another runtime has to preserve.

## Measured envelope

Same machine throughout (Windows, NVIDIA P104-100 8 GB, Vulkan, one GPU job at a
time). 0.6B unless noted.

| clip | mel | audio tower | prefill | decode | total | RTFx |
|---|---|---|---|---|---|---|
| 15 s | 2 ms | 107 | 151 | 346 | 0.64 s | 23.6 |
| 180 s | 25 ms | 1012 | 1534 | 5788 | 8.4 s + 1.45 s front end | 17.8 |
| 15 min | 121 ms | 5236 | 16 259 | 76 556 | 98.3 s | **9.4** |

Memory: 180 s peaks at ~4.6 GiB; 15 minutes at ~5.5 GiB (a 12 065-token prompt
plus a 16 384-token KV cache). The O(s²) attention scratch is gone — the slabbed
attention needs 398 MB at 15 minutes instead of 2 × 4.7 GB.

Where the time goes at 15 minutes: decode 78 % (the KV scan is instruction-bound,
not bandwidth-bound: ~86 GB/s effective against ~250 GB/s available), prefill
17 % (60 % of it the two attention GEMMs, already at this card's ~2.1 TFLOP/s
ceiling), audio tower 5 % (at its weight-bandwidth roofline). `HANDOFF.md` and
`docs/design-tiled-prefill.md` carry the ablation tables behind those numbers.

## Source map

| file | what lives there |
|---|---|
| `src/mel.rs` | log-mel front end (torch-compatible STFT, `center=True`), wav loading, vendored soxr HQ resampling |
| `src/audio_encoder.rs` | CPU audio tower — the reference the GPU one is diffed against (`--cpu-enc`) |
| `src/audio_encoder_gpu.rs` | GPU audio tower: conv stem, window packing, 18 transformer layers |
| `src/decoder.rs` | text decoder: GEMV decode path, prefill (GEMM chain + attention), KV cache |
| `src/shaders.rs` | every WGSL kernel, generated as Rust string functions with the invariants spelled out |
| `src/inference.rs` | orchestration: model load, mel → tower → prompt → prefill → decode, streaming sessions, diagnostics |
| `src/processor.rs` | the reference processor API (`apply_transcription_request` / `generate` / `decode`) |
| `src/prompt.rs` | chat template + the reference's output parsing (`language X<asr_text>…`) |
| `src/bin/` | `transcribe` (CLI) and ~15 probes/benches — see `docs/PORTING.md` |
| `cuda_ref/` | the little CUDA program that produced the golden dumps `prefill_check` diffs against |

## Documentation

| doc | for |
|---|---|
| `docs/PORTING.md` | **start here if you are porting this to another runtime**: the invariants, the traps, the dead ends |
| `docs/design-tiled-prefill.md` | the long-context attention: design, as-built, measurements, eight traps |
| `FEASIBILITY.md` | the original go/no-go: what wgpu can and cannot express, with numbers |
| `ROADMAP-wgpu.md` | the port's phase history |
| `docs/wgpu-best-practices-audit.md` | wgpu-specific findings (batching, descriptors, subgroup experiments) |
| `docs/decode-splitk-analysis.md` | why the decode GEMVs cannot be K-split bit-exactly, and what to tune instead |
| `docs/eval-fleurs*.md` | WER/CER + RTFx against the Python reference on 29 languages |
| `HANDOFF.md` | the live lab notebook (Chinese): current state, bug archaeology, dead ends, next steps |

## License

Apache-2.0, matching upstream Qwen3-ASR. See `LICENSE`.
