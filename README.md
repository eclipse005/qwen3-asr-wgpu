# qwen3-asr-wgpu

A **wgpu** inference engine for [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR):
the whole pipeline — log-mel front end, audio tower, text decoder — runs on the
GPU through one WebGPU-style API, so the same code targets Vulkan, D3D12 and
Metal instead of one vendor's runtime.

It is a *port*, not a re-implementation: every kernel is written against the
reference's arithmetic (down to accumulation order, the bit-exact `exp`, and
where f16 rounding happens), and every claim below is measured on the machine
described under [Measured envelope](#measured-envelope).

**Verified devices** (same verbatim-parity gate on each, 180 s English clip):

| device | backend | subgroup | result |
|---|---|---|---|
| NVIDIA P104-100 | Vulkan | 32 | MATCH |
| NVIDIA GTX 1070 | D3D12 | none | MATCH |
| Intel iGPU | Vulkan | 8..32 | MATCH |

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

### Choosing the device

Because the port's reason to exist is "one binary, several devices", the choice
is a first-class part of the API rather than a hidden default:

```bash
cargo run --release --bin transcribe -- --list-devices
# [0] Intel(R) Graphics (Vulkan, IntegratedGpu) | ... binding 1023 MiB, subgroup 8..32
# [1] NVIDIA P104-100   (Vulkan, DiscreteGpu)   | ... binding 2047 MiB, subgroup 32
# [3] Intel(R) Graphics (Dx12, IntegratedGpu)   | ... binding 2047 MiB, subgroup none

cargo run --release --bin transcribe -- --device intel …      # by name
cargo run --release --bin transcribe -- --device 1 …          # or #1, by index
cargo run --release --bin transcribe -- --device dx12 …       # first adapter on a backend
cargo run --release --bin transcribe -- --device integrated … # by device class
```

```rust
use qwen3_asr_wgpu::{DeviceSelector, WgpuAsr};

for d in WgpuAsr::devices() { println!("{}", d.describe()); }   // no device created
let asr = WgpuAsr::load_on("model-dir".as_ref(), DeviceSelector::Name("intel".into()))?;
println!("running on {}", asr.device_description());
```

`--adapter <name>` (the older spelling) is still accepted and is exactly
`DeviceSelector::Name`.  Whatever a device grants — binding limit, workgroup
storage, subgroup width — the engine reads from the *negotiated* limits, so a
device that cannot run a given clip is refused explicitly rather than producing
garbage (that refusal is what the tiling in `docs/design-tiled-prefill.md` is
about).

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
