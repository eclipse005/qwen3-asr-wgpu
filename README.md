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
| `cpu` | host (20 threads; `gemm` prefill) | — | — | — | 48.8 s | 3.61× | MATCH |
| `vulkan:1` | Intel iGPU | 9176 | 14816 | 25062 | 49 s | 3.6× | MATCH |
| `gl:0` | Intel iGPU (OpenGL) | 7083 | 12131 | 29992 | 51 s | 3.46× | MATCH |

On **D3D12 the pipeline build takes ~5.5 minutes** (344 s on Intel, 328 s on
NVIDIA, against 4.6 / 6.6 s on Vulkan for their own Vulkan adapters), and wgpu
30's D3D12 backend does not expose `Features::PIPELINE_CACHE`, so it cannot be
cached away there.  The timings above are *after* that load; D3D12 stays a
fallback for when Vulkan is unavailable, and `vulkan:N` is the practical choice.

`cpu` is the host backend (`--cpu-dec`): no adapter, f16 weights widened to f32
once at load, rayon over output rows and over `(row, head)` in the attention.
Its six-fixture RTFx: 4.25 / 3.77 / 3.95 / 4.17 / 3.61 / 3.47 (15 s … 180 s).

`Features::SUBGROUP` is a capability, not a width: some drivers grant it with
`subgroup_min_size..max_size = 8..32`, where a 32-lane xor butterfly reduces
across the wrong lanes and the model produces fluent garbage with no error
anywhere.  Shuffle paths are gated on the adapter promising exactly 32 lanes;
the shared-memory fallback is bit-identical.

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

Long audio is the interesting case: materialising the `s²` attention matrix at 15
minutes is 2 × 4.7 GB and does not fit an 8 GB card. This port tiles the prefill
attention into key slabs, which is what turns "refused" into "runs".

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

### Library use

```rust
use qwen3_asr_wgpu::{AsrInference, Backend, TranscribeOptions};

let asr = AsrInference::load("Qwen3-ASR-0.6B-hf".as_ref(), Backend::best())?;
let out = asr.transcribe("clip.wav", TranscribeOptions::default())?;
println!("[{}] {}", out.language, out.text);
```

`AsrInference` owns the model and every method takes `&self`, so one instance can
be *shared* rather than duplicated — 0.6B's weights are ~1.2 GiB and each copy
would carry its own KV cache too:

```rust
let asr = std::sync::Arc::new(AsrInference::load(dir, Backend::best())?);
let worker = std::sync::Arc::clone(&asr);
std::thread::spawn(move || worker.transcribe("clip.wav", TranscribeOptions::default()));
```

Concurrent calls serialize on the internal mutex; a transcription is not split
across threads. Everything else is per-call state — each generation prefills its
own prompt from KV position zero, so nothing leaks between calls.

Errors are one enum, `AsrError` (`ModelLoad`, `AudioDecode`, `Inference`,
`InvalidOptions`), with `Result<T>` as the alias every entry point returns.

`TranscribeOptions` carries the whole request — `language` (ISO code or full
name, validated against the 30 supported ones), `context` (the hotwords that
become the chat template's `system` message) and `max_new_tokens` (default
2048). There is no second, positional copy of the ceiling to fall out of sync
with. The struct is `#[non_exhaustive]` and has `with_*` setters:

```rust
let opts = TranscribeOptions::default()
    .with_language("zh")
    .with_context("Swing trading course. Terms: order block, time frame, …")
    .with_max_new_tokens(700);
```

When a language is forced the result reports it back in
`TranscribeResult::language`; with auto-detection the field is whatever the model
named. Streaming is the same picture — `transcribe_streaming` takes a per-token
callback, and `create_streaming_session` takes audio incrementally
(`push_samples` → `flush`).

With `features = ["hub"]`, `AsrInference::from_pretrained(model_id, cache_dir, backend)`
downloads `Qwen/Qwen3-ASR-0.6B` (or `1.7B`) and loads it; the `hub` feature is
off by default because it pulls in reqwest and TLS.

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
#   cpu        the host backend (CPU audio tower + CPU text decoder)

cargo run --release --bin transcribe -- --device vulkan:1 …   # iGPU through Vulkan
cargo run --release --bin transcribe -- --device dx12:0 …     # dGPU through D3D12
```

The runtimes are `vulkan`, `metal`, `dx12`, `gl` (the four wgpu drives) and
`cpu` (our own host implementation).  There is deliberately **no `cuda` and no
`dml`**: this crate has no CUDA backend, and wgpu drives Windows through D3D12
*compute* — DirectML is a different API that would be a separate integration.
`gl` is wgpu's compatibility runtime (OpenGL/GLES, weakest feature set); it is
kept in the list because old machines only have it.

**What is verified where.** Everything above was measured on Windows + NVIDIA.
The Vulkan, D3D12, GL and host paths are all gated on this machine; the Metal
path has not been run anywhere.  Nothing in the engine is Windows- or
NVIDIA-specific — the shaders are WGSL, the device is whatever
`DeviceSelector` picks, and `build.rs` builds the vendored resampler with CMake
on any platform — but "not verified" is not "works", and the first run on a new
runtime is where a driver-specific surprise would show up.  The likely one is
subgroup handling: shuffle paths are gated on the adapter promising exactly 32
lanes, and the shared-memory fallback is bit-identical, so an adapter that
reports something else takes the slow-but-correct path.

```rust
use qwen3_asr_wgpu::{AsrInference, DeviceSelector};

for t in AsrInference::device_targets() { println!("{}", t.describe()); }  // no device created
let asr = AsrInference::load_on("model-dir".as_ref(),
                                DeviceSelector::Runtime { api: wgpu::Backend::Vulkan, index: 1 })?;
println!("running on {}", asr.device_description());
```

`--adapter <name>` (the older spelling) still works and is the last-resort
substring form.  Whatever a target grants — binding limit, workgroup storage,
subgroup width — the engine reads from the *negotiated* limits, so a target that
cannot run a given clip is refused explicitly rather than producing garbage.

### The reference pipeline's API

The Python side is three steps — `processor.apply_transcription_request(...)`,
`model.generate(**inputs, ...)`, `processor.decode(ids, return_format=...)` —
and so is this crate ([`src/processor.rs`](src/processor.rs)):

```rust
use qwen3_asr_wgpu::{AsrInference, Backend, ReturnFormat, TranscribeOptions};

let asr = AsrInference::load("Qwen3-ASR-0.6B-hf".as_ref(), Backend::best())?;
let opts = TranscribeOptions::default().with_max_new_tokens(512);  // + language, context

let req  = asr.apply_transcription_request_file("clip.wav".as_ref(), &opts)?;
let ids  = asr.generate(&req)?;                          // greedy, stops at EOS
let out  = asr.decode(&ids, ReturnFormat::Parsed)?;      // {"language", "transcription"}
println!("{}: {}", out.language().unwrap_or("?"), out.transcription_text());
```

`max_new_tokens` rides in the request's options rather than being passed again to
`generate`, so the ceiling is fixed where the request is built — the same rule
`transcribe` follows.

`decode` implements all three upstream formats (`Raw`, `Parsed`,
`TranscriptionOnly`) with the reference's exact semantics — including
`Raw` keeping the special tokens and `Parsed` dropping them. Note that `decode`
reports the language the *model* named, which is the reference's
`_parse_single_output`: when the language was forced the metadata lives in the
prompt and the field is empty. `transcribe` fills it in from the request instead
— the one place the two disagree. The `#[ignore]`d test
`processor_api_matches_the_one_call_wrapper` pins them together.

## Verification

* `cargo test --release` — the pure-CPU half (tokenizer / prompt / mel geometry,
  config parsing).
* `cargo test --release --test public_api -- --ignored` — the public API against
  a real model: the frozen-text check, `Backend::Cpu`, the incremental session,
  and one instance serving two threads.  Needs
  `QASR_TEST_MODEL` / `QASR_TEST_WAV`, and `QASR_TEST_BASELINE` for the
  verbatim comparison.
* `cargo run --release --bin transcribe -- --model <dir> --wav clip.wav
  --adapter nvidia --max-new 512 --baseline frozen.txt` — the whole pipeline,
  with a verbatim comparison against a frozen reference transcript
  (`MATCH` / `MISMATCH`).
* `cargo run --release --bin transcribe -- --diag-enc` — the audio tower's own
  oracle: stage by stage against the host reference, including a host
  recomputation of the attention operands.

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
ceiling), audio tower 5 % (at its weight-bandwidth roofline).

## Source map

| file | what lives there |
|---|---|
| `src/mel.rs` | log-mel front end (torch-compatible STFT, `center=True`), wav loading, vendored soxr HQ resampling |
| `src/audio_encoder.rs` | CPU audio tower — the reference the GPU one is diffed against (`--cpu-enc`) |
| `src/audio_encoder_gpu.rs` | GPU audio tower: conv stem, window packing, 18 transformer layers |
| `src/decoder.rs` | text decoder: GEMV decode path, prefill (GEMM chain + attention), KV cache |
| `src/shaders.rs` | every WGSL kernel, generated as Rust string functions with the invariants spelled out |
| `src/inference.rs` | the engine (`Inner`) and its public handle `AsrInference`: model load, mel → tower → prompt → prefill → decode |
| `src/backend.rs` | `Backend` — the one-word backend choice |
| `src/error.rs` | `AsrError` / `Result`, the public error type |
| `src/streaming.rs` | incremental sessions (`create_streaming_session` → `push_samples` → `flush`) |
| `src/diagnostics.rs` | probe and dump hooks — engine internals, deliberately not on `AsrInference` |
| `src/hub.rs` | HuggingFace download (`hub` feature) |
| `src/processor.rs` | the reference processor API (`apply_transcription_request` / `generate` / `decode`) |
| `src/prompt.rs` | chat template + the reference's output parsing (`language X<asr_text>…`) |
| `src/bin/` | `transcribe` (CLI) and 11 probes / benches |

## License

Apache-2.0, matching upstream Qwen3-ASR. See `LICENSE`.
