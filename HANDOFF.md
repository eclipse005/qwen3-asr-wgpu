# 交接提示词：qwen3-asr-rs wgpu 后端（阶段 1 decode 链路，编译收尾 + 验证）

> 把本文件全文交给接手的 AI 即可。它是自包含的。

---

## 任务背景

工作区 `D:\qwen3-asr-rs` 是一个用 Rust 手写的 Qwen3-ASR 语音识别推理项目（无 PyTorch），已有 CUDA 后端（`src/cudarc_engine.rs` + `src/kernels/kernels.cu`，已冻结基线 12/12 逐字一致）和 CPU 后端。现在正在做第三件事：**把推理引擎移植到 wgpu**，目标是跨平台（N/A/Intel/Apple），且在本机 N 卡上性能对齐 CUDA。

**接手点**：独立 crate `D:\qwen3-asr-rs\wgpu\`（自有 `[workspace]`，零依赖 `../src`）已经写完 decode 链路全部代码，编译剩 3 个错误 + 2 个已知连锁问题，修完即可跑首次对齐验证。**你的工作从"修完编译"开始。**

## 硬约束（不可违反）

1. **`wgpu/` 必须完全独立**：只依赖 crates.io（wgpu 30.0.1 / half / safetensors / memmap2 / serde_json / anyhow / bytemuck / pollster / bytes），不 `include!`、不 `path 依赖 ../src`。用户计划移植完成后把整个 `wgpu/` 目录改名移出本仓库，所以一切要自包含。
2. **不要动 `src/` 的 CUDA 后端**（12/12 逐字一致的冻结状态）。唯一例外是 `src/wgpu_golden.rs`——那是移植期临时脚手架（golden dump 工具，`#[cfg(test)]` 才编译），已生成过数据后一般无需再动。
3. **逐 token 对齐是验收标准**：wgpu decode 输出的 token id 序列必须与 CUDA golden 完全一致。为此 WGSL kernel 的**算术顺序必须镜像 CUDA kernel**（f32 加法不结合，末位差异会被几百步自回归放大成不同文本）。归约树、FMA 分组、strict `>` 的 tie-breaking 都已按 CUDA 原样写好，**不要"顺手优化"算术顺序**。

## 当前状态

### 文件清单（全部已写完，等你编译跑通）

| 文件 | 内容 |
|---|---|
| `wgpu/src/gpu.rs` | Gpu 封装：adapter 选择、storage/uniform 分配（16B 对齐）、readback、pipeline 编译（带 error scope） |
| `wgpu/src/weights.rs` | safetensors mmap 加载（支持分片 index.json）、bf16/f32→f16（与 CUDA 位一致）、fused QKV / gate_up 拼接 |
| `wgpu/src/shaders.rs` | 全部 WGSL：`rms_norm`、`gemv`（warp-per-row + xor butterfly 归约 + 残差 epilogue）、`qkv_extract`（per-head norm + rotate-half RoPE + KV 写入）、`gqa_decode_single`（flash 风格，256/512 两档）、`silu_mul_split`、`argmax_into_slot`、`embed_lookup_single` |
| `wgpu/src/decoder.rs` | `WgpuTextDecoder`：28 层 forward_decode 组装、一步 = 一个 command encoder（~256 dispatch 批处理）、token 留在 GPU 跨步传递、`step()` / `bench_steps()` / `step_weight_bytes()` |
| `wgpu/src/mrope.rs` | 自包含 MRoPE cos/sin 表（f64 计算 → f32 一次舍入），interleaved/blocked 两种布局 |
| `wgpu/src/golden.rs` | 读取 CUDA golden dump（f16/i32 bin + meta.json）、diff 工具 |
| `wgpu/src/bin/decode_check.rs` | 验证 + benchmark 主程序：MRoPE 对比 → 载入模型 → 用 golden KV 前缀播种 → 51 步 decode 逐 token 比对 → step0 中间量 + 前 8 步 logits diff → ms/token 与 GB/s |
| `wgpu/golden/q06_15s_en/` | CUDA golden（0.6B，15s_en.wav，seq_len=210，n_decode=51，含 step0_{norm1,qkv,q_out,attn_out,norm2,gate_up,activated,h_out,final_norm}.bin） |

golden 生成工具（已跑过一次，如需重生成）：主 crate 里
`QASR_TAG=q06_15s_en QASR_WAV=tests/fixtures/15s_en.wav cargo test --release --lib dump_decode_golden -- --ignored --nocapture`

### 编译错误：已修 5 处（勿重复修）

1. `decoder.rs` `fused_qkv`/`fused_gate_up`：借用冲突，已在 move 前把 `cols` 存出来。
2. `golden.rs:113` `layer_slice`：已加显式 `<'a>`。
3. `gpu.rs:132` readback：已改 `let mut data`。
4. `shaders.rs` bfly 文档注释里的 `{16,8,4,2,1}` 触发 Rust format-string 解析错误：已改为 `[16,8,4,2,1]`。
5. `shaders.rs:24` 常量名：`std::f32::consts::LOG2E` 不存在，已改为 `LOG2_E`。

### 待修（你从这里开始）

1. **`shaders.rs` 连锁错误**：`silu_mul_split` 里 `log2e = format_f32(LOG2E)` 仍引用旧名 `LOG2E`，常量已改名 `LOG2_E`，把引用同步改掉（应在 436 行附近）。
2. **`gpu.rs:155`**：`self.device.pop_error_scope()` 不存在。wgpu 30 的 API 是 `push_error_scope` 返回 `ErrorScopeGuard`（`impl ErrorScopeGuard { pub fn pop(self) -> impl Future<Output = Option<Error>> }`）。正确写法：
   ```rust
   let guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
   let module = ...; let pipe = ...;
   let err = pollster::block_on(guard.pop());
   if let Some(e) = err { bail!("pipeline {label} failed validation: {e}"); }
   ```
3. **运行时必踩**：`golden/q06_15s_en/meta.json` 缺 `last_token` 字段，而 `golden::Meta` 要求它（`expected_sequence()` 用）。最省事：给 `Meta` 的 `last_token` 加 `#[serde(default)]` 并在 `expected_sequence()` 里对 0 值降级（比对到 `tokens.len()` 为止），或者手动往 meta.json 补 `"last_token": <EOS id，0.6B chat 模板是 151645>`。**别重新设计 Meta 结构。**

### 已知实现细节（改代码前必读）

- **布局约定**（与 CUDA 位级一致，是 token 对齐的前提）：
  - 激活 = f16 按 `array<u32>` 存（word j = 元素 2j/2j+1，即 `__half2` 视图）；
  - 权重 = f16 按 `array<vec4<u32>>` 存（8 half = 16B，即 `uint4` 视图）；
  - KV cache = `[head][max_seq][head_dim]`，decode_check 只把 golden 的前 `seq_len` 个位置写进各 head 槽位。
- `gemv` 前置条件：`n % 8 == 0` 且 `(k/8) % 32 == 0`（0.6B 的 hs=1024、inter=3072、q_dim=2048、fused_qkv=4096、vocab=151936 全部满足；1.7B h=2048/inter=6144 也满足）。
- GQA 单块路径容量 `GQA_SINGLE_CAP=1024`（镜像 CUDA 的 SPLIT_THRESHOLD）；`cur_len <= 512` 用 256 线程版，否则 512。
- Pascal 不暴露 `shaderFloat16`，所以一切 f16 走 `unpack2x16float`/`pack2x16float`，无任何 feature 依赖——这是验证过的核心绕法，别尝试 `array<f16>`。
- WGSL 里 `@builtin(global_invocation_id)` 是全局线程号（spike 时代踩过：当 workgroup 索引用会"带宽虚高 8 倍"）。
- wgpu 30 其它 API 注意：`PollType::wait_indefinitely()`、`BufferSlice::get_mapped_range()` 返回 `Result`、`InstanceDescriptor` 无 `Default`。

## 验证流程（修完编译后）

```bash
cd D:\qwen3-asr-rs\wgpu
cargo build --release
cargo run --release --bin decode_check -- --tag q06_15s_en --adapter nvidia
```

（`model_dir` 在 meta.json 里是相对主仓库根的 `models/Qwen3-ASR-0.6B`；若 cwd 在 wgpu/ 下找不到模型，就在主仓库根用 `cargo run --release --manifest-path wgpu/Cargo.toml --bin decode_check` 跑。）

预期输出分四段：MRoPE 表 diff（应 exact）、step0 九个中间量 diff、前 8 步 logits diff、token 序列比对 + benchmark。**成功 = token 序列 0 mismatch（IDENTICAL）**。step0 中间量允许少量非 exact 但 max|Δ| 在 f16 舍入量级（1e-3 相对值内）；如果 step0 的 norm1/qkv 就大幅偏离，优先查权重布局或 uniform 参数，不要先怀疑 attention。

性能对照（P104-100，CUDA 冻结基线）：

| 指标 | CUDA 当前值 | wgpu 判定 |
|---|---|---|
| 一步 decode | 6.58 ms/token | wgpu ≤ 6.58 即赢（decode 是 wgpu 的主场） |
| GEMV 有效带宽 | 289 GB/s | 对比 decode_check 输出的 GB/s |
| dispatch 开销 | 255 × 5.77 µs ≈ 1.47 ms/tok（22%） | wgpu 批处理 ~256 × 0.62 µs ≈ 0.16 ms |

## 修完之后的路线（按优先级）

1. **换 1.7B 验证**：生成 golden：`QASR_TAG=q17_15s_en QASR_WAV=tests/fixtures/15s_en.wav QASR_MODEL=models/Qwen3-ASR-1.7B cargo test ...`（具体环境变量名看 `src/wgpu_golden.rs` 顶部注释；1.7B 是分片 safetensors，weights.rs 已支持 index.json）。
2. **多 fixture**：90s/180s 长音频（检验长上下文下 KV 布局与 gqa512 路径）。
3. **写 `wgpu/ROADMAP-wgpu.md`**：记录实测数据（不要估算）、文件清单、复现命令、与 CUDA 对照表。
4. **阶段 2（本阶段之后）**：prefill GEMM——目标 2.5 TFLOP/s @ m=384（当前手写 WGSL 只有 1.0，cuBLAS 4.0-5.2，这是 go/no-go 生死线）。spike 里已有微内核扫描代码（`wgpu/src/main.rs`）。

## 环境注意

- 本机 Win11，bash 工具可用但**带转义引号的复杂命令会 eval 失败**，优先用简单命令（cd && cargo ...）；`cmd.exe` 被沙箱拦截。
- `wgpu/target` 已在 `.gitignore`（491MB 构建产物），不要提交。
- github 连接不稳定，push/fetch 偶发 Connection was reset，重试即可。
- **没有明确指令绝不 `git push`**（用户规则：commit 仅指本地提交）。

## 交付要求

- 每步给实测数据，不要估算。
- 报告改动文件清单 + 复现命令。
- 需要决策时问用户，不要自己拍板。
