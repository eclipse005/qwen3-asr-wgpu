# 新窗口启动提示词：wgpu 后端移植

> 使用方式：把下面横线之间的整段内容，原样粘贴到一个新对话里。

---

# 任务：为 qwen3-asr-rs 实现 wgpu 推理后端（跨平台 + 对齐 CUDA 性能）

**工作区**：`D:\qwen3-asr-rs`

## 一、项目是什么

用 Rust 手写实现 Qwen3-ASR 语音识别推理，不依赖 PyTorch。模型在 `models/` 下：
`Qwen3-ASR-0.6B`（单文件 safetensors）和 `Qwen3-ASR-1.7B`（**分片**：`model-00001/00002-of-00002.safetensors` + `model.safetensors.index.json`）。

已有两个后端：
- **CUDA**：`src/cudarc_engine.rs` + `src/kernels/kernels.cu`（35 个 kernel，预编译 7 架构 PTX 放在 `ptx/`）
- **CPU**：`src/cpu_engine.rs`

## 二、这次要做什么

把推理引擎移植到 **wgpu**，目的是**多后端跨平台**（N 卡 / A 卡 / Intel / Apple Silicon 都能跑），
并且在当前这台机器上性能**对齐甚至超过** CUDA 版。

本机显卡：**NVIDIA P104-100**（Pascal sm_61，8GB，实测可用带宽约 320 GB/s）。

跨平台是目的，性能是硬指标——用户明确要求"RTFx 能齐平甚至超过 CUDA 手写版"。

## 三、先读这三份材料（已经做好，不要重做）

1. **`wgpu/FEASIBILITY.md`** — 可行性分析报告，全部基于真实硬件实测。**先读第 0 节结论 + 第 9 节建议路线。**
2. **`wgpu/src/main.rs`**（1240 行）— 可编译可运行的实测 spike：适配器/特性探测、打包 f16 / 原生 f16 / f32
   三条 GEMV 路径的带宽、数值校验（小 shape + qkv 全 K 逐行对照）、纯流式读上限、
   dispatch 开销（批 vs 逐个 submit）、整步 decode 模拟、参数化 GEMM 微内核扫描。
3. **`wgpu/cuda_ref/src/main.rs`** — cuBLAS prefill GEMM 对照基线。

跑起来：
```bash
cd wgpu && cargo run --release -- nvidia     # 也可用 intel 做交叉验证
cd wgpu/cuda_ref && cargo run --release
```

## 四、已确认的四个关键事实（直接用，不必重新验证）

**1. Pascal 的 Vulkan 不暴露 f16。**
P104 的 `shaderFloat16 = false`（Intel 核显反而是 true），`VK_KHR_shader_float16_int8` 要 Turing+。
所以 `array<f16>` 在 P104 上不可用。
**绕法已验证**：f16 打包进 `u32`，用核心内建 `unpack2x16float` 解包。字节布局与 `array<f16>` 完全一致，
不需要任何扩展，数值精确（误差 3e-4 / 值 O(577)）。spike 里的 `SHADER_P16` 就是这条路径。

**2. decode 那一侧已经赢 cuBLAS。**
decode 占 CUDA 总耗时 75-80%，是纯带宽受限。wgpu 实测 283 GB/s vs cuBLAS 197 GB/s。
**这部分移植过来就是赚的。**

**3. prefill GEMM 是唯一的拦路虎，也是生死线。**
手写 WGSL 分块 GEMM 最好只到 **1.0 TFLOP/s**，cuBLAS 是 **4.0-5.2 TFLOP/s**（差 4.7-6.7×）。
微内核越大越慢（4×4 > 8×4 > 8×8），典型的寄存器溢出 / 占用率崩塌——
WGSL 没有寄存器控制、没有 `__launch_bounds__`、没有 `cp.async`。
**达标线：2.5 TFLOP/s @ m=384。达不到，整体 RTFx 会输给 CUDA 版。**

**4. dispatch 必须批处理。**
一整步 decode 约 255 个 dispatch。批进单个 command buffer 是 **0.62 µs/个**；
逐算子 submit 是 **31 µs/个**（差 50 倍，会让 RTFx 直接减半）。

## 五、硬约束

1. **转录内容必须逐字不变**（用户硬要求）。任何影响数值的改动后必须跑：
   ```bash
   cargo run --release --example verify_baseline -- cuda all
   ```
   期望 **12/12 逐字一致**（0.6B + 1.7B 各 6 个 fixture）。同时确认同一二进制重跑结果稳定
   （比对 `target/verify_got_*.txt` 的 md5）。
2. **不要动 CUDA 后端**。`src/` 是已验证的稳定版本（12/12 逐字一致、长音频跨 run 可复现）。
   wgpu 代码先放在 `wgpu/` 独立 workspace 里。
3. **PTX 是生成物**。若改动了 `src/kernels/kernels.cu`，必须重新生成：
   `powershell scripts/compile-ptx.ps1`（7 个架构），否则运行时找不到新符号。
   合并冲突时也**不要手工解 `ptx/*.ptx`**，取任一侧后重新生成即可。
4. **增量验证**。每加一个算子就和参考实现对齐，不要一次性写完再测。

## 六、建议的推进顺序

**先做一个架构决策**：wgpu 引擎做成 `wgpu/` 里的独立实验 crate，还是 `src/` 里 feature-gated 的后端？
建议**先在 `wgpu/` 独立 crate 里跑通并把性能做到达标，再考虑并入主 crate**——
wgpu 依赖很重，过早并入会拖慢主 crate 的编译。
如果需要复用主 crate 的模型加载 / tokenizer / 音频前处理，可以在 `wgpu/Cargo.toml` 里加
`qwen3-asr = { path = "..", default-features = false }`。

**阶段 1 — 打通 decode 链路**
GEMV（打包 f16）、RMSNorm、SiLU、argmax、embed lookup、KV cache 读写。
参考实现是 `src/kernels/kernels.cu` 里的 `gemv_f16`：**一个 warp 管一行输出**，
权重行和激活都用 `uint4`（16 B = 8 个 half）lane-stride 读取（保证每次 warp 访存是连续 512 B），
4 个独立 f32 累加器缩短依赖链，f16 存储 + f32 累加，残差加折进 epilogue。
一步一个 command buffer，token 留在 GPU 上跨步传递。
目标：与 CUDA 输出**逐 token 对齐（token id 级）**。

**阶段 2 — 攻 GEMM（决定成败）**
目标 2.5 TFLOP/s @ m=384。然后接 prefill + 音频编码器，端到端跑通 6 个 fixture，出 RTFx。

**阶段 3 — 可选优化**
INT8 weight-only（权重 1 byte，decode 权重流量减半，核心内建 `unpack4x8snorm`）。
**这是 wgpu 侧唯一可能反超 CUDA 的地方**（CUDA 侧没做 GPU INT8）。

**明确不做**：追现代 N 卡的张量核（追不上，wgpu/naga 至今不支持 cooperative matrix）；
在 Intel 核显上追性能（带宽只有 60 GB/s）。

## 七、环境注意事项（都是踩过的坑）

- wgpu 版本 **30.0.1**，API 有变化：`PollType::wait_indefinitely()`、
  `BufferSlice::get_mapped_range()` 返回 `Result`、`InstanceDescriptor` 无 `Default`、
  `RequestAdapterOptions` 有 `apply_limit_buckets`。
- WGSL 里 `@builtin(global_invocation_id)` 是**全局线程号**，`workgroup_id` 才是 workgroup 索引。
  spike 在这里踩过坑，导致测出的带宽假高 8 倍。
- 本机 `cmd.exe` 被沙箱拦截；跑 nvcc 要用 PowerShell 调 `scripts/compile-ptx.ps1`。
- 到 github.com 的连接不稳定，git 操作偶发 `Connection was reset`，重试即可。
  非交互终端下 push 需要 `git -c credential.helper='!gh auth git-credential' push`。
- `wgpu/target` 已在 `.gitignore` 里，不要提交（491 MB）。

## 八、对齐目标（CUDA 侧当前水平）

| 指标 | 当前值 |
|---|---|
| decode GEMV 带宽（5 个投影合计） | **289 GB/s**（cuBLAS 208 GB/s） |
| 一步 decode | **6.58 ms/token** |
| RTFx 0.6B（15/30/90/180s） | **30.4 / 21.6 / 23.4 / 18.1** |
| RTFx 1.7B（15/30/90/180s） | **12.4 / 11.0 / 11.5 / 11.0** |
| prefill GEMM（cuBLAS） | 4.0-5.2 TFLOP/s |
| decode 剩余开销 | 255 次 launch × 5.77 µs ≈ 1.47 ms/token（占 22%） |

把你的 wgpu 数字和这张表逐项对比，就能知道是赢是输。

## 九、交付要求

- 每个阶段结束给出**实测数据**（不要估算），并和上表对比
- 报告改动文件清单 + 复现命令
- 需要决策时直接问，不要自己拍板
- 完成后更新 `wgpu/FEASIBILITY.md` 或新建 `wgpu/ROADMAP-wgpu.md` 记录进展

---

## 附：这条路的判断依据（供你参考，不必重复验证）

- 在 P104 这种**没有张量核**的 Pascal 卡上，"wgpu 对齐 CUDA"是可行的：
  两边都是同一个 f32 SIMT 算力池，cuBLAS 用不上张量核。
- 换到 Ampere/Ada 反而不行：那边 cuBLAS 走张量核，差距会从 4.7× 拉大到 10×+。
- Intel 核显实测约 60 GB/s（共享内存带宽），能跑但没有性能意义。
- 所以：**跨平台是这条路的价值，N 卡对齐是附加收益，别把目标搞反。**
