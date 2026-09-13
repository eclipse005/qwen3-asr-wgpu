# wgpu 移植可行性分析（N 卡）

> 2026-09-12 · 基于真实硬件实测，非推测。所有数字可在本文档末尾的命令下复现。

---

## 0. 结论

**可行，但成败取决于一件事：prefill / 音频编码器的 GEMM 必须调好。decode 那一侧已经赢了。**

| 维度 | 结论 |
|---|---|
| wgpu 在 N 卡上能否跑起来 | ✅ 可以。Vulkan 后端正常，`vec4<u32>` + `unpack2x16float` 路径数值精确（误差 3e-4 / 值 O(577)） |
| decode（占 CUDA 总耗时 75-80%） | ✅ **已超过 cuBLAS**：实测 283 GB/s vs cuBLAS 197 GB/s，逐 token 估算 5.24ms vs 9.05ms（**1.73×**） |
| prefill（占 10-11%） | ⚠️ **最大风险**：手写 WGSL GEMM 只有 **1.0 TFLOP/s**，cuBLAS 是 **4.0-5.2 TFLOP/s**，差 **4.7-6.7×** |
| dispatch 开销 | ✅ 可接受——但**必须**把一整步的 ~255 个 dispatch 批进单个 command buffer（0.62µs/个），逐个 submit 是 31µs/个，差 50 倍 |
| Pascal 的 f16 | ❌ wgpu 不暴露 `SHADER_F16`（Vulkan 驱动层面就没有 `shaderFloat16`）——已找到等价绕法 |
| 预估 RTFx（0.6B / 15s） | 未调优 GEMM：**18×（输给 CUDA 25.4×）**；GEMM 调到 2.5 TFLOP/s：**29×**；3.5 TFLOP/s：**35×** |

**一句话**：把 decode 的 GEMV 移植过来是稳赢的；但如果不把 GEMM 调到 cuBLAS 的 50% 以上，整体 RTFx 会**输**给现在的 CUDA 版。

---

## 1. 实测环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA P104-100（GP104，Pascal **sm_61**，8 GiB，WDDM） |
| 驱动 / Vulkan | 572.75 / 1.4.303 |
| 显存时钟 | 5005 MHz（GDDR5X）→ 真实可用带宽实测 **≥ 293 GB/s** |
| CPU | Intel Core Ultra 7 265K（对照用 iGPU：Intel Graphics） |
| 工具链 | wgpu 30.0.1 / Rust 1.96 |
| 被移植对象 | 本仓库 CUDA 后端（cuBLAS + 34 个 NVRTC kernel），0.6B 文本解码器形状 |

**关于带宽基准的重要更正**：README / ROADMAP 里写的 "P104 ~200 GB/s" 偏低。实测 wgpu GEMV 在 lm_head（296.8 MB）上跑到 **292.8 GB/s**，纯流式读 241-245 GB/s。按 GDDR5X 口径，P104-100 的真实峰值应在 **320 GB/s** 左右。这意味着 **cuBLAS 的 GEMV 平均只用了可用带宽的 ~62%（197/293）**——CUDA 侧本身就有余量（见 §7.1）。

---

## 2. 决定性发现 ①：Pascal 在 wgpu 里没有 f16

```
[0] Vulkan | Intel(R) Graphics   | IntegratedGpu | SHADER_F16=true
[1] Vulkan | NVIDIA P104-100     | DiscreteGpu   | SHADER_F16=false   ← 目标卡
[4] Dx12   | NVIDIA GTX 1070     | DiscreteGpu   | SHADER_F16=false
```

`vulkaninfo` 确认这就是驱动能力，不是 wgpu 的问题：

- P104-100：`shaderFloat16 = false`，`storageBuffer16BitAccess = true`
- Intel 核显：`shaderFloat16 = true`

即 **Pascal 有 16-bit 存储，但没有 f16 算术单元**（GP104 的 FP16 是 1:64 速率，NVIDIA 干脆不暴露该扩展；`VK_KHR_shader_float16_int8` 从 Turing/sm_75 才提供）。而 wgpu 把这两件事打包成同一个 `Features::SHADER_F16`，所以直接 `array<f16>` 这条路在 P104 上是死的。

### 绕法（已验证）

把 f16 权重按 `u32` 存储（每个 u32 装 2 个 f16），用 WGSL **核心内建** `unpack2x16float` 解包成 `vec2<f32>` 再算：

```wgsl
@group(0) @binding(0) var<storage, read> W: array<vec4<u32>>;   // 8 个 f16 / 元素
...
let a = unpack2x16float(wv.x);   // 无需任何扩展
acc = acc + dot(a, x0.xy) + ...;
```

- **字节布局与 `array<f16>` 完全一致** —— 权重加载路径不用改。
- **不需要任何 feature**，所有后端通用。
- **数值精确**：qkv 4096×1024 全 K 校验，4 行误差 0.0002-0.0004（值 O(577)）。
- **性能无损**：Intel 核显上打包路径 60.2 GB/s，反而略快于原生 f16 路径的 56.4 GB/s。

> **附带收益**：这条路径顺带绕开了历史上干掉 wgpu 方案的那个坑。ROADMAP §5.5 记录的 "Intel Vulkan 对 f16 storage buffer 连续读取 device lost"，本次在 **N 卡和 Intel 核显上都未复现**（打包路径读的是 `vec4<u32>`，f16 路径读的是 8 字节对齐的 `vec4<f16>`，都不是当初的标量 `array<f16>` 读取）。不过这条只是"没复现"，不建议当成结论——真要回 Intel 别忘了重测。

---

## 3. 决定性发现 ②：decode（80% 的时间）wgpu 已经赢过 cuBLAS

decode 是**带宽受限**的：每生成一个 token 要把全部权重扫一遍。0.6B 一层的 4 个 GEMV 读 30 MB，28 层 840 MB，加 lm_head 297 MB ≈ **1.14 GB/token**。

实测（同一台机器、同一时刻，形状与 `examples/cublas_gemv_bench.rs` 完全一致）：

| GEMV | rows×cols | 权重 | cuBLAS | wgpu 打包 f16 | 倍率 |
|---|---|---|---|---|---|
| qkv | 4096×1024 | 8.0 MB | 176.6 GB/s | **211.8 GB/s** | 1.20× |
| o_proj | 1024×2048 | 4.0 MB | 127.6 GB/s | **172.7 GB/s** | 1.35× |
| gate_up | 6144×1024 | 12.0 MB | 87.4 GB/s | **243.1 GB/s** | **2.78×** |
| down_proj | 1024×3072 | 6.0 MB | 133.2 GB/s | **190.8 GB/s** | 1.43× |
| lm_head | 151936×1024 | 296.8 MB | 212.7 GB/s | **292.8 GB/s** | 1.38× |

推演：

| | cuBLAS | wgpu |
|---|---|---|
| 每层 4 个 GEMV | 0.271 ms | **0.149 ms**（1.82×） |
| 整个 token（28 层 + lm_head） | 9.05 ms | **5.24 ms**（1.73×） |

对照 ROADMAP 记录的实测 decode 8.78 ms/tok（15s 音频），与 cuBLAS 的 9.05 ms 推演几乎重合 —— 说明 **decode 基本就是 GEMV 本身，attention / norm / 融合 kernel 的开销接近于零**。所以 GEMV 快多少，decode 就快多少。

**为什么 wgpu 能赢**：cuBLAS 在 n=1 的 GEMV 形状上选择不佳。gate_up（6144 行）只跑到 87 GB/s，是可用带宽的 30%；而 lm_head 能到 213 GB/s。这不是 cuBLAS 的上限问题，是 GEMV 形状下的 kernel 选择问题。手写 kernel 反而更容易打满带宽。

**代价**：f32 对照实验（同样 kernel，权重改 f32，读 2 倍字节）耗时正好 2 倍（2.39ms vs 1.21ms）—— 确认 kernel 确确实实是**带宽瓶颈**，没有任何隐性的算力或拖尾开销。

---

## 4. 决定性发现 ③：dispatch 开销——必须批处理

decode 一步大约 255 个 dispatch（28 层 × 9 + final norm + lm_head + argmax + embed）。

| 方式 | 每 dispatch 开销 |
|---|---|
| 只 encode，不 submit | 0.04 µs |
| encode + **一次** submit（全部批在一起） | **0.62 µs** |
| encode + **逐个** submit | **31-32 µs**（50× 更差！） |

模拟一整步：

| 配置 | ms/step | tok/s |
|---|---|---|
| 113 个 GEMV dispatch | 4.59 ms | 218 |
| + 142 个 no-op（共 255 个） | 4.64 ms | 216 |

**结论**：255 个 dispatch 批进单个 command buffer，dispatch 税只有 **+0.05 ms/step**（约 1%），完全可接受。

**反面**：如果用那种"每个算子自己 new 一个 encoder 然后 submit"的朴素 Rust 封装（很多 wgpu 教程就是这么写的），代价是 255 × 31µs = **8 ms/token** —— 直接翻倍还多。这是移植时的头号工程纪律：**一步 = 一个 encoder = 一次 submit**。

另外 `device.poll()` 每步只能调一次（在读取 argmax 结果时）。当前 CUDA 版把 token 留在 GPU 上跨步传递，wgpu 可以照搬，避免每次 host 同步。

---

## 5. 决定性发现 ④：prefill GEMM 是唯一的拦路虎

prefill 是**算力受限**的（算术强度 ≈ m，权重只读一遍，GB/s(W) 只有 2-3）。cuBLAS 在这里用 f32 SIMT 打到 4-5 TFLOP/s（P104 的 f32 峰值约 6.5 TFLOP/s，即 65-77%）。

我写了一个分块 GEMM（`C[m,n] = A[m,k]·W[n,k]ᵀ`，f32 累加，权重打包 f16），扫了微内核配置：

| 配置 | m=384 | m=1536 |
|---|---|---|
| 4×4 微内核 | 862-910 GFLOP/s | 920-929 |
| **4×4 + k 循环全展开** | **851-1044** | **1015-1080** |
| 8×4 + k 展开 | 560-604 | 596-617 |
| 8×8 | 252-287 | 290-292 |
| **cuBLAS f16（f32 累加）** | **4064-5164** | **4384-5321** |

28 层 prefill 总耗时：

| | m=384 | m=1536 |
|---|---|---|
| 手写 WGSL（最好配置） | **352 ms** | **1284 ms** |
| cuBLAS | **75.3 ms** | **191.1 ms** |
| 差距 | **4.7×** | **6.7×** |

数值全部正确（64×64×64 校验 max err = 0.00000）。

**关键观察：微内核越大越慢**（4×4 > 8×4 > 8×8）。这是典型的**寄存器溢出 / 占用率崩塌**。根因是 WGSL 这条链路的先天短板：

- 没有寄存器数量控制，没有 `__launch_bounds__`
- 没有 `cp.async` 之类的异步拷贝，无法做双缓冲流水
- 无法保证共享内存 load 被向量化
- naga → SPIR-V → NVIDIA 驱动这条链路的代码生成质量不如 nvcc

按 4×4 的 LDS:FMA 比例（8 次 LDS / 16 次 FMA）推算，共享内存带宽的天花板就是 50% 峰值 ≈ 3.25 TFLOP/s；我们实测只到 1.0，说明还有约 3× 的纯调优空间（向量化 LDS、更大 BK、手工双缓冲）。

**音频编码器（占 7-10%）是同一类问题**：GPU 侧走 im2col + cuBLAS，同样是 GEMM 受限。

---

## 6. RTFx 预算推演（0.6B / 15s，CUDA 基线 0.63s / RTFx 25.4）

CUDA 的阶段占比（ROADMAP §1.4）：decode 75-80%、prefill 10-11%、音频编码 7-10%、mel+setup ~4%。

| 情景 | 假设 | decode | prefill | 音频编码 | mel | 总计 | **RTFx** |
|---|---|---|---|---|---|---|---|
| **A. 直接移植，不调优** | GEMV 已达标；GEMM 停在 1.0 TF | 0.28 | 0.31 | 0.21 | 0.025 | 0.83 s | **18×** ❌ |
| **B. GEMM 调到 2.5 TF**（cuBLAS 的 55%） | 需要向量化 LDS + 双缓冲 | 0.28 | 0.11 | 0.10 | 0.025 | 0.51 s | **29×** ✅ |
| **C. GEMM 调到 3.5 TF**（cuBLAS 的 78%） | 接近这条链路的理论上限 | 0.28 | 0.08 | 0.07 | 0.025 | 0.45 s | **35×** ✅ |
| 参考：现在的 CUDA | — | 0.49 | 0.07 | 0.05 | 0.025 | 0.63 s | 25.4× |

**所以**：decode 那 1.73× 是确定的收益，但会被未经调优的 GEMM 全部吃掉还倒亏。**GEMM 调到 cuBLAS 的 50% 是这条路的 go/no-go 线。**

注意这个推演只对 15s 短音频成立。**长音频（180s）下 prefill + 音频编码的占比会显著上升，对 GEMM 的要求更严苛**，需要单独跑一遍分阶段计时。

---

## 7. 跨平台视角

### 7.1 先说一个 CUDA 侧立刻能拿的收益

上面那个"cuBLAS GEMV 只跑到可用带宽 62%"的结论，意味着**现在的 CUDA 版自己就有余量**：ROADMAP §1.4 认为 decode 已经贴近带宽下限（"3.75ms/tok floor"），但那个 floor 是按错误的 200 GB/s 算的。按实测 293 GB/s，真实下限是 **3.9 ms/token**，而实测是 8.78 —— 中间那 4.9ms 主要是 cuBLAS 的 GEMV 效率损失。

把 decode 的 5 个 GEMV 从 cuBLAS 换成手写 kernel（就是本次 wgpu spike 里那个思路，`__launch_bounds__` + 每 warp 一行 + vec 读取），decode 有望 8.78 → ~6 ms/tok，**15s RTFx 25.4 → ~35**，而且完全不依赖 wgpu。这是本次调研里性价比最高的一条建议。

> 副作用：这条做完之后，wgpu 在 decode 上的优势就基本消失了（两边都是"打到带宽上限"），到时候 wgpu 的卖点只剩**跨平台**，而不是**更快**。做决策时要把这点算进去。

### 7.2 Intel 核显（实测）

| 路径 | 带宽 |
|---|---|
| f16 扩展 | 56.4 GB/s |
| 打包 f16 | 60.2 GB/s |
| f32 | 63.2 GB/s |

共享内存带宽就是 ~60 GB/s 这个量级，decode 一步约 5.7 ms → **约 175 tok/s**，比 CPU INT8 路径（3-6 ms/tok）略慢或持平。**结论：核显上 wgpu 跑得动、但没有性能意义**，和 ROADMAP §5.6 的 DirectML 结论一致。真正的跨平台收益在 A 卡 / Apple Silicon 这类有独立显存带宽的目标上。

### 7.3 现代 N 卡（Ampere / Ada / Blackwell）—— 需要单独想清楚

这一条很重要，因为它和"用 wgpu 支持 N 卡"的直觉相反：

- **decode**：仍然是带宽受限，cuBLAS GEMV 与现代卡带宽都接近打满 → wgpu 大致打平，可能小幅落后（无优势）。
- **prefill**：cuBLAS 在现代卡上**直接用张量核**（f16 TC 密度是 f32 SIMT 的 8-64 倍）。而 **wgpu / naga 至今不支持 cooperative matrix**（`VK_NV_cooperative_matrix` / `SPV_NV_cooperative_matrix` 都没有接）。也就是说，在 3080/4090 上，prefill 的差距会从 Pascal 上的 4.7× 拉大到 **10× 以上**。

**所以"在 N 卡上用 wgpu 对齐 CUDA"这件事，Pascal 反而是最好做的代际**（因为 cuBLAS 在 Pascal 上没有张量核，只能用 f32 SIMT，和 wgpu 是同一个算力池子）。上到 Turing 之后，张量核这道墙 wgpu 翻不过去。

如果目标包含现代 N 卡并且要求 RTFx 对齐，需要重新评估：这时"跨平台 + 可接受性能"是现实目标，"齐平甚至超过 CUDA"不现实。

### 7.4 A 卡 / Apple Silicon

推测（未实测，标为待验证）：

- **Apple Silicon（Metal）**：wgpu 的 Metal 后端质量最好；M 系列统一内存带宽 100-200 GB/s。decode 有机会跑得不错。**且 Metal 下 `SHADER_F16` 可用**，不需要打包绕法。prefill 同样输给手写 Metal kernel，但没有 cuBLAS 张量核做对照，差距是"相对 CPU"的，反而好看。
- **AMD（Vulkan）**：`shaderFloat16` 在 RDNA 上可用，打包方案是稳的。ROCm/hiplas 的 GEMM 对照会很不利。

---

## 8. 风险清单

| # | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| R1 | **GEMM 达不到 cuBLAS 的 50%** | 🔴 致命 | 这是唯一的 go/no-go。先花 1-2 天专攻 GEMM：向量化 `vec4<f32>` LDS、BK 提到 32/64、寄存器分块手工双缓冲、workgroup 128 线程。达标线：2.5 TFLOP/s @ m=384 |
| R2 | 长音频下 prefill 占比升高 | 🟠 高 | 先补一份 180s 的分阶段计时，确认 prefill+编码器的真实占比 |
| R3 | 每步多次 submit 导致 8ms/token | 🟠 高 | 工程纪律：一步一个 encoder、一次 submit；用 profile 卡住"每步 submit 次数 == 1" |
| R4 | naga 寄存器溢出（已实测到） | 🟠 高 | 微内核不要贪大，4×4 比 8×8 快 3×；用 `naga` 的 SPIR-V 反汇编确认是否 spill |
| R5 | `maxStorageBufferBindingSize = 2047 MiB` | 🟡 中 | 1.7B 模型 f16 权重 3.4GB > 2GB。按层分 buffer（每层 30MB）天然解决；lm_head/embed 各 297MB 也没问题 |
| R6 | 音频编码器的 windowed attention / conv | 🟡 中 | 18 层、seq 短（~187）；先用简单实现跑通，再对照 CUDA 的 `fused_gqa` 做融合 |
| R7 | 精度 | 🟢 低 | 已校验：GEMV 误差 3e-4，GEMM 误差 0.0。与 CUDA 同为 f16 存储 + f32 计算 |

---

## 9. 建议路线（按风险排序）

**阶段 0 —— 先决定要不要做（半天）**
用本文档的 §6 表格和 §7.1 做决策。如果目标主要是"让 A 卡/Mac 能跑"，直接进阶段 1；如果目标主要是"让 N 卡更快"，**先做 §7.1 的 CUDA 手写 GEMV**，那是个稳赚不赔的独立收益。

**阶段 1 —— 打通链路（1-2 周）**
1. `src/wgpu_backend.rs`：`Instance`/`Adapter`/`Device` 封装，适配器按 vendor 选择（对照现有 `Backend` 枚举 + cfg 模式）
2. `src/wgpu_engine.rs`：GEMV（打包 f16 + k 展开）、RMSNorm、SiLU、argmax、embed lookup、KV cache 读写
3. decode 循环：**一步一个 command buffer**，token 留在 GPU 上跨步传递
4. 与 CPU 参考 / CUDA 输出逐 token 对齐（token id 级）

**阶段 2 —— 决定成败（1-2 周）**
5. 攻 GEMM：目标 2.5 TFLOP/s（m=384）。这是整个项目的生死线
6. prefill + 音频编码器接 GEMM
7. 端到端跑通 7 个 fixture，出 RTFx

**阶段 3 —— 吊打优化（可选）**
8. KV cache f16 / head split-K attention 融合
9. **INT8 weight-only**：CPU 路径已有成熟做法（per-channel 对称量化 + `unpack4x8snorm`，同为 WGSL 核心内建）。权重 1 byte，decode 权重流量直接减半——**这是 wgpu 侧唯一可能反超 CUDA 的地方**（CUDA 侧没做 GPU INT8）

**明确不做**
- 用 wgpu 追现代 N 卡的张量核性能（追不上，见 §7.3）
- 在 Intel 核显上追求性能（带宽就 60 GB/s，见 §7.2）

---

## 10. 复现方式

```bash
# 1. wgpu spike（本目录）
cd wgpu
cargo run --release -- nvidia      # 或不带参数自动选独显
cargo run --release -- intel       # 交叉验证

# 2. cuBLAS 对照基线
cd wgpu/cuda_ref && cargo run --release

# 3. CUDA 侧现有 GEMV 基线（会重建本仓库）
cargo run --release --example cublas_gemv_bench -- models/Qwen3-ASR-0.6B
```

spike 覆盖：适配器能力探测 → 打包 f16 / 原生 f16 / f32 三条 GEMV 路径的带宽 → 数值校验（小 shape + qkv 全 K 逐行对照）→ 纯流式读上限 → dispatch 开销（批 / 逐个 submit）→ 整步 decode 模拟 → prefill GEMM 微内核扫描 + 数值校验。

---

## 附：与 ROADMAP §5.5 / §5.6 的关系

原文把 wgpu 和 DirectML 一起归为"三条路全部否决"，理由是 Intel 核显 Vulkan 的 f16 storage device lost、以及 DirectML 的 GEMM 带宽太低。本次调研修正/补充：

1. **当年测的是 Intel 核显，现在是 N 卡**——前提不同，结论不能直接沿用。
2. **那个 Intel f16 device-lost 本次未复现**（两条路径都跑通了），但样本只有一次，建议真要上 Intel 时重测。
3. **"GEMM 差"这个结论本身是对的，而且比当年估计的更关键**——它现在被精确量化成 4.7-6.7× 的差距，并被确认为唯一的拦路虎。
4. **新增了一个当年没有的结论**：decode（80% 的时间）wgpu 反而**快于** cuBLAS 1.73×。
