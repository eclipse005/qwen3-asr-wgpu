# wgpu 官方最佳实践核对（2026-09-13, session 4）

本文只记录**有出处**的官方指导，以及本仓库逐条核对的结果。凡是没有官方依据的
推断一律标注出来，不混进结论。设备：wgpu **30.0.1** / Vulkan / NVIDIA P104-100。

## 官方资源在哪（先纠正一个前提）

| 资源 | 是否真有内容 | 说明 |
|---|---|---|
| `https://wgpu.rs/` | 只有入口 | 没有性能/最佳实践正文，只链到 docs.rs 和 GitHub |
| `docs.rs/wgpu/latest/wgpu/` 的 crate 根文档 | **无**性能章节 | 正文是 Getting Started / Extension Specs / Shader Support / feature flags / HDR 色彩空间 |
| **`wgpu/documentation/best_practices/dos_and_donts.rs`** | **有，是官方正文** | 源码里的 `/*! … */` 文档，会被 rustdoc 收进去。见 <https://wgpu.rs/doc/src/wgpu/documentation/best_practices/dos_and_donts.rs.html> |
| **wiki "Do's and Dont's"** | 存在，但内容相同 | <https://github.com/gfx-rs/wgpu/wiki/Do's-and-Dont's>；镜像显示 **Last Modified 2020-05-27**，正文与上面那份源码文档逐条一致 |

也就是说：**官方确实有 best-practices 文档，但总共只有 4 条**，而且 wiki 那页是
2020 年的老页。之前那次审计说"读了 wiki 的 Do's and Don'ts 但没得到可执行结论"——
真实原因是**那页只有 4 条通用建议，本来就没有 compute 专项内容**，不是没读懂。
想要更深的只能去 NVIDIA / Khronos 的一手资料（见下节）。

## 官方原文（4 条，逐字）

> ### Don't: create temporary mapped buffers when updating data
> Instead, `Queue::write_buffer` and `Queue::write_texture` can be used conveniently.
> If you are uploading a lot of data that is generated (as opposed to already sitting
> in a data vector), it may be more efficient to recycle staging buffers in a pool.

> ### Do: group resource bindings by change frequency, starting from the lowest
> For example, put per-frame resources into bind group 0, per-pass resources into bind
> group 1, and per-material resources into bind group 2. This allows the WebGPU
> implementation to keep the other bindings intact, reducing state changes.

> ### Don't: create many resources (buffers or textures) per frame
> This puts pressure on the WebGPU memory allocator and tracker. Prefer coalescing
> smaller resources into larger ones. For buffers, you can create a large buffer and
> use different parts of it for different purposes. For textures, consider texture
> atlases and arrays.

> ### Don't: submit many times per frame
> There is a visible CPU cost per submission, and resources are tracked per submission
> by the implementation. It is fine to have multiple `CommandBuffer`s per submission,
> but the number of `Queue::submit` calls should be limited to a few per frame (e.g. 1–5).

## 逐条核对本仓库

### ① 不要临时 mapped buffer 更新数据 —— **已合规**

全仓库的上传走两条路：`BulkUpload`（`gpu.rs`，256 MiB 预算的 staging，load 期权重）
和运行期小量 `queue.write_buffer`（RoPE 表、token slot、per-dispatch uniform）。
没有"每次更新都建 map 缓冲"的写法。**无待办。**

### ② 按变更频率分组 bind group —— **解码器已合规；prefill 有一条待办**

- 解码路径（`decoder.rs::encode_step`）：bind group 在 `load` 期就建好并缓存
  （`l.bg_gemv_*`、`l.bg_gqa*` 等），**每个 decode step 新建 0 个**。
- prefill 路径（`decoder.rs::prefill`，L1276–1445）：**每个 layer 现建 6 个 bind group**
  → 28 层 × 6 ≈ 170 个/pass。这是官方明说的"per-frame 建资源"反模式（虽然它是 per-pass）。
  **待办**：把 prefill 的 bind group 提到 `load` 期一次性建好。
  （预期收益：小，prefill 只占 180s_en 的 16%；但符合官方建议且改动机械。）
- GPU 音频塔（`audio_encoder_gpu.rs`）：同样每层现建 bind group。

### ③ 不要每帧建大量资源 —— **prefill / 音频塔有违反（同 ②**）

除了 bind group，`audio_encoder_gpu.rs` 还有一处更明确的问题：**每个 encode pass
都 `gpu.storage(...)` 重新分配十几个 activation buffer**（L889–914）。官方这条正是
针对它说的（"prefer coalescing smaller resources into larger ones / create a large
buffer and use different parts of it"）。
**待办**：把这些 buffer 改成 `load` 期一次性分配 + 按 offset 复用（当前每次
`encode` 都重新分配，虽然量级是 ms 级，但属于官方点名的反模式）。

### ④ 不要多次 submit —— **两条路径都已合规**

官方口径是"每 frame 1–5 次"。本仓库的 submit 次数：

| 路径 | submit 次数 | 出处 |
|---|---|---|
| 每个 decode step | **1** | `decoder.rs:854`（`step()` = 1 encoder + 1 pass + 1 submit） |
| prefill（180s，28 层） | ~8（每 4 层 submit+poll，防 Pascal TDR） | `decoder.rs:1420-1428` |
| GPU 音频塔 | ~5（每 6 层 submit+poll） | `audio_encoder_gpu.rs:1079` 等 |

**无待办。** 注意这些周期性 submit 是**故意**的：ROADMAP 记录过 Pascal/WDDM 上一次
提交跑几秒会 device-lost（"Deep-async-queue device loss on Pascal/WDDM"），
所以"每层 submit+poll"是正确取舍，不是违反官方建议。

## 官方文档**没有**覆盖的（不要假装有）

以下问题在 wgpu 官方资源里**查不到**指导，不要引用 wgpu 来支持结论：

1. compute 的 workgroup size / occupancy 怎么选。
2. 小 dispatch 太多怎么办（NVIDIA 的 Vulkan Dos-and-Don'ts 有讲，属厂商资料）。
3. 动态 uniform offset 的合法用法（只有 API 层约束，没有性能建议）。
4. GPU timestamp query 的正确用法与坑。
5. Pascal / pre-Turing 的限制与绕过方式。

唯一一条**已确认**的硬事实（来自 docs.rs 官方 API 页）：

> wgpu 30.0.1 的 `WgslLanguageFeatures` 只有 4 个成员：
> `ReadOnlyAndReadWriteStorageTextures`、`Packed4x8IntegerDotProduct`、
> `UnrestrictedPointerParameters`、`PointerCompositeAccess`。
> —— <https://docs.rs/wgpu/30.0.1/wgpu/struct.WgslLanguageFeatures.html>

`WgslLanguageFeatures` 是 **WGSL 语言扩展**（对应 WGSL 的 `enable <x>;` 指令），
subgroups 不在其中。

**但要小心别把结论说大了**（我第一版就说过头了）：`wgpu::Features` 里**有**
subgroup 能力标志，而且 Vulkan 是支持的：

- `SUBGROUP` —— "Allows compute and fragment shaders to use the subgroup operation
  built-ins and perform subgroup operations (except barriers)." Supported Platforms:
  **Vulkan, DX12, Metal**；文档注明它仍是 native-only（<https://github.com/gfx-rs/wgpu/issues/5555>）。
- `SUBGROUP_BARRIER` —— 需要 `SUBGROUP`，"Without it, enables nothing"；Vulkan/Metal。
- `SUBGROUP_VERTEX` —— 仅 Vulkan。

所以准确的说法是：**API 层有能力标志，但 WGSL 前端没有对应的 `enable` 指令**，
两者都对得上"ROADMAP 里 naga 拒绝 `enable subgroups;`"的实测。
至于 WGSL 里 subgroup 内建到底怎么开放、以及 **Pascal sm_61 是否支持**——
**官方文档完全没写**（`SUBGROUP` 的条目里没有 subgroup size、没有 `@builtin(subgroup_size)`、
也没提任何 WGSL enable 指令；对比 `SHADER_F16` 明确写了"you must add `enable f16;`"）。
**这条必须实测，不能引用文档下结论。**

---

## 厂商/标准一手资料（补 wgpu 文档的空白）

标注：**OFFICIAL-VENDOR** = NVIDIA / Khronos 官方。

### V1. **Pascal 的 fp16 算力是 1/64 —— "要不要用快 f16"这个前提本身是错的**

NVIDIA 官方 *Pascal Tuning Guide*（<https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html>）：

> GP100, designed with training deep neural networks in mind, provides FP16 throughput
> **up to 2x that of FP32** arithmetic. **On GP104, FP16 throughput is lower, 1/64th
> that of FP32.**

脚注：GP100 / GP104 的 compute capability 分别是 6.0 / **6.1**，而 **P104-100 是
GP104 = sm_61**。同一份文档也确认 Pascal **有**原生 packed `half2` FMA
（"operands must be stored in a `half2` vector type"），只是在 GP104 上慢 64 倍。

**推论（算术，非引用）**：20 SM × 128 FP32 lane × 2 × ~1.6 GHz ≈ **8 TFLOP/s fp32**；
按 1/64 得 **fp16 ALU 上限 ≈ 0.13 TFLOP/s**。

**我们 `gemm_bench` 实测的 ~2 TFLOP/s 比这个上限高约 15 倍** —— 所以那个 kernel
**根本不在跑 fp16 算术**，而是「f16 存储 + 解包到 f32 计算」。

> **`unpack2x16float` / `pack2x16float` 不是拖累，它恰恰是能拿到 2 TFLOP/s 而不是
> 0.13 TFLOP/s 的原因。**

cuBLAS 的 ~3× 也与"cuBLAS 同样在做 fp32 速率的 f16 存储 GEMM"一致，差距只能来自
分块 / 共享内存 / 启动开销，**不是 f16 算力**。

GP104 上唯一"高于 fp32 速率"的算术是 **INT8 点积**：`__dp4a` / `__dp2a`
"offer a throughput equal to that of FP32 arithmetic"（即与 fp32 持平，不是更快）。
Vulkan 侧对应 `VK_KHR_shader_integer_dot_product`（1.3 进核心）。

**Vulkan `shaderFloat16` 在 Pascal 上有没有暴露：NOT FOUND** —— 厂商文档没有明确说法
（该特性按 `VK_KHR_shader_float16_int8` 是**可选**的，NVIDIA 还是该扩展的贡献者）。
我们实测的 `VK_FALSE` 是设备级事实，**不要拿文档去反驳它**，也不要假设它不存在。

### V2. **NVIDIA 官方对小 command buffer / 小 dispatch 的说法（直接命中我们的结构）**

<https://developer.nvidia.com/blog/vulkan-dos-donts/>：

> Don't record **tiny command buffers** that contain only a few small draw calls or
> small compute dispatches… **compute work from one command buffer can't overlap with
> compute work in subsequent command buffers, even if there are no barriers.**

> Don't submit a small amount of GPU work. If a queue submission is processed on the
> GPU faster than new ones can be submitted on the CPU, it will result in
> wasted / idle GPU cycles.

> **Don't overlap compute work on the graphics queue with compute work on a dedicated
> asynchronous compute queue on pre-Ampere GPUs.**（Pascal 属 pre-Ampere）

> Minimize the number of `vkCmdBindPipeline` calls, each call has significant CPU cost
> **and GPU cost**.

亦有：barrier 建议（"A barrier may cause a GPU pipeline flush"）、
测量卫生（"Lock GPU clocks… `nvidia-smi` 或 `SetStablePowerState()`"、
"Don't test performance with validation layers enabled"）。

**对我们的意义**：encode 的 per-layer submit+poll（防 Pascal TDR）**正好撞在第一条上** ——
相邻 command buffer 之间的 compute **无法重叠**，GPU 会空转。这是**真实的结构性代价**，
不只是保险。值得实测：TDR 防护能否做粗（按时间预算而非固定层数），
或是否有别的机制不必整条队列排空。

### V3. **Nsight Compute 官方诊断，正好描述我们的 attention（9% 带宽）**

<https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html>：

> High occupancy + low SM pipeline utilization: Warps are present but not issuing —
> the workload is **latency-bound**.

> DRAM throughput is well below peak and no other unit is saturated… Classic
> **bytes-in-flight starvation**… Typically pairs with high `long_scoreboard` stalls
> → **Increase bytes in flight via ILP, vectorized loads, or async copy**.

> Increase **instruction-level parallelism (ILP)** to hide dependent latency. Mix
> independent work so schedulers can issue every cycle.

阈值：achieved occupancy `< 60%` 为低；issue `< 0.5`/cycle 为 issue-starved
（`< 0.2` 严重）；接近上限 = "within ~80% of its peak"。
（Pascal Tuning Guide 补充：GP104 每 SM **64 warp**、**32 block**、64k 寄存器；
"Resource constrained kernels that are limited to low occupancy may benefit from
**increasing the number of concurrent memory accesses per thread**"。）

**对我们的意义**：80 workgroup / 20 SM + 64 深串行点积链 = 教科书级的
bytes-in-flight starvation。**官方给的解法是"每线程更多并发 load（ILP / 展开 +
更宽 vector load）"，不是"更多 dispatch"。** 这与子代理提的"K/V ×4 预取"是同一件事，
现在有官方依据；而"chunk 512→256 让 workgroup 翻倍"只解决占用率那一侧。

### V4. Subgroups 的 Vulkan/WebGPU 前提条件

- Khronos subgroup 教程（<https://www.khronos.org/blog/vulkan-subgroup-tutorial>）：
  `supportedStages` 必须包含 compute；`supportedOperations` 必须含 basic，
  **其余类别都是可选的**；"on **NVIDIA (which has a `subgroupSize` of 32**)"；
  并警告 workgroup 小于 subgroup 会浪费 lane。
- `VK_EXT_subgroup_size_control`：compute 要求
  `WorkGroupSize ≤ SubgroupSize × maxComputeWorkgroupSubgroups`；
  **required-size 支持是可选的**；Vulkan 1.3 进核心。
- WebGPU subgroups 提案（<https://raw.githubusercontent.com/gpuweb/gpuweb/main/proposals/subgroups.md>）：
  需要 `subgroupSupportedStages` **包含 compute 和 fragment**，
  且需要 "**Vulkan 1.3 或 `VK_EXT_subgroup_size_control`**"。

**结论**：subgroup 能不能用**取决于设备能力查询**。所以上文 §H 的"申请
`Features::SUBGROUP` 试试"是正确做法；**但在把 64 深归约换成 subgroup 之前，
必须先查这台设备的实际能力**（fragment stage 与 size-control 两项是最可能的拦路条件）。

### V5. Khronos 关于 f16 转换的警告（印证 V1）

Vulkan-Samples 的 16-bit arithmetic 一节：*"Modern GPU architectures rely on
'packed' f16x2 instructions to achieve improved arithmetic performance"* /
**"Don't: Cast between FP16 and FP32 too much."**

对 GP104 而言前半句不成立（1/64），后半句我们已经不可避免（算术必须在 f32 做）。

### V6. 由官方数据推出的、动手前建议先做的 4 个验证

1. **加一个 fp32-compute 变体的同一个 GEMM**（存储仍 f16，算术明确 f32）。
   官方数据预测它与现在的 packed-f16 kernel 在 **~1.3× 以内** —— 若成立，
   就证明"f16 算术"在这个硬件上一分钱不值，别再往那个方向优化。
2. **CUDA 微基测** `__hfma2` vs `__fmaf`（预期 ~1/64）。注意：需要碰
   `D:\qwen3-asr-rs` **但只读不动**，或另建独立小工程。
3. **反汇编已编译的 pipeline**（Nsight Graphics / RenderDoc）找 `HFMA2`：
   不出现就说明 fp16 ALU 从未参与，"位操作"只是运输、不是 bug。
4. **设备查询 `integerDotProduct4x8BitPacked`**：决定唯一 ≥fp32 速率的路径（dp4a）
   能否从 Vulkan 够到。

---

## docs.rs 上真正可执行、有出处的几条

以下每条都来自官方 API 文档，附链接。按对本仓库的相关度排序。

### A. `zero_initialize_workgroup_memory: false` —— 最直接的一条

`PipelineCompilationOptions`（<https://docs.rs/wgpu/30.0.1/wgpu/struct.PipelineCompilationOptions.html>）：

> Whether workgroup scoped memory will be initialized with zero values for this stage.
> This is required by the WebGPU spec, but **may have overhead which can be avoided for
> cross-platform applications**.

`Features::EXPERIMENTAL_MESH_SHADER` 的文档进一步佐证：

> It is recommended to use `Device::create_shader_module_trusted` with
> `ShaderRuntimeChecks::unchecked()` to avoid workgroup memory zero initialization,
> which can be **expensive due to zero initialization being single-threaded currently**.

**本仓库现状：`create_compute_pipeline` 用的是 `compilation_options: Default::default()`**
（`gpu.rs:196`），即**没有关掉** zero-init。而本仓库的 workgroup 数组相当多：

| shader | workgroup 数组 | 字节数（大致） |
|---|---|---|
| `prefill_gemm` (`shaders.rs:1058-1059`) | `As[bm*pad]` + `Bs[bn*pad]` | 64×17×4 ×2 ≈ **8.7 KB** |
| `argmax` (944-945) | `smax[1024]` + `sidx[1024]` | **8 KB** |
| `gqa_decode_split_p1` (776-779) | `sc[chunk]` + `partial[d*t_split]` + 2×`red_*` | **~5.5 KB** |
| `gqa_decode_single` (622-625) | `sc[cap]` + `partial[d*tchunks]` + 2×`red_*` | 数 KB |
| `softmax_causal` (1262-1263) | 2×`red_*` | 小 |
| `gemv` (108-110) / `rms_norm` (51) / `qkv_extract` (251) | `bt0/bt1/rows_out` / `red` | 小 |

`prefill_gemm` 每层、每个 n-tile、每个 k-tile 都重新清零约 8.7 KB；28 层 × 每层几十个
dispatch 累加起来不是噪声。文档明说这是"可避免的开销"且当前实现是**单线程**清零。

**安全性初查（未实测，改动前必须逐 kernel 复核）**：
`prefill_gemm` 的 `As`/`Bs` 是**先写后读**（`load_as`/`load_bs` 写入 → `workgroupBarrier()`
→ 计算），所以跳清零看起来安全。但 `rms_norm` / `argmax` / softmax 那些归约数组的
写-读模式需要逐个确认边界（例如 `argmax` 只有 `n` 个有效项时，`smax[1024]` 的尾部是否
被读到）。**结论：这是一条高价值待办，但不能盲改——要先测收益（预期可达 prefill 的
几个百分点），再逐 kernel 证明不依赖初值。**

**待办：设 `zero_initialize_workgroup_memory: false`** —— 逐 kernel 核对后再开。

### B. `IMMEDIATES` —— 比动态 uniform offset 更合适的机制

- 特性名是 **`Features::IMMEDIATES`**，**没有** `PUSH_CONSTANTS`。
- 描述（<https://docs.rs/wgpu-types/30.0.1/wgpu_types/struct.FeaturesWebGPU.html>）：
  "Allows the use of immediate data: small, fast bits of memory that can be updated
  inside a `RenderPass`." WGSL 里写 `var<immediate>`；**Supported platforms 明确包含 Vulkan**。
- 计算路径有 `ComputePass::set_immediates` —— "Set immediate data for subsequent
  dispatch calls."；`offset` 和长度必须是 `IMMEDIATE_DATA_ALIGNMENT`（**总是 4**）的倍数。
- 大小：`Limits::max_immediate_size` 文档给出 "Expect the size to be: **Vulkan: 128-256
  bytes**"；**默认值是 0**，要显式申请。

**本仓库现状**：per-dispatch 的 `GDims` 走"256 B slot 的 uniform ring + 动态 offset"，
每条 decode step 约 3 次 `write_buffer`。immediates 正是为这种"每条 dispatch 一小块数据"
设计的，且省掉动态 offset 的 256 B 对齐浪费。**待办（需实测收益）**：把 `GDims` 改成
`set_immediates`。注意它是可选特性，要处理不支持时的回退。

### C. 复用同一个 `PipelineLayout` 可以避免重新绑定

`ComputePipelineDescriptor::layout`（<https://docs.rs/wgpu/30.0.1/wgpu/struct.ComputePipelineDescriptor.html>）：

> Using the same `PipelineLayout` for many `RenderPipeline` or `ComputePipeline`
> pipelines guarantees that **you don't have to rebind any resources** when switching
> between those pipelines.

**本仓库已部分做到**：`family_layout` / `family_layout_dyn` 就是为这个写的
（ROADMAP 里也记了"wgpu 隐式 layout 是 pipeline 独占的"）。decode loop 每 step 切
256/512 两个 sibling pipeline 时不需要重绑，符合文档建议。**无待办。**

### D. `Limits::using_alignment()` —— 把 256 B 动态 offset 对齐降下来

`Limits` 的 `min_uniform_buffer_offset_alignment` 默认 **256**（"Lower is 'better'"），
而 `Limits::using_alignment` 的文档说它 "Modify the current limits to use the buffer
alignment limits of the adapter. This is useful for when you'd like to dynamically use
the 'best' supported buffer alignments."

**本仓库现状**：`gpu.rs` 用的是 `required_limits: limits.clone()`（即 adapter 的实际值），
**已经拿到了 adapter 的真实对齐**，所以 GDims ring 的 256 B slot 是跟着 adapter 走的。
**无待办**（但要确认 adapter 在 Vulkan/NVIDIA 上给的是多少——文档没写各家的值）。

### E. `write_buffer_with` / `StagingBelt` 能省掉每次提交的 staging 分配

`Queue::write_buffer` 文档（<https://docs.rs/wgpu/30.0.1/wgpu/struct.Queue.html>）：

> Currently on native platforms, for both of these methods, **the staging memory will be
> a new allocation**. This will then be released after the next submission finishes.
> To entirely avoid short-lived allocations, you might be able to use `StagingBelt`, or
> buffers you explicitly create, map, and unmap yourself.

但同时明确说了：**"for small values (e.g. a typical uniform buffer whose contents come
from a `struct`), there will likely be no difference"** —— 本仓库的 `write_buffer` 全是
小 uniform / 小表，属于文档明说"没差别"的范畴。**无待办**（decode 每 step 3 次小
`write_buffer` 不受影响）。

### F. `PipelineCache` 只在 Vulkan 上有效，且只省启动时间

`PipelineCache` 文档：计算着色器要编译成机器码，"Pipeline caches allow this computation
to be reused between executions of the program… **most desktop GPU drivers will manage
their own caches, meaning that little advantage can be gained from this on those
platforms**"；且 "This resource currently only works on the following backends: **Vulkan**"。

**本仓库现状**：CLI 一次性进程，且 NVIDIA 驱动自带缓存。**收益预计很小，记为可选。**

### G. `BindingResource::Buffer` 的 `size` 与动态 offset 的坑（我们踩过）

`BufferBinding::offset` 文档：

> If the buffer was created with `BufferUsages::UNIFORM`, then this offset must be a
> multiple of `Limits::min_uniform_buffer_offset_alignment`.

这正是本次 session 在音频塔里踩到的：bind group 的 uniform binding 必须给
`size = 一个 slot`（32 B），否则动态 offset 一律越界——`as_entire_binding()`（整个
ring）配上动态 offset 是**非法**的。这条既是 API 约束，也是我们踩过的真 bug。

### H. **不要写 `enable subgroups;`** —— 那正是我们撞过的墙（有源码证据，已修复）
这条值得单独强调，因为它是本项目**实际踩过**的坑，而且现在有了源码级证据。

- `Features::SUBGROUP` / `SUBGROUP_BARRIER` / `SUBGROUP_VERTEX` **都存在且 Vulkan 支持**
  （见上文）；native-only 的跟踪 issue 是
  <https://github.com/gfx-rs/wgpu/issues/5555>（官方，OPEN）。
- 但 naga 的 WGSL 前端把 `enable subgroups;` **显式实现为 Unimplemented**：
  `naga/src/front/wgsl/parse/directive/enable_extension.rs` 里
  `Self::SUBGROUPS => Self::Unimplemented(UnimplementedEnableExtension::Subgroups)`，
  且 `tracking_issue_num() == 5555`。
  （源码：<https://raw.githubusercontent.com/gfx-rs/wgpu/trunk/naga/src/front/wgsl/parse/directive/enable_extension.rs>）
  写 `enable subgroups;` 会直接报 `EnableExtensionNotYetImplemented`。

**所以正确做法是：不要写 enable 指令，改成申请 `Features::SUBGROUP` 能力位。**
subgroup 内建由能力位门控，不由 WGSL 指令门控。

#### 实测结果（2026-09-13，`cargo run --release --bin feature_probe -- --adapter nvidia`）

**结论：这台 Pascal/Vulkan 上 subgroup 完全可用，ROADMAP 那条旧结论是误判。**

```text
adapter : NVIDIA P104-100 (Vulkan, DiscreteGpu)  driver 572.75
SUBGROUP                YES      SUBGROUP_BARRIER  YES
SUBGROUP_VERTEX         YES      IMMEDIATES        YES
PIPELINE_CACHE          YES      SHADER_F16        no
SHADER_I16              YES
min_uniform_offset_align 256   min_storage_offset_align 32   max_immediate_size 256

requesting: SUBGROUP | SUBGROUP_BARRIER | IMMEDIATES   -> accepted

-- subgroup WGSL (no `enable` directive) --
  COMPILED + RAN. out[0..8] = [1431655764, 1431655765, ...]
  -> subgroup built-ins are reachable on this stack

-- `enable subgroups;` -- rejected as expected (confirms naga #5555)
```

验证方式不是"编译过就算"：kernel 用 `subgroupBallot(lid % 2 == 0)` 与
`subgroupShuffleXor(lid, 1)` 算出 `0x55555555 ^ (lid ^ 1)`，回读值
1431655764 = `0x55555554` 起、逐个 +1，**正是该表达式**，说明 subgroup 操作真的执行了。

**对 decode 的影响与落地结果**：四处理论上都能换，但**只有 `gemv` 是安全的换法**。

- **`gemv`：已改并上线。** 它是**单 warp 的 32-lane xor butterfly**
  （`workgroup_size(256)`，一行一个 warp），逐轮映射到
  `subgroupShuffleXor(t, 16/8/4/2/1)` —— **xor 顺序与加法对象完全相同**，
  所以归约树逐位不变。A/B（`subgroup_bfly_bench`）：512 行 1.18×、
  18992 行 1.17×，**0 个输出不同**。端到端 0.6B `180s_en` decode
  **10019 → 9630 ms**（RTFx 9.88 → **10.50**），`1.7B 180s_en` decode
  **15256 → 14851 ms**；**六项 fixture + 1.7B 两项全部 MATCH**。
- **`gqa_decode_single` / `gqa_decode_split_p1` / `gqa_split_merge`：未改，且不应改。**
  它们的归约形状是 `if (lid.x < sh) { red[i] op red[i + sh] }`，
  宽度是 **BS（跨多 warp）**、每轮参与线程减半 —— 换成 subgroup 等于换归约树，
  **会破坏 bit-exactness**。这与 ROADMAP 里"不改变 decode kernel 加法顺序"的约束一致。

这次调查里**唯一一条"原以为不可能、实际可行、且已上线"的优化**，就是 `gemv` 那处。

### H2. `zero_initialize_workgroup_memory` 实测

同一次 probe：

```text
-- zero_initialize_workgroup_memory --
  zero_initialize=true  pipeline OK
  zero_initialize=false pipeline OK
-- correctness with zero_initialize=false --
  cpu sum 14272.000 | gpu 14272.000 | MATCH
```

即：两个取值都能建 pipeline（说明**确实可关**），且对一个"先写后读全宽度"的
workgroup 归约 kernel，关掉之后结果与 CPU 一致。**但这只证明了机制可用**——
本仓库真正的 kernel（`rms_norm` / `argmax` / softmax 的归约数组在 `n < 数组长度`
时会读到未初始化尾部）**仍需逐 kernel 核对**，见 §A / §I。

### H3. 顺带拿到的、对另两条待办有用的限制值

| 限制 | 本机值 | 影响 |
|---|---|---|
| `max_immediate_size` | **256**（Vulkan 文档区间 128–256 的上限） | §B 的 `IMMEDIATES` 方案可用，`GDims` 只有 32 B，绰绰有余 |
| `min_uniform_buffer_offset_alignment` | **256** | 动态 uniform offset 环必须 256 B 对齐（我们已经是） |
| `min_storage_buffer_offset_alignment` | **32** | 若改成 storage buffer 传 per-dispatch 数据，对齐要求低 8 倍 |
| `max_storage_buffer_binding_size` | 2047 MiB | 权重 buffer 够用（默认 128 MiB 不够，我们已按 adapter 申请） |
| `SHADER_F16` | **no** | 与 ROADMAP 记录一致：f16 无原生算术（且 GP104 速率本就是 1/64，见 §V1） |
| `SHADER_I16` | **YES** | INT8/INT16 路径可能可用（配合 §V1 的 `dp4a` 问题） |

**这对本仓库很关键**：ROADMAP 里记着"naga 拒绝 `enable subgroups;` → warp-shuffle
butterfly 不可用 → 只能用 shared-memory butterfly"。**当时很可能只是写错了门控方式**，
而不是这台硬件不支持。Pascal 的 subgroup width 是 32（一个 warp），如果
`SUBGROUP` 在本机 Pascal/Vulkan 上能申请到，decode 的 butterfly 归约就有机会换成
warp shuffle，`gqa` 两条 attention 路径都受影响。**这是一条高价值待验证项。**

（注意：源码证据来自 trunk，未逐字节核对 naga 30.0.0 crate；实测为准。）

### I. `zero_initialize_workgroup_memory` 有官方 changelog 背书

v0.20.0 release notes 的 "Other performance improvements" 里明确写着：

> Support disabling zero-initialization of workgroup local memory in compute shaders.
> By @DJMcNab in #5508

也就是说这不是"文档顺带一提"，而是**作为性能改进正式发布的特性**。与上文 A 节
相互印证，进一步说明这条值得做。另：v28.0.0 changelog 提到 immediates 当前仍会
zero-initialize 你声明的那段范围，"This is not spec compliant…"（对我们用 immediates
有参考价值）。

### J. 官方 wiki 里另外两条能用的（含一条 Pascal/Windows 的直接相关项）

| wiki 页 | 原文/要点 | 与本仓库关系 |
|---|---|---|
| `Encapsulating-Graphics-Work` | "`Queue::submit()` is expensive for wgpu to execute." | 印证我们的 submit 计数纪律（decode 1 次/step） |
| `Debugging-performance-issues` | 只有 3 条：`--release` 构建、`InstanceFlags::DEBUG` 默认在 dev 开启、dev 有更多断言。**没有别的** | 已合规（我们一直用 `--release`） |
| `Debugging-wgpu-Applications` | "If RenderDoc is used with a compute shader without any normal rendering components, `device.start_output()` must be called before enqueuing work for RenderDoc to pick up the pipeline." | 以后用 RenderDoc 抓 compute 要记住 |
| **`Known-Driver-Issues` / Nvidia / Vulkan** | **`write_buffer` only working when aligned to 16**（wgpu#1323，GTX 1050/1070，Windows） | **我们是 Pascal + Windows**，值得留意；不过我们的 per-dispatch uniform slot 是 32 B 对齐、`min_uniform_buffer_offset_alignment` 默认 256，天然满足 |

### K. 仓库里**没有**性能文档（已用 API 核实）

- 根目录无 `PERFORMANCE.md` / `PROFILING.md`。
- `docs/` 只有：`README.md`、`broadcast_license.nu`、`managing-cargo-dependencies.md`、
  `release-checklist.md`、`review-checklist.md`、`running-tests-on-android-and-ios.md`、
  `testing.md`。
- 不存在独立的优化指南。**性能正文只在 `documentation/best_practices/` 那 4 条 +
  散落的条目级文档里。**

### L. `memory_hints` 已是默认最优 —— **无待办**
`MemoryHints::Performance` 文档："Favor performance over memory usage (**the default
value**)"。`gpu.rs` 用 `..Default::default()`，**没有**显式设 `memory_hints`，
所以拿到的就是默认的 `Performance`。**无待办**（文档也没给 compute-only 负载的选择建议）。

### M. `required_limits: limits.clone()` 与官方警告相冲突 —— **低优先级待办**

`Limits` 的中央警告（<https://docs.rs/wgpu/30.0.1/wgpu/struct.Limits.html>）：

> Requesting limits that are 'better' than you need may cause performance to decrease
> because the implementation needs to support more than is needed.
> **You should ideally only request exactly what you need.**
> We recommend starting with the most restrictive limits you can and manually increasing
> the limits you need boosted.

**本仓库现状**（`gpu.rs:65`）：`required_limits: limits.clone()` —— 直接要了
**adapter 的全部上限**。这与官方建议相反。理论上可能让实现做更多校验/预留。

**但**：本仓库的权重 buffer 很大（`max_storage_buffer_binding_size` 默认只有 128 MiB、
`max_buffer_size` 默认 256 MiB），所以我们需要高于默认的 limit。正确做法不是
"clone 全部"，而是**只把我们真正需要的几项抬到 adapter 给的值，其余用默认**。
**待办（低优先级，需实测是否有收益）**：把 `limits.clone()` 换成一份显式的、
最小化的 `required_limits`。

### N. 额外的 decode 循环相关事实

- `ComputePipelineDescriptor::layout` 文档还指出：用隐式默认 layout 的 pipeline，
  "these bind groups **cannot be used with any other pipelines**… using an explicit
  layout is recommended in most cases." —— 本仓库已经用显式 family layout（见 ②/C），一致。
- `CommandEncoder::transition_resources` 有一节 **"Batching Barriers"**：wgpu 不在
  多个 command buffer 之间批量合并它自动插入的 barrier，"may lead to suboptimal barrier
  placement"。本仓库把整条链放在**单个** command buffer / 单个 compute pass 里，
  所以不受影响。**无待办**（但这是"多 command buffer"方案的已知代价，以后若拆分要注意）。



## 未完成

三路网络研究（docs.rs / gfx-rs repo / NVIDIA+Khronos）已完成，内容全部并入本文。
剩下的都是**必须上机实测**、无法靠文档定论的：

1. 本机能否申请到 `Features::SUBGROUP`（以及 `SUBGROUP_BARRIER`）——
   直接决定 decode 的 butterfly 归约能否换 warp shuffle。见 §H / §V4。
2. 关掉 `zero_initialize_workgroup_memory` 的实际收益与安全性（逐 kernel）。
   见 §A / §I。
3. fp32-compute 变体 GEMM 是否落在现 kernel 的 ~1.3× 内（验证 §V1 的推论）。
4. `integerDotProduct4x8BitPacked` 是否可达。
5. `IMMEDIATES` 相对动态 uniform offset 的实际收益（官方无任何测量数据）。

**明确查不到、不要编的**（三路都确认）：

- wgpu 官方对 compute 的 occupancy / workgroup size 指导。
- wgpu 官方对"小 dispatch 太多"的指导（只有 NVIDIA 那份厂商资料有）。
- wgpu 官方关于 Pascal / pre-Turing 限制的任何表述。
- 各家 adapter 的 `min_uniform_buffer_offset_alignment` 实际值。
- NVIDIA 关于"Pascal 不支持 Vulkan `shaderFloat16`"的明确表述（设备查询才是事实来源）。
- Khronos 关于"push constants 比动态 UBO 快"的任何量化结论。


