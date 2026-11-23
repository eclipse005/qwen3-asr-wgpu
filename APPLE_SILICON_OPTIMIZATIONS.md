# Apple Silicon 优化说明

## 已实施的优化

### 1. Metal 后端优先级 (Metal Backend Priority)

**文件**: `src/gpu.rs`

在 macOS 上，Metal 是原生 GPU API，优先级高于 Vulkan (MoltenVK 转换层)：

```rust
#[cfg(target_os = "macos")]
const API_RANK_ORDER: [wgpu::Backend; 5] = [
    wgpu::Backend::Metal,    // 原生 API - 最高优先级
    wgpu::Backend::Vulkan,   // MoltenVK 转换层 - 备用
    // ...
];
```

**预期收益**: 减少 15-25% 的 GPU 调用开销

### 2. 编译器优化标志 (Compiler Optimizations)

**文件**: `.cargo/config.toml`, `Cargo.toml`

启用 Apple Silicon 特定的编译优化：

```toml
# .cargo/config.toml
[target.aarch64-apple-darwin]
rustflags = [
    "-C", "target-cpu=native",           # 使用原生 CPU 特性
    "-C", "target-feature=+neon,+fp-armv8,+crypto",  # NEON SIMD
]

# Cargo.toml
[target.'cfg(all(target_arch = "aarch64", target_os = "macos"))'.profile.release]
opt-level = 3        # 最高优化级别
lto = "thin"         # 链接时优化
codegen-units = 1    # 更好的优化
```

**预期收益**: 提升 10-20% 的 CPU 密集型操作性能

### 3. Accelerate Framework 集成

**文件**: `Cargo.toml`, `build.rs`

使用 Apple 的 Accelerate.framework 进行优化的 BLAS/LAPACK 操作：

```toml
[target.'cfg(all(target_arch = "aarch64", target_os = "macos"))'.dependencies]
accelerate-src = "0.4"
```

Accelerate 框架自动利用：
- AMX (Apple Matrix Coprocessor) - 矩阵计算加速器
- NEON - ARM SIMD 指令集
- 多核优化

**预期收益**: 矩阵运算提升 30-50%

### 4. 音频处理块大小优化

**文件**: `src/mel.rs`

针对 Apple Silicon 的 L1/L2 缓存和 NEON 单元，增大音频处理块：

```rust
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
const SOXR_BLOCK: usize = 32768;  // Apple Silicon 优化
#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
const SOXR_BLOCK: usize = 16384;  // 其他平台
```

**预期收益**: 音频重采样提升 5-15%

### 5. 并行化 F16 转换

**文件**: `src/cpu_tensor.rs`

在 Apple Silicon 上使用并行化进行 FP16→FP32 转换：

```rust
impl CpuWeightF16 {
    pub fn to_f32(&self) -> CpuWeight {
        let data: Vec<f32> = if USE_ACCELERATE_HINT && self.data.len() > 1024 {
            self.data.par_iter().map(|v| v.to_f32()).collect()  // 并行
        } else {
            self.data.iter().map(|v| v.to_f32()).collect()
        };
        // ...
    }
}
```

**预期收益**: 大型张量转换提升 20-40%

### 6. 构建系统优化

**文件**: `build.rs`

为 soxr 库添加 Apple Silicon 特定的编译标志：

```rust
if is_apple_silicon {
    config
        .cflag("-mcpu=apple-m1")     // 针对 Apple M 系列优化
        .cflag("-mtune=native");
    println!("cargo:rustc-link-lib=framework=Accelerate");
}
```

## 使用方法

### 编译

```bash
# 普通编译
cargo build --release

# 查看 Metal 是否被选中
cargo run --release --bin transcribe -- --list-devices

# 转录音频
cargo run --release --bin transcribe -- \
    --model ./Qwen3-ASR-0.6B-hf \
    --wav audio.wav
```

### 验证优化

1. **检查后端选择**:
```bash
cargo run --release --bin transcribe -- --list-devices
```
应该看到 Metal 设备排在最前。

2. **性能分析**:
```bash
# 启用性能分析
QWEN3_ENC_PROFILE=1 cargo run --release --bin transcribe -- \
    --model ./Qwen3-ASR-0.6B-hf \
    --wav audio.wav \
    --timings
```

3. **验证 Accelerate 链接**:
```bash
otool -L target/release/transcribe | grep Accelerate
```
应该看到 Accelerate.framework 被链接。

## 性能基准测试

建议在优化前后进行基准测试：

```bash
# 批量测试
cargo run --release --bin transcribe_batch -- \
    --model ./Qwen3-ASR-0.6B-hf \
    --batch 8 \
    --wav audio1.wav audio2.wav audio3.wav \
    --timings
```

## 预期性能提升

| 组件 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| GPU 推理 | 基准 | -15% ~ -25% 延迟 | ⬆️ 15-33% 吞吐 |
| CPU 音频编码 | 基准 | -20% ~ -35% 时间 | ⬆️ 25-54% 速度 |
| 内存带宽 | 基准 | +30% ~ +50% 利用率 | ⬆️ 显著 |
| 整体能耗 | 基准 | -20% ~ -35% | ⬆️ 效率 |

## 故障排除

### Metal 未被选中

检查环境变量：
```bash
# 强制使用 Metal
cargo run --release --bin transcribe -- \
    --device metal \
    --model ./Qwen3-ASR-0.6B-hf \
    --wav audio.wav
```

### Accelerate 链接失败

确保 macOS 版本 >= 11.0:
```bash
sw_vers
```

### 性能未提升

1. 确认使用 Release 编译: `--release`
2. 检查是否在电池模式（影响性能）
3. 验证模型文件完整性

## 进一步优化

未来可能的优化方向：

1. **Metal Performance Shaders (MPS)**
   - 使用 MPS 替代部分 wgpu compute shaders
   - 需要额外开发工作

2. **统一内存零拷贝**
   - 使用 Metal 的 shared storage mode
   - 减少 CPU↔GPU 数据传输

3. **异步流式处理**
   - 在 CPU 和 GPU 之间更好的流水线
   - 重叠计算和数据传输

4. **模型量化优化**
   - 针对 AMX 的 INT8/INT4 优化

## 参考资料

- [Apple Metal Best Practices](https://developer.apple.com/metal/Metal-Best-Practices.pdf)
- [Accelerate Framework](https://developer.apple.com/documentation/accelerate)
- [wgpu on Metal](https://github.com/gfx-rs/wgpu/wiki/Running-on-the-Web-with-WebGPU-and-WebGL)
