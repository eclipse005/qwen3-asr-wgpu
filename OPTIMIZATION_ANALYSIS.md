# Qwen3-ASR-wgpu Apple Silicon 优化分析报告

## 项目概述

这是一个基于 Rust 和 wgpu 的 Qwen3-ASR 语音识别推理引擎，使用 GPU 加速进行跨平台语音转文本。

### 当前架构
- **语言**: Rust (1.95.0)
- **GPU 抽象**: wgpu 30.0.1
- **音频处理**: CPU fallback + GPU 编码器
- **模型**: Qwen3-ASR 0.6B / 1.7B
- **量化**: 支持 INT8
- **并行**: rayon + gemm (CPU), wgpu compute shaders (GPU)

## Apple Silicon (M系列芯片) 特定问题

### 1. Metal 后端优化不足

**当前状态**:
```rust
// gpu.rs 中的运行时优先级
const API_RANK_ORDER: [wgpu::Backend; 5] = [
    wgpu::Backend::Vulkan,  // 优先级 0
    wgpu::Backend::Metal,   // 优先级 1 ⚠️
    wgpu::Backend::Dx12,
    wgpu::Backend::Gl,
    wgpu::Backend::BrowserWebGpu,
];
```

**问题**: Metal 在 macOS 上是原生 API，但优先级低于 Vulkan (通过 MoltenVK)。MoltenVK 增加了翻译开销。

### 2. 统一内存架构 (UMA) 未充分利用

Apple Silicon 的 CPU 和 GPU 共享统一内存，当前实现：
- 音频编码器默认 CPU (`EncoderBackend::Cpu`)
- CPU↔GPU 数据传输仍然通过标准 wgpu buffer 映射
- 未利用零拷贝共享内存的优势

### 3. AMX (Apple Matrix Coprocessor) 未激活

Apple Silicon 拥有专用的 AMX 矩阵协处理器，但当前的 `gemm` crate 和 CPU 实现不直接利用：
```rust
// audio_encoder.rs
use gemm::{gemm, Parallelism};  // 通用 GEMM，未针对 AMX
```

### 4. 音频重采样性能

```rust
// mel.rs
const SOXR_BLOCK: usize = 16384;  // 固定块大小
```
使用外部 C 库 soxr，未针对 M 芯片的 NEON 向量化优化。

## 优化建议

### 优先级 1: Metal 优先级调整

**影响**: 🔥🔥🔥 高  
**复杂度**: 🟢 低

在 macOS 上优先使用 Metal，避免 MoltenVK 转换开销。

### 优先级 2: 利用统一内存架构

**影响**: 🔥🔥🔥 高  
**复杂度**: 🟡 中

使用 Metal 的 shared storage mode 实现零拷贝数据传输。

### 优先级 3: 启用 Accelerate 框架

**影响**: 🔥🔥 中  
**复杂度**: 🟢 低

使用 Apple Accelerate.framework 替代通用 GEMM，自动利用 AMX。

### 优先级 4: 编译器优化标志

**影响**: 🔥🔥 中  
**复杂度**: 🟢 低

添加 Apple Silicon 特定的编译优化。

### 优先级 5: Shader 优化

**影响**: 🔥 低到中  
**复杂度**: 🔴 高

针对 Apple GPU 架构调整 compute shader 的 workgroup 大小。

## 性能预期

优化后预计性能提升：
- **延迟**: -15% 到 -30%
- **吞吐量**: +20% 到 +40%
- **内存带宽利用率**: +30% 到 +50%
- **能耗**: -20% 到 -35%

## 实施计划

1. ✅ 修改 Metal 优先级和设备选择逻辑
2. ✅ 添加 Apple 特定编译优化
3. ✅ 集成 Accelerate.framework
4. ✅ 优化统一内存使用
5. ⏳ 性能验证和基准测试

