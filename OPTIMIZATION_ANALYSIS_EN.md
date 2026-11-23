# Qwen3-ASR-wgpu Apple Silicon Optimization Analysis Report

## Project Overview

This is a Rust and wgpu-based Qwen3-ASR speech recognition inference engine that uses GPU acceleration for cross-platform audio-to-text conversion.

### Current Architecture
- **Language**: Rust (1.95.0)
- **GPU Abstraction**: wgpu 30.0.1
- **Audio Processing**: CPU fallback + GPU encoder
- **Models**: Qwen3-ASR 0.6B / 1.7B
- **Quantization**: INT8 support
- **Parallelization**: rayon + gemm (CPU), wgpu compute shaders (GPU)

## Apple Silicon (M-series chips) Specific Issues

### 1. Metal Backend Under-optimized

**Current State**:
```rust
// Runtime priority in gpu.rs
const API_RANK_ORDER: [wgpu::Backend; 5] = [
    wgpu::Backend::Vulkan,  // Priority 0
    wgpu::Backend::Metal,   // Priority 1 ⚠️
    wgpu::Backend::Dx12,
    wgpu::Backend::Gl,
    wgpu::Backend::BrowserWebGpu,
];
```

**Issue**: Metal is the native API on macOS, but has lower priority than Vulkan (via MoltenVK). MoltenVK adds translation overhead.

### 2. Unified Memory Architecture (UMA) Not Fully Utilized

Apple Silicon's CPU and GPU share unified memory. Current implementation:
- Audio encoder defaults to CPU (`EncoderBackend::Cpu`)
- CPU↔GPU data transfer still uses standard wgpu buffer mapping
- Doesn't leverage zero-copy shared memory advantages

### 3. AMX (Apple Matrix Coprocessor) Not Activated

Apple Silicon has dedicated AMX matrix coprocessor, but current `gemm` crate and CPU implementation don't directly utilize it:
```rust
// audio_encoder.rs
use gemm::{gemm, Parallelism};  // Generic GEMM, not AMX-optimized
```

### 4. Audio Resampling Performance

```rust
// mel.rs
const SOXR_BLOCK: usize = 16384;  // Fixed block size
```
Uses external C library soxr, not optimized for M-chip NEON vectorization.

## Optimization Recommendations

### Priority 1: Metal Priority Adjustment

**Impact**: 🔥🔥🔥 High  
**Complexity**: 🟢 Low

Prioritize Metal on macOS to avoid MoltenVK translation overhead.

### Priority 2: Leverage Unified Memory Architecture

**Impact**: 🔥🔥🔥 High  
**Complexity**: 🟡 Medium

Use Metal's shared storage mode for zero-copy data transfer.

### Priority 3: Enable Accelerate Framework

**Impact**: 🔥🔥 Medium  
**Complexity**: 🟢 Low

Use Apple Accelerate.framework instead of generic GEMM, automatically utilizing AMX.

### Priority 4: Compiler Optimization Flags

**Impact**: 🔥🔥 Medium  
**Complexity**: 🟢 Low

Add Apple Silicon specific compiler optimizations.

### Priority 5: Shader Optimization

**Impact**: 🔥 Low to Medium  
**Complexity**: 🔴 High

Adjust compute shader workgroup sizes for Apple GPU architecture.

## Performance Expectations

Expected performance improvements after optimization:
- **Latency**: -15% to -30%
- **Throughput**: +20% to +40%
- **Memory Bandwidth Utilization**: +30% to +50%
- **Power Consumption**: -20% to -35%

## Implementation Plan

1. ✅ Modify Metal priority and device selection logic
2. ✅ Add Apple-specific compiler optimizations
3. ✅ Integrate Accelerate.framework
4. ✅ Optimize unified memory usage
5. ⏳ Performance validation and benchmarking

