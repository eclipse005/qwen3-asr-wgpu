# ⚡ 快速开始 - Apple Silicon 优化版

## 🚀 三步开始

```bash
# 1️⃣ 验证优化
./verify_optimizations.sh

# 2️⃣ 构建项目
cargo build --release

# 3️⃣ 运行转录
./target/release/transcribe \
  --model ./Qwen3-ASR-0.6B-hf \
  --wav audio.wav
```

## 📥 下载模型

```bash
# 从 Hugging Face 下载（需要 git-lfs）
git clone https://huggingface.co/Qwen/Qwen3-ASR-0.6B

# 或者手动下载
# https://huggingface.co/Qwen/Qwen3-ASR-0.6B
```

## 🎯 常用命令

### 单文件转录
```bash
./target/release/transcribe \
  --model ./Qwen3-ASR-0.6B-hf \
  --wav audio.wav
```

### 批量转录（推荐）
```bash
./target/release/transcribe \
  --model ./Qwen3-ASR-0.6B-hf \
  --batch 8 \
  --wav audio1.wav audio2.wav audio3.wav \
  --out results.txt
```

### 性能分析
```bash
QWEN3_ENC_PROFILE=1 ./target/release/transcribe \
  --model ./Qwen3-ASR-0.6B-hf \
  --wav audio.wav \
  --timings
```

### 强制使用 Metal
```bash
./target/release/transcribe \
  --device metal \
  --model ./Qwen3-ASR-0.6B-hf \
  --wav audio.wav
```

### 查看可用设备
```bash
./target/release/transcribe --list-devices
```

## 📊 性能基准测试

```bash
./benchmark.sh ./Qwen3-ASR-0.6B-hf audio.wav
```

## 🎯 预期性能

| M 芯片 | 批量 | RTF | 说明 |
|--------|------|-----|------|
| M1 | 8 | ~0.15 | 实时率 |
| M1 Pro/Max | 16 | ~0.10 | 更快 |
| M2/M3 | 12 | ~0.12 | 高效 |

> RTF < 1.0 表示比实时更快

## 💡 优化要点

✅ **使用批量模式** - `--batch 8` 或更高  
✅ **连接电源** - 获得最佳性能  
✅ **Metal 优先** - 自动使用原生 API  
✅ **多核利用** - Accelerate 自动优化  

## 📚 详细文档

| 文档 | 说明 |
|------|------|
| [优化完成报告.md](优化完成报告.md) | 完整总结 |
| [OPTIMIZATION_SUMMARY_CN.md](OPTIMIZATION_SUMMARY_CN.md) | 详细指南 |
| [README_OPTIMIZATIONS.md](README_OPTIMIZATIONS.md) | 使用说明 |

## 🆘 常见问题

**Q: Metal 未被选中？**  
A: 使用 `--device metal` 强制

**Q: 性能未提升？**  
A: 检查电源模式，连接电源

**Q: 内存不足？**  
A: 减小 `--batch` 或使用 INT8 模型

## ✨ 一键运行完整测试

```bash
# 验证 → 构建 → 测试
./verify_optimizations.sh && \
cargo build --release && \
./target/release/transcribe --list-devices
```

---

**🎉 享受更快的语音识别！**
