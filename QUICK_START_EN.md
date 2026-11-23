# ⚡ Quick Start - Apple Silicon Optimized Edition

## 🚀 Three Steps to Start

```bash
# 1️⃣ Verify optimizations
./verify_optimizations.sh

# 2️⃣ Build project
cargo build --release

# 3️⃣ Run transcription
./target/release/transcribe \
  --model ./Qwen3-ASR-0.6B-hf \
  --wav audio.wav
```

## 📥 Download Model

```bash
# From Hugging Face (requires git-lfs)
git clone https://huggingface.co/Qwen/Qwen3-ASR-0.6B

# Or download manually
# https://huggingface.co/Qwen/Qwen3-ASR-0.6B
```

## 🎯 Common Commands

### Single File Transcription
```bash
./target/release/transcribe \
  --model ./Qwen3-ASR-0.6B-hf \
  --wav audio.wav
```

### Batch Transcription (Recommended)
```bash
./target/release/transcribe \
  --model ./Qwen3-ASR-0.6B-hf \
  --batch 8 \
  --wav audio1.wav audio2.wav audio3.wav \
  --out results.txt
```

### Performance Profiling
```bash
QWEN3_ENC_PROFILE=1 ./target/release/transcribe \
  --model ./Qwen3-ASR-0.6B-hf \
  --wav audio.wav \
  --timings
```

### Force Metal Usage
```bash
./target/release/transcribe \
  --device metal \
  --model ./Qwen3-ASR-0.6B-hf \
  --wav audio.wav
```

### List Available Devices
```bash
./target/release/transcribe --list-devices
```

## 📊 Performance Benchmarks

```bash
./benchmark.sh ./Qwen3-ASR-0.6B-hf audio.wav
```

## 🎯 Expected Performance

| M Chip | Batch | RTF | Description |
|--------|-------|-----|-------------|
| M1 | 8 | ~0.15 | Real-time factor |
| M1 Pro/Max | 16 | ~0.10 | Faster |
| M2/M3 | 12 | ~0.12 | Efficient |

> RTF < 1.0 means faster than real-time

## 💡 Optimization Keys

✅ **Use Batch Mode** - `--batch 8` or higher  
✅ **Connect Power** - Get best performance  
✅ **Metal Priority** - Automatically use native API  
✅ **Multi-core Utilization** - Accelerate auto-optimizes  

## 📚 Detailed Documentation

| Document | Description |
|----------|-------------|
| [OPTIMIZATION_SUMMARY_EN.md](OPTIMIZATION_SUMMARY_EN.md) | Complete guide (Recommended) |
| [APPLE_SILICON_OPTIMIZATIONS_EN.md](APPLE_SILICON_OPTIMIZATIONS_EN.md) | Technical details |
| [README_OPTIMIZATIONS_EN.md](README_OPTIMIZATIONS_EN.md) | Usage instructions |

## 🆘 Common Issues

**Q: Metal not selected?**  
A: Use `--device metal` to force

**Q: No performance improvement?**  
A: Check power mode, connect power

**Q: Out of memory?**  
A: Reduce `--batch` or use INT8 model

## ✨ One-Click Complete Test

```bash
# Verify → Build → Test
./verify_optimizations.sh && \
cargo build --release && \
./target/release/transcribe --list-devices
```

---

**🎉 Enjoy faster speech recognition!**
