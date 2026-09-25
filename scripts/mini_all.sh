#!/bin/bash
# Mini-set, all eight arms, strictly serial (one GPU job at a time), in the
# owner's order: python-cuda originals + int8 first, then wgpu-vulkan.
# Logs land in $OUT/log_<arm>.txt; hyps in $OUT/hyps_<arm>.json.
set -u
PY=/c/Users/ADMIN/miniconda3/envs/myenv/python.exe
ROOT=D:/mini_asr_data
OUT=$ROOT/results_all
MODELS=D:/Qwen3-ASR/models
EVAL=D:/Qwen3-ASR/scripts/mini_eval.py
RUN=D:/qwen3-asr-wgpu/scripts/mini_run.py
mkdir -p "$OUT"

echo "=== [1/8] python cuda 0.6B fp16 (original) $(date +%H:%M:%S) ==="
"$PY" "$EVAL" --root "$ROOT" --out "$OUT/py06" --backends fp16 \
  --fp16 "$MODELS/Qwen3-ASR-0.6B-hf" --max-new 256 > "$OUT/log_py06fp16.txt" 2>&1
cp "$OUT/py06/hyps_fp16.json" "$OUT/hyps_py06fp16.json"

echo "=== [2/8] python cuda 0.6B int8 (quantized) $(date +%H:%M:%S) ==="
"$PY" "$EVAL" --root "$ROOT" --out "$OUT/py06" --backends int8 \
  --fp16 "$MODELS/Qwen3-ASR-0.6B-hf" --int8 "$MODELS/Qwen3-ASR-0.6B-int8" --max-new 256 \
  > "$OUT/log_py06int8.txt" 2>&1
cp "$OUT/py06/hyps_int8.json" "$OUT/hyps_py06int8.json"

echo "=== [3/8] python cuda 1.7B fp16 (original) $(date +%H:%M:%S) ==="
"$PY" "$EVAL" --root "$ROOT" --out "$OUT/py17" --backends fp16 \
  --fp16 "$MODELS/Qwen3-ASR-1.7B-hf" --max-new 256 > "$OUT/log_py17fp16.txt" 2>&1
cp "$OUT/py17/hyps_fp16.json" "$OUT/hyps_py17fp16.json"

echo "=== [4/8] python cuda 1.7B int8 (quantized) $(date +%H:%M:%S) ==="
"$PY" "$EVAL" --root "$ROOT" --out "$OUT/py17" --backends int8 \
  --fp16 "$MODELS/Qwen3-ASR-1.7B-hf" --int8 "$MODELS/Qwen3-ASR-1.7B-int8" --max-new 256 \
  > "$OUT/log_py17int8.txt" 2>&1
cp "$OUT/py17/hyps_int8.json" "$OUT/hyps_py17int8.json"

echo "=== [5/8] wgpu vulkan 0.6B fp16 (original) $(date +%H:%M:%S) ==="
"$PY" "$RUN" --arm wgpu06fp16 --model "$MODELS/Qwen3-ASR-0.6B-hf" \
  --out "$OUT" --device vulkan --max-new 256 > "$OUT/log_wgpu06fp16.txt" 2>&1

echo "=== [6/8] wgpu vulkan 0.6B int8 (quantized) $(date +%H:%M:%S) ==="
"$PY" "$RUN" --arm wgpu06int8 --model "$MODELS/Qwen3-ASR-0.6B-int8" \
  --out "$OUT" --device vulkan --max-new 256 > "$OUT/log_wgpu06int8.txt" 2>&1

echo "=== [7/8] wgpu vulkan 1.7B fp16 (original) $(date +%H:%M:%S) ==="
"$PY" "$RUN" --arm wgpu17fp16 --model "$MODELS/Qwen3-ASR-1.7B-hf" \
  --out "$OUT" --device vulkan --max-new 256 > "$OUT/log_wgpu17fp16.txt" 2>&1

echo "=== [8/8] wgpu vulkan 1.7B int8 (quantized) $(date +%H:%M:%S) ==="
"$PY" "$RUN" --arm wgpu17int8 --model "$MODELS/Qwen3-ASR-1.7B-int8" \
  --out "$OUT" --device vulkan --max-new 256 > "$OUT/log_wgpu17int8.txt" 2>&1

echo "=== ALL ARMS DONE $(date +%H:%M:%S) ==="
"$PY" D:/qwen3-asr-wgpu/scripts/mini_report.py --out "$OUT"
