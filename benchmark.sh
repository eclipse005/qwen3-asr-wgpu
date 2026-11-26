#!/bin/bash
# 性能基准测试脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🎯 Qwen3-ASR-wgpu 性能基准测试${NC}"
echo "========================================"
echo ""

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <模型路径> <音频文件1> [音频文件2] ..."
    echo ""
    echo "示例:"
    echo "  $0 ./Qwen3-ASR-0.6B-hf audio.wav"
    echo "  $0 ./Qwen3-ASR-0.6B-hf audio1.wav audio2.wav audio3.wav"
    exit 1
fi

MODEL=$1
shift
AUDIO_FILES=("$@")

# 验证模型路径
if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}❌ 模型目录不存在: $MODEL${NC}"
    exit 1
fi

# 验证音频文件
for audio in "${AUDIO_FILES[@]}"; do
    if [ ! -f "$audio" ]; then
        echo -e "${YELLOW}❌ 音频文件不存在: $audio${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✅ 模型: $MODEL${NC}"
echo -e "${GREEN}✅ 音频文件: ${#AUDIO_FILES[@]} 个${NC}"
echo ""

# 确保已编译
if [ ! -f "target/release/transcribe" ]; then
    echo "正在构建 Release 版本..."
    cargo build --release
    echo ""
fi

# 获取系统信息
echo "📊 系统信息"
echo "----------------------------------------"
echo "  架构: $(uname -m)"
echo "  系统: $(uname -s) $(sw_vers -productVersion 2>/dev/null || echo '')"
echo "  Rust: $(rustc --version)"
if [ -x "$(command -v sysctl)" ]; then
    echo "  CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Unknown')"
    echo "  核心: $(sysctl -n hw.ncpu 2>/dev/null || echo 'Unknown')"
    TOTAL_RAM=$(sysctl -n hw.memsize 2>/dev/null)
    if [ -n "$TOTAL_RAM" ]; then
        echo "  内存: $((TOTAL_RAM / 1024 / 1024 / 1024)) GB"
    fi
fi
echo ""

# 显示可用设备
echo "🖥️  可用设备"
echo "----------------------------------------"
./target/release/transcribe --list-devices 2>/dev/null || echo "无法列出设备"
echo ""

# 测试函数
run_test() {
    local test_name=$1
    local batch_size=$2
    local device=$3
    local extra_args=$4
    
    echo -e "${BLUE}📝 测试: $test_name${NC}"
    echo "  批量大小: $batch_size"
    echo "  设备: $device"
    
    local start_time=$(date +%s.%N)
    
    if ./target/release/transcribe \
        --model "$MODEL" \
        --device "$device" \
        --batch "$batch_size" \
        $extra_args \
        --wav "${AUDIO_FILES[@]}" \
        --out /tmp/transcribe_output_$$.txt \
        > /tmp/transcribe_log_$$.txt 2>&1; then
        
        local end_time=$(date +%s.%N)
        local elapsed=$(echo "$end_time - $start_time" | bc)
        
        echo -e "  ${GREEN}✅ 完成${NC}"
        echo "  时间: ${elapsed}s"
        
        # 尝试提取实时率（如果日志中有）
        if grep -q "RTF" /tmp/transcribe_log_$$.txt; then
            local rtf=$(grep "RTF" /tmp/transcribe_log_$$.txt | head -1)
            echo "  $rtf"
        fi
        
        # 显示一些结果
        if [ -f /tmp/transcribe_output_$$.txt ]; then
            local line_count=$(wc -l < /tmp/transcribe_output_$$.txt)
            echo "  输出行数: $line_count"
        fi
        
        rm -f /tmp/transcribe_output_$$.txt /tmp/transcribe_log_$$.txt
        echo ""
        return 0
    else
        echo -e "  ${YELLOW}❌ 失败${NC}"
        if [ -f /tmp/transcribe_log_$$.txt ]; then
            echo "  错误日志:"
            tail -5 /tmp/transcribe_log_$$.txt | sed 's/^/    /'
        fi
        rm -f /tmp/transcribe_output_$$.txt /tmp/transcribe_log_$$.txt
        echo ""
        return 1
    fi
}

# 开始测试
echo "🚀 开始基准测试"
echo "========================================"
echo ""

# 测试 1: Auto 后端，单个
run_test "Auto 后端 - 单个处理" 1 "auto" ""

# 测试 2: Auto 后端，批量 4
if [ ${#AUDIO_FILES[@]} -ge 4 ]; then
    run_test "Auto 后端 - 批量 4" 4 "auto" ""
fi

# 测试 3: Auto 后端，批量 8
if [ ${#AUDIO_FILES[@]} -ge 8 ]; then
    run_test "Auto 后端 - 批量 8" 8 "auto" ""
fi

# 测试 4: 强制 Metal（如果在 macOS 上）
if [ "$(uname -s)" = "Darwin" ]; then
    run_test "Metal 后端 - 批量 1" 1 "metal" ""
    
    if [ ${#AUDIO_FILES[@]} -ge 8 ]; then
        run_test "Metal 后端 - 批量 8" 8 "metal" ""
    fi
fi

# 测试 5: 带性能分析
echo -e "${BLUE}📊 性能分析运行${NC}"
echo "----------------------------------------"
QWEN3_ENC_PROFILE=1 ./target/release/transcribe \
    --model "$MODEL" \
    --batch 1 \
    --wav "${AUDIO_FILES[0]}" \
    --timings \
    > /tmp/profile_$$.txt 2>&1 || true

if [ -f /tmp/profile_$$.txt ]; then
    echo ""
    grep -A 20 "phase profile\|Timings:" /tmp/profile_$$.txt || true
    rm -f /tmp/profile_$$.txt
fi

echo ""
echo "✅ 基准测试完成！"
echo ""
echo "💡 提示:"
echo "  • 连接电源以获得最佳性能"
echo "  • 增加批量大小可以提高吞吐量"
echo "  • 首次运行可能较慢（GPU 预热）"
echo "  • 查看 OPTIMIZATION_SUMMARY_CN.md 了解更多优化建议"
