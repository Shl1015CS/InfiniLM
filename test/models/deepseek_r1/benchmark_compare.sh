#!/bin/bash
# DeepSeek-R1 MLA 性能对比脚本

echo "=================================="
echo "DeepSeek-R1 MLA 性能对比"
echo "=================================="
echo ""

# 检测设备
DEVICE=""
if [ "$1" == "--nvidia" ] || [ "$1" == "" ]; then
    DEVICE="--nvidia"
    echo "设备: NVIDIA GPU"
elif [ "$1" == "--moore" ]; then
    DEVICE="--moore"
    echo "设备: Moore 摩尔线程"
elif [ "$1" == "--iluvatar" ]; then
    DEVICE="--iluvatar"
    echo "设备: Iluvatar 天数智芯"
elif [ "$1" == "--cpu" ]; then
    DEVICE="--cpu"
    echo "设备: CPU"
else
    DEVICE="$1"
    echo "设备: $1"
fi

echo ""
echo "=================================="
echo "1. 运行 PyTorch 版本"
echo "=================================="
python attention_test.py $DEVICE > pytorch_results.txt 2>&1
echo "结果已保存到 pytorch_results.txt"
cat pytorch_results.txt | grep -A 5 "Test Summary"
echo ""

echo "=================================="
echo "2. 运行九齿框架版本"
echo "=================================="
python attention_test_ninetoothed.py $DEVICE > ninetoothed_results.txt 2>&1
echo "结果已保存到 ninetoothed_results.txt"
cat ninetoothed_results.txt | grep -A 5 "Test Summary"
echo ""

echo "=================================="
echo "3. 性能对比"
echo "=================================="

# 提取性能数据
PT_PREFILL=$(grep "Prefill Latency" pytorch_results.txt | awk '{print $5}')
PT_DECODE=$(grep "Decode Throughput" pytorch_results.txt | awk '{print $5}')

NT_PREFILL=$(grep "Prefill Latency" ninetoothed_results.txt | awk '{print $5}')
NT_DECODE=$(grep "Decode Throughput" ninetoothed_results.txt | awk '{print $5}')

echo "预填充延迟:"
echo "  PyTorch:       $PT_PREFILL ms"
echo "  Ninetoothed:   $NT_PREFILL ms"
echo ""

echo "解码吞吐量:"
echo "  PyTorch:       $PT_DECODE tok/s"
echo "  Ninetoothed:   $NT_DECODE tok/s"
echo ""

echo "=================================="
echo "完成！"
echo "=================================="
echo ""
echo "详细结果:"
echo "  - pytorch_results.txt"
echo "  - ninetoothed_results.txt"
