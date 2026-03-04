#!/bin/bash
# M4 Pro 训练监控脚本

echo "=== M4 Pro 训练监控 ==="
echo ""

# 1. GPU 利用率
echo "📊 GPU 活跃进程:"
ps aux | grep -E "python|train" | grep -v grep | head -5

echo ""
echo "⚡️ 实时功耗 (每5秒更新，Ctrl+C 退出):"
echo ""

# 监控功耗
while true; do
    # 获取 GPU 功耗 (mW)
    power=$(sudo powermetrics --samplers gpu_power -i 1000 -n 1 | grep "GPU Power" | awk '{print $3}')
    
    # 获取活跃核心
    stats=$(sudo powermetrics --samplers gpu_stats -i 1000 -n 1 | grep -A 2 "GPU Statistics" | tail -2)
    
    echo "$(date '+%H:%M:%S') | 功耗: ${power}mW"
    echo "$stats"
    echo "---"
    sleep 5
done
