#!/usr/bin/env bash
# 编译并运行 legacy API 卷积 demo，同时保留 cuDNN 调用日志；可传 --all / --find。
set -euo pipefail

NV=/home/liquan/venvs/torch/lib/python3.14/site-packages/nvidia
CUDA_INC=/usr/local/cuda/include
MODE="${1:-}"

g++ -O2 -std=c++17 -Wall -Wextra legacy_conv_backend.cpp -o legacy_conv_backend \
  -I"$NV/cudnn/include" -I"$CUDA_INC" \
  "$NV/cudnn/lib/libcudnn.so.9" "$NV/cu13/lib/libcudart.so.13" \
  -Wl,-rpath,"$NV/cudnn/lib" -Wl,-rpath,"$NV/cu13/lib"

LOG="legacy_conv${MODE:+_${MODE#--}}.log"
CUDNN_LOGLEVEL_DBG=3 CUDNN_LOGDEST_DBG="./$LOG" ./legacy_conv_backend $MODE

echo
echo "== 从 cuDNN 日志复核调用次数（入口记录口径，见文档 §8.1）=="
python3 ../count_calls.py "$LOG"
echo "== 日志里有没有公开 graph API 入口 =="
grep -c "function cudnnBackend\(CreateDescriptor\|SetAttribute\|GetAttribute\|Finalize\|DestroyDescriptor\)" "$LOG" || true
