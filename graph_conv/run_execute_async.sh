#!/usr/bin/env bash
# 编译并运行"execute 只是下发"的计时 demo（大卷积，kernel 有可测量的执行时间）。
set -euo pipefail
NV=/home/liquan/venvs/torch/lib/python3.14/site-packages/nvidia
g++ -O2 -std=c++17 -Wall -Wextra execute_async_demo.cpp -o execute_async_demo \
  -I"$NV/cudnn/include" -I/usr/local/cuda/include \
  "$NV/cudnn/lib/libcudnn.so.9" "$NV/cu13/lib/libcudart.so.13" \
  -Wl,-rpath,"$NV/cudnn/lib" -Wl,-rpath,"$NV/cu13/lib"
./execute_async_demo
