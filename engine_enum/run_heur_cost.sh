#!/usr/bin/env bash
# 测 heuristic 各模式的 CPU 代价（每种形状×模式开新进程跑 3 次），输出 heur_cost.txt
set -euo pipefail
cd "$(dirname "$0")"
NV=/home/liquan/venvs/torch/lib/python3.14/site-packages/nvidia
g++ -O2 -std=c++17 heur_cost.cpp -o heur_cost \
  -I"$NV/cudnn/include" -I/usr/local/cuda/include \
  "$NV/cudnn/lib/libcudnn.so.9" "$NV/cu13/lib/libcudart.so.13" \
  -Wl,-rpath,"$NV/cudnn/lib" -Wl,-rpath,"$NV/cu13/lib"

shapes=("4 3 64 16" "32 256 56 256" "8 1024 14 1024")
for s in "${shapes[@]}"; do
  read -r n c hw k <<<"$s"
  for m in 0 1 2 3; do
    for r in 1 2 3; do
      RUNNER_N=$n RUNNER_C=$c RUNNER_HW=$hw RUNNER_K=$k ./heur_cost "$m" 4
    done
  done
done > heur_cost.txt
grep '^mode=' heur_cost.txt
