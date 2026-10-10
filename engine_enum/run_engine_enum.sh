#!/usr/bin/env bash
# 复现 §11.9 的全部输出：engine 普查表、候选表、kernel 抓取。
set -euo pipefail
cd "$(dirname "$0")"
NV=/home/liquan/venvs/torch/lib/python3.14/site-packages/nvidia

g++ -O2 -std=c++17 -Wall -Wextra engine_enum.cpp -o engine_enum \
  -I"$NV/cudnn/include" -I/usr/local/cuda/include \
  "$NV/cudnn/lib/libcudnn.so.9" "$NV/cu13/lib/libcudart.so.13" \
  -Wl,-rpath,"$NV/cudnn/lib" -Wl,-rpath,"$NV/cu13/lib"

for mode in fwd bwd_data bwd_filter; do
  CUDNN_LOGLEVEL_DBG=3 CUDNN_LOGDEST_DBG="./engine_enum_${mode}.log" ./engine_enum "$mode" > "engine_enum_${mode}.txt"
done

g++ -O2 -std=c++17 -fPIC -shared runner.cpp -o librunner.so \
  -I"$NV/cudnn/include" -I/usr/local/cuda/include \
  "$NV/cudnn/lib/libcudnn.so.9" "$NV/cu13/lib/libcudart.so.13" \
  -Wl,-rpath,"$NV/cudnn/lib" -Wl,-rpath,"$NV/cu13/lib"

/home/liquan/venvs/torch/bin/python trace_engines.py 0 17 20 > trace_all.txt
grep "^cand " trace_all.txt

# 单个全局 engine：precompiled(28/0/6) 与 runtime 编译(71) 的 plan 耗时 / kernel 对照
/home/liquan/venvs/torch/bin/python trace_engine_global.py 28 0 6 71

# ---- §11.12：旋钮（knob）----
g++ -O2 -std=c++17 knob_info.cpp -o knob_info \
  -I"$NV/cudnn/include" -I/usr/local/cuda/include \
  "$NV/cudnn/lib/libcudnn.so.9" "$NV/cu13/lib/libcudart.so.13" \
  -Wl,-rpath,"$NV/cudnn/lib" -Wl,-rpath,"$NV/cu13/lib"
for mode in fwd bwd_data bwd_filter; do
  ./knob_info "$mode" > "knob_info_${mode}.txt"
done

# 4 种 heuristic 模式的候选对照
for hm in 0 1 2 3; do
  echo "########## HEUR_MODE=$hm ##########"
  HEUR_MODE=$hm SKIP_ENUM=1 ./engine_enum fwd 2>/dev/null | sed -n '/heuristic（mode/,$p'
done > engine_enum_modes_fwd.txt

# 手工旋钮网格实测（含无效点报错）与稳定计时
/home/liquan/venvs/torch/bin/python knob_sweep.py > /dev/null
/home/liquan/venvs/torch/bin/python knob_bench.py 2>/dev/null

# ---- §11.13：heuristic 的 CPU 代价 ----
./run_heur_cost.sh > /dev/null
