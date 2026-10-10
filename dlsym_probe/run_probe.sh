#!/usr/bin/env bash
# 编译 dlsym 探针并跑两个 demo，产出 *.dlsym.log（细节见 README.txt）
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NV=${NV:-/home/liquan/venvs/torch/lib/python3.14/site-packages/nvidia}
SO=${SO:-/tmp/dlsym_who.so}

gcc -shared -fPIC -O2 "$HERE/dlsym_who.c" -o "$SO" -ldl

for demo in "$HERE/../graph_conv/graph_conv_backend" "$HERE/../legacy_conv/legacy_conv_backend"; do
  name=$(basename "$demo" _backend)
  log="$HERE/${name}.dlsym.log"
  echo "== $demo -> $log"
  LD_PRELOAD="$SO" LD_LIBRARY_PATH="$NV/cudnn/lib:$NV/cu13/lib" "$demo" >/dev/null 2>"$log"
  grep -E "cuLaunchKernel|cuModuleLoad" "$log" | sed 's/.*\[dlsym\] //' | sort | uniq -c
done
