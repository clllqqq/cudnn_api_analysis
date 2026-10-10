#!/usr/bin/env bash
# 把 /home/liquan/cudnn_per_op.py 里的每一种情况各跑一遍（每次独立进程），
# 保存 cuDNN level-3 日志 + stdout，并汇总调用次数。
#
# 用法:
#   ./run_per_op_all.sh              # 跑全部 case
#   ./run_per_op_all.sh conv rnn     # 只跑指定 case
#   CUDNN_LOGLEVEL=2 ./run_per_op_all.sh   # 换日志级别（默认 3）
set -uo pipefail

PY=${PY:-/home/liquan/venvs/torch/bin/python}
TARGET=${TARGET:-/home/liquan/cudnn_per_op.py}
HEREDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTDIR=${OUTDIR:-$HEREDIR/logs_per_op}
LEVEL=${CUDNN_LOGLEVEL:-3}
COUNT_CALLS=$HEREDIR/count_calls.py

mkdir -p "$OUTDIR"

# 1) 从目标脚本里自动提取所有 case（形如: op == "conv"），保持与实际代码同步
mapfile -t DISCOVERED < <(grep -oP 'op\s*==\s*"\K[^"]+' "$TARGET")
if [ "${#DISCOVERED[@]}" -eq 0 ]; then
  DISCOVERED=(conv batchnorm rnn attention attention_forced grid_sample ctc)
fi

if [ "$#" -gt 0 ]; then
  CASES=("$@")
else
  CASES=("${DISCOVERED[@]}")
fi

echo "python : $PY"
echo "target : $TARGET"
echo "outdir : $OUTDIR   (CUDNN_LOGLEVEL_DBG=$LEVEL)"
echo "cases  : ${#CASES[@]} 个 —— ${CASES[*]}"
echo

declare -a ROW_OP=() ROW_RC=() ROW_TIME=() ROW_LOG=() ROW_OUT=()
FAILED=0

for op in "${CASES[@]}"; do
  log="$OUTDIR/${op}.log"
  out="$OUTDIR/${op}.stdout"
  : >"$log"          # cuDNN 日志是追加写，先清空保证只含本次运行
  : >"$out"

  start=$(date +%s.%N)
  CUDNN_LOGLEVEL_DBG="$LEVEL" CUDNN_LOGDEST_DBG="$log" \
    "$PY" "$TARGET" "$op" >"$out" 2>&1
  rc=$?
  end=$(date +%s.%N)
  elapsed=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.2f", b-a}')

  size=$(stat -c %s "$log")
  ok=$(grep -c '^ok$' "$out" || true)
  [ "$rc" -eq 0 ] && [ "$ok" -ge 1 ] && status="OK" || { status="FAIL"; FAILED=1; }
  printf '%-18s %-5s %6ss  log=%8s B  stdout=%s\n' \
    "$op" "$status" "$elapsed" "$size" "$(tr '\n' ' ' <"$out" | cut -c1-60)"

  ROW_OP+=("$op"); ROW_RC+=("$rc"); ROW_TIME+=("$elapsed")
  ROW_LOG+=("$size"); ROW_OUT+=("$out")
done

if [ "$#" -gt 0 ]; then
  SUMMARY=$OUTDIR/summary_subset.txt   # 子集运行不覆盖全量汇总
else
  SUMMARY=$OUTDIR/summary.txt
fi
{
  echo "== run_per_op_all.sh  $(date '+%F %T')"
  echo "target=$TARGET  CUDNN_LOGLEVEL_DBG=$LEVEL"
  echo
  printf '%-18s %-6s %8s %12s\n' case exit 用时s 日志字节
  for i in "${!ROW_OP[@]}"; do
    printf '%-18s %-6s %8s %12s\n' "${ROW_OP[$i]}" "${ROW_RC[$i]}" "${ROW_TIME[$i]}" "${ROW_LOG[$i]}"
  done
  echo
  echo "== 每份日志的 cuDNN 公开入口调用次数（口径见 README §8.1）"
} >"$SUMMARY"

for op in "${ROW_OP[@]}"; do
  {
    echo
    echo "---- $op ----"
    "$PY" "$COUNT_CALLS" "$OUTDIR/${op}.log" 2>/dev/null || echo "(日志为空或无入口记录)"
  } >>"$SUMMARY"
done

echo
echo "汇总 -> $SUMMARY"
cat "$SUMMARY"
exit "$FAILED"
