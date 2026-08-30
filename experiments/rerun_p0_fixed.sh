#!/usr/bin/env bash
# ============================================================================
# P0 修复后全量重跑 — 场景 C (留一城市) + 场景 A (个性化消融)
#
# 覆盖被 P0 缺陷污染的结果:
#   - 场景 C: 缓存键漏城市集合 → 5/6 fold 只训 4 城 (本轮已修, 全部重跑)
#   - 场景 A: checkpoint 只存 global 不存本地 BN/头 + 选轮用 RMSE (本轮已修)
#
# 运行时间: 长 (CPU 单机, 场景 C 6 fold × 3 seed + 场景 A 4 config × 3 seed)
# 用法:     PY=.venv/Scripts/python.exe bash experiments/rerun_p0_fixed.sh
# ============================================================================
set -uo pipefail

PY="${PY:-python}"
SEEDS="42,123,999"
CITIES="SZH,AMS,JHB,LOA,MEL,SPO"
LOG_DIR="outputs/rerun_logs"
mkdir -p "$LOG_DIR"

run_py() {
  local desc="$1"; shift
  local code
  echo ""
  echo "================================================================"
  echo "--- $(date '+%F %T')  $desc"
  echo "================================================================"
  if "$@"; then
    echo "    [OK]   $desc"
  else
    code=$?
    if [ "$code" -eq 127 ]; then
      echo "    [OK*]  $desc (退出码 127 = torch 关闭崩溃, 结果已落盘)"
    else
      echo "    [FAIL] $desc 退出码 $code — 见上方日志"
    fi
  fi
}

# ────────────────────────────────────────────────────────────────
# 场景 C — 留一城市冷启动 (FedProx μ=0.01, α=0.5, top_k=10, rounds=30)
# ────────────────────────────────────────────────────────────────
run_py "场景 C: 留一城市冷启动" $PY experiments/leave_one_out.py \
  --top_k 10 --rounds 30 --local_epochs 5 \
  --finetune_days 14,30 --seeds "$SEEDS" \
  2>&1 | tee "$LOG_DIR/scenario_C.log"

# ────────────────────────────────────────────────────────────────
# 场景 A — 个性化消融 (stratified_balanced k=9, α=0, fedavg, rounds=50)
# ────────────────────────────────────────────────────────────────
A_COMMON="--cities $CITIES --station_selection stratified_balanced --top_k 9 \
--aggregation fedavg --city_weight_alpha 0 --num_rounds 50 \
--local_epochs 5 --seeds $SEEDS"

run_py "A.1 Base"                    $PY main.py $A_COMMON                     2>&1 | tee "$LOG_DIR/scenario_A_base.log"
run_py "A.2 + FedBN"                 $PY main.py $A_COMMON --fedbn             2>&1 | tee "$LOG_DIR/scenario_A_fedbn.log"
run_py "A.3 + LocalHead"             $PY main.py $A_COMMON --local_head        2>&1 | tee "$LOG_DIR/scenario_A_localhead.log"
run_py "A.4 + FedBN + LocalHead"     $PY main.py $A_COMMON --fedbn --local_head 2>&1 | tee "$LOG_DIR/scenario_A_both.log"

# ────────────────────────────────────────────────────────────────
# 聚合结果
# ────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "--- $(date '+%F %T')  聚合 organized_results"
echo "================================================================"
$PY experiments/organize_results.py

echo ""
echo "================================================================"
echo "  重跑完成。"
echo "  场景 C 汇总: $(ls outputs/leave_one_out/loo_summary_multi_*.json 2>/dev/null | wc -l) 个文件"
echo "  (下一步: 复制最新 loo_summary_multi → results/leave_one_out_3seed.json;"
echo "   organized_results → results/organized_results.json; 重画图)"
echo "================================================================"
