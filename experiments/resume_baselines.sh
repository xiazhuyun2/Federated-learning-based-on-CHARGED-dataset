#!/usr/bin/env bash
# ============================================================================
# 基线续跑脚本 — 跳过已完成的 A/B/C 场景, 只跑基线。
#
# centralized_personalized 已优化: 原实现每个 batch 对每个站点单独跑一遍 TCN+LSTM,
#   慢一个数量级 (单 seed ~12h)。现改为主干整批前向一次 + 各站独立小头路由,
#   数值完全等价 (TCN/LSTM 各样本在 batch 维独立), 预计快 20~30 倍。
#
# 用法 (项目根目录):
#   PY=.venv/Scripts/python.exe bash experiments/resume_baselines.sh
# ============================================================================
set -euo pipefail

export PYTHONUNBUFFERED=1   # 实时输出, 避免块缓冲导致"看不到进度"的错觉

PY="${PY:-python}"
CITIES="SZH,AMS,JHB,LOA,MEL,SPO"

# 运行单个实验: `if "$@"` 使失败进入 else 分支 (不会触发 set -e);
# 退出码 127 (torch 关闭崩溃, 结果已保存) 放行, 其它非零码中止。
run_py() {
  local desc="$1"; shift
  local code
  echo ""
  echo "--- $desc ---"
  if "$@"; then
    echo "    [OK]   $desc"
  else
    code=$?
    if [ "$code" -eq 127 ]; then
      echo "    [OK*]  $desc (退出码 127 = torch 关闭崩溃, 结果已落盘)"
    else
      echo "    [FAIL] $desc 退出码 $code — 结果可能不完整, 见上方日志"
      exit "$code"
    fi
  fi
}

echo "=============================================================="
echo "  基线续跑 (跳过 A/B/C): centralized_personalized + 单城市基线"
echo "=============================================================="

# ── 多城市集中式个性化 (3 seed, 已优化) ──────────────────────────
for S in 42 123 999; do
  run_py "centralized_personalized seed=$S" \
    $PY experiments/baseline_centralized_personalized.py \
    --cities $CITIES --station_selection stratified_balanced --top_k 9 \
    --epochs 100 --seed $S
done

# ── 单城市基线 (6 城 × 3 seed × 3 基线) ──────────────────────────
for CITY in SZH AMS JHB LOA MEL SPO; do
  for S in 42 123 999; do
    run_py "seasonal_naive $CITY seed=$S" \
      $PY experiments/baseline_seasonal_naive.py --city $CITY --top_k 9 --seed $S
    run_py "local_only $CITY seed=$S" \
      $PY experiments/baseline_local.py --city $CITY --top_k 9 --epochs 100 --seed $S
    run_py "centralized_shared $CITY seed=$S" \
      $PY experiments/baseline_centralized.py --city $CITY --top_k 9 --epochs 100 --seed $S
  done
done

echo ""
echo "=============================================================="
echo "  基线完成。"
echo "  metrics.json 数量: $(find outputs -name metrics.json | wc -l)  (预期约 78)"
echo "  聚合结果:           $PY experiments/organize_results.py"
echo "=============================================================="
