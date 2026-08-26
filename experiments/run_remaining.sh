#!/usr/bin/env bash
# ============================================================================
# 续跑脚本 — 跳过已完成的场景 A.1 (FedAvg base, 已完成 3 seeds)
#   继续: A.2/A.3/A.4 (个性化消融) + 场景 B + 场景 C + 基线
#
# 用法 (项目根目录):
#   PY=.venv/Scripts/python.exe bash experiments/run_remaining.sh
# ============================================================================
set -euo pipefail

PY="${PY:-python}"
SEEDS="42,123,999"
CITIES="SZH,AMS,JHB,LOA,MEL,SPO"
ROUNDS=50
LOCAL_EPOCHS=5

echo "=============================================================="
echo "  续跑: 场景 A.2-A.4 + B + C + 基线  (跳过 A.1)"
echo "=============================================================="

# ────────────────────────────────────────────────────────────────
# 场景 A — 个性化消融 (已跳过 A.1 Base; 跑 A.2/A.3/A.4)
# ────────────────────────────────────────────────────────────────
A_COMMON="--cities $CITIES --station_selection stratified_balanced --top_k 9 \
--aggregation fedavg --city_weight_alpha 0 --num_rounds $ROUNDS \
--local_epochs $LOCAL_EPOCHS --seeds $SEEDS"

echo "--- A.2 + FedBN ---"
$PY main.py $A_COMMON --fedbn

echo "--- A.3 + LocalHead ---"
$PY main.py $A_COMMON --local_head

echo "--- A.4 + FedBN + LocalHead ---"
$PY main.py $A_COMMON --fedbn --local_head

# ────────────────────────────────────────────────────────────────
# 场景 B — 城市权重指数扫描 (比例分配, 共 60 站, α ∈ {0,0.5,1})
# ────────────────────────────────────────────────────────────────
B_COMMON="--cities $CITIES --station_selection proportional --top_k 60 \
--aggregation fedavg --num_rounds $ROUNDS --local_epochs $LOCAL_EPOCHS \
--seeds $SEEDS"

for A in 0 0.5 1.0; do
  echo "--- B.α=$A ---"
  $PY main.py $B_COMMON --city_weight_alpha $A
done

# ────────────────────────────────────────────────────────────────
# 场景 C — 留一城市冷启动 (FedProx μ=0.01, α=0.5, top_k=10, rounds=30)
# ────────────────────────────────────────────────────────────────
echo "--- 场景 C: 留一城市冷启动 ---"
$PY experiments/leave_one_out.py \
  --top_k 10 --rounds 30 --local_epochs 5 \
  --finetune_days 14,30 --seeds $SEEDS

# ────────────────────────────────────────────────────────────────
# 基线 (基线脚本仅支持 --seed, 故按种子循环)
# ────────────────────────────────────────────────────────────────
echo "--- centralized_personalized (多城市, 分层平衡 9 站/城) ---"
for S in 42 123 999; do
  $PY experiments/baseline_centralized_personalized.py \
    --cities $CITIES --station_selection stratified_balanced --top_k 9 \
    --epochs 100 --seed $S
done

echo "--- seasonal_naive / local_only / centralized_shared (单城市, 每城 top_k=9) ---"
for CITY in SZH AMS JHB LOA MEL SPO; do
  for S in 42 123 999; do
    $PY experiments/baseline_seasonal_naive.py --city $CITY --top_k 9 --seed $S
    $PY experiments/baseline_local.py        --city $CITY --top_k 9 --epochs 100 --seed $S
    $PY experiments/baseline_centralized.py  --city $CITY --top_k 9 --epochs 100 --seed $S
  done
done

echo ""
echo "=============================================================="
echo "  续跑完成。聚合:  $PY experiments/organize_results.py"
echo "  场景 C: outputs/leave_one_out/loo_summary_multi_*.json"
echo "=============================================================="
