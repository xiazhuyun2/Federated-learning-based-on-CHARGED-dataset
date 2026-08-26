#!/usr/bin/env bash
# ============================================================================
# 全量重跑脚本 — 修复 P0 数据缺陷后重新产出可信实验数字
#
# 用法 (在项目根目录, PY 指向 venv python, 或已激活 venv 直接用 python):
#   PY=.venv/Scripts/python.exe bash experiments/run_all_experiments.sh
#   # 或
#   bash experiments/run_all_experiments.sh
#
# 可只跑某一段: 注释掉其余段落即可。建议先跑冒烟确认无误再全量。
#
# 重要: 修复前的 outputs/ 里是「有缺陷」的旧数字。重跑前请先清空或重命名
#   旧 outputs/ 目录, 避免 organize_results 混入旧 run:
#     mv outputs outputs_buggy_backup
#
# 所有实验默认写入 outputs/ (main.py / 基线 / leave_one_out 的默认目录),
# 与 organize_results.py 的默认读取目录一致, 故本脚本不显式传 --output_dir。
#
# 预计耗时 1–2 天 (RTX 4060 等 GPU)。
# ============================================================================
set -euo pipefail

PY="${PY:-python}"
SEEDS="42,123,999"
CITIES="SZH,AMS,JHB,LOA,MEL,SPO"
ROUNDS=50
LOCAL_EPOCHS=5

echo "=============================================================="
echo "  全量重跑: 场景 A / B / C + 基线"
echo "  种子: $SEEDS"
echo "=============================================================="

# ============================================================================
# 场景 A — 个性化消融 (分层平衡采样 stratified_balanced, 每城 9 站, FedAvg μ=0, α=0)
#   {Base, +FedBN, +LocalHead, +FedBN+LocalHead}
# ============================================================================
echo ""
echo "##### 场景 A: 个性化消融 #####"
A_COMMON="--cities $CITIES --station_selection stratified_balanced --top_k 9 \
--aggregation fedavg --city_weight_alpha 0 --num_rounds $ROUNDS \
--local_epochs $LOCAL_EPOCHS --seeds $SEEDS"

echo "--- A.1 Base (FedAvg) ---"
$PY main.py $A_COMMON

echo "--- A.2 + FedBN ---"
$PY main.py $A_COMMON --fedbn

echo "--- A.3 + LocalHead ---"
$PY main.py $A_COMMON --local_head

echo "--- A.4 + FedBN + LocalHead ---"
$PY main.py $A_COMMON --fedbn --local_head

# ============================================================================
# 场景 B — 城市权重指数扫描 (比例分配 proportional, 共 60 站, FedAvg μ=0)
#   α ∈ {0, 0.5, 1.0}
# ============================================================================
echo ""
echo "##### 场景 B: 城市权重 α 扫描 #####"
B_COMMON="--cities $CITIES --station_selection proportional --top_k 60 \
--aggregation fedavg --num_rounds $ROUNDS --local_epochs $LOCAL_EPOCHS \
--seeds $SEEDS"

for A in 0 0.5 1.0; do
  echo "--- B.α=$A ---"
  $PY main.py $B_COMMON --city_weight_alpha $A
done

# ============================================================================
# 场景 C — 留一城市冷启动 (FedProx μ=0.01, α=0.5, top_k=10, rounds=30)
#   脚本内部已实现 strict zero-shot (5 城共享 scaler) + calibrated zero-shot,
#   紧邻测试期的 few-shot / from-scratch, 以及 full-local; 报告 14/30 天 few-shot。
#   结果写 outputs/leave_one_out/ (organize_results 跳过该目录, 由脚本自身聚合)。
# ============================================================================
echo ""
echo "##### 场景 C: 留一城市冷启动 #####"
$PY experiments/leave_one_out.py \
  --top_k 10 --rounds 30 --local_epochs 5 \
  --finetune_days 14,30 --seeds $SEEDS

# ============================================================================
# 基线 (基线脚本仅支持 --seed, 不支持 --seeds / --output_dir, 故按种子循环)
#   centralized_personalized: 多城市, 分层平衡 9 站/城 (与场景 A 同站点集)
#   seasonal_naive / local_only / centralized_shared: 单城市, 每城 top_k=9
#   ⚠ 基线口径 (top_k/选站策略 与 Macro-City 聚合方式) 在 Phase 6 统一核对。
# ============================================================================
echo ""
echo "##### 基线 #####"

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
echo "  重跑完成。重新聚合并查看结果:"
echo "    $PY experiments/organize_results.py"
echo "  场景 C 结果: outputs/leave_one_out/loo_summary_multi_*.json"
echo "=============================================================="
