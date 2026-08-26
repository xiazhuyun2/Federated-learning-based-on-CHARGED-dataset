#!/usr/bin/env bash
# ============================================================================
# 续跑脚本 v2 — 跳过已完成的 A.1 (FedAvg base) 与 A.2 (FedBN)
#   继续: A.3 / A.4 (个性化消融) + 场景 B + 场景 C + 基线
#
# 为什么重写: 上一版 run_remaining.sh 在 A.2 跑完、A.3 开始前以退出码 127 停掉。
#   原因: main.py 多 seed 训练结束后, torch+CUDA 在解释器关闭阶段偶发崩溃
#   (Windows exit 127 = ERROR_PROC_NOT_FOUND), 而脚本的 `set -e` 把这个退出码
#   当成失败直接中止了整个脚本。此时 A.2 的 metrics.json / 图其实已全部落盘。
#
#   本版把每次 python 调用包进 run_py(): 退出码 127 单独放行 (结果已保存),
#   其它非零退出码仍视为致命错误并中止, 这样真实崩溃不会被掩盖。
#
# 用法 (项目根目录):
#   PY=.venv/Scripts/python.exe bash experiments/resume_from_A3.sh
# ============================================================================
set -euo pipefail

PY="${PY:-python}"
SEEDS="42,123,999"
CITIES="SZH,AMS,JHB,LOA,MEL,SPO"
ROUNDS=50
LOCAL_EPOCHS=5

# 运行单个实验: 退出码 127 (torch 关闭崩溃, 结果已保存) 放行, 其它非零码中止。
run_py() {
  local desc="$1"; shift
  local code
  echo ""
  echo "--- $desc ---"
  "$@"
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "    [OK]   $desc"
  elif [ "$code" -eq 127 ]; then
    echo "    [OK*]  $desc (退出码 127 = torch 关闭崩溃, 结果已落盘)"
  else
    echo "    [FAIL] $desc 退出码 $code — 结果可能不完整, 见上方日志"
    exit "$code"
  fi
}

echo "=============================================================="
echo "  续跑 v2: A.3-A.4 + B + C + 基线  (跳过 A.1 base / A.2 FedBN)"
echo "=============================================================="

# ────────────────────────────────────────────────────────────────
# 场景 A — 个性化消融 (已跳过 A.1 Base 与 A.2 FedBN; 跑 A.3/A.4)
# ────────────────────────────────────────────────────────────────
A_COMMON="--cities $CITIES --station_selection stratified_balanced --top_k 9 \
--aggregation fedavg --city_weight_alpha 0 --num_rounds $ROUNDS \
--local_epochs $LOCAL_EPOCHS --seeds $SEEDS"

run_py "A.3 + LocalHead"         $PY main.py $A_COMMON --local_head
run_py "A.4 + FedBN + LocalHead" $PY main.py $A_COMMON --fedbn --local_head

# ────────────────────────────────────────────────────────────────
# 场景 B — 城市权重指数扫描 (比例分配, 共 60 站, α ∈ {0,0.5,1})
# ────────────────────────────────────────────────────────────────
B_COMMON="--cities $CITIES --station_selection proportional --top_k 60 \
--aggregation fedavg --num_rounds $ROUNDS --local_epochs $LOCAL_EPOCHS \
--seeds $SEEDS"

for A in 0 0.5 1.0; do
  run_py "B.α=$A" $PY main.py $B_COMMON --city_weight_alpha $A
done

# ────────────────────────────────────────────────────────────────
# 场景 C — 留一城市冷启动 (FedProx μ=0.01, α=0.5, top_k=10, rounds=30)
# ────────────────────────────────────────────────────────────────
run_py "场景 C: 留一城市冷启动" $PY experiments/leave_one_out.py \
  --top_k 10 --rounds 30 --local_epochs 5 \
  --finetune_days 14,30 --seeds $SEEDS

# ────────────────────────────────────────────────────────────────
# 基线 (基线脚本仅支持 --seed, 故按种子循环)
# ────────────────────────────────────────────────────────────────
for S in 42 123 999; do
  run_py "centralized_personalized seed=$S" \
    $PY experiments/baseline_centralized_personalized.py \
    --cities $CITIES --station_selection stratified_balanced --top_k 9 \
    --epochs 100 --seed $S
done

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
echo "  续跑完成。"
echo "  metrics.json 数量: $(find outputs -name metrics.json | wc -l)  (预期约 78, 含已完成的 A.1/A.2 共 6)"
echo "  场景 C 汇总:        $(ls outputs/leave_one_out/loo_summary_multi_*.json 2>/dev/null | wc -l) 个文件"
echo "  聚合结果:           $PY experiments/organize_results.py"
echo "=============================================================="
