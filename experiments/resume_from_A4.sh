#!/usr/bin/env bash
# ============================================================================
# 续跑脚本 v3 — 跳过已完成的 A.1 (base) / A.2 (FedBN) / A.3 (LocalHead)
#   继续: A.4 (FedBN+LocalHead) + 场景 B + 场景 C + 基线
#
# 前两次中止原因与本次修复:
#   - main.py 多 seed 训练结束后, torch+CUDA 在解释器关闭阶段偶发崩溃 (exit 127),
#     而脚本 `set -e` 把它当成失败直接终止。
#   - v2 的 run_py() 里 `"$@"` 仍受 `set -e` 约束, 失败时在 `code=$?` 之前就退出了,
#     等于没修到。
#   - 本次: (1) main.py 末尾已加 os._exit(0) 强制干净退出 (不再产生 127);
#           (2) run_py() 改用 `if "$@"` 标准写法, 让失败进入 else 分支而不是触发 set -e;
#           (3) 对退出码 127 单独放行 (leave_one_out.py / 基线脚本仍可能偶发)。
#
# 用法 (项目根目录):
#   PY=.venv/Scripts/python.exe bash experiments/resume_from_A4.sh
# ============================================================================
set -euo pipefail

PY="${PY:-python}"
SEEDS="42,123,999"
CITIES="SZH,AMS,JHB,LOA,MEL,SPO"
ROUNDS=50
LOCAL_EPOCHS=5

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
echo "  续跑 v3: A.4 + B + C + 基线  (跳过 A.1/A.2/A.3)"
echo "=============================================================="

# ────────────────────────────────────────────────────────────────
# 场景 A — 个性化消融 (已跳过 A.1 Base / A.2 FedBN / A.3 LocalHead)
# ────────────────────────────────────────────────────────────────
A_COMMON="--cities $CITIES --station_selection stratified_balanced --top_k 9 \
--aggregation fedavg --city_weight_alpha 0 --num_rounds $ROUNDS \
--local_epochs $LOCAL_EPOCHS --seeds $SEEDS"

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
echo "  metrics.json 数量: $(find outputs -name metrics.json | wc -l)  (预期约 78, 含已完成的 A.1/A.2/A.3 共 9)"
echo "  场景 C 汇总:        $(ls outputs/leave_one_out/loo_summary_multi_*.json 2>/dev/null | wc -l) 个文件"
echo "  聚合结果:           $PY experiments/organize_results.py"
echo "=============================================================="
