"""
公平对比实验 — 所有方法使用相同的站点、时间切分、架构、训练步数和种子

实验矩阵:
  Baseline:
    - Seasonal Naive (季节朴素预测)
    - Local-only TCN-LSTM (孤岛训练)
    - Centralized TCN-LSTM (集中式训练, 理论上限)

  Federated:
    - FedAvg + TCN-LSTM
    - FedProx + TCN-LSTM
    - Clustered FL + TCN-LSTM (簇内 FedProx + 负荷特征聚类)
    - Full Method: FedBN + Local Head + Clustered FL

消融实验 (模型架构):
    - LSTM-only
    - TCN-only
    - TCN-LSTM

使用方式:
  python experiments/fair_comparison.py --city SZH --seeds 42,123,999 --top_k 10 --rounds 30
"""
import sys
import os
import json
import subprocess
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_command(name: str, cmd: str):
    """运行单个实验命令"""
    print(f"\n{'#'*60}")
    print(f"# {name}")
    print(f"# {cmd}")
    print(f"{'#'*60}\n")
    ret = os.system(cmd)
    if ret != 0:
        print(f"  WARNING: {name} returned exit code {ret}")
    return ret


def collect_metrics(output_dir: str, city: str, method: str,
                    seeds: list) -> dict:
    """收集某个方法在所有种子下的平均指标"""
    all_avgs = []
    for seed in seeds:
        # 扫描 run 目录
        seed_dir = os.path.join(output_dir, city, method, f"seed_{seed}")
        if not os.path.exists(seed_dir):
            print(f"  Missing: {seed_dir}")
            continue

        run_dirs = sorted([
            d for d in os.listdir(seed_dir)
            if d.startswith("run_") and os.path.isdir(os.path.join(seed_dir, d))
        ])
        if not run_dirs:
            continue

        # 取最新的 run
        latest_run = os.path.join(seed_dir, run_dirs[-1])
        metrics_file = os.path.join(latest_run, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                data = json.load(f)
            if "AVERAGE" in data:
                all_avgs.append(data["AVERAGE"])

    if not all_avgs:
        return None

    # 计算均值和标准差
    result = {}
    for key in all_avgs[0]:
        vals = [m[key] for m in all_avgs if key in m and m[key] is not None]
        if vals:
            result[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="SZH")
    parser.add_argument("--seeds", default="42,123,999")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100,
                        help="Epochs for non-FL methods (local, centralized)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Skip seasonal naive, local-only, centralized")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    city = args.city
    top_k = args.top_k
    rounds = args.rounds
    local_epochs = args.local_epochs
    base_dir = args.output_dir or "outputs"

    # 实验列表
    experiment_groups = {
        "Baseline": [
            ("seasonal_naive",
             f"python experiments/baseline_seasonal_naive.py --city {city} --top_k {top_k}"),
            ("local_only",
             f"python experiments/baseline_local.py --city {city} --top_k {top_k} --epochs {args.epochs}"),
            ("centralized",
             f"python experiments/baseline_centralized.py --city {city} --top_k {top_k} --epochs {args.epochs}"),
        ],
        "Federated": [
            ("fedavg",
             f"python main.py --city {city} --aggregation fedavg "
             f"--top_k {top_k} --num_rounds {rounds} --local_epochs {local_epochs} "
             f"--output_dir {base_dir}"),
            ("fedprox",
             f"python main.py --city {city} --aggregation fedprox "
             f"--top_k {top_k} --num_rounds {rounds} --local_epochs {local_epochs} "
             f"--mu 0.01 --output_dir {base_dir}"),
            ("clustered",
             f"python main.py --city {city} --aggregation clustered "
             f"--top_k {top_k} --num_rounds {rounds} --local_epochs {local_epochs} "
             f"--mu 0.01 --n_clusters 3 --output_dir {base_dir}"),
            ("fedprox_fedbn_localhead",
             f"python main.py --city {city} --aggregation fedprox "
             f"--top_k {top_k} --num_rounds {rounds} --local_epochs {local_epochs} "
             f"--mu 0.01 --fedbn --local_head --output_dir {base_dir}"),
        ],
    }

    if args.skip_baselines:
        experiment_groups.pop("Baseline", None)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = os.path.join(base_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)

    # 收集所有 baseline 方法的无种子的单次运行 (baseline 脚本还不太支持 --seed + --seeds)
    # 对 Federated 方法, 使用 main.py 的 --seeds 直接在内部处理

    all_results = {}

    for group_name, experiments in experiment_groups.items():
        print(f"\n{'='*60}")
        print(f"  {group_name}")
        print(f"{'='*60}")

        for method, cmd in experiments:
            # Baselines: 对每个种子分别运行
            if group_name == "Baseline":
                full_cmd = cmd
                for seed in seeds:
                    seed_cmd = full_cmd + f" --seed {seed}"
                    run_command(f"{method} (seed={seed})", seed_cmd)

            # Federated: main.py 内部支持 --seeds
            else:
                full_cmd = cmd + f" --seeds {args.seeds}"
                run_command(f"{method}", full_cmd)

            # 收集指标
            metrics = collect_metrics(base_dir, city, method, seeds)
            if metrics:
                all_results[method] = metrics

    # 打印汇总表格
    print(f"\n\n{'='*80}")
    print(f"  FAIR COMPARISON SUMMARY — {city}")
    print(f"  Seeds: {seeds}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}")

    header = f"  {'Method':<30s}"
    for m_name in ["RMSE", "MAE", "WAPE", "SMAPE"]:
        header += f" {m_name:>16s}"
    print(header)
    print(f"  {'-'*30}{'  ' + '-'*16 + '  ' + '-'*16 + '  ' + '-'*16 + '  ' + '-'*16}")

    best_rmse_method = None
    best_rmse = float("inf")

    for method, metrics in sorted(all_results.items()):
        line = f"  {method:<30s}"
        for m_name in ["RMSE", "MAE", "WAPE", "SMAPE"]:
            if m_name in metrics:
                m, s = metrics[m_name]["mean"], metrics[m_name]["std"]
                line += f" {m:>8.2f}±{s:<5.2f}"
            else:
                line += f" {'N/A':>16s}"
        print(line)

        if "RMSE" in metrics and metrics["RMSE"]["mean"] < best_rmse:
            best_rmse = metrics["RMSE"]["mean"]
            best_rmse_method = method

    print(f"\n  Best RMSE: {best_rmse_method} ({best_rmse:.4f})")
    print(f"{'='*80}")

    # 保存汇总
    summary = {
        "city": city,
        "seeds": seeds,
        "timestamp": timestamp,
        "results": all_results,
        "best_method": best_rmse_method,
        "best_rmse": best_rmse,
    }
    summary_path = os.path.join(summary_dir, f"comparison_{city}_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
