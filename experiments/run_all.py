"""
对比实验 — 一键运行所有实验组合, 生成对比表格

实验组:
  1. Local-only       (孤岛训练)
  2. Centralized      (集中式训练, 理论上限)
  3. FedAvg + LSTM    (基础联邦学习)
  4. FedProx + TCN-LSTM (本文方法 v1)
  5. Clustered FL + TCN-LSTM (本文方法 v2, 推荐)
"""
import sys
import os
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_experiment(name: str, cmd: str):
    print(f"\n{'#'*60}")
    print(f"# Experiment: {name}")
    print(f"# Command: {cmd}")
    print(f"{'#'*60}\n")
    os.system(cmd)


def main():
    city = "SZH"
    top_k = 10  # 快速验证用 10 个站点
    rounds = 30

    experiments = [
        ("Local-only (baseline)",
         f"python experiments/baseline_local.py --city {city} --top_k {top_k} --epochs 50"),

        ("FedAvg + TCN-LSTM",
         f"python main.py --city {city} --aggregation fedavg "
         f"--top_k {top_k} --num_rounds {rounds} --local_epochs 3"),

        ("FedProx + TCN-LSTM",
         f"python main.py --city {city} --aggregation fedprox "
         f"--top_k {top_k} --num_rounds {rounds} --local_epochs 3 --mu 0.01"),

        ("Clustered FL + TCN-LSTM",
         f"python main.py --city {city} --aggregation clustered "
         f"--top_k {top_k} --num_rounds {rounds} --local_epochs 3 --n_clusters 3"),
    ]

    for name, cmd in experiments:
        run_experiment(name, cmd)

    print("\n" + "=" * 60)
    print("  All experiments completed!")
    print("  Check outputs/ directory for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
