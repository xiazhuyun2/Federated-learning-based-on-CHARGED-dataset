"""
超参数搜索 — 系统化网格搜索 + 参数敏感性分析

搜索关键超参数的最优组合, 每组用快速配置 (少量站点 + 少轮次) 评估。
结果输出 CSV 和参数敏感性图。

运行: python experiments/hyperparam_search.py --city SZH
"""
import sys
import os
import json
import itertools
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from src.federated.trainer import FederatedTrainer
from src.utils.metrics import set_seed


# ============================================================
# 参数搜索空间定义
# ============================================================

# 每次只搜索一个维度, 其余固定为默认值 (单因素分析)
SEARCH_SPACE = {
    "lr": {
        "values": [5e-4, 1e-3, 2e-3, 5e-3],
        "label": "Learning Rate",
        "default": 1e-3,
    },
    "seq_len": {
        "values": [48, 72, 168, 336],
        "label": "Input Sequence Length (hours)",
        "default": 168,
    },
    "lstm_hidden": {
        "values": [64, 128, 256],
        "label": "LSTM Hidden Size",
        "default": 128,
    },
    "local_epochs": {
        "values": [1, 3, 5, 10],
        "label": "Local Training Epochs",
        "default": 5,
    },
    "fedprox_mu": {
        "values": [0.0, 0.001, 0.01, 0.1, 1.0],
        "label": "FedProx Proximal Term (mu)",
        "default": 0.01,
    },
    "n_clusters": {
        "values": [2, 3, 5, 8],
        "label": "Number of Clusters (Clustered FL)",
        "default": 3,
    },
}


def run_single_config(city: str, param_name: str, param_value,
                      top_k: int = 5, num_rounds: int = 10) -> dict:
    """
    用指定参数值运行一次快速训练, 返回测试指标
    """
    cfg = Config()
    cfg.data.top_k_stations = top_k
    cfg.fed.num_rounds = num_rounds
    cfg.fed.local_epochs = 3  # 默认快速
    cfg.fed.aggregation = "fedprox"

    # 设置搜索参数
    if param_name == "lr":
        cfg.fed.lr = param_value
    elif param_name == "seq_len":
        cfg.data.seq_len = param_value
    elif param_name == "lstm_hidden":
        cfg.model.lstm_hidden = param_value
    elif param_name == "local_epochs":
        cfg.fed.local_epochs = param_value
    elif param_name == "fedprox_mu":
        cfg.fed.fedprox_mu = param_value
    elif param_name == "n_clusters":
        cfg.fed.n_clusters = param_value
        cfg.fed.aggregation = "clustered"

    set_seed(cfg.seed)

    trainer = FederatedTrainer(cfg)
    trainer.prepare_city_clients(city)
    results = trainer.run_federated_training()

    avg = results.get("AVERAGE", {})
    return {
        "param": param_name,
        "value": param_value,
        "RMSE": avg.get("RMSE", float("inf")),
        "MAE": avg.get("MAE", float("inf")),
        "MAPE": avg.get("MAPE", float("inf")),
    }


def plot_sensitivity(all_results: dict, output_dir: str):
    """
    参数敏感性分析图: 每个参数一个子图, 展示指标随参数变化的趋势
    """
    n_params = len(all_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = {"RMSE": "#1976D2", "MAE": "#4CAF50", "MAPE": "#F44336"}

    for i, (param_name, results) in enumerate(all_results.items()):
        if i >= 6:
            break
        ax = axes[i]
        info = SEARCH_SPACE[param_name]

        values = [r["value"] for r in results]
        rmse = [r["RMSE"] for r in results]
        mae = [r["MAE"] for r in results]
        mape = [r["MAPE"] for r in results]

        x = range(len(values))

        ax.plot(x, rmse, "o-", color=colors["RMSE"], linewidth=2, label="RMSE")
        ax.plot(x, mae, "s-", color=colors["MAE"], linewidth=2, label="MAE")

        ax2 = ax.twinx()
        ax2.plot(x, mape, "D--", color=colors["MAPE"], linewidth=2, label="MAPE(%)")
        ax2.set_ylabel("MAPE (%)", color=colors["MAPE"])

        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in values], fontsize=9)
        ax.set_xlabel(info["label"])
        ax.set_ylabel("RMSE / MAE")
        ax.set_title(f"Sensitivity: {info['label']}", fontsize=11, fontweight="bold")

        # 标注最优值
        best_idx = np.argmin(rmse)
        ax.axvline(x=best_idx, color="gray", linestyle=":", alpha=0.5)

        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # 隐藏多余的子图
    for j in range(len(all_results), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Hyperparameter Sensitivity Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "hyperparam_sensitivity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Sensitivity plot saved: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter Search")
    parser.add_argument("--city", default="SZH")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Stations per experiment (small for speed)")
    parser.add_argument("--num_rounds", type=int, default=10,
                        help="FL rounds per experiment (small for speed)")
    parser.add_argument("--params", nargs="+",
                        default=["lr", "seq_len", "lstm_hidden",
                                 "local_epochs", "fedprox_mu", "n_clusters"],
                        help="Parameters to search")
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Hyperparameter Search — City: {args.city}")
    print(f"  Quick mode: {args.top_k} stations, {args.num_rounds} rounds")
    print("=" * 60)

    all_results = {}
    csv_rows = []

    for param_name in args.params:
        if param_name not in SEARCH_SPACE:
            print(f"  Unknown param: {param_name}, skipping")
            continue

        info = SEARCH_SPACE[param_name]
        print(f"\n{'='*60}")
        print(f"  Searching: {info['label']}")
        print(f"  Values: {info['values']}")
        print(f"{'='*60}")

        param_results = []

        for val in info["values"]:
            print(f"\n  --- {param_name} = {val} ---")
            result = run_single_config(
                args.city, param_name, val,
                args.top_k, args.num_rounds
            )
            param_results.append(result)
            csv_rows.append(result)

            print(f"  Result: RMSE={result['RMSE']:.4f}, "
                  f"MAE={result['MAE']:.4f}, MAPE={result['MAPE']:.2f}%")

        all_results[param_name] = param_results

        # 打印本参数最优值
        best = min(param_results, key=lambda r: r["RMSE"])
        print(f"\n  Best {param_name} = {best['value']} "
              f"(RMSE={best['RMSE']:.4f})")

    # 保存 CSV
    csv_path = os.path.join(output_dir, "hyperparam_search.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["param", "value", "RMSE", "MAE", "MAPE"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  Results saved: {csv_path}")

    # 生成敏感性分析图
    plot_sensitivity(all_results, output_dir)

    # 打印总结
    print("\n" + "=" * 60)
    print("  OPTIMAL VALUES SUMMARY")
    print("=" * 60)
    for param_name, results in all_results.items():
        best = min(results, key=lambda r: r["RMSE"])
        info = SEARCH_SPACE[param_name]
        print(f"  {info['label']:35s} = {best['value']:<10} "
              f"(RMSE={best['RMSE']:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
