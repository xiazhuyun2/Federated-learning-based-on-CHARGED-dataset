"""
超参数搜索 — 系统化网格搜索 + 参数敏感性分析

搜索关键超参数的最优组合, 每组用快速配置评估。
**使用验证集指标选参**, 测试集只在最终评估时使用一次。

运行: python experiments/hyperparam_search.py --city SZH
"""
import sys
import os
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config, get_run_dir
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
                      top_k: int = 5, num_rounds: int = 10,
                      base_dir: str = "outputs") -> dict:
    """
    用指定参数值运行一次快速训练, 返回验证集和测试集指标
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

    method = "clustered" if param_name == "n_clusters" else "fedprox"
    run_dir = get_run_dir(city, f"hparam_{method}_{param_name}_{param_value}",
                          cfg.seed, base_dir=base_dir)

    trainer = FederatedTrainer(cfg, run_dir=run_dir,
                               city=city, method=f"hparam_{param_name}")
    trainer.prepare_city_clients(city)
    results = trainer.run_federated_training()

    # 从 history.json 读取验证集最佳指标
    history_path = os.path.join(run_dir, "history.json")
    best_val_rmse = float("inf")
    best_val_wape = float("inf")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        for entry in history.get("val_metrics", []):
            rmse_v = float(entry.get("RMSE", "inf"))
            if rmse_v < best_val_rmse:
                best_val_rmse = rmse_v
                best_val_wape = float(entry.get("WAPE", "inf"))

    # 从 metrics.json 读取测试集指标 (仅用于最终报告, 不用于选参)
    test_avg = results.get("AVERAGE", {})

    return {
        "param": param_name,
        "value": param_value,
        "val_RMSE": best_val_rmse,
        "val_WAPE": best_val_wape,
        "test_RMSE": test_avg.get("RMSE", float("inf")),
        "test_MAE": test_avg.get("MAE", float("inf")),
        "test_WAPE": test_avg.get("WAPE", float("inf")),
        "test_SMAPE": test_avg.get("SMAPE", float("inf")),
    }


def plot_sensitivity(all_results: dict, output_dir: str):
    """参数敏感性分析图"""
    n_params = len(all_results)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = {"val_RMSE": "#1976D2", "test_RMSE": "#F44336", "val_WAPE": "#4CAF50"}

    for i, (param_name, results) in enumerate(all_results.items()):
        if i >= len(axes):
            break
        ax = axes[i]
        info = SEARCH_SPACE[param_name]

        values = [r["value"] for r in results]
        val_rmse = [r["val_RMSE"] for r in results]
        test_rmse = [r["test_RMSE"] for r in results]

        x = range(len(values))

        ax.plot(x, val_rmse, "o-", color=colors["val_RMSE"], linewidth=2,
                label="Val RMSE (select)")
        ax.plot(x, test_rmse, "s--", color=colors["test_RMSE"], linewidth=2,
                label="Test RMSE")

        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in values], fontsize=9)
        ax.set_xlabel(info["label"])
        ax.set_ylabel("RMSE")
        ax.set_title(f"{info['label']}", fontsize=11, fontweight="bold")

        # 标注验证集最优值
        best_idx = np.argmin(val_rmse)
        ax.axvline(x=best_idx, color="gray", linestyle=":", alpha=0.5)
        ax.annotate(f"Best: {values[best_idx]}",
                    xy=(x[best_idx], val_rmse[best_idx]),
                    fontsize=8, color="gray")

        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for j in range(len(all_results), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Hyperparameter Sensitivity (Select by Validation RMSE)",
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
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    search_dir = os.path.join(args.output_dir, "hyperparam_search")
    os.makedirs(search_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Hyperparameter Search — City: {args.city}")
    print(f"  Quick mode: {args.top_k} stations, {args.num_rounds} rounds")
    print(f"  Selection metric: VALIDATION RMSE (not test RMSE!)")
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
                args.top_k, args.num_rounds,
                base_dir=args.output_dir
            )
            param_results.append(result)
            csv_rows.append(result)

            print(f"  Val:  RMSE={result['val_RMSE']:.4f}, WAPE={result['val_WAPE']:.2f}%")
            print(f"  Test: RMSE={result['test_RMSE']:.4f}, WAPE={result['test_WAPE']:.2f}% "
                  f"(FOR REFERENCE ONLY, not used for selection)")

        all_results[param_name] = param_results

        # 打印本参数最优值 (基于验证集)
        best = min(param_results, key=lambda r: r["val_RMSE"])
        print(f"\n  >>> Best {param_name} = {best['value']} "
              f"(Val RMSE={best['val_RMSE']:.4f}, "
              f"Test RMSE={best['test_RMSE']:.4f})")

    # 保存 CSV (标注验证集最佳)
    csv_path = os.path.join(search_dir, "hyperparam_search.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "param", "value", "val_RMSE", "val_WAPE",
            "test_RMSE", "test_MAE", "test_WAPE", "test_SMAPE"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  Results saved: {csv_path}")

    # 生成敏感性分析图
    plot_sensitivity(all_results, search_dir)

    # 打印总结
    print("\n" + "=" * 60)
    print("  OPTIMAL VALUES SUMMARY (selected by VALIDATION RMSE)")
    print("=" * 60)
    optimal_config = {}
    for param_name, results in all_results.items():
        best = min(results, key=lambda r: r["val_RMSE"])
        info = SEARCH_SPACE[param_name]
        optimal_config[param_name] = best["value"]
        print(f"  {info['label']:35s} = {best['value']:<10} "
              f"(Val RMSE={best['val_RMSE']:.4f})")
    print("=" * 60)

    # 保存最优配置
    optimal_path = os.path.join(search_dir, "optimal_config.json")
    with open(optimal_path, "w") as f:
        json.dump(optimal_config, f, indent=2)
    print(f"\n  Optimal config saved: {optimal_path}")
    print(f"\n  WARNING: These results are from QUICK mode ({args.num_rounds} rounds, "
          f"{args.top_k} stations).")
    print(f"  Run final evaluation with these parameters and more rounds/stations "
          f"for paper results.")


if __name__ == "__main__":
    main()
