"""
可视化模块 — 6种论文级图表

"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# 论文级全局样式
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# 配色方案
COLORS = {
    "primary": "#1976D2",
    "secondary": "#F44336",
    "accent": "#4CAF50",
    "orange": "#FF9800",
    "purple": "#9C27B0",
    "gray": "#757575",
}


def plot_training_loss(history: Dict, output_dir: str) -> str:
    """
    图1: 训练损失曲线
    X: 通信轮次, Y: 平均训练Loss
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    rounds = history["rounds"]
    losses = history["avg_loss"]

    ax.plot(rounds, losses, color=COLORS["primary"], linewidth=2, marker="o",
            markersize=3, label="Avg Training Loss")

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Average Loss (MSE)")
    ax.set_title("Federated Training Loss Curve")
    ax.legend()

    path = os.path.join(output_dir, "loss_curve.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_val_metrics(history: Dict, output_dir: str) -> str:
    """
    图2: 验证指标曲线 (双Y轴)
    左轴: RMSE + MAE, 右轴: MAPE(%)
    """
    if not history.get("val_metrics"):
        return ""

    val_data = history["val_metrics"]
    rounds = [d["round"] for d in val_data]
    rmse = [d["RMSE"] for d in val_data]
    mae = [d["MAE"] for d in val_data]
    mape = [d["MAPE"] for d in val_data]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(rounds, rmse, color=COLORS["primary"], linewidth=2,
             marker="s", markersize=5, label="RMSE")
    ax1.plot(rounds, mae, color=COLORS["accent"], linewidth=2,
             marker="^", markersize=5, label="MAE")
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("RMSE / MAE")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(rounds, mape, color=COLORS["secondary"], linewidth=2,
             marker="D", markersize=5, label="MAPE (%)", linestyle="--")
    ax2.set_ylabel("MAPE (%)")
    ax2.tick_params(axis="y")

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Validation Metrics over Training Rounds")
    fig.tight_layout()

    path = os.path.join(output_dir, "val_metrics.png")
    fig.savefig(path)
    plt.close(fig)
    return path


@torch.no_grad()
def plot_prediction_vs_actual(
    model: nn.Module, dataloader: DataLoader, scaler,
    output_dir: str, device: str = "cpu", n_hours: int = 168,
    station_name: str = ""
) -> str:
    """
    图3: 预测 vs 实际对比曲线
    选取测试集最后 n_hours 小时, 将预测值和真实值叠加
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    for x, y in dataloader:
        x = x.to(device)
        pred = model(x).cpu().numpy()
        all_preds.append(pred)
        all_targets.append(y.numpy())

    model.to("cpu")

    if not all_preds:
        return ""

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 反归一化
    preds_inv = scaler.inverse_target(all_preds)
    targets_inv = scaler.inverse_target(all_targets)

    # reshape 回 (N, pred_len) 如果被 flatten 了
    if preds_inv.ndim == 1 and all_preds.ndim == 2:
        pred_len = all_preds.shape[1]
        preds_inv = preds_inv.reshape(-1, pred_len)
        targets_inv = targets_inv.reshape(-1, pred_len)

    # 取最后 n_hours 个预测窗口的第一个值, 拼出连续时间序列
    if preds_inv.ndim == 2:
        pred_series = preds_inv[-n_hours:, 0]
        true_series = targets_inv[-n_hours:, 0]
    else:
        pred_series = preds_inv[-n_hours:]
        true_series = targets_inv[-n_hours:]

    fig, ax = plt.subplots(figsize=(12, 5))
    hours = np.arange(len(true_series))

    ax.plot(hours, true_series, color=COLORS["primary"], linewidth=1.5,
            label="Actual", alpha=0.9)
    ax.plot(hours, pred_series, color=COLORS["secondary"], linewidth=1.5,
            label="Predicted", alpha=0.8, linestyle="--")

    ax.fill_between(hours, true_series, pred_series,
                    alpha=0.15, color=COLORS["secondary"])

    ax.set_xlabel("Hour")
    ax.set_ylabel("Charging Load")
    title = "Prediction vs Actual — Test Set"
    if station_name:
        title += f" ({station_name})"
    ax.set_title(title)
    ax.legend()

    path = os.path.join(output_dir, "pred_vs_actual.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_station_comparison(test_results: Dict, output_dir: str) -> str:
    """
    图4: 各站点指标对比条形图
    每个站点的 RMSE / MAE / MAPE 并排柱状图
    """
    # 过滤掉 AVERAGE
    stations = {k: v for k, v in test_results.items() if k != "AVERAGE"}
    if not stations:
        return ""

    names = list(stations.keys())
    rmse = [stations[n]["RMSE"] for n in names]
    mae = [stations[n]["MAE"] for n in names]
    mape = [stations[n]["MAPE"] for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: RMSE + MAE
    bars1 = ax1.bar(x - width/2, rmse, width, label="RMSE",
                    color=COLORS["primary"], alpha=0.85)
    bars2 = ax1.bar(x + width/2, mae, width, label="MAE",
                    color=COLORS["accent"], alpha=0.85)
    ax1.set_xlabel("Station")
    ax1.set_ylabel("Value")
    ax1.set_title("RMSE & MAE by Station")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax1.legend()

    # 右图: MAPE
    bars3 = ax2.bar(x, mape, width * 1.5, color=COLORS["orange"], alpha=0.85)
    ax2.set_xlabel("Station")
    ax2.set_ylabel("MAPE (%)")
    ax2.set_title("MAPE (%) by Station")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)

    # 标注 AVERAGE 线
    if "AVERAGE" in test_results:
        avg = test_results["AVERAGE"]
        ax1.axhline(y=avg["RMSE"], color=COLORS["secondary"], linestyle="--",
                    alpha=0.7, label=f"Avg RMSE={avg['RMSE']:.1f}")
        ax2.axhline(y=avg["MAPE"], color=COLORS["secondary"], linestyle="--",
                    alpha=0.7, label=f"Avg MAPE={avg['MAPE']:.1f}%")
        ax1.legend(fontsize=8)
        ax2.legend(fontsize=8)

    fig.suptitle("Per-Station Test Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "station_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    return path


@torch.no_grad()
def plot_error_distribution(
    model: nn.Module, dataloader: DataLoader, scaler,
    output_dir: str, device: str = "cpu"
) -> str:
    """
    图5: 预测误差分布直方图
    验证误差是否近似正态分布 (好模型的误差应该集中在0附近)
    """
    model.to(device)
    model.eval()

    all_errors = []
    for x, y in dataloader:
        x = x.to(device)
        pred = model(x).cpu().numpy()
        target = y.numpy()

        pred_inv = scaler.inverse_target(pred)
        target_inv = scaler.inverse_target(target)

        errors = (pred_inv - target_inv).flatten()
        all_errors.append(errors)

    model.to("cpu")

    if not all_errors:
        return ""

    all_errors = np.concatenate(all_errors)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左: 误差分布
    ax1.hist(all_errors, bins=50, color=COLORS["primary"], alpha=0.7,
             edgecolor="white", density=True)
    ax1.axvline(x=0, color=COLORS["secondary"], linestyle="--", linewidth=2)
    ax1.axvline(x=np.mean(all_errors), color=COLORS["orange"], linestyle="-",
                linewidth=1.5, label=f"Mean={np.mean(all_errors):.2f}")
    ax1.set_xlabel("Prediction Error (Predicted - Actual)")
    ax1.set_ylabel("Density")
    ax1.set_title("Error Distribution")
    ax1.legend()

    # 右: Q-Q plot
    from scipy import stats as sp_stats
    errors_sorted = np.sort(all_errors)
    n = len(errors_sorted)
    theoretical_q = sp_stats.norm.ppf(np.arange(1, n + 1) / (n + 1))

    ax2.scatter(theoretical_q, errors_sorted, alpha=0.3, s=3, color=COLORS["primary"])
    # 参考线
    q25, q75 = np.percentile(errors_sorted, [25, 75])
    t25, t75 = sp_stats.norm.ppf([0.25, 0.75])
    slope = (q75 - q25) / (t75 - t25)
    intercept = q25 - slope * t25
    x_line = np.array([theoretical_q[0], theoretical_q[-1]])
    ax2.plot(x_line, slope * x_line + intercept, color=COLORS["secondary"],
             linewidth=2, label="Reference line")
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")
    ax2.set_title("Q-Q Plot (Normality Check)")
    ax2.legend()

    fig.suptitle("Prediction Error Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "error_dist.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_method_comparison(results_dict: Dict[str, Dict], output_dir: str) -> str:
    """
    图6: 多方法对比雷达图 (论文核心图)

    results_dict 格式:
    {
        "Local-only": {"RMSE": ..., "MAE": ..., "MAPE": ...},
        "FedAvg":     {"RMSE": ..., "MAE": ..., "MAPE": ...},
        "FedProx":    {"RMSE": ..., "MAE": ..., "MAPE": ...},
        "Clustered":  {"RMSE": ..., "MAE": ..., "MAPE": ...},
    }
    """
    if not results_dict:
        return ""

    metrics = ["RMSE", "MAE", "MAPE"]
    methods = list(results_dict.keys())
    n_metrics = len(metrics)

    # 归一化到 [0, 1] (越小越好, 所以反转: 1 - normalized)
    max_vals = {}
    for m in metrics:
        max_vals[m] = max(results_dict[method][m] for method in methods) * 1.1

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors_list = [COLORS["gray"], COLORS["primary"],
                   COLORS["orange"], COLORS["secondary"]]

    for i, method in enumerate(methods):
        values = []
        for m in metrics:
            # 归一化: 1 表示最差, 0 表示最好
            normalized = results_dict[method][m] / max_vals[m]
            values.append(normalized)
        values += values[:1]

        color = colors_list[i % len(colors_list)]
        ax.plot(angles, values, "o-", linewidth=2, label=method, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_title("Method Comparison\n(Closer to center = Better)",
                 fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    path = os.path.join(output_dir, "method_radar.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_all_plots(history: Dict, test_results: Dict, output_dir: str):
    """一键生成训练完成后的所有图表"""
    os.makedirs(output_dir, exist_ok=True)

    paths = []
    paths.append(plot_training_loss(history, output_dir))
    paths.append(plot_val_metrics(history, output_dir))
    paths.append(plot_station_comparison(test_results, output_dir))

    generated = [p for p in paths if p]
    print(f"\n  Generated {len(generated)} plots:")
    for p in generated:
        print(f"    - {p}")
