"""
论文配图脚本 — 生成 paper/figures/ 下的 4 张英文图

图 1  分层联邦架构示意图 (fig1_architecture.png)
图 2  场景 A 个性化消融对比柱状图 (fig2_scenario_a_ablation.png)
图 3  场景 B 城市权重指数 α 扫描图 (fig3_scenario_b_alpha.png)
图 4  场景 C 冷启动泛化对比图 (fig4_scenario_c_coldstart.png)

只读 JSON + matplotlib/numpy，不 import torch（无 GPU 也能跑）。
数据源:
  - results/organized_results.json   (场景 A/B)
  - results/leave_one_out_3seed.json (场景 C)

用法:
  python experiments/plot_paper_figures.py
"""
import os
import json
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIG_DIR = os.path.join(PROJECT_ROOT, "paper", "figures")

# 配色（复刻 src/utils/visualization.py 的 COLORS，避免 import 该模块拖入 torch）
COLORS = {
    "primary": "#1976D2",
    "secondary": "#F44336",
    "accent": "#4CAF50",
    "orange": "#FF9800",
    "purple": "#9C27B0",
    "gray": "#757575",
}

# 论文级样式
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

CITIES = ["SZH", "AMS", "JHB", "LOA", "MEL", "SPO"]


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _mean_std(block):
    """从 {mean,std} 块取 (mean, std)，缺失返回 (nan, nan)。"""
    if isinstance(block, dict) and "mean" in block:
        return float(block["mean"]), float(block.get("std", 0.0))
    return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# 图 1: 分层联邦架构示意图
# ---------------------------------------------------------------------------
def fig1_architecture(out_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    n_city = 6
    city_x = np.linspace(1.2, 10.8, n_city)

    # ---- 底部：站级客户端 ----
    ax.text(0.4, 1.6, "Station\nclients", fontsize=10, ha="left", va="center",
            color=COLORS["gray"], weight="bold")
    station_patches = []
    for cx in city_x:
        # 每个城市 3 个小站点方块
        for k in range(3):
            sx = cx - 0.45 + k * 0.4
            p = FancyBboxPatch((sx, 0.5), 0.32, 0.6,
                               boxstyle="round,pad=0.02",
                               facecolor="#E3F2FD", edgecolor=COLORS["primary"],
                               linewidth=1.0)
            ax.add_patch(p)
            station_patches.append((cx, sx + 0.16, 1.1))

    # ---- 中部：城市聚合 ----
    city_boxes = {}
    for i, cx in enumerate(city_x):
        b = FancyBboxPatch((cx - 0.55, 3.2), 1.1, 1.2,
                           boxstyle="round,pad=0.03",
                           facecolor="#C8E6C9", edgecolor=COLORS["accent"],
                           linewidth=1.6)
        ax.add_patch(b)
        ax.text(cx, 3.8, CITIES[i], fontsize=11, ha="center", va="center", weight="bold")
        city_boxes[CITIES[i]] = (cx, 3.2, 4.4)

    ax.text(0.4, 4.0, "City\naggregation", fontsize=10, ha="left", va="center",
            color=COLORS["accent"], weight="bold")

    # ---- 顶部：全局模型 ----
    gx, gy, gw, gh = 4.5, 7.0, 3.0, 1.2
    gbox = FancyBboxPatch((gx, gy), gw, gh, boxstyle="round,pad=0.05",
                          facecolor="#FFEBEE", edgecolor=COLORS["secondary"],
                          linewidth=2.0)
    ax.add_patch(gbox)
    ax.text(gx + gw / 2, gy + gh / 2, "Global model\n(server)",
            fontsize=12, ha="center", va="center", weight="bold")

    # ---- 站点 → 城市 箭头 ----
    for cx, sx, top in station_patches:
        ax.add_patch(FancyArrowPatch((sx, top), (cx, 3.2),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color=COLORS["primary"], linewidth=1.0, alpha=0.6))

    # ---- 城市 → 全局 箭头（带 β 标注） ----
    for cx, _, top in city_boxes.values():
        ax.add_patch(FancyArrowPatch((cx, top), (gx + gw / 2, gy),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=COLORS["secondary"], linewidth=1.4))
    ax.text(10.6, 5.9, "β_c ∝ N_c^α\n(city weight)", fontsize=10,
            ha="center", va="center", color=COLORS["secondary"], weight="bold")

    # ---- 下行广播箭头 ----
    ax.add_patch(FancyArrowPatch((gx + gw, gy + gh / 2), (10.2, 4.4),
                                 arrowstyle="-|>", mutation_scale=12,
                                 linestyle="--", color=COLORS["gray"], linewidth=1.2))
    ax.text(11.5, 5.6, "broadcast", fontsize=9, ha="center", va="center",
            color=COLORS["gray"], rotation=90)

    # ---- 个性化标注（放右上角空白区，避免与站点方块重叠） ----
    ax.text(9.4, 8.7, "FedBN / head\nstay local", fontsize=9, ha="center",
            va="center", color=COLORS["purple"], weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5",
                      edgecolor=COLORS["purple"], alpha=0.9))

    ax.set_title("Two-level Hierarchical Federated Learning Architecture", pad=12)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 2: 场景 A 个性化消融
# ---------------------------------------------------------------------------
def fig2_scenario_a(org, out_path):
    order = [
        ("Base\n(FedAvg)", "fedavg_a0_stratified_balanced"),
        ("FedBN", "fedavg_a0_stratified_balanced_fedbn"),
        ("LocalHead", "fedavg_a0_stratified_balanced_localhead"),
        ("FedBN\n+LocalHead", "fedavg_a0_stratified_balanced_fedbn_localhead"),
    ]
    labels = [o[0] for o in order]
    macro, macro_e = [], []
    worst, worst_e = [], []
    for _, key in order:
        m, s = _mean_std(org[key].get("WAPE"))
        w, ws = _mean_std(org[key].get("worst_city_WAPE"))
        macro.append(m); macro_e.append(s)
        worst.append(w); worst_e.append(ws)

    x = np.arange(len(labels))
    w_bar = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w_bar / 2, macro, w_bar, yerr=macro_e, capsize=4,
           color=COLORS["primary"], label="Macro-City WAPE")
    ax.bar(x + w_bar / 2, worst, w_bar, yerr=worst_e, capsize=4,
           color=COLORS["orange"], label="Worst-City WAPE")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("WAPE (%)")
    ax.set_title("Scenario A: Personalization Ablation")
    ax.set_ylim(0, max(worst + [v + e for v, e in zip(worst, worst_e)]) * 1.15)
    ax.legend(loc="upper left", framealpha=0.9)

    # 标注最佳项
    best_idx = int(np.argmin(macro))
    ax.annotate("best", xy=(x[best_idx] - w_bar / 2, macro[best_idx]),
                xytext=(x[best_idx] - w_bar / 2 - 0.5, macro[best_idx] + 5),
                arrowprops=dict(arrowstyle="->", color=COLORS["accent"]),
                color=COLORS["accent"], weight="bold", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(f"[fig2] Macro-City WAPE: {[round(m, 2) for m in macro]}")
    print(f"[fig2] Worst-City WAPE: {[round(w, 2) for w in worst]}")


# ---------------------------------------------------------------------------
# 图 3: 场景 B 城市权重指数 α 扫描
# ---------------------------------------------------------------------------
def fig3_scenario_b(org, out_path):
    order = [
        ("0", "fedavg_a0_proportional"),
        ("0.5", "fedavg_a0_5_proportional"),
        ("1", "fedavg_a1_proportional"),
    ]
    alphas = [o[0] for o in order]
    macro, macro_e = [], []
    worst, worst_e = [], []
    for _, key in order:
        m, s = _mean_std(org[key].get("WAPE"))
        w, ws = _mean_std(org[key].get("worst_city_WAPE"))
        macro.append(m); macro_e.append(s)
        worst.append(w); worst_e.append(ws)

    x = np.arange(len(alphas))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(x, macro, yerr=macro_e, marker="o", markersize=8, linewidth=2,
                capsize=5, color=COLORS["primary"], label="Macro-City WAPE")
    ax.errorbar(x, worst, yerr=worst_e, marker="s", markersize=8, linewidth=2,
                capsize=5, color=COLORS["orange"], label="Worst-City WAPE")

    ax.set_xticks(x)
    ax.set_xticklabels(["α = 0\n(equal)", "α = 0.5", "α = 1\n(sample-weighted)"])
    ax.set_xlabel("City weight exponent α")
    ax.set_ylabel("WAPE (%)")
    ax.set_title("Scenario B: Effect of City Weight Exponent α")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(f"[fig3] α={alphas}  Macro={[round(m, 2) for m in macro]}  "
          f"Worst={[round(w, 2) for w in worst]}")


# ---------------------------------------------------------------------------
# 图 4: 场景 C 冷启动泛化
# ---------------------------------------------------------------------------
def _macro_across_cities_and_seeds(per_seed, path):
    """per_seed[seed][city][...path...]["WAPE"] -> 跨城市等权、跨 seed 均值±std。"""
    seed_vals = []
    for seed, cities in per_seed.items():
        city_vals = []
        for city in CITIES:
            block = cities[city]
            try:
                v = block
                for k in path:
                    v = v[k]
                city_vals.append(float(v["WAPE"]))
            except (KeyError, TypeError):
                continue
        if city_vals:
            seed_vals.append(float(np.mean(city_vals)))
    if not seed_vals:
        return float("nan"), float("nan")
    return float(np.mean(seed_vals)), float(np.std(seed_vals))


def fig4_scenario_c(loo, out_path):
    per_seed = loo["per_seed"]
    methods = [
        ("Zero-shot\n(calibrated)", ["zero_shot_calibrated"], COLORS["primary"]),
        ("Few-shot\n(14d)", ["few_shot", "14"], COLORS["purple"]),
        ("Few-shot\n(30d)", ["few_shot", "30"], COLORS["purple"]),
        ("From-scratch\n(14d)", ["from_scratch", "14"], COLORS["gray"]),
        ("From-scratch\n(30d)", ["from_scratch", "30"], COLORS["gray"]),
        ("Full-local\nbaseline", ["full_local"], COLORS["accent"]),
    ]
    labels = [m[0] for m in methods]
    vals, errs, colors = [], [], []
    for label, path, color in methods:
        m, s = _macro_across_cities_and_seeds(per_seed, path)
        vals.append(m); errs.append(s); colors.append(color)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, vals, 0.6, yerr=errs, capsize=4, color=colors, edgecolor="black",
           linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("WAPE (%)")
    ax.set_title("Scenario C: Cold-Start Generalization (leave-one-city)")
    ax.set_ylim(0, max(vals) * 1.15)

    # 严格零样本 (scaler 来自其余 5 城、不做校准) 因 SPO/JHB 尺度失配达到 ~426%,
    # 会压扁坐标轴, 故以文字标注保留诚实口径, 主柱用校准零样本 (模型迁移能力本身)。
    strict = _macro_across_cities_and_seeds(per_seed, ["zero_shot"])[0]
    ax.text(0.02, 0.97, f"strict zero-shot (uncalibrated scaler): {strict:.0f}%",
            transform=ax.transAxes, fontsize=9, va="top",
            color=COLORS["secondary"],
            bbox=dict(boxstyle="round", fc="white", ec=COLORS["secondary"], alpha=0.9))

    # 标注预训练增益：从 from-scratch(14d) 顶部垂直下降到 few-shot(14d) 水平
    ax.annotate("", xy=(3, vals[1]), xytext=(3, vals[3]),
                arrowprops=dict(arrowstyle="<->", color=COLORS["secondary"], linewidth=1.6))
    ax.text(3.35, (vals[1] + vals[3]) / 2, f"pre-training gain\n−{vals[3] - vals[1]:.0f} p.p.",
            ha="left", va="center", color=COLORS["secondary"], weight="bold", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(f"[fig4] {labels}")
    print(f"[fig4] WAPE: {[round(v, 1) for v in vals]}  ±  {[round(e, 1) for e in errs]}")


# ---------------------------------------------------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    org = _load_json(os.path.join(RESULTS_DIR, "organized_results.json"))
    loo = _load_json(os.path.join(RESULTS_DIR, "leave_one_out_3seed.json"))

    fig1_architecture(os.path.join(FIG_DIR, "fig1_architecture.png"))
    fig2_scenario_a(org, os.path.join(FIG_DIR, "fig2_scenario_a_ablation.png"))
    fig3_scenario_b(org, os.path.join(FIG_DIR, "fig3_scenario_b_alpha.png"))
    fig4_scenario_c(loo, os.path.join(FIG_DIR, "fig4_scenario_c_coldstart.png"))

    print("\nSaved 4 figures to:", FIG_DIR)


if __name__ == "__main__":
    main()
