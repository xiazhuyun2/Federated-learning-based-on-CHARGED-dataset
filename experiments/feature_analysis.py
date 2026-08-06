"""
特征相关性分析 — Spearman秩相关 + 互信息(MI) + 可视化

用途: 验证当前选用的特征与充电负荷(target)之间的相关性,
      为特征筛选提供数据依据, 论文中可作为特征工程章节的支撑图表。

运行: python experiments/feature_analysis.py --city SZH --top_k 5
输出: outputs/spearman_heatmap.png, outputs/mic_barplot.png, outputs/feature_scatter.png
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

from src.data.data_loader import load_city_data, select_top_stations, build_station_dataframe
from config import DATA_DIR


def spearman_analysis(df: pd.DataFrame, output_dir: str):
    """
    Spearman 秩相关系数矩阵 + 热力图
    Spearman 相比 Pearson 能捕捉非线性单调关系, 更适合气象-负荷关系
    """
    feature_cols = [c for c in df.columns if c not in ("timestamp", "target")]
    analysis_df = df[["target"] + feature_cols].copy()

    # 计算 Spearman 相关矩阵
    corr_matrix = analysis_df.corr(method="spearman")

    # 提取 target 行并排序
    target_corr = corr_matrix["target"].drop("target").sort_values(
        key=abs, ascending=False)

    print("\n" + "=" * 60)
    print("  Spearman Rank Correlation with Target (charging load)")
    print("=" * 60)
    for feat, val in target_corr.items():
        bar = "+" * int(abs(val) * 30)
        sign = "+" if val > 0 else "-"
        print(f"  {feat:20s}  {val:+.4f}  {sign}{bar}")

    # 热力图
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Spearman Correlation"}
    )
    ax.set_title("Spearman Rank Correlation Matrix\n(Features vs Charging Load)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "spearman_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    return target_corr


def mutual_info_analysis(df: pd.DataFrame, output_dir: str):
    """
    互信息 (Mutual Information) 分析
    MI 能度量任意类型的统计依赖 (不仅限于线性/单调), 特别适合评估
    离散变量 (is_weekend, conditions) 与连续目标的关系
    """
    feature_cols = [c for c in df.columns if c not in ("timestamp", "target")]
    X = df[feature_cols].values
    y = df["target"].values

    # 标记离散特征
    discrete_mask = np.array([
        col in ("is_weekend",) for col in feature_cols
    ])

    # 计算互信息 (多次取平均, 减少随机性)
    mi_scores = np.zeros(len(feature_cols))
    n_repeats = 5
    for _ in range(n_repeats):
        mi = mutual_info_regression(
            X, y, discrete_features=discrete_mask, random_state=None, n_neighbors=5
        )
        mi_scores += mi
    mi_scores /= n_repeats

    # 排序
    mi_df = pd.DataFrame({
        "feature": feature_cols,
        "MI": mi_scores
    }).sort_values("MI", ascending=False)

    print("\n" + "=" * 60)
    print("  Mutual Information Score (feature → target)")
    print("=" * 60)
    for _, row in mi_df.iterrows():
        bar = "#" * int(row["MI"] / mi_df["MI"].max() * 30)
        print(f"  {row['feature']:20s}  {row['MI']:.4f}  {bar}")

    # 条形图
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3" if mi > mi_df["MI"].median() else "#90CAF9"
              for mi in mi_df["MI"]]
    ax.barh(mi_df["feature"], mi_df["MI"], color=colors, edgecolor="white")
    ax.set_xlabel("Mutual Information Score", fontsize=12)
    ax.set_title("Mutual Information: Features → Charging Load\n"
                 "(Higher = More Predictive)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    # 标注中位线
    median_mi = mi_df["MI"].median()
    ax.axvline(x=median_mi, color="red", linestyle="--", alpha=0.7, label=f"Median={median_mi:.3f}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "mic_barplot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    return mi_df


def scatter_analysis(df: pd.DataFrame, output_dir: str, top_n: int = 6):
    """
    Top-N 特征与 target 的散点图矩阵
    直观展示特征-负荷关系的形态 (线性/非线性/无关)
    """
    feature_cols = [c for c in df.columns if c not in ("timestamp", "target")]

    # 选 Spearman 相关最高的 top_n 个特征
    corrs = {}
    for col in feature_cols:
        rho, _ = stats.spearmanr(df[col], df["target"])
        corrs[col] = abs(rho)

    top_features = sorted(corrs, key=corrs.get, reverse=True)[:top_n]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        # 降采样绘图 (太多点会很慢)
        sample_idx = np.random.choice(len(df), min(2000, len(df)), replace=False)
        ax.scatter(
            df[feat].iloc[sample_idx], df["target"].iloc[sample_idx],
            alpha=0.3, s=5, c="#1976D2"
        )
        rho = corrs[feat]
        ax.set_xlabel(feat, fontsize=11)
        ax.set_ylabel("Charging Load", fontsize=11)
        ax.set_title(f"{feat} (|ρ|={rho:.3f})", fontsize=12, fontweight="bold")

    fig.suptitle("Top Features vs Charging Load (Scatter)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")


def print_recommendations(spearman_corr, mi_df):
    """根据分析结果输出特征选择建议"""
    print("\n" + "=" * 60)
    print("  FEATURE SELECTION RECOMMENDATIONS")
    print("=" * 60)

    # 强相关特征 (|rho| > 0.1)
    strong = [f for f, v in spearman_corr.items() if abs(v) > 0.1]
    weak = [f for f, v in spearman_corr.items() if abs(v) <= 0.05]

    # MI 高分特征
    mi_threshold = mi_df["MI"].median()
    high_mi = mi_df[mi_df["MI"] > mi_threshold]["feature"].tolist()

    print(f"\n  Strong Spearman correlation (|ρ|>0.1): {strong}")
    print(f"  Weak Spearman correlation  (|ρ|≤0.05): {weak}")
    print(f"  High mutual information (>median):  {high_mi}")

    # 建议保留: Spearman 或 MI 至少一个强
    keep = set(strong) | set(high_mi)
    remove = set(spearman_corr.index) - keep

    print(f"\n  >> KEEP ({len(keep)}): {sorted(keep)}")
    print(f"  >> CONSIDER REMOVING ({len(remove)}): {sorted(remove)}")
    print(f"\n  Note: Cyclical encodings (sin/cos) may show low linear correlation")
    print(f"        but capture important periodic patterns. Keep them.")


def main():
    parser = argparse.ArgumentParser(description="Feature Correlation Analysis")
    parser.add_argument("--city", default="SZH", help="City to analyze")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Analyze top-k stations and average")
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Feature Correlation Analysis — City: {args.city}")
    print("=" * 60)

    # 加载数据
    city_data = load_city_data(DATA_DIR, args.city, use_remove_zero=True)
    stations = select_top_stations(city_data["volume"], "Unnamed: 0", args.top_k)

    # 合并多个站点数据做分析 (更稳健)
    all_dfs = []
    for sid in stations:
        df = build_station_dataframe(city_data, sid)
        all_dfs.append(df)
        print(f"  Loaded station {sid}: {len(df)} samples, "
              f"mean load = {df['target'].mean():.1f}")

    # 使用第一个站点做详细分析 (热力图), 用所有站点做 MI 分析
    main_df = all_dfs[0]

    # 1. Spearman 分析
    spearman_corr = spearman_analysis(main_df, output_dir)

    # 2. 互信息分析
    mi_df = mutual_info_analysis(main_df, output_dir)

    # 3. 散点图
    scatter_analysis(main_df, output_dir)

    # 4. 综合建议
    print_recommendations(spearman_corr, mi_df)

    print("\n  Analysis complete! Check outputs/ for plots.")


if __name__ == "__main__":
    main()
