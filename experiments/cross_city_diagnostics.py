"""
跨城市诊断实验 — 回答"多城市FL是否比单城市FL更好"、"哪些城市之间有正/负迁移"

三大实验:
  1. 城市间负荷模式相似性 (日曲线/周曲线/自相关/Wasserstein距离)
  2. 6×6 跨城市迁移矩阵
  3. 单城市 vs 多城市预实验

用法:
  python experiments/cross_city_diagnostics.py                    # 完整诊断
  python experiments/cross_city_diagnostics.py --quick            # 快速预实验
  python experiments/cross_city_diagnostics.py --task similarity  # 仅相似性
  python experiments/cross_city_diagnostics.py --task transfer    # 仅迁移矩阵
  python experiments/cross_city_diagnostics.py --task comparison  # 仅对比实验
"""
import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import (
    load_city_data, select_top_stations, build_station_dataframe
)
from src.data.feature_engineering import prepare_station_data
from src.models.tcn_lstm import build_model
from src.utils.metrics import evaluate_model, set_seed

ALL_CITIES = ["SZH", "AMS", "JHB", "LOA", "MEL", "SPO"]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ═══════════════════════════════════════════════════════════════
# Task 1: 城市间负荷模式相似性
# ═══════════════════════════════════════════════════════════════

def extract_city_profiles(city: str, top_k: int = 30) -> dict:
    """提取单个城市的负荷模式特征"""
    print(f"\n  Extracting profiles for {city}...")
    city_data = load_city_data(DATA_DIR, city, use_remove_zero=True)
    volume = city_data["volume"]
    time_col = "Unnamed: 0"
    station_cols = [c for c in volume.columns if c != time_col]

    # 取训练期数据
    n_train = int(len(volume) * 0.85)
    train_vol = volume.iloc[:n_train]

    # 选有效站点
    stations = select_top_stations(
        city_data["volume"], time_col, min(top_k, len(station_cols)),
        train_ratio=0.85
    )
    if not stations:
        return {}

    daily_profiles = []
    weekly_profiles = []
    acf_24_list, acf_168_list = [], []

    for sid in stations[:top_k]:
        data = train_vol[sid].values
        # 日曲线
        if len(data) >= 24:
            profile_24 = np.array([np.mean(data[i::24]) for i in range(24)])
            if profile_24.max() > 0.01:
                profile_24 = profile_24 / profile_24.max()
            daily_profiles.append(profile_24)
        # 周曲线
        if len(data) >= 168:
            profile_168 = np.array([np.mean(data[i::168]) for i in range(168)])
            if profile_168.max() > 0.01:
                profile_168 = profile_168 / profile_168.max()
            weekly_profiles.append(profile_168)
        # 自相关
        if len(data) > 168:
            acf_24 = np.corrcoef(data[:-24], data[24:])[0, 1]
            acf_168 = np.corrcoef(data[:-168], data[168:])[0, 1]
            if not np.isnan(acf_24): acf_24_list.append(acf_24)
            if not np.isnan(acf_168): acf_168_list.append(acf_168)

    return {
        "city": city,
        "n_stations": len(stations),
        "avg_daily_profile": np.mean(daily_profiles, axis=0).tolist() if daily_profiles else [],
        "avg_weekly_profile": np.mean(weekly_profiles, axis=0).tolist() if weekly_profiles else [],
        "acf_24h": float(np.median(acf_24_list)) if acf_24_list else 0,
        "acf_168h": float(np.median(acf_168_list)) if acf_168_list else 0,
        "acf_24h_std": float(np.std(acf_24_list)) if acf_24_list else 0,
        "acf_168h_std": float(np.std(acf_168_list)) if acf_168_list else 0,
    }


def compute_wasserstein_matrix(profiles: dict) -> np.ndarray:
    """计算6×6 Wasserstein 距离矩阵"""
    from scipy.stats import wasserstein_distance
    cities = sorted(profiles.keys())
    n = len(cities)
    matrix = np.zeros((n, n))

    for i, ci in enumerate(cities):
        for j, cj in enumerate(cities):
            if i == j:
                matrix[i, j] = 0.0
                continue
            # 用各城市的平均周曲线计算 Wasserstein 距离
            pw = profiles[ci].get("avg_weekly_profile", [])
            qw = profiles[cj].get("avg_weekly_profile", [])
            if pw and qw:
                matrix[i, j] = wasserstein_distance(pw, qw)
            else:
                matrix[i, j] = float("nan")

    return matrix, cities


def plot_similarity_analysis(profiles: dict, wasserstein: np.ndarray,
                              cities_order: list, output_dir: str):
    """绘制城市相似性分析图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#1976D2", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
    n = len(cities_order)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 1. 标准化日曲线
    ax = axes[0]
    for i, c in enumerate(cities_order):
        p = profiles[c].get("avg_daily_profile", [])
        if p:
            ax.plot(range(24), p, color=colors[i], linewidth=2, label=c)
    ax.set_title("Normalized Daily Load Profile")
    ax.set_xlabel("Hour"); ax.set_ylabel("Normalized Load")
    ax.legend(fontsize=8)

    # 2. 标准化周曲线
    ax = axes[1]
    for i, c in enumerate(cities_order):
        p = profiles[c].get("avg_weekly_profile", [])
        if p:
            ax.plot(range(min(168, len(p))), p[:168],
                    color=colors[i], linewidth=1.5, alpha=0.8, label=c)
    ax.set_title("Normalized Weekly Load Profile")
    ax.set_xlabel("Hour of Week"); ax.set_ylabel("Normalized Load")
    ax.legend(fontsize=7, ncol=2)

    # 3. 24h自相关
    ax = axes[2]
    acf24_vals = [profiles[c]["acf_24h"] for c in cities_order]
    ax.bar(cities_order, acf24_vals, color=colors, alpha=0.85)
    ax.set_title("24h Autocorrelation")
    ax.set_ylabel("Correlation")

    # 4. 168h自相关
    ax = axes[3]
    acf168_vals = [profiles[c]["acf_168h"] for c in cities_order]
    ax.bar(cities_order, acf168_vals, color=colors, alpha=0.85)
    ax.set_title("168h (Weekly) Autocorrelation")
    ax.set_ylabel("Correlation")

    # 5. Wasserstein 距离热力图
    ax = axes[4]
    im = ax.imshow(wasserstein, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(cities_order); ax.set_yticklabels(cities_order)
    for i in range(n):
        for j in range(n):
            val = wasserstein[i, j]
            if not np.isnan(val) and val > 0:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if val > np.nanmax(wasserstein)*0.5 else "black")
    ax.set_title("Wasserstein Distance Matrix")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 6. Summary text
    ax = axes[5]
    ax.axis("off")
    summary_lines = ["City Profile Summary"]
    for i, c in enumerate(cities_order):
        p = profiles[c]
        summary_lines.append(
            f"{c}: {p['n_stations']} stations, "
            f"ACF24={p['acf_24h']:.3f}, ACF168={p['acf_168h']:.3f}"
        )
    ax.text(0.05, 0.95, "\n".join(summary_lines), fontsize=9,
            va="top", family="monospace", transform=ax.transAxes)

    fig.suptitle("Cross-City Load Pattern Similarity Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "city_similarity.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def run_similarity_task(output_dir: str):
    """任务1: 城市负荷模式相似性"""
    print(f"\n{'='*60}")
    print("  Task 1: Cross-City Load Pattern Similarity")
    print(f"{'='*60}")

    profiles = {}
    for city in ALL_CITIES:
        try:
            profiles[city] = extract_city_profiles(city, top_k=30)
        except Exception as e:
            print(f"  Failed to extract {city}: {e}")

    wasserstein, cities_order = compute_wasserstein_matrix(profiles)

    # 保存JSON
    json_path = os.path.join(output_dir, "city_similarity.json")
    serializable = {}
    for c, p in profiles.items():
        serializable[c] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in p.items()
        }
    serializable["wasserstein_matrix"] = {
        "cities": cities_order,
        "matrix": wasserstein.tolist(),
    }
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    plot_similarity_analysis(profiles, wasserstein, cities_order, output_dir)

    # 打印关键发现
    print(f"\n  Similarity Findings:")
    for c in cities_order:
        p = profiles.get(c, {})
        print(f"    {c}: ACF24={p.get('acf_24h', 0):.3f}, "
              f"ACF168={p.get('acf_168h', 0):.3f}")


# ═══════════════════════════════════════════════════════════════
# Task 2: 6×6 跨城市迁移矩阵
# ═══════════════════════════════════════════════════════════════

def train_on_city_test_on_city(train_city: str, test_city: str,
                                top_k: int = 10, epochs: int = 30,
                                seed: int = 42) -> dict:
    """用 train_city 数据训练模型, 在 test_city 上评估"""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载训练城市数据
    train_data = load_city_data(DATA_DIR, train_city, use_remove_zero=True)
    test_data = load_city_data(DATA_DIR, test_city, use_remove_zero=True)

    train_stations = select_top_stations(
        train_data["volume"], "Unnamed: 0", top_k, train_ratio=0.85
    )
    test_stations = select_top_stations(
        test_data["volume"], "Unnamed: 0",
        min(top_k, len([c for c in test_data["volume"].columns if c != "Unnamed: 0"])),
        train_ratio=0.85
    )

    # 合并训练城市的所有站点数据
    train_datasets = []
    for sid in train_stations:
        try:
            df = build_station_dataframe(train_data, sid)
            train_ds, _, _, _ = prepare_station_data(df)
            if len(train_ds) > 0:
                train_datasets.append(train_ds)
        except Exception:
            continue

    if not train_datasets:
        return {"RMSE": float("inf"), "MAE": float("inf")}

    merged_train = ConcatDataset(train_datasets)
    train_loader = DataLoader(merged_train, batch_size=64, shuffle=True)

    # 训练
    model = build_model(
        train_datasets[0][0][0].shape[1], 24,
        type("M", (), {"tcn_channels": [64,64,64], "tcn_kernel_size": 3,
                        "tcn_dropout": 0.2, "lstm_hidden": 64,
                        "lstm_layers": 2, "lstm_dropout": 0.2,
                        "fc_hidden": 64, "use_fedbn": False,
                        "use_local_head": False})()
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss={total_loss/len(train_loader):.4f}")
    model.eval()

    # 在测试城市的每个站点上评估
    test_results = []
    for tsid in test_stations:
        try:
            df = build_station_dataframe(test_data, tsid)
            _, _, test_ds, scaler = prepare_station_data(df)
            if len(test_ds) == 0:
                continue
            test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
            metrics = evaluate_model(model, test_loader, scaler, device)
            test_results.append(metrics)
        except Exception:
            continue

    if not test_results:
        return {"RMSE": float("inf"), "MAE": float("inf")}

    return {
        "RMSE": float(np.mean([m["RMSE"] for m in test_results])),
        "MAE": float(np.mean([m["MAE"] for m in test_results])),
        "WAPE": float(np.mean([m.get("WAPE", 0) for m in test_results])),
        "n_tested": len(test_results),
    }


def run_transfer_task(output_dir: str, top_k: int = 10,
                       epochs: int = 20, cities: list = None):
    """任务2: 6×6 跨城市迁移矩阵"""
    cities = cities or ALL_CITIES
    print(f"\n{'='*60}")
    print(f"  Task 2: 6×6 Cross-City Transfer Matrix")
    print(f"  ({len(cities)} cities, top_k={top_k}, epochs={epochs})")
    print(f"{'='*60}")

    n = len(cities)
    rmse_matrix = np.zeros((n, n))
    mae_matrix = np.zeros((n, n))

    for i, train_c in enumerate(cities):
        for j, test_c in enumerate(cities):
            print(f"\n  [{i+1},{j+1}] Train={train_c} → Test={test_c}")
            try:
                result = train_on_city_test_on_city(
                    train_c, test_c, top_k=top_k, epochs=epochs
                )
                rmse_matrix[i, j] = result["RMSE"]
                mae_matrix[i, j] = result["MAE"]
                print(f"    RMSE={result['RMSE']:.4f}, MAE={result['MAE']:.4f}")
            except Exception as e:
                print(f"    Failed: {e}")
                rmse_matrix[i, j] = float("inf")
                mae_matrix[i, j] = float("inf")

    # 保存
    result = {
        "cities": cities,
        "rmse_matrix": rmse_matrix.tolist(),
        "mae_matrix": mae_matrix.tolist(),
    }
    json_path = os.path.join(output_dir, "transfer_matrix.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # 绘图
    plot_transfer_matrix(rmse_matrix, cities, output_dir)

    # 分析
    print(f"\n  Transfer Analysis:")
    for j, test_c in enumerate(cities):
        diag_val = rmse_matrix[j, j]
        best_val = np.min(rmse_matrix[:, j])
        best_train = cities[np.argmin(rmse_matrix[:, j])]
        if best_train != test_c:
            improvement = (diag_val - best_val) / diag_val * 100
            print(f"    {test_c}: self RMSE={diag_val:.2f}, "
                  f"best transfer={best_train} ({best_val:.2f}, "
                  f"{improvement:+.1f}%)")

    return result


def plot_transfer_matrix(matrix: np.ndarray, cities: list, output_dir: str):
    """绘制迁移矩阵热力图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(cities)
    # 对角线归一化: 每列除以对角线值
    relative = matrix.copy()
    for j in range(n):
        if matrix[j, j] > 0:
            relative[:, j] = matrix[:, j] / matrix[j, j]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左: 绝对 RMSE
    im1 = ax1.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax1.set_title("Absolute RMSE (lower = better)")
    ax1.set_xticks(range(n)); ax1.set_yticks(range(n))
    ax1.set_xticklabels(cities); ax1.set_yticklabels(cities)
    ax1.set_ylabel("Train City"); ax1.set_xlabel("Test City")
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if np.isfinite(val) and val < 1e6:
                ax1.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # 右: 相对 RMSE (1.0 = 与本地训练相当)
    im2 = ax2.imshow(relative, cmap="RdYlGn_r", aspect="auto", vmin=0.5, vmax=2.0)
    ax2.set_title("Relative RMSE (1.0 = self-train, <1 = positive transfer)")
    ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
    ax2.set_xticklabels(cities); ax2.set_yticklabels(cities)
    ax2.set_ylabel("Train City"); ax2.set_xlabel("Test City")
    for i in range(n):
        for j in range(n):
            val = relative[i, j]
            if np.isfinite(val):
                ax2.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle("Cross-City Transfer Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "transfer_matrix.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# Task 3: 单城市 vs 多城市 预实验对比
# ═══════════════════════════════════════════════════════════════

def run_comparison_task(output_dir: str, top_k: int = 5,
                         rounds: int = 10, cities: list = None):
    """任务3: 快速对比 单城市FL vs 多城市FL"""
    cities = cities or ALL_CITIES
    print(f"\n{'='*60}")
    print(f"  Task 3: Single-City vs Multi-City Pilot Comparison")
    print(f"  ({len(cities)} cities, top_k={top_k}, rounds={rounds})")
    print(f"{'='*60}")

    results = {}

    # 3a. 每个城市单独FL
    print(f"\n  --- 3a: Single-city FL (baseline) ---")
    for city in cities:
        print(f"\n  Running single-city FL: {city}")
        ret = os.system(
            f"python main.py --city {city} --aggregation fedprox "
            f"--top_k {top_k} --num_rounds {rounds} --local_epochs 3 "
            f"--output_dir {output_dir}/pilot_single "
            f"2>&1 | tail -5"
        )
        if ret != 0:
            print(f"  WARNING: {city} single-city FL failed")

    # 3b. 多城市普通 FedAvg (α=1)
    print(f"\n  --- 3b: Multi-city FedAvg (α=1.0, standard weighting) ---")
    ret = os.system(
        f"python main.py --cities {','.join(cities)} "
        f"--aggregation fedprox --top_k {top_k} --num_rounds {rounds} "
        f"--local_epochs 3 --city_weight_alpha 1.0 "
        f"--output_dir {output_dir}/pilot_alpha1 "
        f"2>&1 | tail -10"
    )

    # 3c. 多城市平衡聚合 (α=0.5)
    print(f"\n  --- 3c: Multi-city Balanced (α=0.5) ---")
    ret = os.system(
        f"python main.py --cities {','.join(cities)} "
        f"--aggregation fedprox --top_k {top_k} --num_rounds {rounds} "
        f"--local_epochs 3 --city_weight_alpha 0.5 "
        f"--output_dir {output_dir}/pilot_alpha05 "
        f"2>&1 | tail -10"
    )

    # 3d. 多城市平衡 + 本地预测头 (α=0.5 + local_head)
    print(f"\n  --- 3d: Multi-city Balanced + Local Head (α=0.5) ---")
    ret = os.system(
        f"python main.py --cities {','.join(cities)} "
        f"--aggregation fedprox --top_k {top_k} --num_rounds {rounds} "
        f"--local_epochs 3 --city_weight_alpha 0.5 --local_head "
        f"--output_dir {output_dir}/pilot_alpha05_lh "
        f"2>&1 | tail -10"
    )

    # 收集并比较结果
    print(f"\n  --- Pilot Results Summary ---")
    collect_and_compare(output_dir, cities)


def collect_and_compare(output_dir: str, cities: list):
    """收集各实验的结果并汇总"""
    import glob

    pilot_dirs = [
        ("single_FL", f"{output_dir}/pilot_single"),
        ("FedAvg_α1", f"{output_dir}/pilot_alpha1"),
        ("Balanced_α0.5", f"{output_dir}/pilot_alpha05"),
        ("Balanced+LocalH", f"{output_dir}/pilot_alpha05_lh"),
    ]

    print(f"\n{'='*80}")
    print(f"  PILOT COMPARISON SUMMARY")
    print(f"{'='*80}")

    for label, base_dir in pilot_dirs:
        metrics_files = glob.glob(
            os.path.join(base_dir, "*", "*", "seed_*", "run_*", "metrics.json")
        )
        if not metrics_files:
            print(f"  {label:<20s}: No results found")
            continue

        latest = sorted(metrics_files)[-1]
        try:
            with open(latest) as f:
                data = json.load(f)
            macro = data.get("macro_city", data.get("AVERAGE", {}))
            rmse = macro.get("RMSE", "N/A")
            wape = macro.get("WAPE", "N/A")
            print(f"  {label:<20s}: RMSE={rmse}, WAPE={wape}")

            # 如果有 per_city 数据
            per_city = data.get("per_city", {})
            for c in sorted(per_city.keys()):
                city_m = per_city[c]
                print(f"    {c}: RMSE={city_m.get('RMSE', 'N/A'):.1f}, "
                      f"WAPE={city_m.get('WAPE', 'N/A'):.1f}%")
        except Exception as e:
            print(f"  {label:<20s}: Parse error: {e}")

    print(f"{'='*80}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="跨城市诊断实验")
    parser.add_argument("--task", type=str, default="all",
                        choices=["similarity", "transfer", "comparison", "all"],
                        help="选择要执行的任务")
    parser.add_argument("--cities", type=str, default="SZH,AMS,JHB,LOA,MEL,SPO",
                        help="逗号分隔的城市列表")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20,
                        help="迁移矩阵实验中统一训练的epoch数")
    parser.add_argument("--rounds", type=int, default=10,
                        help="对比实验中的FL轮数")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式: top_k=5, epochs=10, rounds=5")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cities = [c.strip() for c in args.cities.split(",")]
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "diagnostics")
    os.makedirs(output_dir, exist_ok=True)

    if args.quick:
        args.top_k = 5
        args.epochs = 10
        args.rounds = 5
        print("  Quick mode: smaller scale for fast iteration")

    if args.task in ("similarity", "all"):
        run_similarity_task(output_dir)

    if args.task in ("transfer", "all"):
        run_transfer_task(output_dir, top_k=args.top_k,
                           epochs=args.epochs, cities=cities)

    if args.task in ("comparison", "all"):
        run_comparison_task(output_dir, top_k=args.top_k,
                             rounds=args.rounds, cities=cities)

    print(f"\n  All diagnostics saved to {output_dir}")


if __name__ == "__main__":
    main()
