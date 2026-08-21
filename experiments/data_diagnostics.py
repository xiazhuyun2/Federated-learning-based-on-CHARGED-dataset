"""
六城市数据诊断报告 — 回答"六个城市到底有多不均衡"、"哪些城市可以参与训练"

对每个城市生成完整统计表, 分两轮:
  1. 原始数据统计 (保留真实差异)
  2. 统一规则筛选后统计 (实际可用数据)

输出: outputs/diagnostics/
  - {city}_stats.json      每城市结构化统计
  - city_comparison.csv     六城市汇总对比表
  - city_comparison.png     六城市并排对比面板图

用法:
  python experiments/data_diagnostics.py
  python experiments/data_diagnostics.py --cities SZH,AMS
  python experiments/data_diagnostics.py --skip_raw
"""
import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import (
    load_city_data, select_top_stations, _parse_timestamps
)
from src.data.feature_engineering import prepare_station_data

# ============================================================
# 城市时区映射 (UTC偏移)
# ============================================================
CITY_TIMEZONES = {
    "SZH": 8,    # Asia/Shanghai
    "AMS": 2,    # Europe/Amsterdam (CEST in summer, CET+1)
    "JHB": 2,    # Africa/Johannesburg (SAST)
    "LOA": 1,    # America/Los_Angeles → 实际是 -7 或 -8? 按数据验证
    "MEL": 10,   # Australia/Melbourne (AEST)
    "SPO": -3,   # America/Sao_Paulo (BRT)
}

# 各城市币种 (用于电价分析)
CITY_CURRENCIES = {
    "SZH": "CNY",
    "AMS": "EUR",
    "JHB": "ZAR",
    "LOA": "USD",
    "MEL": "AUD",
    "SPO": "BRL",
}

# 共享特征列表 (所有城市weather CSV的交集)
SHARED_WEATHER_FEATURES = [
    "temp", "humidity", "windspeed", "precip",
    "cloudcover", "solarradiation", "pressure"
]


def diagnose_one_city(city: str, data_dir: str, skip_raw: bool = False) -> dict:
    """对单个城市执行完整诊断, 返回统计字典"""
    print(f"\n{'='*60}")
    print(f"  Diagnosing: {city}")
    print(f"{'='*60}")

    city_data = load_city_data(data_dir, city, use_remove_zero=True)
    volume = city_data["volume"]
    weather = city_data.get("weather")
    e_price = city_data.get("e_price")
    s_price = city_data.get("s_price")
    sites = city_data.get("sites")
    chargers = city_data.get("chargers")
    info = city_data.get("info")

    time_col = "Unnamed: 0"
    station_cols = [c for c in volume.columns if c != time_col]
    n_total_stations = len(station_cols)
    n_timesteps = len(volume)

    # ── 解析时间戳 ──
    timestamps = _parse_timestamps(volume[time_col], "volume")

    stats = {
        "city": city,
        "timezone": CITY_TIMEZONES.get(city, 0),
        "currency": CITY_CURRENCIES.get(city, "?"),
    }

    # ════════════════════════════════════════════════════════
    # 1. 数据规模
    # ════════════════════════════════════════════════════════
    stats["data_scale"] = {
        "total_stations": n_total_stations,
        "total_timesteps": n_timesteps,
        "time_range_start": str(timestamps.iloc[0]),
        "time_range_end": str(timestamps.iloc[-1]),
        "duration_hours": (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 3600,
    }

    # 有效站点筛选统计 (三步过滤的中间结果)
    train_ratio = 0.85
    n_train = max(int(n_timesteps * train_ratio), 1)
    train_vol = volume.iloc[:n_train]

    means = train_vol[station_cols].mean()
    stds = train_vol[station_cols].std()
    n_non_constant = int((stds >= 0.01).sum())
    n_constant = n_total_stations - n_non_constant

    non_const_means = means[stds >= 0.01]
    q1, q3 = non_const_means.quantile(0.25), non_const_means.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr
    n_normal = int(((non_const_means <= upper_bound) & (non_const_means >= 1.0)).sum())
    n_outlier = n_non_constant - n_normal

    zero_ratios = (train_vol[station_cols] < 0.01).sum() / n_train
    low_zero_mask = zero_ratios < 0.3
    n_low_zero = int(low_zero_mask.sum())
    n_high_zero = n_normal - n_low_zero

    stats["data_scale"]["filtering"] = {
        "constant_removed": n_constant,
        "outlier_removed": n_outlier,
        "high_zero_removed": n_high_zero,
        "effective_stations": n_low_zero,
        "effective_rate": round(n_low_zero / max(n_total_stations, 1) * 100, 1),
    }

    # 每站有效窗口数 (取样评估前20个有效站点)
    effective_stations = low_zero_mask[low_zero_mask].index.tolist()
    window_counts = []
    for sid in effective_stations[:min(20, len(effective_stations))]:
        try:
            df = _build_minimal_df(city_data, sid, time_col)
            train_ds, _, _, _ = prepare_station_data(
                df, seq_len=168, pred_len=24,
                train_ratio=0.7, val_ratio=0.15
            )
            window_counts.append(len(train_ds))
        except Exception:
            continue

    if window_counts:
        stats["data_scale"]["window_counts_sample"] = {
            "mean": round(np.mean(window_counts), 1),
            "median": round(np.median(window_counts), 1),
            "min": int(np.min(window_counts)),
            "max": int(np.max(window_counts)),
            "n_sampled": len(window_counts),
        }
        total_effective_windows = n_low_zero * np.mean(window_counts)
        stats["data_scale"]["estimated_total_windows"] = int(total_effective_windows)
    else:
        stats["data_scale"]["window_counts_sample"] = None
        stats["data_scale"]["estimated_total_windows"] = 0

    # ════════════════════════════════════════════════════════
    # 2. 时间完整性
    # ════════════════════════════════════════════════════════
    expected_hours = n_timesteps
    duplicates = timestamps.duplicated().sum()
    # 检查是否有缺小时 (按预期频率判断)
    if len(timestamps) >= 2:
        typical_gap = (timestamps.iloc[-1] - timestamps.iloc[0]) / (len(timestamps) - 1)
        # 找最大连续缺失
        diffs = timestamps.diff().dropna()
        max_gap = diffs.max()
        missing_gaps = diffs[diffs > typical_gap * 1.5]
        stats["temporal"] = {
            "duplicate_timestamps": int(duplicates),
            "max_gap_hours": round(max_gap.total_seconds() / 3600, 1) if pd.notna(max_gap) else 0,
            "gaps_larger_than_typical": len(missing_gaps),
            "typical_interval_minutes": round(typical_gap.total_seconds() / 60, 1),
        }

    # ════════════════════════════════════════════════════════
    # 3. 活跃程度
    # ════════════════════════════════════════════════════════
    zero_rates_stations = (train_vol[effective_stations[:min(50, len(effective_stations))]] < 0.01).mean()
    stats["activity"] = {
        "zero_rate_p25": round(float(zero_rates_stations.quantile(0.25)), 3),
        "zero_rate_p50": round(float(zero_rates_stations.quantile(0.50)), 3),
        "zero_rate_p75": round(float(zero_rates_stations.quantile(0.75)), 3),
        "zero_rate_p90": round(float(zero_rates_stations.quantile(0.90)), 3),
        "zero_rate_mean": round(float(zero_rates_stations.mean()), 3),
    }

    # 连续零值长度 (取中间站点)
    if effective_stations:
        mid_station = effective_stations[min(len(effective_stations)//2, len(effective_stations)-1)]
        mid_data = train_vol[mid_station].values
        zero_runs = _count_zero_runs(mid_data)
        stats["activity"]["max_consecutive_zeros_example"] = zero_runs.get("max_run", 0)
        stats["activity"]["active_ratio_example"] = round(1 - float((mid_data < 0.01).mean()), 3)

    # ════════════════════════════════════════════════════════
    # 4. 负荷规模 (训练期)
    # ════════════════════════════════════════════════════════
    train_means = train_vol[effective_stations[:min(50, len(effective_stations))]].mean()
    stats["load_scale"] = {
        "mean": round(float(train_means.mean()), 2),
        "median": round(float(train_means.median()), 2),
        "p75": round(float(train_means.quantile(0.75)), 2),
        "p90": round(float(train_means.quantile(0.90)), 2),
        "p95": round(float(train_means.quantile(0.95)), 2),
    }
    # 峰谷比
    peak_valley_ratios = []
    for sid in effective_stations[:min(20, len(effective_stations))]:
        data = train_vol[sid].values
        p90_val = np.percentile(data, 90)
        p10_val = np.percentile(data, 10)
        if p10_val > 0.01:
            peak_valley_ratios.append(p90_val / p10_val)
    if peak_valley_ratios:
        stats["load_scale"]["peak_valley_ratio_median"] = round(float(np.median(peak_valley_ratios)), 1)

    # ════════════════════════════════════════════════════════
    # 5. 周期性
    # ════════════════════════════════════════════════════════
    acf_24_list, acf_168_list = [], []
    daily_profiles = []
    for sid in effective_stations[:min(30, len(effective_stations))]:
        data = train_vol[sid].values
        if len(data) > 168:
            acf24 = np.corrcoef(data[:-24], data[24:])[0, 1]
            acf168 = np.corrcoef(data[:-168], data[168:])[0, 1]
            if not np.isnan(acf24): acf_24_list.append(acf24)
            if not np.isnan(acf168): acf_168_list.append(acf168)
        if len(data) >= 24:
            profile = np.array([np.mean(data[i::24]) for i in range(24)])
            if profile.max() > 0:
                profile = profile / profile.max()
            daily_profiles.append(profile)

    stats["periodicity"] = {
        "autocorr_24h_median": round(float(np.median(acf_24_list)), 3) if acf_24_list else None,
        "autocorr_24h_p25": round(float(np.percentile(acf_24_list, 25)), 3) if acf_24_list else None,
        "autocorr_24h_p75": round(float(np.percentile(acf_24_list, 75)), 3) if acf_24_list else None,
        "autocorr_168h_median": round(float(np.median(acf_168_list)), 3) if acf_168_list else None,
    }
    if daily_profiles:
        avg_profile = np.mean(daily_profiles, axis=0).tolist()
        stats["periodicity"]["avg_daily_profile"] = [round(v, 4) for v in avg_profile]

    # ════════════════════════════════════════════════════════
    # 6. 站点属性
    # ════════════════════════════════════════════════════════
    try:
        id_col = "site_id" if "site_id" in sites.columns else "site"
        site_stats = {}
        for col in ["charger_num", "avg_power", "perimeter"]:
            if col in sites.columns:
                vals = pd.to_numeric(sites[col], errors="coerce").dropna()
                if len(vals) > 0:
                    site_stats[col] = {
                        "mean": round(float(vals.mean()), 2),
                        "median": round(float(vals.median()), 2),
                        "p90": round(float(vals.quantile(0.90)), 2),
                    }
        stats["station_attributes"] = site_stats
        if chargers is not None:
            stats["station_attributes"]["total_chargers_in_city"] = len(chargers)
    except Exception as e:
        stats["station_attributes"] = {"error": str(e)}

    # ════════════════════════════════════════════════════════
    # 7. 特征质量
    # ════════════════════════════════════════════════════════
    feature_quality = {}

    # weather
    if weather is not None:
        wf_available = [f for f in SHARED_WEATHER_FEATURES if f in weather.columns]
        wf_missing = [f for f in SHARED_WEATHER_FEATURES if f not in weather.columns]
        wf_nan_rates = {}
        for f in wf_available:
            rate = weather[f].isna().mean()
            wf_nan_rates[f] = round(float(rate), 4)
        feature_quality["weather"] = {
            "available_features": wf_available,
            "missing_features": wf_missing,
            "nan_rates": wf_nan_rates,
        }

    # e_price
    if e_price is not None:
        price_cols = [c for c in e_price.columns if c not in ("Unnamed: 0", "time", "Time", "hour")]
        price_means = e_price[price_cols].mean()
        feature_quality["e_price"] = {
            "station_count": len(price_cols),
            "mean_price": round(float(price_means.mean()), 4),
            "price_std": round(float(price_means.std()), 4),
            "nan_rate": round(float(e_price[price_cols].isna().mean().mean()), 4),
        }

    # static features
    if sites is not None:
        static_cols = ["charger_num", "avg_power", "perimeter", "total_volume"]
        static_rates = {}
        for col in static_cols:
            if col in sites.columns:
                static_rates[col] = round(float(sites[col].isna().mean()), 4)
        feature_quality["static"] = static_rates

    stats["feature_quality"] = feature_quality

    # ════════════════════════════════════════════════════════
    # 8. 可用样本估算
    # ════════════════════════════════════════════════════════
    stats["usable_samples"] = {
        "effective_stations": n_low_zero,
        "estimated_train_windows": int(n_low_zero * np.mean(window_counts)) if window_counts else 0,
        "estimated_val_windows": int(n_low_zero * np.mean(window_counts) * 0.15 / 0.7) if window_counts else 0,
        "estimated_test_windows": int(n_low_zero * np.mean(window_counts) * 0.15 / 0.7) if window_counts else 0,
        "verdict": "",
    }
    effective_total = stats["usable_samples"]["estimated_train_windows"]
    if effective_total >= 50000:
        stats["usable_samples"]["verdict"] = "充足 — 适合作为训练城市"
    elif effective_total >= 10000:
        stats["usable_samples"]["verdict"] = "一般 — 可以参与训练但需关注"
    elif effective_total >= 2000:
        stats["usable_samples"]["verdict"] = "偏少 — 建议作为冷启动/外部测试城市"
    else:
        stats["usable_samples"]["verdict"] = "极少 — 必须作为冷启动/外部测试城市"

    return stats


def _build_minimal_df(city_data, station_id, time_col):
    """快速构建最小 DataFrame (仅用于窗口数估算) — 节省内存"""
    from src.data.data_loader import build_station_dataframe
    return build_station_dataframe(city_data, station_id, time_col)


def _count_zero_runs(data: np.ndarray, threshold: float = 0.01) -> dict:
    """统计连续零值长度"""
    is_zero = data < threshold
    runs = []
    current = 0
    for z in is_zero:
        if z:
            current += 1
        else:
            if current > 0:
                runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)
    active_ratio = 1 - is_zero.mean()
    return {
        "max_run": max(runs) if runs else 0,
        "mean_run": np.mean(runs) if runs else 0,
        "n_runs": len(runs),
        "active_ratio": round(float(active_ratio), 3),
    }


def generate_comparison_plots(all_stats: list, output_dir: str):
    """生成六城市并排对比图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    cities = [s["city"] for s in all_stats]
    colors = ["#1976D2", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    # 1. 有效站点数
    ax = axes[0]
    effective = [s["data_scale"]["filtering"]["effective_stations"] for s in all_stats]
    total = [s["data_scale"]["total_stations"] for s in all_stats]
    x = np.arange(len(cities))
    ax.bar(x - 0.2, total, 0.35, color="#BBDEFB", label="Total")
    ax.bar(x + 0.2, effective, 0.35, color=colors, label="Effective")
    ax.set_title("Station Count: Total vs Effective")
    ax.set_xticks(x)
    ax.set_xticklabels(cities)
    ax.legend(fontsize=8)

    # 2. 有效窗口数估算
    ax = axes[1]
    windows = [s["data_scale"].get("estimated_total_windows", 0) for s in all_stats]
    bars = ax.bar(cities, windows, color=colors, alpha=0.85)
    ax.set_title("Estimated Total Training Windows")
    ax.set_ylabel("Windows")
    for bar, val in zip(bars, windows):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f"{val/1000:.0f}k" if val > 1000 else str(val),
                ha="center", fontsize=8)

    # 3. 零值率分布
    ax = axes[2]
    zero_data = [s["activity"]["zero_rate_p50"] for s in all_stats]
    ax.bar(cities, [v*100 for v in zero_data], color=colors, alpha=0.85)
    ax.set_title("Median Zero Rate (%)")
    ax.set_ylabel("%")

    # 4. 负荷均值
    ax = axes[3]
    load_means = [s["load_scale"]["mean"] for s in all_stats]
    ax.bar(cities, load_means, color=colors, alpha=0.85)
    ax.set_title("Mean Load (training period)")
    ax.set_ylabel("Load")

    # 5. 24h自相关
    ax = axes[4]
    acf24 = [s["periodicity"].get("autocorr_24h_median") or 0 for s in all_stats]
    ax.bar(cities, acf24, color=colors, alpha=0.85)
    ax.set_title("Median 24h Autocorrelation")
    ax.set_ylabel("Correlation")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # 6. 168h自相关
    ax = axes[5]
    acf168 = [s["periodicity"].get("autocorr_168h_median") or 0 for s in all_stats]
    ax.bar(cities, acf168, color=colors, alpha=0.85)
    ax.set_title("Median 168h (Weekly) Autocorrelation")
    ax.set_ylabel("Correlation")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # 7. 标准化日负荷曲线 (叠加)
    ax = axes[6]
    for i, s in enumerate(all_stats):
        profile = s["periodicity"].get("avg_daily_profile")
        if profile:
            ax.plot(range(24), profile, color=colors[i], linewidth=1.5, label=s["city"])
    ax.set_title("Normalized Daily Load Profile")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Normalized Load")
    ax.legend(fontsize=7, ncol=2)

    # 8. 站点属性对比
    ax = axes[7]
    charger_nums = []
    for s in all_stats:
        attrs = s.get("station_attributes", {})
        charger_num = attrs.get("charger_num", {})
        charger_nums.append(charger_num.get("mean", 0) if isinstance(charger_num, dict) else 0)
    ax.bar(cities, charger_nums, color=colors, alpha=0.85)
    ax.set_title("Mean Chargers per Station")

    # 9. 决策摘要表
    ax = axes[8]
    ax.axis("off")
    table_data = [["City", "Effective\nStations", "Est. Windows", "Verdict"]]
    for s in all_stats:
        table_data.append([
            s["city"],
            str(s["data_scale"]["filtering"]["effective_stations"]),
            f"{s['data_scale'].get('estimated_total_windows', 0)//1000}k",
            s["usable_samples"]["verdict"],
        ])
    tbl = ax.table(cellText=table_data, cellLoc="center", loc="center",
                   colWidths=[0.12, 0.12, 0.12, 0.45])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    ax.set_title("Decision Summary", fontsize=12, fontweight="bold", y=1.02)

    fig.suptitle("CHARGED Dataset: 6-City Data Diagnostics",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "city_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="六城市数据诊断报告")
    parser.add_argument("--cities", type=str, default="SZH,AMS,JHB,LOA,MEL,SPO",
                        help="逗号分隔的城市列表")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))), "data"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_raw", action="store_true")
    args = parser.parse_args()

    cities = [c.strip() for c in args.cities.split(",")]
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "diagnostics")
    os.makedirs(output_dir, exist_ok=True)

    all_stats = []
    for city in cities:
        try:
            stats = diagnose_one_city(city, args.data_dir, args.skip_raw)
            all_stats.append(stats)

            # 保存单城市 JSON
            json_path = os.path.join(output_dir, f"{city}_stats.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
            print(f"  Saved: {json_path}")
        except Exception as e:
            print(f"  ERROR diagnosing {city}: {e}")
            import traceback
            traceback.print_exc()

    if not all_stats:
        print("\n  No cities diagnosed successfully.")
        return

    # ════════════════════════════════════════════════════════
    # 汇总 CSV
    # ════════════════════════════════════════════════════════
    csv_path = os.path.join(output_dir, "city_comparison.csv")
    rows = []
    for s in all_stats:
        ds = s["data_scale"]
        f = ds["filtering"]
        rows.append({
            "City": s["city"],
            "Timezone": s["timezone"],
            "Currency": s["currency"],
            "Total_Stations": ds["total_stations"],
            "Effective_Stations": f["effective_stations"],
            "Effective_Rate_%": f["effective_rate"],
            "Const_Removed": f["constant_removed"],
            "Outlier_Removed": f["outlier_removed"],
            "HighZero_Removed": f["high_zero_removed"],
            "Est_Train_Windows": ds.get("estimated_total_windows", 0),
            "ZeroRate_Median_%": round(s["activity"].get("zero_rate_p50", 0) * 100, 1),
            "Load_Mean": s["load_scale"]["mean"],
            "Load_PeakValley": s["load_scale"].get("peak_valley_ratio_median", 0),
            "ACF_24h": s["periodicity"].get("autocorr_24h_median") or 0,
            "ACF_168h": s["periodicity"].get("autocorr_168h_median") or 0,
            "Verdict": s["usable_samples"]["verdict"],
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ════════════════════════════════════════════════════════
    # 生成汇总图表
    # ════════════════════════════════════════════════════════
    generate_comparison_plots(all_stats, output_dir)

    # ════════════════════════════════════════════════════════
    # 打印关键发现
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  KEY FINDINGS")
    print(f"{'='*60}")
    for s in all_stats:
        print(f"  {s['city']}: {s['data_scale']['filtering']['effective_stations']} "
              f"effective stations, ~{s['data_scale'].get('estimated_total_windows', 0)//1000}k "
              f"train windows → {s['usable_samples']['verdict']}")

    # 识别冷启动候选
    cold_start = [s for s in all_stats
                  if s["data_scale"].get("estimated_total_windows", 0) < 10000]
    if cold_start:
        print(f"\n  Cold-start candidates (<10k windows): "
              f"{[s['city'] for s in cold_start]}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
