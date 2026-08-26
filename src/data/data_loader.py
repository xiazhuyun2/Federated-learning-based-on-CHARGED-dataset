"""
数据加载器 — 读取 UrbanEV 数据集, 构建站点级时序数据
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def load_city_data(data_dir: str, city: str, use_remove_zero: bool = True) -> Dict:
    """
    加载单个城市的所有数据文件
    Returns: dict with keys: volume, weather, e_price, s_price, poi, sites, chargers, info
    """
    folder = f"{city}_remove_zero" if use_remove_zero else city
    city_dir = os.path.join(data_dir, folder)
    if not os.path.isdir(city_dir):
        city_dir = os.path.join(data_dir, city)  # fallback to non-remove-zero

    if not os.path.isdir(city_dir):
        raise FileNotFoundError(f"Data directory not found: {city_dir}")

    data = {}
    data["volume"] = pd.read_csv(os.path.join(city_dir, "volume.csv"))
    data["weather"] = pd.read_csv(os.path.join(city_dir, "weather.csv"))
    data["e_price"] = pd.read_csv(os.path.join(city_dir, "e_price.csv"))
    data["s_price"] = pd.read_csv(os.path.join(city_dir, "s_price.csv"))

    # 静态数据 (poi/sites/chargers/info 也在 _remove_zero 目录中)
    data["poi"] = pd.read_csv(os.path.join(city_dir, "poi.csv"))
    data["sites"] = pd.read_csv(os.path.join(city_dir, "sites.csv"))
    data["chargers"] = pd.read_csv(os.path.join(city_dir, "chargers.csv"))
    data["info"] = pd.read_csv(os.path.join(city_dir, "info.csv"))

    return data


def load_all_cities(data_dir: str, cities: list,
                    use_remove_zero: bool = True) -> Dict[str, Dict]:
    """
    批量加载多个城市的数据
    Returns: {city: city_data_dict}
    """
    result = {}
    for city in cities:
        try:
            result[city] = load_city_data(data_dir, city, use_remove_zero)
        except Exception as e:
            print(f"  WARNING: Failed to load {city}: {e}")
    return result


def select_top_stations(volume_df: pd.DataFrame, time_col: str, k: int,
                       train_ratio: float = 0.85) -> List[str]:
    """
    数据清洗 + 选站: 三步过滤后按总充电量选取 top-k 站点

    为避免测试集泄漏, 所有统计量仅基于训练时间段计算。

    过滤规则:
      1. 剔除恒定值站点 (std < 0.01): 6个月充电量完全不变, 无法学习模式
      2. 剔除异常大值站点 (IQR法): mean > Q3 + 3*IQR, 量级远超正常站点
      3. 剔除极稀疏站点 (mean < 1): 几乎无充电活动

    Args:
        train_ratio: 训练期占比, 统计量仅基于此前缀行计算 (默认 0.85)
    """
    station_cols = [c for c in volume_df.columns if c != time_col]

    # 只使用训练时间段的数据计算统计量 (避免测试集泄漏)
    n_train = max(int(len(volume_df) * train_ratio), 1)
    train_volume = volume_df.iloc[:n_train]

    # 计算每个站点的统计量 (仅训练期)
    means = train_volume[station_cols].mean()
    stds = train_volume[station_cols].std()

    # Step 1: 剔除恒定值站点 (std < 0.01)
    non_constant = stds[stds >= 0.01].index.tolist()

    # Step 2: 在非恒定站点中, 用 IQR 法剔除异常大值
    non_const_means = means[non_constant]
    q1 = non_const_means.quantile(0.25)
    q3 = non_const_means.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr  # 3倍 IQR, 宽松阈值
    normal_stations = non_const_means[
        (non_const_means <= upper_bound) & (non_const_means >= 1.0)
    ].index.tolist()

    # Step 3: 剔除零值占比过高的站点 (>30% 的时间为零, 仅训练期)
    zero_ratios = (train_volume[normal_stations] < 0.01).sum() / len(train_volume)
    low_zero_stations = zero_ratios[zero_ratios < 0.3].index.tolist()

    # Step 4: 按训练期总充电量排序, 选 top-k
    total_volume = train_volume[low_zero_stations].sum().sort_values(ascending=False)
    selected = total_volume.head(k).index.tolist()

    # 打印清洗统计
    n_total = len(station_cols)
    n_const = n_total - len(non_constant)
    n_outlier = len(non_constant) - len(normal_stations)
    n_zero = len(normal_stations) - len(low_zero_stations)
    print(f"    Data cleaning (train-period only, first {n_train}/{len(volume_df)} rows): "
          f"{n_total} total -> "
          f"-{n_const} constant, -{n_outlier} outlier, -{n_zero} high-zero -> "
          f"{len(low_zero_stations)} valid -> selected top {len(selected)}")

    return selected


def stratified_sample_stations(
    volume_df: pd.DataFrame,
    city_data: Dict,
    time_col: str,
    k: int,
    train_ratio: float = 0.85,
    distribution: str = "natural",  # "natural" | "balanced"
    seed: int = 42,
) -> Tuple[List[str], Dict]:
    """
    分层抽样选站: 按负荷水平、零值率、站点类型分层, 每层内随机选取。

    与 select_top_stations 的区别:
      - Top-K: 只保留总充电量最大的 k 个站点 (偏向大型高活跃站点)
      - Stratified: 每层都保留代表性站点 (适合跨城市对比)

    Args:
        distribution: "natural" 按层比例抽样; "balanced" 每层等量抽样

    Returns:
        (selected_stations, strata_info)
    """
    import random
    random.seed(seed)

    station_cols = [c for c in volume_df.columns if c != time_col]
    n_train = max(int(len(volume_df) * train_ratio), 1)
    train_volume = volume_df.iloc[:n_train]

    # Step 1: 先用三步过滤清洗 (与 select_top_stations 一致)
    means = train_volume[station_cols].mean()
    stds = train_volume[station_cols].std()
    non_constant = stds[stds >= 0.01].index.tolist()

    non_const_means = means[non_constant]
    q1, q3 = non_const_means.quantile(0.25), non_const_means.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr
    normal_stations = non_const_means[
        (non_const_means <= upper_bound) & (non_const_means >= 1.0)
    ].index.tolist()

    zero_ratios = (train_volume[normal_stations] < 0.01).sum() / n_train
    valid_stations = zero_ratios[zero_ratios < 0.3].index.tolist()

    if len(valid_stations) == 0:
        print(f"    Stratified: no valid stations after cleaning")
        return [], {"error": "no_valid_stations"}

    if k >= len(valid_stations):
        print(f"    Stratified: k={k} >= valid={len(valid_stations)}, returning all")
        return valid_stations, {"all_valid": True, "n_valid": len(valid_stations)}

    # Step 2: 为每个有效站点标注分层标签
    valid_means = means[valid_stations]
    valid_zeros = zero_ratios[valid_stations]

    # 负荷层级 (低/中/高)
    p33 = valid_means.quantile(0.33)
    p66 = valid_means.quantile(0.66)
    load_levels = []
    for v in valid_means.values:
        if v < p33:
            load_levels.append("L_low")
        elif v < p66:
            load_levels.append("M_mid")
        else:
            load_levels.append("H_high")

    # 零值率层级
    zero_p33 = valid_zeros.quantile(0.33)
    zero_p66 = valid_zeros.quantile(0.66)
    zero_levels = []
    for v in valid_zeros.values:
        if v < zero_p33:
            zero_levels.append("Z_low")
        elif v < zero_p66:
            zero_levels.append("Z_mid")
        else:
            zero_levels.append("Z_high")

    # 站点类型 (从 sites 表获取)
    sites = city_data.get("sites")
    type_col = None
    if sites is not None:
        for col in ["type", "station_type", "site_type", "category"]:
            if col in sites.columns:
                type_col = col
                break

    station_types = {}
    if type_col:
        id_col = "site_id" if "site_id" in sites.columns else "site"
        for _, row in sites.iterrows():
            sid = str(row[id_col])
            if sid in valid_stations:
                station_types[sid] = str(row.get(type_col, "unknown"))
    # 未分类的统一为 "unknown"
    for sid in valid_stations:
        if sid not in station_types:
            station_types[sid] = "unknown"

    # Step 3: 分层
    strata = defaultdict(list)
    for i, sid in enumerate(valid_stations):
        key = f"{load_levels[i]}|{zero_levels[i]}"
        strata[key].append(sid)

    # Step 4: 按层抽样
    selected = []
    strata_info = {}
    for key, members in strata.items():
        n_total_stratum = len(members)
        if distribution == "balanced":
            n_per_stratum = max(1, k // len(strata))
        else:
            n_per_stratum = max(1, int(k * n_total_stratum / len(valid_stations)))

        n_sample = min(n_per_stratum, n_total_stratum)
        sampled = random.sample(members, n_sample)
        selected.extend(sampled)
        strata_info[key] = {"total": n_total_stratum, "sampled": n_sample}

    # 如果采样过多, 随机裁剪到 k
    if len(selected) > k:
        selected = random.sample(selected, k)

    print(f"    Stratified ({distribution}): {len(selected)} stations from "
          f"{len(strata)} strata (valid={len(valid_stations)}, k={k})")
    for key in sorted(strata_info.keys()):
        print(f"      {key}: {strata_info[key]['sampled']}/{strata_info[key]['total']}")

    return selected, strata_info


def _parse_timestamps(series: pd.Series, col_label: str) -> pd.Series:
    """
    统一解析时间戳列，兼容多种格式:
      - "2023/4/1 0:00" (weather, volume)
      - "2023-04-01 00:00:00" (e_price, s_price)
    """
    try:
        return pd.to_datetime(series)
    except Exception:
        pass
    # 尝试常见格式
    for fmt in ["%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                "%Y/%m/%d %H:%M:%S"]:
        try:
            return pd.to_datetime(series, format=fmt)
        except Exception:
            continue
    raise ValueError(f"无法解析 {col_label} 列的时间戳: {series.head(3).tolist()}")


def build_station_dataframe(
    city_data: Dict,
    station_id: str,
    time_col: str = "Unnamed: 0",
    timezone_offset: int = 0,
    price_normalization: bool = False,
    add_load_norm: bool = False,
    train_ratio: float = 0.7,
    use_lag_features: bool = True,
    use_rolling_features: bool = True,
    use_static_features: bool = True,
) -> pd.DataFrame:
    """
    为单个站点构建包含所有特征的 DataFrame:
      - target: 充电负荷 (volume)
      - 气象特征: temp, humidity, windspeed, ...   (按时间戳合并)
      - 价格特征: e_price, s_price                 (按时间戳合并)
      - 时间特征: 循环编码 (数据已是本地时间, 不再做时区偏移)
      - 负荷归一化: target_per_charger
      - 电价标准化: e_price_zscore, e_price_rel_daily, e_price_quantile

    Args:
        timezone_offset: 已废弃 (CHARGED 时间戳已是各城市本地时间), 保留仅为向后兼容
        price_normalization: 是否添加城市内标准化电价特征
        add_load_norm: 是否添加每桩特征
        train_ratio: 训练期占比; 电价标准化/分位数统计量仅用此前缀拟合 (防测试集泄漏)
        use_lag_features: 是否添加滞后特征
        use_rolling_features: 是否添加滚动统计特征
        use_static_features: 是否添加站点静态特征
    """
    volume = city_data["volume"]
    weather = city_data["weather"]
    e_price = city_data["e_price"]
    s_price = city_data["s_price"]

    # ── 1. 构建基础 DataFrame: 时间戳 + 目标 ──
    timestamps = _parse_timestamps(volume[time_col], "volume")
    n_original = len(timestamps)

    # 时间戳已是各城市本地时间 (CHARGED 官方约定), 不再做时区偏移;
    # timezone_offset 参数保留仅为向后兼容, 现已忽略。
    local_ts = timestamps

    df = pd.DataFrame()
    df["timestamp"] = timestamps
    df["target"] = volume[station_id].values.astype(np.float32)

    # ── 2. 气象特征: 按时间戳合并 ──
    weather_time_col = "time" if "time" in weather.columns else weather.columns[0]
    weather_ts = _parse_timestamps(weather[weather_time_col], "weather")
    weather_features = ["temp", "humidity", "windspeed", "precip",
                        "cloudcover", "solarradiation", "pressure"]
    available_wf = [f for f in weather_features if f in weather.columns]
    if available_wf:
        wdf = weather[[weather_time_col] + available_wf].copy()
        wdf["__ts__"] = weather_ts
        wdf.drop(columns=[weather_time_col], inplace=True)

        df["__ts__"] = timestamps
        df = pd.merge(df, wdf, left_on="__ts__", right_on="__ts__", how="left")
        df.drop(columns=["__ts__"], inplace=True)
        for f in available_wf:
            df[f] = df[f].astype(np.float32)
        if len(df) != n_original:
            print(f"    WARNING: weather merge changed row count "
                  f"{n_original} -> {len(df)} for station {station_id}")

    # ── 3. 电价特征: 按时间戳合并 ──
    price_time_col = "time" if "time" in e_price.columns else e_price.columns[0]
    price_ts = _parse_timestamps(e_price[price_time_col], "e_price")

    if station_id in e_price.columns:
        pdf = e_price[[price_time_col, station_id]].copy()
        pdf.columns = [price_time_col, "e_price"]
        pdf["__ts__"] = price_ts
        pdf.drop(columns=[price_time_col], inplace=True)

        df["__ts__"] = timestamps
        df = pd.merge(df, pdf, left_on="__ts__", right_on="__ts__", how="left")
        df.drop(columns=["__ts__"], inplace=True)
        df["e_price"] = df["e_price"].astype(np.float32)
        if len(df) != n_original:
            print(f"    WARNING: e_price merge changed row count "
                  f"{n_original} -> {len(df)} for station {station_id}")

    if station_id in s_price.columns:
        pdf = s_price[[price_time_col, station_id]].copy()
        pdf.columns = [price_time_col, "s_price"]
        pdf["__ts__"] = price_ts
        pdf.drop(columns=[price_time_col], inplace=True)

        df["__ts__"] = timestamps
        df = pd.merge(df, pdf, left_on="__ts__", right_on="__ts__", how="left")
        df.drop(columns=["__ts__"], inplace=True)
        df["s_price"] = df["s_price"].astype(np.float32)
        if len(df) != n_original:
            print(f"    WARNING: s_price merge changed row count "
                  f"{n_original} -> {len(df)} for station {station_id}")

    # ── 3.5 电价标准化 (城市内, 消除币种差异; 统计量仅用 train 前缀拟合, 防泄漏) ──
    if price_normalization:
        n_train = max(int(len(df) * train_ratio), 1)
        train_slice = df.iloc[:n_train]

        if "e_price" in df.columns:
            e_train = train_slice["e_price"].values
            e_vals = df["e_price"].values
            e_mean = np.nanmean(e_train)
            e_std = np.nanstd(e_train) + 1e-8
            df["e_price_zscore"] = ((e_vals - e_mean) / e_std).astype(np.float32)
            # 相对近期平均电价 (因果滚动 24h 均值, 避免同日未来信息泄漏)
            df["e_price_rel_daily"] = (
                df["e_price"]
                / (df["e_price"].rolling(24, min_periods=1).mean() + 1e-8)
            ).astype(np.float32)
            # 电价分位数: 用 train 分布的经验 CDF 映射到 [0,1]
            e_sorted = np.sort(e_train[~np.isnan(e_train)])
            e_quantile = np.full_like(e_vals, np.nan, dtype=np.float64)
            valid = ~np.isnan(e_vals)
            if e_sorted.size and valid.any():
                e_quantile[valid] = (
                    np.searchsorted(e_sorted, e_vals[valid], side="right")
                    / e_sorted.size
                )
            df["e_price_quantile"] = e_quantile.astype(np.float32)
        if "s_price" in df.columns:
            s_train = train_slice["s_price"].values
            s_vals = df["s_price"].values
            s_mean = np.nanmean(s_train)
            s_std = np.nanstd(s_train) + 1e-8
            df["s_price_zscore"] = ((s_vals - s_mean) / s_std).astype(np.float32)

    # ── 4. 时间特征 (循环编码, 使用时区感知的本地时间) ──
    df["hour"] = local_ts.dt.hour
    df["dayofweek"] = local_ts.dt.dayofweek
    df["month"] = local_ts.dt.month
    df["is_weekend"] = (local_ts.dt.dayofweek >= 5).astype(np.float32)

    # 正弦/余弦循环编码
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)

    # 删除原始离散时间列 (已编码)
    df.drop(columns=["hour", "dayofweek", "month"], inplace=True)

    # ── 5. 滞后特征 (充电负荷历史值) ──
    if use_lag_features:
        lag_hours = [24, 48, 168]  # 1天, 2天, 7天
        for lag in lag_hours:
            df[f"target_lag_{lag}h"] = df["target"].shift(lag).astype(np.float32)

    # ── 6. 滚动统计特征 ──
    if use_rolling_features:
        for window in [24, 168]:
            roll = df["target"].rolling(window=window, min_periods=1)
            df[f"target_roll_mean_{window}h"] = roll.mean().astype(np.float32)
            df[f"target_roll_std_{window}h"] = roll.std().astype(np.float32)
            df[f"target_roll_max_{window}h"] = roll.max().astype(np.float32)

    # ── 7. 静态特征 (站点属性, 广播到所有时间步) ──
    # 注: 原含 avg_power, 但它是 sites.csv 的全时段充电统计量 (测试集泄漏), 已移除。
    static = get_station_static_features(city_data, station_id)
    if use_static_features:
        for key in ["charger_num", "perimeter"]:
            val = static.get(key, 0)
            df[key] = np.float32(val)

    # ── 7.5 负荷归一化: 区分 "站点规模大" vs "行为模式不同" ──
    # 注: 原 load_rate = target / (avg_power * charger_num) 依赖全时段 avg_power, 已删除。
    if add_load_norm:
        charger_num = float(static.get("charger_num", 0))
        if charger_num > 0:
            df["target_per_charger"] = (df["target"] / charger_num).astype(np.float32)
        else:
            df["target_per_charger"] = df["target"].copy()

    # ── 8. 缺失值标记 + 前向填充 ──
    for col in df.columns:
        if col != "timestamp" and df[col].isna().any():
            df[f"{col}_is_missing"] = df[col].isna().astype(np.float32)
    df.ffill(inplace=True)  # 前向填充 (比 fillna(0) 更合理)
    df.fillna(0, inplace=True)  # 开头无可前向填充的仍填0

    return df


def get_station_static_features(city_data: Dict, station_id: str) -> Dict:
    """获取站点静态特征 (充电桩数量、平均功率等), 用于聚类"""
    sites = city_data["sites"]
    # sites 的 id 列名不统一, 兼容处理
    id_col = "site_id" if "site_id" in sites.columns else "site"
    row = sites[sites[id_col].astype(str) == str(station_id)]
    if len(row) == 0:
        return {"charger_num": 0, "avg_power": 0, "perimeter": 0}
    row = row.iloc[0]
    return {
        "charger_num": float(row.get("charger_num", 0)),
        "avg_power": float(row.get("avg_power", 0)),
        "perimeter": float(row.get("perimeter", 0)),
        "total_volume": float(row.get("total_volume", 0)),
    }
