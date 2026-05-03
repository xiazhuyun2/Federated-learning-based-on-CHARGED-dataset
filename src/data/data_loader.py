"""
数据加载器 — 读取 UrbanEV 数据集, 构建站点级时序数据
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def load_city_data(data_dir: str, city: str, use_remove_zero: bool = True) -> Dict:
    """
    加载单个城市的所有数据文件
    Returns: dict with keys: volume, weather, e_price, s_price, poi, sites, chargers, info
    """
    folder = f"{city}_remove_zero" if use_remove_zero else city
    city_dir = os.path.join(data_dir, folder)

    data = {}
    data["volume"] = pd.read_csv(os.path.join(city_dir, "volume.csv"))
    data["weather"] = pd.read_csv(os.path.join(city_dir, "weather.csv"))
    data["e_price"] = pd.read_csv(os.path.join(city_dir, "e_price.csv"))
    data["s_price"] = pd.read_csv(os.path.join(city_dir, "s_price.csv"))

    # 静态数据从原始文件夹读取
    orig_dir = os.path.join(data_dir, city)
    data["poi"] = pd.read_csv(os.path.join(orig_dir, "poi.csv"))
    data["sites"] = pd.read_csv(os.path.join(orig_dir, "sites.csv"))
    data["chargers"] = pd.read_csv(os.path.join(orig_dir, "chargers.csv"))
    data["info"] = pd.read_csv(os.path.join(orig_dir, "info.csv"))

    return data


def select_top_stations(volume_df: pd.DataFrame, time_col: str, k: int) -> List[str]:
    """
    数据清洗 + 选站: 三步过滤后按总充电量选取 top-k 站点

    过滤规则:
      1. 剔除恒定值站点 (std < 0.01): 6个月充电量完全不变, 无法学习模式
      2. 剔除异常大值站点 (IQR法): mean > Q3 + 3*IQR, 量级远超正常站点
      3. 剔除极稀疏站点 (mean < 1): 几乎无充电活动
    """
    station_cols = [c for c in volume_df.columns if c != time_col]

    # 计算每个站点的统计量
    means = volume_df[station_cols].mean()
    stds = volume_df[station_cols].std()

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

    # Step 3: 剔除零值占比过高的站点 (>30% 的时间为零)
    zero_ratios = (volume_df[normal_stations] < 0.01).sum() / len(volume_df)
    low_zero_stations = zero_ratios[zero_ratios < 0.3].index.tolist()

    # Step 4: 按总充电量排序, 选 top-k
    total_volume = volume_df[low_zero_stations].sum().sort_values(ascending=False)
    selected = total_volume.head(k).index.tolist()

    # 打印清洗统计
    n_total = len(station_cols)
    n_const = n_total - len(non_constant)
    n_outlier = len(non_constant) - len(normal_stations)
    n_zero = len(normal_stations) - len(low_zero_stations)
    print(f"    Data cleaning: {n_total} total -> "
          f"-{n_const} constant, -{n_outlier} outlier, -{n_zero} high-zero -> "
          f"{len(low_zero_stations)} valid -> selected top {len(selected)}")

    return selected


def build_station_dataframe(
    city_data: Dict,
    station_id: str,
    time_col: str = "Unnamed: 0"
) -> pd.DataFrame:
    """
    为单个站点构建包含所有特征的 DataFrame:
      - target: 充电负荷 (volume)
      - 气象特征: temp, humidity, windspeed, ...
      - 价格特征: e_price, s_price
      - 时间特征: hour, dayofweek, month, is_weekend
    """
    volume = city_data["volume"]
    weather = city_data["weather"]
    e_price = city_data["e_price"]
    s_price = city_data["s_price"]

    # 时间索引
    timestamps = pd.to_datetime(volume[time_col])

    df = pd.DataFrame()
    df["timestamp"] = timestamps
    df["target"] = volume[station_id].values.astype(np.float32)

    # 气象特征 (选取关键列)
    weather_features = ["temp", "humidity", "windspeed", "precip",
                        "cloudcover", "solarradiation", "pressure"]
    for feat in weather_features:
        if feat in weather.columns:
            df[feat] = weather[feat].values.astype(np.float32)

    # 电价特征
    if station_id in e_price.columns:
        df["e_price"] = e_price[station_id].values.astype(np.float32)
    if station_id in s_price.columns:
        df["s_price"] = s_price[station_id].values.astype(np.float32)

    # 时间特征 (循环编码)
    df["hour"] = timestamps.dt.hour
    df["dayofweek"] = timestamps.dt.dayofweek
    df["month"] = timestamps.dt.month
    df["is_weekend"] = (timestamps.dt.dayofweek >= 5).astype(np.float32)

    # 正弦/余弦循环编码
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)

    # 删除原始离散时间列 (已编码)
    df.drop(columns=["hour", "dayofweek", "month"], inplace=True)

    # 填充缺失值
    df.fillna(0, inplace=True)

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
