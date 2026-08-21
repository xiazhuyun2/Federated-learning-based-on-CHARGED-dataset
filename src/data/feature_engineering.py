"""
特征工程 — 标准化、滑动窗口构建、VMD信号分解
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


# ============================================================
# VMD (Variational Mode Decomposition) — 纯 NumPy 实现
# 注意: VMD 当前未被训练链路调用, 仅作为可选组件保留。
#       在消融实验确认有效之前, 项目标题不使用 VMD。
# ============================================================

def vmd(signal: np.ndarray, K: int = 6, alpha: int = 2000,
        tau: float = 0, tol: float = 1e-7, max_iter: int = 500) -> np.ndarray:
    """
    变分模态分解 (VMD)
    Args:
        signal: 一维信号 (T,)
        K: 分解模态数
        alpha: 带宽约束惩罚因子
        tau: 噪声容限 (0 = 无噪声)
        tol: 收敛阈值
        max_iter: 最大迭代次数
    Returns:
        u: (K, T) 各模态子信号
    """
    T = len(signal)
    t = np.arange(1, T + 1) / T
    freqs = t - 0.5 - 1 / T

    # 镜像延拓
    f_mirror = np.concatenate([signal[::-1], signal, signal[::-1]])
    T_mirror = len(f_mirror)
    t_mirror = np.arange(1, T_mirror + 1) / T_mirror

    # FFT
    f_hat = np.fft.fft(f_mirror)
    f_hat_plus = f_hat.copy()
    f_hat_plus[:T_mirror // 2] = 0

    freqs_mirror = np.arange(T_mirror) / T_mirror - 0.5

    # 初始化
    u_hat_plus = np.zeros((max_iter, K, T_mirror), dtype=complex)
    omega_plus = np.zeros((max_iter, K))

    # 初始中心频率均匀分布
    for k in range(K):
        omega_plus[0, k] = (0.5 / K) * k

    lambda_hat = np.zeros((max_iter, T_mirror), dtype=complex)

    # 主迭代
    n = 0
    uDiff = tol + 1

    while uDiff > tol and n < max_iter - 1:
        # 逐模态更新
        for k in range(K):
            # 其他模态之和
            sum_uk = np.sum(u_hat_plus[n, :, :], axis=0) - u_hat_plus[n, k, :]
            if k > 0:
                sum_uk += u_hat_plus[n + 1, :k, :].sum(axis=0)

            u_hat_plus[n + 1, k, :] = (
                (f_hat_plus - sum_uk - lambda_hat[n, :] / 2)
                / (1 + alpha * (freqs_mirror - omega_plus[n, k]) ** 2)
            )

            # 更新中心频率
            numerator = np.sum(
                freqs_mirror[T_mirror // 2:T_mirror] *
                np.abs(u_hat_plus[n + 1, k, T_mirror // 2:T_mirror]) ** 2
            )
            denominator = np.sum(
                np.abs(u_hat_plus[n + 1, k, T_mirror // 2:T_mirror]) ** 2
            ) + 1e-12
            omega_plus[n + 1, k] = numerator / denominator

        # 更新拉格朗日乘子
        lambda_hat[n + 1, :] = (
            lambda_hat[n, :]
            + tau * (f_hat_plus - np.sum(u_hat_plus[n + 1, :, :], axis=0))
        )

        # 收敛判断
        uDiff = 0
        for k in range(K):
            uDiff += np.sum(
                np.abs(u_hat_plus[n + 1, k, :] - u_hat_plus[n, k, :]) ** 2
            ) / np.sum(np.abs(u_hat_plus[n, k, :]) ** 2 + 1e-12)
        uDiff = np.abs(uDiff)

        n += 1

    # 重构
    u = np.zeros((K, T))
    for k in range(K):
        u_hat_full = np.zeros(T_mirror, dtype=complex)
        u_hat_full[T_mirror // 2:T_mirror] = u_hat_plus[n, k, T_mirror // 2:T_mirror]
        u_hat_full[1:T_mirror // 2] = np.conj(u_hat_plus[n, k, T_mirror // 2 + 1:])[::-1]
        u_hat_full[0] = np.conj(u_hat_full[-1])
        u_k = np.real(np.fft.ifft(u_hat_full))
        u[k, :] = u_k[T:2 * T]

    return u


# ============================================================
# 标准化 + 滑动窗口 Dataset
# ============================================================

class TimeSeriesScaler:
    """对目标和特征分别标准化, 保留 scaler 用于反归一化"""

    def __init__(self):
        self.target_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()

    def fit_transform(self, target: np.ndarray, features: np.ndarray):
        target_scaled = self.target_scaler.fit_transform(
            target.reshape(-1, 1)).flatten()
        features_scaled = self.feature_scaler.fit_transform(features)
        return target_scaled, features_scaled

    def transform(self, target: np.ndarray, features: np.ndarray):
        target_scaled = self.target_scaler.transform(
            target.reshape(-1, 1)).flatten()
        features_scaled = self.feature_scaler.transform(features)
        return target_scaled, features_scaled

    def inverse_target(self, target_scaled: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(
            target_scaled.reshape(-1, 1)).flatten()


class ChargingDataset(Dataset):
    """滑动窗口时序数据集"""

    def __init__(self, target: np.ndarray, features: np.ndarray,
                 seq_len: int = 168, pred_len: int = 24):
        self.target = target
        self.features = features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = max(0, len(target) - seq_len - pred_len + 1)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # 输入: [seq_len, num_features] 包含 target 作为第一个特征
        x_target = self.target[idx: idx + self.seq_len]
        x_feat = self.features[idx: idx + self.seq_len]
        x = np.column_stack([x_target, x_feat]).astype(np.float32)

        # 输出: [pred_len]
        y = self.target[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        y = y.astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


def prepare_station_data(
    df, seq_len: int = 168, pred_len: int = 24,
    train_ratio: float = 0.7, val_ratio: float = 0.15,
    external_target_scaler=None,
) -> Tuple[ChargingDataset, ChargingDataset, ChargingDataset, TimeSeriesScaler]:
    """
    将站点 DataFrame 切分为 train/val/test 数据集

    流程:
      1. 先在整个序列上构建滑动窗口 (所有可能的 X,y 对)
      2. 按预测区间的起始索引划分 train/val/test
      3. 标准化在训练时段拟合, 避免信息泄漏

    这样 val/test 窗口可以使用训练期的历史数据,
    但预测目标不会包含未来信息。
    """
    target = df["target"].values
    feature_cols = [c for c in df.columns if c not in ("timestamp", "target")]
    features = df[feature_cols].values

    n_full = len(target)
    n_train_window_end = int(n_full * train_ratio)          # 训练期最后一条数据的索引
    n_val_window_end = int(n_full * (train_ratio + val_ratio))  # 验证期最后

    # Step 1: 先在完整序列上构造所有可能的滑动窗口
    # 每个窗口 i 对应预测区间 [i+seq_len, i+seq_len+pred_len)
    num_windows = max(0, n_full - seq_len - pred_len + 1)
    if num_windows <= 0:
        raise ValueError(f"序列太短: {n_full} 行, 需要至少 {seq_len + pred_len} 行")

    # Step 2: 按预测区间起点划分 train/val/test
    # 预测区间: [i+seq_len, i+seq_len+pred_len)
    # 训练窗口: 预测区间终点 <= 训练期最后索引
    # 验证窗口: 预测区间起点 > 训练期最后索引, 预测区间终点 <= 验证期最后索引
    # 测试窗口: 预测区间起点 > 验证期最后索引
    train_indices = []  # 窗口中预测起始索引的列表
    val_indices = []
    test_indices = []

    for i in range(num_windows):
        pred_start = i + seq_len          # 预测区间的起始行索引
        pred_end = i + seq_len + pred_len  # 预测区间的结束行索引

        if pred_end <= n_train_window_end:
            train_indices.append(i)
        elif pred_start >= n_train_window_end and pred_end <= n_val_window_end:
            val_indices.append(i)
        elif pred_start >= n_val_window_end:
            test_indices.append(i)
        # else: 跨界的窗口丢弃

    print(f"    Window split: {len(train_indices)} train / "
          f"{len(val_indices)} val / {len(test_indices)} test windows "
          f"(from {num_windows} total)")

    if len(train_indices) == 0:
        raise ValueError("没有训练窗口! 请检查 train_ratio 或数据长度。")

    # Step 3: 标准化 — 只在训练窗口的输入/目标上拟合
    scaler = TimeSeriesScaler()

    # 收集所有训练窗口中的 target 和 feature 值 (用于拟合 scaler)
    # 使用原始值拟合
    train_target_raw_list = []
    train_feat_raw_list = []
    for i in train_indices:
        train_target_raw_list.append(target[i: i + seq_len])
        train_feat_raw_list.append(features[i: i + seq_len])
    train_target_raw = np.concatenate(train_target_raw_list)
    train_feat_raw = np.concatenate(train_feat_raw_list)

    # 拟合 scaler — 如果提供了外部 target_scaler 则直接使用
    if external_target_scaler is not None:
        scaler.target_scaler = external_target_scaler
    else:
        scaler.target_scaler.fit(train_target_raw.reshape(-1, 1))
    scaler.feature_scaler.fit(train_feat_raw)

    # 标准化全部数据
    target_scaled = scaler.target_scaler.transform(
        target.reshape(-1, 1)).flatten()
    features_scaled = scaler.feature_scaler.transform(features)

    # Step 4: 构建子 Dataset (使用窗口索引)
    train_ds = _SubsetChargingDataset(
        target_scaled, features_scaled, seq_len, pred_len, train_indices)
    val_ds = _SubsetChargingDataset(
        target_scaled, features_scaled, seq_len, pred_len, val_indices) if val_indices else None
    test_ds = _SubsetChargingDataset(
        target_scaled, features_scaled, seq_len, pred_len, test_indices) if test_indices else None

    # 处理空数据集
    if val_ds is None:
        val_ds = train_ds  # fallback (不应该发生)

    return train_ds, val_ds, test_ds, scaler


class _SubsetChargingDataset(Dataset):
    """按指定窗口索引创建数据集子集"""

    def __init__(self, target: np.ndarray, features: np.ndarray,
                 seq_len: int, pred_len: int, indices: list):
        self.target = target
        self.features = features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.indices = indices  # 窗口起始索引列表

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]  # 原始窗口索引
        x_target = self.target[i: i + self.seq_len]
        x_feat = self.features[i: i + self.seq_len]
        x = np.column_stack([x_target, x_feat]).astype(np.float32)
        y = self.target[i + self.seq_len: i + self.seq_len + self.pred_len]
        y = y.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)
