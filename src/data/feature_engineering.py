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
        self.total_len = len(target) - seq_len - pred_len + 1

    def __len__(self):
        return max(0, self.total_len)

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
    train_ratio: float = 0.7, val_ratio: float = 0.15
) -> Tuple[ChargingDataset, ChargingDataset, ChargingDataset, TimeSeriesScaler]:
    """
    将站点 DataFrame 切分为 train/val/test 数据集
    """
    target = df["target"].values
    feature_cols = [c for c in df.columns if c not in ("timestamp", "target")]
    features = df[feature_cols].values

    # 时序切分 (不能 shuffle)
    n = len(target)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))

    scaler = TimeSeriesScaler()

    # fit on train only
    train_target, train_features = scaler.fit_transform(
        target[:n_train], features[:n_train])
    val_target, val_features = scaler.transform(
        target[n_train:n_val], features[n_train:n_val])
    test_target, test_features = scaler.transform(
        target[n_val:], features[n_val:])

    train_ds = ChargingDataset(train_target, train_features, seq_len, pred_len)
    val_ds = ChargingDataset(val_target, val_features, seq_len, pred_len)
    test_ds = ChargingDataset(test_target, test_features, seq_len, pred_len)

    return train_ds, val_ds, test_ds, scaler
