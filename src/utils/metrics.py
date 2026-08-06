"""
评估指标 + 训练工具函数

指标:
  - RMSE, MAE (标准)
  - MAPE_active (排除 |y|≤1 的样本, 原名 MAPE)
  - MAPE_raw (标准 MAPE)
  - WAPE (加权绝对百分比误差: sum|y-pred| / sum|y|)
  - SMAPE (对称 MAPE)
  - NRMSE (归一化 RMSE: RMSE / y_mean)
  - MASE (相对 Seasonal Naive 的缩放误差)
  - 分时段误差: RMSE@1h, 6h, 12h, 24h
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    seasonal_naive_mae: float = None,
                    pred_len: int = 24) -> Dict[str, float]:
    """
    计算完整指标体系

    Args:
        y_true: 真实值 (N, pred_len) 或 (N*pred_len,)
        y_pred: 预测值
        seasonal_naive_mae: Seasonal Naive 预测的 MAE (用于 MASE)
        pred_len: 预测窗口长度
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # MAPE_active: 排除 |y|≤1 的样本 (旧版 "MAPE", 保留兼容)
    mask_active = np.abs(y_true) > 1.0
    if mask_active.sum() > 0:
        mape_active = float(np.mean(np.abs(
            (y_true[mask_active] - y_pred[mask_active]) / y_true[mask_active])) * 100)
    else:
        mape_active = 0.0

    # MAPE_raw: 标准 MAPE, 排除 y=0
    mask_nz = np.abs(y_true) > 1e-8
    if mask_nz.sum() > 0:
        mape_raw = float(np.mean(np.abs(
            (y_true[mask_nz] - y_pred[mask_nz]) / y_true[mask_nz])) * 100)
    else:
        mape_raw = 0.0

    # WAPE: sum(|y-pred|) / sum(|y|) * 100
    y_sum = np.sum(np.abs(y_true))
    if y_sum > 0:
        wape = float(np.sum(np.abs(y_true - y_pred)) / y_sum * 100)
    else:
        wape = 0.0

    # SMAPE: mean(2*|y-pred| / (|y|+|pred|+ε)) * 100
    smape = float(np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100)

    # NRMSE: RMSE / mean(y)
    y_mean = np.mean(y_true)
    if abs(y_mean) > 1e-8:
        nrmse = float(rmse / abs(y_mean))
    else:
        nrmse = float("inf")

    # MASE: MAE / Seasonal Naive MAE
    if seasonal_naive_mae is not None and seasonal_naive_mae > 0:
        mase = float(mae / seasonal_naive_mae)
    else:
        mase = float("inf")

    result = {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape_active,          # 向后兼容: 保留 MAPE 为 MAPE_active
        "MAPE_active": mape_active,
        "MAPE_raw": mape_raw,
        "WAPE": wape,
        "SMAPE": smape,
        "NRMSE": nrmse,
    }
    if mase != float("inf"):
        result["MASE"] = mase

    # 分时段误差 (如果 pred_len > 1)
    if pred_len > 1:
        y_true_2d = y_true.reshape(-1, pred_len)
        y_pred_2d = y_pred.reshape(-1, pred_len)
        horizons = [1, 6, 12, 24]
        for h in horizons:
            if h <= pred_len:
                idx = min(h - 1, pred_len - 1)
                err_h = np.sqrt(np.mean((y_true_2d[:, idx] - y_pred_2d[:, idx]) ** 2))
                result[f"RMSE@{h}h"] = float(err_h)
            else:
                result[f"RMSE@{h}h"] = float(rmse)

    return result


@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader: DataLoader,
                   scaler, device: str = "cpu",
                   return_predictions: bool = False,
                   seasonal_naive_mae: float = None,
                   pred_len: int = 24) -> Dict[str, float]:
    """
    在验证/测试集上评估模型, 返回反归一化后的指标

    Args:
        return_predictions: 若 True, 返回 (metrics, preds, targets)
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

    if len(all_preds) == 0:
        empty = {"RMSE": float("inf"), "MAE": float("inf"),
                 "MAPE": float("inf"), "WAPE": float("inf"),
                 "SMAPE": float("inf")}
        if return_predictions:
            return empty, np.array([]), np.array([])
        return empty

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 反归一化
    all_preds_inv = scaler.inverse_target(all_preds)
    all_targets_inv = scaler.inverse_target(all_targets)

    metrics = compute_metrics(all_targets_inv, all_preds_inv,
                              seasonal_naive_mae=seasonal_naive_mae,
                              pred_len=pred_len)
    model.to("cpu")

    if return_predictions:
        return metrics, all_preds_inv, all_targets_inv
    return metrics


def compute_seasonal_naive_mae(target: np.ndarray, seasonality: int = 168,
                               pred_len: int = 24) -> float:
    """
    计算 Seasonal Naive 预测的 MAE (用于 MASE 计算)

    Seasonal Naive: 用 seasonality 小时前的值作为预测
    """
    y_true = target.flatten()
    # 构造预测: 每个预测位置的对应历史值
    total_len = len(y_true)

    all_errors = []
    for i in range(seasonality, total_len):
        pred = y_true[i - seasonality]
        all_errors.append(abs(y_true[i] - pred))

    if all_errors:
        return float(np.mean(all_errors))
    return float("inf")


def set_seed(seed: int):
    """设置全局随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
