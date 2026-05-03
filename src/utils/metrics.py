"""
评估指标 + 训练工具函数
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算 RMSE, MAE, MAPE
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    # MAPE: 仅在 y_true 有显著值的位置计算, 避免除以极小值导致爆炸
    mask = np.abs(y_true) > 1.0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader: DataLoader,
                   scaler, device: str = "cpu") -> Dict[str, float]:
    """
    在验证/测试集上评估模型, 返回反归一化后的指标
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
        return {"RMSE": float("inf"), "MAE": float("inf"), "MAPE": float("inf")}

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 反归一化
    all_preds_inv = scaler.inverse_target(all_preds)
    all_targets_inv = scaler.inverse_target(all_targets)

    metrics = compute_metrics(all_targets_inv, all_preds_inv)
    model.to("cpu")
    return metrics


def set_seed(seed: int):
    """设置全局随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
