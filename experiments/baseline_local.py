"""
对比实验 — 单站点本地训练 (孤岛模式) vs 联邦学习
用于论文的 baseline 对比
"""
import sys
import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, DATA_DIR
from src.data.data_loader import load_city_data, select_top_stations, build_station_dataframe
from src.data.feature_engineering import prepare_station_data
from src.models.tcn_lstm import build_model
from src.utils.metrics import evaluate_model, set_seed, compute_metrics


def train_local_only(city: str = "SZH", top_k: int = 20,
                     epochs: int = 100, lr: float = 1e-3):
    """
    Baseline 1: 每个站点只用自己的数据训练 (Local-only, 孤岛模式)
    """
    cfg = Config()
    cfg.data.top_k_stations = top_k
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)

    city_data = load_city_data(DATA_DIR, city, cfg.data.use_remove_zero)
    stations = select_top_stations(
        city_data["volume"], cfg.data.time_col, top_k)

    results = {}

    for sid in stations:
        df = build_station_dataframe(city_data, sid, cfg.data.time_col)
        train_ds, val_ds, test_ds, scaler = prepare_station_data(
            df, cfg.data.seq_len, cfg.data.pred_len)

        if len(train_ds) == 0:
            continue

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        input_dim = train_ds[0][0].shape[1]
        model = build_model(input_dim, cfg.data.pred_len, cfg.model)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        # 训练
        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        # 测试
        model.to("cpu")
        metrics = evaluate_model(model, test_loader, scaler, "cpu")
        results[f"{city}_{sid}"] = metrics
        print(f"  Local-only {city}_{sid}: "
              f"RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, "
              f"MAPE={metrics['MAPE']:.2f}%")

    avg = {
        "RMSE": np.mean([m["RMSE"] for m in results.values()]),
        "MAE": np.mean([m["MAE"] for m in results.values()]),
        "MAPE": np.mean([m["MAPE"] for m in results.values()]),
    }
    results["AVERAGE"] = avg
    print(f"\n  Local-only AVERAGE: RMSE={avg['RMSE']:.4f}, "
          f"MAE={avg['MAE']:.4f}, MAPE={avg['MAPE']:.2f}%")

    # 保存
    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/baseline_local_{city}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="SZH")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("  Baseline: Local-only Training (No Federation)")
    print("=" * 60)
    train_local_only(args.city, args.top_k, args.epochs)
