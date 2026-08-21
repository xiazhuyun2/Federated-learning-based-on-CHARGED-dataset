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

from config import Config, DATA_DIR, get_run_dir
from src.data.data_loader import load_city_data, select_top_stations, build_station_dataframe
from src.data.feature_engineering import prepare_station_data
from src.models.tcn_lstm import build_model
from src.utils.metrics import evaluate_model, set_seed, compute_metrics

TIMEZONE_OFFSETS = {"SZH": 8, "AMS": 2, "JHB": 2, "LOA": -7, "MEL": 10, "SPO": -3}


def train_local_only(city: str = "SZH", top_k: int = 20,
                     epochs: int = 100, lr: float = 1e-3,
                     seed: int = 42, output_dir: str = None):
    """
    Baseline 1: 每个站点只用自己的数据训练 (Local-only, 孤岛模式)
    """
    cfg = Config()
    cfg.data.top_k_stations = top_k
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)

    run_dir = get_run_dir(city, "local_only", seed,
                          base_dir=output_dir or cfg.output_dir)

    city_data = load_city_data(DATA_DIR, city, cfg.data.use_remove_zero)
    stations = select_top_stations(
        city_data["volume"], cfg.data.time_col, top_k,
        train_ratio=cfg.data.train_ratio + cfg.data.val_ratio
    )

    results = {}
    predictions = {}

    tz = TIMEZONE_OFFSETS.get(city, 0)

    for sid in stations:
        df = build_station_dataframe(city_data, sid, cfg.data.time_col,
                                     timezone_offset=tz,
                                     price_normalization=True,
                                     add_load_norm=True)
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
        metrics, preds, targets = evaluate_model(
            model, test_loader, scaler, "cpu",
            return_predictions=True, pred_len=cfg.data.pred_len)
        results[f"{city}_{sid}"] = metrics
        predictions[f"{city}_{sid}"] = {"pred": preds, "target": targets}
        print(f"  Local-only {city}_{sid}: "
              f"RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, "
              f"WAPE={metrics['WAPE']:.2f}%")

    # 汇总
    avg = {}
    for key in ["RMSE", "MAE", "MAPE", "WAPE", "SMAPE", "NRMSE"]:
        vals = [m[key] for m in results.values() if key in m]
        if vals:
            avg[key] = np.mean(vals)
    results["AVERAGE"] = avg
    print(f"\n  Local-only AVERAGE: RMSE={avg.get('RMSE', 0):.4f}, "
          f"MAE={avg.get('MAE', 0):.4f}, WAPE={avg.get('WAPE', 0):.2f}%")

    # 保存
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to {run_dir}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="SZH")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  Baseline: Local-only Training (No Federation)")
    print("=" * 60)
    train_local_only(args.city, args.top_k, args.epochs, seed=args.seed)
