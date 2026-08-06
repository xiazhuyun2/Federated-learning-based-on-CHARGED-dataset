"""
Baseline: Centralized Training (集中式训练)

所有站点数据合并训练单个 TCN-LSTM 模型。
这是联邦学习的理论上限 — 没有隐私/通信约束时的最佳效果。
"""
import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, DATA_DIR, get_run_dir
from src.data.data_loader import load_city_data, select_top_stations, build_station_dataframe
from src.data.feature_engineering import prepare_station_data, TimeSeriesScaler
from src.models.tcn_lstm import build_model
from src.utils.metrics import evaluate_model, set_seed


def train_centralized(city: str = "SZH", top_k: int = 20,
                      epochs: int = 100, lr: float = 1e-3,
                      seed: int = 42, output_dir: str = None):
    """
    Centralized baseline: 合并所有站点训练数据, 训练单个模型
    """
    cfg = Config()
    cfg.data.top_k_stations = top_k
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = get_run_dir(city, "centralized", seed,
                          base_dir=output_dir or cfg.output_dir)

    city_data = load_city_data(DATA_DIR, city, cfg.data.use_remove_zero)
    stations = select_top_stations(
        city_data["volume"], cfg.data.time_col, top_k,
        train_ratio=cfg.data.train_ratio + cfg.data.val_ratio
    )

    # 收集所有站点的训练/测试数据
    all_train_datasets = []
    per_station_test = {}  # 每个站点仍需单独评估
    per_station_scaler = {}

    for sid in stations:
        df = build_station_dataframe(city_data, sid, cfg.data.time_col)
        train_ds, val_ds, test_ds, scaler = prepare_station_data(
            df, cfg.data.seq_len, cfg.data.pred_len)

        if len(train_ds) == 0:
            continue

        all_train_datasets.append(train_ds)
        per_station_test[sid] = test_ds
        per_station_scaler[sid] = scaler

    if not all_train_datasets:
        raise ValueError("No training data available!")

    # 合并所有训练集
    merged_train = ConcatDataset(all_train_datasets)
    train_loader = DataLoader(merged_train, batch_size=64, shuffle=True)

    input_dim = merged_train[0][0].shape[1]
    model = build_model(input_dim, cfg.data.pred_len, cfg.model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print(f"\n  Centralized Training: {len(stations)} stations, "
          f"{len(merged_train)} windows")
    print(f"  Epochs: {epochs}, LR: {lr}")

    # 训练
    history = {"epoch": [], "loss": []}
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f}")

    # 测试 (每个站点独立评估)
    model.to("cpu")
    results = {}
    predictions = {}

    for sid, test_ds in per_station_test.items():
        if len(test_ds) == 0:
            continue
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        scaler = per_station_scaler[sid]
        metrics = evaluate_model(model, test_loader, scaler, "cpu",
                                 pred_len=cfg.data.pred_len)
        results[f"{city}_{sid}"] = metrics
        print(f"  {city}_{sid}: RMSE={metrics['RMSE']:.4f}, "
              f"MAE={metrics['MAE']:.4f}, WAPE={metrics['WAPE']:.2f}%")

    # 汇总
    avg = {}
    for key in ["RMSE", "MAE", "MAPE", "WAPE", "SMAPE", "NRMSE"]:
        vals = [m[key] for m in results.values() if key in m]
        if vals:
            avg[key] = np.mean(vals)
    results["AVERAGE"] = avg

    print(f"\n  AVERAGE: RMSE={avg.get('RMSE', 0):.4f}, "
          f"MAE={avg.get('MAE', 0):.4f}, WAPE={avg.get('WAPE', 0):.2f}%")

    # 保存
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    # 保存模型
    torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))

    print(f"\n  Results saved to {run_dir}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="SZH")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  Baseline: Centralized Training")
    print("=" * 60)
    train_centralized(args.city, args.top_k, args.epochs, args.lr, args.seed)
