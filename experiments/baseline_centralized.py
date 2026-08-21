"""
Baseline: Centralized Training — Shared (集中式共享)

所有站点数据合并训练单个 TCN-LSTM 模型, 使用单一全局 target scaler。
注意: 这不是「理论上限」, 只是集中式共享模型 (问题与解决3.txt 第六节);
真正的性能参考上限见 baseline_centralized_personalized.py (共享主干+每站头)。
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

TIMEZONE_OFFSETS = {"SZH": 8, "AMS": 2, "JHB": 2, "LOA": -7, "MEL": 10, "SPO": -3}


def train_centralized(city: str = "SZH", top_k: int = 20,
                      epochs: int = 100, lr: float = 1e-3,
                      seed: int = 42, output_dir: str = None):
    """
    Centralized baseline: 合并所有站点训练数据, 训练单个模型

    与 local_only 的关键区别:
      - 使用全局 target 归一化 (所有站点共享一个 scaler)
      - 按站点等权采样 (防止大站 dominate loss)
    """
    from sklearn.preprocessing import StandardScaler

    cfg = Config()
    cfg.data.top_k_stations = top_k
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = get_run_dir(city, "centralized_shared", seed,
                          base_dir=output_dir or cfg.output_dir)

    city_data = load_city_data(DATA_DIR, city, cfg.data.use_remove_zero)
    stations = select_top_stations(
        city_data["volume"], cfg.data.time_col, top_k,
        train_ratio=cfg.data.train_ratio + cfg.data.val_ratio
    )

    tz = TIMEZONE_OFFSETS.get(city, 0)

    # ═══════════════════════════════════════════════════════════
    # Phase 1: 在所有站点训练数据上 fit 全局 target scaler
    # ═══════════════════════════════════════════════════════════
    all_train_targets_raw = []
    for sid in stations:
        df = build_station_dataframe(city_data, sid, cfg.data.time_col,
                                     timezone_offset=tz,
                                     price_normalization=True,
                                     add_load_norm=True)
        target = df["target"].values
        n_train_end = int(len(target) * (cfg.data.train_ratio + cfg.data.val_ratio))
        all_train_targets_raw.append(target[:n_train_end])

    global_target_scaler = StandardScaler()
    global_target_scaler.fit(
        np.concatenate(all_train_targets_raw).reshape(-1, 1))
    print(f"  Global target scaler: "
          f"mean={global_target_scaler.mean_[0]:.2f}, "
          f"std={global_target_scaler.scale_[0]:.2f}")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: 用全局 scaler 创建各站点数据集
    # ═══════════════════════════════════════════════════════════
    all_train_datasets = []
    station_sample_counts = {}  # 用于加权
    per_station_test = {}
    per_station_scaler = {}

    for sid in stations:
        df = build_station_dataframe(city_data, sid, cfg.data.time_col,
                                     timezone_offset=tz,
                                     price_normalization=True,
                                     add_load_norm=True)
        train_ds, val_ds, test_ds, scaler = prepare_station_data(
            df, cfg.data.seq_len, cfg.data.pred_len,
            external_target_scaler=global_target_scaler)

        if len(train_ds) == 0:
            continue

        all_train_datasets.append(train_ds)
        station_sample_counts[sid] = len(train_ds)
        per_station_test[sid] = test_ds
        per_station_scaler[sid] = scaler

    if not all_train_datasets:
        raise ValueError("No training data available!")

    # 合并所有训练集
    merged_train = ConcatDataset(all_train_datasets)

    # 计算每站等权的样本权重
    n_stations = len(all_train_datasets)
    total_samples = sum(station_sample_counts.values())
    sample_weights = []
    for i, ds in enumerate(all_train_datasets):
        n_s = len(ds)
        # 每站贡献相等的总权重 → w = total / (n_stations * n_s)
        w = total_samples / (n_stations * n_s)
        sample_weights.extend([w] * n_s)

    # WeightedRandomSampler: 按站等权采样
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(merged_train, batch_size=64, sampler=sampler)

    input_dim = merged_train[0][0].shape[1]
    model = build_model(input_dim, cfg.data.pred_len, cfg.model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print(f"\n  Centralized Training: {len(stations)} stations, "
          f"{len(merged_train)} windows")
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"  Sample weights: range [{min(sample_weights):.2f}, "
          f"{max(sample_weights):.2f}] (per-station equal weight)")

    # ═══════════════════════════════════════════════════════════
    # Phase 3: 训练
    # ═══════════════════════════════════════════════════════════
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

    # ═══════════════════════════════════════════════════════════
    # Phase 4: 测试 (每个站点用全局 scaler 独立评估)
    # ═══════════════════════════════════════════════════════════
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
