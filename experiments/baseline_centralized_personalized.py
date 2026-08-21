"""
Baseline: Centralized Training — Personalized (集中式个性化)

共享 TCN-LSTM 主干 + 每站独立预测头, 每站用训练期统计量分别标准化。
这是与多城市 FL 相同主干/损失/时间切分下的「集中式性能参考上限」。

与 baseline_centralized.py (Centralized-shared) 的区别:
  - shared   : 单共享模型 + 单一全局 target scaler (大站主导, 小站信号被压缩)
  - personalized: 共享主干 + 每站独立头 + 每站独立 scaler (问题与解决3.txt 第六节推荐)
"""
import sys
import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, DATA_DIR, get_run_dir
from src.data.data_loader import (
    load_city_data, select_top_stations, stratified_sample_stations,
    build_station_dataframe,
)
from src.data.feature_engineering import prepare_station_data
from src.models.tcn_lstm import build_model
from src.utils.metrics import compute_metrics, set_seed

TIMEZONE_OFFSETS = {"SZH": 8, "AMS": 2, "JHB": 2, "LOA": -7, "MEL": 10, "SPO": -3}


class StationTaggedDataset(Dataset):
    """为数据集样本附加站点整数索引, 便于混合批次内按站路由到各自预测头。"""

    def __init__(self, base_ds, station_idx: int):
        self.base = base_ds
        self.idx = station_idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, self.idx


def _allocate_budget(city_sizes, total_budget, min_city=2):
    """按 (有效站点数 - 保底) 超出部分分配预算 (最大余数法), 与 trainer 一致。"""
    cities = list(city_sizes.keys())
    excess = {c: max(0.0, city_sizes[c] - min_city) for c in cities}
    excess_total = sum(excess.values())
    raw = {c: (total_budget - min_city * len(cities)) * (
        excess[c] / excess_total) if excess_total > 0 else 0.0 for c in cities}

    alloc = {c: min_city + int(raw[c]) for c in cities}
    frac = {c: raw[c] - int(raw[c]) for c in cities}
    order = sorted(cities, key=lambda c: (-frac[c], c))
    i = 0
    while sum(alloc.values()) < total_budget:
        alloc[order[i % len(order)]] += 1
        i += 1
    return alloc


def _forward_backbone_head(shared, head, x):
    tcn_out = shared.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)
    lstm_out, _ = shared.lstm(tcn_out)
    last = lstm_out[:, -1, :]
    return head(last)


def train_centralized_personalized(cities, top_k=20, epochs=100, lr=1e-3,
                                   seed=42, station_selection="top_k",
                                   output_dir=None):
    cfg = Config()
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = get_run_dir("+".join(cities), "centralized_personalized", seed,
                          base_dir=output_dir or cfg.output_dir)
    train_ratio = cfg.data.train_ratio + cfg.data.val_ratio

    # ── 1. 逐城选站 + 建数据集 + 每站独立 scaler ──────────────────
    records = []  # dict: city, sid, station_idx, train_ds, val_ds, test_ds, scaler
    city_sizes = {}

    # 比例分配预算 (proportional 模式需要跨城规模)
    budget = None
    if station_selection == "proportional":
        for city in cities:
            cd = load_city_data(DATA_DIR, city, cfg.data.use_remove_zero)
            all_valid = select_top_stations(cd["volume"], cfg.data.time_col,
                                            k=10 ** 6, train_ratio=train_ratio)
            city_sizes[city] = len(all_valid)
        budget = _allocate_budget(city_sizes, top_k, cfg.data.min_city_clients)

    for city in cities:
        cd = load_city_data(DATA_DIR, city, cfg.data.use_remove_zero)
        tz = TIMEZONE_OFFSETS.get(city, 0)

        if station_selection.startswith("stratified"):
            dist = "natural" if "natural" in station_selection else "balanced"
            stations, _ = stratified_sample_stations(
                cd["volume"], cd, cfg.data.time_col,
                budget[city] if budget else top_k,
                train_ratio=train_ratio, distribution=dist, seed=seed)
        else:
            stations = select_top_stations(
                cd["volume"], cfg.data.time_col,
                budget[city] if budget else top_k, train_ratio=train_ratio)

        for sid in stations:
            df = build_station_dataframe(cd, sid, cfg.data.time_col,
                                         timezone_offset=tz,
                                         price_normalization=True,
                                         add_load_norm=True)
            train_ds, val_ds, test_ds, scaler = prepare_station_data(
                df, cfg.data.seq_len, cfg.data.pred_len,
                train_ratio=cfg.data.train_ratio, val_ratio=cfg.data.val_ratio)
            if len(train_ds) == 0:
                continue
            records.append({"city": city, "sid": sid, "train_ds": train_ds,
                            "val_ds": val_ds, "test_ds": test_ds, "scaler": scaler})

    if not records:
        raise ValueError("No training data available!")

    # ── 2. 共享主干 + 每站独立预测头 ────────────────────────────────
    input_dim = records[0]["train_ds"][0][0].shape[1]
    shared = build_model(input_dim, cfg.data.pred_len, cfg.model)
    heads = nn.ModuleList([copy.deepcopy(shared.fc) for _ in records])
    shared.fc = nn.Identity()  # 移除无用的共享头, 只保留 tcn+lstm 主干

    # ── 3. 合并训练集 + 每站等权采样 ───────────────────────────────
    merged = ConcatDataset([
        StationTaggedDataset(rec["train_ds"], i) for i, rec in enumerate(records)
    ])
    n_stations = len(records)
    total_samples = sum(len(rec["train_ds"]) for rec in records)
    sample_weights = []
    for rec in records:
        n_s = len(rec["train_ds"])
        w = total_samples / (n_stations * n_s)
        sample_weights.extend([w] * n_s)
    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)
    train_loader = DataLoader(merged, batch_size=cfg.fed.batch_size, sampler=sampler)

    optimizer = torch.optim.Adam(
        list(shared.parameters()) + list(heads.parameters()),
        lr=lr, weight_decay=cfg.fed.weight_decay)
    criterion = nn.MSELoss()

    shared.to(device)
    heads.to(device)

    print(f"\n  Centralized-Personalized: {n_stations} stations across "
          f"{len(cities)} cities, {len(merged)} windows")
    print(f"  Epochs: {epochs}, LR: {lr}, per-station head + per-station scaler")

    # ── 4. 训练 ──────────────────────────────────────────────────
    history = {"epoch": [], "loss": []}
    for epoch in range(epochs):
        shared.train()
        heads.train()
        total_loss = 0.0
        total_batches = 0
        for x, y, sidx in train_loader:
            x, y, sidx = x.to(device), y.to(device), sidx.to(device)
            preds = torch.zeros_like(y)
            for s in sidx.unique():
                m = (sidx == s)
                preds[m] = _forward_backbone_head(shared, heads[s.item()], x[m])
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(shared.parameters()) + list(heads.parameters()), 5.0)
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
        avg_loss = total_loss / max(total_batches, 1)
        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f}")

    # ── 5. 测试 (每站用自己 head + 自己 scaler 反归一化) ────────────
    shared.eval()
    heads.eval()
    results = {}
    micro_abs_err = micro_abs_target = micro_sq_err = 0.0
    micro_n = 0

    for i, rec in enumerate(records):
        loader = DataLoader(rec["test_ds"], batch_size=cfg.fed.batch_size,
                            shuffle=False)
        preds_list, targets_list = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                p = _forward_backbone_head(shared, heads[i], x).cpu().numpy()
                preds_list.append(p)
                targets_list.append(y.numpy())
        preds = rec["scaler"].inverse_target(np.concatenate(preds_list, axis=0))
        targets = rec["scaler"].inverse_target(np.concatenate(targets_list, axis=0))
        m = compute_metrics(targets, preds, pred_len=cfg.data.pred_len)
        results[f"{rec['city']}_{rec['sid']}"] = m
        micro_abs_err += float(np.sum(np.abs(preds - targets)))
        micro_abs_target += float(np.sum(np.abs(targets)))
        micro_sq_err += float(np.sum((preds - targets) ** 2))
        micro_n += int(targets.size)

    # 汇总: 宏平均 (每站等权)
    avg = {}
    for key in ["RMSE", "MAE", "WAPE", "SMAPE", "NRMSE", "MAPE_raw"]:
        vals = [m[key] for m in results.values() if key in m]
        if vals:
            avg[key] = float(np.mean(vals))
    results["AVERAGE"] = avg

    results["micro"] = {
        "WAPE": float(micro_abs_err / micro_abs_target * 100) if micro_abs_target > 0 else 0.0,
        "RMSE": float(np.sqrt(micro_sq_err / micro_n)) if micro_n > 0 else 0.0,
        "MAE": float(micro_abs_err / micro_n) if micro_n > 0 else 0.0,
    }

    # 多城市: per-city 宏平均 + 最差城市
    if len(cities) > 1:
        per_city = {}
        for city in cities:
            cids = [k for k in results if k.startswith(city + "_")]
            if not cids:
                continue
            per_city[city] = {
                key: float(np.mean([results[k][key] for k in cids if key in results[k]]))
                for key in ["RMSE", "MAE", "WAPE", "SMAPE", "NRMSE", "MAPE_raw"]
            }
        results["macro_city"] = {
            "RMSE": float(np.mean([v["RMSE"] for v in per_city.values()])),
            "MAE": float(np.mean([v["MAE"] for v in per_city.values()])),
            "WAPE": float(np.mean([v["WAPE"] for v in per_city.values()])),
        }
        results["per_city"] = per_city
        worst = max(per_city.items(), key=lambda kv: kv[1]["WAPE"])
        results["worst_city"] = {"city": worst[0], "WAPE": worst[1]["WAPE"],
                                 "RMSE": worst[1]["RMSE"], "MAE": worst[1]["MAE"]}

    print(f"\n  AVERAGE: RMSE={avg.get('RMSE', 0):.4f}, "
          f"MAE={avg.get('MAE', 0):.4f}, WAPE={avg.get('WAPE', 0):.2f}%")
    if "macro_city" in results:
        print(f"  MACRO-CITY: WAPE={results['macro_city']['WAPE']:.2f}%  "
              f"WORST={results['worst_city']['city']} "
              f"WAPE={results['worst_city']['WAPE']:.2f}%")

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    torch.save({"shared": shared.state_dict(),
                "heads": {i: h.state_dict() for i, h in enumerate(heads)}},
               os.path.join(run_dir, "best_model.pt"))

    print(f"\n  Results saved to {run_dir}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", default="SZH",
                        help="逗号分隔城市列表, 如 SZH,AMS,JHB,LOA,MEL,SPO")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--station_selection", default="top_k",
                        choices=["top_k", "stratified_balanced",
                                 "stratified_natural", "proportional"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    print("=" * 60)
    print(f"  Baseline: Centralized-Personalized ({len(cities)} cities)")
    print("=" * 60)
    train_centralized_personalized(
        cities, args.top_k, args.epochs, args.lr, args.seed,
        args.station_selection)
