"""
Baseline: Seasonal Naive 预测

Seasonal Naive: 用上周同时刻 (168小时前) 的值作为预测。
这是时序预测的最基础 baseline，任何模型都应该大幅优于它。
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, DATA_DIR, get_run_dir
from src.data.data_loader import load_city_data, select_top_stations, build_station_dataframe
from src.data.feature_engineering import prepare_station_data
from src.utils.metrics import compute_metrics, set_seed

TIMEZONE_OFFSETS = {"SZH": 8, "AMS": 2, "JHB": 2, "LOA": -7, "MEL": 10, "SPO": -3}


def evaluate_seasonal_naive(city: str = "SZH", top_k: int = 20,
                            seed: int = 42, output_dir: str = None):
    """
    Seasonal Naive 基线评估
    预测 = 168小时 (7天) 前的实际值
    """
    cfg = Config()
    cfg.data.top_k_stations = top_k
    set_seed(seed)

    run_dir = get_run_dir(city, "seasonal_naive", seed,
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

        if len(test_ds) == 0:
            continue

        # 收集测试集真实值
        all_targets = []
        all_preds = []

        for x, y in test_ds:
            target = y.numpy()  # (pred_len,) — 已标准化
            all_targets.append(target)

            # Seasonal naive: 使用输入窗口最后168小时前的值
            # 输入 x 包含 [target, features], 取第一列 (=target)
            # 取 seq_len - 168 位置的值作为预测
            input_target = x[:, 0].numpy()  # (seq_len,)
            seasonality = 168
            pred_len = cfg.data.pred_len

            # 用输入序列中 seasonality 小时前的值
            if len(input_target) >= seasonality:
                # 对于每个预测步, 使用对应的季节滞后
                naive_pred = np.array([
                    input_target[-seasonality + i] if seasonality - i <= len(input_target) else input_target[-seasonality]
                    for i in range(pred_len)
                ])
            else:
                naive_pred = np.full(pred_len, input_target[-1])

            all_preds.append(naive_pred)

        all_preds = np.stack(all_preds)
        all_targets = np.stack(all_targets)

        # 反归一化
        preds_inv = scaler.inverse_target(all_preds)
        targets_inv = scaler.inverse_target(all_targets)

        metrics = compute_metrics(targets_inv, preds_inv, pred_len=cfg.data.pred_len)
        results[f"{city}_{sid}"] = metrics
        predictions[f"{city}_{sid}"] = {"pred": preds_inv, "target": targets_inv}
        print(f"  {city}_{sid}: RMSE={metrics['RMSE']:.4f}, "
              f"MAE={metrics['MAE']:.4f}, WAPE={metrics['WAPE']:.2f}%")

    # 汇总
    avg = {}
    for key in ["RMSE", "MAE", "MAPE", "WAPE", "SMAPE", "NRMSE"]:
        vals = [m[key] for m in results.values() if key in m]
        if vals:
            avg[key] = np.mean(vals)
    results["AVERAGE"] = avg

    # 保存冬季naive的MAE (用于后续MASE计算)
    seasonal_mae = avg.get("MAE", 0)

    print(f"\n  AVERAGE: RMSE={avg.get('RMSE', 0):.4f}, "
          f"MAE={avg.get('MAE', 0):.4f}, WAPE={avg.get('WAPE', 0):.2f}%")

    # 保存结果
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 保存 seasonal_naive_mae 供其他实验使用
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"seasonal_naive_mae": seasonal_mae, "city": city}, f)

    print(f"\n  Results saved to {run_dir}")
    return results, seasonal_mae


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="SZH")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  Baseline: Seasonal Naive")
    print("=" * 60)
    evaluate_seasonal_naive(args.city, args.top_k, args.seed)
