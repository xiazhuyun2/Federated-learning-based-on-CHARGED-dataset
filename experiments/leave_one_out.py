"""
留一城市冷启动测试 — 评估多城市FL对新城市的泛化能力

实验流程 (对每个被留出的城市):
  1. 用其余5个城市训练一个「可迁移」的全局模型 (普通 FedAvg/FedProx, 不开 FedBN/LocalHead)
  2. Zero-shot: 直接用全局模型预测留出城市 (不微调)
  3. Few-shot: 用留出城市 N 天 (7/14/30) 数据微调后评估
  4. From-scratch: 用同样 N 天数据从头训练 (与 few-shot 对照, 量化预训练价值)
  5. Full local: 用留出城市全量数据本地训练 (oracle 参考)

循环6次 (每次留不同城市), 报告平均冷启动性能。

关于「N 天」的语义: 模型输入窗口 seq_len=168h (7天) + 预测 pred_len=24h, 即每个训练样本
需要 8 天 (192h) 连续数据。因此 7 天数据不足以构成任何训练窗口 (窗口数=0), 会打印
「insufficient」并跳过; 14 天→145 窗口, 30 天→529 窗口。

用法:
  python experiments/leave_one_out.py --top_k 10 --rounds 30
  python experiments/leave_one_out.py --quick            # 快速验证
  python experiments/leave_one_out.py --finetune_days 14,30
"""
import sys
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, DATA_DIR
from src.data.data_loader import (
    load_city_data, select_top_stations, build_station_dataframe
)
from src.data.feature_engineering import prepare_station_data
from src.federated.trainer import FederatedTrainer
from src.models.tcn_lstm import build_model
from src.utils.metrics import evaluate_model, set_seed

ALL_CITIES = ["SZH", "AMS", "JHB", "LOA", "MEL", "SPO"]
TIMEZONE_OFFSETS = {"SZH": 8, "AMS": 2, "JHB": 2, "LOA": -7, "MEL": 10, "SPO": -3}


def _infer_input_dim(state_dict: dict) -> int:
    """从 checkpoint 的 TCN 第一层卷积权重推断真实 input_dim

    之前错误地用 fc.0.weight.shape[1] (LSTM hidden dim=64),
    正确做法是用 tcn.network.0.conv1.conv.weight.shape[1] (真实输入特征数).
    """
    tcn_keys = [
        "tcn.network.0.conv1.conv.weight",  # TCNBlock -> CausalConv1d -> nn.Conv1d
        "tcn.network.0.conv1.weight",       # 旧版可能直接用 nn.Conv1d
    ]
    for key in tcn_keys:
        w = state_dict.get(key)
        if w is not None and w.ndim >= 2:
            return w.shape[1]
    return None


def _infer_pred_len(state_dict: dict) -> int:
    """从 checkpoint 预测头末层 Linear 权重推断 pred_len (shape[0])."""
    for key in ("fc.3.weight", "fc.weight"):
        w = state_dict.get(key)
        if w is not None and w.ndim == 2:
            return w.shape[0]
    return None


def _model_cfg():
    """与 config.ModelConfig 默认值一致 (build_model 只读架构字段, 不读 fedbn/localhead)."""
    return type("M", (), {
        "tcn_channels": [64, 64, 64], "tcn_kernel_size": 3,
        "tcn_dropout": 0.2, "lstm_hidden": 64, "lstm_layers": 2,
        "lstm_dropout": 0.2, "fc_hidden": 64,
        "use_fedbn": False, "use_local_head": False,
    })()


def _prepare_station_tasks(test_city: str, top_k: int) -> list:
    """构建留出城市每个站点的数据集任务 (只做一次, 供各评估函数复用).

    返回 [{sid, train_ds, val_ds, test_ds, scaler, input_dim, seq_len, pred_len}]
    """
    test_data = load_city_data(DATA_DIR, test_city, use_remove_zero=True)
    stations = select_top_stations(
        test_data["volume"], "Unnamed: 0", top_k, train_ratio=0.85)
    tz = TIMEZONE_OFFSETS.get(test_city, 0)

    tasks = []
    for sid in stations:
        try:
            df = build_station_dataframe(
                test_data, sid,
                timezone_offset=tz,
                price_normalization=True,
                add_load_norm=True,
            )
            train_ds, val_ds, test_ds, scaler = prepare_station_data(df)
            if len(train_ds) == 0 or len(test_ds) == 0:
                continue
            x0, y0 = train_ds[0]
            tasks.append({
                "sid": sid,
                "train_ds": train_ds,
                "val_ds": val_ds,
                "test_ds": test_ds,
                "scaler": scaler,
                "input_dim": x0.shape[1],
                "seq_len": x0.shape[0],
                "pred_len": int(y0.numel()),
            })
        except Exception as e:
            print(f"    Station {sid} failed: {e}")
            continue
    return tasks


def _n_windows_from_days(days: int, seq_len: int, pred_len: int, n_train: int) -> int:
    """N 天目标城市数据能构成的 (seq_len 输入 + pred_len 输出) 训练窗口数."""
    return min(n_train, max(0, days * 24 - seq_len - pred_len + 1))


def _adaptive_epochs(n_windows: int, base_epochs: int, base_windows: int = 145) -> int:
    """按窗口数自适应 few-shot 微调 epoch, 保持总梯度步数 ≈ base_epochs * base_windows。

    few-shot 用固定 epoch 时, 数据越多 (30d=529 窗口 vs 14d=145 窗口) 总步数线性膨胀,
    小样本更容易过拟合 (如 SZH 30d 用固定 3 epoch 退化到 113.4%)。自适应后
    14d≈3 epoch、30d≈1 epoch, 总步数恒定, 避免「更多数据反而更差」。
    """
    total_steps = base_epochs * base_windows
    return max(1, int(round(total_steps / max(1, n_windows))))


def _aggregate_metrics(all_metrics: list) -> dict:
    if not all_metrics:
        return {"RMSE": float("inf"), "MAE": float("inf"), "WAPE": float("inf"),
                "n_stations": 0}
    return {
        "RMSE": float(np.mean([m["RMSE"] for m in all_metrics])),
        "MAE": float(np.mean([m["MAE"] for m in all_metrics])),
        "WAPE": float(np.mean([m.get("WAPE", 0) for m in all_metrics])),
        "n_stations": len(all_metrics),
    }


def _freeze_bn_layers(model) -> int:
    """把模型里的 BN 层置为 eval 模式 (冻结 running_mean/var), 返回冻结层数.

    few-shot 微调背景: 全局模型已在 5 城数据上学到稳定的 BN running stats,
    但小样本 (14/30 天, 仅 ~145/529 窗口) 下 model.train() 会用噪声 batch 统计量
    归一化并更新 running stats, 把好的全局统计量污染掉, 导致微调后性能崩塌
    (如 AMS zero-shot 45.6 -> few-shot 100.5)。置 eval 后 BN 改用固定 running stats
    归一化且不再更新, 只保留 affine 参数 (weight/bias) 可微调。
    """
    cnt = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            cnt += 1
    return cnt


def _train_on_windows(model, train_ds, n_windows: int, epochs: int, lr: float,
                      device: str, batch_size: int = 16, verbose: bool = False,
                      freeze_bn: bool = False, head_only: bool = False) -> None:
    """在 train_ds 最近 n_windows 个窗口上就地训练 model (改 model 状态并置 eval).

    冷启动语义: 用「最近的 N 天」(train_ds 尾部, 紧邻测试期) 而非「最早的 N 天」。
    EV 负荷有季节/趋势漂移, 最早的数据与测试期相隔数月, 会让 few-shot 学到
    不具代表性的模式 (如 AMS zero-shot 22.9 -> few-shot 55.6)。
    """
    start = max(0, len(train_ds) - n_windows)
    subset = Subset(train_ds, list(range(start, len(train_ds))))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()

    model.train()
    if freeze_bn:
        n_bn = _freeze_bn_layers(model)
        if verbose:
            print(f"      [freeze_bn] 冻结 {n_bn} 个 BN 层 running stats")
    if head_only:
        n_frozen = 0
        for name, p in model.named_parameters():
            if not name.startswith("fc"):
                p.requires_grad_(False)
                n_frozen += 1
        if verbose:
            print(f"      [head_only] 冻结 {n_frozen} 个 backbone 参数, 仅微调 fc 预测头")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr)
    for epoch in range(epochs):
        epoch_loss, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        if verbose and (epoch + 1) % max(1, epochs // 2) == 0:
            print(f"      epoch {epoch+1}/{epochs}, loss={epoch_loss/max(1, n):.4f}")
    model.eval()


def train_global_model(train_cities: list, top_k: int, rounds: int,
                         local_epochs: int, seed: int, output_dir: str) -> str:
    """在给定城市集上训练全局FL模型, 返回 best_model 路径.

    冷启动需要一个「可迁移」的单一全局模型, 因此这里**不开 FedBN / LocalHead**
    (否则 BN running stats 与预测头只存在于各站点本地、不会进入全局模型,
    zero-shot 加载到的将是一个随机 BN/头的坏模型)。
    """
    print(f"\n  Training global model on {train_cities}...")

    cfg = Config()
    cfg.data.top_k_stations = top_k
    cfg.fed.num_rounds = rounds
    cfg.fed.local_epochs = local_epochs
    cfg.fed.aggregation = "fedprox"
    cfg.fed.city_weight_alpha = 0.5
    cfg.fed.multi_city_mode = "multi_city"
    cfg.data.station_selection = "top_k"
    cfg.model.use_fedbn = False
    cfg.model.use_local_head = False
    cfg.seed = seed

    trainer = FederatedTrainer(cfg, run_dir=output_dir, city="MULTI",
                               method="leave_one_out")
    trainer.prepare_multi_city_clients(train_cities)
    trainer.run_federated_training()

    model_path = os.path.join(output_dir, "best_model.pt")
    return model_path


def evaluate_zero_shot(model_path: str, tasks: list, device: str) -> dict:
    """Zero-shot: 直接用全局模型在留出城市上评估 (不微调)."""
    print(f"\n  Zero-shot evaluation ({len(tasks)} stations)...")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    input_dim = _infer_input_dim(state_dict) or (tasks[0]["input_dim"] if tasks else 60)
    pred_len = _infer_pred_len(state_dict) or (tasks[0]["pred_len"] if tasks else 24)

    model = build_model(input_dim, pred_len, _model_cfg())
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    all_metrics = []
    for t in tasks:
        if t["input_dim"] != input_dim:
            print(f"    Station {t['sid']}: input_dim mismatch "
                  f"({t['input_dim']} vs {input_dim}), skipped")
            continue
        loader = DataLoader(t["test_ds"], batch_size=64, shuffle=False)
        all_metrics.append(evaluate_model(model, loader, t["scaler"], device))

    return _aggregate_metrics(all_metrics)


def evaluate_few_shot(model_path: str, tasks: list, device: str, seed: int,
                      finetune_days: int, finetune_epochs: int,
                      finetune_lr: float = 1e-4, freeze_bn: bool = False,
                      head_only: bool = False, adaptive_epochs: bool = True) -> dict:
    """Few-shot: 用留出城市前 N 天数据微调全局模型后评估."""
    print(f"\n  Few-shot ({finetune_days}d) evaluation...")
    set_seed(seed)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    input_dim = _infer_input_dim(state_dict) or (tasks[0]["input_dim"] if tasks else 60)
    pred_len = _infer_pred_len(state_dict) or (tasks[0]["pred_len"] if tasks else 24)

    all_metrics = []
    for t in tasks:
        n_win = _n_windows_from_days(
            finetune_days, t["seq_len"], t["pred_len"], len(t["train_ds"]))
        if n_win <= 0:
            print(f"    Station {t['sid']}: {finetune_days}d < "
                  f"{t['seq_len'] + t['pred_len']}h window, insufficient, skipped")
            continue
        if t["input_dim"] != input_dim:
            print(f"    Station {t['sid']}: input_dim mismatch, skipped")
            continue

        epochs = (_adaptive_epochs(n_win, finetune_epochs)
                  if adaptive_epochs else finetune_epochs)
        model = build_model(input_dim, pred_len, _model_cfg())
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        _train_on_windows(model, t["train_ds"], n_win, epochs,
                          finetune_lr, device, freeze_bn=freeze_bn,
                          head_only=head_only)
        loader = DataLoader(t["test_ds"], batch_size=64, shuffle=False)
        m = evaluate_model(model, loader, t["scaler"], device)
        all_metrics.append(m)
        print(f"    Station {t['sid']}: WAPE={m.get('WAPE', float('nan')):.1f}% "
              f"({n_win} windows, {epochs} epoch)")

    return _aggregate_metrics(all_metrics)


def evaluate_from_scratch(tasks: list, device: str, seed: int,
                          finetune_days: int, scratch_epochs: int,
                          scratch_lr: float = 1e-3) -> dict:
    """From-scratch: 用与 few-shot 完全相同的前 N 天数据从头训练 (量化预训练价值)."""
    print(f"\n  From-scratch ({finetune_days}d) evaluation...")
    set_seed(seed)

    all_metrics = []
    for t in tasks:
        n_win = _n_windows_from_days(
            finetune_days, t["seq_len"], t["pred_len"], len(t["train_ds"]))
        if n_win <= 0:
            print(f"    Station {t['sid']}: {finetune_days}d < "
                  f"{t['seq_len'] + t['pred_len']}h window, insufficient, skipped")
            continue

        model = build_model(t["input_dim"], t["pred_len"], _model_cfg())
        model.to(device)
        _train_on_windows(model, t["train_ds"], n_win, scratch_epochs,
                          scratch_lr, device)
        loader = DataLoader(t["test_ds"], batch_size=64, shuffle=False)
        m = evaluate_model(model, loader, t["scaler"], device)
        all_metrics.append(m)
        print(f"    Station {t['sid']}: WAPE={m.get('WAPE', float('nan')):.1f}% "
              f"({n_win} windows)")

    return _aggregate_metrics(all_metrics)


def evaluate_full_local(tasks: list, epochs: int, device: str, seed: int) -> dict:
    """Full local: 留出城市全量本地训练 (oracle 参考)."""
    print(f"\n  Full local training ({len(tasks)} stations)...")
    set_seed(seed)

    all_metrics = []
    for t in tasks:
        train_loader = DataLoader(t["train_ds"], batch_size=64, shuffle=True)
        test_loader = DataLoader(t["test_ds"], batch_size=64, shuffle=False)

        model = build_model(t["input_dim"], t["pred_len"], _model_cfg())
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"    Epoch {epoch+1}/{epochs}, "
                      f"loss={epoch_loss/len(train_loader):.4f}")

        model.eval()
        all_metrics.append(evaluate_model(model, test_loader, t["scaler"], device))

    return _aggregate_metrics(all_metrics)


def _mean_std(vals):
    """跨 seed 求 mean/std, 忽略 None 与 inf. 返回 {mean, std, n}."""
    vals = [float(v) for v in vals if v is not None and v < float("inf")]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals)}


def _city_wape_across_seeds(seed_results, city, *path):
    """取某城市某指标在所有 seed 下的 WAPE 列表。

    path 形如 ("zero_shot",) 或 ("few_shot", "14") / ("from_scratch", "14")。
    """
    vals = []
    for seed in seed_results:
        node = seed_results[seed].get(city, {})
        for k in path:
            node = node.get(k, {}) if isinstance(node, dict) else {}
        v = node.get("WAPE") if isinstance(node, dict) else None
        if v is not None and v < float("inf"):
            vals.append(float(v))
    return vals


def _fmt_mean_std(ms):
    if ms.get("n", 0) == 0:
        return " " * 14
    if ms["n"] == 1:
        return f"{ms['mean']:>14.1f}"
    return f"{ms['mean']:>8.1f}±{ms['std']:<5.1f}"


def run_once(args, cities, finetune_days, seed, device, base_dir, timestamp):
    """跑单个 seed 的完整留一城市流程, 返回 {left_out: {zero_shot, few_shot,
    from_scratch, full_local}}."""
    all_results = {}
    for left_out in cities:
        print(f"\n{'='*70}")
        print(f"  LEAVE-ONE-OUT [seed={seed}]: Test on {left_out}")
        print(f"  Train on: {[c for c in cities if c != left_out]}")
        print(f"{'='*70}")

        train_cities = [c for c in cities if c != left_out]

        # 每个 seed 独立训练全局模型 (目录按 seed 区分, 避免互相覆盖)
        model_dir = os.path.join(base_dir, f"train_{left_out}_s{seed}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "best_model.pt")
        if args.skip_train and os.path.exists(model_path):
            print(f"\n  [skip_train] 复用已保存的全局模型 {model_path}")
        else:
            model_path = train_global_model(
                train_cities, args.top_k, args.rounds,
                args.local_epochs, seed, model_dir)

        tasks = _prepare_station_tasks(left_out, args.top_k)

        zero_shot = evaluate_zero_shot(model_path, tasks, device)

        few_shot = {}
        for day in finetune_days:
            few_shot[str(day)] = evaluate_few_shot(
                model_path, tasks, device, seed,
                finetune_days=day, finetune_epochs=args.finetune_epochs,
                finetune_lr=args.finetune_lr, freeze_bn=args.freeze_bn,
                head_only=args.head_only, adaptive_epochs=args.adaptive_epochs)

        from_scratch = {}
        for day in finetune_days:
            from_scratch[str(day)] = evaluate_from_scratch(
                tasks, device, seed,
                finetune_days=day, scratch_epochs=args.scratch_epochs,
                scratch_lr=args.scratch_lr)

        if args.skip_full_local:
            full_local = {"RMSE": float("inf"), "WAPE": float("inf")}
        else:
            full_local = evaluate_full_local(tasks, args.epochs, device, seed)

        all_results[left_out] = {
            "zero_shot": zero_shot,
            "few_shot": few_shot,
            "from_scratch": from_scratch,
            "full_local": full_local,
        }

        print(f"\n  Results for {left_out}:")
        print(f"    Zero-shot:  WAPE={zero_shot.get('WAPE', float('nan')):.1f}%")
        for day in finetune_days:
            fs = few_shot[str(day)].get("WAPE", float("inf"))
            sc = from_scratch[str(day)].get("WAPE", float("inf"))
            print(f"    Few-shot ({day:>2d}d):   WAPE={fs:.1f}%  |  "
                  f"from-scratch ({day:>2d}d): WAPE={sc:.1f}%")
        if not args.skip_full_local:
            print(f"    Full local: WAPE={full_local.get('WAPE', float('nan')):.1f}%")

    return all_results


def _print_summary(seed_results, cities, finetune_days, skip_full_local):
    n_seeds = len(seed_results)
    print(f"\n{'='*100}")
    print(f"  LEAVE-ONE-OUT SUMMARY  (WAPE %, {n_seeds} seed"
          f"{'s' if n_seeds > 1 else ''}, mean±std)")
    print(f"{'='*100}")

    header = f"  {'City':<6s} {'ZS':>14s} "
    for day in finetune_days:
        header += f"{'FS' + str(day) + 'd':>14s} {'SC' + str(day) + 'd':>14s} "
    if not skip_full_local:
        header += f"{'FullLocal':>14s}"
    print(header)
    print("  " + "-" * 96)

    for city in cities:
        zs = _mean_std(_city_wape_across_seeds(seed_results, city, "zero_shot"))
        line = f"  {city:<6s} {_fmt_mean_std(zs)} "
        for day in finetune_days:
            fs = _mean_std(_city_wape_across_seeds(
                seed_results, city, "few_shot", str(day)))
            sc = _mean_std(_city_wape_across_seeds(
                seed_results, city, "from_scratch", str(day)))
            line += f"{_fmt_mean_std(fs)} {_fmt_mean_std(sc)} "
        if not skip_full_local:
            fl = _mean_std(_city_wape_across_seeds(
                seed_results, city, "full_local"))
            line += f"{_fmt_mean_std(fl)}"
        print(line)

    # 平均与预训练增益 (先对每城跨 seed 求 mean, 再跨城平均)
    print("\n  --- 平均 WAPE 与预训练增益 (few-shot 相对 from-scratch 的下降) ---")
    zero_vals = [_mean_std(_city_wape_across_seeds(
        seed_results, c, "zero_shot"))["mean"] for c in cities]
    zero_vals = [v for v in zero_vals if not np.isnan(v)]
    if zero_vals:
        print(f"    Zero-shot 平均 WAPE: {np.mean(zero_vals):.1f}%")
    for day in finetune_days:
        fs_m = [_mean_std(_city_wape_across_seeds(
            seed_results, c, "few_shot", str(day)))["mean"] for c in cities]
        sc_m = [_mean_std(_city_wape_across_seeds(
            seed_results, c, "from_scratch", str(day)))["mean"] for c in cities]
        fs_m = [v for v in fs_m if not np.isnan(v)]
        sc_m = [v for v in sc_m if not np.isnan(v)]
        if fs_m and sc_m:
            fs_avg, sc_avg = np.mean(fs_m), np.mean(sc_m)
            gain = (sc_avg - fs_avg) / sc_avg * 100
            print(f"    {day:>2d} 天: few-shot={fs_avg:.1f}%  "
                  f"from-scratch={sc_avg:.1f}%  预训练增益={gain:+.1f}%")
        else:
            print(f"    {day:>2d} 天: 数据不足 (所有站点窗口数=0), 无法评估")
    if not skip_full_local:
        fl_m = [_mean_std(_city_wape_across_seeds(
            seed_results, c, "full_local"))["mean"] for c in cities]
        fl_m = [v for v in fl_m if not np.isnan(v)]
        if fl_m:
            print(f"    Full local 平均 WAPE: {np.mean(fl_m):.1f}% (oracle)")
    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser(
        description="留一城市冷启动泛化测试")
    parser.add_argument("--cities", type=str,
                        default="SZH,AMS,JHB,LOA,MEL,SPO")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50,
                        help="Full local 训练的 epochs")
    parser.add_argument("--finetune_days", type=str, default="7,14,30",
                        help="few-shot/from-scratch 使用的目标城市数据天数 (逗号分隔)")
    parser.add_argument("--finetune_epochs", type=int, default=3)
    parser.add_argument("--finetune_lr", type=float, default=1e-4,
                        help="few-shot 微调学习率 (低 lr, 避免灾难性遗忘)")
    parser.add_argument("--adaptive_epochs", action="store_true", default=True,
                        help="few-shot 按窗口数自适应 epoch (14d≈3, 30d≈1, 默认开)")
    parser.add_argument("--no_adaptive_epochs", dest="adaptive_epochs",
                        action="store_false",
                        help="关闭自适应, 所有天数用固定 --finetune_epochs")
    parser.add_argument("--freeze_bn", action="store_true",
                        help="few-shot 微调时冻结 BN running stats (防小样本污染全局统计量)")
    parser.add_argument("--head_only", action="store_true",
                        help="few-shot 微调时只训练 fc 预测头, 冻结 TCN+LSTM 主干")
    parser.add_argument("--scratch_epochs", type=int, default=10,
                        help="from-scratch 从头训练的 epochs (需更多轮收敛)")
    parser.add_argument("--scratch_lr", type=float, default=1e-3,
                        help="from-scratch 从头训练学习率")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None,
                        help="逗号分隔多种子 (如 42,123,999), 覆盖 --seed 并聚合 mean±std")
    parser.add_argument("--quick", action="store_true",
                        help="快速验证: rounds=5, top_k=3, epochs=10")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_full_local", action="store_true",
                        help="跳过 full local oracle (耗时)")
    parser.add_argument("--skip_train", action="store_true",
                        help="复用已保存的全局模型 (跳过 train_global_model)")
    args = parser.parse_args()

    if args.quick:
        args.top_k = min(args.top_k, 3)
        args.rounds = min(args.rounds, 5)
        args.epochs = min(args.epochs, 10)
        args.local_epochs = min(args.local_epochs, 2)
        print("  Quick mode enabled")

    finetune_days = [int(d.strip()) for d in args.finetune_days.split(",") if d.strip()]

    cities = [c.strip() for c in args.cities.split(",")]
    base_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "leave_one_out")
    os.makedirs(base_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    seeds = ([int(s.strip()) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else [args.seed])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_results = {}
    for i, seed in enumerate(seeds):
        print(f"\n{'#'*70}")
        print(f"#  SEED {seed} ({i+1}/{len(seeds)})")
        print(f"{'#'*70}")
        res = run_once(args, cities, finetune_days, seed, device,
                       base_dir, timestamp)
        seed_results[seed] = res
        # 每个 seed 独立存一份摘要, 便于追溯
        per_seed_path = os.path.join(base_dir, f"loo_summary_s{seed}_{timestamp}.json")
        with open(per_seed_path, "w", encoding="utf-8") as f:
            json.dump({"timestamp": timestamp, "seed": seed, "cities": cities,
                       "finetune_days": finetune_days, "results": res},
                      f, indent=2, default=str)

    _print_summary(seed_results, cities, finetune_days, args.skip_full_local)

    # 保存聚合结果 (每城每指标 mean/std + 原始 per-seed)
    agg = {"cities": {}}
    for city in cities:
        agg["cities"][city] = {
            "zero_shot": _mean_std(_city_wape_across_seeds(
                seed_results, city, "zero_shot")),
            "few_shot": {str(d): _mean_std(_city_wape_across_seeds(
                seed_results, city, "few_shot", str(d))) for d in finetune_days},
            "from_scratch": {str(d): _mean_std(_city_wape_across_seeds(
                seed_results, city, "from_scratch", str(d))) for d in finetune_days},
            "full_local": _mean_std(_city_wape_across_seeds(
                seed_results, city, "full_local")),
        }
    summary = {
        "timestamp": timestamp,
        "seeds": seeds,
        "cities": cities,
        "finetune_days": finetune_days,
        "aggregated": agg,
        "per_seed": seed_results,
    }
    summary_path = os.path.join(base_dir, f"loo_summary_multi_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
