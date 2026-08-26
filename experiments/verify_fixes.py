"""
防回归自动验收 — 针对 问题与解决4.txt 四个 P0 缺陷的三项自检

用法:
  python experiments/verify_fixes.py

三项检查 (无需 GPU / 真实数据, 秒级完成):
  1. α=1 时 CityBalancedServer 的 global 聚合应 == 普通样本加权 FedAvg
     (对应 #4: α 扫描此前用错模型; 这里锁定聚合数学本身正确)。
  2. 保存的最佳模型 == 验证集最优的 global 参数, 而非 clients[0].model
     (对应 #1: 最佳模型保存错误)。
  3. 特征/scaler 泄漏审计: 电价 z-score/分位数/相对日均、静态特征只能读 train 前缀,
     且 avg_power / load_rate 这两个全时段泄漏特征已删除
     (对应 #2a / #2b)。

全部通过时 exit code = 0, 任一失败 exit code = 1。
"""
import sys
import os
import copy
import tempfile
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.federated.aggregation import FLServer, CityBalancedServer


# ────────────────────────────────────────────────────────────
# 检查 1: α=1 的 global 聚合 == 标准样本加权 FedAvg
# ────────────────────────────────────────────────────────────
def check_alpha1_equals_fedavg():
    torch.manual_seed(0)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 3)
            self.b = nn.Linear(3, 1)

        def forward(self, x):
            return self.b(torch.relu(self.a(x)))

    def random_state(model, scale):
        sd = copy.deepcopy(model.state_dict())
        for k in sd:
            sd[k] = torch.randn_like(sd[k]) * scale
        return sd

    base = Tiny()
    # 5 客户端: 城市 A 三个 (小), 城市 B 两个 (大), 权重差异显著
    client_params = [random_state(base, float(i + 1)) for i in range(5)]
    client_weights = [100.0, 200.0, 300.0, 1000.0, 2000.0]
    client_city_map = ["A", "A", "A", "B", "B"]
    city_sizes = {"A": 600.0, "B": 3000.0}

    # CityBalanced α=1
    cb = CityBalancedServer(copy.deepcopy(base), alpha=1.0, aggregation="fedavg")
    cb.set_city_groups(["A", "B"], city_sizes)
    cb.aggregate_with_city_balance(
        [copy.deepcopy(p) for p in client_params],
        list(client_weights), list(client_city_map))
    global_alpha1 = cb.get_global_params()

    # 普通 FedAvg (标准样本加权)
    fs = FLServer(copy.deepcopy(base), aggregation="fedavg")
    fs.aggregate([copy.deepcopy(p) for p in client_params], list(client_weights))
    global_fedavg = fs.get_global_params()

    max_diff = max(
        (global_alpha1[k] - global_fedavg[k]).abs().max().item()
        for k in global_alpha1
    )
    assert max_diff < 1e-5, f"α=1 global 与 FedAvg 不一致 (max_diff={max_diff:.2e})"

    # sanity: α=0.5 应显著不同于 FedAvg (城市平衡确实生效)
    cb2 = CityBalancedServer(copy.deepcopy(base), alpha=0.5, aggregation="fedavg")
    cb2.set_city_groups(["A", "B"], city_sizes)
    cb2.aggregate_with_city_balance(
        [copy.deepcopy(p) for p in client_params],
        list(client_weights), list(client_city_map))
    global_a05 = cb2.get_global_params()
    diff05 = max(
        (global_a05[k] - global_fedavg[k]).abs().max().item()
        for k in global_a05
    )
    assert diff05 > 1e-6, "α=0.5 应与普通 FedAvg 不同 (城市平衡未生效)"

    return max_diff, diff05


# ────────────────────────────────────────────────────────────
# 检查 2: 保存的最佳模型 == global 最优参数, 而非 clients[0].model
# ────────────────────────────────────────────────────────────
def check_best_model_saved_correctly():
    from src.federated.trainer import FederatedTrainer

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))

        def forward(self, x):
            return self.fc(x)

    tmp = tempfile.mkdtemp(prefix="verify_fixes_")
    run_dir = os.path.join(tmp, "A", "B", "seed_0", "run_test")

    cfg = Config()
    cfg.seed = 0
    cfg.model.use_fedbn = False
    cfg.model.use_local_head = False

    trainer = FederatedTrainer(cfg, run_dir=run_dir, city="SZH", method="test")

    # global 最优参数: 全 1.0
    best_model = Tiny()
    for p in best_model.parameters():
        p.data.fill_(1.0)
    trainer.best_model_state = copy.deepcopy(best_model.state_dict())
    trainer.best_val_rmse = 0.5

    # clients[0].model: 全 2.0 (若保存逻辑回退到 clients[0].model, 会存成 2.0)
    client0_model = Tiny()
    for p in client0_model.parameters():
        p.data.fill_(2.0)
    fake_client = SimpleNamespace(model=client0_model, client_id="SZH_1")
    trainer.clients = [fake_client]
    trainer.scalers = {"SZH_1": None}
    trainer.test_loaders = {"SZH_1": None}
    trainer.val_loaders = {}
    trainer.cities = ["SZH"]
    trainer.city_client_map = {}
    trainer.city_data_sizes = {}
    trainer.first_round_hash = None
    trainer._city_weights = None
    trainer.excluded_param_names = []

    with mock.patch("src.federated.trainer.generate_all_plots"), \
         mock.patch("src.federated.trainer.plot_prediction_vs_actual"), \
         mock.patch("src.federated.trainer.plot_error_distribution"):
        trainer._save_results(
            {}, {"AVERAGE": {"RMSE": 1.0, "MAE": 1.0, "WAPE": 1.0}}, None)

    ckpt = torch.load(os.path.join(run_dir, "best_model.pt"),
                      map_location="cpu", weights_only=False)
    saved = ckpt["model_state_dict"]

    for k in trainer.best_model_state:
        assert torch.allclose(saved[k], trainer.best_model_state[k]), \
            f"best_model.pt[{k}] != global 最优参数"
    # 反证: 必须不等于 clients[0].model (否则说明又存回了客户端模型)
    for k in trainer.best_model_state:
        assert not torch.allclose(saved[k], fake_client.model.state_dict()[k]), \
            f"best_model.pt[{k}] 存成了 clients[0].model (回归)"

    return run_dir


# ────────────────────────────────────────────────────────────
# 检查 3: 特征/scaler 泄漏审计
# ────────────────────────────────────────────────────────────
def check_no_leakage():
    import pandas as pd
    from src.data.data_loader import build_station_dataframe

    n = 400
    n_train = int(n * 0.7)
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    tstr = idx.strftime("%Y-%m-%d %H:%M:%S")

    def make_data(test_e_price):
        volume = pd.DataFrame({
            "Unnamed: 0": tstr,
            "S1": np.linspace(1.0, 5.0, n).astype(np.float32),
        })
        weather = pd.DataFrame({
            "time": tstr,
            "temp": np.linspace(0.0, 30.0, n).astype(np.float32),
            "humidity": np.linspace(20.0, 80.0, n).astype(np.float32),
        })
        e_price = pd.DataFrame({
            "time": tstr,
            "S1": np.concatenate([
                np.linspace(5.0, 15.0, n_train),
                np.full(n - n_train, test_e_price),
            ]),
        })
        s_price = pd.DataFrame({
            "time": tstr, "S1": np.linspace(3.0, 9.0, n),
        })
        sites = pd.DataFrame({
            "site_id": ["S1"], "charger_num": [4], "perimeter": [100.0],
            "avg_power": [50.0], "total_volume": [1000.0],
        })
        return {"volume": volume, "weather": weather, "e_price": e_price,
                "s_price": s_price, "sites": sites}

    def build(test_e_price):
        return build_station_dataframe(
            make_data(test_e_price), "S1", "Unnamed: 0",
            price_normalization=True, add_load_norm=True, train_ratio=0.7)

    df_up = build(1000.0)
    df_down = build(-1000.0)

    # 泄漏断言: 训练期内的价格特征必须与测试期取值无关
    price_cols = ["e_price_zscore", "e_price_rel_daily",
                  "e_price_quantile", "s_price_zscore"]
    for c in price_cols:
        train_up = df_up[c].iloc[:n_train].to_numpy(dtype=np.float64)
        train_down = df_down[c].iloc[:n_train].to_numpy(dtype=np.float64)
        assert np.allclose(train_up, train_down, equal_nan=True), \
            f"训练期 {c} 依赖测试期数据 (泄漏)"

    # 全时段泄漏特征必须已删除
    assert "avg_power" not in df_up.columns, "avg_power (全时段统计) 仍存在"
    assert "load_rate" not in df_up.columns, "load_rate (全时段统计) 仍存在"
    assert "target_per_charger" in df_up.columns, "target_per_charger 缺失"

    # 静态特征只保留 charger_num / perimeter
    assert "charger_num" in df_up.columns and "perimeter" in df_up.columns

    return n_train


def main():
    checks = [
        ("α=1 global == FedAvg", check_alpha1_equals_fedavg),
        ("最佳模型保存 == global 最优", check_best_model_saved_correctly),
        ("特征/scaler 无泄漏", check_no_leakage),
    ]
    failures = 0
    print("=" * 70)
    print("  verify_fixes — 三项 P0 防回归验收")
    print("=" * 70)
    for name, fn in checks:
        try:
            info = fn()
            print(f"  [PASS] {name}" + (f"  ({info})" if info is not None else ""))
        except Exception as e:
            failures += 1
            print(f"  [FAIL] {name}: {e}")
    print("=" * 70)
    if failures == 0:
        print("  All checks passed")
        return 0
    print(f"  {failures} check(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
