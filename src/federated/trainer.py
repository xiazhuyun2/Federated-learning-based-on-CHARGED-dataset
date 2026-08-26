"""
联邦训练引擎 — 编排完整的联邦学习训练流程

支持:
  - FedAvg / FedProx / Clustered FL 聚合策略
  - FedBN (BN 参数不聚合)
  - 本地预测头 (FC head 不聚合)
  - 全局训练后本地微调
  - 实验追踪 (唯一输出目录、完整配置保存、最佳 checkpoint)
"""
import copy
import os
import random
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import List, Dict, Optional

from config import DATA_DIR, OUTPUT_DIR
from src.data.data_loader import (
    load_city_data, select_top_stations, stratified_sample_stations,
    build_station_dataframe, get_station_static_features
)
from src.data.feature_engineering import prepare_station_data
from src.models.tcn_lstm import build_model
from src.federated.aggregation import (
    FLClient, FLServer, ClusteredFLServer, CityBalancedServer
)
from src.utils.metrics import evaluate_model, set_seed
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.visualization import (
    generate_all_plots, plot_prediction_vs_actual, plot_error_distribution
)


def _get_excluded_param_names(model, use_fedbn: bool = False,
                             use_local_head: bool = False) -> List[str]:
    """获取不参与联邦聚合的参数名列表 (FedBN + 本地预测头)。

    遍历 state_dict 的全部 key (含 BN 的 running_mean/var 等 buffer),
    按开关分别排除:
      - 所有 BatchNorm 层的 weight/bias/running_* (FedBN)
      - 输出头 fc 序列中最后一个 nn.Linear 的 weight/bias (LocalHead)

    注意: 两个开关必须独立。若忽略开关而永远同时排除 BN+头,
    会让 FedBN / LocalHead / 两者 三种消融产出完全相同的结果。
    """
    excluded = set()

    # 1) FedBN: 所有 BatchNorm 层 (含 running stats buffer, 避免被聚合/广播覆盖)
    if use_fedbn:
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                prefix = f"{name}." if name else ""
                for sub in ("weight", "bias", "running_mean", "running_var",
                            "num_batches_tracked"):
                    excluded.add(f"{prefix}{sub}")

    # 2) LocalHead: fc 序列中最后一个 Linear 层 (预测头)
    if use_local_head:
        head = getattr(model, "fc", None)
        last_linear = None
        if isinstance(head, nn.Sequential):
            for sub in reversed(list(head.children())):
                if isinstance(sub, nn.Linear):
                    last_linear = sub
                    break
        elif isinstance(head, nn.Linear):
            last_linear = head

        if last_linear is not None:
            for name, module in model.named_modules():
                if module is last_linear:
                    prefix = f"{name}." if name else ""
                    excluded.add(f"{prefix}weight")
                    excluded.add(f"{prefix}bias")
                    break

    return sorted(excluded)


def _hash_params(state_dict, exclude=None) -> str:
    """对 state_dict 中非排除参数计算 SHA256 哈希 (用于跨 α 比较第一轮模型)。"""
    import hashlib
    h = hashlib.sha256()
    exclude = set(exclude or [])
    for key in sorted(state_dict.keys()):
        if key in exclude:
            continue
        h.update(key.encode("utf-8"))
        arr = state_dict[key].float().cpu().detach().numpy()
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


class FederatedTrainer:
    """
    联邦学习训练器
    负责: 数据准备 -> 客户端创建 -> 联邦训练循环 -> 评估与日志

    支持单城市模式 (prepare_city_clients) 和多城市模式 (prepare_multi_city_clients).
    """

    def __init__(self, config, run_dir: str = None, city: str = None,
                 method: str = None):
        self.cfg = config
        self.device = config.device if torch.cuda.is_available() else "cpu"
        set_seed(config.seed)

        self.clients: List[FLClient] = []
        self.scalers = {}  # client_id -> scaler
        self.test_loaders = {}  # client_id -> test_loader
        self.val_loaders = {}  # client_id -> val_loader

        # 多城市支持
        self.cities: List[str] = []
        self.city_client_map: Dict[str, List[int]] = {}  # city -> [client_indices]
        self.city_data_sizes: Dict[str, float] = {}       # city -> total N_c

        # 实验追踪
        self.city = city or "unknown"
        self.method = method or config.fed.aggregation
        if run_dir:
            self.run_dir = run_dir
        else:
            from config import get_run_dir, OUTPUT_DIR
            self.run_dir = get_run_dir(self.city, self.method, config.seed, OUTPUT_DIR)
        self.tracker = ExperimentTracker(
            os.path.dirname(os.path.dirname(self.run_dir)),
            self.city, self.method, config.seed
        )
        self.tracker.run_dir = self.run_dir  # 使用已有目录

        # FedBN / 本地预测头: 聚合时排除的参数名
        self.excluded_param_names: List[str] = []
        self._use_fedbn = config.model.use_fedbn
        self._use_local_head = config.model.use_local_head

        # 最佳模型追踪
        self.best_val_rmse = float("inf")
        self.best_round = 0
        self.best_model_state = None

        # 聚合验收 (问题与解决3.txt 第三节三层证据)
        self._city_weights = None
        self.first_round_hash = None

    def _add_station_client(self, city: str, sid: str, city_data: Dict):
        """
        为单个站点构建客户端并注册 (被 prepare_city_clients 和
        prepare_multi_city_clients 共用).
        """
        df = build_station_dataframe(
            city_data, sid, self.cfg.data.time_col,
            timezone_offset=self.cfg.data.timezone_offsets.get(city, 0),
            price_normalization=self.cfg.data.price_normalization,
            add_load_norm=self.cfg.data.load_normalization,
            train_ratio=self.cfg.data.train_ratio,
            use_lag_features=self.cfg.data.use_lag_features,
            use_rolling_features=self.cfg.data.use_rolling_features,
            use_static_features=self.cfg.data.use_static_features,
        )
        print(f"  Station {sid}: {len(df)} samples, "
              f"{len(df.columns)-2} features, "
              f"mean load={df['target'].mean():.2f}")

        train_ds, val_ds, test_ds, scaler = prepare_station_data(
            df,
            seq_len=self.cfg.data.seq_len,
            pred_len=self.cfg.data.pred_len,
            train_ratio=self.cfg.data.train_ratio,
            val_ratio=self.cfg.data.val_ratio,
        )

        if len(train_ds) == 0:
            print(f"    WARNING: Station {sid} has no training samples, skipping")
            return

        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.fed.batch_size,
            shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.fed.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.cfg.fed.batch_size, shuffle=False
        )

        input_dim = train_ds[0][0].shape[1]
        model = build_model(input_dim, self.cfg.data.pred_len, self.cfg.model)

        client_id = f"{city}_{sid}"
        client = FLClient(client_id, model, train_loader, val_loader, self.device)
        self.clients.append(client)
        self.scalers[client_id] = scaler
        self.test_loaders[client_id] = test_loader
        self.val_loaders[client_id] = val_loader

        return client

    def _select_clients(self, round_idx: int) -> List[int]:
        """按 client_fraction 抽样本轮参与训练/聚合的客户端索引 (确定性, 每轮不同)。

        client_fraction=1.0 (默认) 时全部参与, 行为与旧版一致。
        """
        n = len(self.clients)
        frac = float(getattr(self.cfg.fed, "client_fraction", 1.0))
        if frac >= 1.0:
            return list(range(n))
        n_sample = max(1, int(round(n * frac)))
        rng = random.Random(self.cfg.seed * 10000 + round_idx)
        return sorted(rng.sample(range(n), n_sample))

    # ── 选站策略 (场景A分层 / 场景B比例分配) ──────────────────

    def _allocate_city_budget(self, cities: List[str], total_budget: int,
                              size_metric: str = "count"):
        """
        按各城有效站点规模分配客户端预算 (场景B: 自然不均衡)。

        分配逻辑 (对应 问题与解决3.txt 第二节):
          1. 每城保底 min_city_clients 个;
          2. 剩余预算按「超出保底部分的有效站点数」比例分配;
          3. 最大余数法保证 sum(alloc) == total_budget。

        Returns: (alloc: {city: n_clients}, sizes: {city: n_valid_stations})
        """
        sizes = {}
        for city in cities:
            cd = load_city_data(DATA_DIR, city, self.cfg.data.use_remove_zero)
            all_valid = select_top_stations(
                cd["volume"], self.cfg.data.time_col, k=10 ** 6,
                train_ratio=self.cfg.data.train_ratio + self.cfg.data.val_ratio)
            sizes[city] = float(len(all_valid))

        total_size = sum(sizes.values())
        min_city = max(1, int(getattr(self.cfg.data, "min_city_clients", 2)))

        # 剩余预算按 (N_c - min_city) 超出保底的部分分配
        excess = {c: max(0.0, sizes[c] - min_city) for c in cities}
        excess_total = sum(excess.values())

        raw = {}
        for c in cities:
            raw[c] = (total_budget - min_city * len(cities)) * (
                excess[c] / excess_total) if excess_total > 0 else 0.0

        # 最大余数法: 取整 + 按小数部分补足
        alloc = {c: min_city + int(raw[c]) for c in cities}
        frac = {c: raw[c] - int(raw[c]) for c in cities}
        # 小数部分降序, 并列时按城市名保证确定性
        order = sorted(cities, key=lambda c: (-frac[c], c))
        i = 0
        while sum(alloc.values()) < total_budget:
            alloc[order[i % len(order)]] += 1
            i += 1

        print(f"  City budget allocation (total={total_budget}, min/city={min_city}):")
        for c in cities:
            print(f"    {c}: {alloc[c]} clients (valid_stations={int(sizes[c])})")
        return alloc, sizes

    def _select_stations_for_city(self, city: str, city_data: Dict, k: int):
        """按配置的选站策略为单个城市选择 k 个站点。"""
        sel = self.cfg.data.station_selection
        train_ratio = self.cfg.data.train_ratio + self.cfg.data.val_ratio
        if sel.startswith("stratified"):
            dist = "natural" if "natural" in sel else "balanced"
            stations, _ = stratified_sample_stations(
                city_data["volume"], city_data, self.cfg.data.time_col, k,
                train_ratio=train_ratio, distribution=dist, seed=self.cfg.seed)
        else:
            stations = select_top_stations(
                city_data["volume"], self.cfg.data.time_col, k,
                train_ratio=train_ratio)
        return stations

    def _station_list_path(self) -> str:
        d = self.cfg.data
        sig = (f"{d.station_selection}_k{d.top_k_stations}_"
               f"tr{d.train_ratio}_vr{d.val_ratio}_rz{int(d.use_remove_zero)}"
               f"_s{self.cfg.seed}")
        return os.path.join(OUTPUT_DIR, "station_lists", f"{sig}.json")

    def _load_or_create_station_list(self, cities: List[str]) -> Dict[str, List[str]]:
        """
        返回 {city: [station_ids]}, 并持久化到 outputs/station_lists/。

        保证同一选站策略+seed 下, α/FedBN/LocalHead 等所有实验使用完全相同的站点
        (问题与解决3.txt 第二节第 127 行要求)。
        """
        path = self._station_list_path()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                station_list = json.load(f)
            print(f"  Loaded station list from {path}")
            return station_list

        budget = None
        if self.cfg.data.station_selection == "proportional":
            budget, _ = self._allocate_city_budget(
                cities, self.cfg.data.top_k_stations, size_metric="count")

        station_list = {}
        for city in cities:
            cd = load_city_data(DATA_DIR, city, self.cfg.data.use_remove_zero)
            k = budget[city] if budget else self.cfg.data.top_k_stations
            station_list[city] = self._select_stations_for_city(city, cd, k)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(station_list, f, indent=2)
        print(f"  Saved station list to {path}")
        return station_list

    def prepare_city_clients(self, city: str):
        """为单个城市准备所有客户端 (向后兼容)"""
        self.city = city
        self.cities = [city]
        print(f"\n{'='*60}")
        print(f"  Loading data for city: {city}")
        print(f"{'='*60}")

        data_dir = DATA_DIR
        city_data = load_city_data(data_dir, city, self.cfg.data.use_remove_zero)

        # 选择站点
        if self.cfg.data.station_selection.startswith("stratified"):
            dist = "natural" if "natural" in self.cfg.data.station_selection else "balanced"
            stations, _ = stratified_sample_stations(
                city_data["volume"], city_data, self.cfg.data.time_col,
                self.cfg.data.top_k_stations,
                train_ratio=self.cfg.data.train_ratio + self.cfg.data.val_ratio,
                distribution=dist,
                seed=self.cfg.seed,
            )
        else:
            stations = select_top_stations(
                city_data["volume"], self.cfg.data.time_col,
                self.cfg.data.top_k_stations,
                train_ratio=self.cfg.data.train_ratio + self.cfg.data.val_ratio
            )
        print(f"  Selected {len(stations)} stations: {stations[:5]}...")

        for sid in stations:
            self._add_station_client(city, sid, city_data)

        # 更新城市-客户端映射
        self.city_client_map[city] = list(range(len(self.clients)))
        if self.clients:
            self.city_data_sizes[city] = sum(
                c.data_size for c in self.clients
            )

        print(f"\n  Total clients for {city}: {len(self.clients)}")

    def prepare_multi_city_clients(self, cities: List[str]):
        """为多个城市准备所有客户端 (多城市联邦学习)"""
        self.cities = cities
        print(f"\n{'='*60}")
        print(f"  Loading data for {len(cities)} cities: {cities}")
        print(f"{'='*60}")

        data_dir = DATA_DIR

        # 统一选站 (场景A分层 / 场景B比例分配), 并持久化保证跨实验一致
        station_list = self._load_or_create_station_list(cities)

        for city in cities:
            print(f"\n  --- City: {city} ---")
            try:
                city_data = load_city_data(data_dir, city, self.cfg.data.use_remove_zero)
                stations = station_list.get(city, [])
                print(f"  Selected {len(stations)} stations: {stations[:5]}...")

                # 记录该城市客户端在当前 client 列表中的起始位置
                city_start_idx = len(self.clients)
                city_clients_added = 0

                for sid in stations:
                    result = self._add_station_client(city, sid, city_data)
                    if result is not None:
                        city_clients_added += 1

                # 更新城市-客户端映射
                city_end_idx = len(self.clients)
                self.city_client_map[city] = list(
                    range(city_start_idx, city_end_idx)
                )
                self.city_data_sizes[city] = sum(
                    self.clients[i].data_size
                    for i in range(city_start_idx, city_end_idx)
                )
                print(f"  City {city}: {city_clients_added} clients "
                      f"(N_c={self.city_data_sizes[city]:.0f})")

            except Exception as e:
                print(f"  ERROR loading city {city}: {e}")
                import traceback
                traceback.print_exc()

        self.city = "+".join(cities)  # 用于目录名
        print(f"\n  Total clients across {len(cities)} cities: {len(self.clients)}")
        for c in cities:
            n_clients = len(self.city_client_map.get(c, []))
            n_data = self.city_data_sizes.get(c, 0)
            print(f"    {c}: {n_clients} stations, N_c={n_data:.0f}")

    def run_federated_training(self) -> Dict:
        """
        执行联邦学习训练循环
        """
        if len(self.clients) == 0:
            raise ValueError("No clients prepared. Call prepare_city_clients first.")

        # 初始化全局模型
        global_model = copy.deepcopy(self.clients[0].model)

        # 确定聚合时排除的参数
        if self._use_fedbn or self._use_local_head:
            self.excluded_param_names = _get_excluded_param_names(
                global_model, self._use_fedbn, self._use_local_head)
            if self._use_fedbn:
                bn_count = sum(1 for n in self.excluded_param_names if "bn" in n)
                print(f"  FedBN enabled: {bn_count} BN params excluded from aggregation")
            if self._use_local_head:
                head_count = sum(1 for n in self.excluded_param_names if "fc" in n)
                print(f"  Local head enabled: {head_count} head params excluded")

        # 选择聚合策略
        is_multi_city = len(self.cities) > 1
        server_cls = None
        use_city_balance = (
            is_multi_city
            and self.cfg.fed.multi_city_mode == "multi_city"
        )

        if self.cfg.fed.aggregation == "clustered":
            server = ClusteredFLServer(
                global_model, self.cfg.fed.n_clusters,
                mu=self.cfg.fed.fedprox_mu,
                min_cluster_size=self.cfg.fed.min_cluster_size
            )
        elif use_city_balance and self.cfg.fed.multi_city_mode == "multi_city":
            from src.federated.aggregation import CityBalancedServer
            server = CityBalancedServer(
                global_model,
                alpha=self.cfg.fed.city_weight_alpha,
                aggregation=self.cfg.fed.aggregation,
            )
            server.set_city_groups(self.cities, self.city_data_sizes)
            self._city_weights = dict(getattr(server, "city_weights", {}))
            print(f"  City-Balanced Server: α={self.cfg.fed.city_weight_alpha}")
        else:
            server = FLServer(global_model, self.cfg.fed.aggregation)

        # FedProx mu
        if self.cfg.fed.aggregation in ("fedprox",):
            mu = self.cfg.fed.fedprox_mu
        elif self.cfg.fed.aggregation == "clustered":
            mu = self.cfg.fed.fedprox_mu
        else:
            mu = 0.0

        history = {"rounds": [], "avg_loss": [], "val_metrics": []}

        print(f"\n{'='*60}")
        print(f"  Starting Federated Training")
        print(f"  Strategy: {self.cfg.fed.aggregation}")
        print(f"  Rounds: {self.cfg.fed.num_rounds}")
        print(f"  Clients: {len(self.clients)}")
        print(f"  FedProx mu: {mu}")
        print(f"  FedBN: {self._use_fedbn}")
        print(f"  Local Head: {self._use_local_head}")
        print(f"  Run Dir: {self.run_dir}")
        print(f"{'='*60}\n")

        for round_idx in range(self.cfg.fed.num_rounds):
            # 0. 抽样本轮参与的客户端 (client_fraction; 1.0 = 全部)
            selected_indices = self._select_clients(round_idx)

            # 1. 广播模型参数
            global_params = server.get_global_params()

            # 2. 本地训练
            client_params_list = []
            client_weights = []
            client_city_list = []
            round_loss = 0

            for i in selected_indices:
                client = self.clients[i]
                # 确定发送给客户端的参数
                if isinstance(server, ClusteredFLServer):
                    if round_idx > 0:
                        params_to_send = server.get_cluster_params(i)
                    else:
                        params_to_send = copy.deepcopy(global_params)
                    # 簇内 FedProx: 使用簇参数作为近端参考
                    proximal_ref = server.get_cluster_proximal_params(i) if round_idx > 0 else global_params
                else:
                    params_to_send = copy.deepcopy(global_params)
                    proximal_ref = global_params

                # FedBN/LocalHead: 第 0 轮全量广播 (统一初始化), 之后跳过 BN/头,
                # 保留各客户端本地训练得到的 BN 统计量 / 本地预测头。
                exclude = (self.excluded_param_names
                           if (round_idx > 0 and self.excluded_param_names) else None)
                client.set_parameters(params_to_send, exclude_names=exclude)

                stats = client.train_local(
                    epochs=self.cfg.fed.local_epochs,
                    lr=self.cfg.fed.lr,
                    weight_decay=self.cfg.fed.weight_decay,
                    global_params=proximal_ref,
                    mu=mu,
                )

                client_params_list.append(client.get_parameters())
                client_weights.append(float(client.data_size))
                client_city_list.append(client.client_id.split("_")[0])
                round_loss += stats["loss"]

            # 3. 聚合 (支持排除 BN/Head 参数)
            # 3. 聚合 (支持城市平衡)
            if use_city_balance and isinstance(server, CityBalancedServer):
                server.aggregate(
                    client_params_list, client_weights,
                    exclude_param_names=self.excluded_param_names if self.excluded_param_names else None,
                    client_city_map=client_city_list,
                )
            else:
                server.aggregate(
                    client_params_list, client_weights,
                    exclude_param_names=self.excluded_param_names if self.excluded_param_names else None)

            # 记录第一轮聚合后的共享参数哈希 (用于跨 α 比较, 验收第三层)
            if round_idx == 0:
                self.first_round_hash = _hash_params(
                    server.global_model.state_dict(),
                    exclude=(self.excluded_param_names
                             if self.excluded_param_names else None))

            avg_loss = round_loss / max(1, len(selected_indices))
            history["rounds"].append(round_idx + 1)
            history["avg_loss"].append(avg_loss)

            # 4. 验证 + 保存最佳模型
            if (round_idx + 1) % 5 == 0 or round_idx == 0:
                val_metrics = self._evaluate_all_clients(server)
                history["val_metrics"].append({
                    "round": round_idx + 1, **val_metrics
                })
                print(f"  Round {round_idx+1:3d}/{self.cfg.fed.num_rounds} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Val RMSE: {val_metrics['RMSE']:.4f} | "
                      f"MAE: {val_metrics['MAE']:.4f} | "
                      f"WAPE: {val_metrics.get('WAPE', 0):.2f}%")

                # 追踪最佳模型
                if val_metrics["RMSE"] < self.best_val_rmse:
                    self.best_val_rmse = val_metrics["RMSE"]
                    self.best_round = round_idx + 1
                    self.best_model_state = copy.deepcopy(server.get_global_params())
            else:
                print(f"  Round {round_idx+1:3d}/{self.cfg.fed.num_rounds} | "
                      f"Loss: {avg_loss:.4f}")

        # ── 全局训练后本地微调 ──
        if self.cfg.fed.finetune_epochs > 0:
            print(f"\n{'='*60}")
            print(f"  Local Fine-tuning: {self.cfg.fed.finetune_epochs} epochs each")
            print(f"{'='*60}")
            for client in self.clients:
                if isinstance(server, ClusteredFLServer):
                    idx = self.clients.index(client)
                    params = server.get_cluster_params(idx)
                else:
                    params = server.get_global_params()
                client.set_parameters(
                    params,
                    exclude_names=(self.excluded_param_names
                                   if self.excluded_param_names else None))
                stats = client.train_local(
                    epochs=self.cfg.fed.finetune_epochs,
                    lr=self.cfg.fed.lr * 0.1,
                    weight_decay=self.cfg.fed.weight_decay,
                    global_params=params,
                    mu=mu,
                )
                print(f"  {client.client_id}: fine-tune loss={stats['loss']:.4f}")

        # 5. 最终测试评估 (使用验证集最优模型)
        print(f"\n{'='*60}")
        print(f"  Final Test Evaluation (best model from round {self.best_round})")
        print(f"{'='*60}")

        # 恢复最佳模型
        if self.best_model_state is not None:
            if isinstance(server, ClusteredFLServer):
                server.global_model.load_state_dict(self.best_model_state)
            else:
                server.global_model.load_state_dict(self.best_model_state)

        test_results, predictions = self._test_all_clients_with_preds(server)

        # 保存结果
        self._save_results(history, test_results, predictions)

        return test_results

    def _evaluate_all_clients(self, server) -> Dict[str, float]:
        """在所有客户端的验证集上评估, 多城市模式返回多层级指标"""
        is_multi_city = len(self.cities) > 1
        per_client_metrics = {}

        for i, client in enumerate(self.clients):
            if isinstance(server, ClusteredFLServer):
                params = server.get_cluster_params(i)
            else:
                # CityBalanced 的最佳轮选择统一用 global 模型 (α 只影响 global, Step2),
                # 与测试阶段保持一致; city 模型 (Step1) 另在测试阶段单独报告。
                params = server.get_global_params()

            client.set_parameters(
                params,
                exclude_names=(self.excluded_param_names
                               if self.excluded_param_names else None))
            metrics = evaluate_model(
                client.model, client.val_loader,
                self.scalers[client.client_id], self.device
            )
            per_client_metrics[client.client_id] = metrics

        # 基础宏平均 (所有站点等权)
        all_rmse = [m["RMSE"] for m in per_client_metrics.values()]
        all_mae = [m["MAE"] for m in per_client_metrics.values()]
        all_wape = [m.get("WAPE", 0) for m in per_client_metrics.values()]

        result = {
            "RMSE": np.mean(all_rmse),
            "MAE": np.mean(all_mae),
            "WAPE": np.mean(all_wape),
        }

        # 多城市: 额外计算城市级宏平均
        if is_multi_city and self.city_client_map:
            city_rmse, city_mae, city_wape = [], [], []
            for city in self.cities:
                city_client_ids = [
                    self.clients[idx].client_id
                    for idx in self.city_client_map.get(city, [])
                ]
                if not city_client_ids:
                    continue
                city_vals_rmse = [
                    per_client_metrics[cid]["RMSE"]
                    for cid in city_client_ids if cid in per_client_metrics
                ]
                city_vals_mae = [
                    per_client_metrics[cid]["MAE"]
                    for cid in city_client_ids if cid in per_client_metrics
                ]
                city_vals_wape = [
                    per_client_metrics[cid].get("WAPE", 0)
                    for cid in city_client_ids if cid in per_client_metrics
                ]
                if city_vals_rmse:
                    city_rmse.append(np.mean(city_vals_rmse))
                    city_mae.append(np.mean(city_vals_mae))
                    city_wape.append(np.mean(city_vals_wape))

            if city_rmse:
                result["macro_city_RMSE"] = np.mean(city_rmse)
                result["macro_city_MAE"] = np.mean(city_mae)
                result["macro_city_WAPE"] = np.mean(city_wape)

        return result

    def _test_all_clients_with_preds(self, server) -> tuple:
        """在所有客户端的测试集上评估, 同时返回预测值。

        对 CityBalancedServer, 主指标 macro_city 使用 **global 模型 (Step2)** —
        因为 α 只影响 global 聚合, 若仍用 city 模型 (Step1) 会让 α 扫描失效;
        city 模型另存为 macro_city_city_model / per_city_city_model / worst_city_city_model。
        """
        # 主评估: 统一用 global 模型 (非 CityBalanced 本就是 global / cluster)
        results, predictions = self._test_clients(server, use_city_model=False)

        # CityBalanced: 额外报告 city 模型 (Step1 城市内聚合)
        if isinstance(server, CityBalancedServer):
            city_results, _ = self._test_clients(server, use_city_model=True)
            for key in ("macro_city", "per_city", "worst_city"):
                if key in city_results:
                    results[f"{key}_city_model"] = city_results[key]

        return results, predictions

    def _test_clients(self, server, use_city_model: bool = False) -> tuple:
        """在测试集上评估所有客户端, 返回 (results, predictions)。

        use_city_model=True 时对 CityBalancedServer 使用 city 模型 (Step1),
        否则使用 global 模型 (Step2)。
        """
        results = {}
        predictions = {}
        micro_abs_err = 0.0
        micro_abs_target = 0.0
        micro_sq_err = 0.0
        micro_n = 0

        for i, client in enumerate(self.clients):
            if use_city_model and isinstance(server, CityBalancedServer):
                client_city = client.client_id.split("_")[0]
                city_params = server.get_city_params(client_city)
                params = city_params if city_params else server.get_global_params()
            elif isinstance(server, ClusteredFLServer):
                params = server.get_cluster_params(i)
            else:
                params = server.get_global_params()

            # FedBN/LocalHead: 保留本地 BN/头, 只更新共享参数
            client.set_parameters(
                params,
                exclude_names=(self.excluded_param_names
                               if self.excluded_param_names else None))

            metrics, preds, targets = evaluate_model(
                client.model, self.test_loaders[client.client_id],
                self.scalers[client.client_id], self.device,
                return_predictions=True
            )
            results[client.client_id] = metrics
            predictions[client.client_id] = {"pred": preds, "target": targets}
            micro_abs_err += float(np.sum(np.abs(preds - targets)))
            micro_abs_target += float(np.sum(np.abs(targets)))
            micro_sq_err += float(np.sum((preds - targets) ** 2))
            micro_n += int(targets.size)
            if not use_city_model:
                print(f"  {client.client_id}: RMSE={metrics['RMSE']:.4f}, "
                      f"MAE={metrics['MAE']:.4f}, WAPE={metrics.get('WAPE', 0):.2f}%")

        # 汇总: 宏平均 (每个站点等权重)
        macro_avg = {
            "RMSE": np.mean([m["RMSE"] for m in results.values()]),
            "MAE": np.mean([m["MAE"] for m in results.values()]),
        }
        for key in results[list(results.keys())[0]]:
            if key not in macro_avg:
                vals = [m[key] for m in results.values() if key in m]
                if vals:
                    macro_avg[key] = np.mean(vals)

        results["AVERAGE"] = macro_avg
        if not use_city_model:
            print(f"\n  MACRO-STATION: RMSE={macro_avg['RMSE']:.4f}, "
                  f"MAE={macro_avg['MAE']:.4f}, WAPE={macro_avg.get('WAPE', 0):.2f}%")

        # Micro 指标 (所有站点样本合并计算, 反映自然数据分布)
        results["micro"] = {
            "WAPE": float(micro_abs_err / micro_abs_target * 100) if micro_abs_target > 0 else 0.0,
            "RMSE": float(np.sqrt(micro_sq_err / micro_n)) if micro_n > 0 else 0.0,
            "MAE": float(micro_abs_err / micro_n) if micro_n > 0 else 0.0,
        }
        if not use_city_model:
            print(f"  MICRO (all stations pooled): RMSE={results['micro']['RMSE']:.4f}, "
                  f"WAPE={results['micro']['WAPE']:.2f}%")

        results.update(self._city_level_summary(
            results, label="city model" if use_city_model else ""))

        return results, predictions

    def _city_level_summary(self, client_metrics: Dict[str, dict],
                            label: str = "") -> dict:
        """从 {client_id: metrics} 计算城市级宏平均 / per_city / worst_city。

        仅多城市模式返回有效内容; 单城市返回空 dict。
        """
        is_multi_city = len(self.cities) > 1
        if not (is_multi_city and self.city_client_map):
            return {}

        tag = f"[{label}] " if label else ""
        per_city_metrics = {}
        city_rmse_list, city_mae_list, city_wape_list = [], [], []
        for city in self.cities:
            city_client_ids = [
                self.clients[idx].client_id
                for idx in self.city_client_map.get(city, [])
            ]
            city_vals = {
                cid: client_metrics[cid]
                for cid in city_client_ids if cid in client_metrics
            }
            if not city_vals:
                continue
            city_avg = {}
            for key in ["RMSE", "MAE", "WAPE", "SMAPE", "NRMSE", "MAPE_raw"]:
                vals = [m[key] for m in city_vals.values() if key in m]
                if vals:
                    city_avg[key] = float(np.mean(vals))
            city_rmse_list.append(city_avg["RMSE"])
            city_mae_list.append(city_avg["MAE"])
            city_wape_list.append(city_avg.get("WAPE", 0))
            per_city_metrics[city] = city_avg
            print(f"  {tag}{city}: RMSE={city_avg['RMSE']:.4f}, "
                  f"MAE={city_avg['MAE']:.4f}, WAPE={city_avg.get('WAPE', 0):.2f}%")

        worst_city = max(per_city_metrics.items(),
                         key=lambda kv: kv[1].get("WAPE", 0))
        summary = {
            "macro_city": {
                "RMSE": float(np.mean(city_rmse_list)),
                "MAE": float(np.mean(city_mae_list)),
                "WAPE": float(np.mean(city_wape_list)),
            },
            "per_city": per_city_metrics,
            "worst_city": {
                "city": worst_city[0],
                "WAPE": worst_city[1].get("WAPE", 0),
                "RMSE": worst_city[1].get("RMSE", 0),
                "MAE": worst_city[1].get("MAE", 0),
            },
        }
        print(f"  {tag}WORST-CITY: {worst_city[0]} "
              f"WAPE={worst_city[1].get('WAPE', 0):.2f}%")
        print(f"\n  {tag}MACRO-CITY"
              f"{' (paper primary)' if not label else ''}: "
              f"RMSE={summary['macro_city']['RMSE']:.4f}, "
              f"MAE={summary['macro_city']['MAE']:.4f}, "
              f"WAPE={summary['macro_city']['WAPE']:.2f}%")
        return summary

    def _save_results(self, history: Dict, test_results: Dict,
                      predictions: Dict = None):
        """保存训练日志、测试结果、模型和可视化图表"""
        os.makedirs(self.run_dir, exist_ok=True)

        # 保存配置
        self.tracker.run_dir = self.run_dir
        self.tracker.save_config(self.cfg)

        # 保存训练历史
        with open(os.path.join(self.run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2, default=str)

        # 保存测试结果
        with open(os.path.join(self.run_dir, "metrics.json"), "w") as f:
            json.dump(test_results, f, indent=2, default=str)

        # 保存聚合验收信息 (N_c / β_c / 第一轮模型哈希) —— 问题与解决3.txt 第三节三层证据
        total_nc = sum(self.city_data_sizes.values())
        meta = {
            "alpha": getattr(self.cfg.fed, "city_weight_alpha", None),
            "station_selection": self.cfg.data.station_selection,
            "cities": list(self.cities),
            "per_city": {},
            "city_weights": self._city_weights,
            "first_round_hash": self.first_round_hash,
        }
        for c in self.cities:
            meta["per_city"][c] = {
                "n_stations": len(self.city_client_map.get(c, [])),
                "N_c": float(self.city_data_sizes.get(c, 0.0)),
                "data_fraction": (float(self.city_data_sizes.get(c, 0.0) / total_nc)
                                  if total_nc > 0 else 0.0),
            }
        with open(os.path.join(self.run_dir, "aggregation_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # 保存最佳模型 (验证集最优的 global 模型, 而非 clients[0].model)
        if self.best_model_state is not None:
            best_model = copy.deepcopy(self.clients[0].model)
            best_model.load_state_dict(self.best_model_state)
            self.tracker.save_best_model(best_model, self.best_val_rmse)

        # FedBN/LocalHead: 同时保存所有客户端的本地 BN/头 (个性化参数),
        # 否则只存 client[0] 会丢失其余站点的预测头。
        if self.excluded_param_names:
            local_models = {
                client.client_id: client.model.state_dict()
                for client in self.clients
            }
            torch.save(local_models, os.path.join(self.run_dir, "local_models.pt"))

        # 保存预测值
        if predictions:
            self.tracker.save_predictions(predictions)

        # 生成可视化图表
        print(f"\n  Generating visualizations...")
        generate_all_plots(history, test_results, self.run_dir)

        # 为第一个客户端生成预测对比图和误差分布图
        if self.clients:
            first_client = self.clients[0]
            first_id = first_client.client_id
            plot_prediction_vs_actual(
                first_client.model, self.test_loaders[first_id],
                self.scalers[first_id], self.run_dir,
                self.device, station_name=first_id
            )
            plot_error_distribution(
                first_client.model, self.test_loaders[first_id],
                self.scalers[first_id], self.run_dir, self.device
            )

        print(f"\n  Results saved to {self.run_dir}")
