"""
联邦训练引擎 — 编排完整的联邦学习训练流程
"""
import copy
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import List, Dict

from src.data.data_loader import (
    load_city_data, select_top_stations,
    build_station_dataframe, get_station_static_features
)
from src.data.feature_engineering import prepare_station_data
from src.models.tcn_lstm import build_model
from src.federated.aggregation import FLClient, FLServer, ClusteredFLServer
from src.utils.metrics import evaluate_model, set_seed
from src.utils.visualization import (
    generate_all_plots, plot_prediction_vs_actual, plot_error_distribution
)


class FederatedTrainer:
    """
    联邦学习训练器
    负责: 数据准备 -> 客户端创建 -> 联邦训练循环 -> 评估与日志
    """

    def __init__(self, config):
        self.cfg = config
        self.device = config.device if torch.cuda.is_available() else "cpu"
        set_seed(config.seed)

        self.clients: List[FLClient] = []
        self.scalers = {}  # client_id -> scaler
        self.test_loaders = {}  # client_id -> test_loader

    def prepare_city_clients(self, city: str):
        """为单个城市准备所有客户端"""
        print(f"\n{'='*60}")
        print(f"  Loading data for city: {city}")
        print(f"{'='*60}")

        data_dir = os.path.join(self.cfg.output_dir, "..", "data")
        city_data = load_city_data(data_dir, city, self.cfg.data.use_remove_zero)

        # 选择 top-k 站点
        stations = select_top_stations(
            city_data["volume"], self.cfg.data.time_col,
            self.cfg.data.top_k_stations
        )
        print(f"  Selected {len(stations)} stations: {stations[:5]}...")

        for sid in stations:
            # 构建站点 DataFrame
            df = build_station_dataframe(city_data, sid, self.cfg.data.time_col)
            print(f"  Station {sid}: {len(df)} samples, "
                  f"{len(df.columns)-2} features, "
                  f"mean load={df['target'].mean():.2f}")

            # 构建数据集
            train_ds, val_ds, test_ds, scaler = prepare_station_data(
                df,
                seq_len=self.cfg.data.seq_len,
                pred_len=self.cfg.data.pred_len,
                train_ratio=self.cfg.data.train_ratio,
                val_ratio=self.cfg.data.val_ratio,
            )

            if len(train_ds) == 0:
                print(f"    WARNING: Station {sid} has no training samples, skipping")
                continue

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

            # 构建本地模型
            input_dim = train_ds[0][0].shape[1]  # (seq_len, features)
            model = build_model(input_dim, self.cfg.data.pred_len, self.cfg.model)

            client_id = f"{city}_{sid}"
            client = FLClient(client_id, model, train_loader, val_loader, self.device)
            self.clients.append(client)
            self.scalers[client_id] = scaler
            self.test_loaders[client_id] = test_loader

        print(f"\n  Total clients for {city}: {len(self.clients)}")

    def run_federated_training(self) -> Dict:
        """
        执行联邦学习训练循环
        """
        if len(self.clients) == 0:
            raise ValueError("No clients prepared. Call prepare_city_clients first.")

        # 初始化全局模型 (与客户端模型结构相同)
        global_model = copy.deepcopy(self.clients[0].model)

        # 选择聚合策略
        if self.cfg.fed.aggregation == "clustered":
            server = ClusteredFLServer(global_model, self.cfg.fed.n_clusters)
        else:
            server = FLServer(global_model, self.cfg.fed.aggregation)

        mu = self.cfg.fed.fedprox_mu if self.cfg.fed.aggregation == "fedprox" else 0.0

        history = {"rounds": [], "avg_loss": [], "val_metrics": []}

        print(f"\n{'='*60}")
        print(f"  Starting Federated Training")
        print(f"  Strategy: {self.cfg.fed.aggregation}")
        print(f"  Rounds: {self.cfg.fed.num_rounds}")
        print(f"  Clients: {len(self.clients)}")
        print(f"  FedProx mu: {mu}")
        print(f"{'='*60}\n")

        for round_idx in range(self.cfg.fed.num_rounds):
            # 1. 广播全局模型
            global_params = server.get_global_params()

            # 2. 本地训练
            client_params_list = []
            client_weights = []
            round_loss = 0

            for i, client in enumerate(self.clients):
                # 聚类联邦: 使用簇模型; 其他: 使用全局模型
                if isinstance(server, ClusteredFLServer) and round_idx > 0:
                    params_to_send = server.get_cluster_params(i)
                else:
                    params_to_send = copy.deepcopy(global_params)

                client.set_parameters(params_to_send)

                stats = client.train_local(
                    epochs=self.cfg.fed.local_epochs,
                    lr=self.cfg.fed.lr,
                    weight_decay=self.cfg.fed.weight_decay,
                    global_params=global_params,
                    mu=mu,
                )

                client_params_list.append(client.get_parameters())
                client_weights.append(float(client.data_size))
                round_loss += stats["loss"]

            # 3. 聚合
            server.aggregate(client_params_list, client_weights)

            avg_loss = round_loss / len(self.clients)
            history["rounds"].append(round_idx + 1)
            history["avg_loss"].append(avg_loss)

            # 4. 验证 (每5轮)
            if (round_idx + 1) % 5 == 0 or round_idx == 0:
                val_metrics = self._evaluate_all_clients(server)
                history["val_metrics"].append({
                    "round": round_idx + 1, **val_metrics
                })
                print(f"  Round {round_idx+1:3d}/{self.cfg.fed.num_rounds} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Val RMSE: {val_metrics['RMSE']:.4f} | "
                      f"MAE: {val_metrics['MAE']:.4f} | "
                      f"MAPE: {val_metrics['MAPE']:.2f}%")
            else:
                print(f"  Round {round_idx+1:3d}/{self.cfg.fed.num_rounds} | "
                      f"Loss: {avg_loss:.4f}")

        # 5. 最终测试评估
        print(f"\n{'='*60}")
        print(f"  Final Test Evaluation")
        print(f"{'='*60}")
        test_results = self._test_all_clients(server)

        # 保存结果
        self._save_results(history, test_results)

        return test_results

    def _evaluate_all_clients(self, server) -> Dict[str, float]:
        """在所有客户端的验证集上评估"""
        all_rmse, all_mae, all_mape = [], [], []

        for i, client in enumerate(self.clients):
            if isinstance(server, ClusteredFLServer):
                params = server.get_cluster_params(i)
            else:
                params = server.get_global_params()

            client.set_parameters(params)
            metrics = evaluate_model(
                client.model, client.val_loader,
                self.scalers[client.client_id], self.device
            )
            all_rmse.append(metrics["RMSE"])
            all_mae.append(metrics["MAE"])
            all_mape.append(metrics["MAPE"])

        return {
            "RMSE": np.mean(all_rmse),
            "MAE": np.mean(all_mae),
            "MAPE": np.mean(all_mape),
        }

    def _test_all_clients(self, server) -> Dict:
        """在所有客户端的测试集上评估 (最终结果)"""
        results = {}

        for i, client in enumerate(self.clients):
            if isinstance(server, ClusteredFLServer):
                params = server.get_cluster_params(i)
            else:
                params = server.get_global_params()

            client.set_parameters(params)
            metrics = evaluate_model(
                client.model, self.test_loaders[client.client_id],
                self.scalers[client.client_id], self.device
            )
            results[client.client_id] = metrics
            print(f"  {client.client_id}: RMSE={metrics['RMSE']:.4f}, "
                  f"MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%")

        # 汇总
        avg = {
            "RMSE": np.mean([m["RMSE"] for m in results.values()]),
            "MAE": np.mean([m["MAE"] for m in results.values()]),
            "MAPE": np.mean([m["MAPE"] for m in results.values()]),
        }
        results["AVERAGE"] = avg
        print(f"\n  AVERAGE: RMSE={avg['RMSE']:.4f}, "
              f"MAE={avg['MAE']:.4f}, MAPE={avg['MAPE']:.2f}%")

        return results

    def _save_results(self, history: Dict, test_results: Dict):
        """保存训练日志、测试结果和可视化图表"""
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        # 保存历史
        with open(os.path.join(self.cfg.output_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2, default=str)

        # 保存测试结果
        with open(os.path.join(self.cfg.output_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=2, default=str)

        # 生成可视化图表
        print(f"\n  Generating visualizations...")
        generate_all_plots(history, test_results, self.cfg.output_dir)

        # 为第一个客户端生成预测对比图和误差分布图
        if self.clients:
            first_client = self.clients[0]
            first_id = first_client.client_id
            plot_prediction_vs_actual(
                first_client.model, self.test_loaders[first_id],
                self.scalers[first_id], self.cfg.output_dir,
                self.device, station_name=first_id
            )
            plot_error_distribution(
                first_client.model, self.test_loaders[first_id],
                self.scalers[first_id], self.cfg.output_dir, self.device
            )

        print(f"\n  Results saved to {self.cfg.output_dir}")
