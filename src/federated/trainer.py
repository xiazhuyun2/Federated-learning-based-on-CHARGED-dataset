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
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import List, Dict, Optional

from src.data.data_loader import (
    load_city_data, select_top_stations,
    build_station_dataframe, get_station_static_features
)
from src.data.feature_engineering import prepare_station_data
from src.models.tcn_lstm import build_model
from src.federated.aggregation import FLClient, FLServer, ClusteredFLServer
from src.utils.metrics import evaluate_model, set_seed
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.visualization import (
    generate_all_plots, plot_prediction_vs_actual, plot_error_distribution
)


def _get_excluded_param_names(model) -> List[str]:
    """获取不参与联邦聚合的参数名列表 (FedBN + 本地预测头)"""
    excluded = []
    for name, _ in model.named_parameters():
        # FedBN: 跳过所有 BatchNorm 参数
        if "bn" in name:
            excluded.append(name)
        # 本地预测头: 跳过最后的线性层
        if "fc.3" in name:  # nn.Sequential 中第4个子模块 (Linear)
            excluded.append(name)
    return excluded


class FederatedTrainer:
    """
    联邦学习训练器
    负责: 数据准备 -> 客户端创建 -> 联邦训练循环 -> 评估与日志
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

    def prepare_city_clients(self, city: str):
        """为单个城市准备所有客户端"""
        self.city = city
        print(f"\n{'='*60}")
        print(f"  Loading data for city: {city}")
        print(f"{'='*60}")

        data_dir = os.path.join(os.path.dirname(self.cfg.output_dir), "data")
        city_data = load_city_data(data_dir, city, self.cfg.data.use_remove_zero)

        # 选择 top-k 站点 (统计量仅基于训练期, 避免测试集泄漏)
        stations = select_top_stations(
            city_data["volume"], self.cfg.data.time_col,
            self.cfg.data.top_k_stations,
            train_ratio=self.cfg.data.train_ratio + self.cfg.data.val_ratio
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
            self.val_loaders[client_id] = val_loader

        print(f"\n  Total clients for {city}: {len(self.clients)}")

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
            self.excluded_param_names = _get_excluded_param_names(global_model)
            if self._use_fedbn:
                bn_count = sum(1 for n in self.excluded_param_names if "bn" in n)
                print(f"  FedBN enabled: {bn_count} BN params excluded from aggregation")
            if self._use_local_head:
                head_count = sum(1 for n in self.excluded_param_names if "fc" in n)
                print(f"  Local head enabled: {head_count} head params excluded")

        # 选择聚合策略
        if self.cfg.fed.aggregation == "clustered":
            server = ClusteredFLServer(
                global_model, self.cfg.fed.n_clusters,
                mu=self.cfg.fed.fedprox_mu,
                min_cluster_size=self.cfg.fed.min_cluster_size
            )
        else:
            server = FLServer(global_model, self.cfg.fed.aggregation)

        mu = self.cfg.fed.fedprox_mu if self.cfg.fed.aggregation in ("fedprox", "clustered") else 0.0

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
            # 1. 广播模型参数
            global_params = server.get_global_params()

            # 2. 本地训练
            client_params_list = []
            client_weights = []
            round_loss = 0

            for i, client in enumerate(self.clients):
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

                client.set_parameters(params_to_send)

                stats = client.train_local(
                    epochs=self.cfg.fed.local_epochs,
                    lr=self.cfg.fed.lr,
                    weight_decay=self.cfg.fed.weight_decay,
                    global_params=proximal_ref,
                    mu=mu,
                )

                client_params_list.append(client.get_parameters())
                client_weights.append(float(client.data_size))
                round_loss += stats["loss"]

            # 3. 聚合 (支持排除 BN/Head 参数)
            server.aggregate(client_params_list, client_weights,
                             exclude_param_names=self.excluded_param_names if self.excluded_param_names else None)

            avg_loss = round_loss / len(self.clients)
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
                client.set_parameters(params)
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
        """在所有客户端的验证集上评估"""
        all_rmse, all_mae, all_wape = [], [], []

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
            all_wape.append(metrics.get("WAPE", 0))

        return {
            "RMSE": np.mean(all_rmse),
            "MAE": np.mean(all_mae),
            "WAPE": np.mean(all_wape),
        }

    def _test_all_clients_with_preds(self, server) -> tuple:
        """在所有客户端的测试集上评估, 同时返回预测值"""
        results = {}
        predictions = {}

        for i, client in enumerate(self.clients):
            if isinstance(server, ClusteredFLServer):
                params = server.get_cluster_params(i)
            else:
                params = server.get_global_params()

            client.set_parameters(params)
            metrics, preds, targets = evaluate_model(
                client.model, self.test_loaders[client.client_id],
                self.scalers[client.client_id], self.device,
                return_predictions=True
            )
            results[client.client_id] = metrics
            predictions[client.client_id] = {"pred": preds, "target": targets}
            print(f"  {client.client_id}: RMSE={metrics['RMSE']:.4f}, "
                  f"MAE={metrics['MAE']:.4f}, WAPE={metrics.get('WAPE', 0):.2f}%")

        # 汇总: 宏平均 (每个站点等权重) + 微平均 (按样本数加权)
        macro_avg = {
            "RMSE": np.mean([m["RMSE"] for m in results.values()]),
            "MAE": np.mean([m["MAE"] for m in results.values()]),
        }
        # 包含所有指标的平均
        for key in results[list(results.keys())[0]]:
            if key not in macro_avg:
                vals = [m[key] for m in results.values() if key in m]
                if vals:
                    macro_avg[key] = np.mean(vals)

        results["AVERAGE"] = macro_avg
        print(f"\n  AVERAGE: RMSE={macro_avg['RMSE']:.4f}, "
              f"MAE={macro_avg['MAE']:.4f}, WAPE={macro_avg.get('WAPE', 0):.2f}%")

        return results, predictions

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

        # 保存最佳模型
        if self.best_model_state is not None:
            self.tracker.save_best_model(
                self.clients[0].model, self.best_val_rmse)

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
