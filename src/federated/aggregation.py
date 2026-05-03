"""
联邦学习聚合策略 — FedAvg / FedProx / Clustered FL

每个 Client 持有一个本地模型, Server 负责聚合全局模型。
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Optional
from collections import OrderedDict


# ============================================================
# Client — 本地训练
# ============================================================

class FLClient:
    """联邦学习客户端 (单个充电站)"""

    def __init__(self, client_id: str, model: nn.Module,
                 train_loader: DataLoader, val_loader: DataLoader,
                 device: str = "cpu"):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.data_size = len(train_loader.dataset)

    def get_parameters(self) -> OrderedDict:
        return copy.deepcopy(self.model.state_dict())

    def set_parameters(self, params: OrderedDict):
        self.model.load_state_dict(params)

    def train_local(self, epochs: int, lr: float, weight_decay: float,
                    global_params: Optional[OrderedDict] = None,
                    mu: float = 0.0) -> Dict:
        """
        本地训练
        Args:
            global_params: 全局模型参数 (用于 FedProx 近端项)
            mu: FedProx 近端系数, 0 则退化为 FedAvg
        Returns:
            训练统计信息
        """
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        total_loss = 0
        total_samples = 0

        for epoch in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                pred = self.model(x)
                loss = criterion(pred, y)

                # FedProx: 添加近端正则化项
                if mu > 0 and global_params is not None:
                    proximal_term = 0.0
                    for name, param in self.model.named_parameters():
                        global_param = global_params[name].to(self.device)
                        proximal_term += ((param - global_param) ** 2).sum()
                    loss += (mu / 2) * proximal_term

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        self.model.to("cpu")

        return {"client_id": self.client_id, "loss": avg_loss,
                "samples": total_samples}


# ============================================================
# Server — 聚合策略
# ============================================================

class FLServer:
    """联邦学习服务器"""

    def __init__(self, global_model: nn.Module, aggregation: str = "fedavg"):
        self.global_model = global_model
        self.aggregation = aggregation

    def get_global_params(self) -> OrderedDict:
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_params_list: List[OrderedDict],
                  client_weights: List[float]):
        """
        FedAvg / FedProx 加权平均聚合
        """
        total_weight = sum(client_weights)
        new_params = OrderedDict()

        for key in client_params_list[0]:
            new_params[key] = sum(
                params[key].float() * (w / total_weight)
                for params, w in zip(client_params_list, client_weights)
            )

        self.global_model.load_state_dict(new_params)


# ============================================================
# Clustered FL — 谱聚类联邦学习
# ============================================================

def compute_model_similarity(params_list: List[OrderedDict]) -> np.ndarray:
    """
    计算客户端模型参数之间的余弦相似度矩阵
    用于谱聚类分组
    """
    # 将参数展平为一维向量
    flat_params = []
    for params in params_list:
        flat = torch.cat([p.float().flatten() for p in params.values()])
        flat_params.append(flat)

    n = len(flat_params)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cos_sim = torch.nn.functional.cosine_similarity(
                flat_params[i].unsqueeze(0),
                flat_params[j].unsqueeze(0)
            ).item()
            sim_matrix[i, j] = cos_sim
            sim_matrix[j, i] = cos_sim

    return sim_matrix


def cluster_clients(params_list: List[OrderedDict],
                    n_clusters: int = 3) -> List[List[int]]:
    """
    基于模型参数相似度进行谱聚类
    Returns: 每个簇包含的客户端索引列表
    """
    from sklearn.cluster import SpectralClustering

    sim_matrix = compute_model_similarity(params_list)

    # 确保相似度矩阵非负
    sim_matrix = (sim_matrix + 1) / 2  # 将 [-1,1] 映射到 [0,1]

    n_clients = len(params_list)
    actual_clusters = min(n_clusters, n_clients)

    clustering = SpectralClustering(
        n_clusters=actual_clusters,
        affinity="precomputed",
        random_state=42
    ).fit(sim_matrix)

    clusters = [[] for _ in range(actual_clusters)]
    for idx, label in enumerate(clustering.labels_):
        clusters[label].append(idx)

    return clusters


class ClusteredFLServer:
    """
    聚类联邦学习服务器
    - 先用谱聚类将客户端分组
    - 每个簇内独立 FedAvg 聚合
    - 每个客户端接收本簇的聚合模型
    """

    def __init__(self, global_model: nn.Module, n_clusters: int = 3):
        self.global_model = global_model
        self.n_clusters = n_clusters
        self.cluster_models: List[OrderedDict] = []
        self.client_cluster_map: Dict[int, int] = {}

    def get_global_params(self) -> OrderedDict:
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_params_list: List[OrderedDict],
                  client_weights: List[float]):
        """
        聚类后分组聚合
        """
        # 聚类
        clusters = cluster_clients(client_params_list, self.n_clusters)

        # 更新客户端到簇的映射
        self.client_cluster_map = {}
        for cluster_idx, members in enumerate(clusters):
            for member_idx in members:
                self.client_cluster_map[member_idx] = cluster_idx

        # 为每个簇独立聚合
        self.cluster_models = []
        for cluster_idx, members in enumerate(clusters):
            if len(members) == 0:
                self.cluster_models.append(self.get_global_params())
                continue

            cluster_params = [client_params_list[i] for i in members]
            cluster_w = [client_weights[i] for i in members]
            total_w = sum(cluster_w)

            new_params = OrderedDict()
            for key in cluster_params[0]:
                new_params[key] = sum(
                    p[key].float() * (w / total_w)
                    for p, w in zip(cluster_params, cluster_w)
                )
            self.cluster_models.append(new_params)

        # 全局模型用所有客户端的加权平均更新 (用于初始化新客户端)
        total_weight = sum(client_weights)
        global_params = OrderedDict()
        for key in client_params_list[0]:
            global_params[key] = sum(
                p[key].float() * (w / total_weight)
                for p, w in zip(client_params_list, client_weights)
            )
        self.global_model.load_state_dict(global_params)

    def get_cluster_params(self, client_idx: int) -> OrderedDict:
        """获取客户端所属簇的聚合模型参数"""
        cluster_idx = self.client_cluster_map.get(client_idx, 0)
        if cluster_idx < len(self.cluster_models):
            return copy.deepcopy(self.cluster_models[cluster_idx])
        return self.get_global_params()
