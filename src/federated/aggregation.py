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
            global_params: 全局/簇模型参数 (用于 FedProx 近端项)
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
    """联邦学习服务器 (FedAvg / FedProx)"""

    def __init__(self, global_model: nn.Module, aggregation: str = "fedavg"):
        self.global_model = global_model
        self.aggregation = aggregation

    def get_global_params(self) -> OrderedDict:
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_params_list: List[OrderedDict],
                  client_weights: List[float],
                  exclude_param_names: Optional[List[str]] = None):
        """
        FedAvg / FedProx 加权平均聚合

        Args:
            exclude_param_names: 不参与聚合的参数名 (用于 FedBN / 本地预测头)
        """
        total_weight = sum(client_weights)
        new_params = OrderedDict()

        exclude_set = set(exclude_param_names) if exclude_param_names else set()

        for key in client_params_list[0]:
            if key in exclude_set:
                # 共享参数的全局聚合结果对排除参数无意义, 保留当前值
                new_params[key] = self.global_model.state_dict()[key].clone()
                continue
            new_params[key] = sum(
                params[key].float() * (w / total_weight)
                for params, w in zip(client_params_list, client_weights)
            )

        self.global_model.load_state_dict(new_params)


# ============================================================
# Clustered FL — 谱聚类联邦学习
# ============================================================

def compute_station_features(load_data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    从训练期负荷数据提取站点特征, 用于谱聚类

    每个站点的特征向量:
      - 24小时平均日负荷曲线 (24维)
      - 负荷均值、标准差
      - 峰谷比
      - 24小时/168小时自相关
      - 零值率

    Args:
        load_data: {station_id: 1D array of hourly load (training period only)}

    Returns:
        (n_stations, n_features) 特征矩阵
    """
    features = []
    station_ids = list(load_data.keys())

    for sid in station_ids:
        data = load_data[sid]
        feats = []

        # 24小时平均日负荷曲线
        if len(data) >= 24:
            daily_profile = np.array([
                np.mean(data[i::24]) for i in range(24)
            ])
            # 归一化
            profile_max = daily_profile.max()
            if profile_max > 0:
                daily_profile = daily_profile / profile_max
            feats.extend(daily_profile.tolist())
        else:
            feats.extend([0] * 24)

        # 统计量
        feats.append(np.mean(data))
        feats.append(np.std(data))
        feats.append(np.max(data) / (np.mean(data) + 1e-8))  # 峰均比
        feats.append(np.percentile(data, 90) / (np.percentile(data, 10) + 1e-8))  # 峰谷比

        # 自相关
        if len(data) > 168:
            acf_24 = np.corrcoef(data[:-24], data[24:])[0, 1]
            acf_168 = np.corrcoef(data[:-168], data[168:])[0, 1]
        else:
            acf_24, acf_168 = 0, 0
        feats.append(acf_24 if not np.isnan(acf_24) else 0)
        feats.append(acf_168 if not np.isnan(acf_168) else 0)

        # 零值率
        feats.append(np.mean(data < 0.01))

        features.append(feats)

    return np.array(features, dtype=np.float32)


def compute_model_similarity(params_list: List[OrderedDict]) -> np.ndarray:
    """
    计算客户端模型参数之间的余弦相似度矩阵 (旧方法, 不推荐)
    """
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
                    n_clusters: int = 3,
                    load_features: np.ndarray = None,
                    min_cluster_size: int = 2) -> List[List[int]]:
    """
    谱聚类分组客户端

    推荐使用 load_features (训练期负荷特征), 比模型参数聚类更稳定。

    Args:
        params_list: 客户端模型参数列表
        n_clusters: 目标簇数 (会自动调整)
        load_features: 站点负荷特征矩阵, None 则退化为模型参数聚类
        min_cluster_size: 最小簇大小

    Returns: 每个簇包含的客户端索引列表
    """
    from sklearn.cluster import SpectralClustering

    n_clients = len(params_list)

    # 限制簇数, 保证最小簇大小
    max_clusters = max(1, n_clients // min_cluster_size)
    actual_clusters = min(n_clusters, max_clusters)
    actual_clusters = max(1, actual_clusters)

    if load_features is not None and len(load_features) == n_clients:
        # 基于负荷特征的聚类 (推荐)
        from sklearn.preprocessing import StandardScaler
        feats_scaled = StandardScaler().fit_transform(load_features)

        # 构建 RBF 亲和矩阵
        from sklearn.metrics.pairwise import rbf_kernel
        gamma = 1.0 / feats_scaled.shape[1]
        affinity = rbf_kernel(feats_scaled, gamma=gamma)

        clustering = SpectralClustering(
            n_clusters=actual_clusters,
            affinity="precomputed",
            random_state=42
        ).fit(affinity)
    else:
        # 基于模型参数的聚类 (旧方法, 备用)
        sim_matrix = compute_model_similarity(params_list)
        sim_matrix = (sim_matrix + 1) / 2  # 将 [-1,1] 映射到 [0,1]

        clustering = SpectralClustering(
            n_clusters=actual_clusters,
            affinity="precomputed",
            random_state=42
        ).fit(sim_matrix)

    clusters = [[] for _ in range(actual_clusters)]
    for idx, label in enumerate(clustering.labels_):
        clusters[label].append(idx)

    # 验证簇大小
    cluster_sizes = [len(c) for c in clusters]
    print(f"  Clusters: {cluster_sizes} "
          f"(min={min(cluster_sizes)}, "
          f"method={'load_profile' if load_features is not None else 'model_params'})")

    return clusters


class ClusteredFLServer:
    """
    聚类联邦学习服务器

    - 基于负荷特征谱聚类分组
    - 每个簇内独立 FedProx 聚合
    - 支持排除参数 (FedBN / 本地预测头)
    """

    def __init__(self, global_model: nn.Module, n_clusters: int = 3,
                 mu: float = 0.01, min_cluster_size: int = 2):
        self.global_model = global_model
        self.n_clusters = n_clusters
        self.mu = mu
        self.min_cluster_size = min_cluster_size
        self.cluster_models: List[OrderedDict] = []
        self.cluster_proximal_params: List[OrderedDict] = []  # 簇内 FedProx 参考
        self.client_cluster_map: Dict[int, int] = {}
        self.load_features: Optional[np.ndarray] = None

    def set_load_features(self, features: np.ndarray):
        """设置用于聚类的负荷特征 (在 prepare 阶段调用)"""
        self.load_features = features

    def get_global_params(self) -> OrderedDict:
        return copy.deepcopy(self.global_model.state_dict())

    def get_cluster_proximal_params(self, client_idx: int) -> OrderedDict:
        """获取客户端所属簇的 FedProx 近端参考参数"""
        cluster_idx = self.client_cluster_map.get(client_idx, 0)
        if cluster_idx < len(self.cluster_proximal_params):
            return copy.deepcopy(self.cluster_proximal_params[cluster_idx])
        return self.get_global_params()

    def aggregate(self, client_params_list: List[OrderedDict],
                  client_weights: List[float],
                  exclude_param_names: Optional[List[str]] = None):
        """
        聚类后分组聚合: 每个簇内独立 FedAvg/FedProx

        Args:
            exclude_param_names: 不参与聚合的参数名
        """
        # 聚类
        clusters = cluster_clients(
            client_params_list, self.n_clusters,
            load_features=self.load_features,
            min_cluster_size=self.min_cluster_size
        )

        # 更新客户端到簇的映射
        self.client_cluster_map = {}
        for cluster_idx, members in enumerate(clusters):
            for member_idx in members:
                self.client_cluster_map[member_idx] = cluster_idx

        exclude_set = set(exclude_param_names) if exclude_param_names else set()

        # 为每个簇独立聚合
        self.cluster_models = []
        self.cluster_proximal_params = []
        for cluster_idx, members in enumerate(clusters):
            if len(members) == 0:
                default_params = self.get_global_params()
                self.cluster_models.append(default_params)
                self.cluster_proximal_params.append(copy.deepcopy(default_params))
                continue

            cluster_params = [client_params_list[i] for i in members]
            cluster_w = [client_weights[i] for i in members]
            total_w = sum(cluster_w)

            new_params = OrderedDict()
            for key in cluster_params[0]:
                if key in exclude_set:
                    new_params[key] = self.global_model.state_dict()[key].clone()
                else:
                    new_params[key] = sum(
                        p[key].float() * (w / total_w)
                        for p, w in zip(cluster_params, cluster_w)
                    )
            self.cluster_models.append(new_params)
            # 保存簇参数作为 FedProx 近端参考
            self.cluster_proximal_params.append(copy.deepcopy(new_params))

        # 全局模型使用所有客户端的加权平均 (用于初始化新客户端)
        total_weight = sum(client_weights)
        global_params = OrderedDict()
        for key in client_params_list[0]:
            if key in exclude_set:
                global_params[key] = self.global_model.state_dict()[key].clone()
            else:
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
