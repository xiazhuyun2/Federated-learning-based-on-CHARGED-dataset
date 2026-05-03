"""
全局配置文件 — 联邦学习充电站负荷预测
"""
import os
from dataclasses import dataclass, field
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class DataConfig:
    """数据与特征工程配置"""
    cities: List[str] = field(default_factory=lambda: [
        "SZH", "AMS", "JHB", "LOA", "MEL", "SPO"
    ])
    use_remove_zero: bool = True          # 使用去零站点版本
    time_col: str = "Unnamed: 0"          # volume.csv 时间列名
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seq_len: int = 48                    # 输入窗口: 7天 * 24h
    pred_len: int = 24                    # 预测窗口: 未来24h
    top_k_stations: int = 20              # 每城市选取负荷最大的 k 个站点作为客户端
    vmd_K: int = 6                        # VMD 分解模态数
    vmd_alpha: int = 2000                 # VMD 惩罚因子


@dataclass
class ModelConfig:
    """TCN-LSTM 模型配置"""
    tcn_channels: List[int] = field(default_factory=lambda: [64, 64, 64])
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.2
    lstm_hidden: int = 64
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    fc_hidden: int = 64
    input_dim: int = 1                    # 将在运行时根据特征数更新


@dataclass
class FedConfig:
    """联邦学习配置"""
    num_rounds: int = 50                  # 全局通信轮次
    local_epochs: int = 5                 # 本地训练轮次
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    aggregation: str = "fedprox"          # fedavg / fedprox / clustered
    fedprox_mu: float = 0.01             # FedProx 近端项系数
    n_clusters: int = 3                   # 聚类联邦的簇数
    min_clients_per_round: int = 5        # 每轮最少参与客户端


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fed: FedConfig = field(default_factory=FedConfig)
    seed: int = 42
    device: str = "cuda"                  # cuda / cpu
    output_dir: str = OUTPUT_DIR
