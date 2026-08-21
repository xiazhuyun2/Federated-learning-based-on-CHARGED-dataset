"""
全局配置文件 — 联邦学习充电站负荷预测
"""
import os
from dataclasses import dataclass, field
from typing import List
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_git_commit() -> str:
    """获取当前 Git commit 短哈希"""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def get_run_dir(city: str, method: str, seed: int,
                base_dir: str = None) -> str:
    """
    生成唯一实验输出目录:
      outputs/{city}/{method}/seed_{seed}/{timestamp}_{git_commit}/
    """
    base = base_dir or OUTPUT_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git = get_git_commit()
    run_dir = os.path.join(base, city, method, f"seed_{seed}",
                           f"run_{timestamp}_{git}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


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
    seq_len: int = 168                    # 输入窗口: 7天 * 24h
    pred_len: int = 24                    # 预测窗口: 未来24h
    top_k_stations: int = 20              # 每城市选取负荷最大的 k 个站点作为客户端
    vmd_K: int = 6                        # VMD 分解模态数 (当前未被训练链路使用)
    vmd_alpha: int = 2000                 # VMD 惩罚因子
    # 增强特征开关 (P1)
    use_lag_features: bool = True
    use_rolling_features: bool = True
    use_static_features: bool = True
    # 多城市预处理 (P2)
    station_selection: str = "top_k"       # "top_k" | "stratified_natural" | "stratified_balanced" | "proportional"
    min_city_clients: int = 2              # proportional 分配时每城保底客户端数
    price_normalization: bool = True       # 电价城市内标准化
    load_normalization: bool = True        # 添加 per-charger 和 load_rate 特征
    timezone_offsets: dict = field(default_factory=lambda: {
        "SZH": 8, "AMS": 2, "JHB": 2,
        "LOA": -7, "MEL": 10, "SPO": -3,
    })


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
    # FedBN (P1)
    use_fedbn: bool = False               # 是否启用 FedBN (BN 参数不参与聚合)
    use_local_head: bool = False          # 本地预测头不参与聚合


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
    # 聚类方法 (P1)
    cluster_method: str = "load_profile"  # "load_profile" / "model_params"
    min_cluster_size: int = 2             # 最小簇大小, 防止无效聚类
    # 本地微调 (P1)
    finetune_epochs: int = 0              # 全局训练后本地微调轮次, 0=禁用
    # 多城市平衡聚合 (P2)
    city_weight_alpha: float = 0.5         # 0=等权, 1=样本量加权, 0.5=折中
    multi_city_mode: str = "single"        # "single" | "multi_city"
    client_fraction: float = 1.0           # 每轮参与聚合的客户端比例 (0,1]; 1.0=全部参与


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fed: FedConfig = field(default_factory=FedConfig)
    seed: int = 42
    device: str = "cuda"                  # cuda / cpu
    output_dir: str = OUTPUT_DIR
    experiment_name: str = ""             # 实验名称, 为空则自动生成
    git_commit: str = field(default_factory=get_git_commit)
