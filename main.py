"""
主入口 — 基于联邦学习的电动车充电站负荷预测

使用方式:
  python main.py                         # 默认: SZH城市, FedProx策略
  python main.py --city AMS              # 指定城市
  python main.py --aggregation fedavg    # FedAvg策略
  python main.py --aggregation clustered # 聚类联邦学习
  python main.py --num_rounds 30 --top_k 10  # 调参
"""
import argparse
import sys
import os

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.federated.trainer import FederatedTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="FL-EV: Federated Learning for EV Charging Load Prediction"
    )
    parser.add_argument("--city", type=str, default="SZH",
                        choices=["SZH", "AMS", "JHB", "LOA", "MEL", "SPO"],
                        help="City to use (default: SZH)")
    parser.add_argument("--aggregation", type=str, default="fedprox",
                        choices=["fedavg", "fedprox", "clustered"],
                        help="Federated aggregation strategy")
    parser.add_argument("--num_rounds", type=int, default=50,
                        help="Number of FL communication rounds")
    parser.add_argument("--local_epochs", type=int, default=5,
                        help="Local training epochs per round")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-k stations as clients")
    parser.add_argument("--seq_len", type=int, default=168,
                        help="Input sequence length (hours)")
    parser.add_argument("--pred_len", type=int, default=24,
                        help="Prediction horizon (hours)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mu", type=float, default=0.01,
                        help="FedProx proximal term coefficient")
    parser.add_argument("--n_clusters", type=int, default=3,
                        help="Number of clusters for Clustered FL")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device: auto (detect GPU), cuda, cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # 构建配置
    cfg = Config()
    cfg.seed = args.seed
    cfg.data.top_k_stations = args.top_k
    cfg.data.seq_len = args.seq_len
    cfg.data.pred_len = args.pred_len
    cfg.fed.num_rounds = args.num_rounds
    cfg.fed.local_epochs = args.local_epochs
    cfg.fed.batch_size = args.batch_size
    cfg.fed.lr = args.lr
    cfg.fed.aggregation = args.aggregation
    cfg.fed.fedprox_mu = args.mu
    cfg.fed.n_clusters = args.n_clusters

    # 设备选择
    import torch
    if args.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        cfg.device = args.device

    print("=" * 60)
    print("  FL-EV: Federated Learning for EV Charging Load Prediction")
    print("=" * 60)
    print(f"  City:         {args.city}")
    print(f"  Aggregation:  {args.aggregation}")
    print(f"  Rounds:       {args.num_rounds}")
    print(f"  Local Epochs: {args.local_epochs}")
    print(f"  Top-K:        {args.top_k}")
    print(f"  Seq/Pred:     {args.seq_len} -> {args.pred_len}")
    if cfg.device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Device:       {cfg.device} ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        print(f"  Device:       {cfg.device}")
    print("=" * 60)

    # 训练
    trainer = FederatedTrainer(cfg)
    trainer.prepare_city_clients(args.city)
    results = trainer.run_federated_training()

    print("\n  Done!")


if __name__ == "__main__":
    main()
