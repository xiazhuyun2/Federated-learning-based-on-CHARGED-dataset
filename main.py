"""
主入口 — 基于联邦学习的电动车充电站负荷预测

使用方式:
  python main.py                         # 默认: SZH城市, FedProx策略
  python main.py --city AMS              # 指定城市
  python main.py --aggregation fedavg    # FedAvg策略
  python main.py --aggregation clustered # 聚类联邦学习
  python main.py --aggregation fedprox --fedbn  # FedProx + FedBN
  python main.py --aggregation fedprox --local_head  # 本地预测头
  python main.py --num_rounds 30 --top_k 10  # 调参
  python main.py --seeds 42,123,999      # 多种子运行
"""
import argparse
import sys
import os

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_run_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="FL-EV: Federated Learning for EV Charging Load Prediction"
    )
    # 基本参数
    parser.add_argument("--city", type=str, default="SZH",
                        choices=["SZH", "AMS", "JHB", "LOA", "MEL", "SPO"],
                        help="City to use (default: SZH)")
    parser.add_argument("--aggregation", type=str, default="fedprox",
                        choices=["fedavg", "fedprox", "clustered"],
                        help="Federated aggregation strategy")

    # 联邦学习参数
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

    # 聚类参数
    parser.add_argument("--n_clusters", type=int, default=3,
                        help="Number of clusters for Clustered FL")
    parser.add_argument("--cluster_method", type=str, default="load_profile",
                        choices=["load_profile", "model_params"],
                        help="Clustering method")

    # FedBN / 本地预测头
    parser.add_argument("--fedbn", action="store_true",
                        help="Enable FedBN (BN params stay local)")
    parser.add_argument("--local_head", action="store_true",
                        help="Enable local prediction head (not aggregated)")
    parser.add_argument("--finetune_epochs", type=int, default=0,
                        help="Local fine-tuning epochs after global training")

    # 运行参数
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device: auto (detect GPU), cuda, cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-seed runs (e.g. 42,123,999)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")

    return parser.parse_args()


def run_single(cfg: Config, args, seed: int):
    """单次运行"""
    from src.federated.trainer import FederatedTrainer
    import torch

    cfg.seed = seed
    if args.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        cfg.device = args.device

    # 生成唯一输出目录
    method = args.aggregation
    if args.fedbn:
        method += "_fedbn"
    if args.local_head:
        method += "_localhead"
    run_dir = get_run_dir(args.city, method, seed,
                          base_dir=args.output_dir)

    print("=" * 60)
    print("  FL-EV: Federated Learning for EV Charging Load Prediction")
    print("=" * 60)
    print(f"  City:         {args.city}")
    print(f"  Aggregation:  {args.aggregation}")
    print(f"  Method tag:   {method}")
    print(f"  Rounds:       {args.num_rounds}")
    print(f"  Local Epochs: {args.local_epochs}")
    print(f"  Top-K:        {args.top_k}")
    print(f"  Seq/Pred:     {args.seq_len}h -> {args.pred_len}h")
    print(f"  FedBN:        {args.fedbn}")
    print(f"  Local Head:   {args.local_head}")
    print(f"  Seed:         {seed}")
    print(f"  Output:       {run_dir}")
    if cfg.device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Device:       {cfg.device} ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        print(f"  Device:       {cfg.device}")
    print("=" * 60)

    # 训练
    trainer = FederatedTrainer(cfg, run_dir=run_dir,
                               city=args.city, method=method)
    trainer.prepare_city_clients(args.city)
    results = trainer.run_federated_training()

    print("\n  Done!")
    return results


def main():
    args = parse_args()

    # 构建配置
    cfg = Config()
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
    cfg.fed.cluster_method = args.cluster_method
    cfg.fed.finetune_epochs = args.finetune_epochs
    cfg.model.use_fedbn = args.fedbn
    cfg.model.use_local_head = args.local_head

    if args.output_dir:
        cfg.output_dir = args.output_dir

    # 多种子运行
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
        all_results = {}
        for seed in seeds:
            print(f"\n{'#'*60}")
            print(f"#  Running with seed={seed}")
            print(f"{'#'*60}")
            results = run_single(cfg, args, seed)
            all_results[f"seed_{seed}"] = results.get("AVERAGE", {})

        # 汇总多种子结果
        print(f"\n{'='*60}")
        print(f"  Multi-Seed Summary ({len(seeds)} seeds)")
        print(f"{'='*60}")
        import numpy as np
        for metric in ["RMSE", "MAE", "WAPE", "SMAPE"]:
            vals = [r[metric] for r in all_results.values() if metric in r]
            if vals:
                print(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        print("=" * 60)
    else:
        run_single(cfg, args, args.seed)


if __name__ == "__main__":
    main()
