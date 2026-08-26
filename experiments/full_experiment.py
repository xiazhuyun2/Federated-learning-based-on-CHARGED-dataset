"""
完整论文实验系统 — 多城市分层联邦学习完整实验矩阵

实验矩阵:
  A. 时序基线 (每城市)
     - Seasonal Naive (24h) — 一天前同时刻
     - Seasonal Naive (168h) — 一周前同时刻

  B. 非联邦基线 (每城市)
     - Local-only TCN-LSTM (孤岛训练)
     - Centralized TCN-LSTM (集中式训练, 理论上限)

  C. 单城市联邦 (每城市各自)
     - FedProx (标准单城市FL)

  D. 多城市联邦
     - FedAvg (标准样本加权, α=1.0)
     - FedProx (处理统计异质性)
     - City-Balanced FedAvg (α=0.5, 防止大城主导)
     - City-Balanced FedProx (α=0.5 + 近端项)
     - City-Balanced + FedBN + LocalHead (推荐完整方法)

  E. 跨城市聚类联邦
     - Clustered FL (基于负荷特征聚类)

  F. 消融实验
     - LSTM-only
     - TCN-only
     - FedBN only vs LocalHead only vs Both

  G. 泛化测试
     - 留一城市冷启动 (另见 leave_one_out.py)

用法:
  # 完整6城市实验
  python experiments/full_experiment.py --mode all

  # 仅单城市实验
  python experiments/full_experiment.py --mode single_city --city SZH

  # 仅多城市实验
  python experiments/full_experiment.py --mode multi_city --cities SZH,AMS,JHB,LOA,MEL,SPO

  # 快速验证
  python experiments/full_experiment.py --mode quick --rounds 10 --top_k 5

  # 消融实验
  python experiments/full_experiment.py --mode ablation --city SZH
"""
import sys
import os
import json
import argparse
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALL_CITIES = ["SZH", "AMS", "JHB", "LOA", "MEL", "SPO"]
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cmd(name: str, cmd: str) -> int:
    """运行单个实验命令"""
    print(f"\n{'#'*70}")
    print(f"# {name}")
    print(f"# {cmd[:120]}{'...' if len(cmd) > 120 else ''}")
    print(f"{'#'*70}\n")
    ret = os.system(cmd)
    if ret != 0:
        print(f"  ⚠ WARNING: {name} returned exit code {ret}")
    return ret


def collect_organized_results(base_dir: str) -> Dict:
    """收集全部实验结果, 复用 organize_results 的逻辑。

    organize_results 读取每个 run 的 config.json 重新归类为逻辑实验名 (含 α、
    FedBN/LocalHead、station_selection、单/多城市标记), 并按「轮次优先 + 时间戳」
    挑最优 run; 这消除了旧 find_latest_* 依赖目录名拼接 (MULTI/multi_city 前缀、
    run_dirs[-1] 最新覆盖) 带来的错位。
    """
    from experiments.organize_results import collect_runs, summarize
    return summarize(collect_runs(base_dir))


def print_summary_table(results: Dict, title: str = "EXPERIMENT SUMMARY"):
    """打印实验结果汇总表"""
    metric_keys = ["RMSE", "MAE", "WAPE", "SMAPE"]
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*100}")

    header = f"  {'Method':<35s}"
    for mk in metric_keys:
        header += f" {mk:>18s}"
    print(header)
    print(f"  {'-'*35}{'  ' + '-'*18 + '  ' + '-'*18 + '  ' + '-'*18 + '  ' + '-'*18}")

    for method, metrics in sorted(results.items()):
        line = f"  {method:<35s}"
        for mk in metric_keys:
            if mk in metrics:
                m, s = metrics[mk]["mean"], metrics[mk]["std"]
                line += f" {m:>10.2f}±{s:<5.2f}"
            else:
                line += f" {'N/A':>18s}"
        print(line)

    print(f"{'='*100}")


# ═══════════════════════════════════════════════════════════════
# 实验定义
# ═══════════════════════════════════════════════════════════════

def build_single_city_experiments(city: str, top_k: int, rounds: int,
                                    local_epochs: int, epochs: int,
                                    base_dir: str) -> Dict[str, str]:
    """构建单城市实验命令字典"""
    # main.py 用 main_opts，基线脚本有自己的参数
    main_opts = f"--city {city} --top_k {top_k} --output_dir {base_dir}"
    # 基线脚本不认 --output_dir 和 --device，只用城市和 top_k
    base_opts = f"--city {city} --top_k {top_k}"
    fl_common = f"--num_rounds {rounds} --local_epochs {local_epochs}"

    return {
        # A. 时序基线 (这个脚本只支持 168h 季节性推理)
        f"{city}/seasonal_naive":
            f"python experiments/baseline_seasonal_naive.py {base_opts}",

        # B. 非联邦基线
        f"{city}/local_only":
            f"python experiments/baseline_local.py {base_opts} --epochs {epochs}",
        f"{city}/centralized":
            f"python experiments/baseline_centralized.py {base_opts} --epochs {epochs}",

        # C. 单城市FL (main.py)
        f"{city}/fedprox":
            f"python main.py {main_opts} --aggregation fedprox "
            f"{fl_common} --mu 0.01",
        f"{city}/fedprox_fedbn_lh":
            f"python main.py {main_opts} --aggregation fedprox "
            f"{fl_common} "
            f"--mu 0.01 --fedbn --local_head",
    }


def build_multi_city_experiments(cities: str, top_k: int, rounds: int,
                                   local_epochs: int, base_dir: str) -> Dict[str, str]:
    """构建多城市实验命令字典"""
    opts = f"--cities {cities} --top_k {top_k} --output_dir {base_dir}"
    common = f"--num_rounds {rounds} --local_epochs {local_epochs}"

    return {
        # D. 多城市联邦
        "multi/fedavg_a1":
            f"python main.py {opts} --aggregation fedavg "
            f"{common} --city_weight_alpha 1.0",
        "multi/fedprox_a1":
            f"python main.py {opts} --aggregation fedprox "
            f"{common} --mu 0.01 --city_weight_alpha 1.0",
        "multi/fedavg_balanced":
            f"python main.py {opts} --aggregation fedavg "
            f"{common} --city_weight_alpha 0.5",
        "multi/fedprox_balanced":
            f"python main.py {opts} --aggregation fedprox "
            f"{common} --mu 0.01 --city_weight_alpha 0.5",
        "multi/full_method":
            f"python main.py {opts} --aggregation fedprox "
            f"{common} --mu 0.01 --city_weight_alpha 0.5 "
            f"--fedbn --local_head",

        # E. 聚类FL
        "multi/clustered":
            f"python main.py {opts} --aggregation clustered "
            f"{common} --mu 0.01 --n_clusters 3 --city_weight_alpha 0.5",
        "multi/clustered_fedbn":
            f"python main.py {opts} --aggregation clustered "
            f"{common} --mu 0.01 --n_clusters 3 --city_weight_alpha 0.5 --fedbn",

        # 消融: 城市平衡的核心对比
        "multi/fedavg_a0":
            f"python main.py {opts} --aggregation fedavg "
            f"{common} --city_weight_alpha 0.0",
        "multi/fedavg_a1_szh_dominate":
            f"python main.py {opts} --aggregation fedavg "
            f"{common} --city_weight_alpha 1.0 --station_selection top_k",
    }


def build_ablation_experiments(city: str, top_k: int, rounds: int,
                                 local_epochs: int, base_dir: str) -> Dict[str, str]:
    """构建消融实验命令字典"""
    opts = f"--city {city} --top_k {top_k} --output_dir {base_dir}"
    common = f"--num_rounds {rounds} --local_epochs {local_epochs}"

    return {
        f"{city}/fedprox_baseline":
            f"python main.py {opts} --aggregation fedprox {common} --mu 0.01",
        f"{city}/fedprox_fedbn":
            f"python main.py {opts} --aggregation fedprox {common} --mu 0.01 --fedbn",
        f"{city}/fedprox_localhead":
            f"python main.py {opts} --aggregation fedprox {common} --mu 0.01 --local_head",
        f"{city}/fedprox_both":
            f"python main.py {opts} --aggregation fedprox {common} --mu 0.01 --fedbn --local_head",
    }


def build_multi_city_ablation(cities: str, top_k: int, rounds: int,
                                local_epochs: int, base_dir: str) -> Dict[str, str]:
    """构建多城市消融实验 — FedBN/LocalHead/Both (消融实验注意事项.txt 要求)"""
    opts = f"--cities {cities} --top_k {top_k} --output_dir {base_dir}"
    common = f"--num_rounds {rounds} --local_epochs {local_epochs} --city_weight_alpha 0.5 --mu 0.01"

    return {
        "multi_ablation/neither":
            f"python main.py {opts} --aggregation fedprox {common}",
        "multi_ablation/fedbn":
            f"python main.py {opts} --aggregation fedprox {common} --fedbn",
        "multi_ablation/localhead":
            f"python main.py {opts} --aggregation fedprox {common} --local_head",
        "multi_ablation/both":
            f"python main.py {opts} --aggregation fedprox {common} --fedbn --local_head",
    }


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="完整论文实验系统 — 多城市分层联邦学习")

    # 模式
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["single_city", "multi_city", "all",
                                 "quick", "ablation", "transfer"],
                        help="实验模式")

    # 城市
    parser.add_argument("--city", type=str, default="SZH",
                        help="单城市模式下的城市")
    parser.add_argument("--cities", type=str, default="SZH,AMS,JHB,LOA,MEL,SPO",
                        help="多城市模式下的城市列表")

    # 训练参数
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100,
                        help="非FL方法的epoch数")
    parser.add_argument("--seeds", type=str, default="42,123,999")

    # 输出
    parser.add_argument("--output_dir", type=str, default=None)
    # 快速模式 (可用于任何 mode)
    parser.add_argument("--quick", action="store_true",
                        help="快速验证: top_k=5, rounds=10, epochs=20, 2 seeds")

    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    base_dir = args.output_dir or os.path.join(PROJECT_ROOT, "outputs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 快速模式参数 (--quick 或 --mode quick 均触发)
    if args.quick or args.mode == "quick":
        args.top_k = min(args.top_k, 5)
        args.rounds = min(args.rounds, 10)
        args.local_epochs = min(args.local_epochs, 2)
        args.epochs = min(args.epochs, 20)
        seeds = seeds[:2]  # 仅2个种子
        print(f"  Quick mode: top_k={args.top_k}, rounds={args.rounds}, "
              f"epochs={args.epochs}, seeds={seeds}")

    # --- 单城市实验 ---
    if args.mode in ("single_city", "all", "quick"):
        city = args.city
        print(f"\n{'='*70}")
        print(f"  SINGLE-CITY EXPERIMENTS: {city}")
        print(f"{'='*70}")

        experiments = build_single_city_experiments(
            city, args.top_k, args.rounds,
            args.local_epochs, args.epochs, base_dir)

        all_results = {}
        for method, cmd in experiments.items():
            for seed in seeds:
                seed_cmd = cmd + f" --seed {seed}"
                run_cmd(f"{method} (seed={seed})", seed_cmd)

        # 收集结果 (按 config.json 重新归类 + 轮次优先, 而非目录名拼接)
        all_results = collect_organized_results(base_dir)

        print_summary_table(all_results, f"Single-City Results — {city}")

        # 保存
        summary_path = os.path.join(
            base_dir, "summaries",
            f"single_city_{city}_{timestamp}.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({"city": city, "seeds": seeds,
                       "timestamp": timestamp, "results": all_results},
                      f, indent=2, default=str)
        print(f"  Summary saved: {summary_path}")

    # --- 多城市实验 ---
    if args.mode in ("multi_city", "all", "quick"):
        cities_str = args.cities
        print(f"\n{'='*70}")
        print(f"  MULTI-CITY EXPERIMENTS: {cities_str}")
        print(f"{'='*70}")

        experiments = build_multi_city_experiments(
            cities_str, args.top_k, args.rounds,
            args.local_epochs, base_dir)

        all_results = {}
        for method, cmd in experiments.items():
            # 对每个seed单独运行
            for seed in seeds:
                seed_cmd = cmd + f" --seed {seed}"
                run_cmd(f"{method} (seed={seed})", seed_cmd)

        # 收集多城市指标 (按 config.json 重新归类 + 轮次优先)
        all_results = collect_organized_results(base_dir)

        print_summary_table(all_results, f"Multi-City Results — {cities_str}")

        # 保存
        summary_path = os.path.join(
            base_dir, "summaries",
            f"multi_city_{timestamp}.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({"cities": cities_str.split(","), "seeds": seeds,
                       "timestamp": timestamp, "results": all_results},
                      f, indent=2, default=str)
        print(f"  Summary saved: {summary_path}")

    # --- 消融实验 ---
    if args.mode == "ablation":
        city = args.city
        cities_str = args.cities

        # 2a. 单城市消融 (FedBN/LocalHead/Both)
        print(f"\n{'='*70}")
        print(f"  ABLATION 2a: Single-City ({city}) — FedBN / LocalHead / Both")
        print(f"{'='*70}")

        experiments = build_ablation_experiments(
            city, args.top_k, args.rounds,
            args.local_epochs, base_dir)

        all_results = {}
        for method, cmd in experiments.items():
            for seed in seeds:
                seed_cmd = cmd + f" --seed {seed}"
                run_cmd(f"{method} (seed={seed})", seed_cmd)

        all_results = collect_organized_results(base_dir)

        print_summary_table(all_results, f"Ablation 2a — Single-City: {city}")

        summary_path = os.path.join(
            base_dir, "summaries",
            f"ablation_single_{city}_{timestamp}.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({"city": city, "seeds": seeds,
                       "timestamp": timestamp, "results": all_results},
                      f, indent=2, default=str)
        print(f"  Summary saved: {summary_path}")

        # 2b. 多城市消融 (FedBN/LocalHead/Both, α=0.5)
        print(f"\n{'='*70}")
        print(f"  ABLATION 2b: Multi-City ({cities_str}) — FedBN / LocalHead / Both")
        print(f"  (α=0.5, 消融实验注意事项.txt: 需单城市+多城市各做一遍)")
        print(f"{'='*70}")

        multi_experiments = build_multi_city_ablation(
            cities_str, args.top_k, args.rounds,
            args.local_epochs, base_dir)

        multi_results = {}
        for method, cmd in multi_experiments.items():
            for seed in seeds:
                seed_cmd = cmd + f" --seed {seed}"
                run_cmd(f"{method} (seed={seed})", seed_cmd)

        multi_results = collect_organized_results(base_dir)

        print_summary_table(multi_results,
                           f"Ablation 2b — Multi-City: {cities_str}")

        summary_path = os.path.join(
            base_dir, "summaries",
            f"ablation_multi_{timestamp}.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({"cities": cities_str.split(","), "seeds": seeds,
                       "timestamp": timestamp, "results": multi_results},
                      f, indent=2, default=str)
        print(f"  Summary saved: {summary_path}")

        # 汇总单城市 + 多城市消融
        print(f"\n{'='*70}")
        print(f"  ABLATION COMPLETE — Both single-city and multi-city done.")
        print(f"  Paper tables: α=0/0.5/1 from multi_city mode; "
              f"FedBN/LocalHead from ablation mode.")
        print(f"{'='*70}")

    print(f"\n{'='*70}")
    print(f"  All experiments complete!")
    print(f"  Output: {base_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
