"""
独立评估脚本 — 加载已保存的实验结果, 生成对比可视化

扫描 outputs/{city}/{method}/seed_*/run_*/ 目录结构,
加载各方法的 metrics.json, 生成方法对比图表。

运行: python experiments/evaluate.py
"""
import sys
import os
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.visualization import plot_method_comparison


def find_run_dirs(output_dir: str) -> dict:
    """
    扫描新的目录结构: outputs/{city}/{method}/seed_{seed}/run_*/metrics.json
    返回 {method_label: metrics_dict}
    """
    method_results = {}

    # 遍历 outputs/ 下的所有 city/method/seed/run 目录
    for metrics_path in glob.glob(os.path.join(
            output_dir, "*", "*", "seed_*", "run_*", "metrics.json")):
        try:
            with open(metrics_path) as f:
                data = json.load(f)

            # 从路径中提取 method 名
            parts = metrics_path.replace("\\", "/").split("/")
            # parts: [..., city, method, seed_X, run_Y, metrics.json]
            city = parts[-5]
            method = parts[-4]
            label = f"{city}/{method}"

            if "AVERAGE" in data:
                avg = data["AVERAGE"]
                method_results[label] = {
                    k: float(v) for k, v in avg.items()
                    if k != "MASE" and v != float("inf")
                }
                print(f"\n  {label}:")
                for k, v in method_results[label].items():
                    print(f"    {k} = {v:.4f}")
        except Exception as e:
            print(f"  Skipping {metrics_path}: {e}")
            continue

    return method_results


def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "outputs")

    print("=" * 60)
    print("  Post-hoc Evaluation & Method Comparison")
    print("=" * 60)

    method_results = find_run_dirs(output_dir)

    if not method_results:
        print("\n  No results found in the new directory structure.")
        print("  Run experiments first:")
        print("    python main.py --aggregation fedavg --city SZH")
        print("    python main.py --aggregation fedprox --city SZH")
        print("    python main.py --aggregation clustered --city SZH")
        return

    if len(method_results) >= 2:
        print("\n  Generating radar comparison plot...")
        path = plot_method_comparison(method_results, output_dir)
        print(f"  Saved: {path}")
    else:
        print("\n  Need at least 2 methods to generate comparison plot.")

    # 打印对比表格
    if method_results:
        # 确定要展示的指标
        first_result = next(iter(method_results.values()))
        display_metrics = [m for m in ["RMSE", "MAE", "WAPE", "SMAPE"]
                          if m in first_result]

        print("\n" + "=" * 80)
        header = f"  {'Method':<35s}"
        for m in display_metrics:
            header += f" {m:>12s}"
        print(header)
        print("-" * 80)

        for name, m in sorted(method_results.items(),
                               key=lambda x: x[1].get("RMSE", float("inf"))):
            line = f"  {name:<35s}"
            for metric in display_metrics:
                if metric in m:
                    line += f" {m[metric]:>12.4f}"
                else:
                    line += f" {'N/A':>12s}"
            print(line)
        print("=" * 80)


if __name__ == "__main__":
    main()
