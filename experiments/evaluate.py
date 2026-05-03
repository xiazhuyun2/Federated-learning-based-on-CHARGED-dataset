"""
独立评估脚本 — 加载已保存的实验结果, 生成对比可视化

用于: 在所有实验跑完后, 统一生成方法对比图 (雷达图)
运行: python experiments/evaluate.py
"""
import sys
import os
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.visualization import plot_method_comparison


def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "outputs")

    print("=" * 60)
    print("  Post-hoc Evaluation & Method Comparison")
    print("=" * 60)

    # 扫描 outputs/ 下所有结果文件
    result_files = glob.glob(os.path.join(output_dir, "*results*.json"))
    result_files += glob.glob(os.path.join(output_dir, "baseline*.json"))

    print(f"\n  Found {len(result_files)} result files:")
    for f in result_files:
        print(f"    - {os.path.basename(f)}")

    # 加载所有结果的 AVERAGE
    method_results = {}

    for fpath in result_files:
        with open(fpath) as f:
            data = json.load(f)

        name = os.path.basename(fpath).replace(".json", "")

        # 提取 AVERAGE (如果有)
        if "AVERAGE" in data:
            avg = data["AVERAGE"]
            method_results[name] = {
                "RMSE": float(avg["RMSE"]),
                "MAE": float(avg["MAE"]),
                "MAPE": float(avg["MAPE"]),
            }
            print(f"\n  {name}:")
            print(f"    RMSE = {avg['RMSE']:.4f}")
            print(f"    MAE  = {avg['MAE']:.4f}")
            print(f"    MAPE = {avg['MAPE']:.2f}%")

    if len(method_results) >= 2:
        print("\n  Generating radar comparison plot...")
        path = plot_method_comparison(method_results, output_dir)
        print(f"  Saved: {path}")
    else:
        print("\n  Need at least 2 methods to generate comparison plot.")
        print("  Run different experiments first:")
        print("    python experiments/baseline_local.py --city SZH")
        print("    python main.py --aggregation fedavg --city SZH")
        print("    python main.py --aggregation fedprox --city SZH")
        print("    python main.py --aggregation clustered --city SZH")

    # 打印对比表格
    if method_results:
        print("\n" + "=" * 60)
        print(f"  {'Method':<25s} {'RMSE':>10s} {'MAE':>10s} {'MAPE':>10s}")
        print("-" * 60)
        for name, m in sorted(method_results.items(),
                               key=lambda x: x[1]["RMSE"]):
            print(f"  {name:<25s} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} "
                  f"{m['MAPE']:>9.2f}%")
        print("=" * 60)


if __name__ == "__main__":
    main()
