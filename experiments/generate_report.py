"""
自动报告生成 — 从实验结果生成论文所需的 LaTeX 表格和文本

读取 outputs/ 下所有实验结果，生成:
  - outputs/reports/main_results.tex     — 主结果表 (含 mean ± std)
  - outputs/reports/ablation.tex         — 消融实验表
  - outputs/reports/transfer_table.tex   — 跨城市迁移矩阵表
  - outputs/reports/city_details.tex     — 各城市详细指标表
  - outputs/reports/paper_numbers.txt    — 论文中可直接引用的数值

用法:
  python experiments/generate_report.py
  python experiments/generate_report.py --output_dir outputs/my_runs
"""
import sys
import os
import json
import argparse
import numpy as np
import glob as glob_mod
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_all_metrics_files(base_dir: str) -> Dict[str, List[str]]:
    """扫描实验输出目录, 按方法分组查找 metrics.json"""
    method_files = defaultdict(list)
    pattern = os.path.join(base_dir, "**", "metrics.json")
    for path in glob_mod.glob(pattern, recursive=True):
        # 从路径提取方法名
        rel = os.path.relpath(path, base_dir)
        parts = rel.replace("\\", "/").split("/")
        # 跳过 summaries 目录
        if "summaries" in parts or "reports" in parts:
            continue
        # 尝试提取方法名: .../city/method/seed_X/run_Y/metrics.json
        # 或 .../multi_city/MULTI/method/seed_X/run_Y/metrics.json
        method = "unknown"
        for i, p in enumerate(parts):
            if p.startswith("seed_") and i > 0:
                method = parts[i - 1]
                break
        if method == "unknown":
            # fallback: 使用目录名
            candidates = [p for p in parts if not p.startswith("seed_")
                          and not p.startswith("run_") and p != "metrics.json"]
            if candidates:
                method = candidates[-1]

        method_files[method].append(path)

    return dict(method_files)


def aggregate_method_metrics(metrics_paths: List[str]) -> Dict:
    """聚合同一方法多个种子/运行的指标"""
    all_avg = []
    all_macro_city = []
    per_city_collect = defaultdict(list)

    for path in metrics_paths:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue

        avg = data.get("AVERAGE", {})
        if avg:
            all_avg.append(avg)

        macro = data.get("macro_city", {})
        if macro:
            all_macro_city.append(macro)

        per_city = data.get("per_city", {})
        for city, m in per_city.items():
            if isinstance(m, dict) and "RMSE" in m:
                per_city_collect[city].append(m)

    result = {}
    if all_avg:
        result["AVERAGE"] = _agg_dicts(all_avg)
    if all_macro_city:
        result["macro_city"] = _agg_dicts(all_macro_city)
    for city, metrics_list in per_city_collect.items():
        result[f"per_city/{city}"] = _agg_dicts(metrics_list)

    return result


def _agg_dicts(dicts: List[Dict]) -> Dict:
    """计算 mean ± std"""
    if not dicts:
        return {}
    result = {}
    for key in dicts[0]:
        vals = [d[key] for d in dicts if key in d and d[key] is not None
                and not (isinstance(d[key], float) and np.isinf(d[key]))]
        if vals:
            result[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
    return result


def format_mean_std(metrics_dict: Dict, key: str, digits: int = 2) -> str:
    """格式化 mean ± std"""
    entry = metrics_dict.get(key, {})
    if not entry:
        return "—"
    mean, std = entry["mean"], entry.get("std", 0)
    if digits == 2:
        return f"{mean:.2f} \\pm {std:.2f}"
    elif digits == 1:
        return f"{mean:.1f} \\pm {std:.1f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def extract_single_value(metrics_dict: Dict, key: str) -> float:
    entry = metrics_dict.get(key, {})
    if isinstance(entry, dict):
        return entry.get("mean", float("inf"))
    return entry if isinstance(entry, (int, float)) else float("inf")


# ═══════════════════════════════════════════════════════════════
# LaTeX 表格生成
# ═══════════════════════════════════════════════════════════════

def build_main_results_table(all_methods: Dict[str, Dict],
                             metric_keys: List[str] = None) -> str:
    """生成主结果 LaTeX 表格"""
    if metric_keys is None:
        metric_keys = ["RMSE", "MAE", "WAPE", "SMAPE"]

    # 按 macro_city.RMSE 排序 (优先多城市指标)
    def sort_key(item):
        name, metrics = item
        macro = metrics.get("macro_city", metrics.get("AVERAGE", {}))
        return extract_single_value(macro, "RMSE")

    sorted_methods = sorted(all_methods.items(), key=sort_key)

    n_metrics = len(metric_keys)
    col_def = "l" + "c" * n_metrics
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Main Results — Multi-City Federated Learning for EV Charging Load Prediction}")
    lines.append(r"  \label{tab:main_results}")
    lines.append(f"  \\begin{{tabular}}{{{col_def}}}")
    lines.append(r"    \toprule")
    header = "    Method"
    for mk in metric_keys:
        header += f" & {mk}"
    header += r" \\"
    lines.append(header)
    lines.append(r"    \midrule")

    for method, metrics in sorted_methods:
        display_name = _method_display_name(method)
        macro = metrics.get("macro_city", metrics.get("AVERAGE", {}))
        row = f"    {display_name}"
        for mk in metric_keys:
            row += f" & {format_mean_std(macro, mk)}"
        row += r" \\"
        lines.append(row)

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def build_ablation_table(all_methods: Dict[str, Dict]) -> str:
    """生成消融实验 LaTeX 表格"""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Ablation Study — Effect of Each Component}")
    lines.append(r"  \label{tab:ablation}")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Method & FedBN & LocalHead & RMSE & WAPE (\\%) \\")
    lines.append(r"    \midrule")

    for method, metrics in sorted(all_methods.items()):
        display_name = _method_display_name(method)
        has_fedbn = "✓" if "fedbn" in method.lower() else "—"
        has_lh = "✓" if "localhead" in method.lower() or "local_head" in method.lower() else "—"
        macro = metrics.get("macro_city", metrics.get("AVERAGE", {}))
        rmse = format_mean_std(macro, "RMSE")
        wape = format_mean_std(macro, "WAPE", digits=1)
        lines.append(f"    {display_name} & {has_fedbn} & {has_lh} & {rmse} & {wape} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_transfer_table(transfer_matrix_path: str) -> str:
    """从 transfer_matrix.json 生成 LaTeX 迁移矩阵表"""
    if not os.path.exists(transfer_matrix_path):
        return "% Transfer matrix data not found"

    with open(transfer_matrix_path) as f:
        data = json.load(f)

    cities = data.get("cities", data.get("cities_order", []))
    matrix = np.array(data.get("rmse_matrix", data.get("matrix", [])))
    if matrix.size == 0 or len(cities) == 0:
        return "% Transfer matrix data empty"

    n = len(cities)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Cross-City Transfer Matrix (RMSE, Train$\rightarrow$Test)}")
    lines.append(r"  \label{tab:transfer_matrix}")
    lines.append(f"  \\begin{{tabular}}{{{'l' + 'c' * n}}}")
    lines.append(r"    \toprule")
    header = "    Train \\textbackslash Test"
    for c in cities:
        header += f" & {c}"
    header += r" \\"
    lines.append(header)
    lines.append(r"    \midrule")

    for i, train_c in enumerate(cities):
        row = f"    {train_c}"
        for j in range(n):
            val = matrix[i, j]
            if np.isfinite(val) and val < 1e6:
                # 标记对角线
                if i == j:
                    row += f" & \\textbf{{{val:.1f}}}"
                else:
                    row += f" & {val:.1f}"
            else:
                row += " & —"
        row += r" \\"
        lines.append(row)

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_city_details_table(all_methods: Dict[str, Dict],
                              city: str) -> str:
    """生成某个城市在各方法下的详细指标表"""
    metric_keys = ["RMSE", "MAE", "WAPE", "SMAPE"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{Per-Method Results for {city}}}")
    lines.append(f"  \\label{{tab:city_{city.lower()}}}")
    col_def = "l" + "c" * len(metric_keys)
    lines.append(f"  \\begin{{tabular}}{{{col_def}}}")
    lines.append(r"    \toprule")
    header = "    Method"
    for mk in metric_keys:
        header += f" & {mk}"
    header += r" \\"
    lines.append(header)
    lines.append(r"    \midrule")

    for method, metrics in sorted(all_methods.items()):
        per_city_key = f"per_city/{city}"
        city_metrics = metrics.get(per_city_key, metrics.get("AVERAGE", {}))
        display_name = _method_display_name(method)
        row = f"    {display_name}"
        for mk in metric_keys:
            row += f" & {format_mean_std(city_metrics, mk)}"
        row += r" \\"
        lines.append(row)

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _method_display_name(method: str) -> str:
    """规范化方法显示名"""
    mapping = {
        "seasonal_naive24h": "Seasonal Naive (24h)",
        "seasonal_naive168h": "Seasonal Naive (168h)",
        "local_only": "Local-Only",
        "centralized": "Centralized",
        "fedavg": "FedAvg",
        "fedprox": "FedProx",
        "clustered": "Clustered FL",
        "clustered_fedbn": "Clustered + FedBN",
        "fedprox_fedbn_lh": "FedProx + FedBN + LocalHead",
        "fedprox_fedbn_localhead": "Full Method",
        "full_method": "Full Method (Balanced + FedBN + LH)",
        "fedprox_baseline": "FedProx (baseline)",
        "fedprox_fedbn": "FedProx + FedBN",
        "fedprox_localhead": "FedProx + LocalHead",
        "fedprox_both": "FedProx + FedBN + LocalHead",
        "fedavg_a1": "FedAvg ($\\alpha{=}1$)",
        "fedavg_a1_szh_dominate": "FedAvg ($\\alpha{=}1$, Top-K)",
        "fedavg_a0": "FedAvg ($\\alpha{=}0$, Equal)",
        "fedavg_balanced": "FedAvg Balanced ($\\alpha{=}0.5$)",
        "fedprox_a1": "FedProx ($\\alpha{=}1$)",
        "fedprox_balanced": "FedProx Balanced ($\\alpha{=}0.5$)",
    }
    # 移除城市前缀
    for city in ["SZH/", "AMS/", "JHB/", "LOA/", "MEL/", "SPO/",
                  "multi/", "MULTI/"]:
        if method.startswith(city):
            method = method[len(city):]
            break
    return mapping.get(method, method.replace("_", " "))


# ═══════════════════════════════════════════════════════════════
# 论文数值提取
# ═══════════════════════════════════════════════════════════════

def build_paper_numbers(all_methods: Dict[str, Dict]) -> str:
    """提取论文中可直接引用的数值"""
    lines = []
    lines.append("=" * 70)
    lines.append("  Paper-Ready Numbers")
    lines.append("  Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    lines.append("=" * 70)

    for method, metrics in sorted(all_methods.items()):
        display_name = _method_display_name(method)
        macro = metrics.get("macro_city", metrics.get("AVERAGE", {}))
        rmse = macro.get("RMSE", {})
        mae = macro.get("MAE", {})
        wape = macro.get("WAPE", {})

        if rmse:
            rmse_str = f"{rmse['mean']:.2f}±{rmse.get('std', 0):.2f}"
            lines.append(f"\n  {display_name}:")
            lines.append(f"    RMSE = {rmse_str}")
            if mae:
                lines.append(f"    MAE  = {mae['mean']:.2f}±{mae.get('std', 0):.2f}")
            if wape:
                lines.append(f"    WAPE = {wape['mean']:.1f}±{wape.get('std', 0):.1f}%")

        # 各城市详细指标
        for key, city_metrics in metrics.items():
            if key.startswith("per_city/"):
                city = key.split("/", 1)[1]
                c_rmse = city_metrics.get("RMSE", {})
                c_wape = city_metrics.get("WAPE", {})
                if c_rmse:
                    lines.append(f"      {city}: RMSE={c_rmse['mean']:.2f}, WAPE={c_wape.get('mean', 0):.1f}%")

    # 找出最佳方法
    best_method, best_rmse = None, float("inf")
    for method, metrics in all_methods.items():
        macro = metrics.get("macro_city", metrics.get("AVERAGE", {}))
        rmse = extract_single_value(macro, "RMSE")
        if rmse < best_rmse:
            best_rmse = rmse
            best_method = _method_display_name(method)

    lines.append(f"\n  Best method: {best_method} (RMSE = {best_rmse:.2f})")
    lines.append("=" * 70)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="自动报告生成 — 从实验结果生成 LaTeX 表格和论文数值")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="实验结果目录 (默认: outputs/)")
    parser.add_argument("--cities", type=str, default="SZH,AMS,JHB,LOA,MEL,SPO")
    args = parser.parse_args()

    base_dir = args.output_dir or os.path.join(PROJECT_ROOT, "outputs")
    report_dir = os.path.join(base_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)

    cities = [c.strip() for c in args.cities.split(",")]

    print(f"\nScanning experiment results in: {base_dir}")
    method_files = find_all_metrics_files(base_dir)

    if not method_files:
        print("  No metrics.json files found! Run experiments first.")
        return

    print(f"  Found {sum(len(v) for v in method_files.values())} metrics files "
          f"across {len(method_files)} methods")

    # 聚合并行结果
    all_methods = {}
    for method, paths in method_files.items():
        aggregated = aggregate_method_metrics(paths)
        if aggregated:
            all_methods[method] = aggregated
            print(f"    {method}: {len(paths)} runs aggregated")

    if not all_methods:
        print("  No aggregatable results found.")
        return

    # 生成主结果表
    main_table = build_main_results_table(all_methods)
    main_path = os.path.join(report_dir, "main_results.tex")
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(main_table)
    print(f"\n  Main results table: {main_path}")

    # 生成消融表
    ablation_methods = {k: v for k, v in all_methods.items()
                        if any(t in k.lower() for t in ["fedprox_baseline",
                                                         "fedprox_fedbn",
                                                         "fedprox_localhead",
                                                         "fedprox_both",
                                                         "ablation"])}
    if ablation_methods:
        ablation_table = build_ablation_table(ablation_methods)
        ablation_path = os.path.join(report_dir, "ablation.tex")
        with open(ablation_path, "w", encoding="utf-8") as f:
            f.write(ablation_table)
        print(f"  Ablation table: {ablation_path}")

    # 生成迁移矩阵表
    transfer_path = os.path.join(base_dir, "diagnostics", "transfer_matrix.json")
    if os.path.exists(transfer_path):
        transfer_table = build_transfer_table(transfer_path)
        transfer_tex_path = os.path.join(report_dir, "transfer_table.tex")
        with open(transfer_tex_path, "w", encoding="utf-8") as f:
            f.write(transfer_table)
        print(f"  Transfer matrix table: {transfer_tex_path}")

    # 生成各城市详细表
    for city in cities:
        city_table = build_city_details_table(all_methods, city)
        city_path = os.path.join(report_dir, f"city_{city.lower()}.tex")
        with open(city_path, "w", encoding="utf-8") as f:
            f.write(city_table)
    print(f"  Per-city tables: {report_dir}/city_*.tex")

    # 生成论文数值
    paper_numbers = build_paper_numbers(all_methods)
    numbers_path = os.path.join(report_dir, "paper_numbers.txt")
    with open(numbers_path, "w", encoding="utf-8") as f:
        f.write(paper_numbers)
    print(f"  Paper numbers: {numbers_path}")

    # 在主结果表旁也保存一份 .txt
    print(f"\n{paper_numbers}")

    print(f"\n  All reports saved to: {report_dir}")


if __name__ == "__main__":
    main()
