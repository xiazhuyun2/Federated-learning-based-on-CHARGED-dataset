"""
第二阶段快速筛选脚本 (问题与解决3.txt 第九节)

用 10 轮、单种子跑一批小规模实验, 只用于「验证集决策」, 决定哪些方法进入
第三阶段 50 轮正式实验。覆盖:
  1. 场景A (分层抽样每城10站): FedAvg 的 2×2 消融 Base/FedBN/LocalHead/Both
  2. 场景B (比例分配60站):   FedAvg α=0 / 0.5 / 1
  3. FedProx μ 搜索:          μ=0 / 0.001 / 0.01 / 0.1 + local_epochs 1/3
  4. centralized smoke test:   shared vs personalized

用法:
  python experiments/quick_screen.py --dry-run        # 只打印命令
  python experiments/quick_screen.py --subset A       # 只跑场景A
  python experiments/quick_screen.py                  # 全部顺序跑

输出: outputs/summaries/screening.md (按 Macro-City WAPE 排序)
"""
import sys
import os
import json
import subprocess
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OUTPUT_DIR

CITIES = "SZH,AMS,JHB,LOA,MEL,SPO"
SCREEN_DIR = os.path.join(OUTPUT_DIR, "summaries")


def _build_experiments():
    """返回实验清单: [{name, argv, city, method, seed, src}]"""
    exps = []
    common_multi = ["--cities", CITIES, "--num_rounds", "10", "--seed", "42"]

    # ── 场景A: 分层抽样 每城10站, α=0 等权 ──
    a_common = common_multi + [
        "--station_selection", "stratified_balanced", "--top_k", "10",
        "--aggregation", "fedavg", "--city_weight_alpha", "0.0"]
    exps.append({"name": "A_base", "argv": a_common,
                 "city": "SZH", "method": "fedavg_a0_stratified_balanced", "seed": 42, "src": "macro_city"})
    exps.append({"name": "A_fedbn", "argv": a_common + ["--fedbn"],
                 "city": "SZH", "method": "fedavg_fedbn_a0_stratified_balanced", "seed": 42, "src": "macro_city"})
    exps.append({"name": "A_localhead", "argv": a_common + ["--local_head"],
                 "city": "SZH", "method": "fedavg_localhead_a0_stratified_balanced", "seed": 42, "src": "macro_city"})
    exps.append({"name": "A_both", "argv": a_common + ["--fedbn", "--local_head"],
                 "city": "SZH", "method": "fedavg_fedbn_localhead_a0_stratified_balanced", "seed": 42, "src": "macro_city"})

    # ── 场景B: 比例分配60站, FedAvg α=0/0.5/1 ──
    for tag, alpha in [("B_a0", "0.0"), ("B_a0_5", "0.5"), ("B_a1", "1.0")]:
        exps.append({
            "name": tag,
            "argv": common_multi + [
                "--station_selection", "proportional", "--top_k", "60",
                "--aggregation", "fedavg", "--city_weight_alpha", alpha],
            "city": "SZH", "method": f"fedavg_a{alpha.replace('.', '_')}_proportional",
            "seed": 42, "src": "macro_city"})

    # ── FedProx μ 搜索 (场景B, α=0.5) ──
    for tag, mu in [("P_mu0", "0"), ("P_mu0_001", "0.001"),
                    ("P_mu0_01", "0.01"), ("P_mu0_1", "0.1")]:
        exps.append({
            "name": tag,
            "argv": common_multi + [
                "--station_selection", "proportional", "--top_k", "60",
                "--aggregation", "fedprox", "--mu", mu,
                "--city_weight_alpha", "0.5", "--local_epochs", "5"],
            "city": "SZH", "method": "fedprox_a0_5_proportional", "seed": 42, "src": "macro_city"})
    for tag, le in [("P_e1", "1"), ("P_e3", "3")]:
        exps.append({
            "name": tag,
            "argv": common_multi + [
                "--station_selection", "proportional", "--top_k", "60",
                "--aggregation", "fedprox", "--mu", "0.01",
                "--city_weight_alpha", "0.5", "--local_epochs", le],
            "city": "SZH", "method": "fedprox_a0_5_proportional", "seed": 42, "src": "macro_city"})

    # ── centralized smoke test (单城 SZH, top_k=10) ──
    exps.append({
        "name": "C_shared",
        "argv": ["experiments/baseline_centralized.py",
                 "--city", "SZH", "--top_k", "10", "--epochs", "20", "--seed", "42"],
        "city": "SZH", "method": "centralized_shared", "seed": 42, "src": "AVERAGE"})
    exps.append({
        "name": "C_personalized",
        "argv": ["experiments/baseline_centralized_personalized.py",
                 "--cities", "SZH", "--top_k", "10", "--epochs", "20", "--seed", "42"],
        "city": "SZH", "method": "centralized_personalized", "seed": 42, "src": "AVERAGE"})
    return exps


def find_newest_metrics(city, method, seed):
    """返回最新 run 的 (metrics_dict, history_dict)。"""
    path = os.path.join(OUTPUT_DIR, city, method, f"seed_{seed}")
    if not os.path.isdir(path):
        return None, None
    run_dirs = [os.path.join(path, d) for d in os.listdir(path)
                if d.startswith("run_")]
    if not run_dirs:
        return None, None
    newest = max(run_dirs, key=os.path.getmtime)
    metrics = history = None
    mp = os.path.join(newest, "metrics.json")
    hp = os.path.join(newest, "history.json")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            metrics = json.load(f)
    if os.path.exists(hp):
        with open(hp, encoding="utf-8") as f:
            history = json.load(f)
    return metrics, history


def best_val_wape(history):
    """从 history.json 的 val_metrics 取最优验证 WAPE (每5轮记录一次)。"""
    if not history:
        return None
    vm = history.get("val_metrics", [])
    wapes = [v.get("WAPE") for v in vm if isinstance(v, dict) and v.get("WAPE") is not None]
    if not wapes:
        return None
    return float(min(wapes))


def run_all(exps, subset=None, dry_run=False):
    rows = []
    for exp in exps:
        if subset and not exp["name"].startswith(subset):
            continue
        print("\n" + "=" * 70)
        print(f"  [{exp['name']}]  python {' '.join(exp['argv'])}")
        print("=" * 70)
        if not dry_run:
            subprocess.run([sys.executable] + exp["argv"], check=False)

        metrics, history = find_newest_metrics(exp["city"], exp["method"], exp["seed"])
        src = exp["src"]
        block = metrics.get(src, {}) if metrics else {}
        wape = block.get("WAPE") if isinstance(block, dict) else None
        worst = (metrics.get("worst_city", {}).get("WAPE")
                 if metrics and isinstance(metrics.get("worst_city"), dict) else None)
        val_wape = best_val_wape(history)
        rows.append({
            "name": exp["name"],
            "src": src,
            "test_WAPE": wape,
            "worst_city_WAPE": worst,
            "best_val_WAPE": val_wape,
        })
        print(f"  test_WAPE={wape}  worst_city_WAPE={worst}  best_val_WAPE={val_wape}")

    return rows


def write_md(rows):
    os.makedirs(SCREEN_DIR, exist_ok=True)
    # 按 val WAPE 排序 (缺失则用 test WAPE)
    def sort_key(r):
        v = r["best_val_WAPE"] if r["best_val_WAPE"] is not None else r["test_WAPE"]
        return v if v is not None else float("inf")
    rows = sorted(rows, key=sort_key)

    out = os.path.join(SCREEN_DIR, "screening.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# 第二阶段快速筛选结果\n\n")
        f.write("> 10 轮、单 seed；决策应优先看 `best_val_WAPE`（验证集），"
                "`test_WAPE`/`worst_city_WAPE` 作参考。\n\n")
        f.write("| 实验 | 指标源 | test_WAPE | worst_city_WAPE | best_val_WAPE |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            def fmt(v):
                return f"{v:.2f}" if v is not None else "—"
            f.write(f"| {r['name']} | {r['src']} | {fmt(r['test_WAPE'])} | "
                    f"{fmt(r['worst_city_WAPE'])} | {fmt(r['best_val_WAPE'])} |\n")
    print(f"\n  Saved screening summary to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default=None,
                        help="只跑 name 以此前缀开头的实验 (如 A / B / P / C)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印命令不执行")
    args = parser.parse_args()

    exps = _build_experiments()
    rows = run_all(exps, subset=args.subset, dry_run=args.dry_run)
    if rows:
        write_md(rows)


if __name__ == "__main__":
    main()
