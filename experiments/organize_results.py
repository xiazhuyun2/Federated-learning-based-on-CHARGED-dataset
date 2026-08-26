"""
结果整理脚本 — 把 outputs/ 里散落的 run 按 config.json 重新归类

背景:
  main.py 曾只用聚合类型命名输出目录 (fedavg/fedprox/...), 不含 α、不含单/多城市标记,
  导致 fedavg 的 α=0/0.5/1、单城市与多城市 FL 全部挤在同一目录, 靠 run_ 时间戳区分。
  本脚本不重跑任何实验, 只读取每个 run 的 config.json, 重新归类为「逻辑实验名」,
  并对同实验下的 3 个 seed 求 mean±std。

用法:
  python experiments/organize_results.py
  python experiments/organize_results.py --base_dir outputs --json-only

输出:
  - 控制台打印 markdown 表格 (RMSE/MAE/WAPE mean±std)
  - outputs/summaries/organized_results.json
"""
import os
import sys
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALL_CITIES = ["SZH", "AMS", "JHB", "LOA", "MEL", "SPO"]
BASELINE_DIRS = {"local_only", "centralized", "centralized_shared",
                 "centralized_personalized", "seasonal_naive"}
# 单城市基线: 每城独立训练, 需跨城宏平均 (macro-city) 才能与多城市 FL 口径一致。
SINGLE_CITY_BASELINES = {"local_only", "centralized_shared", "seasonal_naive"}
# 顶层非实验目录 (不包含 method/seed_/run_ 结构), 遍历时跳过
SKIP_TOP_DIRS = {"summaries", "leave_one_out"}
METRIC_KEYS = ["RMSE", "MAE", "WAPE", "SMAPE", "MAPE", "NRMSE"]


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def classify_run(cfg, method_dir, city=None):
    """根据 config.json + 目录名, 返回 (逻辑实验名, 是否多城市, 主指标来源)。

    返回 None 表示无法归类 (如 config 缺失且非 baseline)。
    city 为 run 所在的城市目录名, 用于单城市实验按城市分桶 (修复前 cfg.data.cities
    在单城市运行下仍写默认 6 城, 无法从 config 得知真实城市)。
    """
    if method_dir in BASELINE_DIRS:
        if method_dir == "centralized_personalized":
            # 多城市集中式个性化: 分层平衡 9 站/城, 宏站==宏城, 用 macro_city 与 FL 口径一致
            return (method_dir, True, "macro_city")
        # 单城市基线 (local_only / centralized_shared / seasonal_naive):
        # 必须按城市分桶, 否则 6 城挤进同一 seed 桶, pick_best 只留最新一城 (SPO),
        # 得到的是 SPO-only 数字 (seasonal_naive 的 std=0 即由此而来)。
        return (f"{method_dir}__{city}", False, "AVERAGE")

    if cfg is None:
        return None

    fed = cfg.get("fed", {}) or {}
    model = cfg.get("model", {}) or {}
    data = cfg.get("data", {}) or {}

    agg = fed.get("aggregation", method_dir)
    alpha = fed.get("city_weight_alpha", None)
    fb = bool(model.get("use_fedbn", False))
    lh = bool(model.get("use_local_head", False))
    cities = data.get("cities") or []
    n_cities = len(cities) if isinstance(cities, list) else 0
    # 优先用 multi_city_mode 判定单/多城市。cfg.data.cities 默认就是全部 6 城,
    # 单城市运行也会写入 6 城, 因此 n_cities>=2 不能作为多城市依据。
    mm = fed.get("multi_city_mode")
    is_multi = (mm == "multi_city") or (mm is None and n_cities >= 2)

    if agg == "clustered":
        name = "clustered_fedbn" if fb else "clustered"
        return (name, is_multi, "macro_city" if is_multi else "AVERAGE")

    if is_multi:
        # 统一命名: {agg}_a{alpha}_{station_selection}[_fedbn][_localhead]
        # station_selection 用于区分场景A(分层)与场景B(比例)在同一 α 下的不同实验。
        a = alpha
        if a is None or abs(a - 1.0) < 1e-6:
            a_tag = "a1"
        elif abs(a - 0.5) < 1e-6:
            a_tag = "a0_5"
        elif abs(a - 0.0) < 1e-6:
            a_tag = "a0"
        else:
            a_tag = f"a{float(a):g}".replace(".", "_")
        ss = data.get("station_selection") or "top_k"
        key = f"{agg}_{a_tag}_{ss}"
        if fb:
            key += "_fedbn"
        if lh:
            key += "_localhead"
        return (key, True, "macro_city")

    # 单城市 FL
    name = agg
    if fb:
        name += "_fedbn"
    if lh:
        name += "_localhead"
    # 单城市必须按城市分开报告, 否则 6 城挤进同一个 "fedavg" 桶,
    # 每个 seed 用 pick_best 乱挑一个城市, 得到无意义的均值。
    single_city = (cities[0] if (isinstance(cities, list) and len(cities) == 1)
                   else city)
    if single_city:
        name += f"_single_{single_city}"
    return (name, False, "AVERAGE")


def collect_runs(base_dir):
    """遍历 outputs/, 返回 {逻辑实验名: {seed: [run记录...]}}"""
    organized = defaultdict(lambda: defaultdict(list))

    for city in sorted(os.listdir(base_dir)):
        if city in SKIP_TOP_DIRS:
            continue
        city_dir = os.path.join(base_dir, city)
        # 单城市实验的城市目录在 ALL_CITIES 内; 多城市基线 (centralized_personalized)
        # 用 "+".join(cities) 作目录名 (如 "SZH+AMS+..."), 不在 ALL_CITIES, 但也需收集。
        if not os.path.isdir(city_dir):
            continue
        for method_dir in sorted(os.listdir(city_dir)):
            md = os.path.join(city_dir, method_dir)
            if not os.path.isdir(md):
                continue
            for seed_dir in sorted(os.listdir(md)):
                sd = os.path.join(md, seed_dir)
                if not os.path.isdir(sd) or not seed_dir.startswith("seed_"):
                    continue
                try:
                    seed = int(seed_dir.split("_")[1])
                except (IndexError, ValueError):
                    seed = seed_dir
                for run_dir in sorted(os.listdir(sd)):
                    rd = os.path.join(sd, run_dir)
                    if not os.path.isdir(rd) or not run_dir.startswith("run_"):
                        continue
                    cfg = _read_json(os.path.join(rd, "config.json"))
                    metrics = _read_json(os.path.join(rd, "metrics.json"))
                    if metrics is None:
                        continue
                    cls = classify_run(cfg, method_dir, city)
                    if cls is None:
                        continue
                    name, is_multi, src = cls
                    rounds = (cfg or {}).get("fed", {}).get("num_rounds", None) if cfg else None
                    organized[name][seed].append({
                        "run_dir": rd,
                        "run": run_dir,
                        "rounds": rounds,
                        "is_multi": is_multi,
                        "src": src,
                        "metrics": metrics,
                    })
    return organized


def pick_best(runs):
    """同一 (实验, seed) 下挑最优 run: 轮数最高优先, 其次时间戳最新 (run_ 后缀字典序)。"""
    if not runs:
        return None
    # 优先 rounds 大; rounds 为 None 视为 0
    return max(runs, key=lambda r: ((r["rounds"] or 0), r["run"]))


def summarize(organized):
    """返回 {逻辑实验名: {metric: {mean, std, n, per_seed}}}"""
    result = {}
    for name, seeds in sorted(organized.items()):
        # 每个 seed 取最优 run
        per_seed = {}
        for seed, runs in seeds.items():
            best = pick_best(runs)
            if best is None:
                continue
            m = best["metrics"]
            src = best["src"]
            block = m.get(src, m.get("AVERAGE", m.get("macro_city", {})))
            if isinstance(block, dict) and "RMSE" in block:
                per_seed[seed] = {
                    "metrics": block,
                    "rounds": best["rounds"],
                    "src": src,
                    "worst_city_WAPE": (m.get("worst_city") or {}).get("WAPE"),
                    "micro_WAPE": (m.get("micro") or {}).get("WAPE"),
                }
        if not per_seed:
            continue
        entry = {"n_seeds": len(per_seed), "seeds": {}, "src": None, "rounds": None}
        for metric in METRIC_KEYS:
            vals = [v["metrics"][metric] for v in per_seed.values()
                    if metric in v["metrics"] and v["metrics"][metric] is not None]
            if vals:
                entry[metric] = {
                    "mean": float(sum(vals) / len(vals)),
                    "std": float((sum((x - sum(vals) / len(vals)) ** 2
                                       for x in vals) / len(vals)) ** 0.5),
                    "n": len(vals),
                }
        # 公平性 / 自然分布补充指标
        for metric in ["worst_city_WAPE", "micro_WAPE"]:
            vals = [v.get(metric) for v in per_seed.values()
                    if v.get(metric) is not None]
            if vals:
                entry[metric] = {
                    "mean": float(sum(vals) / len(vals)),
                    "std": float((sum((x - sum(vals) / len(vals)) ** 2
                                       for x in vals) / len(vals)) ** 0.5),
                    "n": len(vals),
                }
        for seed, v in per_seed.items():
            entry["seeds"][seed] = {
                "RMSE": v["metrics"].get("RMSE"),
                "MAE": v["metrics"].get("MAE"),
                "WAPE": v["metrics"].get("WAPE"),
                "worst_city_WAPE": v.get("worst_city_WAPE"),
                "rounds": v["rounds"],
            }
            entry["src"] = v["src"]
            entry["rounds"] = v["rounds"]
        result[name] = entry
    return result


def aggregate_single_city_baselines(result):
    """把单城市基线 (local_only/centralized_shared/seasonal_naive) 的
    每城 3-seed 均值跨城宏平均, 得到与多城市 FL 口径一致的 macro-city 数字。

    每城详情保留在 `{method}__{city}` 条目里; 汇总写回 `{method}` 条目。
    std 为跨城 (6 城) 标准差, 表征城市间离散度 (与 3-seed 标准差含义不同)。
    """
    for method in SINGLE_CITY_BASELINES:
        city_means = {}
        for city in ALL_CITIES:
            e = result.get(f"{method}__{city}")
            if not e or "WAPE" not in e:
                continue
            city_means[city] = {k: e[k]["mean"] for k in METRIC_KEYS if k in e}
        if not city_means:
            continue
        n_cities = len(city_means)
        entry = {"n_seeds": n_cities, "src": "macro_city",
                 "aggregation": "macro-city over single-city baselines"}
        for metric in METRIC_KEYS:
            vals = [m[metric] for m in city_means.values() if metric in m]
            if vals:
                mean = sum(vals) / len(vals)
                std = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
                entry[metric] = {"mean": mean, "std": std, "n": len(vals)}
        worst_city = max(city_means.items(), key=lambda kv: kv[1]["WAPE"])
        entry["worst_city_WAPE"] = {"mean": worst_city[1]["WAPE"],
                                    "std": 0.0, "n": 1,
                                    "city": worst_city[0]}
        entry["per_city_WAPE"] = {c: m["WAPE"] for c, m in city_means.items()}
        result[method] = entry


def print_table(result):
    print("\n" + "=" * 90)
    print("  整理后的实验结果 (按 config.json 重映射, 同实验 3 seeds 求 mean±std)")
    print("=" * 90)
    header = (f"  {'实验':<30s} {'n':>3s} {'RMSE':>15s} {'MAE':>15s} "
              f"{'WAPE':>15s} {'worst':>10s}")
    print(header)
    print("  " + "-" * 88)
    for name, e in sorted(result.items()):
        if "__" in name:  # 单城市基线每城详情 (local_only__SZH 等), 已在汇总行体现, 不重复打印
            continue
        def fmt(k):
            if k in e:
                return f"{e[k]['mean']:>8.2f}±{e[k]['std']:<5.2f}"
            return " " * 14
        def fmt_worst():
            if "worst_city_WAPE" in e:
                return f"{e['worst_city_WAPE']['mean']:>9.2f}"
            return " " * 10
        line = (f"  {name:<30s} {e['n_seeds']:>3d} "
                f"{fmt('RMSE')} {fmt('MAE')} {fmt('WAPE')} {fmt_worst()}")
        print(line)
    print("=" * 90)
    print("  注: 多城市实验用 macro_city (每城市等权), 单城市/基线用 AVERAGE (每站平均);")
    print("      worst = 最差城市 WAPE (公平性); micro 见 organized_results.json\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=None)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    base_dir = args.base_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

    organized = collect_runs(base_dir)
    result = summarize(organized)
    aggregate_single_city_baselines(result)

    if not args.json_only:
        print_table(result)

    out_dir = os.path.join(base_dir, "summaries")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "organized_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
