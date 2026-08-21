# 最终实验结果

本目录存放项目最终实验结果（已从 `outputs/` 中抽取，原始模型权重/预测图等大文件不纳入 git）。

## 文件说明

| 文件 | 内容 |
|---|---|
| `organized_results.json` | 场景 A（分层抽样 FedBN/LocalHead 消融）、场景 B（比例分配 α 扫描）、单城市 FL、各基线，由 `experiments/organize_results.py` 生成，含 3 seed 的 mean±std |
| `leave_one_out_3seed.json` | 场景 C（留一城市冷启动）3 seed 结果，由 `experiments/leave_one_out.py --seeds 42,123,999` 生成 |

## 核心结论

### 场景 A · 分层抽样（α=0，个性化消融，Macro-WAPE %）

| 方法 | Macro-WAPE | 最差城市 WAPE |
|---|---|---|
| Base (FedAvg) | 33.10±2.41 | 53.61±3.13 |
| **LocalHead** | **32.71±1.30** | **51.41±1.76** |
| FedBN | 34.74±2.61 | 55.50±4.70 |
| FedBN + LocalHead | 35.63±3.16 | 56.81±9.37 |

结论：本地预测头（LocalHead）是个性化的关键；FedBN 在此任务上有害。

### 场景 B · 比例分配（城市权重指数 α）

| α | Macro-WAPE | 最差城市 WAPE |
|---|---|---|
| 0（等权） | 37.65±0.98 | 58.39±1.53 |
| 0.5 | 37.12±1.02 | 53.85±1.26 |
| **1（样本加权）** | **36.31±0.56** | **50.33±0.75** |

结论：α=1（按样本数加权）同时提升精度与公平性（最差城市 WAPE 下降 8 个点）。

### 场景 C · 留一城市冷启动（3 seed，平均 WAPE %）

| 方法 | 平均 WAPE |
|---|---|
| Zero-shot | 43.6 |
| Few-shot (14d / 30d) | 44.7 / 45.8 |
| From-scratch (14d / 30d) | 66.6 / 64.4 |
| Full-local (oracle) | 42.1 |

结论：预训练价值显著（few-shot 较 from-scratch 降 ~30% 误差）；zero-shot 已逼近本地 oracle，few-shot 相对 zero-shot 增益有限。

### 基线（Macro-WAPE %）

| 基线 | WAPE |
|---|---|
| centralized_personalized | 31.51±1.57 |
| seasonal_naive | 35.01 |
| local_only | 39.17 |
| centralized_shared | 54.00±10.00 |
