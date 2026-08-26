# 最终实验结果

本目录存放项目最终实验结果（已从 `outputs/` 中抽取，原始模型权重/预测图等大文件不纳入 git）。

> ✅ 数字为「修复 P0 数据缺陷 + 全量重跑」后的**可信结果**，由 `experiments/organize_results.py`（场景 A/B/基线）与 `experiments/leave_one_out.py --seeds 42,123,999`（场景 C）聚合，主指标为 **Macro-City WAPE**（每城等权）。
> 可复现 commit：`13dafe3`（P0 修复 + 全量重跑 + 排除内部备份与笔记）。

## 文件说明

| 文件 | 内容 |
|---|---|
| `organized_results.json` | 场景 A（分层抽样 FedBN/LocalHead 消融）、场景 B（比例分配 α 扫描）、各基线，由 `experiments/organize_results.py` 生成，含 3 seed 的 mean±std |
| `leave_one_out_3seed.json` | 场景 C（留一城市冷启动）3 seed 结果，由 `experiments/leave_one_out.py` 生成 |

## 核心结论

### 场景 A · 分层抽样（α=0，个性化消融，Macro-City WAPE %）

| 方法 | Macro-City WAPE | 最差城市 WAPE |
|---|---|---|
| Base (FedAvg) | 34.07±2.01 | 56.33±5.65 |
| **LocalHead** | **32.87±1.33** | **50.07±1.51** |
| FedBN | 37.10±2.28 | 60.18±5.87 |
| FedBN + LocalHead | 36.22±1.70 | 55.40±2.91 |

结论：本地预测头（LocalHead）最优，Macro-City 均值小幅改善（34.07→32.87，1.2 点，仍小于跨种子 std 2.01），但**最差城市 WAPE 明显改善**（56.33→50.07，6.3 点）；FedBN 在本实验协议与模型下未获益（37.10 > 34.07）。

### 场景 B · 比例分配（城市权重指数 α）

| α | Macro-City WAPE | 最差城市 WAPE |
|---|---|---|
| 0（等权） | 40.03±0.15 | **55.38±1.32** |
| **0.5（折中）** | **37.28±1.46** | 56.97±1.22 |
| 1（样本加权） | 37.42±0.99 | 60.49±0.95 |

结论：α 在「精度」与「公平性」之间呈**权衡**——α=0.5 取得最优 Macro-City 精度（37.28），α=0 取得最优公平性（最差城市 55.38），而 α=1（标准样本加权 FedAvg）精度接近 α=0.5 但公平性最差（60.49）。**不存在「α=1 同时最优」**。

### 场景 C · 留一城市冷启动（跨城等权平均 WAPE %）

| 方法 | 平均 WAPE |
|---|---|
| strict zero-shot（严格零样本，scaler 不校准） | 426.5（被 SPO 1916.7 / JHB 322.1 的尺度失配主导） |
| calibrated zero-shot（校准零样本） | 44.97 |
| few-shot (14d / 30d) | 36.91 / **34.52** |
| from-scratch (14d / 30d) | 41.81 / 38.87 |
| full-data local baseline | 46.04 |

结论：**few-shot（30 天）最优（34.52），优于 full-data local baseline（46.04）与校准零样本（44.97）**；from-scratch 30 天（38.87）也反超 full-local。strict zero-shot（426.5）揭示 scaler 不迁移（SPO/JHB 尺度失配），故必须与 calibrated zero-shot 分列报告。预训练平均有益但存在明显负迁移（few-shot 在 4/6 城市劣于校准零样本）。

### 基线（Macro-City WAPE %）

| 基线 | Macro-City WAPE | 最差城市 WAPE |
|---|---|---|
| centralized_personalized（集中训练、每城独立头） | 35.55±2.72 | 52.75±2.92 |
| seasonal_naive（季节朴素） | 35.21±15.12 | MEL 49.40 |
| local_only（各站孤立训练） | 44.54±14.18 | SPO 63.99 |
| centralized_shared（集中共享模型） | 47.94±13.69 | SPO 60.51 |

> ⚠️ 口径提示：单城市基线（seasonal_naive / local_only / centralized_shared）为各城 `top_k=9` 独立训练后**跨城宏平均**，其 std 为**跨 6 城标准差**（非 3-seed std），且站点集合（按电量 top-9）与 FL 场景 A（分层平衡采样）不完全一致；centralized_personalized 为多城集中训练（分层平衡 9 站/城），3-seed std。故基线仅作参考对照，不与 FL 做严格同口径并列比较。
