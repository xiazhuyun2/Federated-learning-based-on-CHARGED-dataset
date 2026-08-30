# 最终实验结果

本目录存放项目最终实验结果（已从 `outputs/` 中抽取，原始模型权重/预测图等大文件不纳入 git）。

> ✅ 数字为「修复 P0 数据缺陷 + 全量重跑」后的**可信结果**，由 `experiments/organize_results.py`（场景 A/B/基线）与 `experiments/leave_one_out.py --seeds 42,123,999`（场景 C）聚合，主指标为 **Macro-City WAPE**（每城等权）。
> 可复现 commit：待回填（P0 修复 + 全量重跑后的新提交；依赖已锁定 `requirements.txt`；FL 站点 ID 见 `outputs/station_lists/*.json`）。上一轮可复现引用 `13dafe3` 已被 P0 缺陷污染，不再作为本目录数字的可复现基线。

## 文件说明

| 文件 | 内容 |
|---|---|
| `organized_results.json` | 场景 A（分层抽样 FedBN/LocalHead 消融）、场景 B（比例分配 α 扫描）、各基线，由 `experiments/organize_results.py` 生成，含 3 seed 的 mean±std |
| `leave_one_out_3seed.json` | 场景 C（留一城市冷启动）3 seed 结果，由 `experiments/leave_one_out.py` 生成 |

## 核心结论

### 场景 A · 分层抽样（α=0，个性化消融，Macro-City WAPE %）

| 方法 | Macro-City WAPE | 最差城市 WAPE |
|---|---|---|
| **Base (FedAvg)** | **33.68±1.56** | 55.65±4.70 |
| LocalHead | 34.37±2.10 | **55.20±5.57** |
| FedBN | 36.90±2.18 | 60.51±6.89 |
| FedBN + LocalHead | 35.89±2.31 | 56.53±5.44 |

结论：在这套协议与模型下，两种轻量个性化机制都**没有带来显著增益**——纯分层 FedAvg（Base）反而最优（33.68）。LocalHead（34.37）与 Base 几乎打平（平均差 0.7 点、最差城市改善 0.45 点，均在跨种子 std 内）；FedBN 明显变差（36.90，约 +3.2 点）。

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
| strict zero-shot（严格零样本，scaler 不校准） | 273.6（被 SPO / JHB 的尺度失配主导） |
| calibrated zero-shot（校准零样本） | 49.15 |
| few-shot (14d / 30d) | 39.97 / **38.32** |
| from-scratch (14d / 30d) | 41.81 / 38.87 |
| full-data local baseline | 39.31 |

结论：**冷启动的头号杀手是「尺度不对齐」而非「模型参数迁移」**——strict zero-shot（273.6，被 SPO 1246/JHB 139.9 主导）与 calibrated zero-shot（49.15）用的是同一预训练主干、同样零训练，只差 scaler 来源就差了 224 点，证明预训练主干可跨城复用、卡点在数据归一化。在此之上 30 天 few-shot 再降约 11 点（49.15→38.32），但迁移的边际价值随本地数据量递减：few-shot 对 from-scratch 只差 0.55 点，且逐城看，JHB/LOA/MEL/SPO 四个小城迁移正向（5–10 点）、SZH/AMS 两个数据富集城反而受损（SZH 差 22 点）。

### 基线（Macro-City WAPE %）

| 基线 | Macro-City WAPE | 最差城市 WAPE |
|---|---|---|
| centralized_personalized（集中训练、每城独立头） | 35.55±2.72 | 52.75±2.92 |
| seasonal_naive（季节朴素） | 35.21±15.12 | MEL 49.40 |
| local_only（各站孤立训练） | 44.54±14.18 | SPO 63.99 |
| centralized_shared（集中共享模型） | 47.94±13.69 | SPO 60.51 |

> ⚠️ 口径提示：单城市基线（seasonal_naive / local_only / centralized_shared）为各城 `top_k=9` 独立训练后**跨城宏平均**，其 std 为**跨 6 城标准差**（非 3-seed std），且站点集合（按电量 top-9）与 FL 场景 A（分层平衡采样）不完全一致；centralized_personalized 为多城集中训练（分层平衡 9 站/城），3-seed std。故基线仅作参考对照，不与 FL 做严格同口径并列比较。
