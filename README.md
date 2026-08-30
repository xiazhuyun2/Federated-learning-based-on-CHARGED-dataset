# 基于分层联邦学习的多城市电动汽车充电负荷预测

> 面向「数据分散在各城市、跨城市行为异质、数据量不均衡」的充电负荷预测，提出**站级—城市级—全局**两级分层联邦学习框架，在 CHARGED 六城市数据集上系统验证个性化消融、城市加权与冷启动泛化。

---

## 概述

电动汽车充电负荷预测要落地，绕不开三个现实问题：充电站数据散落在各家运营商、各个城市手里（隐私与数据孤岛）；不同城市充电行为差异大（跨城市 Non-IID）；数据量严重不均衡（深圳 + 阿姆斯特丹约占全部有效数据的 92%）。标准 FedAvg 按样本量加权聚合，会让「数据大户」主导全局模型、小城市精度与公平性受损，单一全局模型也难以适配各城迥异的负荷模式。

本文提出一套可落地、可解释的联邦方案：全局共享 TCN-LSTM 主干之上，用**城市内站点聚合 + 城市间样本量指数加权**两层聚合平衡精度与公平性，再用 **FedBN / 本地预测头**两种轻量个性化缓解跨城市负迁移，并在三种现实场景下逐一验证。

> 说明：本文实现的是**单进程联邦仿真**——所有城市数据在同一进程内按「逻辑分区」模拟各客户端本地训练、只交换模型参数（不共享原始数据），并非跨进程/跨机器的真实分布式部署；未引入差分隐私或安全聚合，只声称 privacy-aware。

## 核心贡献

1. **两层分层联邦聚合框架**：先在城市内部做站点级聚合，再在城市之间做全局聚合，用城市样本量的指数 α 控制城市权重（β_c ∝ N_c^α），在「精度」和「公平性」之间留出可调档位，避免大数据城市一家独大。
2. **个性化机制消融**：在全局共享主干之上，分别考察 FedBN（BN 统计量本地化）与本地预测头（预测头不参与聚合），量化各自对负迁移的缓解效果。
3. **三种现实场景系统验证**：平衡采样（场景 A）、自然不均衡分配（场景 B）、留一城市冷启动（场景 C），覆盖「数据怎么选」「城市怎么加权」「新城市怎么接入」三个实际问题。

## 方法

- **模型**：TCN-LSTM 混合结构（3 个因果空洞卷积块 + 2 层 LSTM + 预测头），可训练参数 **144,664**，输入为 **38 维**（1 维目标 + 37 维特征），输入窗口 168h、预测窗口 24h，轻量到可在资源受限的充电站端本地训练。
- **两层聚合**：站级客户端本地训练（Adam，weight_decay=1e-5，梯度裁剪 norm=5.0）→ 城市内聚合 → 城市间按 β_c ∝ N_c^α 加权。α=0 城市等权，α=1 退化为标准样本加权 FedAvg。
- **个性化**：FedBN 把 BN 统计量留在各站点本地继续更新、不参与全局聚合；本地预测头只共享特征提取主干、预测头留在各站点本地。两者可叠加，构成 4 种消融配置。

## 数据

**CHARGED**（Guo et al., *Scientific Data*, 2025, DOI: 10.1038/s41597-025-05584-7；github.com/IntelligentSystemsLab/CHARGED），6 座城市约 6 个月、每小时粒度的充电记录：

| 城市 | 代码 | 原始站点 | 有效站点 | 24h 自相关 | 168h 自相关 |
|---|---|---|---|---|---|
| 深圳 | SZH | 1379 | ~1347 | 0.70 | 0.50 |
| 阿姆斯特丹 | AMS | 1388 | ~1083 | 0.89 | 0.69 |
| 约翰内斯堡 | JHB | 35 | ~28 | 0.69 | 0.19 |
| 洛杉矶 | LOA | 224 | ~139 | 0.63 | 0.64 |
| 墨尔本 | MEL | 62 | ~22 | 0.50 | 0.26 |
| 圣保罗 | SPO | 41 | ~17 | 0.80 | 0.34 |

数据按 7:1.5:1.5 划分训练/验证/测试集。

## 实验设计

| 场景 | 目标 | 站点选择 | 聚合 | 对比维度 |
|---|---|---|---|---|
| **A** 个性化消融 | 验证个性化机制 | 分层平衡（每城 9 站） | FedAvg（μ=0），α=0 | Base / +FedBN / +LocalHead / +两者 |
| **B** 城市加权扫描 | 验证 α 的精度—公平性权衡 | 比例分配（60 客户端） | FedAvg（μ=0） | α ∈ {0, 0.5, 1.0} |
| **C** 冷启动泛化 | 验证预训练知识迁移 | 留一城市（其余 5 城训练） | FedProx（μ=0.01），α=0.5 | zero-shot / few-shot / from-scratch / full-local |

- **主指标**：Macro-City WAPE（先对每城内站点 WAPE 等权平均、再对城市等权平均），辅以最差城市 WAPE（公平性）与 micro-WAPE（样本加权口径）。
- **基线**：centralized_personalized（集中训练、每城独立头）、centralized_shared（集中共享模型）、local_only（各站孤立训练）、seasonal_naive（季节朴素）。
- **随机种子**：42 / 123 / 999，报告均值 ± 标准差。

## 主要结果

| 场景 | 核心结论 | 关键数字 |
|---|---|---|
| A 个性化 | 纯分层 FedAvg 最优，LocalHead 打平、FedBN 变差 | Macro-City WAPE 最优 **33.68**（Base）；LocalHead 34.37、FedBN 36.90 |
| B 城市加权 | α 是「精度—公平性」权衡旋钮，非单调占优 | α=0.5 精度最优 **37.28**；α=0 公平最优 55.38；α=1 公平最差 60.49 |
| C 冷启动 | 冷启动卡在「尺度不对齐」而非「模型参数迁移」；迁移边际价值随数据量递减 | strict zero-shot 273.6 → calibrated 49.15（零训练仅换 scaler，降 224 点）→ few-shot **38.32**；小城迁移正向、大城（SZH/AMS）受损 |

- 场景 A：纯分层 FedAvg（Base）平均精度最优（33.68%），LocalHead（34.37%）与它几乎打平、FedBN（36.90%）明显变差，最差城市上各方法也都落在跨种子噪声内——两种轻量个性化机制在本协议/模型下均无显著增益。
- 场景 B：α=0.5 精度最好、α=0 公平性最好，而标准样本加权 FedAvg（α=1）精度贴近但公平性最差，代价就是最差城市服务能力被牺牲。
- 场景 C：冷启动的头号杀手是「数据尺度不对齐」而非「模型参数迁移」——strict zero-shot（273.6）与 calibrated zero-shot（49.15）同一主干、同样零训练，只差 scaler 来源就差了 224 点；30 天 few-shot 再降约 11 点（38.32%），但对 from-scratch 只差 0.55 点，且逐城看，四个小城迁移正向、两个数据富集城（SZH/AMS）反而受损。
- 与基线对比：**分层联邦（33.68%）**跑赢全部基线（centralized_personalized 35.55、seasonal_naive 35.21、local_only 44.54、centralized_shared 47.94）。

> 逐场景完整表格、基线口径提示与逐 seed 原始数据见 [`results/README.md`](results/README.md) 与 `results/*.json`。

## 项目结构

```
├── config.py                    # 全局配置（数据/模型/联邦）
├── main.py                      # FL 训练主入口（单/多城市，场景 A/B）
├── requirements.txt             # 固定依赖版本（可复现）
├── src/
│   ├── data/                    # 数据加载、站点筛选、特征工程
│   ├── models/                  # TCN-LSTM 混合网络
│   ├── federated/               # 客户端、聚合服务器（FedAvg/FedProx/Clustered/CityBalanced）
│   └── utils/                   # 评估指标、可视化、实验追踪
├── experiments/
│   ├── leave_one_out.py         # 场景 C：留一城市冷启动
│   ├── full_experiment.py       # 场景 A/B 实验编排
│   ├── baseline_*.py            # 各基线方法
│   ├── organize_results.py      # 聚合结果 → results/
│   ├── plot_paper_figures.py    # 论文图（fig1–4）
│   └── rerun_p0_fixed.sh        # P0 修复后一键重跑（场景 C + A + 聚合）
├── paper/                       # 论文初稿 + figures/
├── results/                     # 最终实验结果 JSON + 逐项说明
└── outputs/                     # 运行输出（模型权重/日志/图，不纳入 git）
```

## 快速开始

### 环境

```bash
# Python 3.11 + 依赖（版本锁定于 requirements.txt；GPU 可选，代码自动检测）
pip install -r requirements.txt
# GPU（可选）：torch 经 CUDA 12.6 预编译包安装，见 requirements.txt 注释
```

### 一键复现

```bash
PY=python bash experiments/rerun_p0_fixed.sh
# 依次：场景 C（6 fold × 3 seed）→ 场景 A（4 配置 × 3 seed）→ organize_results.py 聚合
```

### 单场景运行

```bash
# 场景 A — 个性化消融（分层平衡 9 站/城，α=0，FedAvg，50 轮）
python main.py --cities SZH,AMS,JHB,LOA,MEL,SPO --station_selection stratified_balanced \
  --top_k 9 --aggregation fedavg --city_weight_alpha 0 --num_rounds 50 --local_epochs 5 \
  --seeds 42,123,999 [--fedbn] [--local_head]

# 场景 B — 城市加权扫描（比例分配 60 客户端，FedAvg，扫描 α）
python main.py --cities SZH,AMS,JHB,LOA,MEL,SPO --station_selection proportional \
  --top_k 60 --aggregation fedavg --city_weight_alpha {0|0.5|1.0} \
  --num_rounds 50 --local_epochs 5 --seeds 42,123,999

# 场景 C — 留一城市冷启动（FedProx μ=0.01，α=0.5，30 轮，top_k=10）
python experiments/leave_one_out.py --top_k 10 --rounds 30 --local_epochs 5 \
  --finetune_days 14,30 --seeds 42,123,999
```

### 聚合与绘图

```bash
python experiments/organize_results.py     # 聚合场景 A/B/基线 → results/organized_results.json
python experiments/plot_paper_figures.py   # 重画论文图 fig1–4
```

## 复现说明

- **依赖锁定**：`requirements.txt` 中 7 项依赖全部 `==` 固定（torch 2.11.0 + CUDA 12.6 经 `.venv` 于 Windows 11 / RTX 4060 验证）。
- **随机性控制**：全部 FL 实验取 3 个种子（42/123/999），报告均值 ± 标准差；站点选择为确定性流程。
- **站点清单**：FL 各 run 站点 ID 见 `outputs/station_lists/*.json`，单城市基线/留一城市为确定性 top-k 选站，可从各 run `metrics.json` 站点键恢复。
- **可复现提交**：完整实验对应的提交哈希与逐项数字见 [`results/README.md`](results/README.md)。

## 引用

数据来源：Guo et al., *A City-scale and Harmonized Dataset for Global Electric Vehicle Charging Demand Analysis*, Scientific Data, 2025, DOI: 10.1038/s41597-025-05584-7（github.com/IntelligentSystemsLab/CHARGED）。
