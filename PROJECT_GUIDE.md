# 多城市分层联邦学习 EV 充电负荷预测 — 项目指南

> 基于 CHARGED 六城市数据集，面向跨城市数据异质性与样本不均衡的分层个性化联邦学习。

---

## 1. 整体架构

```
config.py          → 全局配置 (数据/模型/联邦)
src/data/          → 数据加载、站点筛选、特征工程
src/models/        → TCN-LSTM 混合神经网络
src/federated/     → FL 客户端、聚合服务器 (FedAvg/FedProx/Clustered/CityBalanced)
src/utils/         → 评估指标、可视化、实验追踪
main.py            → 单城市/多城市 FL 训练入口
experiments/       → 诊断、基线、完整实验矩阵、报告生成
```

**论文三项贡献**：
1. 城市内站点聚合 + 城市间 β_c ∝ N_c^α 平衡聚合，防深圳数据主导
2. 全局共享 TCN-LSTM 主干 + 本地 BN/预测头，个性化减少负迁移
3. 自然不均衡、平衡采样、新城市冷启动三种场景验证

---

## 2. 实施阶段 (6 Phases，全部已完成)

| Phase | 内容 | 新建/修改文件 | 状态 |
|-------|------|-------------|------|
| A | 六城市数据诊断 | `experiments/data_diagnostics.py` | ✅ |
| B | 跨城市预处理 (时区/电价/负荷归一化/分层抽样) | `src/data/data_loader.py`, `config.py` | ✅ |
| C | 多城市分层FL架构 (CityBalancedServer/两级聚合/多层级指标) | `src/federated/trainer.py`, `aggregation.py`, `main.py` | ✅ |
| D | 跨城市诊断实验 (相似性/迁移矩阵/预实验对比) | `experiments/cross_city_diagnostics.py` | ✅ |
| E | 完整论文实验体系 | `experiments/full_experiment.py`, `leave_one_out.py` | ✅ |
| F | 多城市可视化与 LaTeX 报告 | `src/utils/visualization.py`, `experiments/generate_report.py` | ✅ |

---

## 3. 数据诊断结果

六城市 4393 小时数据 (~6个月)，站点数严重不均：

| 城市 | 原始站点 | 有效站点 | 24h自相关 | 168h自相关 | 定位 |
|------|---------|---------|-----------|-----------|------|
| SZH | 1379 | ~1347 | 0.70 | 0.50 | 主训练城市 |
| AMS | 1388 | ~1083 | 0.89 | 0.69 | 主训练城市 |
| JHB | 35 | ~28 | 0.69 | 0.19 | 小数据/冷启动候选 |
| LOA | 224 | ~139 | 0.63 | 0.64 | 中等训练城市 |
| MEL | 62 | ~22 | 0.50 | 0.26 | 小数据/冷启动候选 |
| SPO | 41 | ~17 | 0.80 | 0.34 | 小数据/冷启动候选 |

**关键发现**：SZH+AMS 占 95%+ 有效数据，FedAvg(α=1) 下深圳完全主导。小城市 JHB/MEL/SPO 做训练城市意义有限，建议作为留一法冷启动测试城市。

---

## 4. 已完成的端到端验证

```bash
python main.py --cities SZH,AMS --aggregation fedprox --top_k 3 \
  --num_rounds 5 --local_epochs 2 --city_weight_alpha 0.5 \
  --fedbn --local_head --seed 42
```

结果 (5轮快速训练，仅供参考量级)：

| 层级 | RMSE | MAE | WAPE |
|------|------|-----|------|
| **Macro-City (论文主指标)** | 51.86 | 45.51 | 49.01% |
| SZH (3站) | 95.23 | 83.64 | 77.14% |
| AMS (3站) | 8.50 | 7.37 | 20.88% |

> 完整 50 轮 + 20 站 + 6 城市 + 3 种子的结果会显著优于此快速验证。

---

## 5. 运行指南

### 5.1 快速验证 (--quick 模式，约 10 分钟)

```bash
# 跨城市诊断 (只跑相似性+迁移矩阵，top_k=5，epochs=10)
python experiments/cross_city_diagnostics.py --task all --quick

# 完整实验矩阵验证 (top_k=5，rounds=10，2种子)
python experiments/full_experiment.py --mode quick

# 留一城市验证 (rounds=5，top_k=3)
python experiments/leave_one_out.py --quick --skip_full_local
```

### 5.2 数据诊断

```bash
# 生成6城市完整诊断报告
python experiments/data_diagnostics.py --cities SZH,AMS,JHB,LOA,MEL,SPO
# 输出: outputs/diagnostics/{city}_stats.json, city_comparison.csv, city_comparison.png
```

### 5.3 单城市 FL (调试/消融)

```bash
# 深圳单城市 FedProx + FedBN + LocalHead
python main.py --city SZH --aggregation fedprox --top_k 20 \
  --num_rounds 50 --local_epochs 5 --mu 0.01 --fedbn --local_head

# 多种子运行
python main.py --city SZH --aggregation fedprox --top_k 20 \
  --num_rounds 50 --seeds 42,123,999
```

### 5.4 多城市 FL (论文主实验)

```bash
# 推荐主方法: 6城市 FedProx + 城市平衡(α=0.5) + FedBN + LocalHead
python main.py --cities SZH,AMS,JHB,LOA,MEL,SPO \
  --aggregation fedprox --top_k 20 --num_rounds 50 --local_epochs 5 \
  --city_weight_alpha 0.5 --mu 0.01 --fedbn --local_head

# 对比: 标准样本加权 (α=1，SZH主导)
python main.py --cities SZH,AMS,JHB,LOA,MEL,SPO \
  --aggregation fedprox --top_k 20 --num_rounds 50 \
  --city_weight_alpha 1.0 --mu 0.01

# 对比: 六城市等权 (α=0)
python main.py --cities SZH,AMS,JHB,LOA,MEL,SPO \
  --aggregation fedprox --top_k 20 --num_rounds 50 \
  --city_weight_alpha 0.0 --mu 0.01

# 分层抽样替代 Top-K
python main.py --cities SZH,AMS,JHB,LOA,MEL,SPO \
  --aggregation fedprox --top_k 20 --station_selection stratified_natural
```

### 5.5 完整论文实验 (一键运行)

`full_experiment.py` 是实验编排脚本，会自动运行一系列实验矩阵。与 5.3/5.4 不同，5.3/5.4 用 `main.py` 手动跑单个实验。

```bash
# 任何 mode 都可以加 --quick 快速验证 (top_k=5, rounds=10, 2 seeds)
python experiments/full_experiment.py --mode ablation --quick
python experiments/full_experiment.py --mode multi_city --quick

# 所有实验矩阵 (耗时数小时，建议后台运行)
python experiments/full_experiment.py --mode all \
  --top_k 20 --rounds 50 --seeds 42,123,999

# 仅多城市实验
python experiments/full_experiment.py --mode multi_city \
  --top_k 20 --rounds 50

# 消融实验 — 单城市(SZH) + 多城市(6城)各跑一遍 FedBN/LocalHead/Both
python experiments/full_experiment.py --mode ablation --quick
python experiments/full_experiment.py --mode ablation --top_k 20 --rounds 50
# 输出: outputs/summaries/ablation_single_SZH_*.json + ablation_multi_*.json

# 留一城市冷启动 (单独脚本，不在 full_experiment 内)
python experiments/leave_one_out.py --quick --skip_full_local
python experiments/leave_one_out.py --top_k 10 --rounds 30 --seeds 42,123,999
```

> **注意**: `full_experiment.py --mode ablation` 是自动编排的消融矩阵；`main.py` 5.3 是手动单次实验。前者用于论文，后者用于调试。

### 5.6 生成论文报告

```bash
# 扫描 outputs/ 下所有结果，生成 LaTeX 表格
python experiments/generate_report.py
# 输出: outputs/reports/{main_results,ablation,transfer_table,city_*}.tex + paper_numbers.txt
```

---

## 6. 关键参数调优建议

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `--top_k` | 20 | 10 (多城) / 20 (单城) | 多城市时减少每城站点数防止 GPU OOM |
| `--num_rounds` | 50 | 50-100 | 6城市需要更多通信轮次收敛 |
| `--local_epochs` | 5 | 3-5 | 多城市时减少本地epoch防止过拟合 |
| `--city_weight_alpha` | 0.5 | **0.5** (推荐) | 0=等权, 1=SZH主导, 0.5=折中 |
| `--mu` | 0.01 | 0.005-0.02 | FedProx 近端项，异质性大时加大 |
| `--aggregation` | fedprox | **fedprox** | 多城市异质性强，FedProx 优于 FedAvg |
| `--fedbn` | False | **True** | 跨城市强烈推荐，BN统计量本地化 |
| `--local_head` | False | **True** | 推荐，预测头不参与全局聚合 |
| `--station_selection` | top_k | stratified_natural | 论文需要分层抽样验证 |

---

## 7. 下一步行动 (按优先级)

### ✅ 已完成 (5.1 + 5.2)

1. **跨城市诊断** — 相似性 + 迁移矩阵 + 预实验对比
   - 迁移矩阵关键发现: AMS→SZH 优于 SZH 自训练 (81.3 vs 88.3，正迁移)
   - 小城市 JHB/SPO 外部模型误差 ≈ 自训练，跨城市知识迁移对小城市可行
2. **数据诊断** — SZH_stats.json, AMS_stats.json, JHB_stats.json + city_comparison.csv/png

### 现在做

3. **验证 leave_one_out 修复** (已修 input_dim 推断 bug):
   ```bash
   python experiments/leave_one_out.py --quick --skip_full_local
   ```

4. **运行消融实验** (单城市 + 多城市, FedBN/LocalHead/Both):
   ```bash
   python experiments/full_experiment.py --mode ablation --quick
   ```

### 完整论文实验 (后台运行)

5. **完整实验矩阵**:
   ```bash
   # 先 quick 验证管道无报错
   python experiments/full_experiment.py --mode quick
   # 确认无误后跑完整版 (数小时，top_k=10 防 OOM)
   python experiments/full_experiment.py --mode all --rounds 50 --top_k 10 --seeds 42,123,999
   ```

6. **留一城市冷启动** (先 quick 验证):
   ```bash
   python experiments/leave_one_out.py --quick --skip_full_local
   # 确认无误后跑完整版
   python experiments/leave_one_out.py --top_k 10 --rounds 30 --seeds 42,123,999
   ```

### 论文写作前

7. **生成 LaTeX 表格**:
   ```bash
   python experiments/generate_report.py
   ```
   将 `outputs/reports/` 下的 `.tex` 文件直接复制到论文。

8. **论文数据确认清单**:
   - [ ] α=0 / 0.5 / 1.0 对比 → 验证"城市平衡"的必要性
   - [ ] FedBN / LocalHead / Both 消融 → 验证个性化的有效性 (单城+多城)
   - [ ] 迁移矩阵中正/负迁移城市对 → 论文 insight 亮点
   - [ ] 小城市相比 Local-only 的提升
   - [ ] Macro-City 指标 vs Micro 指标的差异

---

## 8. 输出目录结构

```
outputs/
├── diagnostics/             ← data_diagnostics.py + cross_city_diagnostics.py 输出
├── {city}/                  ← 单城市实验 (e.g. SZH/fedprox_fedbn_localhead/seed_42/run_xxx/)
├── test_multi/              ← 多城市快速验证
├── validation/              ← 端到端验证
├── leave_one_out/           ← 留一城市冷启动
└── reports/                 ← generate_report.py LaTeX 输出
     ├── main_results.tex    ← 主结果表 (直接复制到论文)
     ├── ablation.tex        ← 消融表
     ├── transfer_table.tex  ← 迁移矩阵
     ├── city_*.tex          ← 每城市详细表
     └── paper_numbers.txt   ← 论文可直接引用的数值
```

每个实验运行目录 (`run_xxx/`) 包含:
`config.json` / `history.json` / `metrics.json` / `best_model.pt` / `predictions.npz` / `*.png`

`metrics.json` 结构 (多城市): `{station_id: metrics, AVERAGE, macro_city, per_city: {city: metrics}}`
