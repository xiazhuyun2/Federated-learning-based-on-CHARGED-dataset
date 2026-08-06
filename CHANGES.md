# 联邦学习电动汽车负荷预测 — 问题修复与优化日志

> **日期**: 2026-08-06  
> **基准**: 原始项目代码（基于 CHARGED 数据集，TCN-LSTM + 联邦学习）  
> **参考文档**: `问题与解决1.txt`

---

## 一、修复总览

| 优先级 | 问题编号 | 问题描述 | 严重程度 | 状态 |
|--------|---------|---------|---------|------|
| P0 | #9 | 气象/价格按行拼接而非时间戳合并 | 🔴 致命 | ✅ 已修复 |
| P0 | #8 | 选站使用全时间段数据（测试集泄漏） | 🔴 致命 | ✅ 已修复 |
| P0 | #2 | 超参数搜索使用测试集选参（数据泄漏） | 🔴 致命 | ✅ 已修复 |
| P0 | #5 | 每次实验覆盖 test_results.json | 🔴 严重 | ✅ 已修复 |
| P0 | #3 | 聚类模式没有使用 FedProx | 🔴 严重 | ✅ 已修复 |
| P0 | #4 | 缺少 Centralized 和 LSTM-only 基线 | 🔴 严重 | ✅ 已修复 |
| P0 | #7 | MAPE 排除 |y|≤1 的样本 | 🟡 中等 | ✅ 已修复 |
| P0 | #6 | 配置/文档/产物不一致 | 🟡 中等 | ✅ 已修复 |
| P1 | #3 | 聚类使用绝对模型参数（初始化主导） | 🟡 中等 | ✅ 已修复 |
| P1 | - | FedBN + 本地预测头 | 🟢 增强 | ✅ 已实现 |
| P1 | - | 增强特征：滞后/滚动/静态特征 | 🟢 增强 | ✅ 已实现 |

---

## 二、逐问题详细修复说明

### 问题 #9: 气象/价格按行拼接 → 按时间戳合并

**问题**: `build_station_dataframe()` 中，气象和电价特征使用 `.values` 直接按行拼接，而非按时间戳合并。一旦 CSV 文件缺行或错位，会产生静默特征错配。

**修复**（文件: `src/data/data_loader.py`）:
1. 新增 `_parse_timestamps()` 函数，统一解析三种时间格式：
   - `"2023/4/1 0:00"` (weather/volume)
   - `"2023-04-01 00:00:00"` (e_price/s_price)
2. 使用 `pd.merge()` 按时间戳列左连接 weather、e_price、s_price
3. 合并后检查行数是否变化，变化时打印 WARNING
4. 用 `ffill()` 替代 `fillna(0)` 处理缺失值，更合理
5. 新增缺失值标记列（`_is_missing` 二值标记）

**影响**: 所有依赖 `build_station_dataframe()` 的代码自动受益。

---

### 问题 #8: 选站使用全时间段数据 → 仅训练期统计

**问题**: `select_top_stations()` 在全时间段数据上计算均值、标准差、零值率，测试期的行为会影响站点选择。

**修复**（文件: `src/data/data_loader.py`）:
1. `select_top_stations()` 新增 `train_ratio` 参数（默认 0.85）
2. 所有统计量（mean/std/zero_ratio）仅在 `volume_df.iloc[:n_train]` 上计算
3. 调用方（`trainer.py`、`baseline_local.py`）传入 `train_ratio + val_ratio`
4. `load_city_data()` 统一从 `_remove_zero` 目录加载所有文件

**影响**: 选站逻辑不再依赖测试期数据，消除信息泄漏。

---

### 问题 #2: 超参数搜索 → 使用验证集选参

**问题**: `experiments/hyperparam_search.py` 中 `run_single_config()` 调用 `run_federated_training()` 返回的是测试集指标，搜索直接用测试集 RMSE 选参。

**修复**（文件: `experiments/hyperparam_search.py`）:
1. 每次运行同时记录验证集指标（从 `history.json` 读取 val_metrics）
2. **使用验证集 RMSE 选择最佳参数**（`val_RMSE`）
3. 测试集指标仅用于最终报告（标注 "FOR REFERENCE ONLY"）
4. 敏感性图中同时绘制 val_RMSE 和 test_RMSE 两条线
5. 结果保存到独立目录 `outputs/hyperparam_search/`

**影响**: 消除测试集泄漏，确保报告的指标无偏。

---

### 问题 #5: 每次实验覆盖结果 → 唯一输出目录

**问题**: `trainer._save_results()` 固定写入 `outputs/test_results.json`，每次运行覆盖前次结果。

**修复**:
1. **新建** `src/utils/experiment_tracker.py` — 实验追踪模块
2. **修改** `src/federated/trainer.py` — 使用 `ExperimentTracker` 管理输出
3. **修改** `config.py` — 添加 `get_run_dir()` 和 `get_git_commit()` 辅助函数
4. **修改** `main.py` — 自动生成唯一目录并保存完整配置

**新目录结构**:
```
outputs/{city}/{method}/seed_{seed}/run_{timestamp}_{git_commit}/
├── config.json          # 完整配置（含命令行参数和默认值）
├── history.json         # 训练损失 + 验证指标
├── metrics.json         # 最终测试指标
├── best_model.pt        # 验证集最优 checkpoint
├── predictions.npz      # 所有客户端预测值和真实值
└── *.png                # 可视化图表
```

**每个 metrics.json 包含的指标**:
```json
{
  "SZH_509": {
    "RMSE": 79.39, "MAE": 73.21,
    "MAPE": 82.01, "MAPE_active": 82.01, "MAPE_raw": 82.01,
    "WAPE": 62.52, "SMAPE": 54.88, "NRMSE": 0.72,
    "RMSE@1h": 82.03, "RMSE@6h": 87.52,
    "RMSE@12h": 87.46, "RMSE@24h": 84.72
  },
  "AVERAGE": { ... }
}
```

---

### 问题 #3: 聚类未使用 FedProx → 簇内独立 FedProx

**问题**: `ClusteredFLServer.aggregate()` 使用普通加权平均，本地训练时的 FedProx 近端参考是全局模型而非簇模型。

**修复**（文件: `src/federated/aggregation.py`）:
1. `ClusteredFLServer` 新增 `mu` 和 `cluster_proximal_params` 属性
2. 每个簇独立聚合后保存簇参数作为 FedProx 近端参考
3. 新增 `get_cluster_proximal_params(client_idx)` 方法
4. `trainer.py` 中聚类模式下传递簇参考参数给 `client.train_local()`
5. 添加 `min_cluster_size` 防止 5 个客户端时 n_clusters=5/8 结果相同

---

### 聚类方法: 模型参数 → 负荷特征（P1）

**问题**: 使用绝对模型参数余弦相似度，公共初始化参数会主导相似度。

**修复**（文件: `src/federated/aggregation.py`）:
1. 新增 `compute_station_features()` — 从训练期提取 24h 日负荷曲线、均值、峰谷比、24/168h 自相关、零值率
2. `cluster_clients()` 支持 `load_features` 参数（负荷特征聚类，推荐）
3. 使用 RBF 核构建亲和矩阵，自动适配特征数
4. 保留旧方法作为 `model_params` fallback
5. 配置项 `cluster_method` 默认 `"load_profile"`

---

### 问题 #4: 缺失基线 → 新增 Centralized + Seasonal Naive

**修复**:
1. **新建** `experiments/baseline_centralized.py`:
   - 合并所有站点训练数据训练单个 TCN-LSTM
   - 每个站点独立测试评估
   - 保存到独立目录
2. **新建** `experiments/baseline_seasonal_naive.py`:
   - 实现 `ŷ(t+h) = y(t+h-168)` 朴素预测
   - 使用数据的 train/val/test 切分和标准化
   - 保存 seasonal_naive_mae 供 MASE 计算
3. **新建** `experiments/fair_comparison.py`:
   - 一键运行所有对比实验（使用相同的站点/切分/种子）
   - 支持 `--seeds 42,123,999` 多种子
   - 汇总表格：均值 ± 标准差

---

### 问题 #7: MAPE 排除低值样本 → 完整指标体系

**问题**: `compute_metrics()` 中 `mask = |y_true| > 1.0` 排除了负荷 ≤1 的样本，实际上是"活跃时段 MAPE"。

**修复**（文件: `src/utils/metrics.py`）:
1. **MAPE** → **MAPE_active**（保留原逻辑，旧代码兼容）
2. 新增标准 **MAPE_raw**（排除 |y|=0 的样本）
3. 新增 **WAPE**: `sum|y-pred| / sum|y| × 100`（推荐用于负荷预测）
4. 新增 **SMAPE**: `mean(2|y-pred| / (|y|+|pred|+ε)) × 100`
5. 新增 **NRMSE**: `RMSE / mean(y)`
6. 新增 **MASE**: `MAE / seasonal_naive_MAE`（需 Seasonal Naive 基线）
7. 新增 **分时段误差**: `RMSE@1h, RMSE@6h, RMSE@12h, RMSE@24h`
8. `evaluate_model()` 新增 `return_predictions=True` 选项

---

### 时序切分修正（问题 #2 相关）

**问题**: `prepare_station_data()` 按行索引比例切割，而非按预测时间划分。

**修复**（文件: `src/data/feature_engineering.py`）:
1. 先在整个序列上构建所有可能窗口
2. 按**预测区间的起始行索引**划分 train/val/test
3. 标准化（StandardScaler）仅用训练窗口数据拟合
4. 新增 `_SubsetChargingDataset` 类支持按索引子集创建 Dataset
5. 验证集窗口可使用训练期历史，但不含未来目标值

---

### 增强特征工程（P1）

**修复**（文件: `src/data/data_loader.py`）:

**新增滞后特征**（`target_lag_24h`, `target_lag_48h`, `target_lag_168h`）:
- 1 天前、2 天前、7 天前的负荷值
- 自动处理边界（开头行填充 NaN → ffill）

**新增滚动统计特征**（窗口 24h / 168h）:
- `target_roll_mean_24h/168h` — 滚动均值
- `target_roll_std_24h/168h` — 滚动标准差
- `target_roll_max_24h/168h` — 滚动最大值

**新增静态站点特征**:
- `charger_num` — 充电桩数量
- `avg_power` — 平均功率
- `perimeter` — 场地周长

**配置开关**: `DataConfig.use_lag_features`, `use_rolling_features`, `use_static_features`

---

### FedBN + 本地预测头（P1）

**修复**:

**FedBN**（文件: `src/models/tcn_lstm.py`, `src/federated/aggregation.py`）:
- `FLServer.aggregate()` 接受 `exclude_param_names` 参数
- 包含 `"bn"` 的参数名不参与联邦聚合
- 每个客户端保留自己的 BatchNorm 统计量
- 命令行: `--fedbn`

**本地预测头**（文件: `src/models/tcn_lstm.py`, `src/federated/aggregation.py`）:
- 最后的 FC 层（`fc.3`）参数不参与聚合
- 每个客户端可以学习站点特定的预测映射
- 与 FedBN 可同时启用
- 命令行: `--local_head`

**本地微调**:
- `FedConfig.finetune_epochs` — 全局训练后对每个客户端微调（默认 0 = 禁用）
- 命令行: `--finetune_epochs 2`

---

### 可视化兼容性修复

**文件**: `src/utils/visualization.py`

- `plot_val_metrics()`: 自动检测 `MAPE` 或 `WAPE` 键名
- `plot_station_comparison()`: 自动适配百分比指标
- `plot_method_comparison()`: 自适应可用指标组合

---

## 三、新增文件清单

| 文件 | 用途 |
|------|------|
| `src/utils/experiment_tracker.py` | 实验追踪模块 — 唯一输出目录、配置保存、预测值导出 |
| `experiments/baseline_centralized.py` | 集中式训练基线 |
| `experiments/baseline_seasonal_naive.py` | 季节性朴素预测基线 |
| `experiments/fair_comparison.py` | 公平对比实验脚本 — 一键运行所有方法 |
| `.gitignore` | Git 忽略规则 |

## 四、修改文件清单

| 文件 | 关键改动 |
|------|---------|
| `config.py` | 新增 `get_run_dir()`, `get_git_commit()`; `ModelConfig` 新增 `use_fedbn`/`use_local_head`; `DataConfig` 新增特征开关; `FedConfig` 新增 `cluster_method`/`min_cluster_size`/`finetune_epochs` |
| `main.py` | 支持 `--seeds` 多种子、`--fedbn`/`--local_head`/`--finetune_epochs` 等命令行参数；自动生成输出目录 |
| `src/data/data_loader.py` | 时间戳合并、训练期选站、增强特征（滞后/滚动/静态）、缺失值标记、`ffill` 填充 |
| `src/data/feature_engineering.py` | 窗口优先的时序切分、`_SubsetChargingDataset`、VMD 注释说明 |
| `src/models/tcn_lstm.py` | 无变动（模型架构不变） |
| `src/federated/aggregation.py` | 簇内 FedProx、`exclude_param_names` 支持、`compute_station_features()` 负荷特征聚类、`min_cluster_size` |
| `src/federated/trainer.py` | `ExperimentTracker` 集成、FedBN/本地头支持、验证集最佳模型追踪、预测值收集、本地微调 |
| `src/utils/metrics.py` | 完整指标体系（WAPE/SMAPE/NRMSE/MASE/分时段）、`return_predictions` 选项 |
| `src/utils/visualization.py` | 兼容 WAPE/MAPE 键名、自适应指标选择 |
| `experiments/baseline_local.py` | 适配新的输出目录结构、训练期选站 |
| `experiments/hyperparam_search.py` | 验证集选参、独立输出目录、标注测试集仅供参考 |
| `experiments/evaluate.py` | 适配新输出目录结构 |

## 五、使用方式

### 基本运行（单次实验）
```bash
# FedProx（推荐）
python main.py --city SZH --aggregation fedprox --top_k 10 --num_rounds 30

# 聚类联邦
python main.py --city SZH --aggregation clustered --top_k 10 --mu 0.01

# FedBN + 本地预测头
python main.py --city SZH --aggregation fedprox --fedbn --local_head

# 多种子运行
python main.py --city SZH --aggregation fedprox --seeds 42,123,999
```

### 公平对比实验
```bash
python experiments/fair_comparison.py --city SZH --seeds 42,123,999 --top_k 10 --rounds 30
```

### 超参数搜索
```bash
python experiments/hyperparam_search.py --city SZH --top_k 5 --num_rounds 10
```

## 六、注意事项

1. **VMD 不在项目标题/简介中出现**，等消融实验确认有效后再加入
2. **MAPE 向后兼容**: `metrics["MAPE"]` 仍然可用（= MAPE_active），推荐使用 `metrics["WAPE"]`
3. **旧输出文件**: `outputs/test_results.json` 和 `outputs/training_history.json` 不再生成，改为新目录结构
4. **超参数搜索**: 旧 `outputs/hyperparam_search.csv` 是基于测试集的结果，已废弃
5. **Git commit 追踪**: 运行目录名包含 `{git_commit}` 哈希，方便追溯代码版本
