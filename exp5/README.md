# Exp5: 数据清洗 + 弱监督上下文增强

## 实验概述

实验五在 Exp4 的基础上，通过两阶段数据清洗和弱监督上下文增强，进一步提升交通方式识别的准确性和鲁棒性。

### 核心改进

1. **两阶段数据清洗**
   - 第一阶段：基础处理（固定序列长度、缺失特征填充）
   - 第二阶段：深度清洗（异常值剔除、轨迹连续性处理、方向修正）

2. **弱监督上下文增强**
   - OSM/天气特征仅用于表示层（encoder）
   - 不直接影响决策层
   - 避免模型对不可靠信号过拟合

3. **GTA-Seg 思想迁移**
   - 不干预决策层
   - 只改善 encoder 表示

## 目录结构

```
exp5/
├── src/
│   ├── __init__.py
│   ├── trajectory_cleaner.py    # 轨迹清洗器
│   └── exp5_adapter.py          # Exp5数据适配器
├── cache/                        # 特征缓存目录
├── checkpoints/                  # 模型保存目录
├── evaluation_results/           # 评估结果目录
├── train.py                      # 训练脚本
├── evaluate.py                   # 评估脚本
└── README.md                     # 本文件
```

## 数据清洗详情

### 第一阶段清洗（基础处理）

- 保持固定序列长度（50）
- 缺失特征用零填充
- 不丢弃任何样本
- 保留原始标签，类别完全一致

### 第二阶段清洗（深度清洗）

#### 1. 异常值剔除
- 速度异常：步行 > 10 m/s，车辆 > 50 m/s
- 加速度异常：超过 10 m/s²
- 删除异常点或用前后均值/中位数平滑

#### 2. 轨迹连续性处理
- 时间戳跳变过大的点进行插值或平滑
- 小段缺失点填充为线性插值

#### 3. 方向/角度异常修正
- 突然改变方向的点，若非合理转弯，做平滑处理

#### 4. 序列长度统一
- 删除或填充后，保证每条轨迹段仍为固定长度（50）

#### 5. 统计过滤
- 删除过短轨迹段或异常比例过高的轨迹
- 保证各类别样本均衡（但不丢弃有效类别）

## 使用方法

### 训练模型

```bash
# 使用 PyCharm 直接运行（推荐）
python exp5/train.py

# 或使用命令行参数
python exp5/train.py --geolife_root ../data/Geolife\ Trajectories\ 1.3 \
                     --osm_path ../data/exp3.geojson \
                     --weather_path ../data/beijing_weather_hourly_2007_2012.csv \
                     --batch_size 32 \
                     --epochs 50 \
                     --lr 5e-5
```

### 评估模型

```bash
python exp5/evaluate.py
```

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `geolife_root` | `../data/Geolife Trajectories 1.3` | GeoLife 数据集路径 |
| `osm_path` | `../data/exp3.geojson` | OSM GeoJSON 路径 |
| `weather_path` | `../data/beijing_weather_hourly_2007_2012.csv` | 天气数据路径 |
| `batch_size` | 32 | 批次大小 |
| `epochs` | 50 | 训练轮数 |
| `lr` | 5e-5 | 学习率 |
| `hidden_dim` | 128 | 隐藏层维度 |
| `num_layers` | 2 | LSTM 层数 |
| `dropout` | 0.3 | Dropout 比例 |
| `max_grad_norm` | 1.0 | 梯度裁剪阈值 |

## 输出文件

### 训练输出

- `checkpoints/exp5_model.pth`: 最佳模型文件
- `cache/processed_features_weather_v5_cleaning.pkl`: 特征缓存
- `cache/cache_meta_cleaning.json`: 缓存元数据（含清洗统计）

### 评估输出

- `evaluation_results/evaluation_report.json`: 评估报告（含清洗统计）
- `evaluation_results/predictions_exp5.csv`: 预测结果
- `evaluation_results/confusion_matrix.png`: 混淆矩阵图
- `evaluation_results/per_class_f1_scores.png`: 各类别 F1 分数图
- `evaluation_results/error_analysis.csv`: 错误分析
- `evaluation_results/cleaning_stats.json`: 清洗统计报告

## 与其他实验的对比

| 实验 | 特征 | 准确率 | 特点 |
|------|------|--------|------|
| Exp1 | 仅轨迹 | 62.82% | 基线模型 |
| Exp2 | 轨迹+KG | 75.05% | 知识图谱增强 |
| Exp3 | 轨迹+增强KG | 72.22% | 增强知识图谱 |
| Exp4 | 轨迹+KG+天气 | 71.17% | 天气数据增强 |
| Exp5 | 轨迹+KG+天气+清洗 | 待测试 | 数据清洗+弱监督 |

## 技术细节

### 轨迹清洗器 (TrajectoryCleaner)

主要方法：
- `clean_segment()`: 清洗单个轨迹段
- `_remove_outliers()`: 剔除异常值
- `_smooth_outliers()`: 平滑异常点
- `_handle_continuity()`: 处理轨迹连续性
- `_smooth_bearing()`: 平滑方向异常
- `normalize_sequence_length()`: 统一序列长度

### Exp5 数据适配器 (Exp5DataAdapter)

继承自 `Exp4DataAdapter`，添加：
- 第二阶段清洗功能
- 清洗统计收集
- 清洗摘要打印

## 注意事项

1. **缓存管理**
   - 首次运行会生成缓存文件
   - 如需重新生成，删除 `cache/` 目录下的文件或使用 `--clear_cache` 参数

2. **数据清洗**
   - 清洗过程会剔除部分异常样本
   - 保留率通常在 85-95% 之间
   - 可通过调整 `TrajectoryCleaner` 参数控制清洗强度

3. **弱监督上下文**
   - OSM/天气特征仅用于表示层
   - 不直接影响决策层
   - 避免对不可靠信号过拟合

4. **Windows 兼容性**
   - 建议设置 `num_workers=0` 避免多进程问题

## 依赖项

- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Scikit-learn >= 1.3.0
- SciPy >= 1.10.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- tqdm >= 4.65.0

## 引用

如果使用本实验代码，请引用：

```
Exp5: 数据清洗 + 弱监督上下文增强
基于多源数据融合的交通方式识别系统
```
