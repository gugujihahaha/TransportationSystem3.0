# 实验1：基于深度学习的交通方式识别（仅轨迹特征）

本实验仅使用GeoLife GPS轨迹数据，不涉及知识图谱，通过深度学习模型识别交通方式。

## 目录结构

```
exp1/
├── src/                    # 源代码模块
│   ├── __init__.py
│   ├── data_loader.py      # 数据加载和预处理
│   └── model.py            # 深度学习模型
├── train.py                # 训练脚本
├── evaluate.py             # 评估脚本
├── predict.py              # 预测脚本
├── README.md               # 本文件
└── requirements.txt        # 依赖包
```

## 数据特征

从GPS轨迹中提取的9维特征：

1. **latitude** - 纬度
2. **longitude** - 经度
3. **speed** - 速度 (m/s)
4. **acceleration** - 加速度 (m/s²)
5. **bearing_change** - 方向变化 (度)
6. **distance** - 相邻点距离 (米)
7. **time_diff** - 时间差 (秒)
8. **total_distance** - 累积距离 (米)
9. **total_time** - 累积时间 (秒)

## 模型架构

### TransportationModeClassifier

- **输入**: 轨迹特征序列 (batch_size, seq_len, 9)
- **LSTM编码器**: 双向LSTM，隐藏层维度128，2层
- **特征提取**: 全连接层
- **分类层**: 输出6个类别的logits

## 使用方法

### 重要提示：运行位置

**必须在 `exp1` 目录下运行脚本**，因为使用了相对导入 `from src.xxx import ...`

```bash
cd exp1
```

### 1. 训练模型

```bash
python train.py \
    --geolife_root "../data/Geolife Trajectories 1.3" \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --save_dir checkpoints
```

**参数说明**：

- `--geolife_root`: GeoLife数据根目录（相对于exp1目录）
- `--batch_size`: 批次大小（默认32）
- `--epochs`: 训练轮数（默认50）
- `--lr`: 学习率（默认0.001）
- `--hidden_dim`: LSTM隐藏层维度（默认128）
- `--num_layers`: LSTM层数（默认2）
- `--dropout`: Dropout比率（默认0.3）
- `--max_users`: 最大用户数（可选，用于快速测试）
- `--save_dir`: 模型保存目录（默认checkpoints）

### 2. 评估模型

```bash
python evaluate.py \
    --model_path checkpoints/exp1_model.pth \
    --geolife_root "../data/Geolife Trajectories 1.3" \
    --output_dir results
```

**输出**：

- `evaluation_report.json`: 详细评估报告
- `confusion_matrix.png`: 混淆矩阵图
- `predictions.csv`: 预测结果

### 3. 预测新轨迹

```bash
python predict.py \
    --model_path checkpoints/exp1_model.pth \
    --trajectory_path "../data/Geolife Trajectories 1.3/Data/080/Trajectory/20070627094922.plt"
```

## 预期结果

基于仅轨迹特征的模型，预期准确率约在**75-80%**左右。

**性能限制**：

- 仅依赖运动模式，缺乏地理环境信息
- 某些交通方式（如car和taxi）可能难以区分
- 速度相似的交通方式可能混淆

## 与完整模型对比

| 特征    | 仅轨迹特征  | 轨迹+知识图谱     |
| ----- | ------ | ----------- |
| 准确率   | ~78%   | ~86%        |
| 特征维度  | 9维     | 20维（9+11）   |
| 数据需求  | 仅GPS轨迹 | GPS轨迹+OSM数据 |
| 计算复杂度 | 低      | 中等          |

## 依赖

- torch >= 2.0.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- geopy >= 2.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- tqdm >= 4.65.0

安装：

```bash
pip install -r requirements.txt
```

## 代码结构说明

### 模块导入

代码使用相对导入：

```python
from src.data_loader import GeoLifeDataLoader, preprocess_segments
from src.model import TransportationModeClassifier
```

这意味着：

- 必须在 `exp1` 目录下运行脚本
- `src` 目录必须包含 `__init__.py` 文件（已创建）
- Python会将 `src` 识别为一个包

### 如果遇到导入错误

如果遇到 `ModuleNotFoundError: No module named 'src'`，请确保：

1. 在 `exp1` 目录下运行脚本
2. `src/__init__.py` 文件存在
3. 使用 `python train.py` 而不是 `python -m train`

## 注意事项

1. 确保GeoLife数据路径正确（相对于exp1目录）
2. 首次运行需要加载所有轨迹数据，可能需要一些时间
3. 建议使用GPU训练以加快速度
4. 如果内存不足，可以减少`--max_users`参数限制用户数

## 下一步

- 可以尝试不同的模型架构（如CNN-LSTM混合模型）
- 可以调整超参数（隐藏层维度、LSTM层数等）
- 可以尝试数据增强技术
- 对比加入知识图谱后的性能提升
