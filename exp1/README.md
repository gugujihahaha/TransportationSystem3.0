# 实验1：基于深度学习的交通方式识别（仅轨迹特征）

## 📋 实验概述

本实验是基线模型，**仅使用 GeoLife GPS 轨迹数据**，不涉及知识图谱，通过 **Bi-LSTM 深度学习模型**识别交通方式。作为后续实验（Exp2、Exp3）的对比基准，验证仅使用轨迹特征进行交通方式识别的可行性。

**核心特点：**
- ✅ 纯轨迹特征（9维）
- ✅ 单输入 Bi-LSTM 模型
- ✅ 标签自动合并（taxi → car & taxi）
- ✅ 向量化特征计算优化
- ✅ 鲁棒的数据格式处理

---

## 🎯 实验目标

1. **建立基线性能指标**：验证仅使用轨迹特征进行交通方式识别的可行性
2. **为后续实验提供对比基准**：为加入知识图谱特征（Exp2、Exp3）提供性能对比
3. **验证模型架构**：验证 Bi-LSTM 在序列分类任务中的有效性

---

## 📁 目录结构

```
exp1/
├── cache/                      # 缓存目录（自动生成）
├── checkpoints/                # 模型保存目录
│   └── exp1_model.pth         # 训练好的模型
├── results/                    # 评估结果目录
│   └── exp1/
│       ├── evaluation_report.json
│       ├── confusion_matrix.png
│       ├── per_class_metrics.png
│       └── predictions.csv
├── src/                        # 源代码模块
│   ├── __init__.py
│   ├── data_loader.py         # 数据加载和预处理（核心）
│   └── model.py               # 深度学习模型（Bi-LSTM）
├── train.py                    # 训练脚本
├── evaluate.py                 # 评估脚本
├── predict.py                  # 预测脚本
├── requirements.txt            # 依赖包
└── README.md                   # 本文件
```

---

## 📊 数据特征

### 从 GPS 轨迹中提取的 9 维特征

| 序号 | 特征名 | 维度 | 说明 | 计算方法 |
|------|--------|------|------|----------|
| 1 | `latitude` | 1 | 纬度 | 原始数据 |
| 2 | `longitude` | 1 | 经度 | 原始数据 |
| 3 | `speed` | 1 | 速度 (m/s) | `distance / time_diff` |
| 4 | `acceleration` | 1 | 加速度 (m/s²) | `Δspeed / time_diff` |
| 5 | `bearing_change` | 1 | 方向变化 (度) | 方位角差值（处理360度跨越） |
| 6 | `distance` | 1 | 相邻点距离 (米) | Haversine 公式（向量化） |
| 7 | `time_diff` | 1 | 时间差 (秒) | `Δdatetime` |
| 8 | `total_distance` | 1 | 累积距离 (米) | `∑distance`（从起点累计） |
| 9 | `total_time` | 1 | 累积时间 (秒) | `∑time_diff`（从起点累计） |

**特征计算优化：**
- ✅ **向量化 Haversine 公式**：批量计算距离，避免循环
- ✅ **批量计算方位角**：使用 `np.arctan2` 向量化计算
- ✅ **鲁棒处理 6 列/7 列 GeoLife 格式**：自动检测并标准化
- ✅ **自动清洗无效坐标点**：删除 lat ∈ [-90, 90], lon ∈ [-180, 180] 范围外的点

---

## 🏷️ 类别标签处理

### 标签重映射机制

**核心逻辑：** 在 `src/data_loader.py` 的 `preprocess_segments()` 函数中，自动将 `taxi` 归类到 `car & taxi`。

```python
# 标签映射规则
MAPPING = {
    'taxi': 'car',  # taxi → car
}
NEW_CLASS_NAME = 'car & taxi'  # 最终标签名称
```

**最终类别（6类）：**
- `walk`
- `bike`
- `car & taxi`（包含原始 `car` 和 `taxi`）
- `bus`
- `train`
- `subway`

**为什么合并 taxi 和 car？**
- 两者在轨迹特征上非常相似（速度、加速度模式接近）
- 仅使用轨迹特征难以区分
- 合并后提高模型训练稳定性

---

## 🧠 模型架构

### TransportationModeClassifier (Bi-LSTM)

```
输入: (batch_size, seq_len=100, input_dim=9)
  ↓
Bi-LSTM 编码器
  - 隐藏层维度: 128
  - 层数: 2
  - 双向: True
  - Dropout: 0.3
  ↓ (batch_size, seq_len, 256)  # 128 * 2 (双向)
取最后时间步
  ↓ (batch_size, 256)
特征提取层
  - Linear(256 → 128) + ReLU + Dropout
  - Linear(128 → 64) + ReLU + Dropout
  ↓ (batch_size, 64)
分类层
  - Linear(64 → 6)
  ↓
输出: (batch_size, 6) [walk, bike, car & taxi, bus, train, subway]
```

**模型参数：**
- **输入维度**: 9（轨迹特征）
- **LSTM 隐藏层**: 128
- **LSTM 层数**: 2
- **双向**: True
- **Dropout**: 0.3
- **总参数量**: ~200K

---

## 🔄 数据处理流程

### 1. 轨迹加载（`src/data_loader.py`）

**支持两种 GeoLife 格式：**

**7 列格式（标准）：**
```
Latitude,Longitude,Reserved,Altitude,Date_days,Date,Time
39.984702,116.318417,0,492,39744.1201,2008-10-23,02:52:58
```

**6 列格式（非标准）：**
```
Latitude,Longitude,Altitude,Date_days,Date,Time
39.984702,116.318417,492,39744.1201,2008-10-23,02:52:58
```

**自动处理逻辑：**
- 检测列数（6 或 7）
- 自动补全缺失列（Reserved 列填充 0）
- 清洗无效坐标（lat ∈ [-90, 90], lon ∈ [-180, 180]）

### 2. 序列预处理（`preprocess_segments()`）

**序列长度规范化：**
- **最小长度**: 10 个点（过滤太短的轨迹）
- **目标长度**: 100 个点（固定序列长度）
- **最大长度**: 200 个点（超过则均匀采样）

**处理策略：**
```python
if L > 200:
    # 均匀采样到 100
    indices = np.linspace(0, L-1, 100, dtype=int)
elif L > 100:
    # 随机裁剪到 100
    start_index = np.random.randint(0, L - 100 + 1)
    features = features[start_index:start_index + 100]
elif L < 100:
    # 零填充到 100
    padding = np.zeros((100 - L, 9))
    features = np.vstack([features, padding])
```

**标签重映射：**
- 自动将 `taxi` → `car` → `car & taxi`
- 确保训练和评估使用一致的标签

### 3. 特征归一化（`train.py`）

**全局 Z-score 归一化：**
```python
# 计算训练集的全局均值和标准差
mean = features.mean(axis=(0, 1))  # (9,)
std = features.std(axis=(0, 1))     # (9,)

# 归一化
normalized = (features - mean) / (std + 1e-8)

# 截断异常值
normalized = np.clip(normalized, -5, 5)
```

---

## 🚀 使用方法

### ⚠️ 重要提示：运行位置

**必须在 `exp1` 目录下运行脚本**，因为使用了相对导入 `from src.xxx import ...`

```bash
cd exp1
```

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**依赖包列表：**
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
geopy>=2.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

### 2. 训练模型

#### 基础训练（使用默认参数）

```bash
python train_maso.py \
    --geolife_root "../data/Geolife Trajectories 1.3" \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --save_dir checkpoints
```

#### 快速测试（使用部分用户）

```bash
python train_maso.py \
    --geolife_root "../data/Geolife Trajectories 1.3" \
    --max_users 5 \
    --epochs 10
```

#### 完整训练参数

```bash
python train_maso.py \
    --geolife_root "../data/Geolife Trajectories 1.3" \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --hidden_dim 128 \
    --num_layers 2 \
    --dropout 0.3 \
    --save_dir checkpoints \
    --device cuda  # 或 cpu
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--geolife_root` | `../data/Geolife Trajectories 1.3` | GeoLife 数据根目录 |
| `--max_users` | `None` | 最大用户数（用于快速测试） |
| `--batch_size` | `32` | 批次大小 |
| `--epochs` | `50` | 训练轮数 |
| `--lr` | `0.001` | 学习率 |
| `--hidden_dim` | `128` | LSTM 隐藏层维度 |
| `--num_layers` | `2` | LSTM 层数 |
| `--dropout` | `0.3` | Dropout 比率 |
| `--save_dir` | `checkpoints` | 模型保存目录 |
| `--device` | `cuda/cpu` | 训练设备 |

**训练输出示例：**
```
============================================================
加载GeoLife数据...
============================================================
找到 182 个用户

总共加载 16048 个轨迹段

预处理轨迹段...
标签重映射完成: 'taxi' 已并入 'car & taxi'
预处理后剩余 14532 个有效轨迹段

过滤稀疏类别：仅保留主要 6 种交通方式
原始轨迹段总数: 14532
保留轨迹段总数: 14234 (移除 298 个稀疏类别)

类别数: 6
类别: ['bike' 'bus' 'car & taxi' 'subway' 'train' 'walk']

各类别样本数:
  bike: 1234
  bus: 2345
  car & taxi: 3456
  subway: 1234
  train: 2345
  walk: 3620

训练集大小: 11387
测试集大小: 2847

模型参数数量: 201,094
使用设备: cuda

============================================================
开始训练...
============================================================
Epoch 1/50
训练损失: 1.2345, 训练准确率: 0.5432
测试损失: 1.1234, 测试准确率: 0.6123
✓ 保存最佳模型
...
```

### 3. 评估模型

```bash
python evaluate_maso.py \
    --model_path checkpoints/exp1_model.pth \
    --geolife_root "../data/Geolife Trajectories 1.3" \
    --output_dir results/exp1 \
    --batch_size 32
```

**评估输出文件：**
- `evaluation_report.json`: 详细评估报告（JSON格式）
- `confusion_matrix.png`: 混淆矩阵图
- `per_class_metrics.png`: 各类别性能指标图
- `predictions.csv`: 预测结果（包含置信度）

**评估报告示例：**
```json
{
  "accuracy": 0.7812,
  "macro avg": {
    "precision": 0.7654,
    "recall": 0.7543,
    "f1-score": 0.7598
  },
  "walk": {
    "precision": 0.8234,
    "recall": 0.8567,
    "f1-score": 0.8397,
    "support": 1234
  },
  ...
}
```

### 4. 预测新轨迹

```bash
python predict.py \
    --model_path checkpoints/exp1_model.pth \
    --trajectory_path "../data/Geolife Trajectories 1.3/Data/080/Trajectory/20070627094922.plt"
```

**预测输出示例：**
```
============================================================
预测结果
============================================================

交通方式: car & taxi
置信度: 0.8523 (85.23%)

所有类别的概率:
🥇 car & taxi: 0.8523 (85.23%) ██████████████████████████████████████████████
🥈 bus       : 0.0821 ( 8.21%) ████
🥉 train     : 0.0432 ( 4.32%) ██
4. walk      : 0.0143 ( 1.43%) 
5. bike      : 0.0054 ( 0.54%) 
6. subway    : 0.0027 ( 0.27%) 
```

---

## 📈 预期结果

### 性能指标

| 指标 | 预期值 | 说明 |
|------|--------|------|
| **总体准确率** | **~78%** | 基线模型性能 |
| Macro F1 | ~0.76 | 平均 F1 分数 |
| Weighted F1 | ~0.78 | 加权 F1 分数 |

### 各类别性能

| 交通方式 | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| **Walk** | ~0.82 | ~0.86 | ~0.84 |
| **Bike** | ~0.75 | ~0.73 | ~0.74 |
| **Car & Taxi** | ~0.80 | ~0.82 | ~0.81 |
| **Bus** | ~0.72 | ~0.70 | ~0.71 |
| **Train** | ~0.78 | ~0.75 | ~0.76 |
| **Subway** | ~0.75 | ~0.73 | ~0.74 |

### 性能限制分析

**为什么准确率只有 ~78%？**

1. **缺乏地理环境信息**
   - 无法区分道路类型（高速公路 vs 城市道路）
   - 无法利用 POI 信息（公交站、地铁站）

2. **速度相似的交通方式难以区分**
   - Car vs Taxi: 速度模式非常相似（因此合并为 car & taxi）
   - Bus vs Car: 在某些路段速度接近

3. **仅依赖运动模式**
   - 缺少外部知识（如公交线路、地铁网络）
   - 无法利用空间上下文信息

---

## 🔬 实验对比

### Exp1 vs Exp2 vs Exp3

| 特征 | Exp1 | Exp2 | Exp3 |
|------|------|------|------|
| **轨迹特征** | 9维 | 9维 | 9维 |
| **KG特征** | - | 11维 | 15维 |
| **总特征维度** | **9维** | 20维 | 24维 |
| **序列长度** | 100 | 50 | 50 |
| **模型类型** | 单输入 Bi-LSTM | 双输入 Bi-LSTM | 双输入 Bi-LSTM |
| **准确率** | **~78%** | ~86% | ~88% |
| **数据需求** | 仅 GPS 轨迹 | GPS + 基础 OSM | GPS + 完整 OSM |
| **计算复杂度** | 低 | 中等 | 中等 |
| **训练时间** | 快 | 中等 | 中等 |

**性能提升来源（Exp2 vs Exp1）：**
- ✅ 道路类型信息 → 提升 Car/Bike 区分度
- ✅ POI 信息 → 提升 Bus/Train 识别率
- ✅ 道路密度 → 提供空间上下文

---

## 💡 技术特色

### 1. 向量化特征计算

**传统方法（循环）：**
```python
for i in range(len(df)):
    distance[i] = haversine(lat[i-1], lon[i-1], lat[i], lon[i])
```

**优化方法（向量化）：**
```python
# 批量计算所有点对的距离
lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
distances = haversine_vectorized(lat1_rad, lon1_rad, lat2_rad, lon2_rad)
```

**性能提升：** 10-100倍加速

### 2. 鲁棒的数据格式处理

- 自动检测 6 列/7 列格式
- 自动补全缺失列
- 清洗无效坐标点
- 处理时间格式异常

### 3. 标签自动合并

- 在数据预处理阶段自动合并 `taxi` → `car & taxi`
- 确保训练和评估使用一致的标签
- 避免类别不平衡问题

---

## ⚠️ 注意事项

### 1. 运行位置

❌ **错误**：在项目根目录运行
```bash
# 项目根目录
python exp1/train_maso.py  # 会报错：ModuleNotFoundError: No module named 'src'
```

✅ **正确**：在 exp1 目录下运行
```bash
cd exp1
python train_maso.py  # 正确
```

### 2. 数据路径

确保 GeoLife 数据路径正确：
```
data/
└── Geolife Trajectories 1.3/
    └── Data/
        ├── 000/
        │   ├── Trajectory/
        │   │   ├── 20081023025304.plt
        │   │   └── ...
        │   └── labels.txt
        ├── 001/
        ...
```

### 3. 内存要求

- **最小内存**: 8GB RAM
- **推荐内存**: 16GB RAM
- **GPU 显存**: 4GB（可选，可使用 CPU 训练）

### 4. 训练时间

| 配置 | 时间 |
|------|------|
| CPU (i7) | ~2-3 小时 |
| GPU (GTX 1060) | ~30-45 分钟 |
| GPU (RTX 3080) | ~15-20 分钟 |

---

## 🐛 故障排除

### 问题 1: ModuleNotFoundError

**错误信息：**
```
ModuleNotFoundError: No module named 'src'
```

**解决方案：**
```bash
# 确保在 exp1 目录下运行
cd exp1
python train_maso.py
```

### 问题 2: 数据加载失败

**错误信息：**
```
错误：未找到 GeoLife 数据目录
```

**解决方案：**
```bash
# 检查数据路径
ls -la "../data/Geolife Trajectories 1.3/Data"

# 或使用绝对路径
python train_maso.py --geolife_root "/absolute/path/to/Geolife Trajectories 1.3"
```

### 问题 3: CUDA out of memory

**错误信息：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**
```bash
# 方案1: 减小 batch_size
python train_maso.py --batch_size 16

# 方案2: 使用 CPU
python train_maso.py --device cpu
```

### 问题 4: 无效坐标警告

**警告信息：**
```
警告: 文件 xxx.plt 发现无效坐标，正在删除。
```

**说明：** 这是正常的数据清洗过程，不影响训练。

---

## 📝 代码结构说明

### 模块导入

代码使用相对导入：
```python
from src.data_loader import GeoLifeDataLoader, preprocess_segments
from src.model import TransportationModeClassifier
```

**这意味着：**
- 必须在 `exp1` 目录下运行脚本
- `src` 目录必须包含 `__init__.py` 文件（已创建）
- Python 会将 `src` 识别为一个包

### 关键文件说明

**`src/data_loader.py`：**
- `GeoLifeDataLoader`: 加载和预处理 GeoLife 数据
- `preprocess_segments`: 序列标准化、长度规范化、标签重映射

**`src/model.py`：**
- `TransportationModeClassifier`: Bi-LSTM 分类器（主模型）
- `CNNLSTMModel`: CNN+LSTM 混合模型（可选，未使用）

**`train.py`：**
- 主训练循环
- 全局特征归一化
- 模型保存和加载
- 学习率调度

---

## 🔄 下一步

### 实验改进方向

1. **模型架构**
   - 尝试 CNN-LSTM 混合模型（已实现但未使用）
   - 添加注意力机制
   - 使用 Transformer 编码器

2. **数据增强**
   - 轨迹插值
   - 随机裁剪
   - 噪声注入

3. **超参数优化**
   - 调整 LSTM 层数
   - 调整隐藏层维度
   - 尝试不同的学习率策略

### 进阶实验

- **Exp2**: 加入基础 OSM 知识图谱（准确率提升至 ~86%）
- **Exp3**: 使用完整 OSM 知识图谱（准确率提升至 ~88%）

---

## 📚 参考文献

1. **GeoLife GPS Trajectories Dataset**
   - Microsoft Research Asia
   - 182 users, 17,621 trajectories
   - [数据集链接](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)

2. **Bi-LSTM for Sequential Data**
   - Hochreiter & Schmidhuber, 1997
   - Long Short-Term Memory

3. **Transportation Mode Detection**
   - Zheng et al., "Understanding Mobility Based on GPS Data"

---

## 📧 联系方式

如有问题，请提交 Issue 或联系项目维护者。

---

**Exp1 - 基线模型 - 仅轨迹特征**

🎯 准确率: ~78% | 特征维度: 9维 | 模型: Bi-LSTM | 序列长度: 100
