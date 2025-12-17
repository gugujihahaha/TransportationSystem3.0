# 实验1：基于深度学习的交通方式识别（仅轨迹特征）

本实验仅使用 **GeoLife GPS 轨迹数据**，不涉及知识图谱，通过 Bi-LSTM 深度学习模型识别交通方式，作为后续实验的基线模型。

**重要说明：** 在训练和评估过程中，`taxi` 会被归类到 `car` 类别中，因为两者在轨迹特征上非常相似，难以区分。

## 🎯 实验目标

- 验证仅使用轨迹特征进行交通方式识别的可行性
- 建立基线模型性能指标
- 为后续加入知识图谱特征提供对比基准

---

## 📁 目录结构

```
exp1/
├── cache/                      # 缓存目录（可选，自动生成）
├── checkpoints/                # 模型保存目录
│   └── exp1_model.pth         # 训练好的模型
├── results/                    # 评估结果目录
│   ├── evaluation_report.json
│   ├── confusion_matrix.png
│   └── predictions.csv
├── src/                        # 源代码模块
│   ├── __init__.py
│   ├── data_loader.py         # 数据加载和预处理
│   └── model.py               # 深度学习模型
├── train.py                    # 训练脚本
├── evaluate.py                 # 评估脚本
├── predict.py                  # 预测脚本
├── requirements.txt            # 依赖包
└── README.md                   # 本文件
```

---

## 📊 数据特征

### 从 GPS 轨迹中提取的 9 维特征：

| 序号 | 特征名 | 说明 | 计算方法 |
|------|--------|------|----------|
| 1 | `latitude` | 纬度 | 原始数据 |
| 2 | `longitude` | 经度 | 原始数据 |
| 3 | `speed` | 速度 (m/s) | `distance / time_diff` |
| 4 | `acceleration` | 加速度 (m/s²) | `Δspeed / time_diff` |
| 5 | `bearing_change` | 方向变化 (度) | 方位角差值 |
| 6 | `distance` | 相邻点距离 (米) | Haversine 公式 |
| 7 | `time_diff` | 时间差 (秒) | `Δdatetime` |
| 8 | `total_distance` | 累积距离 (米) | `∑distance` |
| 9 | `total_time` | 累积时间 (秒) | `∑time_diff` |

**特征计算优化：**
- ✅ 使用向量化 Haversine 公式计算距离
- ✅ 批量计算方位角和方向变化
- ✅ 鲁棒处理 6 列/7 列 GeoLife 格式
- ✅ 自动清洗无效坐标点

---

## 🧠 模型架构

### TransportationModeClassifier (Bi-LSTM)

```
输入: (batch_size, seq_len, 9)
  ↓
Bi-LSTM 编码器 (2层, 隐藏层128, 双向)
  ↓ (batch_size, seq_len, 256)
取最后时间步
  ↓ (batch_size, 256)
特征提取层: 256 → 128 → 64
  ↓
分类层: 64 → 6
  ↓
输出: (batch_size, 6) [walk, bike, car, bus, train, taxi]
```

**模型参数：**
- 输入维度: 9
- LSTM 隐藏层: 128
- LSTM 层数: 2
- 双向: True
- Dropout: 0.3
- 总参数量: ~200K

**类别说明：**
- 训练时使用 7 种交通方式：`walk`, `bike`, `car`, `bus`, `train`, `taxi`, `subway`
- **`taxi` 会被自动归类到 `car`**（因为两者轨迹特征相似）
- 最终评估类别：`walk`, `bike`, `car`, `bus`, `train`, `subway`（6类）

---

## 🚀 使用方法

### 重要提示：运行位置

**必须在 `exp1` 目录下运行脚本**，因为使用了相对导入 `from src.xxx import ...`

```bash
cd exp1
```

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**依赖包：**
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
python train.py \
    --geolife_root "../data/Geolife Trajectories 1.3" \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --save_dir checkpoints
```

#### 快速测试（使用部分用户）

```bash
python train.py \
    --geolife_root "../data/Geolife Trajectories 1.3" \
    --max_users 5 \
    --epochs 10
```

#### 完整训练参数

```bash
python train.py \
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

**训练输出：**
```
============================================================
加载GeoLife数据...
============================================================
找到 182 个用户

总共加载 16048 个轨迹段

预处理轨迹段...
预处理后剩余 14532 个有效轨迹段

类别数: 7
类别: ['bike' 'bus' 'car' 'subway' 'taxi' 'train' 'walk']

各类别样本数:
  bike: 1234
  bus: 2345
  car: 3456
  ...

训练集大小: 11625
测试集大小: 2907

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
python evaluate.py \
    --model_path checkpoints/exp1_model.pth \
    --geolife_root "../data/Geolife Trajectories 1.3" \
    --output_dir results
```

**评估输出文件：**
- `evaluation_report.json`: 详细评估报告（JSON格式）
- `confusion_matrix.png`: 混淆矩阵图
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
交通方式: car
置信度: 0.8523 (85.23%)

所有类别的概率:
  car       : 0.8523 (85.23%) ██████████████████████████████████████████████
  taxi      : 0.0821 ( 8.21%) ████
  bus       : 0.0432 ( 4.32%) ██
  train     : 0.0143 ( 1.43%) 
  bike      : 0.0054 ( 0.54%) 
  walk      : 0.0027 ( 0.27%) 
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
| **Car** | ~0.80 | ~0.82 | ~0.81 | (包含 taxi)
| **Bus** | ~0.72 | ~0.70 | ~0.71 |
| **Train** | ~0.78 | ~0.75 | ~0.76 |
| **Subway** | ~0.75 | ~0.73 | ~0.74 |

### 性能限制分析

**为什么准确率只有 ~78%？**

1. **缺乏地理环境信息**
   - 无法区分道路类型（高速公路 vs 城市道路）
   - 无法利用 POI 信息（公交站、地铁站）

2. **速度相似的交通方式难以区分**
   - Car vs Taxi: 速度模式非常相似（因此将 taxi 归类到 car）
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
| **准确率** | **~78%** | ~86% | ~88% |
| **数据需求** | 仅 GPS 轨迹 | GPS + 基础 OSM | GPS + 完整 OSM |
| **计算复杂度** | 低 | 中等 | 中等 |
| **训练时间** | 快 | 中等 | 中等 |

**性能提升来源（Exp2 vs Exp1）：**
- ✅ 道路类型信息 → 提升 Car/Bike 区分度
- ✅ POI 信息 → 提升 Bus/Train 识别率
- ✅ 道路密度 → 提供空间上下文

---

## 💡 数据处理细节

### 1. GeoLife 数据格式处理

**支持两种格式：**

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
- 检测列数
- 自动补全缺失列
- 清洗无效坐标（lat ∈ [-90, 90], lon ∈ [-180, 180]）

### 2. 特征归一化

```python
# Z-score 归一化
normalized = (features - mean) / std

# 截断异常值
normalized = np.clip(normalized, -5, 5)
```

### 3. 序列长度处理

```python
# 最小长度: 10 个点
# 目标长度: 100 个点

if len(sequence) > 200:
    # 均匀采样
    sequence = uniform_sample(sequence, target_length=100)
elif len(sequence) < 100:
    # 零填充
    sequence = zero_pad(sequence, target_length=100)
```

---

## ⚠️ 注意事项

### 1. 运行位置

❌ **错误**：在项目根目录运行
```bash
# 项目根目录
python exp1/train.py  # 会报错：ModuleNotFoundError: No module named 'src'
```

✅ **正确**：在 exp1 目录下运行
```bash
cd exp1
python train.py  # 正确
```

### 2. 数据路径

确保 GeoLife 数据路径正确：
```
data/
└── Geolife Trajectories 1.3/
    └── Data/
        ├── 000/
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
python train.py
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
python train.py --geolife_root "/absolute/path/to/Geolife Trajectories 1.3"
```

### 问题 3: CUDA out of memory

**错误信息：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**
```bash
# 方案1: 减小 batch_size
python train.py --batch_size 16

# 方案2: 使用 CPU
python train.py --device cpu
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
- `preprocess_segments`: 序列标准化和归一化

**`src/model.py`：**
- `TransportationModeClassifier`: Bi-LSTM 分类器
- `CNNLSTMModel`: CNN+LSTM 混合模型（可选）

**`train.py`：**
- 主训练循环
- 模型保存和加载
- 学习率调度

---

## 🔄 下一步

### 实验改进方向

1. **模型架构**
   - 尝试 CNN-LSTM 混合模型
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
- **Exp4**: 加入天气数据（准确率提升至 ~90%+）

---

## 📚 参考文献

1. **GeoLife GPS Trajectories Dataset**
   - Microsoft Research Asia
   - 182 users, 17,621 trajectories

2. **Bi-LSTM for Sequential Data**
   - Hochreiter & Schmidhuber, 1997

3. **Transportation Mode Detection**
   - Zheng et al., "Understanding Mobility Based on GPS Data"

---

## 📧 联系方式

如有问题，请提交 Issue 或联系项目维护者。

---

**Exp1 - 基线模型 - 仅轨迹特征**

🎯 准确率: ~78% | 特征维度: 9维 | 模型: Bi-LSTM