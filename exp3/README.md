# 实验3：基于增强知识图谱的交通方式识别

本实验在 Exp2 的基础上，使用**完整 OSM 数据**构建增强知识图谱，将 KG 特征从 11 维扩展到 **15 维**，并支持**跨机器训练**（主机生成缓存 → 游戏本 GPU 训练）。

## 🎯 核心改进

### 1. 增强知识图谱特征 (11维 → 15维)

| 特征组 | Exp2 (11维) | Exp3 (15维) | 说明 |
|--------|------------|------------|------|
| **道路类型** | 6维 (one-hot) | 6维 (one-hot) | walk, bike, car, bus, train, unknown |
| **附近POI** | 4维 | **6维** ✨ | 新增：地铁入口、共享单车、出租车点 |
| **道路属性** | - | **2维** ✨ | 新增：速度限制、公交/地铁线路 |
| **道路密度** | 1维 | 1维 | 附近道路节点数量 |

### 2. 完整 OSM 查询

**Exp3 新增的 OSM 实体：**
```overpass
// 地铁站入口
node(area.a)[railway=subway_entrance];

// 共享单车点
node(area.a)[amenity=bicycle_rental];

// 出租车停靠点
node(area.a)[amenity=taxi];

// 速度限制标签
way(area.a)[maxspeed];

// 公交/地铁线路
relation(area.a)[route=bus];
relation(area.a)[route=subway];
```

### 3. 三级缓存 + 版本控制

```
cache/
├── kg_data_v1.pkl              # 知识图谱缓存
├── processed_features_v1.pkl   # 特征缓存
└── cache_meta.json             # 元数据 (版本、哈希)
```

**缓存验证机制：**
- ✅ 版本检查 (`v1`)
- ✅ OSM 文件哈希验证
- ✅ 特征维度验证

### 4. 跨机器训练支持

支持在无 GPU 的主机上生成缓存，然后在有 GPU 的游戏本上训练。

---

## 📁 目录结构

```
exp3/
├── cache/                          # 缓存目录 (自动生成)
│   ├── kg_data_v1.pkl
│   ├── processed_features_v1.pkl
│   └── cache_meta.json
├── checkpoints/                    # 模型保存目录
│   └── exp3_model.pth
├── results/                        # 评估结果目录
│   ├── evaluation_report.json
│   ├── confusion_matrix.png
│   ├── per_class_metrics.png
│   └── predictions.csv
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # 数据加载
│   ├── knowledge_graph_enhanced.py # ✨ 增强知识图谱
│   ├── feature_extraction.py       # 特征提取 (15维KG)
│   └── model.py                    # 双LSTM模型
├── train.py                        # 训练脚本
├── evaluate.py                     # 评估脚本
├── predict.py                      # 预测脚本
├── requirements.txt
└── README.md
```

---

## 🚀 使用方法

### 安装依赖

```bash
cd exp3
pip install -r requirements.txt
```

### 方式一：直接训练（单机）

```bash
# 在有 GPU 的机器上直接训练
python train.py \
    --geolife_root "../data/Geolife Trajectories 1.3" \
    --osm_path "../data/beijing_complete.geojson" \
    --batch_size 32 \
    --epochs 50 \
    --device cuda
```

### 方式二：跨机器训练（推荐）

#### 步骤1：在主机上生成缓存（无需 GPU）

```bash
# 主机命令
cd exp3
python train.py \
    --geolife_root "../data/Geolife Trajectories 1.3" \
    --osm_path "../data/beijing_complete.geojson" \
    --generate_cache_only \
    --max_users 10  # 快速测试，去掉此参数使用全部数据
```

**输出：**
```
✓ 缓存生成完成！
缓存文件位置: cache/
  - 知识图谱: cache/kg_data_v1.pkl
  - 特征数据: cache/processed_features_v1.pkl
  - 元数据: cache/cache_meta.json
```

#### 步骤2：打包并传输缓存到游戏本

```bash
# 主机：打包缓存
tar -czf exp3_cache.tar.gz cache/

# 传输到游戏本 (使用 scp 或 U盘)
scp exp3_cache.tar.gz user@gaming-pc:/path/to/exp3/

# 游戏本：解压缓存
cd exp3
tar -xzf exp3_cache.tar.gz
```

#### 步骤3：在游戏本上训练（使用 GPU）

```bash
# 游戏本命令
cd exp3
python train.py \
    --use_cached_data \
    --device cuda \
    --batch_size 32 \
    --epochs 50
```

---

## 📊 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--geolife_root` | `../data/Geolife Trajectories 1.3` | GeoLife 数据路径 |
| `--osm_path` | `../data/beijing_complete.geojson` | 完整 OSM 数据路径 |
| `--max_users` | `None` | 最大用户数（测试用） |
| `--batch_size` | `32` | 批次大小 |
| `--epochs` | `50` | 训练轮数 |
| `--lr` | `0.001` | 学习率 |
| `--hidden_dim` | `128` | LSTM 隐藏层维度 |
| `--num_layers` | `2` | LSTM 层数 |
| `--dropout` | `0.3` | Dropout 比率 |
| `--device` | `cuda/cpu` | 训练设备 |
| `--generate_cache_only` | `False` | 仅生成缓存（主机模式） |
| `--use_cached_data` | `False` | 使用缓存数据（游戏本模式） |

---

## 🧪 评估模型

```bash
python evaluate.py \
    --model_path checkpoints/exp3_model.pth \
    --geolife_root "../data/Geolife Trajectories 1.3" \
    --osm_path "../data/beijing_complete.geojson" \
    --output_dir results
```

**输出文件：**
- `evaluation_report.json`: 详细分类报告
- `confusion_matrix.png`: 混淆矩阵
- `per_class_metrics.png`: 各类别性能指标
- `predictions.csv`: 预测结果
- `evaluation_summary.json`: 评估摘要

---

## 🔮 预测新轨迹

```bash
python predict.py \
    --model_path checkpoints/exp3_model.pth \
    --trajectory_path "../data/Geolife Trajectories 1.3/Data/080/Trajectory/20070627094922.plt" \
    --osm_path "../data/beijing_complete.geojson"
```

**输出示例：**
```
预测结果
============================================================

交通方式: car
置信度: 0.8532 (85.32%)

所有类别的概率:
------------------------------------------------------------
🥇 car       : 0.8532 (85.32%) ██████████████████████████████████████████████
🥈 taxi      : 0.0821 (8.21%)  ████
🥉 bus       : 0.0452 (4.52%)  ██
4. train     : 0.0123 (1.23%)  
5. bike      : 0.0051 (0.51%)  
6. walk      : 0.0021 (0.21%)  
============================================================
```

---

## 📈 预期结果

### 性能对比

| 实验 | 特征维度 | 准确率 | 说明 |
|------|---------|--------|------|
| Exp1 | 9维 (仅轨迹) | ~78% | 基线模型 |
| Exp2 | 20维 (轨迹9 + KG11) | ~86% | 基础 OSM |
| **Exp3** | **24维 (轨迹9 + KG15)** | **~88%** ✨ | 完整 OSM + 增强 KG |

### 各类别性能提升

预计在以下类别有显著提升：
- **train**: 地铁入口特征 ✨
- **bike**: 共享单车点特征 ✨
- **taxi**: 出租车停靠点特征 ✨
- **bus**: 公交线路特征 ✨

---

## ⚠️ 注意事项

### 1. 缓存管理

- 首次运行生成缓存可能需要 **30-60 分钟**
- 缓存文件较大（~500MB-2GB），确保磁盘空间充足
- 如果 OSM 数据更新，需要删除旧缓存重新生成

```bash
# 删除缓存
rm -rf cache/
```

### 2. OSM 数据准备

确保使用完整的 OSM 查询（包含 Exp3 新增实体）：

```bash
# 下载完整 OSM 数据
# 使用 Exp3 的 Overpass 查询脚本
# 保存为 beijing_complete.geojson
```

### 3. 内存要求

- **主机（生成缓存）**: 至少 16GB RAM
- **游戏本（训练）**: 至少 8GB RAM + 4GB VRAM

---

## 🔧 故障排除

### 问题1: 缓存验证失败

```
⚠️  OSM 文件已更改，缓存失效
```

**解决方案：**
```bash
rm -rf cache/
python train.py --generate_cache_only
```

### 问题2: 特征维度不匹配

```
ValueError: KG 特征维度错误：预期 15 维，实际 11 维
```

**解决方案：** 确保使用 Exp3 的完整 OSM 数据，而不是 Exp2 的基础数据。

### 问题3: 内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案：**
```bash
# 减小 batch_size
python train.py --batch_size 16 --use_cached_data --device cuda
```

---

## 🎓 与 Exp4 的关系

Exp3 的缓存可以**直接被 Exp4 继承**，Exp4 只需添加天气特征：

```python
# Exp4 特征组成
- 轨迹特征: 9维 (继承 Exp3)
- 增强 KG 特征: 15维 (继承 Exp3)
- 天气特征: 5维 (新增)
- 总计: 29维
```

---

## 📚 参考文献

- GeoLife GPS 轨迹数据集
- OpenStreetMap (OSM) 完整数据
- Bi-LSTM 特征融合模型

---

## 📧 联系方式

如有问题，请提交 Issue 或联系项目维护者。

---

**Exp3 完整实现 - 增强知识图谱 + 跨机器训练**