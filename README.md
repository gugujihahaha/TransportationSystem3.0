# 基于知识图谱和深度学习的交通方式识别模型

这是一个结合知识图谱和深度学习技术的交通方式识别系统，使用GeoLife GPS轨迹数据和OpenStreetMap (OSM) 数据来识别用户的交通方式（步行、骑行、汽车、公交、地铁、出租车）。

## 项目结构

```
交通方式识别/
├── src/
│   ├── data_preprocessing.py    # 数据预处理模块
│   ├── knowledge_graph.py        # 知识图谱构建模块
│   ├── feature_extraction.py     # 特征提取模块
│   └── model.py                  # 深度学习模型
├── data/
│   ├── Geolife Trajectories 1.3/  # GeoLife轨迹数据
│   └── export.geojson             # OSM数据
├── train.py                      # 训练脚本
├── evaluate.py                   # 评估脚本
├── predict.py                    # 预测脚本
├── config.yaml                   # 配置文件
├── requirements.txt              # 依赖包
└── README.md                     # 项目说明

```

## 功能特点

1. **数据预处理**
   - 加载和解析GeoLife GPS轨迹数据
   - 计算轨迹特征（速度、加速度、方向变化等）
   - 处理OSM地理数据

2. **知识图谱构建**
   - 从OSM数据构建交通知识图谱
   - 包含道路网络、POI（公交站、地铁站、停车场等）
   - 建立道路与POI之间的关联关系

3. **特征提取**
   - 提取轨迹特征（位置、速度、加速度等）
   - 提取知识图谱特征（道路类型、附近POI、道路密度等）
   - 融合多源特征

4. **深度学习模型**
   - 使用双向LSTM编码轨迹特征
   - 使用双向LSTM编码知识图谱特征
   - 特征融合和分类

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备数据

确保数据目录结构如下：
- `data/Geolife Trajectories 1.3/` - GeoLife轨迹数据
- `data/export.geojson` - OSM数据（可通过Overpass API获取）

### 2. 训练模型

```bash
python train.py \
    --geolife_root "data/Geolife Trajectories 1.3" \
    --osm_path "data/export.geojson" \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --save_dir checkpoints
```

参数说明：
- `--geolife_root`: GeoLife数据根目录
- `--osm_path`: OSM数据文件路径
- `--batch_size`: 批次大小
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--max_users`: 最大用户数（可选，用于快速测试）
- `--save_dir`: 模型保存目录

### 3. 评估模型

```bash
python evaluate.py \
    --model_path checkpoints/best_model.pth \
    --geolife_root "data/Geolife Trajectories 1.3" \
    --osm_path "data/export.geojson" \
    --output_dir results
```

### 4. 预测新轨迹

```bash
python predict.py \
    --model_path checkpoints/best_model.pth \
    --trajectory_path "data/Geolife Trajectories 1.3/Data/080/Trajectory/20070627094922.plt" \
    --osm_path "data/export.geojson"
```

## 模型架构

模型采用双流LSTM架构：

1. **轨迹特征流**: 使用双向LSTM处理GPS轨迹特征
   - 输入: 纬度、经度、速度、加速度、方向变化、距离、时间差
   - 输出: 轨迹表示向量

2. **知识图谱特征流**: 使用双向LSTM处理知识图谱特征
   - 输入: 道路类型、POI信息、道路密度等
   - 输出: 知识图谱表示向量

3. **特征融合**: 将两个流的输出融合
   - 拼接两个表示向量
   - 通过全连接层进行融合

4. **分类层**: 输出交通方式类别

## 数据格式

### GeoLife轨迹文件 (.plt)
```
latitude,longitude,altitude,date_days,date,time
39.9751333333333,116.329466666667,0,39260.4092824074,2007-06-27,09:49:22
...
```

### 标签文件 (labels.txt)
```
Start Time	End Time	Transportation Mode
2007/06/27 09:49:22	2007/06/27 10:19:19	bus
...
```

### OSM数据 (GeoJSON)
包含道路网络和POI信息的GeoJSON格式数据。

## 性能指标

模型在测试集上的性能指标包括：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- 混淆矩阵 (Confusion Matrix)

## 注意事项

1. **内存使用**: 处理大量OSM数据时可能需要较大内存
2. **计算资源**: 训练深度学习模型建议使用GPU
3. **数据质量**: 确保轨迹数据和标签数据的时间戳匹配
4. **知识图谱**: OSM数据需要覆盖轨迹数据的区域

## 扩展功能

- [ ] 支持更多交通方式类别
- [ ] 添加注意力机制
- [ ] 支持在线预测
- [ ] 可视化轨迹和预测结果
- [ ] 模型解释性分析

## 参考文献

- GeoLife数据集: https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/
- OpenStreetMap: https://www.openstreetmap.org/

## 许可证

本项目仅供学习和研究使用。



