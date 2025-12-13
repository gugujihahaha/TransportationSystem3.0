"""
训练脚本
- 确保在 GeoLife 加载和模型训练阶段显示实时进度。
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import json
from tqdm import tqdm

# 导入自定义模块
from src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader, preprocess_trajectory_segments
from src.knowledge_graph import TransportationKnowledgeGraph
from src.feature_extraction import FeatureExtractor
# 假设您的模型文件名为 src/model.py
from src.model import TransportationModeClassifier


class TrajectoryDataset(Dataset):
    """轨迹数据集"""

    def __init__(self, segments, feature_extractor, label_encoder):
        self.segments = segments
        self.feature_extractor = feature_extractor
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        trajectory, label = self.segments[idx]

        # 提取特征
        trajectory_features, kg_features = self.feature_extractor.extract_features(trajectory)

        # 转换为tensor
        trajectory_tensor = torch.FloatTensor(trajectory_features)
        kg_tensor = torch.FloatTensor(kg_features)

        # 编码标签
        label_encoded = self.label_encoder.transform([label])[0]
        label_tensor = torch.LongTensor([label_encoded])[0]

        return trajectory_tensor, kg_tensor, label_tensor


def load_data(geolife_root: str, osm_path: str, max_users: int = None):
    """加载所有数据，并确保每个阶段都有进度提示"""

    print("\n================ 阶段 1: 数据加载与知识图谱构建 ==================")
    print(f"1.1 正在初始化 GeoLife 数据加载器: {geolife_root}")
    geolife_loader = GeoLifeDataLoader(geolife_root)

    # 获取用户列表
    users = geolife_loader.get_all_users()
    if max_users:
        users = users[:max_users]

    print(f" -> 找到 {len(users)} 个 GeoLife 用户（将处理其中 {len(users)} 个）")

    # 加载OSM数据并构建知识图谱
    print("\n1.2 正在加载 OSM 数据...")
    osm_loader = OSMDataLoader(osm_path)
    osm_data = osm_loader.load_osm_data()

    road_network = osm_loader.extract_road_network(osm_data)
    pois = osm_loader.extract_pois(osm_data)
    print(f" -> OSM 数据提取完成: {len(road_network)} 条道路, {len(pois)} 个POI。")

    # KG构建和关联阶段 (进度条已在 knowledge_graph.py 内部实现)
    print("\n1.3 正在构建交通知识图谱 (此阶段已优化速度)...")
    kg = TransportationKnowledgeGraph()
    kg.build_from_osm(road_network, pois)
    # 确保打印出统计信息，证明图构建已经结束
    print(f" -> 知识图谱构建阶段已完成。统计: {kg.get_graph_statistics()}")


    print("\n================ 阶段 2: 轨迹加载、分段与特征提取 ==================")

    # 加载所有轨迹段 (使用 tqdm 实时显示进度)
    print("2.1 正在加载 GeoLife 轨迹段 (文件 I/O 密集)...")
    all_segments = []

    # 使用 tqdm 包装 GeoLife 用户循环，实时显示用户加载进度
    for user_id in tqdm(users, desc="[GeoLife用户加载]"):
        labels = geolife_loader.load_labels(user_id)
        if labels.empty:
            continue

        trajectory_dir = os.path.join(geolife_root, f"Data/{user_id}/Trajectory")
        if not os.path.exists(trajectory_dir):
            continue

        # 内部循环（加载单个用户的轨迹文件）
        for traj_file in os.listdir(trajectory_dir):
            if not traj_file.endswith('.plt'):
                continue

            traj_path = os.path.join(trajectory_dir, traj_file)
            try:
                trajectory = geolife_loader.load_trajectory(traj_path)
                segments = geolife_loader.segment_trajectory(trajectory, labels)
                all_segments.extend(segments)
            except Exception as e:
                # 打印警告而不是中断
                print(f"\n[WARN] 加载轨迹文件失败 {traj_path}: {e}")
                continue

    print(f" -> GeoLife 轨迹加载完成。总共加载 {len(all_segments)} 个原始轨迹段。")

    # 预处理轨迹段
    print("\n2.2 正在预处理轨迹段...")
    processed_segments = preprocess_trajectory_segments(all_segments, min_length=10)
    print(f" -> 预处理完成。剩余 {len(processed_segments)} 个可用轨迹段。")

    return processed_segments, kg


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # 训练批次的 tqdm
    for trajectory_features, kg_features, labels in tqdm(dataloader, desc="   [训练批次]"):
        trajectory_features = trajectory_features.to(device)
        kg_features = kg_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(trajectory_features, kg_features)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, label_encoder):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # 评估批次的 tqdm
        for trajectory_features, kg_features, labels in tqdm(dataloader, desc="   [评估批次]"):
            trajectory_features = trajectory_features.to(device)
            kg_features = kg_features.to(device)
            labels = labels.to(device)

            logits = model(trajectory_features, kg_features)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # 分类报告
    class_names = label_encoder.classes_
    report = classification_report(all_labels, all_preds,
                                  target_names=class_names,
                                  output_dict=True)

    return avg_loss, report, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='训练交通方式识别模型')
    parser.add_argument('--geolife_root', type=str,
                       default='data/Geolife Trajectories 1.3',
                       help='GeoLife数据根目录')
    parser.add_argument('--osm_path', type=str,
                       default='data/export.geojson',
                       help='OSM数据路径')
    parser.add_argument('--max_users', type=int, default=None,
                       help='最大用户数（用于快速测试）')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout比率')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')

    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载数据
    segments, kg = load_data(args.geolife_root, args.osm_path, args.max_users)

    if len(segments) == 0:
        print("错误: 没有加载到任何数据，请检查数据路径和预处理条件。")
        return

    # 创建特征提取器
    print("\n================ 阶段 3: 模型初始化与训练 ==================")
    feature_extractor = FeatureExtractor(kg)

    # 准备标签编码器
    all_labels = [label for _, label in segments]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    num_classes = len(label_encoder.classes_)
    print(f" -> 类别数: {num_classes}")
    print(f" -> 类别: {label_encoder.classes_}")

    # 创建数据集
    dataset = TrajectoryDataset(segments, feature_extractor, label_encoder)

    # 划分训练集和测试集
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=[segments[i][1] for i in range(len(segments))]
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f" -> 训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    # 创建模型
    model = TransportationModeClassifier(
        trajectory_feature_dim=7,
        kg_feature_dim=11,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(args.device)

    print(f" -> 模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f" -> 训练设备: {args.device}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 训练循环
    best_test_loss = float('inf')

    print(f"\n================ 阶段 4: 模型训练 (共 {args.epochs} 轮) ==================")
    for epoch in range(args.epochs):
        print(f"\n[EPOCH {epoch+1}/{args.epochs}] 开始训练...")

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)
        print(f"   [结果] 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")

        # 评估
        test_loss, test_report, test_preds, test_labels = evaluate(
            model, test_loader, criterion, args.device, label_encoder
        )
        test_acc = test_report['accuracy']
        print(f"   [结果] 测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")

        # 学习率调度
        scheduler.step(test_loss)

        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
                'test_acc': test_acc,
                'label_encoder': label_encoder,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print("   [INFO] 测试损失降低，保存最佳模型。")

        # 打印分类报告
        if (epoch + 1) % 10 == 0:
            print("\n   [报告] 当前分类报告:")
            print(classification_report(test_labels, test_preds,
                                      target_names=label_encoder.classes_))

    print("\n================ 训练完成! ==================")
    print(f"最终最佳测试损失: {best_test_loss:.4f}")
    print(f"最佳模型已保存到 {os.path.join(args.save_dir, 'best_model.pth')}")


if __name__ == '__main__':
    main()

# python train.py --geolife_root "data/Geolife Trajectories 1.3" --osm_path "data/export.geojson"