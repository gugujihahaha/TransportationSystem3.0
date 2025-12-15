"""
训练脚本 - 仅使用GeoLife轨迹数据
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
from collections import Counter # 导入 Counter 用于统计

# 导入自定义模块
from src.data_loader import GeoLifeDataLoader, preprocess_segments
from src.model import TransportationModeClassifier

# 定义特征维度常量 (与 data_loader.py 提取的 9 维轨迹特征一致)
TRAJECTORY_FEATURE_DIM = 9


class TrajectoryDataset(Dataset):
    """轨迹数据集"""

    def __init__(self, segments, label_encoder):
        self.segments = segments
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        features, label = self.segments[idx]

        # 转换为tensor
        features_tensor = torch.FloatTensor(features)

        # 编码标签
        label_encoded = self.label_encoder.transform([label])[0]
        label_tensor = torch.LongTensor([label_encoded])

        return features_tensor, label_tensor


def load_data(geolife_root: str, max_users: int = None):
    """加载所有数据"""
    print("=" * 60)
    print("加载GeoLife数据...")
    print("=" * 60)

    geolife_loader = GeoLifeDataLoader(geolife_root)

    # 获取用户列表
    users_data_path = os.path.join(geolife_root, 'Data')
    if not os.path.exists(users_data_path):
        print(f"错误：未找到 GeoLife 数据目录: {users_data_path}")
        return []

    users = [d for d in os.listdir(users_data_path) if os.path.isdir(os.path.join(users_data_path, d))]

    if max_users:
        users = users[:max_users]

    print(f"找到 {len(users)} 个用户")

    # 加载所有轨迹段
    all_segments = []
    for user_id in tqdm(users, desc="加载轨迹数据"):
        # 假设 load_labels 接受用户ID并能找到 labels.txt
        labels = geolife_loader.load_labels(user_id)
        if labels.empty:
            continue

        # 加载该用户的所有轨迹文件
        trajectory_dir = os.path.join(geolife_root, f"Data/{user_id}/Trajectory")
        if not os.path.exists(trajectory_dir):
            continue

        for traj_file in os.listdir(trajectory_dir):
            if not traj_file.endswith('.plt'):
                continue

            traj_path = os.path.join(trajectory_dir, traj_file)
            try:
                trajectory = geolife_loader.load_trajectory(traj_path)
                segments = geolife_loader.segment_trajectory(trajectory, labels)
                all_segments.extend(segments)
            except Exception as e:
                # 捕获单个文件错误
                print(f"警告: 加载轨迹文件失败 {traj_path}: {e}")
                continue

    print(f"\n总共加载 {len(all_segments)} 个轨迹段")

    # 预处理轨迹段
    print("\n预处理轨迹段...")
    processed_segments = preprocess_segments(all_segments, min_length=10)

    print(f"预处理后剩余 {len(processed_segments)} 个有效轨迹段")

    return processed_segments


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in tqdm(dataloader, desc="训练中"):
        features = features.to(device)
        labels = labels.squeeze().to(device)

        # 前向传播
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
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
        for features, labels in tqdm(dataloader, desc="评估中"):
            features = features.to(device)
            labels = labels.squeeze().to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # 分类报告
    class_names = label_encoder.classes_
    num_classes = len(class_names)

    # 🔥 修复 ValueError: 显式指定标签 (0 到 num_classes-1)
    target_labels = np.arange(num_classes)
    report = classification_report(all_labels, all_preds,
                                  target_names=class_names,
                                  labels=target_labels,
                                  output_dict=True,
                                  zero_division=0) # 避免稀疏类别带来的警告

    return avg_loss, report, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='训练交通方式识别模型（仅轨迹特征）')
    parser.add_argument('--geolife_root', type=str,
                       default='../data/Geolife Trajectories 1.3',
                       help='GeoLife数据根目录')
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

    # 加载数据 (如果存在缓存，这里会加载 11 个类别的segments)
    segments = load_data(args.geolife_root, args.max_users)

    if len(segments) == 0:
        print("错误: 没有加载到任何数据")
        return

    # =================================================================
    # 🔥 核心修改 1: 过滤掉样本数过少的稀疏类别
    # =================================================================
    print("\n" + "=" * 60)
    print("过滤稀疏类别：仅保留主要 7 种交通方式 (walk, bike, car, bus, train, taxi, subway)")
    TARGET_MODES = ['walk', 'bike', 'car', 'bus', 'train', 'taxi', 'subway']

    original_count = len(segments)
    segments = [seg for seg in segments if seg[1] in TARGET_MODES]

    removed_count = original_count - len(segments)
    print(f"原始轨迹段总数: {original_count}")
    print(f"保留轨迹段总数: {len(segments)} (移除 {removed_count} 个稀疏类别)")
    print("=" * 60)
    # =================================================================

    # 准备标签编码器
    all_labels = [label for _, label in segments]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    num_classes = len(label_encoder.classes_)
    print(f"\n类别数: {num_classes}")
    print(f"类别: {list(label_encoder.classes_)}")

    # 统计各类别数量
    label_counts = Counter(all_labels)
    print("\n各类别样本数:")
    # 使用 label_encoder.classes_ 确保打印顺序一致且包含所有类
    for label in label_encoder.classes_:
        print(f"  {label}: {label_counts.get(label, 0)}")


    # 创建数据集
    dataset = TrajectoryDataset(segments, label_encoder)

    # 划分训练集和测试集
    # 使用 stratify 确保训练集和测试集的类别分布均衡
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

    print(f"\n训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 创建模型
    model = TransportationModeClassifier(
        input_dim=TRAJECTORY_FEATURE_DIM,  # 9维轨迹特征
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(args.device)

    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"使用设备: {args.device}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 训练循环
    best_test_loss = float('inf')
    best_test_acc = 0

    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")

        # 评估
        test_loss, test_report, test_preds, test_labels = evaluate(
            model, test_loader, criterion, args.device, label_encoder
        )
        test_acc = test_report['accuracy']
        print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")

        # 学习率调度
        scheduler.step(test_loss)

        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
                'test_acc': test_acc,
                'label_encoder': label_encoder,
                'model_config': {
                    'input_dim': TRAJECTORY_FEATURE_DIM, # 使用常量
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'num_classes': num_classes,
                    'dropout': args.dropout
                }
            }, os.path.join(args.save_dir, 'exp1_model.pth'))
            print("✓ 保存最佳模型")

        # 每10个epoch打印详细分类报告
        if (epoch + 1) % 10 == 0:
            print("\n详细分类报告:")
            # 🔥 修复 ValueError: 显式指定标签 (0 到 num_classes-1)
            print(classification_report(test_labels, test_preds,
                                      target_names=label_encoder.classes_,
                                      labels=np.arange(num_classes),
                                      zero_division=0))

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳测试损失: {best_test_loss:.4f}")
    print(f"最佳测试准确率: {best_test_acc:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()