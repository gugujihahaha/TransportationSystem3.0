"""
训练脚本 (已实现缓存 Checkpointing)
- 确保在 GeoLife 加载和模型训练阶段显示实时进度。
- 阶段 1 和阶段 2 的结果会被缓存到 'cache/' 目录，避免重复耗时计算。
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
from sklearn.metrics import classification_report
from tqdm import tqdm
import pickle # <-- 缓存核心库
import warnings # 用于缓存错误处理

# 导入自定义模块 (完全依赖您的 src 文件夹)
from src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader, preprocess_trajectory_segments
from src.feature_extraction import FeatureExtractor
from src.model import TransportationModeClassifier
from src.knowledge_graph import TransportationKnowledgeGraph # 导入知识图谱模块


# ========================== 特征维度常量 (用于模型和验证) ==========================
TRAJECTORY_FEATURE_DIM = 7
KG_FEATURE_DIM = 11
# ==============================================================================


# ========================== 缓存配置 ==========================
CACHE_DIR = 'cache'
KG_CACHE_PATH = os.path.join(CACHE_DIR, 'kg_data.pkl')
PROCESSED_DATA_CACHE_PATH = os.path.join(CACHE_DIR, 'processed_data.pkl')
os.makedirs(CACHE_DIR, exist_ok=True)
# =============================================================


class TrajectoryDataset(Dataset):
    """轨迹数据集"""

    def __init__(self, segments, feature_extractor, label_encoder):
        self.segments = segments
        self.feature_extractor = feature_extractor
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        # 注意: 这一步会在 DataLoader 的 worker 进程中并行执行 (CPU)
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
    """加载所有数据，并实现阶段 1 和阶段 2 的缓存检查"""

    geolife_loader = GeoLifeDataLoader(geolife_root)
    users = geolife_loader.get_all_users()
    if max_users and max_users < len(users):
        users = users[:max_users]


    # ================= 阶段 1: 知识图谱构建 (缓存检查) ==================
    kg = None
    if os.path.exists(KG_CACHE_PATH):
        print(f"\n================ 阶段 1: 知识图谱构建 (从缓存加载) ==================")
        try:
            with open(KG_CACHE_PATH, 'rb') as f:
                kg = pickle.load(f)
            print("1.0 知识图谱从缓存加载完成。")
        except Exception as e:
            # 健壮性增强：缓存文件损坏，删除并重建
            warnings.warn(f"[WARN] 知识图谱缓存文件加载失败 ({e})，将重新构建。")
            os.remove(KG_CACHE_PATH)
            kg = None # 强制重建

    if kg is None:
        # --- 正常执行阶段 1 (耗时部分) ---
        print("\n================ 阶段 1: 数据加载与知识图谱构建 (重建) ==================")
        print(f"1.1 正在初始化 GeoLife 数据加载器: {geolife_root}")
        print(f" -> 找到 {len(users)} 个 GeoLife 用户（将处理其中 {len(users)} 个）")

        # 加载OSM数据并构建知识图谱
        print("\n1.2 正在加载 OSM 数据...")
        osm_loader = OSMDataLoader(osm_path)
        osm_data = osm_loader.load_osm_data()

        road_network = osm_loader.extract_road_network(osm_data)
        pois = osm_loader.extract_pois(osm_data)
        print(f" -> OSM 数据提取完成: {len(road_network)} 条道路, {len(pois)} 个POI。")

        # KG构建和关联阶段
        print("\n1.3 正在构建交通知识图谱...")
        kg = TransportationKnowledgeGraph()
        kg.build_from_osm(road_network, pois)
        print(f" -> 知识图谱构建阶段已完成。统计: {kg.get_graph_statistics()}")

        # --- 阶段 1 缓存 ---
        print(f"1.4 知识图谱构建完成，正在存储到缓存文件：{KG_CACHE_PATH}")
        with open(KG_CACHE_PATH, 'wb') as f:
            # 使用 HIGHEST_PROTOCOL 提升序列化速度
            pickle.dump(kg, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("1.4 知识图谱缓存完成。")


    # ================= 阶段 2: 轨迹加载、分段与特征提取 (缓存检查) ==================
    processed_segments = None
    label_encoder = None
    if os.path.exists(PROCESSED_DATA_CACHE_PATH):
        print("\n================ 阶段 2: 轨迹加载与特征提取 (从缓存加载) ==================")
        try:
            with open(PROCESSED_DATA_CACHE_PATH, 'rb') as f:
                processed_segments, label_encoder = pickle.load(f)
            print(f"2.0 预处理数据从缓存加载完成。剩余 {len(processed_segments)} 个可用轨迹段。")
        except Exception as e:
            # 健壮性增强：缓存文件损坏，删除并重建
            warnings.warn(f"[WARN] 预处理数据缓存文件加载失败 ({e})，将重新处理。")
            os.remove(PROCESSED_DATA_CACHE_PATH)
            processed_segments = None # 强制重建

    if processed_segments is None:
        # --- 正常执行阶段 2 (耗时部分) ---
        print("\n================ 阶段 2: 轨迹加载、分段与特征提取 (重建) ==================")

        # 加载所有轨迹段
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
                    # load_trajectory 返回 DataFrame
                    trajectory = geolife_loader.load_trajectory(traj_path)
                    segments = geolife_loader.segment_trajectory(trajectory, labels)
                    all_segments.extend(segments)
                except Exception as e:
                    # 避免在 tqdm 中打印大量内容
                    # print(f"\n[WARN] 加载轨迹文件失败 {traj_path}: {e}")
                    continue

        print(f" -> GeoLife 轨迹加载完成。总共加载 {len(all_segments)} 个原始轨迹段。")

        # 预处理轨迹段
        print("\n2.2 正在预处理轨迹段...")
        # preprocess_trajectory_segments 负责将 DataFrame 转换为 NumPy 数组
        processed_segments = preprocess_trajectory_segments(all_segments, min_length=10)
        print(f" -> 预处理完成。剩余 {len(processed_segments)} 个可用轨迹段。")

        # 创建标签编码器
        all_labels = [label for _, label in processed_segments]
        label_encoder = LabelEncoder()
        label_encoder.fit(all_labels)

        # --- 阶段 2 缓存 ---
        print(f"2.3 轨迹预处理完成，正在存储到缓存文件：{PROCESSED_DATA_CACHE_PATH}")
        data_to_save = (processed_segments, label_encoder)
        with open(PROCESSED_DATA_CACHE_PATH, 'wb') as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("2.3 预处理数据缓存完成。")

    # 返回所有组件
    return processed_segments, kg, label_encoder


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # 训练批次的 tqdm
    for trajectory_features, kg_features, labels in tqdm(dataloader, desc="   [训练批次]"):
        # 将数据从 CPU 移动到 GPU
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
            # 将数据从 CPU 移动到 GPU
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
    # 添加 zero_division=0 避免警告/错误
    report = classification_report(all_labels, all_preds,
                                  target_names=class_names,
                                  output_dict=True, zero_division=0)

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
    # <<< 新增：用于并行数据加载和特征提取 >>>
    parser.add_argument('--num_workers', type=int, default=4,
                       help='用于数据加载的进程数，以利用多核CPU加速I/O和特征提取')
    # <<< 保持自动检测 GPU 逻辑不变 >>>
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')

    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载数据 (利用缓存跳过耗时环节)
    segments, kg, label_encoder = load_data(args.geolife_root, args.osm_path, args.max_users)

    if len(segments) == 0:
        print("错误: 没有加载到任何数据，请检查数据路径和预处理条件。")
        return

    # 创建特征提取器
    print("\n================ 阶段 3: 模型初始化与训练 ==================")
    feature_extractor = FeatureExtractor(kg)

    # 准备标签编码器
    num_classes = len(label_encoder.classes_)
    print(f" -> 类别数: {num_classes}")
    print(f" -> 类别: {label_encoder.classes_}")

    # 创建数据集
    dataset = TrajectoryDataset(segments, feature_extractor, label_encoder)

    # 划分训练集和测试集
    labels_for_stratify = label_encoder.transform([segments[i][1] for i in range(len(segments))])
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=labels_for_stratify
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # <<< 修正：应用 num_workers 参数和 pin_memory=True >>>
    # pin_memory=True: 加速数据从 CPU 到 GPU 的传输
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    print(f" -> 训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    # 创建模型并将其移动到 GPU
    model = TransportationModeClassifier(
        trajectory_feature_dim=TRAJECTORY_FEATURE_DIM,
        kg_feature_dim=KG_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(args.device) # <<< 关键：将模型移动到 GPU/CPU

    print(f" -> 模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f" -> 训练设备: {args.device}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 训练循环
    best_test_loss = float('inf')

    print(f"\n================ 阶段 4: 模型训练 (共 {args.epochs} 轮) ==================")
    # 触发 LSTM 工作原理图
    print("模型结构采用 Bi-LSTM，适用于序列数据和异构特征融合。")


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
            }, os.path.join(args.save_dir, 'exp2_model.pth'))
            print("   [INFO] 测试损失降低，保存最佳模型。")

        # 打印分类报告
        if (epoch + 1) % 10 == 0:
            print("\n   [报告] 当前分类报告:")
            print(classification_report(test_labels, test_preds,
                                      target_names=label_encoder.classes_, zero_division=0))

    print("\n================ 训练完成! ==================")
    print(f"最终最佳测试损失: {best_test_loss:.4f}")
    print(f"最佳模型已保存到 {os.path.join(args.save_dir, 'exp2_model.pth')}")


if __name__ == '__main__':
    main()