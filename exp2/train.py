"""
训练脚本 (混合优化版)

核心优化:
1. ✅ 网格缓存 KG 特征 (首次慢，后续快)
2. ✅ 批量查询未缓存点
3. ✅ 三级缓存：KG 数据 + 网格缓存 + 最终特征
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm
import pickle
import warnings

from src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader, preprocess_trajectory_segments
from src.feature_extraction import FeatureExtractor
from src.model import TransportationModeClassifier
from src.knowledge_graph import TransportationKnowledgeGraph

# ========================== 特征维度常量 ==========================
TRAJECTORY_FEATURE_DIM = 9
KG_FEATURE_DIM = 11
# =================================================================


# ========================== 三级缓存配置 ==========================
CACHE_DIR = 'cache'
KG_CACHE_PATH = os.path.join(CACHE_DIR, 'kg_data.pkl')
GRID_CACHE_PATH = os.path.join(CACHE_DIR, 'grid_cache.pkl')
PROCESSED_FEATURE_CACHE_PATH = os.path.join(CACHE_DIR, 'processed_features.pkl')
os.makedirs(CACHE_DIR, exist_ok=True)


# ==================================================================


class TrajectoryDataset(Dataset):
    """轨迹数据集"""

    def __init__(self, all_features_and_labels):
        self.data = all_features_and_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trajectory_features, kg_features, label_encoded = self.data[idx]

        trajectory_tensor = torch.FloatTensor(trajectory_features)
        kg_tensor = torch.FloatTensor(kg_features)
        label_tensor = torch.LongTensor([label_encoded])[0]

        return trajectory_tensor, kg_tensor, label_tensor


def load_data(geolife_root: str, osm_path: str, max_users: int = None):
    """加载所有数据，实现三级缓存"""

    geolife_loader = GeoLifeDataLoader(geolife_root)
    users = geolife_loader.get_all_users()
    if max_users and max_users < len(users):
        users = users[:max_users]

    # ================= 阶段 1: 知识图谱构建 (一级缓存) ==================
    kg = None
    if os.path.exists(KG_CACHE_PATH):
        print(f"\n========== 阶段 1: 知识图谱加载 (从缓存) ==========")
        try:
            with open(KG_CACHE_PATH, 'rb') as f:
                kg = pickle.load(f)
            print("✅ 知识图谱从缓存加载完成。")
            print(f"   统计: {kg.get_graph_statistics()}")

            # 加载网格缓存 (二级缓存)
            if os.path.exists(GRID_CACHE_PATH):
                kg.load_cache(GRID_CACHE_PATH)

        except Exception as e:
            warnings.warn(f"[WARN] KG 缓存加载失败 ({e})，将重新构建。")
            os.remove(KG_CACHE_PATH)
            kg = None

    if kg is None:
        print("\n========== 阶段 1: 知识图谱构建 (重建) ==========")
        print(f"找到 {len(users)} 个用户")

        osm_loader = OSMDataLoader(osm_path)
        osm_data = osm_loader.load_osm_data()
        road_network = osm_loader.extract_road_network(osm_data)
        pois = osm_loader.extract_pois(osm_data)
        print(f"OSM 数据: {len(road_network)} 条道路, {len(pois)} 个POI")

        kg = TransportationKnowledgeGraph()
        kg.build_from_osm(road_network, pois)
        print(f"统计: {kg.get_graph_statistics()}")

        print(f"正在保存到: {KG_CACHE_PATH}")
        with open(KG_CACHE_PATH, 'wb') as f:
            pickle.dump(kg, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✅ 知识图谱缓存完成。")

    # ================= 阶段 2: 特征提取 (三级缓存) ==================
    if os.path.exists(PROCESSED_FEATURE_CACHE_PATH):
        print(f"\n========== 阶段 2: 特征加载 (从缓存) ==========")
        try:
            with open(PROCESSED_FEATURE_CACHE_PATH, 'rb') as f:
                all_features_and_labels, label_encoder = pickle.load(f)
            print(f"✅ 特征从缓存加载完成: {len(all_features_and_labels)} 条记录")

            # 打印缓存统计
            cache_stats = kg.get_cache_stats()
            print(f"   网格缓存统计: {cache_stats}")

            return all_features_and_labels, kg, label_encoder
        except Exception as e:
            warnings.warn(f"[WARN] 特征缓存加载失败 ({e})，将重新提取。")
            os.remove(PROCESSED_FEATURE_CACHE_PATH)

    # --- 重新提取特征 ---
    print("\n========== 阶段 2: 特征提取 (重建) ==========")

    # 2.1 加载轨迹段
    print("2.1 正在加载轨迹段...")
    all_segments = []
    for user_id in tqdm(users, desc="[用户加载]"):
        labels = geolife_loader.load_labels(user_id)
        if labels.empty:
            continue

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
            except Exception:
                continue

    print(f"   共加载 {len(all_segments)} 个轨迹段")

    # 2.2 预处理
    print("2.2 正在预处理轨迹段...")
    processed_segments = preprocess_trajectory_segments(all_segments, min_length=10)

    # 过滤有效类别
    valid_modes = {'walk', 'bike', 'bus', 'car', 'train', 'taxi'}
    processed_segments = [
        (traj, label) for traj, label in processed_segments
        if label in valid_modes
    ]
    print(f"   剩余 {len(processed_segments)} 个有效轨迹段")

    if not processed_segments:
        print("❌ 错误: 没有可用数据")
        return [], kg, None

    # 2.3 创建标签编码器
    all_labels_str = [label for _, label in processed_segments]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels_str)

    # 2.4 混合特征提取 (缓存 + 批量查询)
    print(f"2.4 正在提取特征 (共 {len(processed_segments)} 个轨迹段)...")
    print("    提示: 首次运行会慢，后续运行会从缓存加速")

    feature_extractor = FeatureExtractor(kg)
    all_features_and_labels = []

    for trajectory, label_str in tqdm(processed_segments, desc="[特征提取]"):
        try:
            # 混合特征提取：自动使用缓存
            trajectory_features, kg_features = feature_extractor.extract_features(trajectory)
            label_encoded = label_encoder.transform([label_str])[0]
            all_features_and_labels.append((trajectory_features, kg_features, label_encoded))
        except Exception as e:
            print(f"警告: 特征提取失败 ({e})")
            continue

    # 2.5 保存缓存
    print(f"2.5 正在保存特征到: {PROCESSED_FEATURE_CACHE_PATH}")
    data_to_save = (all_features_and_labels, label_encoder)
    with open(PROCESSED_FEATURE_CACHE_PATH, 'wb') as f:
        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("✅ 特征缓存完成。")

    # 保存网格缓存
    print(f"2.6 正在保存网格缓存到: {GRID_CACHE_PATH}")
    kg.save_cache(GRID_CACHE_PATH)

    # 打印缓存统计
    cache_stats = kg.get_cache_stats()
    print(f"\n缓存统计:")
    print(f"  - 网格数量: {cache_stats['cache_size']}")
    print(f"  - 缓存命中: {cache_stats['cache_hits']}")
    print(f"  - 缓存未命中: {cache_stats['cache_misses']}")
    print(f"  - 命中率: {cache_stats['hit_rate']}")
    print(f"  - 缓存内存: {cache_stats['cache_memory_mb']:.2f} MB")

    return all_features_and_labels, kg, label_encoder


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for trajectory_features, kg_features, labels in tqdm(dataloader, desc="   [训练]"):
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

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device, label_encoder):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for trajectory_features, kg_features, labels in tqdm(dataloader, desc="   [评估]"):
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
    report = classification_report(
        all_labels, all_preds,
        target_names=label_encoder.classes_,
        output_dict=True, zero_division=0
    )

    return avg_loss, report, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='训练交通方式识别模型 (混合优化版)')
    parser.add_argument('--geolife_root', type=str,
                        default='../data/Geolife Trajectories 1.3')
    parser.add_argument('--osm_path', type=str,
                        default='../data/export.geojson')
    parser.add_argument('--max_users', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--clear_cache', action='store_true',
                        help='清空所有缓存并重新构建')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 可选：清空缓存
    if args.clear_cache:
        print("\n清空所有缓存...")
        for cache_file in [KG_CACHE_PATH, GRID_CACHE_PATH, PROCESSED_FEATURE_CACHE_PATH]:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"  - 已删除: {cache_file}")

    # 加载数据
    all_features_and_labels, kg, label_encoder = load_data(
        args.geolife_root, args.osm_path, args.max_users
    )

    if len(all_features_and_labels) == 0:
        print("❌ 错误: 没有可用数据")
        return

    print("\n========== 阶段 3: 模型训练 ==========")
    num_classes = len(label_encoder.classes_)
    print(f"类别数: {num_classes}")
    print(f"类别: {label_encoder.classes_}")

    dataset = TrajectoryDataset(all_features_and_labels)

    # 划分数据集
    labels_for_stratify = [label for _, _, label in all_features_and_labels]
    train_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42,
        stratify=labels_for_stratify
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    print(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")

    # 创建模型
    model = TransportationModeClassifier(
        trajectory_feature_dim=TRAJECTORY_FEATURE_DIM,
        kg_feature_dim=KG_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(args.device)

    print(f"模型参数: {sum(p.numel() for p in model.parameters())}")
    print(f"训练设备: {args.device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 训练循环
    best_test_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\n[EPOCH {epoch + 1}/{args.epochs}]")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        print(f"   训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")

        test_loss, test_report, test_preds, test_labels = evaluate(
            model, test_loader, criterion, args.device, label_encoder
        )
        test_acc = test_report['accuracy']
        print(f"   测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")

        scheduler.step(test_loss)

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
            print("   ✅ 保存最佳模型")

        if (epoch + 1) % 10 == 0:
            print("\n   分类报告:")
            print(classification_report(
                test_labels, test_preds,
                target_names=label_encoder.classes_, zero_division=0
            ))

    print(f"\n========== 训练完成 ==========")
    print(f"最佳测试损失: {best_test_loss:.4f}")

    # 最终缓存统计
    cache_stats = kg.get_cache_stats()
    print(f"\n最终缓存统计:")
    print(f"  - 网格缓存命中率: {cache_stats['hit_rate']}")
    print(f"  - 缓存内存占用: {cache_stats['cache_memory_mb']:.2f} MB")


if __name__ == '__main__':
    main()