"""
训练脚本 (Exp3)
支持跨机器训练：主机生成缓存 → 游戏本 GPU 训练

核心特性:
1. 三级缓存 + 版本控制
2. 增强 KG 特征 (15维)
3. 跨机器训练支持
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
import multiprocessing
import hashlib
import json
from datetime import datetime
import numpy as np

# 导入自定义模块
from src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader, preprocess_trajectory_segments
from src.feature_extraction import FeatureExtractor
from src.model import TransportationModeClassifier
from src.knowledge_graph_enhanced import EnhancedTransportationKG

# ========================== 特征维度常量 ==========================
TRAJECTORY_FEATURE_DIM = 9
KG_FEATURE_DIM = 15  # Exp3 增强特征
# ==================================================================


# ========================== 缓存配置 ==========================
CACHE_VERSION = "v1"
CACHE_DIR = 'cache'
KG_CACHE_PATH = os.path.join(CACHE_DIR, f'kg_data_{CACHE_VERSION}.pkl')
PROCESSED_FEATURE_CACHE_PATH = os.path.join(CACHE_DIR, f'processed_features_{CACHE_VERSION}.pkl')
META_CACHE_PATH = os.path.join(CACHE_DIR, 'cache_meta.json')
os.makedirs(CACHE_DIR, exist_ok=True)


# ==============================================================


def compute_file_hash(filepath: str) -> str:
    """计算文件 MD5 哈希"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_cache_metadata(osm_path: str, geolife_root: str,
                        num_segments: int, label_encoder: LabelEncoder):
    """保存缓存元数据"""
    meta = {
        "version": CACHE_VERSION,
        "experiment": "exp3",
        "created_at": datetime.now().isoformat(),
        "osm_file": osm_path,
        "osm_file_hash": compute_file_hash(osm_path),
        "geolife_root": geolife_root,
        "kg_feature_dim": KG_FEATURE_DIM,
        "trajectory_feature_dim": TRAJECTORY_FEATURE_DIM,
        "total_feature_dim": TRAJECTORY_FEATURE_DIM + KG_FEATURE_DIM,
        "num_segments": num_segments,
        "num_classes": len(label_encoder.classes_),
        "classes": label_encoder.classes_.tolist()
    }

    with open(META_CACHE_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"✓ 缓存元数据已保存: {META_CACHE_PATH}")


def validate_cache(osm_path: str) -> bool:
    """验证缓存是否有效"""
    if not os.path.exists(META_CACHE_PATH):
        return False

    try:
        with open(META_CACHE_PATH, 'r') as f:
            meta = json.load(f)

        # 检查版本
        if meta.get('version') != CACHE_VERSION:
            print(f"⚠️  缓存版本不匹配: {meta.get('version')} != {CACHE_VERSION}")
            return False

        # 检查实验类型
        if meta.get('experiment') != 'exp3':
            print(f"⚠️  缓存实验类型不匹配: {meta.get('experiment')} != exp3")
            return False

        # 检查 OSM 文件哈希
        current_hash = compute_file_hash(osm_path)
        if meta.get('osm_file_hash') != current_hash:
            print(f"⚠️  OSM 文件已更改，缓存失效")
            return False

        print(f"✓ 缓存验证通过 (版本: {CACHE_VERSION})")
        return True

    except Exception as e:
        print(f"⚠️  缓存验证失败: {e}")
        return False


class TrajectoryDataset(Dataset):
    """轨迹数据集 (直接加载预提取的特征)"""

    def __init__(self, all_features_and_labels):
        """
        Args:
            all_features_and_labels: List[(traj_feat, kg_feat, label_encoded), ...]
        """
        self.data = all_features_and_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trajectory_features, kg_features, label_encoded = self.data[idx]

        # 转换为 tensor
        trajectory_tensor = torch.FloatTensor(trajectory_features)
        kg_tensor = torch.FloatTensor(kg_features)
        label_tensor = torch.LongTensor([label_encoded])[0]

        return trajectory_tensor, kg_tensor, label_tensor


def load_data(geolife_root: str, osm_path: str, max_users: int = None):
    """加载所有数据，并实现三级缓存"""

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
            print(f"     统计: {kg.get_graph_statistics()}")
        except Exception as e:
            warnings.warn(f"[WARN] 知识图谱缓存加载失败 ({e})，将重新构建。")
            os.remove(KG_CACHE_PATH)
            kg = None

    if kg is None:
        # 重新构建知识图谱
        print("\n================ 阶段 1: 数据加载与知识图谱构建 (重建) ==================")
        print(f"1.1 正在初始化 GeoLife 数据加载器: {geolife_root}")
        print(f" -> 找到 {len(users)} 个 GeoLife 用户")

        # 加载 OSM 数据
        osm_loader = OSMDataLoader(osm_path)
        osm_data = osm_loader.load_osm_data()
        road_network = osm_loader.extract_road_network(osm_data)
        pois = osm_loader.extract_pois(osm_data)
        transit_routes = osm_loader.extract_transit_routes(osm_data)

        print(f" -> OSM 数据提取完成:")
        print(f"    - 道路: {len(road_network)} 条")
        print(f"    - POI: {len(pois)} 个")
        print(f"    - 公交/地铁线路: {len(transit_routes)} 条")

        # 构建增强知识图谱
        print("\n1.3 正在构建增强知识图谱 (Exp3)...")
        kg = EnhancedTransportationKG()
        kg.build_from_osm(road_network, pois, transit_routes)

        stats = kg.get_graph_statistics()
        print(f" -> 知识图谱构建完成。统计:")
        print(f"    - 节点: {stats['num_nodes']}")
        print(f"    - 边: {stats['num_edges']}")
        print(f"    - 道路节点: {stats['road_nodes']}")
        print(f"    - POI节点: {stats['poi_nodes']}")
        print(f"    - 速度限制记录: {stats['speed_limits']}")
        print(f"    - 公交线路: {stats['bus_routes']}")
        print(f"    - 地铁线路: {stats['subway_routes']}")

        # 缓存知识图谱
        print(f"1.4 正在缓存知识图谱到: {KG_CACHE_PATH}")
        with open(KG_CACHE_PATH, 'wb') as f:
            pickle.dump(kg, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("1.4 知识图谱缓存完成。")

    # ================= 阶段 2: 特征提取 (缓存检查) ==================
    all_features_and_labels = None
    label_encoder = None

    if os.path.exists(PROCESSED_FEATURE_CACHE_PATH):
        print(f"\n================ 阶段 2: 特征提取 (从缓存加载) ==================")
        try:
            with open(PROCESSED_FEATURE_CACHE_PATH, 'rb') as f:
                all_features_and_labels, label_encoder = pickle.load(f)
            print(f"2.0 预提取特征从缓存加载完成。共有 {len(all_features_and_labels)} 条记录。")
            return all_features_and_labels, kg, label_encoder
        except Exception as e:
            warnings.warn(f"[WARN] 特征缓存加载失败 ({e})，将重新处理。")
            os.remove(PROCESSED_FEATURE_CACHE_PATH)
            all_features_and_labels = None

    # 重新提取特征
    print("\n================ 阶段 2: 轨迹加载、分段与【特征提取】 (重建) ==================")

    # 2.1 加载轨迹段
    print("2.1 正在加载 GeoLife 轨迹段...")
    all_segments = []
    for user_id in tqdm(users, desc="[GeoLife用户加载]"):
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
                if trajectory.empty:
                    continue
                segments = geolife_loader.segment_trajectory(trajectory, labels)
                all_segments.extend(segments)
            except Exception as e:
                continue

    print(f" -> GeoLife 轨迹加载完成。总共 {len(all_segments)} 个原始轨迹段。")

    # 2.2 预处理轨迹段
    print("\n2.2 正在预处理轨迹段...")
    processed_segments = preprocess_trajectory_segments(all_segments, min_length=10)
    print(f" -> 预处理完成。剩余 {len(processed_segments)} 个可用轨迹段。")

    # 2.3 过滤稀疏类别
    valid_modes = {'walk', 'bike', 'bus', 'car', 'train', 'taxi'}
    original_count = len(processed_segments)
    processed_segments = [
        (traj, label) for traj, label in processed_segments
        if label in valid_modes
    ]
    filtered_count = original_count - len(processed_segments)
    if filtered_count > 0:
        print(f" -> 过滤掉 {filtered_count} 个非标准类别的轨迹段")

    if not processed_segments:
        print("错误: 没有加载到任何可用轨迹段。")
        return [], kg, None

    # 2.4 创建标签编码器
    all_labels_str = [label for _, label in processed_segments]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels_str)

    print(f"\n标签统计:")
    from collections import Counter
    label_counts = Counter(all_labels_str)
    for label in label_encoder.classes_:
        print(f"  {label}: {label_counts.get(label, 0)}")

    # 2.5 特征提取
    print("\n2.5 正在进行【增强 KG 特征提取】(首次运行耗时较长)...")
    feature_extractor = FeatureExtractor(kg)
    all_features_and_labels = []

    for trajectory, label_str in tqdm(processed_segments, desc="[Exp3 特征提取]"):
        # 提取特征
        trajectory_features, kg_features = feature_extractor.extract_features(trajectory)

        # 编码标签
        label_encoded = label_encoder.transform([label_str])[0]

        # 存储
        all_features_and_labels.append((trajectory_features, kg_features, label_encoded))

    # 2.6 缓存特征
    print(f"\n2.6 正在缓存特征到: {PROCESSED_FEATURE_CACHE_PATH}")
    data_to_save = (all_features_and_labels, label_encoder)
    with open(PROCESSED_FEATURE_CACHE_PATH, 'wb') as f:
        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("2.6 特征缓存完成。")

    # 2.7 保存元数据
    save_cache_metadata(osm_path, geolife_root, len(all_features_and_labels), label_encoder)

    return all_features_and_labels, kg, label_encoder


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

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
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    return avg_loss, report, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='训练交通方式识别模型 (Exp3)')
    parser.add_argument('--geolife_root', type=str,
                        default='../data/Geolife Trajectories 1.3',
                        help='GeoLife数据根目录')
    parser.add_argument('--osm_path', type=str,
                        default='../data/beijing_complete.geojson',
                        help='OSM数据路径 (完整版)')
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
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader进程数')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')

    # 跨机器训练参数
    parser.add_argument('--generate_cache_only', action='store_true',
                        help='仅生成缓存，不训练模型（主机模式）')
    parser.add_argument('--use_cached_data', action='store_true',
                        help='直接使用缓存数据（游戏本模式）')

    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # ========== 步骤1: 加载数据和生成缓存 ==========
    if args.use_cached_data:
        # 游戏本模式：直接加载缓存
        print("\n" + "=" * 60)
        print("游戏本模式：加载缓存数据")
        print("=" * 60)

        if not validate_cache(args.osm_path):
            print("错误：缓存无效或不存在，请在主机上重新生成缓存")
            print("主机命令: python train.py --generate_cache_only")
            return

        with open(PROCESSED_FEATURE_CACHE_PATH, 'rb') as f:
            all_features_and_labels, label_encoder = pickle.load(f)
        print(f"✓ 从缓存加载 {len(all_features_and_labels)} 条记录")

        # 加载 KG（仅用于统计）
        with open(KG_CACHE_PATH, 'rb') as f:
            kg = pickle.load(f)

    else:
        # 主机模式：正常加载或生成缓存
        print("\n" + "=" * 60)
        print("主机模式：处理数据并生成缓存")
        print("=" * 60)

        all_features_and_labels, kg, label_encoder = load_data(
            args.geolife_root, args.osm_path, args.max_users
        )

    if args.generate_cache_only:
        print("\n" + "=" * 60)
        print("✓ 缓存生成完成！")
        print("=" * 60)
        print(f"缓存文件位置: {CACHE_DIR}/")
        print(f"  - 知识图谱: {KG_CACHE_PATH}")
        print(f"  - 特征数据: {PROCESSED_FEATURE_CACHE_PATH}")
        print(f"  - 元数据: {META_CACHE_PATH}")
        print("\n可以将 cache/ 文件夹复制到游戏本进行训练:")
        print("  tar -czf exp3_cache.tar.gz cache/")
        print("  # 复制到游戏本后解压")
        print("  python train.py --use_cached_data --device cuda --epochs 50")
        return

    # ========== 步骤2: 训练模型 ==========
    print("\n" + "=" * 60)
    print(f"开始训练 (设备: {args.device})")
    print("=" * 60)

    num_classes = len(label_encoder.classes_)
    print(f" -> 类别数: {num_classes}")
    print(f" -> 类别: {label_encoder.classes_}")

    # 创建数据集
    dataset = TrajectoryDataset(all_features_and_labels)

    # 划分训练集和测试集
    labels_for_stratify = [label for _, _, label in all_features_and_labels]
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=labels_for_stratify
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f" -> 训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")

    # 创建模型
    model = TransportationModeClassifier(
        trajectory_feature_dim=TRAJECTORY_FEATURE_DIM,
        kg_feature_dim=KG_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(args.device)

    print(f" -> 模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 训练循环
    best_test_loss = float('inf')
    best_test_acc = 0

    for epoch in range(args.epochs):
        print(f"\n[EPOCH {epoch + 1}/{args.epochs}]")

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        print(f"   训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")

        # 评估
        test_loss, test_report, test_preds, test_labels = evaluate(
            model, test_loader, criterion, args.device, label_encoder
        )
        test_acc = test_report['accuracy']
        print(f"   测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")

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
                    'trajectory_feature_dim': TRAJECTORY_FEATURE_DIM,
                    'kg_feature_dim': KG_FEATURE_DIM,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'num_classes': num_classes,
                    'dropout': args.dropout
                }
            }, os.path.join(args.save_dir, 'exp3_model.pth'))
            print("   ✓ 保存最佳模型")

        # 打印分类报告
        if (epoch + 1) % 10 == 0:
            print("\n   [分类报告]")
            print(classification_report(
                test_labels, test_preds,
                target_names=label_encoder.classes_,
                zero_division=0
            ))

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳测试损失: {best_test_loss:.4f}")
    print(f"最佳测试准确率: {best_test_acc:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    if os.name == 'nt':
        multiprocessing.freeze_support()

    main()