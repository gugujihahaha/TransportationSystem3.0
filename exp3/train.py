"""
训练脚本 (Exp3)
基于 Exp2 成功架构，新增:
1. 增强空间特征 (15维)
2. 缓存版本控制
3. 跨机器训练支持
✅ 已集成快速模式支持
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
import hashlib
import json
from datetime import datetime
import numpy as np

# ===== ✅ 修改 1: 文件开头添加导入 =====
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (BaseGeoLifePreprocessor, Exp3DataAdapter,
                     train_epoch, evaluate, compute_class_weights)
# ===== 新增结束 =====

from src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader, preprocess_trajectory_segments
from src.feature_extraction import FeatureExtractor
from src.model import TransportationModeClassifier
from src.osm_feature_extractor import EnhancedOsmSpatialExtractor

# ========================== 特征维度常量 ==========================
TRAJECTORY_FEATURE_DIM = 9
SPATIAL_FEATURE_DIM = 15  # Exp3: 11 → 15
# ==================================================================

# ========================== 缓存配置 ==========================
CACHE_VERSION = "v1"
CACHE_DIR = 'cache'
SPATIAL_CACHE_PATH = os.path.join(CACHE_DIR, f'spatial_data_{CACHE_VERSION}.pkl')
SPATIAL_GRID_CACHE_PATH = os.path.join(CACHE_DIR, f'spatial_grid_cache_{CACHE_VERSION}.pkl')
PROCESSED_FEATURE_CACHE_PATH = os.path.join(CACHE_DIR, f'processed_features_{CACHE_VERSION}.pkl')
META_CACHE_PATH = os.path.join(CACHE_DIR, 'cache_meta.json')
os.makedirs(CACHE_DIR, exist_ok=True)


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
        "spatial_feature_dim": SPATIAL_FEATURE_DIM,
        "trajectory_feature_dim": TRAJECTORY_FEATURE_DIM,
        "total_feature_dim": TRAJECTORY_FEATURE_DIM + SPATIAL_FEATURE_DIM,
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

        if meta.get('version') != CACHE_VERSION:
            print(f"⚠️  缓存版本不匹配: {meta.get('version')} != {CACHE_VERSION}")
            return False

        if meta.get('experiment') != 'exp3':
            print(f"⚠️  缓存实验类型不匹配: {meta.get('experiment')} != exp3")
            return False

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
    """轨迹数据集"""

    def __init__(self, all_features_and_labels,
                 traj_mean=None, traj_std=None,
                 spatial_mean=None, spatial_std=None,
                 stats_mean=None, stats_std=None):
        self.data = all_features_and_labels
        self.traj_mean = traj_mean
        self.traj_std = traj_std
        self.spatial_mean = spatial_mean
        self.spatial_std = spatial_std
        self.stats_mean = stats_mean
        self.stats_std = stats_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trajectory_features, spatial_features, segment_stats, label_encoded = self.data[idx]

        # 归一化
        if self.traj_mean is not None:
            trajectory_features = (trajectory_features - self.traj_mean) / self.traj_std
        if self.spatial_mean is not None:
            spatial_features = (spatial_features - self.spatial_mean) / self.spatial_std
        if self.stats_mean is not None:
            segment_stats = (segment_stats - self.stats_mean) / self.stats_std

        trajectory_tensor = torch.FloatTensor(trajectory_features)
        spatial_tensor = torch.FloatTensor(spatial_features)
        stats_tensor = torch.FloatTensor(segment_stats)
        label_tensor = torch.LongTensor([label_encoded])[0]

        return trajectory_tensor, spatial_tensor, stats_tensor, label_tensor


# ============================================================
# Data loading (✅ 修改 2: 更新 load_data 函数集成快速模式)
# ============================================================
def load_data(geolife_root: str, osm_path: str, max_users: int = None, use_base_data: bool = True, cleaning_mode: str = 'balanced'):
    """加载所有数据 (支持快速模式与三级缓存)

    Args:
        geolife_root: GeoLife数据根目录
        osm_path: OSM数据路径
        max_users: 最大用户数
        use_base_data: 是否使用预处理的基础数据
        cleaning_mode: 数据清洗模式 ('strict', 'balanced', 'gentle')
    """

    BASE_DATA_PATH = os.path.join(
        os.path.dirname(geolife_root),
        'processed/base_segments.pkl'
    )

    geolife_loader = GeoLifeDataLoader(geolife_root)
    users = geolife_loader.get_all_users()
    if max_users and max_users < len(users):
        users = users[:max_users]

    # ================= 阶段 1: 空间特征提取器构建 ==================
    spatial_extractor = None
    if os.path.exists(SPATIAL_CACHE_PATH):
        print(f"\n========== 阶段 1: 空间特征提取器加载 (从缓存) ==========")
        try:
            with open(SPATIAL_CACHE_PATH, 'rb') as f:
                spatial_extractor = pickle.load(f)
            print("✅ 空间特征提取器从缓存加载完成。")
            if os.path.exists(SPATIAL_GRID_CACHE_PATH):
                spatial_extractor.load_cache(SPATIAL_GRID_CACHE_PATH)
        except Exception as e:
            warnings.warn(f"[WARN] 空间特征提取器缓存加载失败 ({e})")
            spatial_extractor = None

    if spatial_extractor is None:
        print("\n========== 阶段 1: 空间特征提取器构建 (重建) ==========")
        osm_loader = OSMDataLoader(osm_path)
        osm_data = osm_loader.load_osm_data()
        road_network = osm_loader.extract_road_network(osm_data)
        pois = osm_loader.extract_pois(osm_data)
        transit_routes = osm_loader.extract_transit_routes(osm_data)

        spatial_extractor = EnhancedOsmSpatialExtractor()
        spatial_extractor.build_from_osm(road_network, pois, transit_routes)

        with open(SPATIAL_CACHE_PATH, 'wb') as f:
            pickle.dump(spatial_extractor, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✅ 空间特征提取器缓存完成。")

    # ================= 阶段 2: 数据加载与特征提取 ==================
    all_features_and_labels = None
    label_encoder = None

    if os.path.exists(PROCESSED_FEATURE_CACHE_PATH):
        print(f"\n========== 阶段 2: 特征加载 (从最终缓存) ==========")
        try:
            with open(PROCESSED_FEATURE_CACHE_PATH, 'rb') as f:
                all_features_and_labels, label_encoder = pickle.load(f)
            print(f"✅ 预提取特征加载完成: {len(all_features_and_labels)} 条记录")
            return all_features_and_labels, spatial_extractor, label_encoder, {}
        except Exception:
            pass

    processed_segments = None
    cleaning_stats = {}

    # ✅ 快速模式：使用基础数据
    if use_base_data and os.path.exists(BASE_DATA_PATH):
        print(f"\n{'='*80}")
        print(f"阶段 2: 使用预处理的基础数据（快速模式 - 清洗模式: {cleaning_mode}）")
        print(f"{'='*80}\n")

        base_segments = BaseGeoLifePreprocessor.load_from_cache(BASE_DATA_PATH)
        adapter = Exp3DataAdapter(target_length=50, enable_cleaning=True, cleaning_mode=cleaning_mode)
        processed_segments = adapter.process_segments(base_segments)
        cleaning_stats = adapter.get_cleaning_stats()

    # 传统模式：从头处理
    else:
        if use_base_data:
            print(f"\n⚠️  基础数据不存在，使用传统模式")

        print("\n========== 阶段 2: 加载轨迹数据 (传统模式) ==========")
        all_segments = []
        for user_id in tqdm(users, desc="[用户加载]"):
            labels = geolife_loader.load_labels(user_id)
            if labels.empty: continue
            trajectory_dir = os.path.join(geolife_root, f"Data/{user_id}/Trajectory")
            if not os.path.exists(trajectory_dir): continue

            for traj_file in os.listdir(trajectory_dir):
                if not traj_file.endswith('.plt'): continue
                try:
                    trajectory = geolife_loader.load_trajectory(os.path.join(trajectory_dir, traj_file))
                    all_segments.extend(geolife_loader.segment_trajectory(trajectory, labels))
                except: continue

        processed_segments = preprocess_trajectory_segments(all_segments, min_length=10)
        valid_modes = {'Walk', 'Bike', 'Bus', 'Car & taxi', 'Train', 'Subway'}
        processed_segments = [(t, l) for t, l in processed_segments if l in valid_modes]

    # 特征提取阶段
    if not processed_segments:
        print("错误: 没有可用轨迹段")
        return [], spatial_extractor, None

    # 对少数类进行数据增强（仅对训练数据有效，此处对全量做增强后再split）
    processed_segments = BaseGeoLifePreprocessor.oversample_minority_classes(
        processed_segments,
        target_ratio=0.3,
        minority_classes=['Subway', 'Train']
    )

    all_labels_str = [label for _, label in processed_segments]
    label_encoder = LabelEncoder().fit(all_labels_str)

    print("\n========== 阶段 3: 特征提取 ==========")
    feature_extractor = FeatureExtractor(spatial_extractor)
    all_features_and_labels = []

    for trajectory, segment_stats, label_str in tqdm(processed_segments, desc="[Exp3 特征提取]"):
        try:
            trajectory_features, spatial_features = feature_extractor.extract_features(trajectory)
            label_encoded = label_encoder.transform([label_str])[0]
            all_features_and_labels.append((trajectory_features, spatial_features, segment_stats, label_encoded))
        except Exception:
            continue

    # 保存各级缓存
    with open(PROCESSED_FEATURE_CACHE_PATH, 'wb') as f:
        pickle.dump((all_features_and_labels, label_encoder, cleaning_stats), f, protocol=pickle.HIGHEST_PROTOCOL)
    spatial_extractor.save_cache(SPATIAL_GRID_CACHE_PATH)
    save_cache_metadata(osm_path, geolife_root, len(all_features_and_labels), label_encoder)

    return all_features_and_labels, spatial_extractor, label_encoder, cleaning_stats


# ============================================================
# Main (✅ 修改 3: 更新 main 函数添加命令行参数)
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='训练交通方式识别模型 (Exp3)')
    parser.add_argument('--geolife_root', type=str, default='../data/Geolife Trajectories 1.3')
    parser.add_argument('--osm_path', type=str, default='../data/exp3.geojson')

    # ===== ✅ 新增参数 =====
    parser.add_argument('--use_base_data', action='store_true', default=True, help='使用预处理的基础数据（推荐）')
    parser.add_argument('--cleaning_mode', type=str, default='balanced',
                       choices=['strict', 'balanced', 'gentle'],
                       help='数据清洗模式: strict(严格), balanced(平衡), gentle(温和)')
    # ===== 新增结束 =====

    parser.add_argument('--max_users', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--generate_cache_only', action='store_true', help='仅生成缓存')
    parser.add_argument('--use_cached_data', action='store_true', help='直接使用缓存数据')
    parser.add_argument('--clear_cache', action='store_true', help='清空缓存')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.clear_cache:
        for f in [SPATIAL_CACHE_PATH, SPATIAL_GRID_CACHE_PATH, PROCESSED_FEATURE_CACHE_PATH, META_CACHE_PATH]:
            if os.path.exists(f): os.remove(f)

    # ========== 步骤1: 加载数据 ==========
    if args.use_cached_data:
        if not validate_cache(args.osm_path): return
        with open(PROCESSED_FEATURE_CACHE_PATH, 'rb') as f:
            all_features_and_labels, label_encoder = pickle.load(f)
        with open(SPATIAL_CACHE_PATH, 'rb') as f:
            spatial_extractor = pickle.load(f)
    else:
        # ✅ 传递新参数
        all_features_and_labels, spatial_extractor, label_encoder, cleaning_stats = load_data(
            args.geolife_root, args.osm_path, args.max_users,
            use_base_data=args.use_base_data,
            cleaning_mode=args.cleaning_mode
        )

    if args.generate_cache_only or not all_features_and_labels:
        return

    # ========== 步骤2: 训练模型 ==========
    print(f"\n开始训练 (类别: {label_encoder.classes_})")

    # 第一步：先划分索引（不需要 dataset）
    all_indices = np.arange(len(all_features_and_labels))
    labels_stratify = [label_encoder.inverse_transform([label_encoded])[0] for _, _, label_encoded in all_features_and_labels]

    train_indices, temp_indices = train_test_split(
        all_indices, test_size=0.3, random_state=42,
        stratify=labels_stratify
    )
    temp_labels = [labels_stratify[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.6667, random_state=42,
        stratify=temp_labels
    )

    # 第二步：用训练集计算归一化统计量
    from common.train_utils import compute_feature_stats
    train_segments = [all_features_and_labels[i] for i in train_indices]
    traj_list = [s[0] for s in train_segments]
    spatial_list = [s[1] for s in train_segments]
    stats_list = [s[2] for s in train_segments]

    traj_all = np.vstack(traj_list)
    spatial_all = np.vstack(spatial_list)
    stats_all = np.vstack(stats_list)

    traj_mean = traj_all.mean(axis=0).astype(np.float32)
    traj_std = traj_all.std(axis=0).astype(np.float32)
    traj_std = np.where(traj_std < 1e-6, 1.0, traj_std)

    spatial_mean = spatial_all.mean(axis=0).astype(np.float32)
    spatial_std = spatial_all.std(axis=0).astype(np.float32)
    spatial_std = np.where(spatial_std < 1e-6, 1.0, spatial_std)

    stats_mean = stats_all.mean(axis=0).astype(np.float32)
    stats_std = stats_all.std(axis=0).astype(np.float32)
    stats_std = np.where(stats_std < 1e-6, 1.0, stats_std)

    norm_params = {
        'traj_mean': traj_mean, 'traj_std': traj_std,
        'spatial_mean': spatial_mean, 'spatial_std': spatial_std,
        'stats_mean': stats_mean, 'stats_std': stats_std
    }
    print(f"\n✅ 归一化统计量计算完成（基于 {len(train_indices)} 个训练样本）")

    # 第三步：创建带归一化的 dataset
    dataset = TrajectoryDataset(
        all_features_and_labels,
        traj_mean=traj_mean, traj_std=traj_std,
        spatial_mean=spatial_mean, spatial_std=spatial_std,
        stats_mean=stats_mean, stats_std=stats_std
    )

    print(f"\n✅ 数据集大小:")
    print(f"  总样本数: {len(dataset)}")
    print(f"  特征样本数: {len(all_features_and_labels)}")

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"\n✅ 数据加载完成:")
    print(f"  Train: {len(train_indices)} 样本")
    print(f"  Val:   {len(val_indices)} 样本")
    print(f"  Test:  {len(test_indices)} 样本")
    print(f"  训练批次总数: {len(train_loader)}")
    print(f"  验证批次总数: {len(val_loader)}")

    print(f"\n类别分布:")
    for cls in label_encoder.classes_:
        train_count = sum(1 for i in train_indices if labels_stratify[i] == cls)
        val_count = sum(1 for i in val_indices if labels_stratify[i] == cls)
        test_count = sum(1 for i in test_indices if labels_stratify[i] == cls)
        print(f"  {cls:15s}: Train={train_count}, Val={val_count}, Test={test_count}")

    model = TransportationModeClassifier(
        TRAJECTORY_FEATURE_DIM, SPATIAL_FEATURE_DIM, 18, args.hidden_dim, args.num_layers, len(label_encoder.classes_), args.dropout,
        num_segments=5,
        local_hidden=64,
        global_hidden=128,
    ).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Warmup scheduler：前5轮线性升温
    def warmup_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warmup_lambda
    )

    # ReduceLROnPlateau：验证损失不改善时降低学习率
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8
    )

    class_weights = compute_class_weights(
        label_encoder,
        all_features_and_labels,
        label_index=-1,
        mode='sqrt_inverse'
    ).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ========================================================
    # ✅ Early Stopping 配置
    # ========================================================
    CHECKPOINT_PATH = os.path.join(args.save_dir, 'exp3_model.pth')

    # 加载历史最佳 val_loss 作为初始基准
    best_val_loss = float("inf")
    start_epoch = 0

    if os.path.exists(CHECKPOINT_PATH):
        try:
            prev = torch.load(CHECKPOINT_PATH, map_location=args.device, weights_only=False)
            if 'val_loss' in prev:
                best_val_loss = prev['val_loss']
            if 'resume' in prev and prev['resume']:
                model.load_state_dict(prev['model_state_dict'])
                optimizer.load_state_dict(prev['optimizer_state_dict'])
                start_epoch = prev['epoch'] + 1
                print(f"✅ 从 epoch {start_epoch} 继续训练，历史最佳 val_loss={best_val_loss:.4f}")
            else:
                print(f"✅ 检测到历史最佳模型，val_loss={best_val_loss:.4f}，新训练需超过此值才覆盖")
        except Exception as e:
            print(f"⚠️ 历史模型加载失败（{e}），从零开始")

    epochs_no_improve = 0
    patience = args.patience

    # ========================================================
    # ✅ 训练曲线保存到 CSV
    # ========================================================
    import csv
    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/exp3_training_log.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)

        val_loss, report, _, _ = evaluate(model, val_loader, criterion, args.device, label_encoder)
        val_acc = report['accuracy']

        # 在训练循环结束后再打印上一轮指标汇总
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # 学习率调度
        if epoch < 5:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")

        # 写入训练日志
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{train_loss:.4f}", f"{train_acc:.4f}",
                           f"{val_loss:.4f}", f"{val_acc:.4f}",
                           f"{optimizer.param_groups[0]['lr']:.6f}"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_encoder': label_encoder,
                'val_loss': val_loss,
                'norm_params': norm_params,
                'resume': True,
                'model_config': {
                    'trajectory_feature_dim': TRAJECTORY_FEATURE_DIM, 'spatial_feature_dim': SPATIAL_FEATURE_DIM,
                    'hidden_dim': args.hidden_dim, 'num_layers': args.num_layers,
                    'num_classes': len(label_encoder.classes_), 'dropout': args.dropout
                }
            }, os.path.join(args.save_dir, 'exp3_model.pth'))
            print("✓ 保存最佳模型（基于验证集）")
        else:
            epochs_no_improve += 1
            print(f"⏳ 验证损失未改善: {epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                print(f"\n🛑 Early stopping 触发（patience={patience}）")
                break

    # ========================================================
    # ✅ 在测试集上进行最终评估
    # ========================================================
    print("\n" + "=" * 80)
    print("最终测试集评估")
    print("=" * 80)

    test_loss, test_report, _, _ = evaluate(model, test_loader, criterion, args.device, label_encoder)
    test_acc = test_report['accuracy']

    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print("\n各类别详细指标:")
    for cls in label_encoder.classes_:
        if cls in test_report:
            metrics = test_report[cls]
            print(f"  {cls:15s}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()