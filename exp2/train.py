import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import pickle
import warnings
import pandas as pd
import numpy as np
from typing import List, Tuple
import torch.serialization
import sys

# ===== ✅ 修改 1：支持基础数据导入 =====
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (BaseGeoLifePreprocessor, Exp2DataAdapter,
                     train_epoch, evaluate, compute_class_weights)
# =====================================

try:
    from src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader, preprocess_trajectory_segments
    from src.feature_extraction import FeatureExtractor
    from src.model import TransportationModeClassifier
    from src.osm_feature_extractor import OsmSpatialExtractor
except ImportError:
    pass

# 特征维度常量
TRAJECTORY_FEATURE_DIM = 21   # 9轨迹 + 12空间
SPATIAL_FEATURE_DIM = 1       # 占位，不实际使用
FIXED_SEQUENCE_LENGTH = 50

# 缓存配置
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
SPATIAL_CACHE_PATH = os.path.join(CACHE_DIR, 'spatial_data.pkl')
SPATIAL_GRID_CACHE_PATH = os.path.join(CACHE_DIR, 'spatial_grid_cache.pkl')
PROCESSED_SEGMENTS_CACHE_PATH = os.path.join(CACHE_DIR, 'processed_segments.pkl')
PROCESSED_FEATURE_CACHE_PATH = os.path.join(CACHE_DIR, 'processed_features.pkl')
os.makedirs(CACHE_DIR, exist_ok=True)

class TrajectoryDataset(Dataset):
    def __init__(self, all_features_and_labels: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]],
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
        trajectory_features, _, segment_stats, label_encoded = self.data[idx]

        # 归一化
        if self.traj_mean is not None:
            trajectory_features = (trajectory_features - self.traj_mean) / self.traj_std
        if self.stats_mean is not None:
            segment_stats = (segment_stats - self.stats_mean) / self.stats_std

        # spatial已融合进traj，placeholder不需要归一化
        placeholder = np.zeros((trajectory_features.shape[0], 1), dtype=np.float32)
        return (torch.FloatTensor(trajectory_features),
                torch.FloatTensor(placeholder),
                torch.FloatTensor(segment_stats),
                torch.LongTensor([label_encoded])[0])

# ===== ✅ 修改 2：load_data 函数完整更新 =====
def load_data(geolife_root: str, osm_path: str, max_users: int = None, use_base_data: bool = True, cleaning_mode: str = 'balanced'):
    """加载所有数据，实现快速模式与传统模式切换

    Args:
        geolife_root: GeoLife数据根目录
        osm_path: OSM数据路径
        max_users: 最大用户数
        use_base_data: 是否使用预处理的基础数据
        cleaning_mode: 数据清洗模式 ('strict', 'balanced', 'gentle')
    """

    BASE_DATA_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data/processed/base_segments.pkl'
    )

    # 1. 空间特征提取器构建 (保持原有逻辑)
    spatial_extractor = None
    if os.path.exists(SPATIAL_CACHE_PATH):
        print(f"\n========== 阶段 1: 空间特征提取器加载 (从缓存) ==========")
        try:
            with open(SPATIAL_CACHE_PATH, 'rb') as f:
                with torch.serialization.safe_globals({LabelEncoder: LabelEncoder}):
                    spatial_extractor = pickle.load(f)
            if os.path.exists(SPATIAL_GRID_CACHE_PATH): spatial_extractor.load_cache(SPATIAL_GRID_CACHE_PATH)
        except Exception as e:
            spatial_extractor = None

    if spatial_extractor is None:
        print("\n========== 阶段 1: 空间特征提取器构建 (重建) ==========")
        osm_loader = OSMDataLoader(osm_path)
        osm_data = osm_loader.load_osm_data()
        spatial_extractor = OsmSpatialExtractor()
        spatial_extractor.build_from_osm(osm_loader.extract_road_network(osm_data), osm_loader.extract_pois(osm_data))
        with open(SPATIAL_CACHE_PATH, 'wb') as f:
            pickle.dump(spatial_extractor, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 2. 数据准备
    processed_segments = None
    label_encoder = None
    cleaning_stats = {}

    # B. ✅ 快速模式逻辑
    if use_base_data and os.path.exists(BASE_DATA_PATH):
        print(f"\n========== 阶段 2: 使用预处理基础数据 (快速模式 - 清洗模式: {cleaning_mode}) ==========")
        base_segments = BaseGeoLifePreprocessor.load_from_cache(BASE_DATA_PATH)
        adapter = Exp2DataAdapter(enable_cleaning=True, cleaning_mode=cleaning_mode)
        processed_segments = adapter.process_segments(base_segments)
        cleaning_stats = adapter.get_cleaning_stats()

        TARGET_MODES = ['Walk', 'Bike', 'Bus', 'Car & taxi', 'Train', 'Subway']
        processed_segments = [s for s in processed_segments if s[2] in TARGET_MODES]

        all_labels_str = [label for _, _, label in processed_segments]
        label_encoder = LabelEncoder().fit(all_labels_str)
        print(f"✅ 基础数据适配完成: {len(processed_segments)} 个段")

    # C. 传统模式逻辑
    else:
        if use_base_data: print(f"⚠️ 基础数据不存在，回退传统模式")
        if os.path.exists(PROCESSED_SEGMENTS_CACHE_PATH):
            print(f"\n========== 阶段 2.1: 轨迹段加载 (从缓存) ==========")
            with open(PROCESSED_SEGMENTS_CACHE_PATH, 'rb') as f:
                with torch.serialization.safe_globals({LabelEncoder: LabelEncoder}):
                    processed_segments, label_encoder = pickle.load(f)

        if processed_segments is None:
            print("\n========== 阶段 2.1: 轨迹段加载 (从原始文件) ==========")
            geolife_loader = GeoLifeDataLoader(geolife_root)
            users = geolife_loader.get_all_users()
            if max_users: users = users[:max_users]

            all_segments = []
            for user_id in tqdm(users, desc="[用户加载]"):
                labels = geolife_loader.load_labels(user_id)
                if labels.empty: continue
                trajectory_dir = os.path.join(geolife_root, "Data", user_id, "Trajectory")
                for traj_file in os.listdir(trajectory_dir):
                    if not traj_file.endswith('.plt'): continue
                    try:
                        traj = geolife_loader.load_trajectory(os.path.join(trajectory_dir, traj_file))
                        all_segments.extend(geolife_loader.segment_trajectory(traj, labels))
                    except: continue

            processed_segments = preprocess_trajectory_segments(all_segments, min_length=10)
            final_six_modes = {'Walk', 'Bike', 'Bus', 'Car & taxi', 'Train', 'Subway'}
            processed_segments = [(t, l) for t, l in processed_segments if l in final_six_modes]

            label_encoder = LabelEncoder().fit([l for _, l in processed_segments])
            with open(PROCESSED_SEGMENTS_CACHE_PATH, 'wb') as f:
                pickle.dump((processed_segments, label_encoder), f, protocol=pickle.HIGHEST_PROTOCOL)

    # 3. 特征提取 (阶段 D)
    print("\n========== 2.2: 特征提取 ==========")
    feature_extractor = FeatureExtractor(spatial_extractor)
    all_features_and_labels = []
    for traj, stats, label_str in tqdm(processed_segments, desc="[特征提取]"):
        try:
            trajectory_features, spatial_features = feature_extractor.extract_features(traj)
            label_encoded = label_encoder.transform([label_str])[0]
            all_features_and_labels.append((trajectory_features, spatial_features, stats, label_encoded))
        except: continue

    all_features_and_labels = [
        item for item in all_features_and_labels
        if not (np.isnan(item[0]).any() or np.isinf(item[0]).any() or
                np.isnan(item[1]).any() or np.isinf(item[1]).any() or
                np.isnan(item[2]).any() or np.isinf(item[2]).any())
    ]
    print(f"✅ NaN/Inf过滤后剩余: {len(all_features_and_labels)} 个样本")

    with open(PROCESSED_FEATURE_CACHE_PATH, 'wb') as f:
        pickle.dump((all_features_and_labels, label_encoder, cleaning_stats), f, protocol=pickle.HIGHEST_PROTOCOL)
    spatial_extractor.save_cache(SPATIAL_GRID_CACHE_PATH)
    return all_features_and_labels, spatial_extractor, label_encoder, cleaning_stats

# ===== ✅ 修改 3：main 函数参数接入 =====
def main():
    parser = argparse.ArgumentParser(description='训练交通方式识别模型 (Exp2 优化版)')
    parser.add_argument('--geolife_root', type=str, default='../data/Geolife Trajectories 1.3')
    parser.add_argument('--osm_path', type=str, default='../data/exp2.geojson')

    # 新增参数
    parser.add_argument('--use_base_data', action='store_true', default=True, help='使用预处理的基础数据')
    parser.add_argument('--cleaning_mode', type=str, default='balanced',
                       choices=['strict', 'balanced', 'gentle'],
                       help='数据清洗模式: strict(严格), balanced(平衡), gentle(温和)')

    parser.add_argument('--max_users', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--save_dir', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints'))
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--clear_cache', action='store_true')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.clear_cache:
        for f in [SPATIAL_CACHE_PATH, SPATIAL_GRID_CACHE_PATH, PROCESSED_SEGMENTS_CACHE_PATH, PROCESSED_FEATURE_CACHE_PATH]:
            if os.path.exists(f): os.remove(f)

    # 传递 use_base_data 参数
    all_features_and_labels, spatial_extractor, label_encoder, cleaning_stats = load_data(
        args.geolife_root, args.osm_path, args.max_users,
        use_base_data=args.use_base_data,
        cleaning_mode=args.cleaning_mode
    )

    if not all_features_and_labels: return

    # 第一步：先划分索引（不需要 dataset）
    all_indices = np.arange(len(all_features_and_labels))
    labels_stratify = [label_encoder.inverse_transform([label_encoded])[0] for _, _, _, label_encoded in all_features_and_labels]

    train_indices, temp_indices = train_test_split(
        all_indices, test_size=0.3, random_state=42,
        stratify=labels_stratify
    )
    temp_labels = [labels_stratify[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.6667, random_state=42,
        stratify=temp_labels
    )

    # 第二步：对训练集进行数据增强（避免数据泄露）
    train_segments = [all_features_and_labels[i] for i in train_indices]
    train_labels_str = [labels_stratify[i] for i in train_indices]
    
    # 构造训练集的 segments 格式 (traj, stats, label)
    train_segments_for_oversample = [(s[0], s[2], train_labels_str[i]) for i, s in enumerate(train_segments)]
    train_segments_oversampled = BaseGeoLifePreprocessor.oversample_minority_classes(
        train_segments_for_oversample,
        target_ratio=0.3,
        minority_classes=['Subway', 'Train']
    )
    
    # 将增强后的 segments 转换回特征格式
    train_features_oversampled = []
    for traj, stats, label_str in train_segments_oversampled:
        label_encoded = label_encoder.transform([label_str])[0]
        train_features_oversampled.append((traj, None, stats, label_encoded))
    
    # 第三步：用训练集计算归一化统计量
    traj_list = [s[0] for s in train_features_oversampled]
    stats_list = [s[2] for s in train_features_oversampled]

    traj_all = np.vstack(traj_list)
    stats_all = np.vstack(stats_list)

    traj_mean = traj_all.mean(axis=0).astype(np.float32)
    traj_std = traj_all.std(axis=0).astype(np.float32)
    traj_std = np.where(traj_std < 1e-6, 1.0, traj_std)

    stats_mean = stats_all.mean(axis=0).astype(np.float32)
    stats_std = stats_all.std(axis=0).astype(np.float32)
    stats_std = np.where(stats_std < 1e-6, 1.0, stats_std)

    norm_params = {
        'traj_mean': traj_mean, 'traj_std': traj_std,
        'stats_mean': stats_mean, 'stats_std': stats_std
    }
    print(f"\n✅ 归一化统计量计算完成（基于 {len(train_features_oversampled)} 个训练样本）")

    # 第四步：创建包含所有数据的 dataset（验证集和测试集使用原始数据）
    all_features = train_features_oversampled + [all_features_and_labels[i] for i in val_indices] + [all_features_and_labels[i] for i in test_indices]
    dataset = TrajectoryDataset(
        all_features,
        traj_mean=traj_mean, traj_std=traj_std,
        stats_mean=stats_mean, stats_std=stats_std
    )

    # 更新索引：训练集在前，验证集在中间，测试集在后
    new_train_indices = list(range(len(train_features_oversampled)))
    new_val_indices = list(range(len(train_features_oversampled), len(train_features_oversampled) + len(val_indices)))
    new_test_indices = list(range(len(train_features_oversampled) + len(val_indices), len(all_features)))

    print(f"\n✅ 数据集大小:")
    print(f"  总样本数: {len(dataset)}")
    print(f"  训练样本数: {len(new_train_indices)}")
    print(f"  验证样本数: {len(new_val_indices)}")
    print(f"  测试样本数: {len(new_test_indices)}")

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, new_train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, new_val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, new_test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"\n✅ 数据加载完成:")
    print(f"  Train: {len(new_train_indices)} 样本")
    print(f"  Val:   {len(new_val_indices)} 样本")
    print(f"  Test:  {len(new_test_indices)} 样本")
    print(f"  训练批次总数: {len(train_loader)}")
    print(f"  验证批次总数: {len(val_loader)}")
    print(f"  测试批次总数: {len(test_loader)}")

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

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Warmup scheduler：前10轮线性升温
    def warmup_lambda(epoch):
        warmup_epochs = 10
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warmup_lambda
    )

    # ReduceLROnPlateau：验证准确率不改善时降低学习率
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8
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
    CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints/exp2_model.pth')

    # 加载历史最佳 val_acc 作为初始基准
    best_val_acc = 0.0
    start_epoch = 0

    if os.path.exists(CHECKPOINT_PATH):
        try:
            prev = torch.load(CHECKPOINT_PATH, map_location=args.device, weights_only=False)
            prev_config = prev.get('model_config', {})

            # 检查输入维度是否一致
            prev_input_dim = prev_config.get('input_dim', prev_config.get('trajectory_feature_dim'))
            curr_input_dim = TRAJECTORY_FEATURE_DIM  # 21

            if prev_input_dim != curr_input_dim:
                print(f"⚠️ 输入维度不匹配（历史={prev_input_dim}, 当前={curr_input_dim}），从零开始训练")
                # 不加载权重，best_val_acc保持0.0
            elif prev_config.get('num_classes') == len(label_encoder.classes_):
                model.load_state_dict(prev['model_state_dict'])
                if 'val_acc' in prev:
                    best_val_acc = prev['val_acc']
                print(f"✅ 加载历史最佳权重，val_acc={best_val_acc:.4f}，从 epoch 0 重新训练")
            else:
                print(f"⚠️ 模型类别数不匹配，从零开始")
        except Exception as e:
            print(f"⚠️ 历史模型加载失败（{e}），从零开始")

    epochs_no_improve = 0
    patience = args.patience
    os.makedirs("checkpoints", exist_ok=True)

    # 连续 NaN 检测计数器
    consecutive_nan_count = 0
    MAX_CONSECUTIVE_NAN = 3  # 连续3次NaN则停止训练

    # ========================================================
    # ✅ 训练曲线保存到 CSV
    # ========================================================
    import csv
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    csv_path = os.path.join(logs_dir, 'exp2_training_log.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)

        # NaN 检测
        if np.isnan(train_loss) or np.isinf(train_loss):
            consecutive_nan_count += 1
            print(f"⚠️ 检测到训练损失异常（{train_loss}），连续NaN次数: {consecutive_nan_count}/{MAX_CONSECUTIVE_NAN}")
            
            if consecutive_nan_count >= MAX_CONSECUTIVE_NAN:
                print(f"🛑 连续{MAX_CONSECUTIVE_NAN}次NaN，停止训练")
                break
            
            if os.path.exists(CHECKPOINT_PATH):
                try:
                    prev = torch.load(CHECKPOINT_PATH, map_location=args.device, weights_only=False)
                    model.load_state_dict(prev['model_state_dict'])
                    print(f"✅ 已恢复到最佳模型")
                except Exception as e:
                    print(f"⚠️ 模型恢复失败（{e}），从零开始")
            continue  # 跳过本轮，不降lr
        else:
            consecutive_nan_count = 0  # 正常训练，重置计数器

        val_loss, report, _, _ = evaluate(model, val_loader, criterion, args.device, label_encoder.classes_)
        val_acc = report['accuracy']

        # NaN 检测
        if np.isnan(val_loss) or np.isinf(val_loss):
            consecutive_nan_count += 1
            print(f"⚠️ 检测到验证损失异常（{val_loss}），连续NaN次数: {consecutive_nan_count}/{MAX_CONSECUTIVE_NAN}")
            
            if consecutive_nan_count >= MAX_CONSECUTIVE_NAN:
                print(f"🛑 连续{MAX_CONSECUTIVE_NAN}次NaN，停止训练")
                break
            
            if os.path.exists(CHECKPOINT_PATH):
                try:
                    prev = torch.load(CHECKPOINT_PATH, map_location=args.device, weights_only=False)
                    model.load_state_dict(prev['model_state_dict'])
                    print(f"✅ 已恢复到最佳模型")
                except Exception as e:
                    print(f"⚠️ 模型恢复失败（{e}），从零开始")
            continue  # 跳过本轮，不降lr

        # 在训练循环结束后再打印上一轮指标汇总
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # 学习率调度
        if epoch < 10:
            warmup_scheduler.step()
        else:
            # warmup 结束后，重置 plateau_scheduler 的内部状态
            if epoch == 10:
                plateau_scheduler.num_bad_epochs = 0
                print("🔄 Warmup 结束，重置 ReduceLROnPlateau 状态")
            plateau_scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")

        # 写入训练日志
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{train_loss:.4f}", f"{train_acc:.4f}",
                           f"{val_loss:.4f}", f"{val_acc:.4f}",
                           f"{optimizer.param_groups[0]['lr']:.6f}"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_encoder': label_encoder,
                'val_acc': val_acc,
                'norm_params': norm_params,
                'resume': True,
                'model_config': {
                    'fused_input': True,
                    'trajectory_feature_dim': 9,
                    'spatial_feature_dim': 12,
                    'combined_dim': 21,
                    'input_dim': TRAJECTORY_FEATURE_DIM,
                    'hidden_dim': args.hidden_dim, 'num_layers': args.num_layers, 'num_classes': len(label_encoder.classes_), 'dropout': args.dropout
                }
            }, os.path.join(args.save_dir, 'exp2_model.pth'))
            print("✓ 保存最佳模型（基于验证集）")
        else:
            epochs_no_improve += 1
            print(f"⏳ 验证准确率未改善: {epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                print(f"\n🛑 Early stopping 触发（patience={patience}）")
                break

    # ========================================================
    # ✅ 在测试集上进行最终评估
    # ========================================================
    print("\n" + "=" * 80)
    print("最终测试集评估")
    print("=" * 80)

    # 加载最佳checkpoint进行最终评估
    best_ckpt = torch.load(
        os.path.join(args.save_dir, 'exp2_model.pth'),
        map_location=args.device, weights_only=False
    )
    model.load_state_dict(best_ckpt['model_state_dict'])
    print("✅ 已加载最佳checkpoint进行最终评估")

    test_loss, test_report, all_preds, all_labels = evaluate(model, test_loader, criterion, args.device, label_encoder.classes_)
    test_acc = test_report['accuracy']

    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print("\n各类别详细指标:")
    for cls in label_encoder.classes_:
        if cls in test_report:
            metrics = test_report[cls]
            print(f"  {cls:15s}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵（行=真实，列=预测）:")
    classes = label_encoder.classes_
    print(f"{'':15s}", end="")
    for c in classes:
        print(f"{c[:6]:>8s}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{classes[i]:15s}", end="")
        for val in row:
            print(f"{val:8d}", end="")
        print()

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    try:
        with torch.serialization.safe_globals({LabelEncoder: LabelEncoder}):
            main()
    except Exception as e:
        print(f"Error: {e}")