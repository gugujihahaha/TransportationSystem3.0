"""
exp2/train.py（修复版）
======================
修复内容：
1. evaluate() 返回值从4个改为5个
2. 训练完成后生成共享划分（shared_split.pkl），供 EXP1/3/4 使用
3. 特征缓存格式明确：(traj_21, spatial_placeholder, stats_18, label_encoded)

注意：EXP2 必须第一个训练，因为：
    - EXP2 的特征缓存是所有实验的数据基础
    - EXP2 训练完成后自动生成 shared_split.pkl
    - EXP3 依赖 EXP2 的 spatial_data.pkl
"""

import argparse
import csv
import os
import pickle
import sys
import warnings
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.serialization

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
os.chdir(SCRIPT_DIR)

from common import (BaseGeoLifePreprocessor, Exp2DataAdapter,
                    train_epoch, evaluate, compute_class_weights)

try:
    from src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader, preprocess_trajectory_segments
    from src.feature_extraction import FeatureExtractor
    from src.model import TransportationModeClassifier
    from src.osm_feature_extractor import OsmSpatialExtractor
except ImportError:
    pass

TRAJECTORY_FEATURE_DIM = 21
FIXED_SEQUENCE_LENGTH  = 50

CACHE_DIR                  = os.path.join(SCRIPT_DIR, 'cache')
SPATIAL_CACHE_PATH         = os.path.join(CACHE_DIR, 'spatial_data.pkl')
SPATIAL_GRID_CACHE_PATH    = os.path.join(CACHE_DIR, 'spatial_grid_cache.pkl')
PROCESSED_FEATURE_CACHE    = os.path.join(CACHE_DIR, 'processed_features.pkl')
SHARED_SPLIT_PATH          = os.path.join(PARENT_DIR, 'data', 'processed', 'shared_split.pkl')
os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# Dataset
# ============================================================
class TrajectoryDataset(Dataset):
    def __init__(self, all_features_and_labels,
                 traj_mean=None, traj_std=None,
                 stats_mean=None, stats_std=None):
        self.data = all_features_and_labels
        self.traj_mean  = traj_mean
        self.traj_std   = traj_std
        self.stats_mean = stats_mean
        self.stats_std  = stats_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 格式: (traj_21, spatial_placeholder, stats_18, label_encoded)
        traj_21, _, stats, label_encoded = self.data[idx]

        traj  = traj_21.astype(np.float32)
        stats = stats.astype(np.float32)

        if self.traj_mean is not None:
            traj = (traj - self.traj_mean) / self.traj_std
        if self.stats_mean is not None:
            stats = (stats - self.stats_mean) / self.stats_std

        return (torch.FloatTensor(traj),
                torch.FloatTensor(stats),
                torch.LongTensor([label_encoded])[0])


# ============================================================
# 数据加载（与原版相同，保留特征缓存逻辑）
# ============================================================
def load_data(geolife_root, osm_path, max_users=None,
              use_base_data=True, cleaning_mode='balanced',
              use_cleaned_balanced=False):
    BASE_DATA_PATH = os.path.join(PARENT_DIR, 'data', 'processed', 'base_segments.pkl')
    CLEANED_BALANCED_PATH = os.path.join(PARENT_DIR, 'data', 'processed', 'cleaned_balanced.pkl')

    spatial_extractor = None
    if os.path.exists(SPATIAL_CACHE_PATH):
        try:
            with open(SPATIAL_CACHE_PATH, 'rb') as f:
                spatial_extractor = pickle.load(f)
            if os.path.exists(SPATIAL_GRID_CACHE_PATH):
                spatial_extractor.load_cache(SPATIAL_GRID_CACHE_PATH)
        except Exception:
            spatial_extractor = None

    if spatial_extractor is None:
        print("\n========== 构建空间特征提取器 ==========")
        osm_loader = OSMDataLoader(osm_path)
        osm_data   = osm_loader.load_osm_data()
        spatial_extractor = OsmSpatialExtractor()
        spatial_extractor.build_from_osm(
            osm_loader.extract_road_network(osm_data),
            osm_loader.extract_pois(osm_data)
        )
        with open(SPATIAL_CACHE_PATH, 'wb') as f:
            pickle.dump(spatial_extractor, f, protocol=pickle.HIGHEST_PROTOCOL)

    if use_cleaned_balanced and os.path.exists(CLEANED_BALANCED_PATH):
        print("\n========== 使用cleaned_balanced.pkl（与共享测试集一致）==========")
        with open(CLEANED_BALANCED_PATH, 'rb') as f:
            cleaned_data = pickle.load(f)
        
        # 过滤掉Airplane类别
        TARGET_MODES = ['Walk', 'Bike', 'Bus', 'Car & taxi', 'Train', 'Subway']
        processed_segments = []
        for traj, stats, datetime_series, label in cleaned_data:
            if label in TARGET_MODES:
                processed_segments.append((traj, stats, label))
        
        cleaning_stats = {'before': {}, 'after': {}, 'cleaner': {}}
        all_labels_str = [s[2] for s in processed_segments]
        label_encoder = LabelEncoder().fit(all_labels_str)
    elif use_base_data and os.path.exists(BASE_DATA_PATH):
        print("\n========== 快速模式：基础数据 ==========")
        base_segments = BaseGeoLifePreprocessor.load_from_cache(BASE_DATA_PATH)
        adapter = Exp2DataAdapter(enable_cleaning=True, cleaning_mode=cleaning_mode)
        processed_segments = adapter.process_segments(base_segments)
        cleaning_stats = adapter.get_cleaning_stats()

        TARGET_MODES = ['Walk', 'Bike', 'Bus', 'Car & taxi', 'Train', 'Subway']
        processed_segments = [s for s in processed_segments if s[2] in TARGET_MODES]

        all_labels_str = [s[2] for s in processed_segments]
        label_encoder = LabelEncoder().fit(all_labels_str)
    else:
        raise RuntimeError("请先生成 base_segments.pkl 或 cleaned_balanced.pkl，再运行此脚本")

    print("\n========== 特征提取 ==========")
    feature_extractor = FeatureExtractor(spatial_extractor)
    all_features = []
    for traj, stats, label_str in tqdm(processed_segments, desc="[特征提取]"):
        try:
            traj_21, spatial = feature_extractor.extract_features(traj)
            label_encoded = label_encoder.transform([label_str])[0]
            all_features.append((traj_21, spatial, stats, label_encoded))
        except Exception:
            continue

    all_features = [
        item for item in all_features
        if not (np.isnan(item[0]).any() or np.isinf(item[0]).any() or
                np.isnan(item[2]).any() or np.isinf(item[2]).any())
    ]
    print(f"✅ 过滤后: {len(all_features)} 个样本")

    with open(PROCESSED_FEATURE_CACHE, 'wb') as f:
        pickle.dump((all_features, label_encoder, cleaning_stats),
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ EXP2 特征缓存已保存: {PROCESSED_FEATURE_CACHE}")

    spatial_extractor.save_cache(SPATIAL_GRID_CACHE_PATH)
    return all_features, label_encoder, cleaning_stats


def generate_shared_split(all_features, label_encoder):
    """
    训练完成后，基于 EXP2 的完整特征集生成共享划分。
    所有实验都使用这个划分，确保测试集完全一致。
    """
    print("\n========== 生成共享划分 ==========")
    n = len(all_features)
    all_indices = np.arange(n)
    labels_encoded = [item[3] for item in all_features]
    labels_str = label_encoder.inverse_transform(labels_encoded)

    train_indices, temp_indices = train_test_split(
        all_indices, test_size=0.3, random_state=42, stratify=labels_str
    )
    temp_labels = [labels_str[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.6667, random_state=42, stratify=temp_labels
    )

    split_data = {
        'train_indices': train_indices.tolist(),
        'val_indices':   val_indices.tolist(),
        'test_indices':  test_indices.tolist(),
        'n_total':       n,
        'label_encoder': label_encoder,
    }

    os.makedirs(os.path.dirname(SHARED_SPLIT_PATH), exist_ok=True)
    with open(SHARED_SPLIT_PATH, 'wb') as f:
        pickle.dump(split_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ 共享划分已保存: {SHARED_SPLIT_PATH}")
    print(f"   Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    return train_indices, val_indices, test_indices


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--geolife_root',  default='../data/Geolife Trajectories 1.3')
    parser.add_argument('--osm_path',      default='../data/exp2.geojson')
    parser.add_argument('--use_base_data', action='store_true', default=False)
    parser.add_argument('--use_cleaned_balanced', action='store_true', default=True)
    parser.add_argument('--cleaning_mode', default='balanced',
                        choices=['strict', 'balanced', 'gentle'])
    parser.add_argument('--max_users',  type=int,   default=None)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--epochs',     type=int,   default=150)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int,   default=128)
    parser.add_argument('--num_layers', type=int,   default=2)
    parser.add_argument('--dropout',    type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience',   type=int,   default=25)
    parser.add_argument('--save_dir',   default=os.path.join(SCRIPT_DIR, 'checkpoints'))
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--clear_cache', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.clear_cache:
        for f in [SPATIAL_CACHE_PATH, SPATIAL_GRID_CACHE_PATH, PROCESSED_FEATURE_CACHE]:
            if os.path.exists(f):
                os.remove(f)

    # 1. 加载数据
    all_features, label_encoder, cleaning_stats = load_data(
        args.geolife_root, args.osm_path, args.max_users,
        args.use_base_data, args.cleaning_mode, args.use_cleaned_balanced
    )
    if not all_features:
        return

    # 2. 生成共享划分（所有实验共用）
    train_indices, val_indices, test_indices = generate_shared_split(
        all_features, label_encoder
    )

    labels_str = label_encoder.inverse_transform([item[3] for item in all_features])

    # 3. 提取各集合样本
    train_raw = [all_features[i] for i in train_indices]
    val_data  = [all_features[i] for i in val_indices]
    test_data = [all_features[i] for i in test_indices]

    # 4. 仅对训练集过采样
    # oversample 格式：(traj, stats, label_str)
    train_str = [(t, s, label_encoder.inverse_transform([l])[0])
                 for t, _, s, l in train_raw]
    train_oversampled_str = BaseGeoLifePreprocessor.oversample_minority_classes(
        train_str, target_ratio=0.3, minority_classes=['Subway', 'Train']
    )
    # 转回特征格式 (traj_21, placeholder=None, stats_18, label_encoded)
    train_data = [(t, None, s, label_encoder.transform([l])[0])
                  for t, s, l in train_oversampled_str]

    print(f"\n训练集过采样: {len(train_raw)} → {len(train_data)}")

    # 5. 归一化统计量（仅基于训练集）
    traj_all  = np.vstack([s[0] for s in train_data])
    stats_all = np.vstack([s[2] for s in train_data])

    traj_mean  = traj_all.mean(0).astype(np.float32)
    traj_std   = np.where(traj_all.std(0) < 1e-6, 1.0, traj_all.std(0)).astype(np.float32)
    stats_mean = stats_all.mean(0).astype(np.float32)
    stats_std  = np.where(stats_all.std(0) < 1e-6, 1.0, stats_all.std(0)).astype(np.float32)

    norm_params = dict(traj_mean=traj_mean, traj_std=traj_std,
                       stats_mean=stats_mean, stats_std=stats_std)

    # 6. Dataset & DataLoader
    def make_loader(data, shuffle):
        ds = TrajectoryDataset(data,
                               traj_mean=traj_mean, traj_std=traj_std,
                               stats_mean=stats_mean, stats_std=stats_std)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_data, shuffle=True)
    val_loader   = make_loader(val_data,   shuffle=False)
    test_loader  = make_loader(test_data,  shuffle=False)

    print(f"\nTrain={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # 7. 模型
    model = TransportationModeClassifier(
        TRAJECTORY_FEATURE_DIM, 18, args.hidden_dim, args.num_layers,
        len(label_encoder.classes_), args.dropout,
        num_segments=5, local_hidden=64, global_hidden=128,
    ).to(args.device)

    class_weights = compute_class_weights(
        label_encoder, all_features, label_index=3, mode='sqrt_inverse'
    ).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: (ep+1)/10 if ep < 10 else 1.0
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8
    )

    CHECKPOINT_PATH = os.path.join(args.save_dir, 'exp2_model.pth')
    best_val_acc = 0.0
    epochs_no_improve = 0
    consecutive_nan = 0

    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/exp2_training_log.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc',
                                 'val_loss', 'val_acc', 'lr'])

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )

        if np.isnan(train_loss) or np.isinf(train_loss):
            consecutive_nan += 1
            if consecutive_nan >= 3:
                break
            if os.path.exists(CHECKPOINT_PATH):
                prev = torch.load(CHECKPOINT_PATH, map_location=args.device, weights_only=False)
                model.load_state_dict(prev['model_state_dict'])
            continue
        else:
            consecutive_nan = 0

        # ✅ 修复：evaluate 返回5个值
        val_loss, val_acc, val_report, _, _ = evaluate(
            model, val_loader, criterion, args.device, label_encoder.classes_
        )

        if np.isnan(val_loss) or np.isinf(val_loss):
            consecutive_nan += 1
            if consecutive_nan >= 3:
                break
            continue
        else:
            consecutive_nan = 0

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if epoch < 10:
            warmup_scheduler.step()
        else:
            if epoch == 10:
                plateau_scheduler.num_bad_epochs = 0
            plateau_scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"学习率: {current_lr:.6f}")

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, f"{train_loss:.4f}", f"{train_acc:.4f}",
                                     f"{val_loss:.4f}", f"{val_acc:.4f}", f"{current_lr:.6f}"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_encoder': label_encoder,
                'val_acc': val_acc,
                'norm_params': norm_params,
                'model_config': {
                    'fused_input': True,
                    'input_dim': TRAJECTORY_FEATURE_DIM,
                    'trajectory_feature_dim': 9,
                    'spatial_feature_dim': 12,
                    'combined_dim': 21,
                    'segment_stats_dim': 18,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'num_classes': len(label_encoder.classes_),
                    'dropout': args.dropout,
                }
            }, CHECKPOINT_PATH)
            print(f"✓ 保存最佳模型 val_acc={val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"⏳ 未改善: {epochs_no_improve}/{args.patience}")
            if epochs_no_improve >= args.patience:
                print("🛑 Early stopping")
                break

    # 最终测试集评估
    print("\n" + "=" * 60)
    print("最终测试集评估")
    print("=" * 60)
    best_ckpt = torch.load(CHECKPOINT_PATH, map_location=args.device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    # ✅ 修复：evaluate 返回5个值
    test_loss, test_acc, test_report, all_preds, all_labels = evaluate(
        model, test_loader, criterion, args.device, label_encoder.classes_
    )

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    for cls in label_encoder.classes_:
        if cls in test_report:
            m = test_report[cls]
            print(f"  {cls:15s}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1-score']:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵:")
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
    warnings.simplefilter('ignore')
    main()