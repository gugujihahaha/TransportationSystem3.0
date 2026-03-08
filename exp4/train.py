"""
exp4/train.py（修复版）
======================
修复内容：
1. 去掉 placeholder（EXP2 模型是单编码器，不需要）
2. 数据划分改为加载共享划分（shared_split.pkl）
3. Dataset 格式改为 (traj_21, stats_18, label_encoded)，返回3个tensor
4. EXP4 直接从 EXP2 特征缓存复用数据

与 EXP2 的唯一区别：损失函数使用 LabelSmoothingFocalLoss。
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
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
os.chdir(SCRIPT_DIR)

from common import (BaseGeoLifePreprocessor, train_epoch, evaluate,
                    compute_class_weights)
from exp2.src.model import TransportationModeClassifier
from src.focal_loss import LabelSmoothingFocalLoss

TRAJECTORY_FEATURE_DIM = 21
SEGMENT_STATS_DIM      = 18

CACHE_DIR               = os.path.join(SCRIPT_DIR, 'cache')
PROCESSED_FEATURE_CACHE = os.path.join(CACHE_DIR, 'processed_features_exp4.pkl')
EXP2_FEATURE_CACHE      = os.path.join(PARENT_DIR, 'exp2', 'cache', 'processed_features.pkl')
SHARED_SPLIT_PATH       = os.path.join(PARENT_DIR, 'data', 'processed', 'shared_split.pkl')
os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# Dataset（去掉 placeholder）
# ============================================================
class TrajectoryDatasetExp4(Dataset):
    def __init__(self, all_features,
                 traj_mean=None, traj_std=None,
                 stats_mean=None, stats_std=None):
        self.data = all_features
        self.traj_mean  = traj_mean
        self.traj_std   = traj_std
        self.stats_mean = stats_mean
        self.stats_std  = stats_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 格式: (traj_21, stats_18, label_encoded)
        traj_21, stats, label = self.data[idx]

        traj  = traj_21.astype(np.float32)
        stats = stats.astype(np.float32)

        if self.traj_mean is not None:
            traj = (traj - self.traj_mean) / self.traj_std
        if self.stats_mean is not None:
            stats = (stats - self.stats_mean) / self.stats_std

        # ✅ 修复：不再返回 placeholder，模型是单编码器
        return (torch.FloatTensor(traj),
                torch.FloatTensor(stats),
                torch.LongTensor([label])[0])


# ============================================================
# 数据加载
# ============================================================
def load_data():
    """从 EXP2 缓存加载数据，转换为 EXP4 格式"""

    if os.path.exists(PROCESSED_FEATURE_CACHE):
        print(f"\n========== 加载 EXP4 特征缓存 ==========")
        try:
            with open(PROCESSED_FEATURE_CACHE, 'rb') as f:
                all_data, label_encoder, cleaning_stats = pickle.load(f)
            print(f"✅ 缓存加载完成: {len(all_data)} 个样本")
            return all_data, label_encoder, cleaning_stats
        except Exception as e:
            print(f"⚠️ 缓存加载失败 ({e})，重新构建")

    if not os.path.exists(EXP2_FEATURE_CACHE):
        raise FileNotFoundError(
            f"EXP2 特征缓存不存在: {EXP2_FEATURE_CACHE}\n"
            "请先运行 exp2/train.py"
        )

    print(f"\n========== 从 EXP2 缓存加载 EXP4 数据 ==========")
    with open(EXP2_FEATURE_CACHE, 'rb') as f:
        exp2_features, label_encoder, cleaning_stats = pickle.load(f)
    print(f"✅ EXP2 特征加载完成: {len(exp2_features)} 个样本")

    # EXP2 格式：(traj_21, spatial_placeholder, stats_18, label_encoded)
    # EXP4 格式：(traj_21, stats_18, label_encoded)  ← 去掉 placeholder
    all_data = []
    for traj_21, _, stats, label_encoded in exp2_features:
        if np.isnan(traj_21).any() or np.isinf(traj_21).any():
            continue
        if np.isnan(stats).any() or np.isinf(stats).any():
            continue
        all_data.append((traj_21, stats, label_encoded))

    print(f"✅ EXP4 数据转换完成: {len(all_data)} 个样本")

    with open(PROCESSED_FEATURE_CACHE, 'wb') as f:
        pickle.dump((all_data, label_encoder, cleaning_stats),
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ EXP4 缓存已保存: {PROCESSED_FEATURE_CACHE}")

    return all_data, label_encoder, cleaning_stats


def load_shared_split():
    if not os.path.exists(SHARED_SPLIT_PATH):
        raise FileNotFoundError(
            f"共享划分不存在: {SHARED_SPLIT_PATH}\n"
            "请先运行 exp2/train.py（会自动生成）"
        )
    with open(SHARED_SPLIT_PATH, 'rb') as f:
        return pickle.load(f)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=150)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int,   default=128)
    parser.add_argument('--num_layers', type=int,   default=2)
    parser.add_argument('--dropout',    type=float, default=0.3)
    parser.add_argument('--patience',   type=int,   default=25)
    parser.add_argument('--save_dir',   default='checkpoints')
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    DEVICE = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("Exp4 训练 (LabelSmoothing + Focal Loss)")
    print("=" * 60)

    # 1. 加载数据
    all_data, label_encoder, _ = load_data()

    # 2. 加载共享划分
    split = load_shared_split()
    train_indices = split['train_indices']
    val_indices   = split['val_indices']
    test_indices  = split['test_indices']
    print(f"\n[共享划分] Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    n = len(all_data)
    assert max(train_indices + val_indices + test_indices) < n, \
        f"索引越界：EXP4数据量={n}"

    # 3. 提取各集合样本
    train_raw  = [all_data[i] for i in train_indices]
    val_data   = [all_data[i] for i in val_indices]
    test_data  = [all_data[i] for i in test_indices]

    # 4. 仅对训练集过采样
    # oversample 格式: (traj, stats, label_str)
    train_str = [(t, s, label_encoder.inverse_transform([l])[0])
                 for t, s, l in train_raw]
    train_oversampled_str = BaseGeoLifePreprocessor.oversample_minority_classes(
        train_str, target_ratio=0.3, minority_classes=['Subway', 'Train']
    )
    train_data = [(t, s, label_encoder.transform([l])[0])
                  for t, s, l in train_oversampled_str]

    print(f"\n训练集过采样: {len(train_raw)} → {len(train_data)}")

    # 5. 归一化统计量
    traj_all  = np.vstack([s[0] for s in train_data])
    stats_all = np.vstack([s[1] for s in train_data])

    traj_mean  = traj_all.mean(0).astype(np.float32)
    traj_std   = np.where(traj_all.std(0) < 1e-6, 1.0, traj_all.std(0)).astype(np.float32)
    stats_mean = stats_all.mean(0).astype(np.float32)
    stats_std  = np.where(stats_all.std(0) < 1e-6, 1.0, stats_all.std(0)).astype(np.float32)

    norm_params = dict(traj_mean=traj_mean, traj_std=traj_std,
                       stats_mean=stats_mean, stats_std=stats_std)

    # 6. Dataset & DataLoader
    def make_loader(data, shuffle):
        ds = TrajectoryDatasetExp4(data,
                                   traj_mean=traj_mean, traj_std=traj_std,
                                   stats_mean=stats_mean, stats_std=stats_std)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_data, shuffle=True)
    val_loader   = make_loader(val_data,   shuffle=False)
    test_loader  = make_loader(test_data,  shuffle=False)

    print(f"\nTrain={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # 7. 模型（与 EXP2 完全相同的架构）
    model = TransportationModeClassifier(
        trajectory_feature_dim=TRAJECTORY_FEATURE_DIM,
        segment_stats_dim=SEGMENT_STATS_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=len(label_encoder.classes_),
        dropout=args.dropout
    ).to(DEVICE)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 8. 类别权重
    class_weights = compute_class_weights(
        label_encoder, train_data, label_index=2, mode='sqrt_inverse'
    ).to(DEVICE)

    # ✅ EXP4 的核心区别：使用 LabelSmoothingFocalLoss
    criterion = LabelSmoothingFocalLoss(
        num_classes=len(label_encoder.classes_),
        gamma=2.0,
        smoothing=0.1,
        weight=class_weights
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: (ep+1)/10 if ep < 10 else 1.0
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8
    )

    CHECKPOINT_PATH = os.path.join(args.save_dir, 'exp4_model.pth')
    best_val_acc = 0.0
    epochs_no_improve = 0
    consecutive_nan = 0

    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/exp4_training_log.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc',
                                 'val_loss', 'val_acc', 'lr'])

    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        if np.isnan(train_loss) or np.isinf(train_loss):
            consecutive_nan += 1
            if consecutive_nan >= 3:
                break
            if os.path.exists(CHECKPOINT_PATH):
                prev = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
                model.load_state_dict(prev['model_state_dict'])
            continue
        else:
            consecutive_nan = 0

        # ✅ evaluate 返回5个值（与 EXP1/2/3 一致）
        val_loss, val_acc, val_report, _, _ = evaluate(
            model, val_loader, criterion, DEVICE, label_encoder.classes_
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
                'label_encoder': label_encoder,
                'val_acc': val_acc,
                'norm_params': norm_params,
                'class_weights': class_weights.cpu().numpy(),
                'model_config': {
                    'input_dim': TRAJECTORY_FEATURE_DIM,
                    'segment_stats_dim': SEGMENT_STATS_DIM,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'num_classes': len(label_encoder.classes_),
                    'dropout': args.dropout,
                    'loss_type': 'label_smoothing_focal',
                    'focal_gamma': 2.0,
                    'label_smoothing': 0.1,
                }
            }, CHECKPOINT_PATH)
            print(f"✓ 保存最佳模型 val_acc={val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"⏳ 未改善: {epochs_no_improve}/{args.patience}")
            if epochs_no_improve >= args.patience:
                print("🛑 Early stopping")
                break

    print("\n" + "=" * 60)
    print("最终测试集评估")
    print("=" * 60)
    best_ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    test_loss, test_acc, test_report, all_preds, all_labels = evaluate(
        model, test_loader, criterion, DEVICE, label_encoder.classes_
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


if __name__ == "__main__":
    main()