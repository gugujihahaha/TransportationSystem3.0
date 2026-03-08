"""
exp1/train.py（修复版）
======================
修复内容：
1. evaluate() 返回值从4个改为5个（增加 accuracy）
2. 数据划分改为加载共享划分（shared_split.pkl），与EXP2/3/4使用同一批测试集
3. EXP1的特征从EXP2缓存中提取前9维，确保样本完全对齐

注意：必须先运行 exp2/train.py 生成 EXP2 特征缓存，
      再运行 python create_shared_split.py，
      最后才能运行此脚本。
"""

import argparse
import csv
import os
import pickle
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
os.chdir(SCRIPT_DIR)

from common import (BaseGeoLifePreprocessor, train_epoch, evaluate,
                    compute_class_weights)
from src.model import TransportationModeClassifier

TRAJECTORY_FEATURE_DIM = 9  # EXP1只用前9维

# 路径配置
EXP2_FEATURE_CACHE = os.path.join(PARENT_DIR, 'exp2', 'cache', 'processed_features.pkl')
SHARED_SPLIT_PATH  = os.path.join(PARENT_DIR, 'data', 'processed', 'shared_split.pkl')
EXP1_CACHE_PATH    = os.path.join(SCRIPT_DIR, 'cache', 'exp1_processed_features.pkl')


# ============================================================
# Dataset（与原版相同）
# ============================================================
class TrajectoryDataset(Dataset):
    def __init__(self, segments, label_encoder,
                 traj_mean=None, traj_std=None,
                 stats_mean=None, stats_std=None):
        self.segments = segments
        self.label_encoder = label_encoder
        self.traj_mean  = traj_mean
        self.traj_std   = traj_std
        self.stats_mean = stats_mean
        self.stats_std  = stats_std

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        # 格式: (traj_9, stats_18, label_encoded)
        traj_9, stats, label_encoded = self.segments[idx]
        x     = traj_9.astype(np.float32)
        stats = stats.astype(np.float32)

        if self.traj_mean is not None:
            x = (x - self.traj_mean) / self.traj_std
        if self.stats_mean is not None:
            stats = (stats - self.stats_mean) / self.stats_std

        return (torch.FloatTensor(x),
                torch.FloatTensor(stats),
                torch.LongTensor([label_encoded]).squeeze())


# ============================================================
# 数据加载：从 EXP2 缓存提取前9维
# ============================================================
def load_data_from_exp2_cache():
    """
    从 EXP2 特征缓存提取 EXP1 所需的9维轨迹特征。
    EXP2 缓存格式：(all_features, label_encoder, cleaning_stats)
    每个样本：(traj_21, spatial_placeholder, stats_18, label_encoded)
    EXP1 只取 traj_21 的前9维作为轨迹特征。
    """
    if not os.path.exists(EXP2_FEATURE_CACHE):
        raise FileNotFoundError(
            f"EXP2 特征缓存不存在: {EXP2_FEATURE_CACHE}\n"
            "请先运行 exp2/train.py"
        )

    print(f"\n[数据加载] 从 EXP2 缓存提取9维轨迹特征...")
    with open(EXP2_FEATURE_CACHE, 'rb') as f:
        exp2_features, label_encoder, cleaning_stats = pickle.load(f)

    print(f"   EXP2 缓存样本数: {len(exp2_features)}")

    # 转换格式：取前9维，格式改为 (traj_9, stats_18, label_encoded)
    exp1_data = []
    for traj_21, _, stats_18, label_encoded in exp2_features:
        traj_9 = traj_21[:, :9].copy()  # 前9维是原始轨迹特征
        exp1_data.append((traj_9, stats_18, label_encoded))

    print(f"   EXP1 数据样本数: {len(exp1_data)}")
    print(f"   特征维度: traj={exp1_data[0][0].shape}, stats={exp1_data[0][1].shape}")

    # 保存 EXP1 缓存
    os.makedirs(os.path.join(SCRIPT_DIR, 'cache'), exist_ok=True)
    with open(EXP1_CACHE_PATH, 'wb') as f:
        pickle.dump((exp1_data, label_encoder, cleaning_stats), f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   ✓ 保存 EXP1 缓存: {EXP1_CACHE_PATH}")

    return exp1_data, label_encoder, cleaning_stats


def load_shared_split():
    """加载共享划分索引"""
    if not os.path.exists(SHARED_SPLIT_PATH):
        raise FileNotFoundError(
            f"共享划分不存在: {SHARED_SPLIT_PATH}\n"
            "请先运行 python create_shared_split.py"
        )
    with open(SHARED_SPLIT_PATH, 'rb') as f:
        return pickle.load(f)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=150)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--patience",   type=int,   default=25)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 1. 加载数据
    all_data, label_encoder, cleaning_stats = load_data_from_exp2_cache()

    # 2. 加载共享划分
    print(f"\n[共享划分] 加载...")
    split = load_shared_split()
    train_indices = split['train_indices']
    val_indices   = split['val_indices']
    test_indices  = split['test_indices']
    print(f"   Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")

    # 验证索引范围
    n = len(all_data)
    assert max(train_indices + val_indices + test_indices) < n, \
        f"索引越界：最大索引={max(train_indices+val_indices+test_indices)}, 数据量={n}"

    # 3. 提取各集合样本
    train_raw = [all_data[i] for i in train_indices]
    val_data  = [all_data[i] for i in val_indices]
    test_data = [all_data[i] for i in test_indices]

    # 4. 仅对训练集做过采样（避免数据泄露）
    # oversample 格式：(traj, stats, label_str)
    train_str = [(t, s, label_encoder.inverse_transform([l])[0])
                 for t, s, l in train_raw]
    train_oversampled_str = BaseGeoLifePreprocessor.oversample_minority_classes(
        train_str, target_ratio=0.3, minority_classes=['Subway', 'Train']
    )
    # 转回 encoded 格式
    train_data = [(t, s, label_encoder.transform([l])[0])
                  for t, s, l in train_oversampled_str]

    print(f"\n✅ 训练集过采样: {len(train_raw)} → {len(train_data)}")

    # 5. 归一化统计量（仅基于训练集）
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
        ds = TrajectoryDataset(data, label_encoder,
                               traj_mean=traj_mean, traj_std=traj_std,
                               stats_mean=stats_mean, stats_std=stats_std)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_data, shuffle=True)
    val_loader   = make_loader(val_data,   shuffle=False)
    test_loader  = make_loader(test_data,  shuffle=False)

    print(f"\n数据集大小: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # 类别分布
    print("\n类别分布（原始测试集）:")
    for cls in label_encoder.classes_:
        enc = label_encoder.transform([cls])[0]
        cnt = sum(1 for _, _, l in test_data if l == enc)
        print(f"  {cls:15s}: {cnt}")

    # 7. 模型
    model = TransportationModeClassifier(
        trajectory_feature_dim=TRAJECTORY_FEATURE_DIM,
        segment_stats_dim=18,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=len(label_encoder.classes_),
        dropout=args.dropout,
        num_segments=5,
        local_hidden=64,
        global_hidden=128,
    ).to(args.device)

    # 8. 损失函数
    class_weights = compute_class_weights(
        label_encoder, train_data, label_index=2, mode='sqrt_inverse'
    ).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 9. 优化器 & 调度器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: (ep+1)/10 if ep < 10 else 1.0
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-5
    )

    # 10. 训练
    CHECKPOINT_PATH = "checkpoints/exp1_model.pth"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    best_val_acc = 0.0
    epochs_no_improve = 0
    consecutive_nan = 0

    csv_path = "logs/exp1_training_log.csv"
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
                print("🛑 连续NaN，停止训练")
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
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_encoder': label_encoder,
                'val_acc': val_acc,
                'norm_params': norm_params,
                'model_config': {
                    'trajectory_feature_dim': TRAJECTORY_FEATURE_DIM,
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
                print(f"🛑 Early stopping")
                break

    # 11. 最终测试集评估
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
    print("\n各类别详细指标:")
    for cls in label_encoder.classes_:
        if cls in test_report:
            m = test_report[cls]
            print(f"  {cls:15s}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1-score']:.4f}")

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


if __name__ == "__main__":
    main()