"""
exp1/train.py（简化版）
======================
1. 从cleaned_balanced.pkl加载数据
2. 自己划分训练集、验证集、测试集（random_state=42）
3. 使用9维轨迹特征
"""

import argparse
import os
import pickle
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
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

TRAJECTORY_FEATURE_DIM = 9
CLEANED_DATA_PATH = os.path.join(PARENT_DIR, 'data', 'processed', 'cleaned_balanced.pkl')
EXP1_CACHE_PATH = os.path.join(SCRIPT_DIR, 'cache', 'exp1_processed_features.pkl')


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


def load_data():
    """从cleaned_balanced.pkl加载数据，提取9维轨迹特征"""
    if not os.path.exists(CLEANED_DATA_PATH):
        raise FileNotFoundError(f"找不到 {CLEANED_DATA_PATH}")

    print(f"\n[数据加载] 从cleaned_balanced.pkl加载...")
    with open(CLEANED_DATA_PATH, 'rb') as f:
        cleaned_data = pickle.load(f)

    print(f"   原始样本数: {len(cleaned_data)}")

    # 提取9维轨迹特征
    all_data = []
    for traj, stats, datetime_series, label in cleaned_data:
        # traj是9维轨迹特征
        traj_9 = traj.astype(np.float32)
        stats_18 = stats.astype(np.float32)
        all_data.append((traj_9, stats_18, label))

    # 编码标签
    all_labels_str = [item[2] for item in all_data]
    label_encoder = LabelEncoder().fit(all_labels_str)
    all_data = [(traj_9, stats_18, label_encoder.transform([label])[0])
                 for traj_9, stats_18, label in all_data]

    print(f"   处理后样本数: {len(all_data)}")
    print(f"   类别数: {len(label_encoder.classes_)}")
    print(f"   类别: {list(label_encoder.classes_)}")

    # 保存缓存
    os.makedirs(os.path.dirname(EXP1_CACHE_PATH), exist_ok=True)
    with open(EXP1_CACHE_PATH, 'wb') as f:
        pickle.dump((all_data, label_encoder, {}), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   缓存已保存: {EXP1_CACHE_PATH}")

    return all_data, label_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 加载数据
    all_data, label_encoder = load_data()

    # 2. 划分数据集（70/10/20，random_state=42）
    print(f"\n[数据划分] 70/10/20，random_state=42...")
    all_indices = np.arange(len(all_data))
    labels_encoded = [item[2] for item in all_data]

    train_indices, temp_indices = train_test_split(
        all_indices, test_size=0.3, random_state=42, stratify=labels_encoded
    )
    temp_labels = [labels_encoded[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.6667, random_state=42, stratify=temp_labels
    )

    train_data = [all_data[i] for i in train_indices]
    val_data   = [all_data[i] for i in val_indices]
    test_data  = [all_data[i] for i in test_indices]

    print(f"   训练集: {len(train_data)}")
    print(f"   验证集: {len(val_data)}")
    print(f"   测试集: {len(test_data)}")

    # 3. 计算归一化参数（仅基于训练集）
    traj_all  = np.vstack([s[0] for s in train_data])
    stats_all = np.vstack([s[1] for s in train_data])

    traj_mean  = traj_all.mean(0).astype(np.float32)
    traj_std   = np.where(traj_all.std(0) < 1e-6, 1.0, traj_all.std(0)).astype(np.float32)
    stats_mean = stats_all.mean(0).astype(np.float32)
    stats_std  = np.where(stats_all.std(0) < 1e-6, 1.0, stats_all.std(0)).astype(np.float32)

    norm_params = dict(traj_mean=traj_mean, traj_std=traj_std,
                       stats_mean=stats_mean, stats_std=stats_std)

    # 4. Dataset & DataLoader
    def make_loader(data, shuffle):
        ds = TrajectoryDataset(data, label_encoder,
                               traj_mean=traj_mean, traj_std=traj_std,
                               stats_mean=stats_mean, stats_std=stats_std)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_data, shuffle=True)
    val_loader   = make_loader(val_data,   shuffle=False)
    test_loader  = make_loader(test_data,  shuffle=False)

    # 5. 模型
    model = TransportationModeClassifier(
        TRAJECTORY_FEATURE_DIM, 18, args.hidden_dim, args.num_layers,
        len(label_encoder.classes_), args.dropout,
        num_segments=5, local_hidden=64, global_hidden=128,
    ).to(args.device)

    class_weights = compute_class_weights(
        label_encoder, train_data, label_index=2, mode='sqrt_inverse'
    )
    class_weights = torch.FloatTensor(class_weights).to(args.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 6. 训练
    # 加载历史最佳模型（如果存在）
    model_path = os.path.join(args.save_dir, 'exp1_model.pth')
    best_val_acc = 0.0
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
            best_val_acc = checkpoint.get('val_acc', 0.0)
            print(f"\n[加载历史最佳模型] 历史最佳验证准确率: {best_val_acc:.4f}")
        except Exception as e:
            print(f"\n[警告] 无法加载历史最佳模型: {e}")
            best_val_acc = 0.0
    else:
        print(f"\n[训练] 未找到历史最佳模型，从头开始训练")

    epochs_no_improve = 0
    patience = args.patience

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        val_loss, val_acc, val_report, _, _ = evaluate(
            model, val_loader, criterion, args.device, label_encoder.classes_
        )

        if val_acc > best_val_acc:
            old_best = best_val_acc
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_encoder': label_encoder,
                'val_acc': val_acc,
                'norm_params': norm_params,
                'model_config': {
                    'input_dim': TRAJECTORY_FEATURE_DIM,
                    'segment_stats_dim': 18,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'num_classes': len(label_encoder.classes_),
                    'dropout': args.dropout,
                }
            }, os.path.join(args.save_dir, 'exp1_model.pth'))
            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f} "
                  f"[✅ 保存最佳模型: {val_acc:.4f} > {old_best:.4f}]")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f} "
                  f"[❌ 未超最佳: {val_acc:.4f} <= {best_val_acc:.4f}]")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 7. 测试
    test_loss, test_acc, test_report, _, _ = evaluate(
        model, test_loader, criterion, args.device, label_encoder.classes_
    )
    print(f"\nTest Acc: {test_acc:.4f}")
    print(test_report)


if __name__ == '__main__':
    import sys
    main()
