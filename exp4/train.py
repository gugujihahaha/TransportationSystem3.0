"""
训练脚本 (Exp4 - 标签平滑 + Focal Loss)

与 exp2 的关系：
    - 特征完全复用 exp2 的点级融合特征 (21维)
    - 模型架构与 exp2 完全一致
    - 唯一区别：使用 LabelSmoothingFocalLoss 替换 CrossEntropyLoss

核心思想：
    - 标签平滑 (ε=0.1) 防止过拟合
    - Focal Loss (γ=2) 针对难分类样本加大惩罚
    - 专门针对 Bus 和 Car&taxi 的混淆问题

已修复的Bug：
    1. oversample 移到 split 之后，只对训练集做，避免测试集数据泄露
    2. early stopping 改为监控 val_acc（与exp1/exp2/exp3一致）
    3. plateau_scheduler mode='max'（监控acc）
    4. plateau 在 warmup 结束时 reset 内部计数器
    5. labels_stratify 取正确的索引（-1）
    6. evaluate.py 的 split 方式与 train.py 保持一致（70/10/20）
"""
import os
import sys
import argparse
import csv
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ========================== 路径设置 ==========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
os.chdir(SCRIPT_DIR)
# ==============================================================

from common import (BaseGeoLifePreprocessor, Exp4DataAdapter,
                    compute_class_weights, evaluate, train_epoch)

# exp2 专用模块（复用模型）
from exp2.src.model import TransportationModeClassifier

# exp4 专用模块
from src.focal_loss import LabelSmoothingFocalLoss

# ========================== 特征维度常量 ==========================
TRAJECTORY_FEATURE_DIM = 21   # 复用exp2：9轨迹+12空间，点级融合
SEGMENT_STATS_DIM      = 18
FIXED_SEQUENCE_LENGTH  = 50

# ========================== 缓存配置 ==========================
CACHE_DIR = 'cache'
PROCESSED_FEATURE_CACHE = os.path.join(CACHE_DIR, 'processed_features_exp4.pkl')
EXP2_FEATURE_CACHE = os.path.join('..', 'exp2', 'cache', 'processed_features.pkl')
# ==================================================================


def compute_feature_stats(segments):
    """
    计算特征的均值和标准差，用于归一化

    Args:
        segments: List of (traj_features, stats, label)

    Returns:
        mean: 均值
        std: 标准差
    """
    traj_list = []
    stats_list = []

    for item in segments:
        traj = item[0]    # (50, 21)
        stats = item[1]   # (18,)
        traj_list.append(traj)
        stats_list.append(stats)

    traj_all = np.vstack(traj_list)       # (N*50, 21)
    stats_all = np.vstack(stats_list)     # (N, 18)

    traj_mean = traj_all.mean(axis=0).astype(np.float32)
    traj_std  = traj_all.std(axis=0).astype(np.float32)
    traj_std  = np.where(traj_std < 1e-6, 1.0, traj_std)  # 防止除零

    stats_mean = stats_all.mean(axis=0).astype(np.float32)
    stats_std  = stats_all.std(axis=0).astype(np.float32)
    stats_std  = np.where(stats_std < 1e-6, 1.0, stats_std)

    return traj_mean, traj_std, stats_mean, stats_std


class TrajectoryDatasetExp4(Dataset):
    """轨迹数据集（Exp4 版本）"""

    def __init__(self, all_features,
                 traj_mean=None, traj_std=None,
                 stats_mean=None, stats_std=None):
        self.data = all_features
        self.traj_mean = traj_mean
        self.traj_std = traj_std
        self.stats_mean = stats_mean
        self.stats_std = stats_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj_21, stats, label = self.data[idx]

        # 归一化
        if self.traj_mean is not None:
            traj = (traj_21 - self.traj_mean) / self.traj_std
        else:
            traj = traj_21

        if self.stats_mean is not None:
            stats_norm = (stats - self.stats_mean) / self.stats_std
        else:
            stats_norm = stats

        return (torch.FloatTensor(traj),
                torch.FloatTensor(stats_norm),
                torch.LongTensor([label])[0])


def load_data(geolife_root: str, osm_path: str, weather_path: str,
              max_users=None, use_base_data=True, cleaning_mode='balanced'):
    """
    加载数据策略：
    1. 优先检查 exp4 完整缓存
    2. 若无缓存，直接加载 exp2 的特征缓存
    3. exp2 缓存格式: (all_data, label_encoder, cleaning_stats)
    4. exp4 数据格式: (traj_21dim, stats_18dim, label_encoded)
    """

    # ===== 优先检查 exp4 完整缓存 =====
    if os.path.exists(PROCESSED_FEATURE_CACHE):
        print(f"\n========== 加载 Exp4 特征缓存 ==========")
        try:
            with open(PROCESSED_FEATURE_CACHE, 'rb') as f:
                all_data, label_encoder, cleaning_stats = pickle.load(f)
            print(f"✅ 缓存加载完成: {len(all_data)} 个样本")
            return all_data, label_encoder, cleaning_stats
        except Exception as e:
            print(f"⚠️ 缓存加载失败 ({e})，重新构建")

    # ===== 加载 exp2 的特征缓存 =====
    print(f"\n========== 将 Exp2 特征 (复用) ==========")
    if not os.path.exists(EXP2_FEATURE_CACHE):
        raise FileNotFoundError(
            f"找不到 exp2 特征缓存: {EXP2_FEATURE_CACHE}\n"
            "请先确保 exp2/train.py 已经运行并生成缓存。"
        )

    with open(EXP2_FEATURE_CACHE, 'rb') as f:
        exp2_raw, label_encoder, cleaning_stats = pickle.load(f)

    print(f"✅ exp2 特征加载完成: {len(exp2_raw)} 个样本")

    # 验证 exp2 缓存格式
    if len(exp2_raw) > 0:
        print(f"exp2样本格式: {type(exp2_raw[0])}, 长度: {len(exp2_raw[0])}")
        print(f"exp2样本0的形状: {[x.shape if hasattr(x, 'shape') else type(x) for x in exp2_raw[0]]}")

    # ===== 转换数据格式 =====
    # exp2 格式: (traj_21dim, placeholder, stats_18dim, label_encoded)
    # exp4 格式: (traj_21dim, stats_18dim, label_encoded)
    print(f"\n========== 转换数据格式 ==========")

    all_data = []
    for traj_21, _, stats, label_encoded in exp2_raw:
        # NaN 过滤
        if np.isnan(traj_21).any() or np.isinf(traj_21).any():
            continue
        if np.isnan(stats).any() or np.isinf(stats).any():
            continue

        # exp4 格式: (traj_21dim, stats_18dim, label_encoded)
        all_data.append((traj_21, stats, label_encoded))

    print(f"✅ 数据转换完成: {len(all_data)} 个样本")

    # 保存缓存
    with open(PROCESSED_FEATURE_CACHE, 'wb') as f:
        pickle.dump((all_data, label_encoder, cleaning_stats), f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Exp4 特征缓存已保存: {PROCESSED_FEATURE_CACHE}")

    return all_data, label_encoder, cleaning_stats


def main():
    parser = argparse.ArgumentParser(description='Exp4: 标签平滑 + Focal Loss')
    parser.add_argument('--geolife_root', type=str, default='../data/Geolife Trajectories 1.3',
                       help='Geolife 数据根目录')
    parser.add_argument('--osm_path', type=str, default='../data/osm_data',
                       help='OSM 数据路径')
    parser.add_argument('--weather_path', type=str, default='../data/weather_data',
                       help='天气数据路径')
    parser.add_argument('--max_users', type=int, default=None,
                       help='最大用户数（用于快速测试）')
    parser.add_argument('--use_base_data', action='store_true',
                       help='是否使用 base_segments.pkl 作为基础数据')
    parser.add_argument('--cleaning_mode', type=str, default='balanced',
                       choices=['balanced', 'strict'],
                       help='数据清洗模式')
    parser.add_argument('--epochs', type=int, default=150,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='LSTM 层数')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout 比率')
    parser.add_argument('--patience', type=int, default=25,
                       help='Early stopping patience')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='模型保存目录')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='设备')

    args = parser.parse_args()

    DEVICE = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("Exp4 训练 (标签平滑 + Focal Loss)")
    print(f"设备: {DEVICE} | lr: {args.lr} | patience: {args.patience}")
    print("=" * 80)

    # 加载数据
    all_data, label_encoder, cleaning_stats = load_data(
        args.geolife_root,
        args.osm_path,
        args.weather_path,
        max_users=args.max_users,
        use_base_data=args.use_base_data,
        cleaning_mode=args.cleaning_mode
    )

    # 数据划分（与exp1/exp2/exp3完全一致）
    print(f"\n========== 数据划分 (70/10/20) ==========")
    all_indices = np.arange(len(all_data))
    labels_stratify = [item[-1] for item in all_data]

    train_indices, temp_indices = train_test_split(
        all_indices, test_size=0.3, random_state=42, stratify=labels_stratify
    )
    temp_labels = [labels_stratify[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.6667, random_state=42, stratify=temp_labels
    )

    # oversample 只对训练集做，val/test 保持原样
    train_segments_raw = [all_data[i] for i in train_indices]
    # 转换格式以适配 oversample 接口：需要 (feat, feat, label_str, ...) 格式
    # oversample 需要字符串标签，临时转回
    train_with_str = [(t, s, label_encoder.inverse_transform([l])[0])
                     for t, s, l in train_segments_raw]

    # oversample
    train_oversampled = BaseGeoLifePreprocessor.oversample_minority_classes(
        train_with_str,
        target_ratio=0.3,
        minority_classes=['Subway', 'Train']
    )
    # 转回编码标签
    train_oversampled = [(t, s, label_encoder.transform([l])[0])
                        for t, s, l in train_oversampled]

    train_labels = [item[-1] for item in train_oversampled]

    print(f"✅ 数据划分完成 (oversample仅对训练集):")
    print(f"  Train (oversampled): {len(train_oversampled)}")
    print(f"  Val:                 {len(val_indices)}")
    print(f"  Test:                {len(test_indices)}")

    # 归一化统计量（仅基于训练集）
    print(f"\n========== 归一化统计量计算 ==========")
    traj_mean, traj_std, stats_mean, stats_std = compute_feature_stats(train_oversampled)

    norm_params = {
        'traj_mean': traj_mean,
        'traj_std': traj_std,
        'stats_mean': stats_mean,
        'stats_std': stats_std
    }
    print(f"✅ 归一化统计量计算完成（基于 {len(train_oversampled)} 个训练样本）")

    # 创建 Dataset
    train_dataset = TrajectoryDatasetExp4(
        train_oversampled,
        traj_mean=traj_mean, traj_std=traj_std,
        stats_mean=stats_mean, stats_std=stats_std
    )
    val_dataset = TrajectoryDatasetExp4(
        [all_data[i] for i in val_indices],
        traj_mean=traj_mean, traj_std=traj_std,
        stats_mean=stats_mean, stats_std=stats_std
    )
    test_dataset = TrajectoryDatasetExp4(
        [all_data[i] for i in test_indices],
        traj_mean=traj_mean, traj_std=traj_std,
        stats_mean=stats_mean, stats_std=stats_std
    )

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                           shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=0)

    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 类别分布
    print(f"\n类别分布 (训练集, oversampled):")
    from collections import Counter
    label_counts = Counter(train_labels)
    for cls in label_encoder.classes_:
        count = label_counts.get(label_encoder.transform([cls])[0], 0)
        print(f"  {cls}: {count}")

    # 创建模型（与 exp2 完全一致）
    model = TransportationModeClassifier(
        trajectory_feature_dim=21,
        segment_stats_dim=18,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=len(label_encoder.classes_),
        dropout=args.dropout
    ).to(DEVICE)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 类别权重
    class_weights = compute_class_weights(
        label_encoder,
        train_oversampled,
        label_index=-1,
        mode='sqrt_inverse'
    ).to(DEVICE)

    print(f"\n类别权重 (mode=sqrt_inverse):")
    for cls, weight in zip(label_encoder.classes_, class_weights):
        count = sum(1 for l in train_labels if l == label_encoder.transform([cls])[0])
        print(f"  {cls}: count={count:4d}, weight={weight:.4f}")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 学习率调度器
    warmup_epochs = 10
    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)
    )
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8
    )

    # Early stopping 计数器
    epochs_no_improve = 0

    # 连续 NaN 检测计数器
    consecutive_nan_count = 0
    MAX_CONSECUTIVE_NAN = 3  # 连续3次NaN则停止训练

    # 损失函数（标签平滑 + Focal Loss）
    criterion = LabelSmoothingFocalLoss(
        num_classes=len(label_encoder.classes_),
        gamma=2.0,
        smoothing=0.1,
        weight=class_weights
    )

    # Checkpoint 路径
    os.makedirs(args.save_dir, exist_ok=True)
    CHECKPOINT_PATH = os.path.join(args.save_dir, 'exp4_model.pth')

    # 训练曲线保存
    os.makedirs("logs", exist_ok=True)
    csv_path = os.path.join("logs", 'exp4_training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])

    # 加载历史最佳模型
    best_val_acc = 0.0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n========== 加载历史最佳模型 ==========")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_val_acc = checkpoint.get('val_acc', 0.0)
            print(f"✅ 加载历史最佳权重，val_acc={best_val_acc:.4f}")
        except Exception as e:
            print(f"⚠️ 历史模型加载失败（{e}），从零开始")

    epochs_no_improve = 0

    # 连续 NaN 检测计数器
    consecutive_nan_count = 0
    MAX_CONSECUTIVE_NAN = 3  # 连续3次NaN则停止训练

    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Warmup 阶段
        if epoch < warmup_epochs:
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            # warmup 结束时 reset plateau 计数器
            if epoch == warmup_epochs:
                plateau_scheduler.num_bad_epochs = 0
                print("🔄 Warmup 结束，重置 ReduceLROnPlateau 状态")
            current_lr = optimizer.param_groups[0]['lr']

        # 训练一个 epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        # NaN 检测
        if np.isnan(train_loss) or np.isinf(train_loss):
            consecutive_nan_count += 1
            print(f"⚠️ 检测到训练损失异常（{train_loss}），连续NaN次数: {consecutive_nan_count}/{MAX_CONSECUTIVE_NAN}")

            if consecutive_nan_count >= MAX_CONSECUTIVE_NAN:
                print(f"🛑 连续{MAX_CONSECUTIVE_NAN}次NaN，停止训练")
                break

            if os.path.exists(CHECKPOINT_PATH):
                try:
                    prev = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
                    model.load_state_dict(prev['model_state_dict'])
                    print(f"✅ 已恢复到最佳模型")
                except Exception as e:
                    print(f"⚠️ 模型恢复失败（{e}），从零开始")
            continue  # 跳过本轮，不降lr
        else:
            consecutive_nan_count = 0  # 正常训练，重置计数器

        # 验证
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, DEVICE,
            label_names=None  # 验证时不计算 CCA 损失
        )

        # 学习率调度
        if epoch >= warmup_epochs:
            old_lr = optimizer.param_groups[0]['lr']
            plateau_scheduler.step(val_acc)
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"📉 学习率调整: {old_lr:.6f} → {new_lr:.6f}")

        # 在训练循环结束后再打印上一轮指标汇总
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0  # 重置计数器
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_encoder': label_encoder,
                'val_acc': val_acc,
                'norm_params': norm_params,
                'model_config': {
                    'input_dim': 21,
                    'segment_stats_dim': 18,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'num_classes': len(label_encoder.classes_),
                    'dropout': args.dropout,
                    'loss_type': 'label_smoothing_focal',
                    'focal_gamma': 2.0,
                    'label_smoothing': 0.1,
                }
            }, CHECKPOINT_PATH)
            print("✓ 保存最佳模型（基于验证集）")
        else:
            epochs_no_improve += 1
            print(f"⏳ 验证准确率未改善: {epochs_no_improve}/{args.patience}")

        # 记录训练曲线
        csv_writer.writerow([epoch + 1, f"{train_loss:.4f}", f"{train_acc:.4f}",
                          f"{val_loss:.4f}", f"{val_acc:.4f}",
                          f"{optimizer.param_groups[0]['lr']:.6f}"])
        csv_file.flush()

        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"\n🛑 Early stopping 触发（patience={args.patience}）")
            break

    csv_file.close()

    # 最终测试集评估
    print("\n" + "=" * 80)
    print("最终测试集评估")
    print("=" * 80)

    # 加载最佳checkpoint进行最终评估
    if os.path.exists(CHECKPOINT_PATH):
        print("✅ 已加载最佳checkpoint进行最终评估")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_report, all_preds, all_labels = evaluate(
        model, test_loader, criterion, DEVICE, label_encoder.classes_
    )

    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print("\n各类别详细指标:")
    for cls in label_encoder.classes_:
        if cls in test_report:
            metrics = test_report[cls]
            print(f"  {cls:15s}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")

    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
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

    print(f"\n最佳验证准确率: {best_val_acc:.4f}")
    print(f"模型保存路径:   {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
