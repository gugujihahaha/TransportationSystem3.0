"""
训练脚本 (Exp4 - CCA 对比学习)

与 exp2 的关系：
    - 特征完全复用 exp2 的点级融合特征 (21维)
    - 新增 CCA 对比学习损失（InfoNCE）
    - 控制变量：特征维度、数据划分、超参数与 exp2 一致

核心思想：
    - 将 21 维特征拆分为轨迹表示（9 维）和空间上下文表示（12 维）
    - 通过 InfoNCE 对比损失对齐两个表示
    - 主路径用于分类，辅路径仅用于对比学习

已修复的Bug：
    1. oversample 移到 split 之后，只对训练集做，避免测试集数据泄露
    2. early stopping 改为监控 val_acc（与exp1/exp2/exp3一致）
    3. plateau_scheduler mode='max'（监控acc）
    4. plateau 在 warmup 结束时 reset 内部计数器
    5. labels_stratify 取正确的索引（-1）
    6. evaluate.py 的 split 方式与 train.py 保持一致（70/10/20）
    7. patience 使用 args.patience，不硬编码
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
                    train_epoch, evaluate, compute_class_weights)
from common.train_utils import compute_feature_stats

# exp4 专用模块
from src.model_cca import CCATransportationClassifier

# ========================== 特征维度常量 ==========================
TRAJECTORY_FEATURE_DIM = 21   # 复用exp2：9轨迹+12空间，点级融合
SEGMENT_STATS_DIM      = 18
FIXED_SEQUENCE_LENGTH  = 50
# ==================================================================

# ========================== 缓存配置 ==========================
CACHE_DIR = 'cache'
EXP2_FEATURE_CACHE = os.path.join(PARENT_DIR, 'exp2', 'cache', 'processed_features.pkl')
PROCESSED_FEATURE_CACHE = os.path.join(CACHE_DIR, 'processed_features_exp4.pkl')
os.makedirs(CACHE_DIR, exist_ok=True)
# ==============================================================


class TrajectoryDatasetForCCA(Dataset):
    """轨迹数据集（CCA 版本）"""

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
    1. 直接加载 exp2 的特征缓存（exp2/cache/processed_features.pkl）
    2. exp2 缓存格式: (all_data, label_encoder, cleaning_stats)
    3. exp4 数据格式: (traj_21dim, stats_18dim, label_encoded)
    4. CCA 的 context 直接从 traj_21 拆分（前9维=轨迹，后12维=空间）
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
    print(f"\n========== 加载 Exp2 特征 (复用) ==========")
    if not os.path.exists(EXP2_FEATURE_CACHE):
        raise FileNotFoundError(
            f"找不到 exp2 特征缓存: {EXP2_FEATURE_CACHE}\n"
            "请先确保 exp2/train.py 已经运行并生成缓存。"
        )
    
    with open(EXP2_FEATURE_CACHE, 'rb') as f:
        exp2_raw, label_encoder, cleaning_stats = pickle.load(f)
    
    print(f"✅ exp2 特征加载完成: {len(exp2_raw)} 个样本")

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
    parser = argparse.ArgumentParser(description='训练 Exp4 (轨迹+空间+CCA)')
    parser.add_argument('--geolife_root', default='../data/Geolife Trajectories 1.3')
    parser.add_argument('--osm_path',     default='../data/exp2.geojson')
    parser.add_argument('--weather_path', default='../data/beijing_weather_daily_2007_2012.csv')
    parser.add_argument('--use_base_data', action='store_true', default=True)
    parser.add_argument('--cleaning_mode', default='balanced',
                        choices=['strict', 'balanced', 'gentle'])
    parser.add_argument('--max_users',  type=int,   default=None)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--epochs',     type=int,   default=150)
    parser.add_argument('--lr',         type=float, default=1e-4)   # 与exp1/exp2/exp3一致
    parser.add_argument('--hidden_dim', type=int,   default=128)
    parser.add_argument('--num_layers', type=int,   default=2)
    parser.add_argument('--dropout',    type=float, default=0.3)
    parser.add_argument('--patience',   type=int,   default=25)     # 与exp1/exp2/exp3一致
    parser.add_argument('--save_dir',   default='checkpoints')
    parser.add_argument('--context_loss_weight', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.07)
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "=" * 80)
    print("Exp4 训练 (轨迹21维 + CCA对比学习)")
    print(f"设备: {DEVICE} | lr: {args.lr} | patience: {args.patience}")
    print("=" * 80)

    # 加载数据
    all_data, label_encoder, cleaning_stats = load_data(
        args.geolife_root, args.osm_path, args.weather_path,
        args.max_users, args.use_base_data, args.cleaning_mode
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
    train_oversampled = BaseGeoLifePreprocessor.oversample_minority_classes(
        train_with_str, target_ratio=0.3, minority_classes=['Subway', 'Train']
    )
    # 转回编码标签
    train_oversampled = [(t, s, label_encoder.transform([l])[0])
                         for t, s, l in train_oversampled]

    print(f"✅ 数据划分完成 (oversample仅对训练集):")
    print(f"  Train (oversampled): {len(train_oversampled)}")
    print(f"  Val:                 {len(val_indices)}")
    print(f"  Test:                {len(test_indices)}")

    # 归一化统计量（仅基于训练集）
    print(f"\n========== 归一化统计量计算 ==========")
    train_trajs = np.array([t for t, _, _ in train_oversampled])
    train_stats = np.array([s for _, s, _ in train_oversampled])

    traj_mean, traj_std = compute_feature_stats(train_trajs)
    stats_mean, stats_std = compute_feature_stats(train_stats)

    norm_params = {
        'traj_mean': traj_mean,
        'traj_std': traj_std,
        'stats_mean': stats_mean,
        'stats_std': stats_std
    }
    print(f"✅ 归一化统计量计算完成（基于 {len(train_oversampled)} 个训练样本）")

    # 创建数据集
    train_dataset = TrajectoryDatasetForCCA(
        train_oversampled,
        traj_mean=traj_mean, traj_std=traj_std,
        stats_mean=stats_mean, stats_std=stats_std
    )
    val_dataset = TrajectoryDatasetForCCA(
        [all_data[i] for i in val_indices],
        traj_mean=traj_mean, traj_std=traj_std,
        stats_mean=stats_mean, stats_std=stats_std
    )
    test_dataset = TrajectoryDatasetForCCA(
        [all_data[i] for i in test_indices],
        traj_mean=traj_mean, traj_std=traj_std,
        stats_mean=stats_mean, stats_std=stats_std
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                          shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=0)

    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 打印类别分布
    print(f"\n类别分布 (训练集, oversampled):")
    train_labels = [l for _, _, l in train_oversampled]
    for cls in label_encoder.classes_:
        count = sum(1 for l in train_labels if l == label_encoder.transform([cls])[0])
        print(f"  {cls}: {count}")

    # 创建模型
    model = CCATransportationClassifier(
        trajectory_feature_dim=TRAJECTORY_FEATURE_DIM,
        segment_stats_dim=SEGMENT_STATS_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=len(label_encoder.classes_),
        dropout=args.dropout,
        context_loss_weight=args.context_loss_weight,
        temperature=args.temperature
    ).to(DEVICE)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 类别权重
    class_weights = compute_class_weights(train_labels, mode='sqrt_inverse')
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

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
        optimizer, mode='max', factor=0.5, patience=8, verbose=True
    )

    # 损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Checkpoint 路径
    os.makedirs(args.save_dir, exist_ok=True)
    CHECKPOINT_PATH = os.path.join(args.save_dir, 'exp4_model.pth')

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
            print(f"⚠️ 加载失败: {e}")

    # 训练曲线保存
    csv_path = os.path.join(args.save_dir, 'training_curve_exp4.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])

    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)

    for epoch in range(args.epochs):
        # Warmup 阶段
        if epoch < warmup_epochs:
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            # warmup 结束时 reset plateau 计数器
            if epoch == warmup_epochs:
                plateau_scheduler.num_bad_epochs = 0
            current_lr = optimizer.param_groups[0]['lr']

        # 训练一个 epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            use_cca=True, context_loss_weight=args.context_loss_weight
        )

        # 验证
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, DEVICE,
            label_names=None, use_cca=False  # 验证时不计算 CCA 损失
        )

        # 学习率调度
        if epoch >= warmup_epochs:
            plateau_scheduler.step(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_encoder': label_encoder,
                'val_acc': val_acc,
                'norm_params': norm_params,
                'model_config': {
                    'trajectory_feature_dim': TRAJECTORY_FEATURE_DIM,
                    'segment_stats_dim': SEGMENT_STATS_DIM,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'num_classes': len(label_encoder.classes_),
                    'dropout': args.dropout,
                    'context_loss_weight': args.context_loss_weight,
                    'temperature': args.temperature,
                }
            }, CHECKPOINT_PATH)
            print(f"✓ 保存最佳模型 (val_acc={val_acc:.4f})")

        # 记录训练曲线
        csv_writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc, current_lr])
        csv_file.flush()

        # 打印进度
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"学习率: {current_lr:.6f}")

        # Early stopping
        if plateau_scheduler.num_bad_epochs >= args.patience:
            print(f"\n⏹️ Early stopping (patience={args.patience})")
            break

    csv_file.close()

    # 最终测试集评估
    print("\n" + "=" * 80)
    print("最终测试集评估")
    print("=" * 80)

    # 加载最佳模型
    if os.path.exists(CHECKPOINT_PATH):
        print("✅ 已加载最佳 checkpoint")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, _, _ = evaluate(
        model, test_loader, criterion, DEVICE,
        label_names=None, use_cca=False  # 测试时不计算 CCA 损失
    )

    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # 打印各类别指标
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for traj, stats, labels in test_loader:
            traj = traj.to(DEVICE)
            stats = stats.to(DEVICE)
            logits = model(traj, segment_stats=stats, return_context=False)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    from sklearn.metrics import classification_report
    print(f"\n各类别指标:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_,
                             zero_division=0, digits=4))

    print(f"\n最佳验证准确率: {best_val_acc:.4f}")
    print(f"模型保存路径:   {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
