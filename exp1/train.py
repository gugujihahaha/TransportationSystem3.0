"""
训练脚本 - 仅使用GeoLife轨迹数据 (Exp1)
最终版：加入特征缓存，保证训练 / 评估 / 论文三者一致
✅ 已集成快速模式支持
"""

import argparse
import os
import pickle
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ===== ✅ 修改 1: 文件开头添加导入 =====
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (BaseGeoLifePreprocessor, Exp1DataAdapter,
                     train_epoch, evaluate, compute_class_weights)

from src.data_loader import GeoLifeDataLoader, preprocess_segments
from src.model import TransportationModeClassifier

TRAJECTORY_FEATURE_DIM = 9


# ============================================================
# Dataset
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
        features, segment_stats, label = self.segments[idx]
        x     = features.astype(np.float32)
        stats = segment_stats.astype(np.float32)

        # 归一化
        if self.traj_mean is not None:
            x = (x - self.traj_mean) / self.traj_std
        if self.stats_mean is not None:
            stats = (stats - self.stats_mean) / self.stats_std

        y = self.label_encoder.transform([label])[0]
        return (torch.FloatTensor(x),
                torch.FloatTensor(stats),
                torch.LongTensor([y]).squeeze())


# ============================================================
# Data loading (✅ 修改 2: 完整替换 load_data 函数)
# ============================================================
def load_data(geolife_root: str, max_users: int = None, use_base_data: bool = True, cleaning_mode: str = 'balanced'):
    """
    加载数据（支持使用基础数据）

    Args:
        geolife_root: GeoLife数据根目录
        max_users: 最大用户数
        use_base_data: 是否使用预处理的基础数据（推荐）
        cleaning_mode: 数据清洗模式 ('strict', 'balanced', 'gentle')

    Returns:
        processed_segments: List of (features, label)
    """
    BASE_DATA_PATH = os.path.join(
        os.path.dirname(geolife_root),
        'processed/base_segments.pkl'
    )

    # ========== 快速模式：使用基础数据 ==========
    if use_base_data and os.path.exists(BASE_DATA_PATH):
        print("\n" + "=" * 80)
        print("使用预处理的基础数据（快速模式）")
        print("=" * 80)

        # 1. 加载基础数据
        base_segments = BaseGeoLifePreprocessor.load_from_cache(BASE_DATA_PATH)

        # 2. Exp1特定适配（序列长度 50，两阶段清洗）
        adapter = Exp1DataAdapter(enable_cleaning=True, cleaning_mode=cleaning_mode)
        processed = adapter.process_segments(base_segments)
        cleaning_stats = adapter.get_cleaning_stats()
        return processed, cleaning_stats

    # ========== 传统模式：从头处理 ==========
    else:
        if use_base_data:
            print(f"\n⚠️  警告: 基础数据文件不存在: {BASE_DATA_PATH}")
            print("    将使用传统方式处理数据（较慢）")
            print("    建议先运行: python scripts/generate_base_data.py\n")

        print("=" * 80)
        print("加载 GeoLife 数据（传统模式）")
        print("=" * 80)

        loader = GeoLifeDataLoader(geolife_root)
        users_path = os.path.join(geolife_root, "Data")
        users = sorted([u for u in os.listdir(users_path) if u.isdigit()])

        if max_users:
            users = users[:max_users]

        all_segments = []
        for user_id in tqdm(users, desc="读取用户轨迹"):
            labels = loader.load_labels(user_id)
            if labels.empty:
                continue

            traj_dir = os.path.join(users_path, user_id, "Trajectory")
            for f in os.listdir(traj_dir):
                if not f.endswith(".plt"):
                    continue
                try:
                    traj = loader.load_trajectory(os.path.join(traj_dir, f))
                    segments = loader.segment_trajectory(traj, labels)
                    all_segments.extend(segments)
                except Exception:
                    continue

        print(f"原始轨迹段数: {len(all_segments)}")

        print("预处理轨迹段...")
        processed = preprocess_segments(all_segments, min_length=10)
        print(f"预处理后轨迹段数: {len(processed)}")

        cleaning_stats = {}
        return processed, cleaning_stats


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geolife_root", default="../data/Geolife Trajectories 1.3")

    # ===== ✅ 修改 3: main 函数添加参数 =====
    parser.add_argument("--use_base_data", action="store_true", default=True,
                       help="使用预处理的基础数据（推荐，大幅加速）")
    parser.add_argument("--cleaning_mode", type=str, default='balanced',
                       choices=['strict', 'balanced', 'gentle'],
                       help="数据清洗模式: strict(严格), balanced(平衡), gentle(温和)")
    # ===== 修改结束 =====

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ✅ 传递新参数
    segments, cleaning_stats = load_data(
        args.geolife_root,
        use_base_data=args.use_base_data,
        cleaning_mode=args.cleaning_mode
    )

    # 最终 6 类（任务定义统一）
    TARGET_MODES_FINAL = ['Walk', 'Bike', 'Car & taxi', 'Bus', 'Train', 'Subway']
    segments = [s for s in segments if s[2] in TARGET_MODES_FINAL]

    # 对少数类进行数据增强（仅对训练数据有效，此处对全量做增强后再split）
    segments = BaseGeoLifePreprocessor.oversample_minority_classes(
        segments,
        target_ratio=0.3,
        minority_classes=['Subway', 'Train']
    )

    labels = [s[2] for s in segments]
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    print("\n类别分布:")
    for k, v in Counter(labels).items():
        print(f"{k}: {v}")

    # ========================================================
    # 🔥 保存特征缓存（评估 & 论文复现关键）
    # ========================================================
    os.makedirs("cache", exist_ok=True)
    with open("cache/exp1_processed_features.pkl", "wb") as f:
        pickle.dump(
            {"segments": segments, "label_encoder": label_encoder, "cleaning_stats": cleaning_stats},
            f
        )
    print("✓ 已保存特征缓存: cache/exp1_processed_features.pkl")

    # 第一步：先划分索引（不需要 dataset）
    all_indices = np.arange(len(segments))
    labels_stratify = [s[2] for s in segments]

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
    train_segments = [segments[i] for i in train_indices]
    traj_mean, traj_std, stats_mean, stats_std = compute_feature_stats(train_segments)
    norm_params = {
        'traj_mean': traj_mean, 'traj_std': traj_std,
        'stats_mean': stats_mean, 'stats_std': stats_std
    }
    print(f"\n✅ 归一化统计量计算完成（基于 {len(train_indices)} 个训练样本）")

    # 第三步：创建带归一化的 dataset
    dataset = TrajectoryDataset(
        segments, label_encoder,
        traj_mean=traj_mean, traj_std=traj_std,
        stats_mean=stats_mean, stats_std=stats_std
    )

    print(f"\n✅ 数据集大小:")
    print(f"  总样本数: {len(dataset)}")
    print(f"  特征样本数: {len(segments)}")

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=args.batch_size,
        shuffle=False
    )

    print(f"\n✅ 数据加载完成:")
    print(f"  Train: {len(train_indices)} 样本")
    print(f"  Val:   {len(val_indices)} 样本")
    print(f"  Test:  {len(test_indices)} 样本")
    print(f"  训练批次总数: {len(train_loader)}")
    print(f"  验证批次总数: {len(val_loader)}")

    # 检查数据中是否有 NaN 或 Inf 值
    print("\n检查数据质量:")
    has_nan = False
    has_inf = False
    for i in range(min(100, len(segments))):
        features, segment_stats, label = segments[i]
        if np.isnan(features).any():
            print(f"  ❌ 样本 {i}: features 包含 NaN")
            has_nan = True
        if np.isinf(features).any():
            print(f"  ❌ 样本 {i}: features 包含 Inf")
            has_inf = True
        if np.isnan(segment_stats).any():
            print(f"  ❌ 样本 {i}: segment_stats 包含 NaN")
            has_nan = True
        if np.isinf(segment_stats).any():
            print(f"  ❌ 样本 {i}: segment_stats 包含 Inf")
            has_inf = True
    if not has_nan and not has_inf:
        print("  ✅ 前 100 个样本数据质量正常")

    print(f"\n类别分布:")
    for cls in label_encoder.classes_:
        train_count = sum(1 for i in train_indices if segments[i][2] == cls)
        val_count = sum(1 for i in val_indices if segments[i][2] == cls)
        test_count = sum(1 for i in test_indices if segments[i][2] == cls)
        print(f"  {cls:15s}: Train={train_count}, Val={val_count}, Test={test_count}")

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

    all_features_and_labels = segments
    class_weights = compute_class_weights(
        label_encoder,
        all_features_and_labels,
        label_index=2,
        mode='sqrt_inverse'
    ).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        optimizer, mode='min', factor=0.5, patience=8,
        min_lr=1e-5
    )

    # ========================================================
    # ✅ Early Stopping 配置
    # ========================================================
    CHECKPOINT_PATH = "checkpoints/exp1_model.pth"

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
    patience = 10
    os.makedirs("checkpoints", exist_ok=True)

    # ========================================================
    # ✅ 训练曲线保存到 CSV
    # ========================================================
    import csv
    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/exp1_training_log.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )

        # 在验证集上评估
        val_loss, val_report, _, _ = evaluate(
            model, val_loader, criterion, args.device, label_encoder.classes_
        )
        val_acc = val_report['accuracy']

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

        # Early Stopping 检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            torch.save({
                "epoch": epoch,
                "resume": True,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "label_encoder": label_encoder,
                "val_loss": val_loss,
                "norm_params": norm_params,
                "model_config": {
                    "trajectory_feature_dim": TRAJECTORY_FEATURE_DIM,
                    "segment_stats_dim": 18,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "num_classes": len(label_encoder.classes_),
                    "dropout": args.dropout
                }
            }, "checkpoints/exp1_model.pth")
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

    test_loss, test_report, all_preds, all_labels = evaluate(
        model, test_loader, criterion, args.device, label_encoder.classes_
    )
    test_acc = test_report['accuracy']

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


if __name__ == "__main__":
    main()
