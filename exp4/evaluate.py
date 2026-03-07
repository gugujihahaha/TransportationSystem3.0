"""
评估脚本 (Exp4 - 标签平滑 + Focal Loss)

使用训练好的模型在测试集上进行评估
"""
import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ========================== 路径设置 ==========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
os.chdir(SCRIPT_DIR)
# ==============================================================

from common import evaluate as common_evaluate

# exp2 专用模块（复用模型）
from exp2.src.model import TransportationModeClassifier

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


def load_data():
    """
    加载数据（与 train.py 完全一致）

    Returns:
        all_data: List of (traj_21, stats, label)
        label_encoder: LabelEncoder
    """
    # ===== 优先检查 exp4 完整缓存 =====
    if os.path.exists(PROCESSED_FEATURE_CACHE):
        print(f"✅ 加载 Exp4 特征缓存")
        try:
            with open(PROCESSED_FEATURE_CACHE, 'rb') as f:
                all_data, label_encoder, cleaning_stats = pickle.load(f)
            print(f"   缓存加载完成: {len(all_data)} 个样本")
            return all_data, label_encoder
        except Exception as e:
            print(f"⚠️ 缓存加载失败 ({e})，重新构建")

    # ===== 加载 exp2 的特征缓存 =====
    print(f"✅ 加载 Exp2 特征 (复用)")
    if not os.path.exists(EXP2_FEATURE_CACHE):
        raise FileNotFoundError(
            f"找不到 exp2 特征缓存: {EXP2_FEATURE_CACHE}\n"
            "请先确保 exp2/train.py 已经运行并生成缓存。"
        )

    with open(EXP2_FEATURE_CACHE, 'rb') as f:
        exp2_raw, label_encoder, cleaning_stats = pickle.load(f)

    print(f"   exp2 特征加载完成: {len(exp2_raw)} 个样本")

    # ===== 转换数据格式 =====
    all_data = []
    for traj_21, _, stats, label_encoded in exp2_raw:
        # NaN 过滤
        if np.isnan(traj_21).any() or np.isinf(traj_21).any():
            continue
        if np.isnan(stats).any() or np.isinf(stats).any():
            continue

        # exp4 格式: (traj_21dim, stats_18dim, label_encoded)
        all_data.append((traj_21, stats, label_encoded))

    print(f"   数据转换完成: {len(all_data)} 个样本")

    # 保存缓存
    with open(PROCESSED_FEATURE_CACHE, 'wb') as f:
        pickle.dump((all_data, label_encoder, cleaning_stats), f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   Exp4 特征缓存已保存: {PROCESSED_FEATURE_CACHE}")

    return all_data, label_encoder


def main():
    parser = argparse.ArgumentParser(description='Exp4: 模型评估')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/exp4_model.pth',
                       help='模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='设备')

    args = parser.parse_args()

    DEVICE = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("Exp4 模型评估")
    print(f"设备: {DEVICE}")
    print("=" * 80)

    # 加载数据
    all_data, label_encoder = load_data()

    # 数据划分（与 train.py 完全一致：70/10/20）
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

    print(f"✅ 数据划分完成:")
    print(f"  Train: {len(train_indices)}")
    print(f"  Val:   {len(val_indices)}")
    print(f"  Test:  {len(test_indices)}")

    # 计算归一化统计量（基于训练集）
    train_segments = [all_data[i] for i in train_indices]
    traj_mean, traj_std, stats_mean, stats_std = compute_feature_stats(train_segments)

    norm_params = {
        'traj_mean': traj_mean,
        'traj_std': traj_std,
        'stats_mean': stats_mean,
        'stats_std': stats_std
    }

    # 创建测试集 Dataset
    test_dataset = TrajectoryDatasetExp4(
        [all_data[i] for i in test_indices],
        traj_mean=traj_mean, traj_std=traj_std,
        stats_mean=stats_mean, stats_std=stats_std
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=0)

    print(f"  Test batches: {len(test_loader)}")

    # 加载模型
    print(f"\n========== 加载模型 ==========")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"找不到模型文件: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)

    # 检查 checkpoint 格式
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        print(f"✅ 检测到新格式 checkpoint")
        print(f"   模型配置: {model_config}")
    else:
        # 旧格式兼容
        model_config = {
            'input_dim': 21,
            'segment_stats_dim': 18,
            'hidden_dim': 128,
            'num_layers': 2,
            'num_classes': len(label_encoder.classes_),
            'dropout': 0.3,
        }
        print(f"⚠️ 检测到旧格式 checkpoint，使用默认配置")

    # 创建模型
    model = TransportationModeClassifier(
        trajectory_feature_dim=model_config.get('input_dim', 21),
        segment_stats_dim=model_config.get('segment_stats_dim', 18),
        hidden_dim=model_config.get('hidden_dim', 128),
        num_layers=model_config.get('num_layers', 2),
        num_classes=model_config.get('num_classes', len(label_encoder.classes_)),
        dropout=model_config.get('dropout', 0.3)
    ).to(DEVICE)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 模型加载完成")

    # 损失函数（标签平滑 + Focal Loss）
    from src.focal_loss import LabelSmoothingFocalLoss
    class_weights = checkpoint.get('class_weights', None)
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)

    criterion = LabelSmoothingFocalLoss(
        num_classes=len(label_encoder.classes_),
        gamma=2.0,
        smoothing=0.1,
        weight=class_weights
    )

    # 评估
    print(f"\n========== 测试集评估 ==========")
    test_loss, test_acc, test_report, all_preds, all_labels = common_evaluate(
        model, test_loader, criterion, DEVICE, label_encoder.classes_
    )

    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print("\n各类别详细指标:")
    for cls in label_encoder.classes_:
        if cls in test_report:
            metrics = test_report[cls]
            print(f"  {cls:15s}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")

    # 混淆矩阵
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

    print(f"\n模型路径: {args.checkpoint}")


if __name__ == "__main__":
    main()
