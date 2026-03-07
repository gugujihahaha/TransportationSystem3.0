"""
通用训练工具函数。

提供支持任意数量输入模态的 train_epoch 和 evaluate 函数，
避免在 exp2/3/4/5 的 train.py 中重复实现相同逻辑。

使用约定：
    DataLoader 的每个 batch 必须是 (*features, labels) 的元组，
    其中 features 是若干特征张量，labels 是标签张量。
"""
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from typing import Tuple, List


def train_epoch(model, dataloader, criterion, optimizer, device,
                max_grad_norm=1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    batch_count = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch_count += 1
        *features, labels = batch
        labels = labels.view(-1).to(device)

        seq_features = []
        segment_stats = None
        for f in features:
            f = f.to(device)
            if f.dim() == 2:
                segment_stats = f
            else:
                seq_features.append(f)

        optimizer.zero_grad()

        if segment_stats is not None:
            logits = model(*seq_features, segment_stats=segment_stats)
        else:
            logits = model(*seq_features)

        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"[EPOCH END] 实际训练 batches: {batch_count}, 样本数: {total}")
    return total_loss / max(len(dataloader), 1), correct / max(total, 1)


def evaluate(model, dataloader, criterion, device, label_names):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            *features, labels = batch
            labels = labels.view(-1).to(device)

            seq_features = []
            segment_stats = None
            for f in features:
                f = f.to(device)
                if f.dim() == 2:
                    segment_stats = f
                else:
                    seq_features.append(f)

            if segment_stats is not None:
                logits = model(*seq_features, segment_stats=segment_stats)
            else:
                logits = model(*seq_features)

            total_loss += criterion(logits, labels).item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    report = classification_report(
        all_labels, all_preds,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    return avg_loss, report, all_preds, all_labels


def compute_class_weights(label_encoder, all_features_and_labels,
                       label_index: int = -1,
                       mode: str = 'inverse') -> torch.Tensor:
    """
    计算类别权重，用于 CrossEntropyLoss。

    Args:
        label_encoder: sklearn LabelEncoder，已 fit
        all_features_and_labels: 训练数据列表，每个元素最后一项为 label_encoded
        label_index: label 在每个元素 tuple 中的位置，默认 -1（最后一项）
        mode: 权重计算方式
            'inverse'       : w_i = N / (C * n_i)，标准反频率权重
            'sqrt_inverse'  : w_i = sqrt(N / (C * n_i))，温和版，避免极端权重
            'effective'     : 基于有效样本数（论文 Class-Balanced Loss 方法）

    Returns:
        weights: FloatTensor，shape (num_classes,)，已归一化使均值为1
    """
    import numpy as np
    from collections import Counter

    num_classes = len(label_encoder.classes_)
    labels = [item[label_index] for item in all_features_and_labels]

    # 支持字符串和整数标签
    if isinstance(labels[0], str):
        label_counts_dict = Counter(labels)
        label_counts = np.array([
            label_counts_dict.get(cls, 0)
            for cls in label_encoder.classes_
        ], dtype=np.float64)
    else:
        label_counts = np.zeros(num_classes, dtype=np.float64)
        for lbl in labels:
            label_counts[int(lbl)] += 1

    # 防止除零
    label_counts = np.where(label_counts == 0, 1, label_counts)
    N = label_counts.sum()
    C = num_classes

    if mode == 'inverse':
        weights = N / (C * label_counts)
    elif mode == 'sqrt_inverse':
        weights = np.sqrt(N / (C * label_counts))
    elif mode == 'effective':
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 归一化：使均值为1，避免影响整体 loss scale
    weights = weights / weights.mean()

    print(f"\n类别权重 (mode={mode}):")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"  {cls:15s}: count={int(label_counts[i]):5d}, weight={weights[i]:.4f}")

    return torch.FloatTensor(weights)


def compute_feature_stats(segments, feature_index=0):
    """
    计算训练集特征的均值和标准差，用于归一化。

    Args:
        segments: List of (traj_features, segment_stats, label)
        feature_index: 特征在 tuple 中的位置

    Returns:
        traj_mean, traj_std: 轨迹特征的均值和标准差 (feature_dim,)
        stats_mean, stats_std: 段级统计特征的均值和标准差 (18,)
    """
    traj_list = []
    stats_list = []

    for item in segments:
        traj = item[0]    # (50, 9)
        stats = item[1]   # (18,)
        traj_list.append(traj)
        stats_list.append(stats)

    traj_all = np.vstack(traj_list)       # (N*50, 9)
    stats_all = np.vstack(stats_list)     # (N, 18)

    traj_mean = traj_all.mean(axis=0).astype(np.float32)
    traj_std  = traj_all.std(axis=0).astype(np.float32)
    traj_std  = np.where(traj_std < 1e-6, 1.0, traj_std)  # 防止除零

    stats_mean = stats_all.mean(axis=0).astype(np.float32)
    stats_std  = stats_all.std(axis=0).astype(np.float32)
    stats_std  = np.where(stats_std < 1e-6, 1.0, stats_std)

    return traj_mean, traj_std, stats_mean, stats_std
