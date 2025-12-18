"""
MASO-MSF 交通方式识别模型训练脚本 (Exp6)
基于论文: 基于GPS轨迹多尺度表达的交通出行方式识别方法
"""

import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm
from collections import Counter

from src.data_loader import GeoLifeDataLoader, preprocess_segments
from src.maso import MASOConfig, MASOFeatureOrganizer
from src.model_msf import MSFModel


# ============================================================
# Dataset
# ============================================================
class MASOTrajectoryDataset(Dataset):
    """MASO特征数据集"""

    def __init__(self, segments, label_encoder, maso_config=None):
        self.segments = segments
        self.label_encoder = label_encoder
        self.maso_config = maso_config if maso_config else MASOConfig()
        self.organizer = MASOFeatureOrganizer(self.maso_config)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment, label = self.segments[idx]

        # 提取MASO特征
        maso_features = self.organizer.extract_maso_features(segment)
        x = torch.FloatTensor(maso_features)

        # 转换标签
        y = self.label_encoder.transform([label])[0]

        return x, torch.LongTensor([y])


# ============================================================
# Train / Eval
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc="训练")
    for x, y in pbar:
        x, y = x.to(device), y.squeeze().to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

        pbar.set_postfix({
            'loss': total_loss / (pbar.n + 1),
            'acc': correct / total
        })

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device, label_encoder):
    """评估模型"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="评估"):
            x, y = x.to(device), y.squeeze().to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()

            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=label_encoder.classes_,
        labels=np.arange(len(label_encoder.classes_)),
        zero_division=0,
        output_dict=True
    )

    return total_loss / len(loader), report


# ============================================================
# Data Loading
# ============================================================
def load_geolife_data(geolife_root: str, max_users: int = None):
    """加载GeoLife数据"""
    print("=" * 60)
    print("加载 GeoLife 数据")
    print("=" * 60)

    loader = GeoLifeDataLoader(geolife_root)
    users_path = os.path.join(geolife_root, "Data")

    if not os.path.exists(users_path):
        users_path = geolife_root

    users = sorted([u for u in os.listdir(users_path) if len(u) == 3 and u.isdigit()])

    if max_users:
        users = users[:max_users]

    print(f"找到 {len(users)} 个用户")

    all_segments = []

    for user_id in tqdm(users, desc="读取用户轨迹"):
        labels = loader.load_labels(user_id)
        if labels.empty:
            continue

        traj_dir = os.path.join(users_path, user_id, "Trajectory")
        if not os.path.exists(traj_dir):
            continue

        for f in os.listdir(traj_dir):
            if not f.endswith(".plt"):
                continue
            try:
                traj = loader.load_trajectory(os.path.join(traj_dir, f))
                if traj.empty:
                    continue

                segments = loader.segment_trajectory(traj, labels)
                all_segments.extend(segments)
            except Exception as e:
                continue

    print(f"\n原始轨迹段数: {len(all_segments)}")

    # 预处理 (转换为标准9维特征)
    print("预处理轨迹段...")
    processed = preprocess_segments(all_segments, min_length=10)
    print(f"预处理后轨迹段数: {len(processed)}")

    return processed


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geolife_root", default="../data/Geolife Trajectories 1.3")
    parser.add_argument("--max_users", type=int, default=None, help="最大用户数 (用于快速测试)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", default="checkpoints")
    args = parser.parse_args()

    # ========================================================
    # 1. 加载数据
    # ========================================================
    segments = load_geolife_data(args.geolife_root, args.max_users)

    # 只保留6个主要类别
    TARGET_MODES = ['walk', 'bike', 'car & taxi', 'bus', 'train', 'subway']
    segments = [s for s in segments if s[1] in TARGET_MODES]

    print(f"过滤后轨迹段数: {len(segments)}")

    # 统计标签分布
    labels = [s[1] for s in segments]
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    print("\n类别分布:")
    for k, v in Counter(labels).items():
        print(f"  {k}: {v}")

    # ========================================================
    # 2. MASO配置
    # ========================================================
    maso_config = MASOConfig(
        K=6,  # 6个子段
        parts=20,
        spatial_ranges=[0.01, 0.05, 0.2],  # 3个空间范围
        image_sizes=[32],  # 1个图像尺寸
        L=9  # 9维属性
    )

    print(f"\nMASO配置:")
    print(f"  K (子段数): {maso_config.K}")
    print(f"  N (空间范围数): {maso_config.N}")
    print(f"  M (图像尺寸数): {maso_config.M}")
    print(f"  L (属性维度): {maso_config.L}")
    print(f"  总特征维度: {maso_config.L * maso_config.N * maso_config.M}")

    # ========================================================
    # 3. 创建数据集
    # ========================================================
    dataset = MASOTrajectoryDataset(segments, label_encoder, maso_config)

    # 分割训练/测试集
    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_idx)}")
    print(f"  测试集: {len(test_idx)}")

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # ========================================================
    # 4. 创建模型
    # ========================================================
    # MASO特征维度: K * L * N * M = 6 * 9 * 3 * 1 = 162
    maso_feat_dim = maso_config.K * maso_config.L * maso_config.N * maso_config.M
    img_size = maso_config.image_sizes[0]

    model = MSFModel(
        input_channels=maso_feat_dim // maso_config.K,  # L*N*M 作为单个对象的通道
        num_objects=maso_config.K,
        num_scales=maso_config.N * maso_config.M,
        num_classes=len(label_encoder.classes_),
        img_size=img_size
    ).to(args.device)

    print(f"\nMSF模型:")
    print(f"  输入通道: {maso_feat_dim // maso_config.K}")
    print(f"  对象数: {maso_config.K}")
    print(f"  图像尺寸: {img_size}x{img_size}")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========================================================
    # 5. 训练配置
    # ========================================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # ========================================================
    # 6. 训练循环
    # ========================================================
    best_loss = float("inf")
    best_acc = 0.0

    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )

        # 评估
        test_loss, report = evaluate(
            model, test_loader, criterion, args.device, label_encoder
        )

        test_acc = report['accuracy']

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.4f}")

        # 保存最佳模型
        if test_loss < best_loss:
            best_loss = test_loss
            best_acc = test_acc

            checkpoint_path = os.path.join(args.save_dir, "exp6_maso_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_encoder': label_encoder,
                'maso_config': maso_config,
                'model_config': {
                    'input_channels': maso_feat_dim // maso_config.K,
                    'num_objects': maso_config.K,
                    'num_scales': maso_config.N * maso_config.M,
                    'num_classes': len(label_encoder.classes_),
                    'img_size': img_size
                }
            }, checkpoint_path)
            print(f"✓ 保存最佳模型 (Loss: {best_loss:.4f}, Acc: {best_acc:.4f})")

        scheduler.step()

    print("\n" + "=" * 60)
    print(f"训练完成! 最佳准确率: {best_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()