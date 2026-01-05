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
from common import BaseGeoLifePreprocessor, Exp1DataAdapter
# ===== 新增结束 =====

from src.data_loader import GeoLifeDataLoader, preprocess_segments
from src.model import TransportationModeClassifier

TRAJECTORY_FEATURE_DIM = 9


# ============================================================
# Dataset
# ============================================================
class TrajectoryDataset(Dataset):
    def __init__(self, segments, label_encoder):
        self.segments = segments
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        features, label = self.segments[idx]
        x = torch.FloatTensor(features)
        y = self.label_encoder.transform([label])[0]
        return x, torch.LongTensor([y])


# ============================================================
# Data loading (✅ 修改 2: 完整替换 load_data 函数)
# ============================================================
def load_data(geolife_root: str, max_users: int = None, use_base_data: bool = False):
    """
    加载数据（支持使用基础数据）

    Args:
        geolife_root: GeoLife数据根目录
        max_users: 最大用户数
        use_base_data: 是否使用预处理的基础数据（推荐）

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

        # 2. Exp1特定适配（序列长度 100）
        adapter = Exp1DataAdapter(target_length=100)
        processed = adapter.process_segments(base_segments)

        return processed

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

        return processed


# ============================================================
# Train / Eval
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in tqdm(loader, desc="训练"):
        x, y = x.to(device), y.squeeze().to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device, label_encoder):
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
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geolife_root", default="../data/Geolife Trajectories 1.3")

    # ===== ✅ 修改 3: main 函数添加参数 =====
    parser.add_argument("--use_base_data", action="store_true",
                       help="使用预处理的基础数据（推荐，大幅加速）")
    # ===== 修改结束 =====

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ✅ 传递新参数
    segments = load_data(
        args.geolife_root,
        use_base_data=args.use_base_data
    )

    # 最终 6 类（car & taxi 已合并）
    TARGET_MODES_FINAL = ['walk', 'bike', 'car & taxi', 'bus', 'train', 'subway']
    segments = [s for s in segments if s[1] in TARGET_MODES_FINAL]

    labels = [s[1] for s in segments]
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
            {"segments": segments, "label_encoder": label_encoder},
            f
        )
    print("✓ 已保存特征缓存: cache/exp1_processed_features.pkl")

    dataset = TrajectoryDataset(segments, label_encoder)
    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False
    )

    model = TransportationModeClassifier(
        input_dim=TRAJECTORY_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=len(label_encoder.classes_),
        dropout=args.dropout
    ).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True) # 确保 checkpoint 目录存在

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        test_loss, report = evaluate(
            model, test_loader, criterion, args.device, label_encoder
        )

        print(f"Train Acc: {train_acc:.4f} | Test Acc: {report['accuracy']:.4f}")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "label_encoder": label_encoder,
                "model_config": {
                    "input_dim": TRAJECTORY_FEATURE_DIM,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "num_classes": len(label_encoder.classes_),
                    "dropout": args.dropout
                }
            }, "checkpoints/exp1_model.pth")
            print("✓ 保存最佳模型")


if __name__ == "__main__":
    main()