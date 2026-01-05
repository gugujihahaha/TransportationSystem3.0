"""
MASO-MSF 交通方式识别模型训练脚本 (Exp6)
基于论文: 基于GPS轨迹多尺度表达的交通出行方式识别方法

修复说明：
1. 修复了 MASO 提取特征时 numpy 数组导致的 AttributeError: 'numpy.ndarray' object has no attribute 'bloc'
2. 增加了 pickle 缓存机制，避免重复加载 GeoLife 原始数据（耗时约1小时）
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm
from collections import Counter

# 假设这些模块在你的项目 src 目录下
from src.data_loader import GeoLifeDataLoader, preprocess_segments
from src.maso import MASOConfig, MASOFeatureOrganizer
from src.model_msf import SimpleMSFModel

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
        # 定义列名，确保与 data_loader.py 生成的特征顺序一致
        self.feature_cols = ['latitude', 'longitude', 'speed', 'acceleration',
                            'bearing_change', 'distance', 'time_diff',
                            'total_distance', 'total_time']

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        # segments 中的元素是 (data, label)
        segment_data, label = self.segments[idx]

        # 【核心修复】: 转换为 DataFrame。因为 maso.py 的 sliding_crop 使用了 .iloc
        if isinstance(segment_data, np.ndarray):
            segment = pd.DataFrame(segment_data, columns=self.feature_cols)
        else:
            segment = segment_data

        # 提取 MASO 特征
        try:
            maso_features = self.organizer.extract_maso_features(segment)
            x = torch.FloatTensor(maso_features)
            # 转换标签
            y = self.label_encoder.transform([label])[0]
            return x, torch.LongTensor([y]).squeeze()
        except Exception as e:
            # 防止由于某些数据异常导致训练中断
            print(f"\n[错误] 索引 {idx} 特征提取失败: {e}")
            # 返回一个全零占位符
            dummy_x = torch.zeros((self.maso_config.K,
                                   self.maso_config.L * self.maso_config.N * self.maso_config.M,
                                   32, 32))
            return dummy_x, torch.LongTensor([0]).squeeze()


# ============================================================
# Train / Eval 函数
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc="训练")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

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
            'loss': f"{total_loss / (pbar.n + 1):.4f}",
            'acc': f"{correct / total:.4f}"
        })

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device, label_encoder):
    """评估模型"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="评估"):
            x, y = x.to(device), y.to(device)
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
# Data Loading (包含缓存机制)
# ============================================================
def load_geolife_data(geolife_root: str, max_users: int = None, cache_file: str = "cache/geolife_processed.pkl"):
    """加载GeoLife数据，支持缓存"""

    # 1. 尝试读取缓存
    if os.path.exists(cache_file):
        print(f"发现缓存文件 {cache_file}，正在直接加载...")
        with open(cache_file, 'rb') as f:
            processed = pickle.load(f)
        print(f"缓存加载完成，共有 {len(processed)} 条轨迹段。")
        return processed

    print("=" * 60)
    print("未发现缓存，开始加载原始 GeoLife 数据 (此过程可能耗时较久)")
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

        for f_name in os.listdir(traj_dir):
            if not f_name.endswith(".plt"):
                continue
            try:
                traj = loader.load_trajectory(os.path.join(traj_dir, f_name))
                if traj.empty:
                    continue
                segments = loader.segment_trajectory(traj, labels)
                all_segments.extend(segments)
            except Exception:
                continue

    print(f"\n原始轨迹段数: {len(all_segments)}")
    print("预处理轨迹段并提取运动特征...")
    processed = preprocess_segments(all_segments, min_length=10)
    print(f"预处理后轨迹段数: {len(processed)}")

    # 2. 写入缓存
    print(f"正在保存处理后的数据到 {cache_file}...")
    with open(cache_file, 'wb') as f:
        pickle.dump(processed, f)
    print("保存成功！下次运行将自动读取。")

    return processed


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geolife_root", default="../data/Geolife Trajectories 1.3")
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--cache_name", default="cache/geolife_processed_cache.pkl")
    args = parser.parse_args()

    # 1. 加载数据 (优先从缓存读取)
    segments = load_geolife_data(args.geolife_root, args.max_users, args.cache_name)

    # 2. 过滤类别并重编码
    TARGET_MODES = ['walk', 'bike', 'car & taxi', 'bus', 'train', 'subway']
    segments = [s for s in segments if s[1] in TARGET_MODES]
    print(f"过滤后轨迹段数: {len(segments)}")

    labels_list = [s[1] for s in segments]
    label_encoder = LabelEncoder()
    label_encoder.fit(labels_list)

    print("\n类别分布:")
    for k, v in Counter(labels_list).items():
        print(f"  {k}: {v}")

    # 3. MASO配置
    maso_config = MASOConfig(
        K=6,
        parts=20,
        spatial_ranges=[0.01, 0.05, 0.2],
        image_sizes=[32],
        L=9
    )

    print(f"\nMASO配置:")
    print(f"  K (子段数): {maso_config.K}, N (范围): {maso_config.N}, M (尺寸): {maso_config.M}")
    print(f"  总输入特征通道数: {maso_config.L * maso_config.N * maso_config.M}")

    # 4. 创建数据集和加载器
    dataset = MASOTrajectoryDataset(segments, label_encoder, maso_config)
    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=labels_list
    )

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

    # 5. 模型初始化
    input_channels = maso_config.L * maso_config.N * maso_config.M
    img_size = maso_config.image_sizes[0]

    model = SimpleMSFModel(
        input_channels=input_channels,
        num_objects=maso_config.K,
        num_classes=len(label_encoder.classes_),
        img_size=img_size
    ).to(args.device)

    print(f"\nSimpleMSF 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 6. 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    os.makedirs(args.save_dir, exist_ok=True)

    # 7. 训练循环
    best_loss = float("inf")
    best_acc = 0.0

    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)
        test_loss, report = evaluate(model, test_loader, criterion, args.device, label_encoder)

        test_acc = report['accuracy']
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.4f}")

        if test_loss < best_loss:
            best_loss = test_loss
            best_acc = test_acc
            checkpoint_path = os.path.join(args.save_dir, "exp6_maso_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'label_encoder': label_encoder,
                'maso_config': maso_config,
                'model_config': {
                    'input_channels': input_channels,
                    'num_objects': maso_config.K,
                    'num_classes': len(label_encoder.classes_),
                    'img_size': img_size
                }
            }, checkpoint_path)
            print(f"✓ 已保存最佳模型 (Loss: {best_loss:.4f})")

        scheduler.step()

    print("\n" + "=" * 60)
    print(f"训练完成! 最佳测试准确率: {best_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()