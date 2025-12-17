"""
MASO-MSF 训练脚本
复现论文: 《基于GPS轨迹多尺度表达的交通出行方式识别方法》- 马妍莉
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 将 src 目录加入路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import GeoLifeDataLoader, preprocess_segments
from src.model import MSFModel
from src.utils import calculate_metrics


# ============================================================
# Dataset 定义 - 修改后：增加维度处理和 Padding 保护
# ============================================================
class MASODataset(Dataset):
    def __init__(self, data_list, label_encoder):
        self.data = data_list
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        maso_imgs, label_name = self.data[idx]

        # 1. 提取并转为 FloatTensor
        # 原始维度可能是 (32, 11) 或 (64, 11)
        img32 = torch.from_numpy(maso_imgs[32]).float()
        img64 = torch.from_numpy(maso_imgs[64]).float()

        # 2. 核心修复：添加 Padding 逻辑 (防止输入太小导致池化后变0)
        # 确保 img32 至少有 32 行，img64 至少有 64 行
        if img32.size(0) < 32:
            img32 = torch.nn.functional.pad(img32, (0, 0, 0, 32 - img32.size(0)))
        if img64.size(0) < 64:
            img64 = torch.nn.functional.pad(img64, (0, 0, 0, 64 - img64.size(0)))

        # 3. 核心修复：增加通道维度 (Channel)
        # 卷积层要求输入为 (Batch, Channel, Height, Width)
        # 这里将 (Length, Features) 变为 (1, Length, Features)
        img32 = img32.unsqueeze(0)
        img64 = img64.unsqueeze(0)

        # 标签转索引
        label_idx = self.label_encoder.transform([label_name])[0]

        return img32, img64, torch.tensor(label_idx, dtype=torch.long)


# ============================================================
# 训练与验证逻辑
# ============================================================
def train_model(args):
    # 1. 初始化数据加载器
    # 路径处理：支持从 maso 文件夹运行或根目录运行
    data_root = args.geolife_root
    if not os.path.exists(data_root):
        # 尝试上级目录路径 (符合你要求的文件结构)
        data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Geolife Trajectories 1.3"))

    print(f"[*] 正在从以下路径加载数据: {data_root}")
    loader = GeoLifeDataLoader(data_root)
    users = loader.get_all_users()

    if args.max_users:
        users = users[:args.max_users]

    # 2. 预处理数据为 MASO 格式
    raw_data = preprocess_segments(loader, users)
    if not raw_data:
        print("[!] 未找到有效轨迹段，请检查数据路径。")
        return

    # 3. 标签处理
    labels = [item[1] for item in raw_data]
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    num_classes = len(label_encoder.classes_)
    print(f"[*] 识别类别 ({num_classes}): {label_encoder.classes_}")

    # 4. 划分数据集 (论文通常按 8:2 或 7:3 划分)
    train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42, stratify=labels)

    train_loader = DataLoader(MASODataset(train_data, label_encoder), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(MASODataset(test_data, label_encoder), batch_size=args.batch_size, shuffle=False)

    # 5. 初始化模型
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MSFModel(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 6. 训练循环
    best_f1 = 0
    os.makedirs("../checkpoints", exist_ok=True)

    print(f"[*] 开始训练，设备: {device}...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for x32, x64, y in train_loader:
            x32, x64, y = x32.to(device), x64.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x32, x64)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for x32, x64, y in test_loader:
                x32, x64, y = x32.to(device), x64.to(device), y.to(device)
                outputs = model(x32, x64)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_trues.extend(y.cpu().numpy())

        metrics = calculate_metrics(all_trues, all_preds, labels=range(num_classes))

        print(f"Epoch [{epoch + 1}/{args.epochs}] Loss: {train_loss / len(train_loader):.4f} | "
              f"Test F1: {metrics['f1']:.4f} | Acc: {np.mean(np.array(all_preds) == np.array(all_trues)):.4f}")

        # 保存最优模型
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_encoder': label_encoder
            }, "../checkpoints/maso_msf_best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MASO-MSF 训练器")
    parser.add_argument("--geolife_root", type=str, default="../data/Geolife Trajectories 1.3")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_users", type=int, default=None, help="限制用户数用于快速测试")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    train_model(args)