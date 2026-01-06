import torch
from torch import nn
import os
import time
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

# 导入项目中现有的模型定义和评估工具
from MainModel import MASO_MSF
from PrintMetricInformation import getPRF1  # 修正导入

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OBJECT_K = 5  # 对应你数据中的 5 个 segments/parts

# --- 路径设置 ---
# 指向你 Step 4 生成的最终特征文件
DATA_PATH = os.path.join('..', 'data', 'processed', 'final_features.pickle')


def trainModel(net, train_iter, test_iter, criterion, optimizer, num_epochs, device, num_classes):
    print("----- 训练开始 ----------")
    for epoch in range(num_epochs):
        start = time.time()
        net.train()
        train_loss, total, correct = 0, 0, 0

        for i, (X, y) in enumerate(train_iter):
            X = X.to(device)
            # --- 关键修改 1: 扩充标签 ---
            # 原本 y 是 [32]，扩充后 y_expand 变成 [160] (32*5)
            y_expand = []
            for label in y:
                for _ in range(5):  # 这里的 5 对应 OBJECT_K
                    y_expand.append(label)
            y_expand = torch.tensor(y_expand).long().to(device)

            optimizer.zero_grad()

            # --- 关键修改 2: 传 4 个参数适配模型 ---
            output = net(X, X, X, X)

            # 现在 output 形状是 [160, 5], y_expand 也是 [160]
            loss = criterion(output, y_expand)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += y_expand.size(0)
            correct += (predicted == y_expand).sum().item()

        train_acc = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Loss: {train_loss / len(train_iter):.4f} Acc: {train_acc:.2f}% 耗时: {time.time() - start:.2f}s")

def main():
    # 1. 基本参数设置
    num_classes = 5
    num_channel = 9
    value_ratio = 16
    BATCH_SIZE = 32
    NUM_EPOCHS = 50  # 建议设为 50-100

    # 2. 加载数据 (跳过原有的 DataLoader.py 逻辑)
    if not os.path.exists(DATA_PATH):
        print(f"错误：找不到特征文件 {DATA_PATH}")
        return

    with open(DATA_PATH, 'rb') as f:
        X, y = pickle.load(f)

    # 转换为 Tensor
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    # 3. 划分数据集
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_iter = DataLoader(test_data, batch_size=BATCH_SIZE)

    # 4. 初始化模型
    print(f"正在 {DEVICE} 上初始化模型...")
    net = MASO_MSF(num_channel, num_classes, value_ratio)
    net = net.to(DEVICE)

    # 5. 定义优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # 6. 执行训练
    trainModel(net, train_iter, test_iter, criterion, optimizer, NUM_EPOCHS, DEVICE, num_classes)

    # 7. 保存模型
    torch.save(net.state_dict(), "maso_msf_final.pth")
    print("模型已保存！")


if __name__ == "__main__":
    main()