"""
MASO-MSF 模块: 多阶段融合识别模型 (MSF-Net)
复现论文: 《基于GPS轨迹多尺度表达的交通出行方式识别方法》- 马妍莉
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """属性-通道注意力模块: 增强对关键运动属性(如速度)的关注"""

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class ResBlock(nn.Module):
    """残差块: 论文中提到的基础卷积单元"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MSFModel(nn.Module):
    """
    [复现核心] MSF-Net: 多阶段融合网络
    支持双尺度输入 (Small Scale: 32x32, Large Scale: 64x64)
    """

    def __init__(self, num_classes=5, input_channels=3):
        super(MSFModel, self).__init__()

        # 1. 针对 32x32 尺度的分支
        self.branch_32 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 1), stride=1, padding=(1, 0)),  # 保持尺寸
            nn.ReLU(),
            # 关键修改点 1：确保池化层不会把 1 变成 0
            # 如果输入高度已经是 1，则不要再进行高度方向的池化
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0),

            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.ReLU(),

            # 关键修改点 2：使用自适应池化替代普通池化或 Flatten
            # 无论输入多小，它都会强制输出 (1, 1) 的尺寸，彻底避免 0x0 错误
            nn.AdaptiveMaxPool2d((1, 1))
        )

        # 2. 针对 64x64 尺度的分支
        self.branch_64 = nn.Sequential(
            ResBlock(input_channels, 32),
            ChannelAttention(32),
            nn.MaxPool2d(2),  # -> 32x32
            ResBlock(32, 64),
            nn.MaxPool2d(2),  # -> 16x16
            ResBlock(64, 64),
            nn.AdaptiveAvgPool2d(1)  # -> 64
        )

        # 3. 尺度注意力模块 (Scale Attention)
        # 论文 4.2.2 节：通过学习到的权重聚合不同尺度的特征
        self.scale_weight_net = nn.Sequential(
            nn.Linear(64 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

        # 4. 决策分类器
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x32, x64):
        # x32 shape: (Batch, 3, 32, 32)
        # x64 shape: (Batch, 3, 64, 64)

        # 第一阶段：属性特征提取
        feat32 = self.branch_32(x32).view(x32.size(0), -1)  # (B, 64)
        feat64 = self.branch_64(x64).view(x64.size(0), -1)  # (B, 64)

        # 第二阶段：尺度注意力融合
        combined = torch.cat([feat32, feat64], dim=1)  # (B, 128)
        weights = self.scale_weight_net(combined)  # (B, 2)

        # 自适应加权求和
        fused_feat = weights[:, 0:1] * feat32 + weights[:, 1:2] * feat64

        # 第三阶段：分类决策
        logits = self.classifier(fused_feat)
        return logits


# 模型测试代码
if __name__ == "__main__":
    model = MSFModel(num_classes=5)
    test_in_32 = torch.randn(2, 3, 32, 32)
    test_in_64 = torch.randn(2, 3, 64, 64)
    output = model(test_in_32, test_in_64)
    print(f"输入尺度: 32x32 & 64x64")
    print(f"输出维度: {output.shape}")  # 应为 [2, 5]