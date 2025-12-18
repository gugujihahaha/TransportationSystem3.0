"""
MASO 双分支模型 (Exp6)
结合 CNN 空间特征提取和 LSTM 时序特征提取

架构:
- 分支 A (CNN 塔): 提取 (3, 64, 64) 图像的空间形态特征
- 分支 B (LSTM 塔): 提取 (50, 9) 时序特征
- 融合层: Concat + FC → 7 类交通方式
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpatialCNNTower(nn.Module):
    """
    CNN 空间特征提取塔
    输入: (batch, 3, 64, 64)
    输出: (batch, 256) 空间特征向量
    """

    def __init__(self, dropout: float = 0.3):
        super(SpatialCNNTower, self).__init__()

        # 轻量级 CNN (参考 ResNet 思想)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 64 → 32
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 32 → 16
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 16 → 8
        )

        # 全局平均池化 + 全连接
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (batch, 3, 64, 64)

        Returns:
            out: (batch, 256)
        """
        # CNN 特征提取
        x = self.conv1(x)  # (batch, 32, 32, 32)
        x = self.conv2(x)  # (batch, 64, 16, 16)
        x = self.conv3(x)  # (batch, 128, 8, 8)

        # 全局池化
        x = self.global_pool(x)  # (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 128)

        # 全连接
        out = self.fc(x)  # (batch, 256)

        return out


class TemporalLSTMTower(nn.Module):
    """
    LSTM 时序特征提取塔
    输入: (batch, 50, 9)
    输出: (batch, 256) 时序特征向量
    """

    def __init__(self,
                 input_dim: int = 9,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        super(TemporalLSTMTower, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),  # 双向 LSTM
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (batch, 50, 9)

        Returns:
            out: (batch, 256)
        """
        # LSTM 编码
        lstm_out, _ = self.lstm(x)  # (batch, 50, 256)

        # 取最后一个时间步
        last_hidden = lstm_out[:, -1, :]  # (batch, 256)

        # 全连接
        out = self.fc(last_hidden)  # (batch, 256)

        return out


class MASOClassifier(nn.Module):
    """
    MASO 双分支交通方式分类器

    输入:
        - maso_image: (batch, 3, 64, 64) 多尺度空间图像
        - temporal_features: (batch, 50, 9) 时序特征

    输出:
        - logits: (batch, num_classes) 分类 logits
    """

    def __init__(self,
                 num_classes: int = 7,
                 dropout: float = 0.3):
        super(MASOClassifier, self).__init__()

        self.num_classes = num_classes

        # 分支 A: CNN 空间特征提取
        self.spatial_tower = SpatialCNNTower(dropout=dropout)

        # 分支 B: LSTM 时序特征提取
        self.temporal_tower = TemporalLSTMTower(dropout=dropout)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256, 256),  # Concat 两个分支
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # 分类器
        self.classifier = nn.Linear(128, num_classes)

    def forward(self,
                maso_image: torch.Tensor,
                temporal_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            maso_image: (batch, 3, 64, 64)
            temporal_features: (batch, 50, 9)

        Returns:
            logits: (batch, num_classes)
        """
        # 分支 A: 空间特征
        spatial_feat = self.spatial_tower(maso_image)  # (batch, 256)

        # 分支 B: 时序特征
        temporal_feat = self.temporal_tower(temporal_features)  # (batch, 256)

        # 特征融合
        combined = torch.cat([spatial_feat, temporal_feat], dim=1)  # (batch, 512)
        fused = self.fusion(combined)  # (batch, 128)

        # 分类
        logits = self.classifier(fused)  # (batch, num_classes)

        return logits

    def predict(self,
                maso_image: torch.Tensor,
                temporal_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测类别和概率

        Returns:
            preds: (batch,) 预测类别索引
            probs: (batch, num_classes) 预测概率
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(maso_image, temporal_features)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs


# ============================================================
# 模型测试
# ============================================================
if __name__ == "__main__":
    # 测试模型
    model = MASOClassifier(num_classes=7)

    # 模拟输入
    batch_size = 4
    maso_img = torch.randn(batch_size, 3, 64, 64)
    temporal_feat = torch.randn(batch_size, 50, 9)

    # 前向传播
    logits = model(maso_img, temporal_feat)

    print(f"[MASO Model Test]")
    print(f"  输入 - MASO 图像: {maso_img.shape}")
    print(f"  输入 - 时序特征: {temporal_feat.shape}")
    print(f"  输出 - Logits: {logits.shape}")
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")