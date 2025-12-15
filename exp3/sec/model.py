"""
深度学习模型 (Exp3)
双 LSTM 特征融合模型
- 轨迹特征: 9维 → Bi-LSTM
- 增强KG特征: 15维 → Bi-LSTM
- 特征融合 → 分类器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TransportationModeClassifier(nn.Module):
    """交通方式分类器 (Exp3)"""

    def __init__(self,
                 trajectory_feature_dim: int = 9,
                 kg_feature_dim: int = 15,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 6,
                 dropout: float = 0.3):
        """
        Args:
            trajectory_feature_dim: 轨迹特征维度 (9)
            kg_feature_dim: 知识图谱特征维度 (15)
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            num_classes: 分类类别数 (walk, bike, car, bus, train, taxi)
            dropout: Dropout比率
        """
        super(TransportationModeClassifier, self).__init__()

        self.trajectory_feature_dim = trajectory_feature_dim
        self.kg_feature_dim = kg_feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 轨迹特征 LSTM 编码器
        # Bi-LSTM 输出维度: hidden_dim * 2
        self.trajectory_lstm = nn.LSTM(
            input_size=trajectory_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 知识图谱特征 LSTM 编码器
        # Bi-LSTM 输出维度: (hidden_dim // 2) * 2 = hidden_dim
        self.kg_lstm = nn.LSTM(
            input_size=kg_feature_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 特征融合层
        # 融合输入维度 = (trajectory_repr: hidden_dim * 2) + (kg_repr: hidden_dim)
        fusion_input_dim = hidden_dim * 2 + hidden_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 分类层
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, trajectory_features: torch.Tensor,
                kg_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            trajectory_features: (batch_size, seq_len, 9)
            kg_features: (batch_size, seq_len, 15)

        Returns:
            logits: (batch_size, num_classes)
        """
        # 轨迹特征编码
        trajectory_out, _ = self.trajectory_lstm(trajectory_features)
        trajectory_repr = trajectory_out[:, -1, :]  # (batch_size, hidden_dim * 2)

        # 知识图谱特征编码
        kg_out, _ = self.kg_lstm(kg_features)
        kg_repr = kg_out[:, -1, :]  # (batch_size, hidden_dim)

        # 特征融合
        combined = torch.cat([trajectory_repr, kg_repr], dim=1)
        fused = self.fusion_layer(combined)

        # 分类
        logits = self.classifier(fused)

        return logits

    def predict(self, trajectory_features: torch.Tensor,
                kg_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测类别和概率

        Args:
            trajectory_features: (batch_size, seq_len, 9)
            kg_features: (batch_size, seq_len, 15)

        Returns:
            preds: (batch_size,) 预测类别
            probs: (batch_size, num_classes) 类别概率
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(trajectory_features, kg_features)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs


class AttentionFusionModel(nn.Module):
    """带注意力机制的特征融合模型（可选扩展）"""

    def __init__(self,
                 trajectory_feature_dim: int = 9,
                 kg_feature_dim: int = 15,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 6,
                 dropout: float = 0.3):
        super(AttentionFusionModel, self).__init__()

        # 轨迹特征编码器
        self.trajectory_lstm = nn.LSTM(
            input_size=trajectory_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 知识图谱特征编码器
        self.kg_lstm = nn.LSTM(
            input_size=kg_feature_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 特征融合
        fusion_dim = hidden_dim * 2 + hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, trajectory_features: torch.Tensor,
                kg_features: torch.Tensor) -> torch.Tensor:
        # 编码轨迹特征
        trajectory_out, _ = self.trajectory_lstm(trajectory_features)

        # 编码知识图谱特征
        kg_out, _ = self.kg_lstm(kg_features)

        # 维度对齐：将 KG 特征填充到与轨迹特征相同的维度
        kg_out_dim = kg_out.size(-1)
        trajectory_out_dim = trajectory_out.size(-1)

        if kg_out_dim < trajectory_out_dim:
            kg_padded = F.pad(kg_out, (0, trajectory_out_dim - kg_out_dim))
        else:
            kg_padded = kg_out[:, :, :trajectory_out_dim]

        # 注意力融合：轨迹特征作为 query，KG 特征作为 key 和 value
        attended, _ = self.attention(trajectory_out, kg_padded, kg_padded)

        # 取最后一个时间步
        trajectory_repr = trajectory_out[:, -1, :]
        attended_repr = attended[:, -1, :]

        # 融合
        combined = torch.cat([trajectory_repr, attended_repr], dim=1)
        fused = self.fusion(combined)

        # 分类
        logits = self.classifier(fused)

        return logits