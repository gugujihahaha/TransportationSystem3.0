"""
深度学习模型 (exp2)
结合LSTM和空间特征进行交通方式识别
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common.base_model import (BaseTransportationClassifier,
                                HierarchicalTransportationClassifier)


class TransportationModeClassifier(HierarchicalTransportationClassifier):
    """交通方式分类器 - 任务定义统一 num_classes = 7"""

    def __init__(
        self,
        trajectory_feature_dim: int = 21,
        segment_stats_dim: int = 18,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.3,
        num_segments: int = 5,
        local_hidden: int = 64,
        global_hidden: int = 128,
    ):
        """
        Args:
            trajectory_feature_dim: 轨迹特征维度（9轨迹+12空间=21维）
            segment_stats_dim: 段级统计特征维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            num_classes: 分类类别数（任务定义统一：Walk, Bike, Bus, Car&taxi, Train, Subway, Airplane）
            dropout: Dropout比率
            num_segments: 序列分段数
            local_hidden: 局部编码器隐藏维度
            global_hidden: 全局编码器隐藏维度
        """
        super().__init__(
            input_dims=[trajectory_feature_dim],
            hidden_dims=[hidden_dim],
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            num_segments=num_segments,
            local_hidden=local_hidden,
            global_hidden=global_hidden,
            segment_stats_dim=segment_stats_dim,
        )

        # 保存参数供 checkpoint 序列化使用
        self.trajectory_feature_dim = trajectory_feature_dim
        self.segment_stats_dim = segment_stats_dim

    def forward(self, trajectory_features: torch.Tensor,
                segment_stats: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            trajectory_features: (batch_size, seq_len, trajectory_feature_dim) - 21维融合特征
            segment_stats: (batch_size, segment_stats_dim)

        Returns:
            logits: (batch_size, num_classes)
        """
        return super().forward(trajectory_features,
                           segment_stats=segment_stats)

    def predict_proba(self, trajectory_features: torch.Tensor,
                     segment_stats: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """推断类别和置信度（推理模式）。"""
        return super().predict_proba(trajectory_features,
                                    segment_stats=segment_stats)


class AttentionFusionModel(BaseTransportationClassifier):
    """带注意力机制的特征融合模型（可选） - 基于 GeoLife 7 大类修正 num_classes = 7"""

    def __init__(
        self,
        trajectory_feature_dim: int = 9,
        spatial_feature_dim: int = 11,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__(
            input_dims=[trajectory_feature_dim, spatial_feature_dim],
            hidden_dims=[hidden_dim, hidden_dim // 2],
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )

        # 保存参数供 checkpoint 序列化使用
        self.trajectory_feature_dim = trajectory_feature_dim
        self.spatial_feature_dim = spatial_feature_dim

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
                spatial_features: torch.Tensor) -> torch.Tensor:
        # 编码轨迹特征
        trajectory_out, _ = self.encoders[0](trajectory_features)

        # 编码空间特征
        spatial_out, _ = self.encoders[1](spatial_features)

        # 注意力融合
        # 使用轨迹特征作为query，空间特征作为key和value
        # 需要对 spatial_out 进行填充使其维度与 trajectory_out 匹配
        spatial_out_dim = spatial_out.size(-1)
        trajectory_out_dim = trajectory_out.size(-1)

        if spatial_out_dim < trajectory_out_dim:
            # 填充到 hidden_dim * 2 的维度
            spatial_padded = F.pad(spatial_out, (0, trajectory_out_dim - spatial_out_dim))
        else:
            # 确保维度一致，这里简单截断或使用线性层也可以
            spatial_padded = spatial_out[:, :, :trajectory_out_dim]

        attended, _ = self.attention(trajectory_out, spatial_padded, spatial_padded)

        # 取最后一个时间步
        trajectory_repr = trajectory_out[:, -1, :]
        attended_repr = attended[:, -1, :]

        # 融合
        combined = torch.cat([trajectory_repr, attended_repr], dim=1)
        fused = self.fusion(combined)

        # 分类
        logits = self.classifier(fused)

        return logits
