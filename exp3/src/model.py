"""
深度学习模型 (Exp3)
维度更新: spatial_feature_dim = 15
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
    """交通方式分类器 (Exp3) - 任务定义统一 num_classes = 7"""

    def __init__(
        self,
        trajectory_feature_dim: int = 9,
        spatial_feature_dim: int = 15,
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
            trajectory_feature_dim: 轨迹特征维度
            spatial_feature_dim: 空间特征维度
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
            input_dims=[trajectory_feature_dim, spatial_feature_dim],
            hidden_dims=[hidden_dim, hidden_dim // 2],
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
        self.spatial_feature_dim = spatial_feature_dim
        self.segment_stats_dim = segment_stats_dim

    def forward(self, trajectory_features: torch.Tensor,
                spatial_features: torch.Tensor,
                segment_stats: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            trajectory_features: (batch_size, seq_len, 9)
            spatial_features: (batch_size, seq_len, 15)
            segment_stats: (batch_size, segment_stats_dim)

        Returns:
            logits: (batch_size, num_classes)
        """
        return super().forward(trajectory_features, spatial_features,
                           segment_stats=segment_stats)

    def predict_proba(self, trajectory_features: torch.Tensor,
                     spatial_features: torch.Tensor,
                     segment_stats: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """推断类别和置信度（推理模式）。"""
        return super().predict_proba(trajectory_features, spatial_features,
                                    segment_stats=segment_stats)
