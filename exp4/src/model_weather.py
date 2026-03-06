"""
深度学习模型 (Exp4)
三输入架构: 轨迹(9维) + 空间(15维) + 天气(12维)
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


class TransportationModeClassifierWithWeather(HierarchicalTransportationClassifier):
    """交通方式分类器 (Exp4 - 含天气特征) - 任务定义统一 num_classes = 7"""

    def __init__(
        self,
        trajectory_feature_dim: int = 9,
        spatial_feature_dim: int = 15,
        weather_feature_dim: int = 12,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.3,
        num_segments: int = 5,
        local_hidden: int = 64,
        global_hidden: int = 128,
    ):
        super().__init__(
            input_dims=[trajectory_feature_dim, spatial_feature_dim, weather_feature_dim],
            hidden_dims=[hidden_dim, hidden_dim // 2, hidden_dim // 4],
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            num_segments=num_segments,
            local_hidden=local_hidden,
            global_hidden=global_hidden,
        )

        # 保存参数供 checkpoint 序列化使用
        self.trajectory_feature_dim = trajectory_feature_dim
        self.spatial_feature_dim = spatial_feature_dim
        self.weather_feature_dim = weather_feature_dim

    def forward(self, trajectory_features: torch.Tensor,
                spatial_features: torch.Tensor,
                weather_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            trajectory_features: (batch_size, seq_len, 9)
            spatial_features: (batch_size, seq_len, 15)
            weather_features: (batch_size, seq_len, 12)

        Returns:
            logits: (batch_size, num_classes)
        """
        return super().forward(trajectory_features, spatial_features, weather_features)

    def predict_proba(self, trajectory_features: torch.Tensor,
                     spatial_features: torch.Tensor,
                     weather_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测类别和概率"""
        return super().predict_proba(trajectory_features, spatial_features, weather_features)
