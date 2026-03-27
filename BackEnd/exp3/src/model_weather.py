"""
深度学习模型 (Exp3)
三输入架构: 轨迹(21维，点级融合) + 天气(10维)

与 exp2 的对比关系:
    exp2: 轨迹21维(9轨迹+12空间)
    exp3: 轨迹21维(9轨迹+12空间) + 天气10维  ← 增量添加天气
    控制变量: 空间特征来源相同(exp2的OSM数据)，只新增天气通道
"""
import torch
import torch.nn as nn
from typing import Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common.base_model import HierarchicalTransportationClassifier


class TransportationModeClassifierWithWeather(HierarchicalTransportationClassifier):
    """
    Exp3 分类器
    输入:
        trajectory_features : (B, T, 21)  — 点级融合特征(9轨迹+12空间)，复用exp2
        weather_features    : (B, T, 10)  — 逐点广播的日级天气特征
        segment_stats       : (B, 18)     — 段级统计特征
    """

    def __init__(
        self,
        trajectory_feature_dim: int = 21,   # 复用exp2：9轨迹+12空间
        weather_feature_dim: int = 10,       # 日级天气10维
        segment_stats_dim: int = 18,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
        num_segments: int = 5,
        local_hidden: int = 64,
        global_hidden: int = 128,
    ):
        # 两路LSTM: [轨迹21维, 天气10维]
        super().__init__(
            input_dims=[trajectory_feature_dim, weather_feature_dim],
            hidden_dims=[hidden_dim, hidden_dim // 4],
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            num_segments=num_segments,
            local_hidden=local_hidden,
            global_hidden=global_hidden,
            segment_stats_dim=segment_stats_dim,
        )
        self.trajectory_feature_dim = trajectory_feature_dim
        self.weather_feature_dim = weather_feature_dim
        self.segment_stats_dim = segment_stats_dim

    def forward(
        self,
        trajectory_features: torch.Tensor,   # (B, T, 21)
        weather_features: torch.Tensor,       # (B, T, 10)
        segment_stats: torch.Tensor = None    # (B, 18)
    ) -> torch.Tensor:
        return super().forward(
            trajectory_features, weather_features,
            segment_stats=segment_stats
        )

    def predict_proba(
        self,
        trajectory_features: torch.Tensor,
        weather_features: torch.Tensor,
        segment_stats: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().predict_proba(
            trajectory_features, weather_features,
            segment_stats=segment_stats
        )