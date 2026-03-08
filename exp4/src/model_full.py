"""
深度学习模型 (exp4)
结合LSTM、空间特征和天气特征进行交通方式识别
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


class TransportationModeClassifierFull(nn.Module):
    """交通方式分类器 - 支持轨迹、空间、天气特征"""

    def __init__(
        self,
        TRAJECTORY_FEATURE_DIM: int = 9,
        SPATIAL_FEATURE_DIM: int = 12,
        WEATHER_FEATURE_DIM: int = 7,
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
            TRAJECTORY_FEATURE_DIM: 轨迹特征维度（9维）
            SPATIAL_FEATURE_DIM: 空间特征维度（12维）
            WEATHER_FEATURE_DIM: 天气特征维度（7维）
            segment_stats_dim: 段级统计特征维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            num_classes: 分类类别数
            dropout: Dropout比率
            num_segments: 序列分段数
            local_hidden: 局部编码器隐藏维度
            global_hidden: 全局编码器隐藏维度
        """
        super().__init__()

        self.trajectory_feature_dim = TRAJECTORY_FEATURE_DIM
        self.spatial_feature_dim = SPATIAL_FEATURE_DIM
        self.weather_feature_dim = WEATHER_FEATURE_DIM
        self.segment_stats_dim = segment_stats_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        # 轨迹特征编码器
        self.trajectory_encoder = nn.LSTM(
            TRAJECTORY_FEATURE_DIM,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # 空间特征编码器
        self.spatial_encoder = nn.LSTM(
            SPATIAL_FEATURE_DIM,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # 天气特征编码器
        self.weather_encoder = nn.LSTM(
            WEATHER_FEATURE_DIM,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # 融合层
        fusion_dim = hidden_dim * 3 + segment_stats_dim
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, global_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden, num_classes)
        )

    def forward(self, traj: torch.Tensor, spatial: torch.Tensor,
                weather: torch.Tensor, segment_stats: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            traj: (batch_size, seq_len, TRAJECTORY_FEATURE_DIM)
            spatial: (batch_size, seq_len, SPATIAL_FEATURE_DIM)
            weather: (batch_size, seq_len, WEATHER_FEATURE_DIM)
            segment_stats: (batch_size, segment_stats_dim)

        Returns:
            logits: (batch_size, num_classes)
        """
        # 轨迹特征编码
        traj_out, (traj_h, traj_c) = self.trajectory_encoder(traj)
        traj_feat = traj_h[-1]  # (batch_size, hidden_dim)

        # 空间特征编码
        spatial_out, (spatial_h, spatial_c) = self.spatial_encoder(spatial)
        spatial_feat = spatial_h[-1]  # (batch_size, hidden_dim)

        # 天气特征编码
        weather_out, (weather_h, weather_c) = self.weather_encoder(weather)
        weather_feat = weather_h[-1]  # (batch_size, hidden_dim)

        # 融合
        if segment_stats is not None:
            combined = torch.cat([traj_feat, spatial_feat, weather_feat, segment_stats], dim=1)
        else:
            combined = torch.cat([traj_feat, spatial_feat, weather_feat], dim=1)

        logits = self.fusion_fc(combined)
        return logits

    def predict_proba(self, traj: torch.Tensor, spatial: torch.Tensor,
                     weather: torch.Tensor, segment_stats: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """推断类别和置信度（推理模式）。"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(traj, spatial, weather, segment_stats)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
        return preds, probs