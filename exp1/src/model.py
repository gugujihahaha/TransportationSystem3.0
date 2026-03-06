"""
深度学习模型 (exp1)
仅使用GPS轨迹特征进行交通方式识别
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
    """
    exp1 分类器（纯轨迹版）。
    输入：轨迹特征 (trajectory_feature_dim 维） + 段级统计特征 (segment_stats_dim 维）。
    """

    def __init__(
        self,
        trajectory_feature_dim: int = 9,
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
            input_dims=[trajectory_feature_dim, segment_stats_dim],
            hidden_dims=[hidden_dim, hidden_dim // 2],
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            num_segments=num_segments,
            local_hidden=local_hidden,
            global_hidden=global_hidden,
        )
        self.trajectory_feature_dim = trajectory_feature_dim
        self.segment_stats_dim = segment_stats_dim

    def forward(self, trajectory_features: torch.Tensor,
                segment_stats: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            trajectory_features: (batch_size, seq_len, trajectory_feature_dim)
            segment_stats: (batch_size, segment_stats_dim)

        Returns:
            logits: (batch_size, num_classes)
        """
        return super().forward(trajectory_features, segment_stats)

    def predict_proba(self, trajectory_features: torch.Tensor,
                     segment_stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """推断类别和置信度（推理模式）。"""
        return super().predict_proba(trajectory_features, segment_stats)


class CNNLSTMModel(nn.Module):
    """CNN+LSTM混合模型（可选）"""

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
    ):
        super(CNNLSTMModel, self).__init__()

        # 1D CNN用于提取局部特征
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM编码器
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        # 转换为 (batch_size, input_dim, seq_len) 用于CNN
        x = x.transpose(1, 2)

        # CNN特征提取
        x = self.conv1d(x)  # (batch_size, 32, seq_len)

        # 转换回 (batch_size, seq_len, 32) 用于LSTM
        x = x.transpose(1, 2)

        # LSTM编码
        lstm_out, _ = self.lstm(x)

        # 使用最后一个时间步
        last_output = lstm_out[:, -1, :]

        # 分类
        logits = self.classifier(last_output)

        return logits
