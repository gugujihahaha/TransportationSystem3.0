"""
特征提取模块 (Exp3)
结合轨迹特征和增强知识图谱特征
- 轨迹特征: 9维
- 增强KG特征: 15维
- 总计: 24维
"""
import numpy as np
from typing import Tuple
from .knowledge_graph_enhanced import EnhancedTransportationKG


class FeatureExtractor:
    """特征提取器 (Exp3)"""

    def __init__(self, kg: EnhancedTransportationKG):
        self.kg = kg

    def extract_features(self, trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取轨迹特征和增强知识图谱特征

        Args:
            trajectory: (N, 9) NumPy 数组
                       [lat, lon, speed, accel, bearing_change, dist, time_diff, total_dist, total_time]

        Returns:
            trajectory_features: (N, 9) 归一化后的轨迹特征
            kg_features: (N, 15) 增强知识图谱特征
        """
        # 1. 提取和归一化轨迹特征
        trajectory_features = self._extract_trajectory_features(trajectory)

        # 2. 提取增强KG特征 (15维)
        try:
            kg_features = self.kg.extract_kg_features(trajectory)
        except Exception as e:
            print(f"警告: KG 特征提取失败 ({e}). 使用零填充代替。")
            kg_features = np.zeros((trajectory.shape[0], 15), dtype=np.float32)

        # 维度验证
        if kg_features.shape[1] != 15:
            raise ValueError(f"KG 特征维度错误：预期 15 维，实际 {kg_features.shape[1]} 维。")

        if trajectory_features.shape[1] != 9:
            raise ValueError(f"轨迹特征维度错误：预期 9 维，实际 {trajectory_features.shape[1]} 维。")

        return trajectory_features, kg_features

    def _extract_trajectory_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        提取并归一化轨迹特征

        Args:
            trajectory: (N, 9) 原始轨迹特征

        Returns:
            (N, 9) 归一化后的轨迹特征
        """
        features = trajectory.copy()
        features = self._normalize_features(features)
        return features

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Z-score 归一化

        Args:
            features: (N, D) 特征数组

        Returns:
            (N, D) 归一化后的特征数组
        """
        # Z-score 归一化
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8  # 避免除零

        normalized = (features - mean) / std

        # 截断异常值到 [-5, 5]
        normalized = np.clip(normalized, -5, 5)

        return normalized

    def combine_features(self, trajectory_features: np.ndarray,
                         kg_features: np.ndarray) -> np.ndarray:
        """
        合并轨迹特征和知识图谱特征

        Args:
            trajectory_features: (N, 9) 轨迹特征
            kg_features: (N, 15) KG特征

        Returns:
            (N, 24) 合并后的特征
        """
        return np.concatenate([trajectory_features, kg_features], axis=1)