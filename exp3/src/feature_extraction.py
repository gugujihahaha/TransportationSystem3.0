# ============================================================
# exp3/src/feature_extraction.py
# ============================================================
"""
特征提取模块 (Exp3)
维度更新: KG特征 11维 → 15维
"""
import numpy as np
from typing import Tuple
from .knowledge_graph import EnhancedTransportationKG


class FeatureExtractor:
    """特征提取器 (Exp3)"""

    def __init__(self, kg: EnhancedTransportationKG):
        self.kg = kg

    def extract_features(self, trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取特征

        Args:
            trajectory: (N, 9) 轨迹数组

        Returns:
            trajectory_features: (N, 9) 归一化轨迹特征
            kg_features: (N, 15) 增强KG特征
        """
        # 1. 提取和归一化轨迹特征
        trajectory_features = self._extract_trajectory_features(trajectory)

        # 2. 提取增强KG特征
        try:
            kg_features = self.kg.extract_kg_features(trajectory)
        except Exception as e:
            print(f"警告: KG 特征提取失败 ({e}). 使用零填充代替。")
            kg_features = np.zeros((trajectory.shape[0], 15), dtype=np.float32)

        # 3. 验证维度
        if kg_features.shape[1] != 15:
            raise ValueError(f"KG 特征维度错误：预期 15 维，实际 {kg_features.shape[1]} 维。")

        if trajectory_features.shape[1] != 9:
            raise ValueError(f"轨迹特征维度错误：预期 9 维，实际 {trajectory_features.shape[1]} 维。")

        return trajectory_features, kg_features

    def _extract_trajectory_features(self, trajectory: np.ndarray) -> np.ndarray:
        """提取并归一化轨迹特征"""
        features = trajectory.copy()
        features = self._normalize_features(features)
        return features

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Z-score 归一化"""
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8
        normalized = (features - mean) / std
        normalized = np.clip(normalized, -5, 5)
        return normalized