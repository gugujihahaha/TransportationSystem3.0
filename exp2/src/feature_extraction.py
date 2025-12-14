"""
特征提取模块
结合轨迹特征和知识图谱特征
"""
import numpy as np
from typing import Tuple

from .knowledge_graph import TransportationKnowledgeGraph


# 导入 TransportationKnowledgeGraph，现在它能够处理 NumPy 数组


class FeatureExtractor:
    """特征提取器"""

    def __init__(self, kg: TransportationKnowledgeGraph):
        self.kg = kg

    def extract_features(self, trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取轨迹特征和知识图谱特征

        输入 trajectory 为已预处理的 NumPy 数组 (N, 7)。

        Returns:
            trajectory_features: 归一化后的轨迹特征 (N, 7)
            kg_features: 知识图谱特征 (N, 11)
        """
        # 1. 提取和归一化轨迹特征
        trajectory_features = self._extract_trajectory_features(trajectory) # (N, 7)

        # 2. 提取知识图谱特征
        # 传入 NumPy 数组，KG模块现在已修正为接受此格式
        try:
            # 关键修正：将 NumPy 数组直接传入 KG 模块
            kg_features = self.kg.extract_kg_features(trajectory)
        except Exception as e:
            # 如果 KG 特征提取失败，使用零填充代替 (N, 11)
            print(f"警告: KG 特征提取失败 ({e}). 使用零填充代替。")
            kg_features = np.zeros((trajectory.shape[0], 11), dtype=np.float32)

        # 检查维度以确保一致性
        if kg_features.shape[1] != 11:
            raise ValueError(f"KG 特征提取的维度错误：预期 11 维，实际 {kg_features.shape[1]} 维。")

        if trajectory_features.shape[1] != 9:
            raise ValueError(f"轨迹特征的维度错误：预期 9 维，实际 {trajectory_features.shape[1]} 维。")

        return trajectory_features, kg_features

    def _extract_trajectory_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        提取轨迹特征。输入 trajectory 已经是 (N, 7) 的特征数组。
        """
        features = trajectory
        features = self._normalize_features(features)
        return features

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        归一化特征 (Z-score 归一化)
        """
        # 使用Z-score归一化
        mean = np.mean(features, axis=0, keepdims=True)
        # 避免除以零
        std = np.std(features, axis=0, keepdims=True) + 1e-8

        normalized = (features - mean) / std

        # 处理异常值 (截断到 [-5, 5])
        normalized = np.clip(normalized, -5, 5)

        return normalized

    def combine_features(self, trajectory_features: np.ndarray,
                        kg_features: np.ndarray) -> np.ndarray:
        """合并轨迹特征和知识图谱特征"""
        return np.concatenate([trajectory_features, kg_features], axis=1)