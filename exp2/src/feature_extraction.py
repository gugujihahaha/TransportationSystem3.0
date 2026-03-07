"""
特征提取模块 (混合优化版)
结合轨迹特征和空间特征

关键点:
- 调用 spatial_extractor.extract_spatial_features() 进行批量特征提取
- 不包含任何循环或嵌套 tqdm
- 保持简洁高效
"""
import numpy as np
from typing import Tuple

from .osm_feature_extractor import OsmSpatialExtractor


class FeatureExtractor:
    """特征提取器 (混合优化版)"""

    def __init__(self, spatial_extractor: OsmSpatialExtractor):
        """
        初始化特征提取器

        Args:
            spatial_extractor: OSM 空间特征提取器实例
        """
        self.spatial_extractor = spatial_extractor

    def extract_features(self, trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取轨迹特征和空间特征（点级融合）

        Args:
            trajectory: 已预处理的 NumPy 数组 (N, 9)
                [:, 0] = latitude
                [:, 1] = longitude
                [:, 2] = speed
                [:, 3] = acceleration
                [:, 4] = bearing_change
                [:, 5] = distance
                [:, 6] = time_diff
                [:, 7] = total_distance
                [:, 8] = total_time

        Returns:
            combined: 点级融合后的特征 (N, 21) = 9轨迹 + 12空间
            placeholder: 占位符 (N, 1)，保持接口兼容
        """
        # 1. 提取轨迹特征
        trajectory_features = self._extract_trajectory_features(trajectory)

        # 2. 提取空间特征
        try:
            spatial_features = self.spatial_extractor.extract_spatial_features(trajectory)
        except Exception as e:
            print(f"警告: 空间特征提取失败 ({e}). 使用零填充代替。")
            spatial_features = np.zeros((trajectory.shape[0], 12), dtype=np.float32)

        # 3. 验证维度
        if spatial_features.shape[1] != 12:
            raise ValueError(f"空间特征维度错误：预期12维，实际{spatial_features.shape[1]}维")

        if trajectory_features.shape[1] != 9:
            raise ValueError(f"轨迹特征维度错误：预期 9 维，实际 {trajectory_features.shape[1]} 维。")

        # 4. 点级融合：每个时间步直接拼接空间特征
        combined = np.concatenate([trajectory_features, spatial_features], axis=1)
        # 返回 (combined_21dim, zeros占位, ) 保持接口兼容
        placeholder = np.zeros((trajectory.shape[0], 1), dtype=np.float32)
        return combined, placeholder

    def _extract_trajectory_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        提取并返回原始轨迹特征

        Args:
            trajectory: (N, 9) 原始轨迹特征

        Returns:
            features: (N, 9) 原始特征（不做归一化）
        """
        return trajectory.copy().astype(np.float32)

    def combine_features(self, trajectory_features: np.ndarray,
                         spatial_features: np.ndarray) -> np.ndarray:
        """
        合并轨迹特征和空间特征（可选方法）

        Args:
            trajectory_features: (N, 9) 轨迹特征
            spatial_features: (N, 11) 空间特征

        Returns:
            combined: (N, 20) 合并后的特征
        """
        return np.concatenate([trajectory_features, spatial_features], axis=1)