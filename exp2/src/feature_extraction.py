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
        提取轨迹特征和空间特征

        关键：这里只调用一次 spatial_extractor.extract_spatial_features()，
              不会出现嵌套循环问题

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
            trajectory_features: 归一化后的轨迹特征 (N, 9)
            spatial_features: 空间特征 (N, 11)
        """
        # 1. 提取和归一化轨迹特征
        trajectory_features = self._extract_trajectory_features(trajectory)

        # 2. 提取空间特征（关键：这里是批量提取，不是逐点循环）
        try:
            # 这里调用 spatial_extractor.extract_spatial_features(trajectory)
            # 在 osm_feature_extractor.py 中实现了：
            #   - 向量化网格键生成
            #   - 批量缓存查询
            #   - 批量 KDTree 查询
            # 因此不会出现嵌套循环
            spatial_features = self.spatial_extractor.extract_spatial_features(trajectory)
        except Exception as e:
            # 如果空间特征提取失败，使用零填充
            print(f"警告: 空间特征提取失败 ({e}). 使用零填充代替。")
            spatial_features = np.zeros((trajectory.shape[0], 11), dtype=np.float32)

        # 3. 验证维度
        if spatial_features.shape[1] != 11:
            raise ValueError(f"空间特征维度错误：预期 11 维，实际 {spatial_features.shape[1]} 维。")

        if trajectory_features.shape[1] != 9:
            raise ValueError(f"轨迹特征维度错误：预期 9 维，实际 {trajectory_features.shape[1]} 维。")

        return trajectory_features, spatial_features

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