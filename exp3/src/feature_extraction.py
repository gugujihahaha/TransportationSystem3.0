"""
特征提取模块 (Exp3)

功能:
  结合轨迹特征和增强空间特征进行特征提取

关键点:
- 空间特征维度从 11 维扩展到 15 维
- 调用 EnhancedOsmSpatialExtractor 进行批量特征提取
- 不包含任何循环或嵌套 tqdm
- 保持简洁高效

特征维度:
- 轨迹特征: 9 维
- 空间特征: 15 维 (增强版)
"""
import numpy as np
from typing import Tuple
from .osm_feature_extractor import EnhancedOsmSpatialExtractor


class FeatureExtractor:
    """
    特征提取器 (Exp3)

    负责从预处理后的轨迹数据中提取轨迹特征和空间特征
    """

    def __init__(self, spatial_extractor: EnhancedOsmSpatialExtractor):
        """
        初始化特征提取器

        Args:
            spatial_extractor: 增强版 OSM 空间特征提取器实例
        """
        self.spatial_extractor = spatial_extractor

    def extract_features(self, trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取轨迹特征和空间特征

        算法流程:
          1. 提取并归一化轨迹特征（9维）
          2. 调用空间特征提取器获取增强空间特征（15维）
          3. 验证特征维度是否正确

        Args:
            trajectory: 已预处理的 NumPy 数组 (N, 9)
                [:, 0] = latitude (纬度)
                [:, 1] = longitude (经度)
                [:, 2] = speed (速度)
                [:, 3] = acceleration (加速度)
                [:, 4] = bearing_change (方向变化率)
                [:, 5] = distance (距离)
                [:, 6] = time_diff (时间差)
                [:, 7] = total_distance (累计距离)
                [:, 8] = total_time (累计时间)

        Returns:
            trajectory_features: 归一化后的轨迹特征 (N, 9)
            spatial_features: 增强空间特征 (N, 15)

        Raises:
            ValueError: 当空间特征维度不为 15 时抛出
        """
        # 1. 提取和归一化轨迹特征
        trajectory_features = self._extract_trajectory_features(trajectory)

        # 2. 提取增强空间特征（关键：这里是批量提取，不是逐点循环）
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
            spatial_features = np.zeros((trajectory.shape[0], 15), dtype=np.float32)

        # 3. 严格验证维度 (N, 15)
        # trajectory 形状为 (50, 9)，spatial_features 必须为 (50, 15)
        if spatial_features.ndim != 2 or spatial_features.shape[1] != 15:
            # 如果被压平了，重新 reshape
            if spatial_features.size == trajectory.shape[0] * 15:
                spatial_features = spatial_features.reshape(trajectory.shape[0], 15)
            else:
                raise ValueError(f"空间特征维度错误：预期末尾维度 15，实际 shape 为 {spatial_features.shape}")

        return trajectory_features, spatial_features

    def _extract_trajectory_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        提取并归一化轨迹特征

        Args:
            trajectory: (N, 9) 原始轨迹特征

        Returns:
            normalized_features: (N, 9) 归一化后的轨迹特征
        """
        # 复制数组避免修改原始数据
        features = trajectory.copy()

        # Z-score 归一化
        features = self._normalize_features(features)

        return features

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        归一化特征 (Z-score 归一化)

        公式: z = (x - μ) / σ

        Args:
            features: (N, 9) 原始特征

        Returns:
            normalized: (N, 9) 归一化特征，截断到 [-5, 5] 范围
        """
        # 计算均值和标准差（沿着第 0 维，即对每个特征维度单独计算）
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8  # 避免除以零

        # 归一化
        normalized = (features - mean) / std

        # 截断异常值到 [-5, 5] 范围
        normalized = np.clip(normalized, -5, 5)

        return normalized
