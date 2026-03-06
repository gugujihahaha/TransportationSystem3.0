"""
特征提取模块 (Exp4 - 稳定版)
维度: 轨迹(9维) + 空间(15维) + 天气(12维) = 36维
增强: 全面的异常处理和缺失值填充
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional

from .osm_feature_extractor import EnhancedOsmSpatialExtractor
from .weather_preprocessing import WeatherDataProcessor


class FeatureExtractorWithWeather:
    """特征提取器 (Exp4 - 稳定版，支持大量缺失数据)"""

    # 特征维度常量
    TRAJECTORY_DIM = 9
    SPATIAL_DIM = 15
    WEATHER_DIM = 12
    TOTAL_DIM = TRAJECTORY_DIM + SPATIAL_DIM + WEATHER_DIM  # 36

    def __init__(self, spatial_extractor: Optional[EnhancedOsmSpatialExtractor],
                 weather_processor: Optional[WeatherDataProcessor]):
        """
        初始化特征提取器

        Args:
            spatial_extractor: OSM 空间特征提取器（可为 None）
            weather_processor: 天气数据处理器（可为 None）
        """
        self.spatial_extractor = spatial_extractor
        self.weather_processor = weather_processor

    def extract_features(self, trajectory: np.ndarray,
                        trajectory_dates: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        提取特征（包含天气）- 稳定版

        Args:
            trajectory: (N, 9) 轨迹数组
            trajectory_dates: (N,) 轨迹日期时间序列

        Returns:
            trajectory_features: (N, 9) 归一化轨迹特征
            spatial_features: (N, 15) 增强空间特征
            weather_features: (N, 12) 天气特征

        所有返回值保证：
        - 形状正确
        - 无 NaN / Inf
        - dtype = float32
        """
        # 获取序列长度
        seq_len = trajectory.shape[0] if trajectory is not None and len(trajectory) > 0 else 0

        if seq_len == 0:
            # 空轨迹：返回空数组
            return (
                np.zeros((0, self.TRAJECTORY_DIM), dtype=np.float32),
                np.zeros((0, self.SPATIAL_DIM), dtype=np.float32),
                np.zeros((0, self.WEATHER_DIM), dtype=np.float32)
            )

        # 1. 提取和归一化轨迹特征（主模态，必须成功）
        trajectory_features = self._extract_trajectory_features_safe(trajectory)

        # 2. 提取增强空间特征（软模态，允许失败）
        spatial_features = self._extract_spatial_features_safe(trajectory, seq_len)

        # 3. 提取天气特征（软模态，允许失败）
        weather_features = self._extract_weather_features_safe(trajectory_dates, seq_len)

        return trajectory_features, spatial_features, weather_features

    def _extract_trajectory_features_safe(self, trajectory: np.ndarray) -> np.ndarray:
        """
        安全提取轨迹特征

        主模态：必须成功，异常时返回原始数据的安全版本
        """
        try:
            # 确保输入是 float32
            features = trajectory.astype(np.float32).copy()

            # 处理 NaN/Inf
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # 归一化
            features = self._normalize_features(features)

            return features

        except Exception as e:
            # 即使异常也要返回有效数据
            print(f"   ⚠️ 轨迹特征提取异常 ({e})，使用原始数据")
            features = np.nan_to_num(trajectory.astype(np.float32),
                                     nan=0.0, posinf=0.0, neginf=0.0)
            return features

    def _extract_spatial_features_safe(self, trajectory: np.ndarray, seq_len: int) -> np.ndarray:
        """
        安全提取空间特征

        软模态：允许失败，失败时返回全零向量
        """
        # 默认返回值
        default_features = np.zeros((seq_len, self.SPATIAL_DIM), dtype=np.float32)

        # 检查空间特征提取器是否可用
        if self.spatial_extractor is None:
            return default_features

        try:
            spatial_features = self.spatial_extractor.extract_spatial_features(trajectory)

            # 验证维度
            if spatial_features is None:
                return default_features

            if spatial_features.ndim != 2:
                if spatial_features.size == seq_len * self.SPATIAL_DIM:
                    spatial_features = spatial_features.reshape(seq_len, self.SPATIAL_DIM)
                else:
                    return default_features

            if spatial_features.shape != (seq_len, self.SPATIAL_DIM):
                # 尝试修复形状
                if spatial_features.shape[0] == seq_len and spatial_features.shape[1] != self.SPATIAL_DIM:
                    # 列数不对，截断或填充
                    if spatial_features.shape[1] > self.SPATIAL_DIM:
                        spatial_features = spatial_features[:, :self.SPATIAL_DIM]
                    else:
                        padding = np.zeros((seq_len, self.SPATIAL_DIM - spatial_features.shape[1]), dtype=np.float32)
                        spatial_features = np.concatenate([spatial_features, padding], axis=1)
                else:
                    return default_features

            # 清理 NaN/Inf
            spatial_features = np.nan_to_num(spatial_features.astype(np.float32),
                                        nan=0.0, posinf=0.0, neginf=0.0)

            # 裁剪到合理范围
            spatial_features = np.clip(spatial_features, -10, 10)

            return spatial_features

        except Exception as e:
            # 空间特征是软模态，异常时静默返回零向量
            # 仅在调试时打印
            # print(f"   ⚠️ 空间特征提取异常 ({e})，使用零向量")
            return default_features

    def _extract_weather_features_safe(self, trajectory_dates: pd.Series, seq_len: int) -> np.ndarray:
        """
        安全提取天气特征

        软模态：允许失败，失败时返回全零向量
        """
        # 默认返回值
        default_features = np.zeros((seq_len, self.WEATHER_DIM), dtype=np.float32)

        # 检查天气处理器是否可用
        if self.weather_processor is None:
            return default_features

        # 检查日期序列
        if trajectory_dates is None or len(trajectory_dates) == 0:
            return default_features

        try:
            weather_features = self.weather_processor.get_weather_features_for_trajectory(
                trajectory_dates
            )

            # 验证维度
            if weather_features is None:
                return default_features

            if weather_features.shape != (seq_len, self.WEATHER_DIM):
                # 尝试修复形状
                if weather_features.shape[0] != seq_len:
                    # 行数不对，需要重采样或填充
                    if weather_features.shape[0] > seq_len:
                        weather_features = weather_features[:seq_len]
                    else:
                        padding = np.zeros((seq_len - weather_features.shape[0], self.WEATHER_DIM),
                                          dtype=np.float32)
                        weather_features = np.vstack([weather_features, padding])

                if weather_features.shape[1] != self.WEATHER_DIM:
                    # 列数不对，截断或填充
                    if weather_features.shape[1] > self.WEATHER_DIM:
                        weather_features = weather_features[:, :self.WEATHER_DIM]
                    else:
                        padding = np.zeros((seq_len, self.WEATHER_DIM - weather_features.shape[1]),
                                          dtype=np.float32)
                        weather_features = np.concatenate([weather_features, padding], axis=1)

            # 清理 NaN/Inf
            weather_features = np.nan_to_num(weather_features.astype(np.float32),
                                             nan=0.0, posinf=0.0, neginf=0.0)

            # 裁剪到合理范围
            weather_features = np.clip(weather_features, -100, 100)

            return weather_features

        except Exception as e:
            # 天气是软模态，异常时静默返回零向量
            # 仅在调试时打印
            # print(f"   ⚠️ 天气特征提取异常 ({e})，使用零向量")
            return default_features

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Z-score 归一化 - 稳定版

        处理全零列、常数列等边界情况
        """
        # 确保是 float32
        features = features.astype(np.float32)

        # 计算均值和标准差
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)

        # 处理标准差为0的情况（常数列）
        std = np.where(std < 1e-8, 1.0, std)

        # 归一化
        normalized = (features - mean) / std

        # 处理 NaN/Inf
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

        # 裁剪到 [-5, 5] 范围
        normalized = np.clip(normalized, -5, 5)

        return normalized.astype(np.float32)

    def extract_features_batch(self, trajectories: list,
                               dates_list: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        批量提取特征

        Args:
            trajectories: 轨迹列表 [(N, 9), ...]
            dates_list: 日期序列列表

        Returns:
            Tuple of stacked feature arrays
        """
        all_traj = []
        all_spatial = []
        all_weather = []

        for traj, dates in zip(trajectories, dates_list):
            try:
                t_feat, s_feat, w_feat = self.extract_features(traj, dates)
                all_traj.append(t_feat)
                all_spatial.append(s_feat)
                all_weather.append(w_feat)
            except Exception:
                # 即使单个样本失败也不影响整体
                seq_len = traj.shape[0] if traj is not None else 0
                all_traj.append(np.zeros((seq_len, self.TRAJECTORY_DIM), dtype=np.float32))
                all_spatial.append(np.zeros((seq_len, self.SPATIAL_DIM), dtype=np.float32))
                all_weather.append(np.zeros((seq_len, self.WEATHER_DIM), dtype=np.float32))

        return all_traj, all_spatial, all_weather