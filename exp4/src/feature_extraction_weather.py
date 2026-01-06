"""
特征提取模块 (Exp4)
维度更新: 轨迹(9维) + KG(15维) + 天气(12维) = 36维
"""
import numpy as np
import pandas as pd
from typing import Tuple

from exp4.src.knowledge_graph import EnhancedTransportationKG
from exp4.src.weather_preprocessing import WeatherDataProcessor


class FeatureExtractorWithWeather:
    """特征提取器 (Exp4 - 含天气数据)"""

    def __init__(self, kg: EnhancedTransportationKG, weather_processor: WeatherDataProcessor):
        """
        初始化特征提取器

        Args:
            kg: 增强知识图谱
            weather_processor: 天气数据处理器
        """
        self.kg = kg
        self.weather_processor = weather_processor

    def extract_features(self, trajectory: np.ndarray,
                        trajectory_dates: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        提取特征（包含天气）

        Args:
            trajectory: (N, 9) 轨迹数组
            trajectory_dates: (N,) 轨迹日期时间序列

        Returns:
            trajectory_features: (N, 9) 归一化轨迹特征
            kg_features: (N, 15) 增强KG特征
            weather_features: (N, 12) 天气特征
        """
        # 1. 提取和归一化轨迹特征
        trajectory_features = self._extract_trajectory_features(trajectory)

        # 2. 提取增强KG特征
        try:
            kg_features = self.kg.extract_kg_features(trajectory)
        except Exception as e:
            print(f"警告: KG 特征提取失败 ({e}). 使用零填充代替。")
            kg_features = np.zeros((trajectory.shape[0], 15), dtype=np.float32)

        # 3. 验证KG特征维度 - KG is treated as a soft modality
        if kg_features.ndim != 2 or kg_features.shape[1] != 15:
            if kg_features.size == trajectory.shape[0] * 15:
                kg_features = kg_features.reshape(trajectory.shape[0], 15)
            else:
                print(f"警告: KG特征维度异常 {kg_features.shape}，使用零填充")
                kg_features = np.zeros((trajectory.shape[0], 15), dtype=np.float32)

        # 4. 提取天气特征（新增）
        try:
            weather_features = self.weather_processor.get_weather_features_for_trajectory(
                trajectory_dates
            )
        except Exception as e:
            print(f"警告: 天气特征提取失败 ({e}). 使用零填充代替。")
            weather_features = np.zeros((trajectory.shape[0], 12), dtype=np.float32)

        # 5. 验证天气特征维度 - Weather is treated as a soft modality
        if weather_features.shape != (trajectory.shape[0], 12):
            print(f"警告: 天气特征维度异常 {weather_features.shape}，使用零填充")
            weather_features = np.zeros((trajectory.shape[0], 12), dtype=np.float32)

        return trajectory_features, kg_features, weather_features

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