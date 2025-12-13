"""
特征提取模块
结合轨迹特征和知识图谱特征
"""
import numpy as np
import pandas as pd
from typing import Tuple
from src.knowledge_graph import TransportationKnowledgeGraph


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, kg: TransportationKnowledgeGraph):
        self.kg = kg
        
    def extract_features(self, trajectory: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取轨迹特征和知识图谱特征
        
        Returns:
            trajectory_features: 轨迹特征 (N, 7)
            kg_features: 知识图谱特征 (N, 11)
        """
        # 提取轨迹特征
        trajectory_features = self._extract_trajectory_features(trajectory)
        
        # 提取知识图谱特征
        kg_features = self.kg.extract_kg_features(trajectory)
        
        return trajectory_features, kg_features
    
    def _extract_trajectory_features(self, trajectory: pd.DataFrame) -> np.ndarray:
        """提取轨迹特征"""
        features = trajectory[['latitude', 'longitude', 'speed', 'acceleration',
                              'bearing_change', 'distance', 'time_diff']].values
        
        # 归一化特征
        features = self._normalize_features(features)
        
        return features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """归一化特征"""
        # 使用Z-score归一化
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8
        
        normalized = (features - mean) / std
        
        # 处理异常值
        normalized = np.clip(normalized, -5, 5)
        
        return normalized
    
    def combine_features(self, trajectory_features: np.ndarray, 
                        kg_features: np.ndarray) -> np.ndarray:
        """合并轨迹特征和知识图谱特征"""
        return np.concatenate([trajectory_features, kg_features], axis=1)



