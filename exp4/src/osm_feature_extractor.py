"""
OSM 空间特征提取模块 (exp4 - 稳定版)
完全继承 exp3（exp3 已继承 exp2），无任何差异。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from exp3.src.osm_feature_extractor import EnhancedOsmSpatialExtractor as _BaseEnhancedOsmSpatialExtractor


class EnhancedOsmSpatialExtractor(_BaseEnhancedOsmSpatialExtractor):
    """增强版 OSM 空间特征提取器 (Exp4 - 稳定版)"""

    # 空间特征维度常量
    SPATIAL_FEATURE_DIM = 15

    def __init__(self):
        super().__init__()

        # 初始化状态标志
        self._is_built = False

        # 默认特征向量（用于异常情况）
        self._default_features = np.zeros(self.SPATIAL_FEATURE_DIM, dtype=np.float32)

    def build_from_osm(self, road_network: pd.DataFrame, pois: pd.DataFrame,
                       transit_routes: pd.DataFrame = None):
        """从OSM数据构建增强空间特征提取器"""
        super().build_from_osm(road_network, pois, transit_routes)
        self._is_built = True

    def extract_spatial_features(self, trajectory: np.ndarray) -> np.ndarray:
        """提取 15 维增强空间特征（稳定版，包含异常处理）"""
        if not self._is_built:
            return np.tile(self._default_features, (len(trajectory), 1))

        try:
            features = super().extract_spatial_features(trajectory)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            return features
        except Exception as e:
            print(f"⚠️ 空间特征提取失败，使用默认特征: {e}")
            return np.tile(self._default_features, (len(trajectory), 1))
