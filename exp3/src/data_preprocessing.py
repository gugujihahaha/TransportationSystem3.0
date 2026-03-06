"""
数据预处理模块 (exp3)
继承 exp2 的 GeoLifeDataLoader，无需重复实现。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from typing import List, Tuple

from exp2.src.data_preprocessing import (
    GeoLifeDataLoader as _BaseGeoLifeDataLoader,
    preprocess_trajectory_segments,
)
from exp3.src.osm_feature_extractor import OSMDataLoader


class GeoLifeDataLoader(_BaseGeoLifeDataLoader):
    """
    exp3 数据加载器。
    继承 exp2 的全部功能，仅覆盖 segment_trajectory 以修正标签映射。
    """

    LABEL_MAPPING = {
        'taxi': 'Car & taxi', 'car': 'Car & taxi',
        'bus': 'Bus', 'walk': 'Walk', 'bike': 'Bike',
        'train': 'Train', 'subway': 'Subway', 'airplane': 'Airplane',
    }

    def segment_trajectory(self, trajectory: pd.DataFrame, labels: pd.DataFrame) -> List[Tuple[pd.DataFrame, str]]:
        """按标签分割轨迹，并统一标签为7大类。"""
        segments = []
        for _, row in labels.iterrows():
            st = pd.to_datetime(row["Start Time"])
            et = pd.to_datetime(row["End Time"])
            raw_mode = str(row["Transportation Mode"]).lower().strip()
            mode = self.LABEL_MAPPING.get(raw_mode, raw_mode.capitalize())
            mask = (trajectory["datetime"] >= st) & (trajectory["datetime"] <= et)
            seg = trajectory[mask].copy()
            if len(seg) >= 10:
                segments.append((seg, mode))
        return segments


__all__ = ["GeoLifeDataLoader", "OSMDataLoader", "preprocess_trajectory_segments"]
