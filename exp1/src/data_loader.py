"""
GeoLife数据加载和预处理模块 (Robust and Enhanced Version)
处理GeoLife轨迹数据，并确保数据加载的鲁棒性。

关键增强点:
1. 鲁棒地处理 6 列或 7 列 GeoLife 文件。
2. 强制清洗无效的经纬度坐标点。
3. 轨迹特征（包括9维）计算已完全向量化，提高性能。
4. 修复：新增 get_all_users 方法。

注意：已移除序列独立归一化，改为在 train_maso.py 中进行全局归一化。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
from geopy.distance import geodesic
import warnings
from tqdm import tqdm

from common.geolife_data_loader import BaseGeoLifeDataLoader

pd.options.mode.chained_assignment = None


class GeoLifeDataLoader(BaseGeoLifeDataLoader):
    """exp1 数据加载器，继承基类并保持原有功能"""

    def load_labels(self, user_id: str) -> pd.DataFrame:
        """加载用户标签数据"""
        labels_path = os.path.join(self.data_root, f"{user_id}/labels.txt")
        if not os.path.exists(labels_path):
            labels_path = os.path.join(self.data_root, f"Data/{user_id}/labels.txt")
        if not os.path.exists(labels_path):
            return pd.DataFrame()
        df = pd.read_csv(labels_path, sep='\t')
        df['Start Time'] = pd.to_datetime(df['Start Time'])
        df['End Time'] = pd.to_datetime(df['End Time'])
        return df


def preprocess_segments(segments: List[Tuple[pd.DataFrame, str]],
                       min_length: int = 10,
                       max_length: int = 200,
                       target_length: int = 100) -> List[Tuple[np.ndarray, str]]:
    """
    【仅传统模式使用】
    快速模式（use_base_data=True）使用 Exp1DataAdapter，不经过此函数。
    保留此函数仅为向后兼容。

    预处理轨迹段，转换为固定长度的序列，并进行标签重映射。
    确保所有输出序列的长度都严格等于 target_length。
    注意：此函数不再进行归一化，归一化操作在 train_maso.py 中使用全局统计量完成。
    """
    processed_segments = []

    MAPPING = {
        'taxi': 'Car & taxi',
        'car': 'Car & taxi',
        'drive': 'Car & taxi',
        'bus': 'Bus',
        'walk': 'Walk',
        'bike': 'Bike',
        'train': 'Train',
        'subway': 'Subway',
        'railway': 'Train',
        'airplane': 'Airplane'
    }

    feature_cols = ['latitude', 'longitude', 'speed', 'acceleration',
                    'bearing_change', 'distance', 'time_diff',
                    'total_distance', 'total_time']

    for segment, label in tqdm(segments, desc="[轨迹段预处理]"):
        if len(segment) < min_length:
            continue

        features = segment[feature_cols].values
        L = len(features)

        if L >= target_length:
            if L > max_length:
                indices = np.linspace(0, L-1, target_length, dtype=int)
                features = features[indices]
            elif L > target_length:
                start_index = np.random.randint(0, L - target_length + 1)
                features = features[start_index:start_index + target_length]
            else:
                pass

        elif L < target_length:
            padding = np.zeros((target_length - L, features.shape[1]))
            features = np.vstack([features, padding])

        mapped_label = MAPPING.get(label, label)

        final_label = mapped_label.capitalize() if mapped_label else mapped_label

        processed_segments.append((features, final_label))

    print(f"\n标签映射字典: {MAPPING}")
    return processed_segments


__all__ = ["GeoLifeDataLoader", "preprocess_segments"]
