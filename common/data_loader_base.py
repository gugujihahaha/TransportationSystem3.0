"""
基础 GeoLife 数据加载器（完全独立，无依赖）
放置位置: 项目根目录/common/data_loader_base.py

这是完全独立的基础模块，不依赖任何实验代码。
功能：
1. 加载 .plt 文件（支持 6/7 列格式）
2. 计算 9 维轨迹特征（向量化）
3. 根据 labels.txt 分割轨迹
4. 标准化序列长度到 100
5. 标签归一化（taxi → car & taxi）
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple
import warnings
from tqdm import tqdm

pd.options.mode.chained_assignment = None


class GeoLifeDataLoader:
    """GeoLife 数据加载器（基础版）"""

    def __init__(self, data_root: str):
        self.data_root = data_root

    def get_all_users(self) -> List[str]:
        """获取所有用户ID"""
        data_dir = os.path.join(self.data_root, 'Data')
        if not os.path.isdir(data_dir):
            data_dir = self.data_root

        if not os.path.isdir(data_dir):
            return []

        users = [
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
               and len(d) == 3
               and d.isdigit()
        ]

        return sorted(users)

    def load_trajectory(self, file_path: str) -> pd.DataFrame:
        """
        加载单个轨迹文件

        支持 6 列和 7 列格式，自动处理
        """
        try:
            df = pd.read_csv(file_path, skiprows=6, header=None)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

        num_cols = df.shape[1]

        # 标准化列名
        if num_cols == 7:
            df.columns = [
                'latitude', 'longitude', 'reserved',
                'altitude', 'date_days', 'date', 'time'
            ]
            df = df.drop('reserved', axis=1)
        elif num_cols == 6:
            df.columns = [
                'latitude', 'longitude',
                'altitude', 'date_days', 'date', 'time'
            ]
        else:
            return pd.DataFrame()

        # 合并日期时间
        df['datetime'] = pd.to_datetime(
            df['date'] + ' ' + df['time'],
            format='%Y-%m-%d %H:%M:%S'
        )
        df = df.sort_values('datetime').reset_index(drop=True)

        # 清洗无效坐标
        invalid_mask = (
                (df['latitude'] < -90) | (df['latitude'] > 90) |
                (df['longitude'] < -180) | (df['longitude'] > 180)
        )

        if invalid_mask.any():
            df = df[~invalid_mask].reset_index(drop=True)
            if len(df) < 2:
                return pd.DataFrame()

        # 计算 9 维特征
        df = self._calculate_features(df)

        # 清理不需要的列
        df = df.drop(columns=['date', 'time', 'date_days', 'altitude'], errors='ignore')

        return df

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """向量化计算 9 维轨迹特征"""

        # 1. 时间差 (秒)
        df['time_diff'] = df['datetime'].diff().dt.total_seconds().fillna(0)

        # 2. 距离 (Haversine 公式)
        lat1 = df['latitude'].shift(1).fillna(df['latitude'].iloc[0])
        lon1 = df['longitude'].shift(1).fillna(df['longitude'].iloc[0])
        lat2 = df['latitude']
        lon2 = df['longitude']

        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
            np.radians, [lat1, lon1, lat2, lon2]
        )

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = (np.sin(dlat / 2.0) ** 2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2)
        c = 2 * np.arcsin(np.sqrt(a))

        R = 6371000  # 地球半径 (米)
        distances = R * c
        distances.iloc[0] = 0.0
        df['distance'] = distances

        # 3. 速度和加速度
        time_diff_safe = df['time_diff'].replace(0, 1e-6)
        df['speed'] = df['distance'] / time_diff_safe
        df['acceleration'] = df['speed'].diff() / time_diff_safe
        df['acceleration'] = df['acceleration'].fillna(0)

        # 4. 方向 (Bearing)
        dlon = lon2_rad - lon1_rad
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = (np.cos(lat1_rad) * np.sin(lat2_rad) -
             np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon))

        bearing = np.degrees(np.arctan2(y, x))
        bearing = (bearing + 360) % 360
        bearing.iloc[0] = 0.0
        df['bearing'] = bearing

        # 5. 方向变化
        df['bearing_change'] = df['bearing'].diff().abs().fillna(0)
        df['bearing_change'] = np.where(
            df['bearing_change'] > 180,
            360 - df['bearing_change'],
            df['bearing_change']
        )

        # 6. 累积特征
        df['total_distance'] = df['distance'].cumsum()
        df['total_time'] = df['time_diff'].cumsum()

        return df

    def load_labels(self, user_id: str) -> pd.DataFrame:
        """加载用户标签数据"""
        labels_path = os.path.join(self.data_root, f"Data/{user_id}/labels.txt")

        if not os.path.exists(labels_path):
            labels_path = os.path.join(self.data_root, f"{user_id}/labels.txt")

        if not os.path.exists(labels_path):
            return pd.DataFrame()

        df = pd.read_csv(labels_path, sep='\t')
        df['Start Time'] = pd.to_datetime(df['Start Time'])
        df['End Time'] = pd.to_datetime(df['End Time'])

        return df

    def segment_trajectory(self, trajectory: pd.DataFrame,
                           labels: pd.DataFrame) -> List[Tuple[pd.DataFrame, str]]:
        """根据标签分割轨迹"""
        segments = []

        if labels.empty:
            return [(trajectory, 'unknown')]

        for _, label_row in labels.iterrows():
            start_time = label_row['Start Time']
            end_time = label_row['End Time']
            mode = label_row['Transportation Mode']

            mask = (
                    (trajectory['datetime'] >= start_time) &
                    (trajectory['datetime'] <= end_time)
            )
            segment = trajectory[mask].copy()

            if len(segment) > 0:
                segments.append((segment, mode))

        return segments


def preprocess_segments_base(
        segments: List[Tuple[pd.DataFrame, str]],
        min_length: int = 10,
        max_length: int = 200,
        target_length: int = 100
) -> List[Tuple[np.ndarray, str]]:
    """
    标准预处理：提取 9 维特征 + 序列长度规范化

    Args:
        segments: 原始轨迹段列表
        min_length: 最小长度（过滤太短的轨迹）
        max_length: 最大长度（超过则采样）
        target_length: 目标序列长度

    Returns:
        List[(features_array, label_str)]
        - features_array: (target_length, 9) numpy 数组
        - label_str: 标签字符串（已标准化）
    """
    processed = []

    # 标签映射：taxi → car & taxi
    LABEL_MAPPING = {
        'taxi': 'car',
        'drive': 'car'
    }

    FINAL_CLASS_NAME = {
        'car': 'car & taxi'
    }

    # 9 维特征列表
    feature_cols = [
        'latitude', 'longitude', 'speed', 'acceleration',
        'bearing_change', 'distance', 'time_diff',
        'total_distance', 'total_time'
    ]

    for segment, label in tqdm(segments, desc="预处理轨迹段"):
        if len(segment) < min_length:
            continue

        # 1. 提取特征
        features = segment[feature_cols].values
        L = len(features)

        # 2. 序列长度规范化
        if L >= target_length:
            if L > max_length:
                # 均匀采样
                indices = np.linspace(0, L - 1, target_length, dtype=int)
                features = features[indices]
            elif L > target_length:
                # 随机裁剪
                start_idx = np.random.randint(0, L - target_length + 1)
                features = features[start_idx:start_idx + target_length]
            # else: L == target_length, 不操作
        else:  # L < target_length
            # 零填充
            padding = np.zeros((target_length - L, features.shape[1]))
            features = np.vstack([features, padding])

        # 3. 标签标准化
        label_lower = label.lower().strip()
        mapped_label = LABEL_MAPPING.get(label_lower, label_lower)
        final_label = FINAL_CLASS_NAME.get(mapped_label, mapped_label)

        processed.append((features, final_label))

    return processed