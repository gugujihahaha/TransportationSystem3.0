"""
GeoLife数据加载器基类
提供跨实验复用的数据加载和预处理功能。
"""
import os
import pandas as pd
import numpy as np
from typing import List, Tuple
import warnings


class BaseGeoLifeDataLoader:
    """GeoLife数据加载器基类"""

    def __init__(self, data_root: str):
        self.data_root = data_root

    def get_all_users(self) -> List[str]:
        """扫描数据根目录，获取所有用户ID"""
        data_dir = os.path.join(self.data_root, 'Data')
        if not os.path.isdir(data_dir):
            data_dir = self.data_root

        if not os.path.isdir(data_dir):
            print(f"警告: GeoLife数据目录未找到: {self.data_root}")
            return []

        users = [d for d in os.listdir(data_dir)
                 if os.path.isdir(os.path.join(data_dir, d)) and len(d) == 3 and d.isdigit()]
        return sorted(users)

    def load_trajectory(self, file_path: str) -> pd.DataFrame:
        """加载单个轨迹文件，鲁棒地处理 6 列或 7 列数据，并进行数据清洗"""
        try:
            df = pd.read_csv(file_path, skiprows=6, header=None)
        except pd.errors.EmptyDataError:
            warnings.warn(f"文件 {file_path} 为空，跳过。")
            return pd.DataFrame()

        num_cols = df.shape[1]

        if num_cols == 7:
            df.columns = ['latitude', 'longitude', 'reserved', 'altitude', 'date_days', 'date', 'time']
            df['reserved'] = df['reserved'].replace({0.0: 0, 0: 0}).astype(int)
        elif num_cols == 6:
            try:
                df.columns = ['latitude', 'longitude', 'altitude', 'date_days', 'date', 'time']
                df.insert(2, 'reserved', 0)
            except ValueError:
                df.columns = ['latitude', 'longitude', 'reserved', 'date_days', 'date', 'time']
                df.insert(3, 'altitude', np.nan)
                df['reserved'] = df['reserved'].replace({0.0: 0, 0: 0}).astype(int)
            warnings.warn(f"文件 {os.path.basename(file_path)} 只有 6 列，已尝试标准化为 7 列。")
        else:
            return pd.DataFrame()

        df = df.drop('reserved', axis=1, errors='ignore')

        df['datetime'] = pd.to_datetime(
            df['date'] + ' ' + df['time'],
            format='%Y-%m-%d %H:%M:%S'
        )
        df = df.sort_values('datetime').reset_index(drop=True)

        invalid_lat_mask = (df['latitude'] < -90) | (df['latitude'] > 90)
        invalid_lon_mask = (df['longitude'] < -180) | (df['longitude'] > 180)

        if invalid_lat_mask.any() or invalid_lon_mask.any():
            warnings.warn(f"文件 {os.path.basename(file_path)} 发现无效坐标，正在删除。")
            df = df[~invalid_lat_mask & ~invalid_lon_mask].reset_index(drop=True)
            if len(df) < 2:
                return pd.DataFrame()

        df = self._calculate_features(df)
        df = df.drop(columns=['date', 'time', 'date_days', 'altitude'], errors='ignore')
        return df

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """向量化计算轨迹特征，包括 9 维特征"""
        df['time_diff'] = df['datetime'].diff().dt.total_seconds().fillna(0)

        lat1 = df['latitude'].shift(1).fillna(df['latitude'].iloc[0])
        lon1 = df['longitude'].shift(1).fillna(df['longitude'].iloc[0])
        lat2 = df['latitude']
        lon2 = df['longitude']

        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = np.sin(dlat/2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        R = 6371000
        distances = R * c
        distances.iloc[0] = 0.0
        df['distance'] = distances

        time_diff_safe = df['time_diff'].replace(0, 1e-6)
        df['speed'] = df['distance'] / time_diff_safe
        df['acceleration'] = df['speed'].diff() / time_diff_safe
        df['acceleration'] = df['acceleration'].fillna(0)

        df['bearing'] = self._calculate_bearing_vectorized(lat1_rad.values, lon1_rad.values, lat2_rad.values, lon2_rad.values)
        df['bearing_change'] = df['bearing'].diff().abs().fillna(0)
        df['bearing_change'] = np.where(df['bearing_change'] > 180, 360 - df['bearing_change'], df['bearing_change'])

        df['total_distance'] = df['distance'].cumsum()
        df['total_time'] = df['time_diff'].cumsum()

        return df

    def _calculate_bearing_vectorized(self, lat1: np.ndarray, lon1: np.ndarray,
                                        lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """向量化计算方位角（度）"""
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        bearing = np.degrees(np.arctan2(y, x))
        bearing = (bearing + 360) % 360
        bearing[0] = 0.0
        return bearing

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

    def segment_trajectory(self, trajectory: pd.DataFrame, labels: pd.DataFrame) -> List[Tuple[pd.DataFrame, str]]:
        """根据标签分割轨迹（子类可覆盖）"""
        segments = []
        if labels.empty:
            return [(trajectory, 'unknown')]
        for _, label_row in labels.iterrows():
            start_time = label_row['Start Time']
            end_time = label_row['End Time']
            mode = label_row['Transportation Mode']
            mask = (trajectory['datetime'] >= start_time) & (trajectory['datetime'] <= end_time)
            segment = trajectory[mask].copy()
            if len(segment) > 0:
                segments.append((segment, mode))
        return segments


__all__ = ['BaseGeoLifeDataLoader']
