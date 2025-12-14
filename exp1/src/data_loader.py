"""
GeoLife数据加载和预处理模块
仅使用GPS轨迹数据，不涉及知识图谱
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
from geopy.distance import geodesic


class GeoLifeDataLoader:
    """GeoLife数据加载器"""
    
    def __init__(self, data_root: str):
        self.data_root = data_root
        
    def load_trajectory(self, file_path: str) -> pd.DataFrame:
        """加载单个轨迹文件
        
        GeoLife数据格式（7列）：
        Field 1: Latitude in decimal degrees
        Field 2: Longitude in decimal degrees
        Field 3: All set to 0 for this dataset (unused)
        Field 4: Altitude in feet (-777 if not valid)
        Field 5: Date - number of days (with fractional part) that have passed since 12/30/1899
        Field 6: Date as a string
        Field 7: Time as a string
        """
        # 跳过前6行元数据
        df = pd.read_csv(file_path, skiprows=6, header=None)
        df.columns = ['latitude', 'longitude', 'unused', 'altitude', 'date_days', 'date', 'time']
        
        # 删除未使用的列（Field 3，全为0）
        df = df.drop('unused', axis=1)
        
        # 合并日期和时间
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # 计算特征
        df = self._calculate_features(df)
        
        return df
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算轨迹特征"""
        # 计算距离（米）
        distances = []
        for i in range(len(df)):
            if i == 0:
                distances.append(0)
            else:
                dist = geodesic(
                    (df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude']),
                    (df.iloc[i]['latitude'], df.iloc[i]['longitude'])
                ).meters
                distances.append(dist)
        df['distance'] = distances
        
        # 计算时间差（秒）
        df['time_diff'] = df['datetime'].diff().dt.total_seconds()
        df['time_diff'] = df['time_diff'].fillna(0)
        df['time_diff'] = df['time_diff'].replace(0, 1e-6)  # 避免除零
        
        # 计算速度（m/s）
        df['speed'] = df['distance'] / df['time_diff']
        
        # 计算加速度（m/s²）
        df['acceleration'] = df['speed'].diff() / df['time_diff']
        df['acceleration'] = df['acceleration'].fillna(0)
        
        # 计算方向变化（度）
        bearings = []
        for i in range(len(df)):
            if i == 0:
                bearings.append(0)
            else:
                lat1, lon1 = df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude']
                lat2, lon2 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
                bearing = self._calculate_bearing(lat1, lon1, lat2, lon2)
                bearings.append(bearing)
        df['bearing'] = bearings
        
        # 计算方向变化率
        df['bearing_change'] = df['bearing'].diff().abs()
        df['bearing_change'] = df['bearing_change'].fillna(0)
        df['bearing_change'] = df['bearing_change'].apply(lambda x: min(x, 360-x))
        
        # 计算总距离（累积）
        df['total_distance'] = df['distance'].cumsum()
        
        # 计算总时间（累积，秒）
        df['total_time'] = df['time_diff'].cumsum()
        
        return df
    
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """计算方位角（度）"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        bearing = np.degrees(np.arctan2(y, x))
        return (bearing + 360) % 360
    
    def load_labels(self, user_id: str) -> pd.DataFrame:
        """加载用户标签数据"""
        labels_path = os.path.join(self.data_root, f"Data/{user_id}/labels.txt")
        if not os.path.exists(labels_path):
            return pd.DataFrame()
        
        df = pd.read_csv(labels_path, sep='\t')
        df['Start Time'] = pd.to_datetime(df['Start Time'])
        df['End Time'] = pd.to_datetime(df['End Time'])
        return df
    
    def segment_trajectory(self, trajectory: pd.DataFrame, labels: pd.DataFrame) -> List[Tuple[pd.DataFrame, str]]:
        """根据标签分割轨迹"""
        segments = []
        
        for _, label_row in labels.iterrows():
            start_time = label_row['Start Time']
            end_time = label_row['End Time']
            mode = label_row['Transportation Mode']
            
            # 提取时间范围内的轨迹点
            mask = (trajectory['datetime'] >= start_time) & (trajectory['datetime'] <= end_time)
            segment = trajectory[mask].copy()
            
            if len(segment) > 0:
                segments.append((segment, mode))
        
        return segments
    
    def get_all_users(self) -> List[str]:
        """获取所有用户ID"""
        data_path = os.path.join(self.data_root, "Data")
        users = []
        for item in os.listdir(data_path):
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path) and item.isdigit():
                users.append(item)
        return sorted(users)


def preprocess_segments(segments: List[Tuple[pd.DataFrame, str]], 
                       min_length: int = 10,
                       max_length: int = 200,
                       target_length: int = 100) -> List[Tuple[np.ndarray, str]]:
    """
    预处理轨迹段，转换为固定长度的序列
    
    Args:
        segments: 轨迹段列表
        min_length: 最小长度阈值
        max_length: 最大长度阈值
        target_length: 目标序列长度
    
    Returns:
        处理后的特征序列和标签
    """
    processed_segments = []
    
    for segment, label in segments:
        if len(segment) < min_length:
            continue
        
        # 提取特征序列（9维特征）
        features = segment[[
            'latitude', 'longitude', 
            'speed', 'acceleration', 
            'bearing_change', 
            'distance', 'time_diff',
            'total_distance', 'total_time'
        ]].values
        
        # 序列长度处理
        if len(features) > max_length:
            # 如果序列太长，进行均匀采样
            indices = np.linspace(0, len(features)-1, target_length, dtype=int)
            features = features[indices]
        elif len(features) < target_length:
            # 如果序列太短，进行填充
            padding = np.zeros((target_length - len(features), features.shape[1]))
            features = np.vstack([features, padding])
        
        # 特征归一化（每个序列独立归一化）
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std
        features = np.clip(features, -5, 5)  # 处理异常值
        
        processed_segments.append((features, label))
    
    return processed_segments

