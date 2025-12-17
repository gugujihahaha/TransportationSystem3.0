"""
MASO-MSF 模块: 数据加载与 MASO 特征图生成
复现论文: 《基于GPS轨迹多尺度表达的交通出行方式识别方法》- 马妍莉
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict
from tqdm import tqdm
import torch

# 标签映射：遵循论文实验设置，将 taxi 合并入 car
MAPPING = {
    'walk': 'walk',
    'bike': 'bike',
    'bus': 'bus',
    'car': 'car & taxi',
    'taxi': 'car & taxi',
    'train': 'train',
    'subway': 'train'  # 论文通常将轨道交通合并
}


class GeoLifeDataLoader:
    """GeoLife 数据加载器 - MASO 图像版本"""

    def __init__(self, data_root: str):
        self.data_root = data_root

    def get_all_users(self) -> List[str]:
        """扫描数据目录获取所有用户ID"""
        data_dir = os.path.join(self.data_root, 'Data')
        if not os.path.isdir(data_dir):
            data_dir = self.data_root
        if not os.path.isdir(data_dir):
            return []
        users = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        return sorted(users)

    def load_trajectory(self, plt_file: str) -> pd.DataFrame:
        """加载单个 .plt 文件并进行初步清洗"""
        try:
            # 兼容 6 行头部的标准 GeoLife 格式
            df = pd.read_csv(plt_file, skiprows=6, header=None)
            if df.shape[1] < 7:
                return pd.DataFrame()

            df.columns = ['lat', 'lon', 'unused', 'alt', 'days', 'date', 'time']
            # 在 src/data_loader.py 中修改大约第 50 行
            df['datetime'] = pd.to_datetime(
                df['date'] + ' ' + df['time'],
                format='%Y-%m-%d %H:%M:%S',
                errors='coerce'  # 遇到非法格式跳过，增强代码鲁棒性
            )

            # 基础清洗：剔除异常经纬度
            df = df[(df['lat'] >= -90) & (df['lat'] <= 90) &
                    (df['lon'] >= -180) & (df['lon'] <= 180)]
            return df[['lat', 'lon', 'datetime']]
        except Exception:
            return pd.DataFrame()

    def calculate_point_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算点级运动属性 (论文 3.1.1 节)
        计算：距离、速度、加速度、转角变化率
        """
        if len(df) < 2: return df

        # 转换为 numpy 加速计算
        coords = df[['lat', 'lon']].values
        times = df['datetime'].values.astype('datetime64[s]').astype(np.float64)

        # 计算距离 (简单欧氏距离近似，论文在网格化时对空间精度有容忍度)
        d_coords = np.diff(coords, axis=0)
        dist = np.sqrt(np.sum(d_coords ** 2, axis=1)) * 111000  # 近似米

        # 计算时间间隔
        dt = np.diff(times)
        dt[dt <= 0] = 0.1  # 防止除零

        # 速度 (m/s)
        velocity = dist / dt

        # 加速度 (m/s^2)
        dv = np.diff(velocity, prepend=velocity[0])
        accel = dv / dt

        # 方向角 (Bearing) 与方向变化率
        # 这里简化处理：计算相邻向量的夹角
        angles = np.arctan2(d_coords[:, 1], d_coords[:, 0])
        bearing_change = np.diff(angles, prepend=angles[0])

        # 填充回 DataFrame (补齐长度)
        df = df.iloc[1:].copy()
        df['speed'] = velocity
        df['accel'] = accel
        df['bearing_rate'] = bearing_change

        return df

    def generate_maso_images(self, segment_df: pd.DataFrame, scales=[32, 64]) -> Dict[int, np.ndarray]:
        """
        [复现核心] 将轨迹段投影为多尺度图像 (MASO 结构)
        通道 0: Speed, 通道 1: Accel, 通道 2: Bearing Change
        """
        # 1. 坐标平移到原点并归一化
        lats = segment_df['lat'].values
        lons = segment_df['lon'].values

        lat_min, lat_max = lats.min(), lats.max()
        lon_min, lon_max = lons.min(), lons.max()

        # 防止分母为0
        lat_range = (lat_max - lat_min) if lat_max != lat_min else 1e-7
        lon_range = (lon_max - lon_min) if lon_max != lon_min else 1e-7

        norm_lat = (lats - lat_min) / lat_range
        norm_lon = (lons - lon_min) / lon_range

        # 获取属性值
        attrs = segment_df[['speed', 'accel', 'bearing_rate']].values

        images = {}
        for S in scales:
            # 创建空图像 (3, S, S)
            img = np.zeros((3, S, S), dtype=np.float32)

            # 计算网格坐标
            grid_y = np.clip((norm_lat * (S - 1)).astype(int), 0, S - 1)
            grid_x = np.clip((norm_lon * (S - 1)).astype(int), 0, S - 1)

            # 将属性填充到网格中 (如果有重叠点，采用覆盖或均值，此处采用覆盖简化)
            # 论文中建议对网格进行全局 Min-Max 归一化，此处预留
            img[0, grid_y, grid_x] = attrs[:, 0]  # Speed
            img[1, grid_y, grid_x] = attrs[:, 1]  # Accel
            img[2, grid_y, grid_x] = attrs[:, 2]  # Bearing

            images[S] = img

        return images


def preprocess_segments(loader: GeoLifeDataLoader,
                        user_list: List[str],
                        min_points: int = 10) -> List[Tuple[Dict[int, np.ndarray], str]]:
    """扫描所有用户，提取带标签的轨迹图像对"""
    all_data = []

    for user in tqdm(user_list, desc="读取GeoLife用户数据"):
        user_dir = os.path.join(loader.data_root, 'Data', user)
        label_file = os.path.join(user_dir, 'labels.txt')

        if not os.path.exists(label_file):
            continue

        # 读取标签
        labels_df = pd.read_csv(label_file, sep='\t')
        labels_df['Start Time'] = pd.to_datetime(labels_df['Start Time'])
        labels_df['End Time'] = pd.to_datetime(labels_df['End Time'])

        plt_dir = os.path.join(user_dir, 'Trajectory')
        for plt_name in os.listdir(plt_dir):
            df = loader.load_trajectory(os.path.join(plt_dir, plt_name))
            if df.empty: continue

            # 根据时间戳匹配标签
            for _, row in labels_df.iterrows():
                mask = (df['datetime'] >= row['Start Time']) & (df['datetime'] <= row['End Time'])
                seg = df.loc[mask]

                if len(seg) >= min_points:
                    # 计算点级属性
                    seg = loader.calculate_point_features(seg)
                    # 生成 MASO 图像
                    maso_imgs = loader.generate_maso_images(seg, scales=[32, 64])

                    label = MAPPING.get(row['Transportation Mode'], None)
                    if label:
                        all_data.append((maso_imgs, label))

    return all_data