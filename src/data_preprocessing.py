"""
数据预处理模块
处理GeoLife轨迹数据和OSM数据
- 修正 GeoLifeDataLoader 中 load_trajectory 的列名错误。
- 增加对 GeoLife 6列数据的鲁棒性处理。
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict
import json
from geopy.distance import geodesic
import warnings
# 暂时忽略设置副本警告，因为它在 pandas 中通常是良性的
pd.options.mode.chained_assignment = None

class GeoLifeDataLoader:
    """GeoLife数据加载器"""

    def __init__(self, data_root: str):
        self.data_root = data_root

    def load_trajectory(self, file_path: str) -> pd.DataFrame:
        """加载单个轨迹文件，鲁棒地处理 6 列或 7 列数据"""

        # 1. 读取原始数据，跳过前6行元数据
        try:
            df = pd.read_csv(file_path, skiprows=6, header=None)
        except pd.errors.EmptyDataError:
            warnings.warn(f"文件 {file_path} 为空，跳过。")
            return pd.DataFrame()

        num_cols = df.shape[1]

        # 2. 根据列数分配正确的列名并进行标准化
        if num_cols == 7:
            # 标准 GeoLife 格式：7列
            df.columns = ['latitude', 'longitude', 'reserved', 'altitude', 'date_days', 'date', 'time']
            df['reserved'] = df['reserved'].replace({0.0: 0, 0: 0}).astype(int) # 确保保留字段为整数或0
        elif num_cols == 6:
            # 常见错误格式：6列，通常缺失 'reserved' 或 'altitude'。
            # 假设该格式缺失的是 'reserved' 字段，或者 'reserved' 和 'altitude' 合并了
            # 为了简化和鲁棒性，我们假设缺失的是 'reserved' 占位符。
            # 如果缺失的是 altitude，则使用 NaN 填充 altitude。
            try:
                # 尝试用 6 列模式加载，并假设缺失 altitude 列 (第4列)
                # 原始顺序: lat, lon, res, alt, date_days, date, time
                # 6列顺序: lat, lon, res, date_days, date, time （缺失 alt）
                df.columns = ['latitude', 'longitude', 'reserved', 'date_days', 'date', 'time']
                df.insert(3, 'altitude', np.nan)  # 在第4列位置插入 altitude 填充 NaN
                df['reserved'] = df['reserved'].replace({0.0: 0, 0: 0}).astype(int)
            except ValueError:
                # 如果上述尝试失败，可能缺失的是 reserved 列 (第3列)
                # 假设顺序: lat, lon, alt, date_days, date, time （缺失 res）
                df.columns = ['latitude', 'longitude', 'altitude', 'date_days', 'date', 'time']
                df.insert(2, 'reserved', 0) # 在第3列位置插入 reserved 填充 0

            warnings.warn(f"文件 {os.path.basename(file_path)} 只有 6 列，已尝试标准化为 7 列。")
        else:
            raise ValueError(f"文件 {file_path} 列数为 {num_cols}，既非 6 也非 7。无法处理。")

        # 3. 后续处理 (适用于所有 7 列标准化后的数据)

        # 合并日期和时间
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.sort_values('datetime').reset_index(drop=True)

        # 计算速度、加速度等特征
        df = self._calculate_features(df)

        # 清理不必要的列，只保留用于后续分析的列
        df = df.drop(columns=['date', 'time', 'date_days'])

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

        # 计算速度（m/s）
        df['speed'] = df['distance'] / (df['time_diff'] + 1e-6)  # 避免除零

        # 计算加速度（m/s²）
        df['acceleration'] = df['speed'].diff() / (df['time_diff'] + 1e-6)
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


class OSMDataLoader:
    """OSM数据加载器"""

    def __init__(self, geojson_path: str):
        self.geojson_path = geojson_path

    def load_osm_data(self) -> Dict:
        """加载OSM GeoJSON数据（支持大文件）"""
        import os

        file_size = os.path.getsize(self.geojson_path) / (1024 * 1024)  # MB
        print(f"加载OSM数据文件: {self.geojson_path} (大小: {file_size:.2f} MB)")

        # 对于大文件，使用流式加载
        if file_size > 100:  # 大于100MB
            print("检测到大文件，使用流式加载...")
            return self._load_osm_data_streaming()
        else:
            with open(self.geojson_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data

    def _load_osm_data_streaming(self) -> Dict:
        """流式加载大文件（逐行解析）"""
        try:
            import ijson  # 需要安装: pip install ijson

            # 使用ijson进行流式解析
            with open(self.geojson_path, 'rb') as f:
                parser = ijson.items(f, 'features.item')
                features = []
                count = 0

                for feature in parser:
                    features.append(feature)
                    count += 1
                    if count % 10000 == 0:
                        print(f"  已加载 {count} 个特征...")

                print(f"总共加载 {count} 个特征")

                return {
                    'type': 'FeatureCollection',
                    'features': features
                }
        except ImportError:
            print("警告: ijson未安装，使用标准JSON加载（可能较慢且占用内存）")
            print("建议安装: pip install ijson 以获得更好的大文件处理性能")
            print("正在加载...")
            with open(self.geojson_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"加载完成，共 {len(data.get('features', []))} 个特征")
            return data
        except Exception as e:
            print(f"流式加载失败: {e}")
            print("回退到标准JSON加载...")
            with open(self.geojson_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data

    def extract_road_network(self, osm_data: Dict) -> pd.DataFrame:
        """提取道路网络信息"""
        roads = []

        for feature in osm_data.get('features', []):
            props = feature.get('properties', {})
            geometry = feature.get('geometry', {})

            # 提取道路类型
            highway = props.get('highway', '')
            railway = props.get('railway', '')

            if highway or railway:
                road_info = {
                    'id': props.get('@id', '') or props.get('id', ''),
                    'highway': highway,
                    'railway': railway,
                    'geometry_type': geometry.get('type', ''),
                    'coordinates': geometry.get('coordinates', [])
                }
                roads.append(road_info)

        print(f"提取到 {len(roads)} 条道路")
        return pd.DataFrame(roads)

    def extract_pois(self, osm_data: Dict) -> pd.DataFrame:
        """提取POI信息（公交站、地铁站等）"""
        pois = []

        for feature in osm_data.get('features', []):
            props = feature.get('properties', {})
            geometry = feature.get('geometry', {})

            # 提取公交站、地铁站等
            highway = props.get('highway', '')
            railway = props.get('railway', '')
            amenity = props.get('amenity', '')

            # 支持更多POI类型（根据新数据可能包含的类型）
            poi_types = ['bus_stop', 'station', 'parking', 'taxi', 'subway_entrance']

            if (highway in poi_types or
                railway in poi_types or
                amenity in poi_types or
                highway == 'bus_stop' or
                railway == 'station' or
                amenity in ['parking', 'taxi']):

                poi_type = highway or railway or amenity
                poi_info = {
                    'id': props.get('@id', '') or props.get('id', ''),
                    'type': poi_type,
                    'name': props.get('name', ''),
                    'coordinates': geometry.get('coordinates', [])
                }
                pois.append(poi_info)

        print(f"提取到 {len(pois)} 个POI")
        return pd.DataFrame(pois)


def preprocess_trajectory_segments(segments: List[Tuple[pd.DataFrame, str]],
                                   min_length: int = 10) -> List[Tuple[np.ndarray, str]]:
    """预处理轨迹段，转换为固定长度的序列"""
    processed_segments = []

    for segment, label in segments:
        if len(segment) < min_length:
            continue

        # 提取特征序列
        # 注意：这里需要确保 segment 中包含这些特征列
        features = segment[['latitude', 'longitude', 'speed', 'acceleration',
                           'bearing_change', 'distance', 'time_diff']].values

        # 如果序列太长，进行采样
        if len(features) > 200:
            indices = np.linspace(0, len(features)-1, 200, dtype=int)
            features = features[indices]

        # 如果序列太短，进行填充
        if len(features) < 50:
            padding = np.zeros((50 - len(features), features.shape[1]))
            features = np.vstack([features, padding])

        processed_segments.append((features, label))

    return processed_segments