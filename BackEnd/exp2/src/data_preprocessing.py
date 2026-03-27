"""
数据预处理模块 (exp2)
提供 OSM 数据加载功能。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import json
import warnings
from tqdm import tqdm

pd.options.mode.chained_assignment = None


class OSMDataLoader:
    """OSM数据加载器"""

    def __init__(self, geojson_path: str):
        self.geojson_path = geojson_path

    def load_osm_data(self) -> Dict:
        """加载OSM GeoJSON数据（支持大文件）"""
        file_size = os.path.getsize(self.geojson_path) / (1024 * 1024)
        print(f"加载OSM数据文件: {self.geojson_path} (大小: {file_size:.2f} MB)")

        if file_size > 100:
            print("检测到大文件，使用流式加载...")
            return self._load_osm_data_streaming()
        else:
            with open(self.geojson_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data

    def _load_osm_data_streaming(self) -> Dict:
        """流式加载大文件（逐行解析）"""
        try:
            import ijson

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

            highway = props.get('highway', '')
            railway = props.get('railway', '')
            amenity = props.get('amenity', '')

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

    def extract_transit_routes(self, osm_data: Dict) -> pd.DataFrame:
        """提取公交/地铁线路信息"""
        routes = []

        for feature in osm_data.get('features', []):
            props = feature.get('properties', {})
            geometry = feature.get('geometry', {})

            route = props.get('route', '')
            highway = props.get('highway', '')
            railway = props.get('railway', '')

            if route in ('bus', 'subway', 'train') or railway in ('rail', 'subway'):
                route_info = {
                    'id': props.get('@id', '') or props.get('id', ''),
                    'type': route or railway or highway,
                    'name': props.get('name', ''),
                    'road_id': props.get('@id', '') or props.get('id', ''),
                    'geometry_type': geometry.get('type', ''),
                }
                routes.append(route_info)

        print(f"提取到 {len(routes)} 条公交/地铁线路")
        return pd.DataFrame(routes) if routes else pd.DataFrame(
            columns=['id', 'type', 'name', 'road_id', 'geometry_type']
        )


def preprocess_trajectory_segments(segments: List[Tuple[pd.DataFrame, str]],
                                   min_length: int = 10) -> List[Tuple[np.ndarray, str]]:
    """预处理轨迹段，转换为固定长度的序列，提取 9 维特征"""
    processed_segments = []

    FIXED_SEQUENCE_LENGTH = 50

    feature_cols = ['latitude', 'longitude', 'speed', 'acceleration',
                    'bearing_change', 'distance', 'time_diff',
                    'total_distance', 'total_time']

    for segment, label in tqdm(segments, desc="[轨迹段预处理]"):
        if len(segment) < min_length:
            continue

        features = segment[feature_cols].values
        current_length = len(features)

        if current_length > FIXED_SEQUENCE_LENGTH:
            indices = np.linspace(0, current_length - 1, FIXED_SEQUENCE_LENGTH, dtype=int)
            features = features[indices]

        elif current_length < FIXED_SEQUENCE_LENGTH:
            padding = np.zeros((FIXED_SEQUENCE_LENGTH - current_length, features.shape[1]))
            features = np.vstack([features, padding])

        processed_segments.append((features, label))

    return processed_segments


__all__ = ["OSMDataLoader", "preprocess_trajectory_segments"]
