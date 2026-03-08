"""
数据预处理模块 (exp4)
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

            public_transport = props.get('public_transport', '')
            highway = props.get('highway', '')
            railway = props.get('railway', '')
            amenity = props.get('amenity', '')

            if public_transport or highway in ['bus_stop', 'platform'] or railway in ['station', 'stop'] or amenity in ['bus_station', 'taxi']:
                poi_info = {
                    'id': props.get('@id', '') or props.get('id', ''),
                    'type': public_transport or highway or railway or amenity,
                    'geometry_type': geometry.get('type', ''),
                    'coordinates': geometry.get('coordinates', [])
                }
                pois.append(poi_info)

        print(f"提取到 {len(pois)} 个POI")
        return pd.DataFrame(pois)


__all__ = ["OSMDataLoader"]