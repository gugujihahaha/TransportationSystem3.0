"""
OSM 空间特征提取模块 (exp3 - 增强版)
基于 Exp2 的架构，扩展空间特征从 11 维到 15 维

新增特征:
- 地铁入口 (1维)
- 共享单车点 (1维)
- 出租车点 (1维)
- 速度限制 (1维)
- 公交/地铁线路 (1维)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import re
import pickle

from exp2.src.osm_feature_extractor import OsmSpatialExtractor as _BaseOsmSpatialExtractor


class EnhancedOsmSpatialExtractor(_BaseOsmSpatialExtractor):
    """增强版 OSM 空间特征提取器 (Exp3)"""

    # 覆盖特征维度常量
    SPATIAL_FEATURE_DIM = 15

    def __init__(self):
        super().__init__()

        # Exp3 新增数据结构
        self.speed_limits = {}  # 道路速度限制
        self.bus_routes = set()  # 公交线路经过的道路ID
        self.subway_routes = set()  # 地铁线路经过的道路ID

        # 扩展道路类型映射
        self.road_type_mapping.update({
            'motorway': 'car',
            'trunk': 'car',
            'subway_entrance': 'train',
        })

        self.transit_routes = None

    def build_from_osm(self, road_network: pd.DataFrame, pois: pd.DataFrame,
                       transit_routes: pd.DataFrame = None):
        """从OSM数据构建增强空间特征提取器"""
        self.road_network = road_network
        self.pois = pois
        self.transit_routes = transit_routes

        print("\n -> 正在添加道路节点和路段...")
        self._add_road_network()
        print(f" -> 道路网络添加完成。当前图节点数: {self.graph.number_of_nodes()}")

        print(" -> 正在提取速度限制信息...")
        self._extract_speed_limits()
        print(f" -> 速度限制提取完成。覆盖道路数: {len(self.speed_limits)}")

        print(" -> 正在添加 POI 节点...")
        self._add_pois()
        print(f" -> POI 节点添加完成。当前图节点数: {self.graph.number_of_nodes()}")

        if transit_routes is not None:
            print(" -> 正在添加公交/地铁线路...")
            self._add_transit_routes()
            print(f" -> 公交/地铁线路添加完成。覆盖道路数: {len(self.bus_routes) + len(self.subway_routes)}")

        print(" -> 正在构建 KDTree 空间索引...")
        self._build_spatial_indices()
        print(" -> KDTree 索引构建完成！")

        self._link_roads_to_pois()

    def _extract_speed_limits(self):
        """从道路网络中提取速度限制信息"""
        if 'maxspeed' not in self.road_network.columns:
            return

        for _, row in self.road_network.iterrows():
            road_id = row.get('id')
            speed_str = row.get('maxspeed')
            if road_id and speed_str:
                speed = self._parse_speed(speed_str)
                if speed is not None:
                    self.speed_limits[road_id] = speed

    def _parse_speed(self, speed_str) -> Optional[float]:
        """解析速度限制字符串"""
        if pd.isna(speed_str):
            return None

        speed_str = str(speed_str).lower().strip()

        # 处理数值
        match = re.search(r'(\d+\.?\d*)', speed_str)
        if not match:
            return None

        speed = float(match.group(1))

        # 单位转换
        if 'mph' in speed_str:
            speed = speed * 1.60934  # mph -> km/h

        return speed

    def _add_transit_routes(self):
        """添加公交/地铁线路信息"""
        if self.transit_routes is None:
            return

        for _, row in self.transit_routes.iterrows():
            route_type = row.get('type', '').lower()
            road_id = row.get('road_id')

            if road_id:
                if 'bus' in route_type:
                    self.bus_routes.add(road_id)
                elif 'subway' in route_type or 'train' in route_type:
                    self.subway_routes.add(road_id)

    def _batch_query_road_attributes(self, coords: np.ndarray, max_distance: float = 50.0) -> np.ndarray:
        """批量查询道路属性（增强版，包含速度限制和线路信息）"""
        if self.road_kdtree is None:
            return np.zeros((len(coords), 5))

        distances, indices = self.road_kdtree.query(coords, k=1, distance_upper_bound=max_distance)

        results = np.zeros((len(coords), 5))

        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if idx == len(self.road_coords):
                continue

            road_id = self.road_coords[idx][2]

            results[i, 0] = dist
            results[i, 1] = self.road_types.get(road_id, 0)
            results[i, 2] = self.speed_limits.get(road_id, 0)
            results[i, 3] = 1 if road_id in self.bus_routes else 0
            results[i, 4] = 1 if road_id in self.subway_routes else 0

        return results

    def extract_spatial_features(self, trajectory: np.ndarray) -> np.ndarray:
        """提取 15 维增强空间特征 (混合缓存 + 批量查询)"""
        coords = trajectory[:, [0, 1]]

        cached_features = []
        uncached_coords = []
        uncached_indices = []

        for i, (lat, lon) in enumerate(coords):
            grid_key = self._get_grid_key(lat, lon)
            if grid_key in self._grid_cache:
                cached_features.append((i, self._grid_cache[grid_key]))
                self._cache_hits += 1
            else:
                uncached_coords.append([lat, lon])
                uncached_indices.append(i)
                self._cache_misses += 1

        if uncached_coords:
            uncached_coords_array = np.array(uncached_coords)
            new_features = self._batch_query_all(uncached_coords_array)

            for idx, (lat, lon), features in zip(uncached_indices, uncached_coords, new_features):
                grid_key = self._get_grid_key(lat, lon)
                self._grid_cache[grid_key] = features
                cached_features.append((idx, features))

        cached_features.sort(key=lambda x: x[0])
        features = np.array([f for _, f in cached_features])

        return features

    def _batch_query_pois_enhanced(self, coords: np.ndarray,
                                    max_distance: float = 100.0,
                                    poi_types: List[str] = None) -> np.ndarray:
        """批量查询 POI（增强版，包含更多 POI 类型）"""
        if self.poi_kdtree is None:
            return np.zeros((len(coords), 7))

        if poi_types is None:
            poi_types = ['bus_stop', 'parking', 'subway_entrance', 'bicycle_rental', 'taxi']

        results = np.zeros((len(coords), 7))

        for i, coord in enumerate(coords):
            distances, indices = self.poi_kdtree.query(coord, k=5, distance_upper_bound=max_distance)

            nearby_pois = []
            for dist, idx in zip(distances, indices):
                if idx == len(self.poi_coords):
                    continue
                poi_type = self.poi_types.get(idx, 'unknown')
                nearby_pois.append((poi_type, dist))

            for j, poi_type in enumerate(poi_types):
                matching_pois = [(pt, d) for pt, d in nearby_pois if pt == poi_type]
                if matching_pois:
                    results[i, j] = min(d for _, d in matching_pois)
                else:
                    results[i, j] = max_distance

            if nearby_pois:
                results[i, 5] = min(d for _, d in nearby_pois)
                results[i, 6] = len(nearby_pois)

        return results

    def _batch_query_all(self, coords: np.ndarray) -> np.ndarray:
        """批量查询所有特征（增强版，返回15维）"""
        road_attrs = self._batch_query_road_attributes(coords)
        poi_attrs = self._batch_query_pois_enhanced(coords)
        density = self._batch_query_road_density(coords)

        features = np.concatenate([road_attrs, poi_attrs, density], axis=1)
        return features
