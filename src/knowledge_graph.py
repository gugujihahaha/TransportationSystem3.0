"""
知识图谱构建模块
从OSM数据构建交通知识图谱
- 使用 KDTree 优化 POI-道路关联速度，将 O(N^2) 复杂度优化为 O(N log N)。
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from geopy.distance import geodesic
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial import KDTree  # 核心修改：引入 KDTree


class TransportationKnowledgeGraph:
    """交通知识图谱"""

    def __init__(self):
        self.graph = nx.MultiDiGraph()  # 有向多重图
        self.road_network = None
        self.pois = None
        self.road_type_mapping = {
            'footway': 'walk',
            'cycleway': 'bike',
            'primary': 'car',
            'secondary': 'car',
            'tertiary': 'car',
            'residential': 'car',
            'bus_stop': 'bus',
            'station': 'train',
            'parking': 'car'
        }

    def build_from_osm(self, road_network: pd.DataFrame, pois: pd.DataFrame):
        """从OSM数据构建知识图谱"""
        self.road_network = road_network
        self.pois = pois

        print(" -> 正在添加道路节点和路段...")
        self._add_road_network()
        print(f" -> 道路网络添加完成。当前图节点数: {self.graph.number_of_nodes()}")

        print(" -> 正在添加 POI 节点...")
        self._add_pois()
        print(f" -> POI 节点添加完成。当前图节点数: {self.graph.number_of_nodes()}")

        # 添加道路-POI关联 (使用 KDTree 优化)
        self._link_roads_to_pois()

    def _add_road_network(self):
        """添加道路网络到知识图谱"""
        for _, road in self.road_network.iterrows():
            road_id = road['id']
            road_type = road.get('highway') or road.get('railway', '')
            coordinates = road.get('coordinates', [])

            if not coordinates:
                continue

            if road['geometry_type'] == 'LineString':
                coords = coordinates
            elif road['geometry_type'] == 'Polygon':
                coords = coordinates[0]
            else:
                continue

            for i, coord in enumerate(coords):
                node_id = f"{road_id}_node_{i}"
                lon, lat = coord[0], coord[1]

                self.graph.add_node(node_id,
                                  type='road_node',
                                  road_id=road_id,
                                  road_type=road_type,
                                  latitude=lat,
                                  longitude=lon)

                if i > 0:
                    prev_node_id = f"{road_id}_node_{i-1}"
                    distance = geodesic((coords[i-1][1], coords[i-1][0]),
                                       (lat, lon)).meters

                    self.graph.add_edge(prev_node_id, node_id,
                                      type='road_segment',
                                      road_type=road_type,
                                      distance=distance)

    def _add_pois(self):
        """添加POI到知识图谱"""
        for _, poi in self.pois.iterrows():
            poi_id = poi['id']
            poi_type = poi['type']
            coordinates = poi.get('coordinates', [])

            if not coordinates:
                continue

            if isinstance(coordinates[0], list):
                lon, lat = coordinates[0][0], coordinates[0][1]
            else:
                lon, lat = coordinates[0], coordinates[1]

            self.graph.add_node(poi_id,
                              type='poi',
                              poi_type=poi_type,
                              name=poi.get('name', ''),
                              latitude=lat,
                              longitude=lon)

    def _link_roads_to_pois(self, max_distance: float = 100.0):
        """
        使用 KDTree 加速 POI 到最近道路节点的链接。
        """
        poi_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'poi']
        road_nodes_data = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('type') == 'road_node']

        if not road_nodes_data:
            print("警告: 无道路节点，跳过关联。")
            return

        print(f" -> 正在关联 {len(poi_nodes)} 个POI到 {len(road_nodes_data)} 个道路节点...")

        # 1. 准备道路节点坐标和ID
        road_coords = np.array([[d['latitude'], d['longitude']] for n, d in road_nodes_data])
        road_node_ids = [n for n, d in road_nodes_data]

        # 2. 构建 KDTree 空间索引
        print(" -> 正在构建 KDTree 空间索引...")
        road_tree = KDTree(road_coords)

        # 3. 准备 POI 坐标
        poi_coords_list = []
        poi_ids_list = []
        for poi_node in poi_nodes:
            poi_data = self.graph.nodes[poi_node]
            poi_coords_list.append([poi_data['latitude'], poi_data['longitude']])
            poi_ids_list.append(poi_node)

        poi_coords = np.array(poi_coords_list)

        # 4. 使用 query_ball_point 批量查找近邻
        # 将米转换为近似的度数
        max_degree_distance = max_distance / 111300.0

        indices = road_tree.query_ball_point(poi_coords, r=max_degree_distance)

        print(" -> 批量查找近邻完成，正在添加边...")

        # 5. 遍历结果并添加边 (使用 tqdm 确保进度显示)
        link_count = 0

        for i, neighbors_indices in enumerate(tqdm(indices, desc="   [KG关联进度]")):
            poi_node = poi_ids_list[i]
            poi_lat, poi_lon = poi_coords[i]

            min_dist = float('inf')
            nearest_road = None

            for j in neighbors_indices:
                road_node = road_node_ids[j]
                road_lat, road_lon = road_coords[j]

                # 重新计算精确的测地距离 (Haversine)
                dist = geodesic((poi_lat, poi_lon), (road_lat, road_lon)).meters

                if dist < min_dist:
                    min_dist = dist
                    nearest_road = road_node

            if nearest_road is not None:
                self.graph.add_edge(poi_node, nearest_road,
                                  type='nearby_road',
                                  distance=min_dist)
                self.graph.add_edge(nearest_road, poi_node,
                                  type='has_poi',
                                  distance=min_dist)
                link_count += 1

        print(f" -> 知识图谱道路-POI关联完成。共添加 {link_count} 条关联边。")

    def get_road_type_for_location(self, lat: float, lon: float,
                                   max_distance: float = 50.0) -> Optional[str]:
        """获取指定位置的道路类型"""
        min_dist = float('inf')
        nearest_type = None

        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'road_node':
                node_lat = data['latitude']
                node_lon = data['longitude']

                dist = geodesic((lat, lon), (node_lat, node_lon)).meters

                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    road_type = data.get('road_type', '')
                    nearest_type = self.road_type_mapping.get(road_type, 'unknown')

        return nearest_type

    def get_nearby_pois(self, lat: float, lon: float,
                       max_distance: float = 200.0) -> List[Dict]:
        """获取附近的POI"""
        nearby_pois = []

        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'poi':
                poi_lat = data['latitude']
                poi_lon = data['longitude']

                dist = geodesic((lat, lon), (poi_lat, poi_lon)).meters

                if dist < max_distance:
                    nearby_pois.append({
                        'id': node,
                        'type': data.get('poi_type', ''),
                        'name': data.get('name', ''),
                        'distance': dist
                    })

        return sorted(nearby_pois, key=lambda x: x['distance'])

    def extract_kg_features(self, trajectory: pd.DataFrame) -> np.ndarray:
        """从知识图谱提取特征"""
        features = []

        # 为特征提取过程添加 tqdm
        for _, point in tqdm(trajectory.iterrows(), desc="   [KG特征提取]"):
            lat = point['latitude']
            lon = point['longitude']

            # 特征1: 道路类型（one-hot编码）
            road_type = self.get_road_type_for_location(lat, lon)
            road_type_features = self._encode_road_type(road_type)

            # 特征2: 附近POI信息
            nearby_pois = self.get_nearby_pois(lat, lon)
            poi_features = self._encode_pois(nearby_pois)

            # 特征3: 道路密度（附近道路节点数量）
            road_density = self._calculate_road_density(lat, lon)

            # 合并特征
            point_features = np.concatenate([
                road_type_features,
                poi_features,
                [road_density]
            ])

            features.append(point_features)

        return np.array(features)

    def _encode_road_type(self, road_type: Optional[str]) -> np.ndarray:
        """编码道路类型为one-hot向量"""
        types = ['walk', 'bike', 'car', 'bus', 'train', 'unknown']
        encoding = np.zeros(len(types))

        if road_type and road_type in types:
            idx = types.index(road_type)
            encoding[idx] = 1.0

        return encoding

    def _encode_pois(self, pois: List[Dict]) -> np.ndarray:
        """编码POI信息"""
        # 特征: [是否有公交站, 是否有地铁站, 是否有停车场, 最近POI距离]
        features = np.zeros(4)

        if pois:
            poi_types = [poi['type'] for poi in pois]
            if 'bus_stop' in poi_types:
                features[0] = 1.0
            if 'station' in poi_types:
                features[1] = 1.0
            if 'parking' in poi_types:
                features[2] = 1.0
            features[3] = pois[0]['distance'] / 200.0  # 归一化到0-1

        return features

    def _calculate_road_density(self, lat: float, lon: float,
                               radius: float = 100.0) -> float:
        """计算道路密度"""
        count = 0

        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'road_node':
                node_lat = data['latitude']
                node_lon = data['longitude']

                dist = geodesic((lat, lon), (node_lat, node_lon)).meters
                if dist < radius:
                    count += 1

        # 归一化（假设最大密度为50）
        return min(count / 50.0, 1.0)

    def get_graph_statistics(self) -> Dict:
        """获取知识图谱统计信息"""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'road_nodes': len([n for n, d in self.graph.nodes(data=True)
                             if d.get('type') == 'road_node']),
            'poi_nodes': len([n for n, d in self.graph.nodes(data=True)
                            if d.get('type') == 'poi']),
            'poi_links': len([u for u, v, k, d in self.graph.edges(data=True, keys=True)
                              if d.get('type') == 'nearby_road'])
        }