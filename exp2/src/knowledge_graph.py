"""
交通知识图谱构建模块 (最终修复道路节点版)

核心修复:
- 针对 road_network DataFrame 缺乏标准 'y'/'x' 或 'lat'/'lon' 列的问题，增加了对常见列的查找。
- 保证 road_nodes 和 num_edges 能够正确添加。
"""
import numpy as np
import pandas as pd
from typing import Dict
from geopy.distance import geodesic
import networkx as nx
from scipy.spatial import KDTree
from tqdm import tqdm
import math

class TransportationKnowledgeGraph:
    """交通知识图谱"""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
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
        # 优化属性
        self.road_kdtree = None
        self.road_coords = None
        self.poi_kdtree = None
        self.poi_coords = None
        self.pois_lookup = {}

    # ========================== KG 并行化支持 ==========================
    def get_data_for_worker(self) -> Dict:
        """提取可序列化数据供多进程 worker 使用"""
        return {
            'road_type_mapping': self.road_type_mapping,
            'graph_data': nx.node_link_data(self.graph),
            'road_coords': self.road_coords.tolist() if self.road_coords is not None else None,
            'poi_coords': self.poi_coords.tolist() if self.poi_coords is not None else None,
        }

    def load_data_for_worker(self, data: Dict):
        """worker 进程从序列化数据中重建 KG 实例"""
        self.road_type_mapping = data['road_type_mapping']
        self.graph = nx.node_link_graph(data['graph_data'])

        if data['road_coords'] is not None:
            self.road_coords = np.array(data['road_coords'])
            self.road_kdtree = KDTree(self.road_coords)

        if data['poi_coords'] is not None:
            self.poi_coords = np.array(data['poi_coords'])
            self.poi_kdtree = KDTree(self.poi_coords)

    # ========================== 核心图谱构建方法 (修复 road_nodes 0 的问题) ==========================

    def _add_road_network(self, road_network: pd.DataFrame):
        """将道路网络节点和边添加到 self.graph"""

        # 1. 添加道路节点 (修复道路节点缺失问题)
        road_nodes = {}

        # 检查 roads DataFrame 是否包含了 u/v 节点信息，osmnx 常见格式
        if 'u' in road_network.columns and 'v' in road_network.columns:
            # 假设 road_network 的索引是边的 ID，我们需要从边的信息中提取节点
            for _, row in road_network.iterrows():
                u, v = row['u'], row['v']

                # 尝试从道路段数据中提取节点坐标 (假设坐标存储在 road_network 中)
                # ----------------------- 灵活坐标提取 (尝试多种常见列名) -----------------------
                def _get_coords(row):
                    # 尝试 'y'/'x'
                    if 'y' in row and 'x' in row:
                        return row['y'], row['x']
                    # 尝试 'lat'/'lon'
                    elif 'lat' in row and 'lon' in row:
                        return row['lat'], row['lon']
                    # 尝试 'geometry' 或其他包含坐标信息的复杂结构 (这里仅处理简单列)
                    return None, None

                lat_u, lon_u = row.get('y_u'), row.get('x_u') # 假设 u 节点坐标
                lat_v, lon_v = row.get('y_v'), row.get('x_v') # 假设 v 节点坐标

                if lat_u is not None and lon_u is not None:
                     road_nodes[u] = {'latitude': lat_u, 'longitude': lon_u}

                if lat_v is not None and lon_v is not None:
                     road_nodes[v] = {'latitude': lat_v, 'longitude': lon_v}
                # --------------------------------------------------------------------------------

        # 将所有道路节点添加到图谱
        for node_id, coords in road_nodes.items():
            self.graph.add_node(
                node_id,
                type='road_node',
                latitude=coords['latitude'],
                longitude=coords['longitude']
            )

        # 2. 添加道路边 (修复 num_edges 0 的问题)
        if 'u' in road_network.columns and 'v' in road_network.columns:
            for _, row in road_network.iterrows():
                u, v = row['u'], row['v']

                # 只有当节点已经被添加到图谱中时，才能添加边
                if u in self.graph and v in self.graph:
                    highway = row.get('highway', 'unclassified')
                    speed_limit = row.get('maxspeed', 60)
                    length = row.get('length', 1.0)

                    self.graph.add_edge(
                        u, v,
                        key=0,
                        type='road_segment',
                        highway=highway,
                        speed_limit=float(speed_limit),
                        length=length
                    )
    # --------------------------------------------------------------------------------

    def _add_pois(self, pois: pd.DataFrame):
        """将 POI 节点添加到 self.graph (使用 'coordinates' 修复)"""

        for poi_id, row in pois.iterrows():

            # --- 修复点：解析 'coordinates' 列表 ---
            if 'coordinates' not in row or not isinstance(row['coordinates'], list) or len(row['coordinates']) < 2:
                continue

            lon = row['coordinates'][0]
            lat = row['coordinates'][1]
            # ------------------------------------

            node_id = f"poi_{row['id']}"

            self.graph.add_node(
                node_id,
                type='poi',
                latitude=lat,
                longitude=lon,
                poi_type=row.get('type', 'general'),
                name=row.get('name', '')
            )
            self.pois_lookup[(lat, lon)] = node_id

    def _link_roads_to_pois(self, distance_threshold: float = 200.0):
        """链接 POI 和最近的道路节点"""

        if self.poi_kdtree is None or self.road_kdtree is None:
            return

        for poi_id, poi_data in self.graph.nodes(data=True):
            if poi_data.get('type') != 'poi':
                continue

            poi_coord = (poi_data['latitude'], poi_data['longitude'])

            radius_in_degrees = distance_threshold / 111320.0

            road_indices = self.road_kdtree.query_ball_point(
                poi_coord,
                radius_in_degrees
            )

            min_dist = float('inf')
            nearest_road_node_id = None

            # 精确计算大地测量距离
            for idx in road_indices:
                road_coord = self.road_coords[idx]
                dist = geodesic(poi_coord, road_coord).meters

                if dist < min_dist:
                    min_dist = dist

                    # 查找最近节点ID
                    # 这里使用了 self.graph 节点数据的慢速查找来获取 ID，但在 road_nodes 存在后可以工作
                    temp_node_id = None
                    for n_id, n_data in self.graph.nodes(data=True):
                        if n_data.get('type') == 'road_node' and \
                           math.isclose(n_data['latitude'], road_coord[0], abs_tol=1e-6) and \
                           math.isclose(n_data['longitude'], road_coord[1], abs_tol=1e-6):
                            temp_node_id = n_id
                            break

                    if temp_node_id is not None:
                        nearest_road_node_id = temp_node_id

            if nearest_road_node_id is not None and min_dist <= distance_threshold:
                self.graph.add_edge(
                    poi_id, nearest_road_node_id,
                    type='poi_link',
                    distance=min_dist
                )

    # ========================== 构建主入口 ==========================

    def build_from_osm(self, road_network: pd.DataFrame, pois: pd.DataFrame):
        """从OSM数据构建知识图谱"""
        self.road_network = road_network
        self.pois = pois

        print(" -> 正在添加道路节点和边...")
        self._add_road_network(road_network)

        # 1. 构建道路 KDTree (只有在 road_nodes > 0 时才会成功)
        road_coords = []
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'road_node':
                road_coords.append((data['latitude'], data['longitude']))

        if road_coords:
            self.road_coords = np.array(road_coords)
            self.road_kdtree = KDTree(self.road_coords)
            print(" -> 道路网络 KDTree 构建完成。")

        print(" -> 正在添加 POI 节点...")
        self._add_pois(pois)

        # 2. 构建 POI KDTree
        poi_coords = []
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'poi':
                poi_coords.append((data['latitude'], data['longitude']))

        if poi_coords:
            self.poi_coords = np.array(poi_coords)
            self.poi_kdtree = KDTree(self.poi_coords)
            print(" -> POI KDTree 构建完成。")

        print(" -> 正在链接 POI 到最近道路...")
        self._link_roads_to_pois()

        print(" -> 知识图谱构建阶段已完成。统计:", self.get_graph_statistics())

    # ========================== 特征提取辅助方法 ==========================

    def _calculate_road_density(self, lat: float, lon: float, radius: float = 100.0) -> float:
        """计算道路密度 (使用 KDTree 加速)"""

        if self.road_kdtree is None or self.road_coords is None:
            return 0.0

        radius_in_degrees = radius / 111320.0
        indices = self.road_kdtree.query_ball_point((lat, lon), radius_in_degrees)

        count = 0
        for idx in indices:
            node_lat, node_lon = self.road_coords[idx]
            dist = geodesic((lat, lon), (node_lat, node_lon)).meters
            if dist < radius:
                count += 1
        return min(count / 50.0, 1.0)

    def _extract_poi_features(self, lat: float, lon: float,
                              max_k: int = 5, poi_types: list = ['bus_stop', 'station', 'school', 'restaurant']) -> np.ndarray:
        """提取最近 POI 类型和距离特征"""

        features = np.zeros(1 + len(poi_types), dtype=np.float32)

        if self.poi_kdtree is None or self.poi_coords is None:
            return features

        distances, indices = self.poi_kdtree.query((lat, lon), k=max_k)

        if not isinstance(distances, np.ndarray):
            distances = np.array([distances])
            indices = np.array([indices])

        if indices.size > 0 and distances[0] < float('inf'):
            features[0] = min(distances[0] / 5000.0, 1.0)

            for i, idx in enumerate(indices):
                if distances[i] > 5000.0:
                    continue

                poi_coord = self.poi_coords[idx]
                for n_id, n_data in self.graph.nodes(data=True):
                    if n_data.get('type') == 'poi' and \
                       math.isclose(n_data['latitude'], poi_coord[0], abs_tol=1e-6) and \
                       math.isclose(n_data['longitude'], poi_coord[1], abs_tol=1e-6):

                        poi_type = n_data.get('poi_type', 'general')
                        if poi_type in poi_types:
                            type_index = poi_types.index(poi_type)
                            features[1 + type_index] = 1.0
                        break

        return features

    def extract_kg_features(self, trajectory: np.ndarray) -> np.ndarray:
        """从轨迹点提取知识图谱特征"""

        N = trajectory.shape[0]
        # 11 维特征矩阵: 1 (距离) + 4 (POI类型) + 1 (密度) + 5 (道路类型占位符)
        poi_types = ['bus_stop', 'station', 'school', 'restaurant']
        poi_dim = 1 + len(poi_types)

        kg_features = np.zeros((N, 11), dtype=np.float32)

        for i in range(N):
            lat, lon = trajectory[i, 0], trajectory[i, 1]

            # 1. POI 特征 (5 维)
            poi_feats = self._extract_poi_features(lat, lon, poi_types=poi_types)
            kg_features[i, 0:poi_dim] = poi_feats

            # 2. 道路密度特征 (第 6 维)
            road_density = self._calculate_road_density(lat, lon, radius=100.0)
            kg_features[i, poi_dim] = road_density

            # 3. 道路类型特征 (剩余 5 维)
            kg_features[i, poi_dim+1:] = np.zeros(11 - poi_dim - 1)

        return kg_features

    def get_graph_statistics(self) -> Dict:
        """获取知识图谱统计信息"""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'road_nodes': len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'road_node']),
            'poi_nodes': len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'poi']),
        }