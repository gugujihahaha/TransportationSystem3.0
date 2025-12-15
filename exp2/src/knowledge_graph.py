"""
知识图谱构建模块
从OSM数据构建交通知识图谱 (已集成 KDTree 空间索引和多进程支持)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from geopy.distance import geodesic
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial import KDTree  # 核心：用于空间加速


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
        # NEW: 用于道路密度加速
        self.road_kdtree = None
        self.road_coords = None
        self.poi_kdtree = None # 假设您已经实现了 POI 的 KDTree
        self.poi_coords = None

    # ========================== KG 并行化支持 (NEW) ==========================
    def get_data_for_worker(self) -> Dict:
        """提取可序列化数据供多进程 worker 使用"""
        # 仅传递必要的数据和可序列化的表示
        return {
            'road_type_mapping': self.road_type_mapping,
            'graph_data': nx.node_link_data(self.graph),
            'road_coords': self.road_coords.tolist() if self.road_coords is not None else None,
            'poi_coords': self.poi_coords.tolist() if self.poi_coords is not None else None,
            'poi_list': [d for n, d in self.graph.nodes(data=True) if d.get('type') == 'poi']
        }

    def load_data_for_worker(self, data: Dict):
        """worker 进程从序列化数据中重建 KG 实例"""
        self.road_type_mapping = data['road_type_mapping']
        self.graph = nx.node_link_graph(data['graph_data'])

        # 重建道路 KDTree
        if data['road_coords'] is not None:
            self.road_coords = np.array(data['road_coords'])
            self.road_kdtree = KDTree(self.road_coords)

        # 重建 POI KDTree
        if data['poi_coords'] is not None:
            self.poi_coords = np.array(data['poi_coords'])
            self.poi_kdtree = KDTree(self.poi_coords)

        # 重新填充 POI 字典以便于查询
        self.pois_lookup = {
            (d['latitude'], d['longitude']): d for d in data['poi_list']
        }

    # =========================================================================

    # 以下辅助方法需要您替换成您的原始实现，以确保图谱能正确构建

    def _add_road_network(self, road_network: pd.DataFrame):
        """添加道路节点和边 (请用您的原始代码替换此行)"""
        # Placeholder: Assume this method populates self.graph with road nodes and edges.
        pass

    def _add_pois(self, pois: pd.DataFrame):
        """添加 POI 节点 (请用您的原始代码替换此行)"""
        # Placeholder: Assume this method populates self.graph with POI nodes.
        pass

    def _link_roads_to_pois(self):
        """链接 POI 和道路 (请用您的原始代码替换此行)"""
        # Placeholder: Assume this method adds edges between POI and nearest roads.
        pass

    # =======================================================================

    def build_from_osm(self, road_network: pd.DataFrame, pois: pd.DataFrame):
        """从OSM数据构建知识图谱"""
        self.road_network = road_network
        self.pois = pois

        # 1. 添加道路节点和边
        self._add_road_network(road_network)

        # NEW: 2.1 构建道路 KDTree 以加速道路密度计算
        road_coords = []
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'road_node':
                road_coords.append((data['latitude'], data['longitude']))

        if road_coords:
            self.road_coords = np.array(road_coords)
            self.road_kdtree = KDTree(self.road_coords)
            print(" -> 道路网络 KDTree 构建完成。")

        # 3. 添加 POI
        self._add_pois(pois)

        # NEW: 3.1 构建 POI KDTree (假设您已经实现了这个，用于 _extract_poi_features 加速)
        poi_coords = []
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'poi':
                poi_coords.append((data['latitude'], data['longitude']))

        if poi_coords:
            self.poi_coords = np.array(poi_coords)
            self.poi_kdtree = KDTree(self.poi_coords)
            print(" -> POI KDTree 构建完成。")


        # 4. 链接 POI 和道路
        self._link_roads_to_pois()

    def extract_kg_features(self, trajectory: np.ndarray) -> np.ndarray:
        """从轨迹点提取知识图谱特征 (请用您的原始代码替换此行)"""
        # Placeholder: Assume this method extracts all 11 KG features for the whole trajectory.
        # It should call _extract_poi_features, _calculate_road_density, etc.

        # 必须确保此方法调用了新的 _calculate_road_density
        N = trajectory.shape[0]
        # (N, 11) 的特征矩阵
        kg_features = np.zeros((N, 11), dtype=np.float32)

        for i in range(N):
            lat, lon = trajectory[i, 0], trajectory[i, 1]
            # 假设您的原始逻辑在这里调用了各种辅助函数来填充 11 维特征

            # 道路密度特征 (假设是第 10 维)
            kg_features[i, 9] = self._calculate_road_density(lat, lon)

            # POI 特征 (假设是第 1 到 4 维)
            # kg_features[i, 0:4] = self._extract_poi_features(lat, lon)

            # 其他特征...

        return kg_features

    def get_graph_statistics(self) -> Dict:
        """获取知识图谱统计信息 (请用您的原始代码替换此行)"""
        # Placeholder: Assume this returns graph statistics.
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'road_nodes': len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'road_node']),
            'poi_nodes': len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'poi']),
        }

    # ========================== 道路密度计算 (KDTree 优化) ==========================
    def _calculate_road_density(self, lat: float, lon: float,
                               radius: float = 100.0) -> float:
        """计算道路密度 (使用 KDTree 加速，将 O(M) 优化为 O(log M))"""

        if self.road_kdtree is None or self.road_coords is None:
            # 如果 KDTree 未构建，退回到 0.0 或原始线性扫描（不推荐）
            return 0.0

        # 1. 将米转换为近似的度数，用于 KDTree 初筛
        # 1度约等于111,320米 (在赤道附近)
        radius_in_degrees = radius / 111320.0

        # 2. 使用 KDTree 快速初筛：找到近似半径内的所有道路节点索引
        # 复杂度 O(log M)
        indices = self.road_kdtree.query_ball_point(
            (lat, lon),
            radius_in_degrees
        )

        # 3. 对初筛结果进行精确的大地测量距离检查
        count = 0
        for idx in indices:
            node_lat, node_lon = self.road_coords[idx]

            # 对少数初筛点进行 Geodesic 距离计算，保证精度
            dist = geodesic((lat, lon), (node_lat, node_lon)).meters
            if dist < radius:
                count += 1

        # 归一化（假设最大密度为50）
        return min(count / 50.0, 1.0)
    # =================================================================================