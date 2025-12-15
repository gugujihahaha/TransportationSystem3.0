"""
增强知识图谱构建模块 (Exp3)
从完整OSM数据构建交通知识图谱
- 继承 Exp2 的 KDTree 优化
- 新增特征：地铁入口、共享单车、出租车点、速度限制、公交/地铁线路
- KG特征从 11 维扩展到 15 维
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from geopy.distance import geodesic
import networkx as nx
from tqdm import tqdm
from scipy.spatial import KDTree
import re


class EnhancedTransportationKG:
    """增强版交通知识图谱 (Exp3)"""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.road_network = None
        self.pois = None
        self.transit_routes = None

        # 道路类型映射
        self.road_type_mapping = {
            'footway': 'walk',
            'cycleway': 'bike',
            'primary': 'car',
            'secondary': 'car',
            'tertiary': 'car',
            'residential': 'car',
            'motorway': 'car',
            'trunk': 'car',
            'bus_stop': 'bus',
            'station': 'train',
            'subway_entrance': 'train',
            'parking': 'car'
        }

        # 新增数据结构
        self.speed_limits = {}  # 道路速度限制
        self.bus_routes = {}  # 公交线路
        self.subway_routes = {}  # 地铁线路
        self.road_to_routes = {}  # 道路到线路的映射

    def build_from_osm(self, road_network: pd.DataFrame, pois: pd.DataFrame,
                       transit_routes: pd.DataFrame = None):
        """从OSM数据构建增强知识图谱"""
        self.road_network = road_network
        self.pois = pois
        self.transit_routes = transit_routes

        print("\n" + "=" * 60)
        print("构建增强知识图谱 (Exp3)")
        print("=" * 60)

        # 1. 添加道路网络
        print(" -> 正在添加道路节点和路段...")
        self._add_road_network()
        print(f" -> 道路网络添加完成。当前图节点数: {self.graph.number_of_nodes()}")

        # 2. 提取速度限制信息
        print(" -> 正在提取速度限制信息...")
        self._extract_speed_limits()
        print(f" -> 提取到 {len(self.speed_limits)} 条速度限制记录")

        # 3. 添加POI节点
        print(" -> 正在添加 POI 节点...")
        self._add_pois()
        print(f" -> POI 节点添加完成。当前图节点数: {self.graph.number_of_nodes()}")

        # 4. 添加公交/地铁线路 (新增)
        if transit_routes is not None and len(transit_routes) > 0:
            print(" -> 正在添加公交/地铁线路...")
            self._add_transit_routes()
            print(f" -> 公交线路: {len(self.bus_routes)}, 地铁线路: {len(self.subway_routes)}")

        # 5. 链接道路和POI (使用 KDTree 优化)
        self._link_roads_to_pois()

        print("=" * 60)
        print("知识图谱构建完成")
        print("=" * 60)

    def _add_road_network(self):
        """添加道路网络到知识图谱"""
        for _, road in self.road_network.iterrows():
            road_id = road['id']
            road_type = road.get('highway') or road.get('railway', '')
            coordinates = road.get('coordinates', [])
            maxspeed = road.get('maxspeed', None)

            if not coordinates:
                continue

            # 处理不同几何类型
            if road['geometry_type'] == 'LineString':
                coords = coordinates
            elif road['geometry_type'] == 'Polygon':
                coords = coordinates[0]
            else:
                continue

            # 为每个坐标点创建节点
            for i, coord in enumerate(coords):
                node_id = f"{road_id}_node_{i}"
                lon, lat = coord[0], coord[1]

                self.graph.add_node(
                    node_id,
                    type='road_node',
                    road_id=road_id,
                    road_type=road_type,
                    latitude=lat,
                    longitude=lon,
                    maxspeed=maxspeed
                )

                # 连接相邻节点
                if i > 0:
                    prev_node_id = f"{road_id}_node_{i - 1}"
                    distance = geodesic(
                        (coords[i - 1][1], coords[i - 1][0]),
                        (lat, lon)
                    ).meters

                    self.graph.add_edge(
                        prev_node_id, node_id,
                        type='road_segment',
                        road_type=road_type,
                        distance=distance,
                        maxspeed=maxspeed
                    )

    def _extract_speed_limits(self):
        """提取速度限制信息"""
        for _, road in self.road_network.iterrows():
            road_id = road['id']
            maxspeed = road.get('maxspeed', None)

            if maxspeed:
                # 解析速度字符串
                speed_value = self._parse_speed(maxspeed)
                if speed_value is not None:
                    self.speed_limits[road_id] = speed_value

    def _parse_speed(self, speed_str) -> Optional[float]:
        """解析速度字符串 (例如: "50 km/h", "40", "60 mph")"""
        if not speed_str:
            return None

        try:
            # 处理字符串
            speed_str = str(speed_str).strip().lower()

            # 提取数字
            numbers = re.findall(r'\d+', speed_str)
            if not numbers:
                return None

            speed = float(numbers[0])

            # 处理单位转换
            if 'mph' in speed_str:
                speed = speed * 1.60934  # 转换为 km/h

            return speed

        except (ValueError, AttributeError):
            return None

    def _add_pois(self):
        """添加POI到知识图谱"""
        for _, poi in self.pois.iterrows():
            poi_id = poi['id']
            poi_type = poi['type']
            coordinates = poi.get('coordinates', [])

            if not coordinates:
                continue

            # 处理坐标
            if isinstance(coordinates[0], list):
                lon, lat = coordinates[0][0], coordinates[0][1]
            else:
                lon, lat = coordinates[0], coordinates[1]

            self.graph.add_node(
                poi_id,
                type='poi',
                poi_type=poi_type,
                name=poi.get('name', ''),
                latitude=lat,
                longitude=lon
            )

    def _add_transit_routes(self):
        """添加公交和地铁线路信息 (新增)"""
        if self.transit_routes is None:
            return

        for _, route in self.transit_routes.iterrows():
            route_type = route.get('route')
            route_id = route.get('id')
            route_ref = route.get('ref', '')
            route_name = route.get('name', '')
            members = route.get('members', [])

            route_info = {
                'id': route_id,
                'ref': route_ref,
                'name': route_name,
                'members': members
            }

            if route_type == 'bus':
                self.bus_routes[route_id] = route_info
            elif route_type == 'subway':
                self.subway_routes[route_id] = route_info

            # 建立道路到线路的映射
            for member in members:
                if isinstance(member, dict):
                    member_id = member.get('ref')
                    if member_id:
                        if member_id not in self.road_to_routes:
                            self.road_to_routes[member_id] = []
                        self.road_to_routes[member_id].append(route_id)

    def _link_roads_to_pois(self, max_distance: float = 100.0):
        """使用 KDTree 加速 POI 到最近道路节点的链接"""
        poi_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'poi']
        road_nodes_data = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('type') == 'road_node']

        if not road_nodes_data:
            print("警告: 无道路节点，跳过关联。")
            return

        print(f" -> 正在关联 {len(poi_nodes)} 个POI到 {len(road_nodes_data)} 个道路节点...")

        # 1. 准备道路节点坐标和ID
        road_coords = np.array([[d['latitude'], d['longitude']] for n, d in road_nodes_data])
        road_node_ids = [n for n, d in road_nodes_data]

        # 2. 构建 KDTree
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

        # 4. 批量查找近邻
        max_degree_distance = max_distance / 111300.0
        indices = road_tree.query_ball_point(poi_coords, r=max_degree_distance)

        print(" -> 批量查找近邻完成，正在添加边...")

        # 5. 添加边
        link_count = 0
        for i, neighbors_indices in enumerate(tqdm(indices, desc="   [KG关联进度]")):
            poi_node = poi_ids_list[i]
            poi_lat, poi_lon = poi_coords[i]

            min_dist = float('inf')
            nearest_road = None

            for j in neighbors_indices:
                road_node = road_node_ids[j]
                road_lat, road_lon = road_coords[j]

                dist = geodesic((poi_lat, poi_lon), (road_lat, road_lon)).meters

                if dist < min_dist:
                    min_dist = dist
                    nearest_road = road_node

            if nearest_road is not None:
                self.graph.add_edge(
                    poi_node, nearest_road,
                    type='nearby_road',
                    distance=min_dist
                )
                self.graph.add_edge(
                    nearest_road, poi_node,
                    type='has_poi',
                    distance=min_dist
                )
                link_count += 1

        print(f" -> 知识图谱道路-POI关联完成。共添加 {link_count} 条关联边。")

    def extract_kg_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        提取 15 维增强 KG 特征

        Input: (N, 9) 轨迹特征 [lat, lon, speed, accel, bearing_change, dist, time_diff, total_dist, total_time]
        Output: (N, 15) KG 特征

        特征组成:
        - 道路类型 (6维 one-hot): walk, bike, car, bus, train, unknown
        - 附近POI (6维): 公交站, 地铁入口, 停车场, 共享单车, 出租车, 最近POI距离
        - 道路属性 (2维): 速度限制(归一化), 是否在公交/地铁线路上
        - 道路密度 (1维): 附近道路节点数量(归一化)
        """
        features = []

        for i in tqdm(range(trajectory.shape[0]), desc="   [Enhanced KG 特征提取]"):
            lat = trajectory[i, 0]
            lon = trajectory[i, 1]

            # 1. 道路类型 (6维)
            road_type = self.get_road_type_for_location(lat, lon)
            road_type_feat = self._encode_road_type(road_type)

            # 2. 附近POI (6维) - 增强版
            nearby_pois = self.get_nearby_pois(lat, lon)
            poi_feat = self._encode_pois_enhanced(nearby_pois)

            # 3. 道路属性 (2维) - 新增
            speed_limit = self._get_speed_limit_at_location(lat, lon)
            on_transit = self._is_on_transit_route(lat, lon)
            road_attr_feat = np.array([speed_limit, on_transit], dtype=np.float32)

            # 4. 道路密度 (1维)
            road_density = self._calculate_road_density(lat, lon)

            # 合并特征 (6 + 6 + 2 + 1 = 15维)
            point_features = np.concatenate([
                road_type_feat,  # 6维
                poi_feat,  # 6维
                road_attr_feat,  # 2维
                [road_density]  # 1维
            ])

            features.append(point_features)

        return np.array(features, dtype=np.float32)

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

    def _encode_road_type(self, road_type: Optional[str]) -> np.ndarray:
        """编码道路类型为 6 维 one-hot 向量"""
        types = ['walk', 'bike', 'car', 'bus', 'train', 'unknown']
        encoding = np.zeros(len(types), dtype=np.float32)

        if road_type and road_type in types:
            idx = types.index(road_type)
            encoding[idx] = 1.0
        else:
            encoding[types.index('unknown')] = 1.0

        return encoding

    def _encode_pois_enhanced(self, pois: List[Dict]) -> np.ndarray:
        """
        编码 POI 为 6 维特征 (增强版)
        [公交站, 地铁入口, 停车场, 共享单车, 出租车点, 最近POI距离]
        """
        features = np.zeros(6, dtype=np.float32)

        if pois:
            poi_types = [poi['type'] for poi in pois]

            # 公交站
            if 'bus_stop' in poi_types:
                features[0] = 1.0

            # 地铁入口 (新增)
            if 'subway_entrance' in poi_types or 'station' in poi_types:
                features[1] = 1.0

            # 停车场
            if 'parking' in poi_types:
                features[2] = 1.0

            # 共享单车点 (新增)
            if 'bicycle_rental' in poi_types:
                features[3] = 1.0

            # 出租车停靠点 (新增)
            if 'taxi' in poi_types:
                features[4] = 1.0

            # 最近POI距离 (归一化到 0-1)
            features[5] = min(pois[0]['distance'] / 200.0, 1.0)

        return features

    def _get_speed_limit_at_location(self, lat: float, lon: float,
                                     max_distance: float = 50.0) -> float:
        """
        获取位置的速度限制 (归一化到 0-1)
        """
        nearest_road_node = self._find_nearest_road_node(lat, lon, max_distance)

        if nearest_road_node:
            road_id = self.graph.nodes[nearest_road_node].get('road_id')
            speed = self.speed_limits.get(road_id, 60.0)  # 默认 60 km/h
            # 归一化 (假设最大速度 120 km/h)
            return min(speed / 120.0, 1.0)

        return 0.5  # 默认中等速度

    def _is_on_transit_route(self, lat: float, lon: float,
                             max_distance: float = 50.0) -> float:
        """
        判断是否在公交/地铁线路上 (0 或 1)
        """
        nearest_road_node = self._find_nearest_road_node(lat, lon, max_distance)

        if nearest_road_node:
            road_id = self.graph.nodes[nearest_road_node].get('road_id')

            # 检查该道路是否属于公交或地铁线路
            if road_id in self.road_to_routes:
                return 1.0

        return 0.0

    def _calculate_road_density(self, lat: float, lon: float,
                                radius: float = 100.0) -> float:
        """计算道路密度 (归一化到 0-1)"""
        count = 0

        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'road_node':
                node_lat = data['latitude']
                node_lon = data['longitude']

                dist = geodesic((lat, lon), (node_lat, node_lon)).meters
                if dist < radius:
                    count += 1

        # 归一化 (假设最大密度为 50)
        return min(count / 50.0, 1.0)

    def _find_nearest_road_node(self, lat: float, lon: float,
                                max_distance: float = 50.0) -> Optional[str]:
        """查找最近的道路节点"""
        min_dist = float('inf')
        nearest_node = None

        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'road_node':
                node_lat = data['latitude']
                node_lon = data['longitude']

                dist = geodesic((lat, lon), (node_lat, node_lon)).meters

                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    nearest_node = node

        return nearest_node

    def get_graph_statistics(self) -> Dict:
        """获取知识图谱统计信息"""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'road_nodes': len([n for n, d in self.graph.nodes(data=True)
                               if d.get('type') == 'road_node']),
            'poi_nodes': len([n for n, d in self.graph.nodes(data=True)
                              if d.get('type') == 'poi']),
            'poi_links': len([u for u, v, k, d in self.graph.edges(data=True, keys=True)
                              if d.get('type') == 'nearby_road']),
            'speed_limits': len(self.speed_limits),
            'bus_routes': len(self.bus_routes),
            'subway_routes': len(self.subway_routes)
        }
        return stats