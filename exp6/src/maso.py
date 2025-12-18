"""
MASO (Multi-Attribute-Scale-Object) 轨迹特征组织结构实现
基于论文：基于GPS轨迹多尺度表达的交通出行方式识别方法

核心思想：
1. 多对象 (Multi-Object): 从一条轨迹段中裁剪提取K个子段
2. 多尺度 (Multi-Scale): 每个子段在N×M个空间尺度下投影
3. 多属性 (Multi-Attribute): 每个像素包含L类运动属性（通道）

最终输出: (K, L, Wb, Hb) - K个对象, L个属性通道, Wb×Hb图像尺寸
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass
import warnings


@dataclass
class MASOConfig:
    """MASO配置参数"""
    # 多对象参数
    K: int = 6  # 裁剪子段数量
    parts: int = 20  # 轨迹分割份数（用于裁剪间隔）

    # 多尺度参数
    spatial_ranges: List[float] = None  # 裁剪空间范围 (度)
    image_sizes: List[int] = None  # 投影图像尺寸

    # 多属性参数
    L: int = 9  # 属性维度 (9维)

    def __post_init__(self):
        # 默认配置
        if self.spatial_ranges is None:
            self.spatial_ranges = [0.01, 0.05, 0.2]  # 3个空间范围
        if self.image_sizes is None:
            self.image_sizes = [32]  # 1个图像尺寸

    @property
    def N(self) -> int:
        """裁剪空间范围数量"""
        return len(self.spatial_ranges)

    @property
    def M(self) -> int:
        """投影图像尺寸数量"""
        return len(self.image_sizes)


class TrajectorySubSegmentCropper:
    """轨迹子段裁剪与选取 (第2章)"""

    def __init__(self, config: MASOConfig):
        self.config = config

    def sliding_crop(self,
                     segment: pd.DataFrame,
                     spatial_range: float) -> List[pd.DataFrame]:
        """
        轨迹子段滑动裁剪策略 (2.1节)

        Args:
            segment: 轨迹段 DataFrame (包含 latitude, longitude 等列)
            spatial_range: 裁剪空间范围 (度)

        Returns:
            裁剪后的轨迹子段列表
        """
        if len(segment) < 2:
            return [segment]

        # 计算裁剪间隔
        length = len(segment)
        interval = max(1, length // self.config.parts)

        sub_segments = []

        # 滑动裁剪
        for start_idx in range(0, length, interval):
            if start_idx >= length:
                break

            # 获取裁剪中心点
            center_lat = segment.iloc[start_idx]['latitude']
            center_lon = segment.iloc[start_idx]['longitude']

            # 计算裁剪范围
            lat_min = center_lat - spatial_range / 2
            lat_max = center_lat + spatial_range / 2
            lon_min = center_lon - spatial_range / 2
            lon_max = center_lon + spatial_range / 2

            # 裁剪数据
            mask = ((segment['latitude'] >= lat_min) & (segment['latitude'] <= lat_max) &
                    (segment['longitude'] >= lon_min) & (segment['longitude'] <= lon_max))

            sub_seg = segment[mask].copy()

            if len(sub_seg) > 0:
                sub_segments.append(sub_seg)

        return sub_segments if sub_segments else [segment]

    def compute_indicators(self,
                           sub_segments: List[pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        计算多指标综合评价指标 (2.2节)

        Args:
            sub_segments: 裁剪子段列表

        Returns:
            {
                'pts_num': GPS点数量,
                'pts_range': GPS点分布范围,
                'pts_overlap': 重叠GPS点数量
            }
        """
        n_segs = len(sub_segments)

        pts_num = np.array([len(seg) for seg in sub_segments])
        pts_range = np.array([
            (seg['longitude'].max() - seg['longitude'].min()) +
            (seg['latitude'].max() - seg['latitude'].min())
            for seg in sub_segments
        ])

        # 计算重叠GPS点数量
        pts_overlap = np.zeros(n_segs)
        for i, seg_i in enumerate(sub_segments):
            indices_i = set(range(len(seg_i)))
            for j, seg_j in enumerate(sub_segments):
                if i != j:
                    indices_j = set(range(len(seg_j)))
                    pts_overlap[i] += len(indices_i & indices_j)

        return {
            'pts_num': pts_num,
            'pts_range': pts_range,
            'pts_overlap': pts_overlap
        }

    def normalize_indicators(self, indicators: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        指标同向化与无量纲化 (2.2.2节 - 极值标准化)
        """
        normalized = {}

        # 正向指标: pts_num, pts_range (公式2-3)
        for key in ['pts_num', 'pts_range']:
            x = indicators[key]
            x_min, x_max = x.min(), x.max()
            if x_max - x_min == 0:
                normalized[key] = np.ones_like(x)
            else:
                normalized[key] = (x - x_min) / (x_max - x_min)

        # 负向指标: pts_overlap (公式2-4)
        x = indicators['pts_overlap']
        # 非负平移处理 (避免全0)
        if x.max() == 0:
            x = x + 0.01
        x_min, x_max = x.min(), x.max()
        if x_max - x_min == 0:
            normalized['pts_overlap'] = np.ones_like(x)
        else:
            normalized['pts_overlap'] = (x_max - x) / (x_max - x_min)

        return normalized

    def calculate_weights_ahp(self) -> np.ndarray:
        """
        层次分析法 (AHP) 权重计算 (2.2.3节)

        返回: [w_pts_num, w_pts_range, w_pts_overlap]
        """
        # 论文中的判断矩阵 (表2-3)
        judgment_matrix = np.array([
            [1, 6, 1],
            [0.167, 1, 0.167],
            [1, 6, 1]
        ])

        # 方根法计算特征向量
        n = judgment_matrix.shape[0]
        row_products = np.prod(judgment_matrix, axis=1) ** (1 / n)
        weights = row_products / row_products.sum()

        return weights

    def calculate_weights_entropy(self, normalized: Dict[str, np.ndarray]) -> np.ndarray:
        """
        熵值法权重计算 (2.2.3节)

        返回: [w_pts_num, w_pts_range, w_pts_overlap]
        """
        k = 1.0 / np.log(len(normalized['pts_num']))  # 熵计算系数
        weights = []

        for key in ['pts_num', 'pts_range', 'pts_overlap']:
            x = normalized[key]
            # 避免 log(0)
            x = np.clip(x, 1e-10, 1)

            # 计算熵
            entropy = -k * np.sum(x * np.log(x + 1e-10))

            # 信息熵冗余度
            d = 1 - entropy
            weights.append(d)

        # 归一化
        weights = np.array(weights)
        weights = weights / weights.sum()

        return weights

    def calculate_combined_weights(self,
                                   normalized: Dict[str, np.ndarray]) -> np.ndarray:
        """
        组合赋权法 (2.2.3节 - 公式2-12)
        """
        w_ahp = self.calculate_weights_ahp()
        w_entropy = self.calculate_weights_entropy(normalized)

        # 简单线性加权 (λ1 = λ2 = 0.5)
        combined_weights = 0.5 * w_ahp + 0.5 * w_entropy

        return combined_weights

    def calculate_comprehensive_score(self,
                                      sub_segments: List[pd.DataFrame]) -> np.ndarray:
        """
        计算综合评价值 (2.2.4节 - 公式2-17)
        """
        indicators = self.compute_indicators(sub_segments)
        normalized = self.normalize_indicators(indicators)
        weights = self.calculate_combined_weights(normalized)

        # 综合得分 = sum(w_j * x_ij)
        scores = np.zeros(len(sub_segments))
        for i, key in enumerate(['pts_num', 'pts_range', 'pts_overlap']):
            scores += weights[i] * normalized[key]

        return scores

    def select_optimal_segments(self,
                                segment: pd.DataFrame,
                                spatial_range: float) -> List[pd.DataFrame]:
        """
        最优轨迹子段分段选取 (2.3节)

        返回K个最优子段
        """
        # 1. 滑动裁剪
        sub_segments = self.sliding_crop(segment, spatial_range)

        if len(sub_segments) <= self.config.K:
            return sub_segments

        # 2. 计算综合评价值
        scores = self.calculate_comprehensive_score(sub_segments)

        # 3. 分段选取 (将轨迹分为K份，每份选1个最优)
        length = len(sub_segments)
        segment_size = length / self.config.K

        optimal_indices = []
        for k in range(self.config.K):
            start_idx = int(k * segment_size)
            end_idx = int((k + 1) * segment_size)

            # 每个部分中选得分最高的
            part_indices = np.arange(start_idx, min(end_idx, length))
            if len(part_indices) > 0:
                best_idx = part_indices[np.argmax(scores[part_indices])]
                optimal_indices.append(best_idx)

        return [sub_segments[idx] for idx in optimal_indices]


class MASOProjection:
    """MASO投影变换与属性计算 (第3章)"""

    def __init__(self, config: MASOConfig):
        self.config = config

    def project_to_image(self,
                         segment: pd.DataFrame,
                         spatial_range: float,
                         image_size: int) -> np.ndarray:
        """
        投影变换: 地理坐标 -> 图像坐标 (3.2.1节)

        Args:
            segment: 轨迹子段
            spatial_range: 裁剪空间范围
            image_size: 输出图像尺寸

        Returns:
            img_data: (image_size, image_size) 图像，每个像素包含落在其中的GPS点列表
        """
        if len(segment) == 0:
            return np.empty((image_size, image_size), dtype=object)

        # 初始化图像
        img_data = np.empty((image_size, image_size), dtype=object)
        for i in range(image_size):
            for j in range(image_size):
                img_data[i, j] = []

        # 找到中心点
        center_idx = len(segment) // 2
        center_lat = segment.iloc[center_idx]['latitude']
        center_lon = segment.iloc[center_idx]['longitude']

        # 计算最小值 (用于坐标变换)
        min_lat = segment['latitude'].min()
        min_lon = segment['longitude'].min()

        # 投影每个GPS点
        for idx, row in segment.iterrows():
            lat, lon = row['latitude'], row['longitude']

            # 地理坐标 -> 图像坐标 (线性变换)
            x = (lon - min_lon) * image_size / spatial_range
            y = (lat - min_lat) * image_size / spatial_range

            # 计算中心对齐偏移
            center_x = (center_lon - min_lon) * image_size / spatial_range
            center_y = (center_lat - min_lat) * image_size / spatial_range

            offset_x = image_size // 2 - int(center_x)
            offset_y = image_size // 2 - int(center_y)

            # 应用偏移
            img_x = int(x) + offset_x
            img_y = int(y) + offset_y

            # 检查边界
            if 0 <= img_x < image_size and 0 <= img_y < image_size:
                img_data[img_y, img_x].append(row)

        return img_data

    def compute_pixel_attributes(self,
                                 pixel_points: List,
                                 segment: pd.DataFrame) -> np.ndarray:
        """
        计算像素的运动属性 (3.2.2节)

        输入像素内的GPS点，计算9维属性的统计值

        Returns: (9,) 数组，包含各属性的统计值
        """
        if len(pixel_points) == 0:
            return np.zeros(9)

        attributes = []

        for attr_idx, attr_name in enumerate([
            'latitude', 'longitude', 'speed', 'acceleration',
            'bearing_change', 'distance', 'time_diff',
            'total_distance', 'total_time'
        ]):
            if attr_name in ['latitude', 'longitude']:
                # 位置属性取平均
                values = [p[attr_name] for p in pixel_points if attr_name in p]
                attr_val = np.mean(values) if values else 0
            else:
                # 运动属性取最大值
                values = [p[attr_name] for p in pixel_points if attr_name in p]
                attr_val = np.max(values) if values else 0

            attributes.append(attr_val)

        return np.array(attributes)

    def create_multi_channel_image(self,
                                   img_data: np.ndarray,
                                   segment: pd.DataFrame,
                                   spatial_range: float,
                                   image_size: int) -> np.ndarray:
        """
        创建多通道图像 (3.2.2-3.2.3节)

        Returns: (L, image_size, image_size) 多通道图像
        """
        L = self.config.L  # 9个属性通道
        img_channels = np.zeros((L, image_size, image_size))

        # 对每个像素计算属性
        for i in range(image_size):
            for j in range(image_size):
                pixel_points = img_data[i, j]

                if len(pixel_points) > 0:
                    # 计算该像素的运动属性
                    attrs = self.compute_pixel_attributes(pixel_points, segment)

                    # 填充通道
                    for c in range(L):
                        img_channels[c, i, j] = attrs[c]

        return img_channels


class MASOFeatureOrganizer:
    """MASO特征组织结构 - 整合上述模块"""

    def __init__(self, config: MASOConfig = None):
        self.config = config if config else MASOConfig()
        self.cropper = TrajectorySubSegmentCropper(self.config)
        self.projector = MASOProjection(self.config)

    def extract_maso_features(self, segment: pd.DataFrame) -> np.ndarray:
        """
        完整MASO特征提取流程

        Args:
            segment: 原始轨迹段 (包含轨迹特征列)

        Returns:
            maso_features: (K, L, N*M, Wb, Hb)
            或简化为: (K*N*M, L, Wb, Hb) 用于训练
        """
        if len(segment) < 10:
            warnings.warn(f"轨迹段过短 ({len(segment)}), 返回零填充")
            return np.zeros((
                self.config.K,
                self.config.L,
                self.config.image_sizes[0],
                self.config.image_sizes[0]
            ))

        # 1. 多对象: 裁剪得到K个子段
        # 使用第一个空间范围进行裁剪
        spatial_range = self.config.spatial_ranges[0]
        optimal_sub_segments = self.cropper.select_optimal_segments(segment, spatial_range)

        K = min(len(optimal_sub_segments), self.config.K)

        # 2. 多尺度 & 多属性: 对每个子段在所有尺度上投影并提取属性
        maso_output = []

        for k, sub_seg in enumerate(optimal_sub_segments[:self.config.K]):
            scale_outputs = []

            for n, sp_range in enumerate(self.config.spatial_ranges):
                for m, img_size in enumerate(self.config.image_sizes):
                    # 投影变换
                    img_data = self.projector.project_to_image(
                        sub_seg, sp_range, img_size
                    )

                    # 计算多通道属性图像
                    multi_ch_img = self.projector.create_multi_channel_image(
                        img_data, sub_seg, sp_range, img_size
                    )

                    scale_outputs.append(multi_ch_img)

            # 在多尺度维度拼接 (K, L, N*M*Wb, Hb)
            # 或简化为 (K*N*M, L, Wb, Hb)
            scale_concat = np.concatenate(scale_outputs, axis=0)  # (L*N*M, Wb, Hb)
            maso_output.append(scale_concat)

        # 如果子段少于K个，进行零填充
        while len(maso_output) < self.config.K:
            maso_output.append(np.zeros_like(maso_output[0]))

        # 返回形状: (K, L*N*M, Wb, Hb)
        return np.stack(maso_output[:self.config.K], axis=0)

    def batch_extract(self, segments: List[pd.DataFrame]) -> np.ndarray:
        """
        批量提取MASO特征

        Args:
            segments: 轨迹段列表

        Returns:
            features: (batch_size, K, L*N*M, Wb, Hb)
        """
        features = []
        for seg in segments:
            feat = self.extract_maso_features(seg)
            features.append(feat)

        return np.array(features)