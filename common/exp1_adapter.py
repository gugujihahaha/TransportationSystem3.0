"""
Exp1 数据适配器
从基础数据生成 Exp1 所需的格式（序列长度100，仅轨迹特征）
"""
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


class Exp1DataAdapter:
    """Exp1 数据适配器"""

    def __init__(self, target_length: int = 100):
        self.target_length = target_length

        # 统一7类标签（任务定义统一）
        self.valid_labels = {
            'Walk', 'Bike', 'Bus', 'Car & taxi',
            'Train', 'Subway', 'Airplane'
        }

        # 标签映射（car & taxi合并）
        self.label_mapping = {
            'Car & taxi': 'Car & taxi',
            'Walk': 'Walk',
            'Bike': 'Bike',
            'Bus': 'Bus',
            'Train': 'Train',
            'Subway': 'Subway',
            'Airplane': 'Airplane'
        }

    def process_segments(self, base_segments: List[dict]) -> List[Tuple[np.ndarray, str]]:
        """
        将基础数据转换为 Exp1 格式

        返回:
            [(features, label), ...]
            features: (100, 9) numpy array
            label: str
        """
        print(f"\n{'=' * 80}")
        print(f"Exp1 数据适配 (序列长度: {self.target_length})")
        print(f"{'=' * 80}\n")

        processed = []

        for seg in tqdm(base_segments, desc="[Exp1适配]"):
            # 1. 标签过滤（只保留7类）
            if seg['label'] not in self.valid_labels:
                continue

            # 2. 提取9维特征
            points = seg['raw_points']
            feature_cols = [
                'latitude', 'longitude', 'speed', 'acceleration',
                'bearing_change', 'distance', 'time_diff',
                'total_distance', 'total_time'
            ]
            features = points[feature_cols].values

            # 3. 长度规范化到100
            features = self._normalize_length(features, self.target_length)

            # 4. 标签映射
            label = self.label_mapping.get(seg['label'], seg['label'])

            processed.append((features, label))

        print(f"✅ Exp1适配完成: {len(processed)} 个样本")

        return processed

    def _normalize_length(self, features: np.ndarray, target: int) -> np.ndarray:
        """序列长度规范化"""
        L = len(features)

        if L > 200:
            # 均匀采样
            indices = np.linspace(0, L - 1, target, dtype=int)
            return features[indices]
        elif L > target:
            # 随机裁剪
            start = np.random.randint(0, L - target + 1)
            return features[start:start + target]
        elif L < target:
            # 零填充
            padding = np.zeros((target - L, features.shape[1]))
            return np.vstack([features, padding])
        else:
            return features