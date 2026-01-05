"""
Exp3 数据适配器
从基础数据生成 Exp3 所需的格式（序列长度50，轨迹特征 + 增强KG特征）
"""
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


class Exp3DataAdapter:
    """Exp3 数据适配器（与Exp2相同的数据格式，但KG特征维度不同）"""

    def __init__(self, target_length: int = 50):
        self.target_length = target_length

        # Exp3使用的标签（与Exp2相同）
        self.valid_labels = {
            'Walk', 'Bike', 'Bus', 'Car & taxi',
            'Train', 'Subway', 'Airplane'
        }

    def process_segments(self, base_segments: List[dict]) -> List[Tuple[np.ndarray, str]]:
        """
        将基础数据转换为 Exp3 格式

        返回:
            [(features, label), ...]
            features: (50, 9) numpy array
            label: str
        """
        print(f"\n{'=' * 80}")
        print(f"Exp3 数据适配 (序列长度: {self.target_length})")
        print(f"{'=' * 80}\n")

        processed = []

        for seg in tqdm(base_segments, desc="[Exp3适配]"):
            # 1. 标签过滤
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

            # 3. 长度规范化到50
            features = self._normalize_length(features, self.target_length)

            processed.append((features, seg['label']))

        print(f"✅ Exp3适配完成: {len(processed)} 个样本")
        self._print_label_distribution(processed)

        return processed

    def _normalize_length(self, features: np.ndarray, target: int) -> np.ndarray:
        """序列长度规范化"""
        L = len(features)

        if L > target:
            # 均匀采样
            indices = np.linspace(0, L - 1, target, dtype=int)
            return features[indices]
        elif L < target:
            # 零填充
            padding = np.zeros((target - L, features.shape[1]))
            return np.vstack([features, padding])
        else:
            return features

    def _print_label_distribution(self, processed: List[Tuple]):
        """打印标签分布"""
        from collections import Counter
        labels = [label for _, label in processed]
        counts = Counter(labels)

        print("\n标签分布:")
        for label in sorted(counts.keys()):
            print(f"  {label:15s}: {counts[label]:6d}")