"""
Exp4 数据适配器
从基础数据生成 Exp4 所需的格式（序列长度50，轨迹特征 + 增强KG特征 + 天气特征）
特别保留时间序列信息用于天气数据匹配
"""
import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm


class Exp4DataAdapter:
    """Exp4 数据适配器（需要保留时间信息）"""

    def __init__(self, target_length: int = 50):
        self.target_length = target_length

        # Exp4使用的标签
        self.valid_labels = {
            'Walk', 'Bike', 'Bus', 'Car & taxi',
            'Train', 'Subway', 'Airplane'
        }

    def process_segments(self, base_segments: List[dict]) -> List[Tuple[np.ndarray, pd.Series, str]]:
        """
        将基础数据转换为 Exp4 格式（含时间序列）

        返回:
            [(features, datetime_series, label), ...]
            features: (50, 9) numpy array
            datetime_series: (50,) pandas Series
            label: str
        """
        print(f"\n{'=' * 80}")
        print(f"Exp4 数据适配 (序列长度: {self.target_length}, 保留时间序列)")
        print(f"{'=' * 80}\n")

        processed = []

        for seg in tqdm(base_segments, desc="[Exp4适配]"):
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

            # 3. 提取时间序列
            datetime_series = seg['datetime_series']

            # 4. 同步长度规范化（特征 + 时间）
            features, datetime_series = self._normalize_length_with_time(
                features, datetime_series, self.target_length
            )

            processed.append((features, datetime_series, seg['label']))

        print(f"✅ Exp4适配完成: {len(processed)} 个样本")
        self._print_label_distribution(processed)

        return processed

    def _normalize_length_with_time(self, features: np.ndarray,
                                    datetime_series: pd.Series,
                                    target: int) -> Tuple[np.ndarray, pd.Series]:
        """序列长度规范化（同步特征和时间）"""
        L = len(features)

        if L > target:
            # 均匀采样
            indices = np.linspace(0, L - 1, target, dtype=int)
            return features[indices], datetime_series.iloc[indices].reset_index(drop=True)

        elif L < target:
            # 零填充特征
            padding = np.zeros((target - L, features.shape[1]))
            features_padded = np.vstack([features, padding])

            # 时间填充（使用最后一个时间）
            last_time = datetime_series.iloc[-1]
            time_padding = pd.Series([last_time] * (target - L))
            datetime_padded = pd.concat(
                [datetime_series.reset_index(drop=True), time_padding],
                ignore_index=True
            )

            return features_padded, datetime_padded

        else:
            return features, datetime_series.reset_index(drop=True)

    def _print_label_distribution(self, processed: List[Tuple]):
        """打印标签分布"""
        from collections import Counter
        labels = [label for _, _, label in processed]
        counts = Counter(labels)

        print("\n标签分布:")
        for label in sorted(counts.keys()):
            print(f"  {label:15s}: {counts[label]:6d}")