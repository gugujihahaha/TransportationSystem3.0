import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from .exp4_adapter import Exp4DataAdapter
from exp5.src.trajectory_cleaner import TrajectoryCleaner


class Exp5DataAdapter(Exp4DataAdapter):
    """Exp5 数据适配器 - 基于Exp4，添加第二阶段数据清洗"""

    def __init__(self, target_length: int = 50, enable_cleaning: bool = True):
        """
        初始化Exp5数据适配器

        Args:
            target_length: 目标序列长度
            enable_cleaning: 是否启用第二阶段清洗
        """
        super().__init__(target_length)
        self.enable_cleaning = enable_cleaning

        self.cleaner = TrajectoryCleaner(
            max_speed_walk=10.0,
            max_speed_vehicle=50.0,
            max_acceleration=10.0,
            max_time_gap=300.0,
            max_bearing_change=135.0,
            min_segment_length=10,
            max_outlier_ratio=0.3
        )

        self.cleaning_stats = {
            'before': {},
            'after': {}
        }

    def process_segments(self, base_segments: List[dict]) -> List[Tuple[np.ndarray, pd.Series, str]]:
        """
        将基础数据转换为 Exp5 格式（含第二阶段清洗）

        Args:
            base_segments: 基础数据段列表

        Returns:
            处理后的数据列表 [(trajectory_features, time_series, label), ...]
        """
        if not self.enable_cleaning:
            return super().process_segments(base_segments)

        processed_data = []

        total_segments = len(base_segments)
        valid_segments = 0
        discarded_segments = 0

        self.cleaning_stats['before'] = {
            'total_segments': total_segments,
            'total_points': sum(len(seg.get('trajectory', [])) for seg in base_segments)
        }

        for segment in base_segments:
            trajectory = np.array(segment.get('trajectory', []))
            time_series = segment.get('time_series')
            label = segment.get('label', 'Unknown')

            if len(trajectory) < 10:
                discarded_segments += 1
                continue

            if label not in self.valid_labels:
                discarded_segments += 1
                continue

            cleaned_trajectory, is_valid = self.cleaner.clean_segment(trajectory, label)

            if not is_valid:
                discarded_segments += 1
                continue

            cleaned_trajectory = self.cleaner.normalize_sequence_length(
                cleaned_trajectory, self.target_length
            )

            if len(time_series) >= self.target_length:
                time_series = time_series.iloc[:self.target_length]
            else:
                pad_length = self.target_length - len(time_series)
                padding = pd.Series([time_series.iloc[-1]] * pad_length, index=time_series.index[-1:])
                time_series = pd.concat([time_series, padding], ignore_index=True)

            processed_data.append((cleaned_trajectory, time_series, label))
            valid_segments += 1

        self.cleaning_stats['after'] = {
            'valid_segments': valid_segments,
            'discarded_segments': discarded_segments,
            'total_points': sum(len(data[0]) for data in processed_data)
        }

        self.cleaning_stats['cleaner'] = self.cleaner.get_cleaning_stats()

        return processed_data

    def get_cleaning_stats(self) -> Dict:
        """
        获取清洗统计信息

        Returns:
            清洗统计字典
        """
        return self.cleaning_stats.copy()

    def print_cleaning_summary(self):
        """打印清洗摘要"""
        print("\n" + "="*60)
        print("实验五 - 数据清洗摘要")
        print("="*60)

        before = self.cleaning_stats.get('before', {})
        after = self.cleaning_stats.get('after', {})
        cleaner = self.cleaning_stats.get('cleaner', {})

        print(f"\n第一阶段（基础处理）:")
        print(f"  总轨迹段数: {before.get('total_segments', 0)}")
        print(f"  总轨迹点数: {before.get('total_points', 0)}")

        print(f"\n第二阶段（深度清洗）:")
        print(f"  有效轨迹段数: {after.get('valid_segments', 0)}")
        print(f"  丢弃轨迹段数: {after.get('discarded_segments', 0)}")
        print(f"  保留率: {after.get('valid_segments', 0) / before.get('total_segments', 1) * 100:.2f}%")

        print(f"\n清洗详情:")
        print(f"  剔除异常点数: {cleaner.get('outliers_removed', 0)}")
        print(f"  插值点数: {cleaner.get('points_interpolated', 0)}")
        print(f"  方向平滑点数: {cleaner.get('bearing_smoothed', 0)}")
        print(f"  最终轨迹点数: {after.get('total_points', 0)}")

        print("="*60 + "\n")

    def reset_stats(self):
        """重置统计信息"""
        self.cleaning_stats = {
            'before': {},
            'after': {}
        }
        self.cleaner.reset_stats()
