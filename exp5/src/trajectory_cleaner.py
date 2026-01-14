import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class TrajectoryCleaner:
    """轨迹数据清洗器 - 实验五核心模块（GTA-Seg 风格的温柔清洗）"""

    def __init__(self,
                 max_speed_walk: float = 10.0,
                 max_speed_vehicle: float = 50.0,
                 max_acceleration: float = 10.0,
                 max_time_gap: float = 300.0,
                 max_bearing_change: float = 135.0,
                 min_segment_length: int = 10,
                 max_outlier_ratio: float = 0.5):
        """
        初始化轨迹清洗器

        Args:
            max_speed_walk: 步行最大速度 (m/s)
            max_speed_vehicle: 车辆最大速度 (m/s)
            max_acceleration: 最大加速度 (m/s²)
            max_time_gap: 最大时间间隔 (秒)
            max_bearing_change: 最大方向变化 (度)
            min_segment_length: 最小轨迹段长度
            max_outlier_ratio: 最大异常点比例（GTA-Seg: 50%）
        """
        self.max_speed_walk = max_speed_walk
        self.max_speed_vehicle = max_speed_vehicle
        self.max_acceleration = max_acceleration
        self.max_time_gap = max_time_gap
        self.max_bearing_change = max_bearing_change
        self.min_segment_length = min_segment_length
        self.max_outlier_ratio = max_outlier_ratio

        self.cleaning_stats = {
            'total_segments': 0,
            'segments_kept': 0,
            'segments_discarded': 0,
            'total_points': 0,
            'outliers_removed': 0,
            'points_interpolated': 0,
            'bearing_smoothed': 0
        }

    def clean_segment(self, trajectory: np.ndarray, label: str) -> Tuple[np.ndarray, bool]:
        """
        清洗单个轨迹段（GTA-Seg 风格的温柔清洗）

        Args:
            trajectory: 轨迹特征矩阵 (n_points, 9)
            label: 轨迹标签

        Returns:
            (cleaned_trajectory, is_valid): 清洗后的轨迹和是否有效
        """
        self.cleaning_stats['total_segments'] += 1
        self.cleaning_stats['total_points'] += len(trajectory)

        if len(trajectory) < self.min_segment_length:
            self.cleaning_stats['segments_discarded'] += 1
            return np.array([]), False

        cleaned = trajectory.copy()

        # GTA-Seg: 异常点软处理（不删除，而是平滑）
        cleaned, outliers_count = self._remove_outliers_soft(cleaned, label)
        self.cleaning_stats['outliers_removed'] += outliers_count

        # GTA-Seg: 时间连续性处理（少量插值，不删除轨迹段）
        cleaned, interpolated = self._handle_continuity_soft(cleaned)
        self.cleaning_stats['points_interpolated'] += interpolated

        # GTA-Seg: 方向异常平滑（不丢弃，而是平滑）
        cleaned, smoothed = self._smooth_bearing_soft(cleaned, label)
        self.cleaning_stats['bearing_smoothed'] += smoothed

        if len(cleaned) < self.min_segment_length:
            self.cleaning_stats['segments_discarded'] += 1
            return np.array([]), False

        # GTA-Seg: 只有当异常点比例超过 50% 时，才丢弃整段轨迹
        outlier_ratio = outliers_count / len(trajectory) if len(trajectory) > 0 else 0
        if outlier_ratio > self.max_outlier_ratio:
            self.cleaning_stats['segments_discarded'] += 1
            return np.array([]), False

        # GTA-Seg: 其他轨迹都保留，哪怕仍有少量异常点
        self.cleaning_stats['segments_kept'] += 1
        return cleaned, True

    def _remove_outliers_soft(self, trajectory: np.ndarray, label: str) -> Tuple[np.ndarray, int]:
        """
        GTA-Seg 风格的异常点软处理
        不再直接删除异常点，而是使用前后点中值平滑

        Args:
            trajectory: 轨迹特征矩阵
            label: 轨迹标签

        Returns:
            (cleaned_trajectory, outliers_count): 清洗后的轨迹和异常点数量
        """
        if len(trajectory) == 0:
            return trajectory, 0

        n_points = len(trajectory)
        cleaned = trajectory.copy()

        speed = trajectory[:, 2]
        acceleration = trajectory[:, 3]

        # 根据交通方式设置不同的速度阈值（软阈值）
        max_speed = self.max_speed_vehicle if label in ['Car & taxi', 'Bus', 'Train', 'Subway'] else self.max_speed_walk

        # 识别异常点（但不删除）
        outlier_mask = np.zeros(n_points, dtype=bool)

        # 速度异常（软阈值）
        speed_outliers = (speed < 0) | (speed > max_speed)
        outlier_mask |= speed_outliers

        # 加速度异常（软阈值）
        accel_outliers = np.abs(acceleration) > self.max_acceleration
        outlier_mask |= accel_outliers

        # NaN/Inf 异常
        nan_inf_outliers = np.isnan(speed) | np.isnan(acceleration) | np.isinf(speed) | np.isinf(acceleration)
        outlier_mask |= nan_inf_outliers

        outliers_count = np.sum(outlier_mask)

        # GTA-Seg: 对异常点进行软处理：使用前后点中值平滑
        if outliers_count > 0:
            for idx in np.where(outlier_mask)[0]:
                # 获取前后点的索引
                prev_idx = max(0, idx - 1)
                next_idx = min(n_points - 1, idx + 1)

                # 对速度、加速度、方向变化进行中值平滑
                for col in [2, 3, 4]:
                    # 获取前后点的值
                    prev_val = cleaned[prev_idx, col]
                    next_val = cleaned[next_idx, col]
                    curr_val = cleaned[idx, col]

                    # 如果当前值是 NaN/Inf，直接用中值替换
                    if np.isnan(curr_val) or np.isinf(curr_val):
                        cleaned[idx, col] = (prev_val + next_val) / 2
                    # 如果当前值超出阈值，用前后点中值平滑
                    elif col == 2 and (curr_val < 0 or curr_val > max_speed):
                        cleaned[idx, col] = (prev_val + next_val) / 2
                    elif col == 3 and np.abs(curr_val) > self.max_acceleration:
                        cleaned[idx, col] = (prev_val + next_val) / 2

        return cleaned, outliers_count

    def _handle_continuity_soft(self, trajectory: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        GTA-Seg 风格的时间连续性处理
        对大于 max_time_gap 的时间间隔进行少量插值（最多 5 个点）
        不要强制删除整个轨迹段

        Args:
            trajectory: 轨迹特征矩阵

        Returns:
            (cleaned_trajectory, points_interpolated): 清洗后的轨迹和插值的点数
        """
        if len(trajectory) < 2:
            return trajectory, 0

        time_diffs = trajectory[1:, 6] - trajectory[:-1, 6]

        large_gap_indices = np.where(time_diffs > self.max_time_gap)[0]

        if len(large_gap_indices) == 0:
            return trajectory, 0

        cleaned = trajectory.copy()
        points_interpolated = 0

        for idx in large_gap_indices:
            gap_size = time_diffs[idx]
            # GTA-Seg: 最多插值 5 个点（温柔处理）
            num_interpolate = min(int(gap_size / 10), 5)

            if num_interpolate < 2:
                continue

            start_point = cleaned[idx]
            end_point = cleaned[idx + 1]

            for i in range(1, num_interpolate):
                ratio = i / num_interpolate
                interpolated_point = start_point + ratio * (end_point - start_point)
                cleaned = np.insert(cleaned, idx + i, interpolated_point, axis=0)
                points_interpolated += 1

        return cleaned, points_interpolated

    def _smooth_bearing_soft(self, trajectory: np.ndarray, label: str) -> Tuple[np.ndarray, int]:
        """
        GTA-Seg 风格的方向异常平滑
        对方向变化大于 max_bearing_change 的点进行平滑处理，而不是丢弃
        不同交通方式可以设置不同软阈值（例如步行乘以 0.7）

        Args:
            trajectory: 轨迹特征矩阵
            label: 轨迹标签

        Returns:
            (cleaned_trajectory, points_smoothed): 清洗后的轨迹和平滑的点数
        """
        if len(trajectory) < 3:
            return trajectory, 0

        cleaned = trajectory.copy()
        bearing_changes = cleaned[1:, 4]
        points_smoothed = 0

        # 根据交通方式设置不同的方向变化阈值（软阈值）
        if label in ['Car & taxi', 'Bus']:
            max_change = self.max_bearing_change
        else:
            max_change = self.max_bearing_change * 0.7

        # 识别方向变化异常的点
        abnormal_indices = np.where(bearing_changes > max_change)[0]

        # GTA-Seg: 对异常点进行平滑处理（不删除）
        for idx in abnormal_indices:
            if idx > 0 and idx < len(cleaned) - 1:
                prev_bearing = cleaned[idx - 1, 4]
                next_bearing = cleaned[idx + 1, 4]
                # 使用前后点的平均值进行平滑
                avg_bearing = (prev_bearing + next_bearing) / 2
                cleaned[idx, 4] = avg_bearing
                points_smoothed += 1

        return cleaned, points_smoothed

    def normalize_sequence_length(self, trajectory: np.ndarray, target_length: int = 50) -> np.ndarray:
        """
        统一序列长度

        Args:
            trajectory: 轨迹特征矩阵
            target_length: 目标长度

        Returns:
            统一长度后的轨迹
        """
        if len(trajectory) == 0:
            return np.zeros((target_length, 9))

        if len(trajectory) == target_length:
            return trajectory

        if len(trajectory) > target_length:
            indices = np.linspace(0, len(trajectory) - 1, target_length, dtype=int)
            return trajectory[indices]
        else:
            pad_length = target_length - len(trajectory)
            padding = np.zeros((pad_length, 9))
            return np.vstack([trajectory, padding])

    def get_cleaning_stats(self) -> Dict:
        """
        获取清洗统计信息

        Returns:
            清洗统计字典
        """
        return self.cleaning_stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        self.cleaning_stats = {
            'total_segments': 0,
            'segments_kept': 0,
            'segments_discarded': 0,
            'total_points': 0,
            'outliers_removed': 0,
            'points_interpolated': 0,
            'bearing_smoothed': 0
        }
