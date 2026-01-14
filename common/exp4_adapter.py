"""
Exp4 数据适配器
从基础数据生成 Exp4 所需的格式（序列长度50，轨迹特征 + 增强KG特征 + 天气特征）
特别保留时间序列信息用于天气数据匹配
集成两阶段数据清洗逻辑
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from tqdm import tqdm

# 导入轨迹清洗器
try:
    from common.trajectory_cleaner import TrajectoryCleaner
except ImportError:
    # 如果导入失败，创建基础类
    class TrajectoryCleaner:
        def __init__(self, **kwargs):
            pass
        def clean_segment(self, trajectory, label):
            return trajectory, True
        def normalize_sequence_length(self, trajectory, target_length):
            return trajectory
        def get_cleaning_stats(self):
            return {}


class Exp4DataAdapter:
    """Exp4 数据适配器 - 集成两阶段清洗（保留时间序列）"""

    def __init__(self,
                 target_length: int = 50,
                 enable_cleaning: bool = True,
                 cleaning_mode: str = 'balanced'):
        """
        初始化Exp4数据适配器

        Args:
            target_length: 目标序列长度
            enable_cleaning: 是否启用第二阶段清洗
            cleaning_mode: 清洗模式
                - 'strict': 严格模式 (高质量，低保留率)
                - 'balanced': 平衡模式 (推荐)
                - 'gentle': 温和模式 (高保留率)
        """
        self.target_length = target_length
        self.enable_cleaning = enable_cleaning
        self.cleaning_mode = cleaning_mode

        # Exp4使用的标签
        self.valid_labels = {
            'Walk', 'Bike', 'Bus', 'Car & taxi',
            'Train', 'Subway', 'Airplane'
        }

        # 根据模式设置清洗参数
        self._setup_cleaning_params()

        # 初始化清洗器
        self.cleaner = TrajectoryCleaner(
            max_time_gap=self.max_time_gap,
            max_bearing_change=self.max_bearing_change,
            min_segment_length=self.min_segment_length,
            max_outlier_ratio=self.max_outlier_ratio,
            enable_smoothing=self.enable_smoothing,
            smoothing_window=self.smoothing_window
        )

        # 统计信息
        self.cleaning_stats = {
            'before': {},
            'after': {},
            'cleaner': {}
        }

    def _setup_cleaning_params(self):
        """根据清洗模式设置参数"""
        if self.cleaning_mode == 'strict':
            self.max_time_gap = 180.0
            self.max_bearing_change = 120.0
            self.min_segment_length = 15
            self.max_outlier_ratio = 0.15
            self.enable_smoothing = True
            self.smoothing_window = 7

        elif self.cleaning_mode == 'gentle':
            self.max_time_gap = 600.0
            self.max_bearing_change = 180.0
            self.min_segment_length = 8
            self.max_outlier_ratio = 0.35
            self.enable_smoothing = False
            self.smoothing_window = 3

        else:  # balanced (默认)
            self.max_time_gap = 300.0
            self.max_bearing_change = 150.0
            self.min_segment_length = 10
            self.max_outlier_ratio = 0.25
            self.enable_smoothing = True
            self.smoothing_window = 5

    def process_segments(self, base_segments: List[dict]) -> List[Tuple[np.ndarray, pd.Series, str]]:
        """
        两阶段数据处理流程（保留时间序列）

        第一阶段 (基础预处理):
        - 标签过滤
        - 特征提取
        - 长度过滤

        第二阶段 (深度清洗):
        - 物理异常检测
        - 统计异常处理
        - 轨迹平滑优化

        Args:
            base_segments: 基础数据段列表

        Returns:
            处理后的数据列表 [(features, datetime_series, label), ...]
        """
        print(f"\n{'=' * 80}")
        print(f"Exp4 数据适配 - 两阶段清洗 (模式: {self.cleaning_mode}, 保留时间序列)")
        print(f"{'=' * 80}\n")

        # ========== 第一阶段: 基础预处理 ==========
        print("第一阶段: 基础预处理...")

        total_segments = len(base_segments)
        self.cleaning_stats['before'] = {
            'total_segments': total_segments,
            'total_points': sum(len(seg.get('raw_points', [])) for seg in base_segments)
        }

        # 提取有效段
        valid_segments = []
        stage1_discarded = 0

        for seg in tqdm(base_segments, desc="[阶段1: 基础过滤]"):
            # 标签过滤
            if seg['label'] not in self.valid_labels:
                stage1_discarded += 1
                continue

            # 长度过滤
            if len(seg.get('raw_points', [])) < self.min_segment_length:
                stage1_discarded += 1
                continue

            # 提取9维特征
            points = seg['raw_points']
            feature_cols = [
                'latitude', 'longitude', 'speed', 'acceleration',
                'bearing_change', 'distance', 'time_diff',
                'total_distance', 'total_time'
            ]

            # 确保所有特征列存在
            for col in feature_cols:
                if col not in points.columns:
                    points[col] = 0.0

            trajectory = points[feature_cols].values.astype(np.float32)
            datetime_series = seg['datetime_series']
            label = seg['label']

            valid_segments.append((trajectory, datetime_series, label))

        print(f"  基础过滤: {total_segments} → {len(valid_segments)} "
              f"(丢弃 {stage1_discarded})")

        # ========== 第二阶段: 深度清洗 ==========
        if not self.enable_cleaning:
            print("⚠️ 跳过第二阶段清洗")
            return self._finalize_segments(valid_segments)

        print(f"\n第二阶段: 深度清洗 (模式: {self.cleaning_mode})...")

        cleaned_segments = []
        stage2_discarded = 0

        for trajectory, datetime_series, label in tqdm(valid_segments, desc="[阶段2: 深度清洗]"):
            # 执行清洗
            cleaned_traj, is_valid = self.cleaner.clean_segment(trajectory, label)

            if not is_valid:
                stage2_discarded += 1
                continue

            # 同步清洗时间序列（根据清洗后的轨迹长度）
            cleaned_datetime = self._sync_time_series(
                datetime_series, len(trajectory), len(cleaned_traj)
            )

            # 长度规范化（同步特征和时间）
            cleaned_traj, cleaned_datetime = self._normalize_length_with_time(
                cleaned_traj, cleaned_datetime, self.target_length
            )

            cleaned_segments.append((cleaned_traj, cleaned_datetime, label))

        print(f"  深度清洗: {len(valid_segments)} → {len(cleaned_segments)} "
              f"(丢弃 {stage2_discarded})")

        # ========== 统计与报告 ==========
        self.cleaning_stats['after'] = {
            'valid_segments': len(cleaned_segments),
            'stage1_discarded': stage1_discarded,
            'stage2_discarded': stage2_discarded,
            'total_discarded': stage1_discarded + stage2_discarded,
            'retention_rate': len(cleaned_segments) / max(total_segments, 1),
            'total_points': sum(len(data[0]) for data in cleaned_segments)
        }

        self.cleaning_stats['cleaner'] = self.cleaner.get_cleaning_stats()

        # 打印摘要
        self.print_cleaning_summary()

        # 打印标签分布
        self._print_label_distribution(cleaned_segments)

        return cleaned_segments

    def _sync_time_series(self, datetime_series: pd.Series,
                           original_length: int, cleaned_length: int) -> pd.Series:
        """
        同步时间序列到清洗后的轨迹长度

        Args:
            datetime_series: 原始时间序列
            original_length: 原始轨迹长度
            cleaned_length: 清洗后轨迹长度

        Returns:
            同步后的时间序列
        """
        if original_length == cleaned_length:
            return datetime_series.reset_index(drop=True)

        # 如果清洗后长度变化，使用等间隔采样
        indices = np.linspace(0, original_length - 1, cleaned_length, dtype=int)
        return datetime_series.iloc[indices].reset_index(drop=True)

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

    def _finalize_segments(self, segments: List[Tuple]) -> List[Tuple]:
        """完成段处理 (跳过清洗模式)"""
        finalized = []

        for trajectory, datetime_series, label in segments:
            # 长度规范化（同步特征和时间）
            normalized_traj, normalized_time = self._normalize_length_with_time(
                trajectory, datetime_series, self.target_length
            )

            finalized.append((normalized_traj, normalized_time, label))

        return finalized

    def get_cleaning_stats(self) -> Dict:
        """获取清洗统计信息"""
        return self.cleaning_stats.copy()

    def print_cleaning_summary(self):
        """打印清洗摘要"""
        print("\n" + "=" * 80)
        print("Exp4 数据清洗摘要报告")
        print("=" * 80)

        before = self.cleaning_stats.get('before', {})
        after = self.cleaning_stats.get('after', {})
        cleaner = self.cleaning_stats.get('cleaner', {})

        # 第一阶段统计
        print(f"\n📌 第一阶段 (基础预处理):")
        print(f"  输入轨迹段数: {before.get('total_segments', 0):,}")
        print(f"  输入轨迹点数: {before.get('total_points', 0):,}")
        print(f"  基础过滤丢弃: {after.get('stage1_discarded', 0):,}")

        # 第二阶段统计
        print(f"\n🛠️ 第二阶段 (深度清洗, 模式: {self.cleaning_mode}):")
        print(f"  深度清洗丢弃: {after.get('stage2_discarded', 0):,}")
        print(f"  最终保留段数: {after.get('valid_segments', 0):,}")
        print(f"  总体保留率: {after.get('retention_rate', 0):.2%}")

        # 清洗详情
        print(f"\n🔧 清洗操作详情:")
        print(f"  物理异常修复: {cleaner.get('outliers_removed', 0):,} 个点")
        print(f"  时间间隔插值: {cleaner.get('points_interpolated', 0):,} 个点")
        print(f"  轨迹平滑优化: {cleaner.get('points_smoothed', 0):,} 个点")

        # 丢弃原因
        discard_reasons = cleaner.get('discard_reasons', {})
        if discard_reasons:
            print(f"\n❌ 丢弃原因分布:")
            print(f"  长度过短: {discard_reasons.get('too_short', 0):,}")
            print(f"  异常点过多: {discard_reasons.get('too_many_outliers', 0):,}")
            print(f"  清洗后无效: {discard_reasons.get('invalid_after_cleaning', 0):,}")

        print("=" * 80 + "\n")

    def _print_label_distribution(self, processed: List[Tuple]):
        """打印标签分布"""
        from collections import Counter
        labels = [label for _, _, label in processed]
        counts = Counter(labels)

        print("\n标签分布:")
        for label in sorted(counts.keys()):
            print(f"  {label:15s}: {counts[label]:6d}")
