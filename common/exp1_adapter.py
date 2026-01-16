"""
Exp1 数据适配器
从基础数据生成 Exp1 所需的格式（序列长度100，仅轨迹特征）
集成两阶段数据清洗逻辑
"""
import numpy as np
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


class Exp1DataAdapter:
    """Exp1 数据适配器 - 集成两阶段清洗"""

    def __init__(self,
                 target_length: int = 100,
                 enable_cleaning: bool = True,
                 cleaning_mode: str = 'balanced'):
        """
        初始化Exp1数据适配器

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

    def process_segments(self, base_segments: List[dict]) -> List[Tuple[np.ndarray, str]]:
        """
        两阶段数据处理流程

        第一阶段 (基础预处理):
        - 标签过滤
        - 特征提取
        - 长度规范化

        第二阶段 (深度清洗):
        - 物理异常检测
        - 统计异常处理
        - 轨迹平滑优化

        Args:
            base_segments: 基础数据段列表

        Returns:
            处理后的数据列表 [(features, label), ...]
        """
        print(f"\n{'=' * 80}")
        print(f"Exp1 数据适配 - 两阶段清洗 (模式: {self.cleaning_mode})")
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
            label = self.label_mapping.get(seg['label'], seg['label'])

            valid_segments.append((trajectory, label))

        print(f"  基础过滤: {total_segments} → {len(valid_segments)} "
              f"(丢弃 {stage1_discarded})")

        # ========== 第二阶段: 深度清洗 ==========
        if not self.enable_cleaning:
            print("⚠️ 跳过第二阶段清洗")
            return self._finalize_segments(valid_segments)

        print(f"\n第二阶段: 深度清洗 (模式: {self.cleaning_mode})...")

        cleaned_segments = []
        stage2_discarded = 0

        for trajectory, label in tqdm(valid_segments, desc="[阶段2: 深度清洗]"):
            # 执行清洗
            cleaned_traj, is_valid = self.cleaner.clean_segment(trajectory, label)

            if not is_valid:
                stage2_discarded += 1
                continue

            # 长度规范化
            cleaned_traj = self.cleaner.normalize_sequence_length(
                cleaned_traj, self.target_length
            )

            cleaned_segments.append((cleaned_traj, label))

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

        return cleaned_segments

    def _finalize_segments(self, segments: List[Tuple]) -> List[Tuple]:
        """完成段处理 (跳过清洗模式)"""
        finalized = []

        for trajectory, label in segments:
            # 长度规范化
            if len(trajectory) != self.target_length:
                if len(trajectory) > self.target_length:
                    indices = np.linspace(0, len(trajectory) - 1,
                                          self.target_length, dtype=int)
                    trajectory = trajectory[indices]
                else:
                    padding = np.zeros((self.target_length - len(trajectory), 9),
                                       dtype=np.float32)
                    trajectory = np.vstack([trajectory, padding])

            finalized.append((trajectory, label))

        return finalized

    def get_cleaning_stats(self) -> Dict:
        """获取清洗统计信息"""
        return self.cleaning_stats.copy()

    def print_cleaning_summary(self):
        """打印清洗摘要"""
        print("\n" + "=" * 80)
        print("Exp1 数据清洗摘要报告")
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