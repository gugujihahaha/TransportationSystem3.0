import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from tqdm import tqdm

# 导入父类和清洗器
try:
    from common.exp4_adapter import Exp4DataAdapter
except ImportError:
    # 如果导入失败，创建基础类
    class Exp4DataAdapter:
        def __init__(self, target_length: int = 50):
            self.target_length = target_length
            self.valid_labels = {
                'Walk', 'Bike', 'Bus', 'Car & taxi',
                'Train', 'Subway', 'Airplane'
            }


class Exp5DataAdapter(Exp4DataAdapter):
    """
    改进的Exp5数据适配器 - 两阶段清洗 + 质量评估

    核心改进:
    1. 第一阶段: 基础预处理 (长度规范化、缺失值填充)
    2. 第二阶段: 深度清洗 (物理异常、统计异常、轨迹平滑)
    3. 质量评估: 为每个轨迹段计算质量分数
    4. 自适应阈值: 根据数据分布动态调整清洗参数
    """

    def __init__(self,
                 target_length: int = 50,
                 enable_cleaning: bool = True,
                 cleaning_mode: str = 'balanced',
                 use_cleaned_data: bool = False,
                 cleaned_data_path: str = None,
                 kg=None,
                 weather=None):
        """
        初始化Exp5数据适配器

        Args:
            target_length: 目标序列长度
            enable_cleaning: 是否启用第二阶段清洗（当use_cleaned_data=True时忽略）
            cleaning_mode: 清洗模式（当use_cleaned_data=True时忽略）
                - 'strict': 严格模式 (高质量，低保留率)
                - 'balanced': 平衡模式 (推荐)
                - 'gentle': 温和模式 (高保留率)
            use_cleaned_data: 是否使用清洗后的数据（推荐）
            cleaned_data_path: 清洗后数据路径
            kg: 知识图谱实例（用于生成增强KG特征）
            weather: 天气数据实例（用于生成天气特征）
        """
        super().__init__(target_length, enable_cleaning, cleaning_mode,
                        use_cleaned_data, cleaned_data_path, kg, weather)
        
        self.enable_cleaning = enable_cleaning
        self.cleaning_mode = cleaning_mode
        self.use_cleaned_data = use_cleaned_data
        self.cleaned_data_path = cleaned_data_path
        self.kg = kg
        self.weather = weather

        # 根据模式设置清洗参数（仅在不使用清洗后数据时）
        if not self.use_cleaned_data:
            self._setup_cleaning_params()

            # 导入清洗器
            from common.trajectory_cleaner import TrajectoryCleaner

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
            'quality': {}
        }

        # 质量评估阈值
        self.quality_thresholds = {
            'high': 0.8,  # 高质量轨迹
            'medium': 0.6,  # 中等质量
            'low': 0.4  # 低质量 (但保留)
        }

    def _setup_cleaning_params(self):
        """根据清洗模式设置参数"""
        if self.cleaning_mode == 'strict':
            self.max_time_gap = 180.0  # 3分钟
            self.max_bearing_change = 120.0  # 120度
            self.min_segment_length = 15  # 最少15个点
            self.max_outlier_ratio = 0.15  # 15%异常点
            self.enable_smoothing = True
            self.smoothing_window = 7

        elif self.cleaning_mode == 'gentle':
            self.max_time_gap = 600.0  # 10分钟
            self.max_bearing_change = 180.0  # 180度
            self.min_segment_length = 8  # 最少8个点
            self.max_outlier_ratio = 0.35  # 35%异常点
            self.enable_smoothing = False
            self.smoothing_window = 3

        else:  # balanced (默认)
            self.max_time_gap = 300.0  # 5分钟
            self.max_bearing_change = 150.0  # 150度
            self.min_segment_length = 10  # 最少10个点
            self.max_outlier_ratio = 0.25  # 25%异常点
            self.enable_smoothing = True
            self.smoothing_window = 5

    def load_cleaned_data(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        """
        加载清洗后的数据并生成KG和天气特征（推荐方式）

        Returns:
            处理后的数据列表 [(trajectory, kg_features, weather_features, label), ...]
        """
        if not self.use_cleaned_data:
            print("⚠️ 未启用清洗后数据模式，请设置 use_cleaned_data=True")
            return []

        if not self.cleaned_data_path:
            print("❌ 错误: 未指定清洗后数据路径")
            return []

        if self.kg is None:
            print("❌ 错误: 未传入知识图谱实例（kg参数）")
            return []

        if self.weather is None:
            print("❌ 错误: 未传入天气数据实例（weather参数）")
            return []

        print(f"\n{'=' * 80}")
        print(f"Exp5 数据适配 - 加载清洗后数据 + 生成KG和天气特征")
        print(f"{'=' * 80}\n")

        import os
        import pickle

        if not os.path.exists(self.cleaned_data_path):
            print(f"❌ 错误: 找不到清洗后数据: {self.cleaned_data_path}")
            print("\n请先运行以下命令生成清洗后数据:")
            print("  python scripts/generate_cleaned_base_data.py")
            return []

        print(f"正在加载清洗后数据: {self.cleaned_data_path}")

        with open(self.cleaned_data_path, 'rb') as f:
            data = pickle.load(f)
            
            if len(data) == 3:
                cleaned_segments, cleaning_stats, cleaning_mode = data
            else:
                cleaned_segments, cleaning_stats = data
                cleaning_mode = 'unknown'

        print(f"   ✓ 加载完成: {len(cleaned_segments)} 个轨迹段")
        print(f"   ✓ 清洗模式: {cleaning_mode}")

        # 打印清洗统计
        self._print_cleaned_stats(cleaning_stats)

        # 处理清洗后的数据（轨迹 + KG + 天气特征）
        features = []
        quality_scores = []

        for seg in tqdm(cleaned_segments, desc="[生成KG和天气特征]"):
            # 提取清洗后的轨迹
            trajectory = seg['cleaned_trajectory']
            label = seg['label']

            # 长度规范化（如果需要）
            if len(trajectory) != self.target_length:
                if len(trajectory) > self.target_length:
                    indices = np.linspace(0, len(trajectory) - 1,
                                          self.target_length, dtype=int)
                    trajectory = trajectory[indices]
                else:
                    padding = np.zeros((self.target_length - len(trajectory), 9),
                                       dtype=np.float32)
                    trajectory = np.vstack([trajectory, padding])

            # 生成KG特征（15维）
            try:
                kg_features = self.kg.extract_kg_features(trajectory)
                
                # 扩展KG特征为15维（Exp5使用增强KG）
                enhanced_kg_features = np.zeros((self.target_length, 15), dtype=np.float32)
                enhanced_kg_features[:, :11] = kg_features
                
                # 添加额外的4维特征
                enhanced_kg_features[:, 11] = np.mean(kg_features[:, 2], axis=0)  # 速度均值
                enhanced_kg_features[:, 12] = np.std(kg_features[:, 2], axis=0)   # 速度标准差
                enhanced_kg_features[:, 13] = np.mean(kg_features[:, 3], axis=0)  # 加速度均值
                enhanced_kg_features[:, 14] = np.std(kg_features[:, 3], axis=0)   # 加速度标准差
                
            except Exception as e:
                print(f"⚠️ KG特征生成失败: {e}")
                enhanced_kg_features = np.zeros((self.target_length, 15), dtype=np.float32)

            # 生成天气特征（12维）
            try:
                weather_features = self.weather.extract_weather_features(trajectory)
            except Exception as e:
                print(f"⚠️ 天气特征生成失败: {e}")
                weather_features = np.zeros((self.target_length, 12), dtype=np.float32)

            # 计算质量分数
            quality_score = self._calculate_quality_score(trajectory)
            quality_scores.append(quality_score)

            features.append((trajectory, enhanced_kg_features, weather_features, label))

        print(f"\n✅ 处理完成: {len(features)} 个样本")
        print(f"   - 轨迹特征: {trajectory.shape}")
        print(f"   - KG特征: {enhanced_kg_features.shape}")
        print(f"   - 天气特征: {weather_features.shape}")

        # 打印质量统计
        self._print_quality_stats(quality_scores)

        return features

    def _calculate_quality_score(self, trajectory: np.ndarray) -> float:
        """
        计算轨迹质量分数

        Args:
            trajectory: 轨迹特征数组 (N, 9)

        Returns:
            质量分数 (0-1)
        """
        # 速度合理性 (0-0.4)
        speeds = trajectory[:, 2]
        speed_score = 0.4 * (1 - np.mean(np.abs(speeds) > 30))

        # 加速度合理性 (0-0.3)
        accelerations = trajectory[:, 3]
        accel_score = 0.3 * (1 - np.mean(np.abs(accelerations) > 10))

        # 方向变化合理性 (0-0.3)
        bearing_changes = trajectory[:, 4]
        bearing_score = 0.3 * (1 - np.mean(np.abs(bearing_changes) > 180))

        return speed_score + accel_score + bearing_score

    def _print_quality_stats(self, quality_scores: List[float]):
        """打印质量统计信息"""
        print("\n" + "=" * 80)
        print("轨迹质量评估")
        print("=" * 80)

        quality_scores = np.array(quality_scores)
        print(f"\n质量分数统计:")
        print(f"  - 平均质量: {np.mean(quality_scores):.3f}")
        print(f"  - 中位数质量: {np.median(quality_scores):.3f}")
        print(f"  - 最低质量: {np.min(quality_scores):.3f}")
        print(f"  - 最高质量: {np.max(quality_scores):.3f}")

        # 质量分布
        high_quality = np.sum(quality_scores >= self.quality_thresholds['high'])
        medium_quality = np.sum((quality_scores >= self.quality_thresholds['medium']) & 
                               (quality_scores < self.quality_thresholds['high']))
        low_quality = np.sum((quality_scores >= self.quality_thresholds['low']) & 
                             (quality_scores < self.quality_thresholds['medium']))
        very_low_quality = np.sum(quality_scores < self.quality_thresholds['low'])

        print(f"\n质量分布:")
        print(f"  - 高质量 (≥{self.quality_thresholds['high']}): {high_quality} ({high_quality/len(quality_scores)*100:.1f}%)")
        print(f"  - 中等质量 ({self.quality_thresholds['medium']}-{self.quality_thresholds['high']}): {medium_quality} ({medium_quality/len(quality_scores)*100:.1f}%)")
        print(f"  - 低质量 ({self.quality_thresholds['low']}-{self.quality_thresholds['medium']}): {low_quality} ({low_quality/len(quality_scores)*100:.1f}%)")
        print(f"  - 极低质量 (<{self.quality_thresholds['low']}): {very_low_quality} ({very_low_quality/len(quality_scores)*100:.1f}%)")

        print("=" * 80 + "\n")

    def _print_cleaned_stats(self, cleaning_stats: Dict):
        """打印清洗统计信息"""
        print("\n" + "=" * 80)
        print("数据清洗统计")
        print("=" * 80)

        print(f"\n清洗统计:")
        print(f"  - 原始轨迹段: {cleaning_stats.get('total_segments', 0):,}")
        print(f"  - 保留轨迹段: {cleaning_stats.get('segments_kept', 0):,}")
        print(f"  - 丢弃轨迹段: {cleaning_stats.get('segments_discarded', 0):,}")
        print(f"  - 保留率: {cleaning_stats.get('segments_kept', 0) / max(cleaning_stats.get('total_segments', 1), 1) * 100:.2f}%")

        print(f"\n清洗操作:")
        print(f"  - 剔除异常点: {cleaning_stats.get('outliers_removed', 0):,}")
        print(f"  - 插值点数: {cleaning_stats.get('points_interpolated', 0):,}")
        print(f"  - 平滑点数: {cleaning_stats.get('points_smoothed', 0):,}")

        print("=" * 80 + "\n")

    def process_segments(self, base_segments: List[dict]) -> List[Tuple[np.ndarray, pd.Series, str]]:
        """
        两阶段数据处理流程

        第一阶段 (基础预处理):
        - 长度过滤
        - 标签过滤
        - 特征提取

        第二阶段 (深度清洗):
        - 物理异常检测
        - 统计异常处理
        - 轨迹平滑优化
        - 质量评估

        Args:
            base_segments: 基础数据段列表

        Returns:
            处理后的数据列表 [(trajectory_features, time_series, label), ...]
        """
        print(f"\n{'=' * 80}")
        print(f"Exp5 数据适配 - 两阶段清洗 (模式: {self.cleaning_mode})")
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
        quality_scores = []

        for trajectory, datetime_series, label in tqdm(valid_segments,
                                                       desc="[阶段2: 深度清洗]"):
            # 执行清洗
            cleaned_traj, is_valid = self.cleaner.clean_segment(trajectory, label)

            if not is_valid:
                stage2_discarded += 1
                continue

            # 计算质量分数
            quality_score = self._compute_quality_score(cleaned_traj, trajectory)
            quality_scores.append(quality_score)

            # 长度规范化
            cleaned_traj = self.cleaner.normalize_sequence_length(
                cleaned_traj, self.target_length
            )

            # 时间序列规范化
            datetime_normalized = self._normalize_time_series(
                datetime_series, self.target_length
            )

            cleaned_segments.append((cleaned_traj, datetime_normalized, label))

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

        self.cleaning_stats['quality'] = self._analyze_quality(quality_scores)
        self.cleaning_stats['cleaner'] = self.cleaner.get_cleaning_stats()

        # 打印摘要
        self.print_cleaning_summary()

        return cleaned_segments

    def _normalize_time_series(self, datetime_series: pd.Series,
                               target_length: int) -> pd.Series:
        """规范化时间序列长度"""
        current_length = len(datetime_series)

        if current_length == target_length:
            return datetime_series.reset_index(drop=True)

        elif current_length > target_length:
            # 下采样
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)
            return datetime_series.iloc[indices].reset_index(drop=True)

        else:
            # 上采样 (填充最后时间)
            last_time = datetime_series.iloc[-1]
            padding = pd.Series([last_time] * (target_length - current_length))
            return pd.concat([datetime_series.reset_index(drop=True), padding],
                             ignore_index=True)

    def _compute_quality_score(self, cleaned: np.ndarray,
                               original: np.ndarray) -> float:
        """
        计算轨迹质量分数 (0-1)

        评估指标:
        1. 数据完整性: 清洗后保留了多少原始点
        2. 平滑度: 速度和加速度的变化是否平滑
        3. 物理合理性: 是否符合物理约束

        Returns:
            质量分数 (0-1, 越高越好)
        """
        try:
            # 指标1: 数据完整性 (30%)
            completeness = len(cleaned) / max(len(original), 1)
            completeness_score = min(completeness, 1.0) * 0.3

            # 指标2: 平滑度 (40%)
            speed_var = np.var(cleaned[:, 2])  # 速度方差
            accel_var = np.var(cleaned[:, 3])  # 加速度方差

            # 归一化方差 (方差越小越平滑)
            speed_smoothness = 1.0 / (1.0 + speed_var)
            accel_smoothness = 1.0 / (1.0 + accel_var)
            smoothness_score = (speed_smoothness + accel_smoothness) / 2 * 0.4

            # 指标3: 物理合理性 (30%)
            # 检查异常值比例
            valid_speed = np.sum((cleaned[:, 2] >= 0) & (cleaned[:, 2] < 100))
            valid_accel = np.sum(np.abs(cleaned[:, 3]) < 20)

            physical_score = (valid_speed + valid_accel) / (2 * len(cleaned)) * 0.3

            # 总分
            total_score = completeness_score + smoothness_score + physical_score

            return min(max(total_score, 0.0), 1.0)

        except Exception:
            return 0.5  # 默认中等质量

    def _analyze_quality(self, quality_scores: List[float]) -> Dict:
        """分析质量分数分布"""
        if not quality_scores:
            return {
                'mean_quality': 0.0,
                'high_quality_count': 0,
                'medium_quality_count': 0,
                'low_quality_count': 0
            }

        scores = np.array(quality_scores)

        high_count = np.sum(scores >= self.quality_thresholds['high'])
        medium_count = np.sum((scores >= self.quality_thresholds['medium']) &
                              (scores < self.quality_thresholds['high']))
        low_count = np.sum(scores < self.quality_thresholds['medium'])

        return {
            'mean_quality': float(np.mean(scores)),
            'median_quality': float(np.median(scores)),
            'std_quality': float(np.std(scores)),
            'high_quality_count': int(high_count),
            'medium_quality_count': int(medium_count),
            'low_quality_count': int(low_count),
            'high_quality_ratio': float(high_count / len(scores)) if len(scores) > 0 else 0.0
        }

    def _finalize_segments(self, segments: List[Tuple]) -> List[Tuple]:
        """完成段处理 (跳过清洗模式)"""
        finalized = []

        for trajectory, datetime_series, label in segments:
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

            # 时间序列规范化
            datetime_normalized = self._normalize_time_series(
                datetime_series, self.target_length
            )

            finalized.append((trajectory, datetime_normalized, label))

        return finalized

    def get_cleaning_stats(self) -> Dict:
        """获取清洗统计信息"""
        return self.cleaning_stats.copy()

    def print_cleaning_summary(self):
        """打印清洗摘要"""
        print("\n" + "=" * 80)
        print("Exp5 数据清洗摘要报告")
        print("=" * 80)

        before = self.cleaning_stats.get('before', {})
        after = self.cleaning_stats.get('after', {})
        quality = self.cleaning_stats.get('quality', {})
        cleaner = self.cleaning_stats.get('cleaner', {})

        # 第一阶段统计
        print(f"\n📌 第一阶段 (基础预处理):")
        print(f"  输入轨迹段数: {before.get('total_segments', 0):,}")
        print(f"  输入轨迹点数: {before.get('total_points', 0):,}")
        print(f"  基础过滤丢弃: {after.get('stage1_discarded', 0):,}")

        # 第二阶段统计
        print(f"\n🛠️ 第二阶段 (深度清洗, 模式: {self.cleaning_mode}):")
        print(f"  清洗模式: {self.cleaning_mode}")
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

        # 质量评估
        print(f"\n⭐ 数据质量评估:")
        print(f"  平均质量分数: {quality.get('mean_quality', 0):.3f}")
        print(f"  高质量轨迹 (≥0.8): {quality.get('high_quality_count', 0):,} "
              f"({quality.get('high_quality_ratio', 0):.1%})")
        print(f"  中等质量 (0.6-0.8): {quality.get('medium_quality_count', 0):,}")
        print(f"  低质量 (<0.6): {quality.get('low_quality_count', 0):,}")

        # 最终结果
        print(f"\n✅ 最终数据集:")
        print(f"  有效轨迹段数: {after.get('valid_segments', 0):,}")
        print(f"  有效轨迹点数: {after.get('total_points', 0):,}")

        print("=" * 80 + "\n")

    def reset_stats(self):
        """重置统计信息"""
        self.cleaning_stats = {
            'before': {},
            'after': {},
            'quality': {}
        }
        self.cleaner.reset_stats()