"""
通用GeoLife数据预处理器
一次性提取所有实验共用的基础数据，避免重复处理

输出数据结构:
{
    'user_id': str,
    'trajectory_id': str,
    'raw_points': DataFrame,  # 原始GPS点 + 9维特征
    'label': str,             # 归一化后的标签
    'start_time': datetime,
    'end_time': datetime,
    'datetime_series': Series # 时间序列（用于天气匹配）
}
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from tqdm import tqdm
import pickle
import warnings

warnings.filterwarnings('ignore')


FEATURE_COLUMNS = [
    'latitude', 'longitude', 'speed', 'acceleration',
    'bearing_change', 'distance', 'time_diff',
    'total_distance', 'total_time'
]

# 特征维度常量（供外部引用）
POINT_FEATURE_DIM = 9   # 原始点级特征维度（不变）
SEGMENT_STATS_DIM = 18  # 段级统计特征维度
TOTAL_FEATURE_DIM = POINT_FEATURE_DIM + SEGMENT_STATS_DIM  # 27


class BaseGeoLifePreprocessor:
    """GeoLife基础数据预处理器（所有实验共用）"""

    def __init__(self, geolife_root: str):
        self.geolife_root = geolife_root

        # 标签映射规则（统一所有实验）
        self.label_mapping = {
            'taxi': 'Car & taxi',
            'car': 'Car & taxi',
            'drive': 'Car & taxi',
            'bus': 'Bus',
            'walk': 'Walk',
            'bike': 'Bike',
            'train': 'Train',
            'subway': 'Subway',
            'railway': 'Train',
            'airplane': 'Airplane'
        }

    def process_all_users(self, max_users: int = None,
                          min_segment_length: int = 10) -> List[Dict]:
        """
        一次性处理所有用户数据

        返回:
            所有轨迹段的列表，每个元素包含完整信息
        """
        print("\n" + "=" * 80)
        print("GeoLife 基础数据预处理（所有实验通用）")
        print("=" * 80)

        users = self._get_all_users()
        if max_users:
            users = users[:max_users]

        print(f"\n找到 {len(users)} 个用户")

        all_segments = []

        for user_id in tqdm(users, desc="[处理用户]"):
            # 加载标签
            labels = self._load_labels(user_id)
            if labels.empty:
                continue

            # 处理该用户的所有轨迹
            trajectory_dir = os.path.join(
                self.geolife_root, f"Data/{user_id}/Trajectory"
            )

            if not os.path.exists(trajectory_dir):
                continue

            for traj_file in os.listdir(trajectory_dir):
                if not traj_file.endswith('.plt'):
                    continue

                traj_path = os.path.join(trajectory_dir, traj_file)
                trajectory_id = traj_file.replace('.plt', '')

                try:
                    # 加载并计算特征
                    trajectory = self._load_and_compute_features(traj_path)

                    if trajectory.empty or len(trajectory) < min_segment_length:
                        continue

                    # 按标签分割轨迹
                    segments = self._segment_trajectory(
                        trajectory, labels, user_id, trajectory_id
                    )

                    all_segments.extend(segments)

                except Exception as e:
                    warnings.warn(f"处理失败 {traj_path}: {e}")
                    continue

        print(f"\n✅ 预处理完成，共 {len(all_segments)} 个轨迹段")

        # 统计信息
        self._print_statistics(all_segments)

        return all_segments

    def _get_all_users(self) -> List[str]:
        """获取所有用户ID"""
        data_path = os.path.join(self.geolife_root, "Data")
        users = []
        for item in os.listdir(data_path):
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path) and item.isdigit():
                users.append(item)
        return sorted(users)

    def _load_labels(self, user_id: str) -> pd.DataFrame:
        """加载用户标签"""
        labels_path = os.path.join(
            self.geolife_root, f"Data/{user_id}/labels.txt"
        )

        if not os.path.exists(labels_path):
            return pd.DataFrame()

        df = pd.read_csv(labels_path, sep='\t')
        df['Start Time'] = pd.to_datetime(df['Start Time'])
        df['End Time'] = pd.to_datetime(df['End Time'])

        return df

    def _load_and_compute_features(self, file_path: str) -> pd.DataFrame:
        """
        加载轨迹文件并计算9维特征

        核心特征计算（所有实验共用）：
        1. latitude, longitude (原始)
        2. speed, acceleration (运动特征)
        3. bearing_change (方向变化)
        4. distance, time_diff (基础)
        5. total_distance, total_time (累积)
        """
        # 1. 读取文件（处理6/7列格式）
        try:
            df = pd.read_csv(file_path, skiprows=6, header=None)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

        num_cols = df.shape[1]

        # 2. 标准化列名
        if num_cols == 7:
            df.columns = [
                'latitude', 'longitude', 'reserved',
                'altitude', 'date_days', 'date', 'time'
            ]
        elif num_cols == 6:
            df.columns = [
                'latitude', 'longitude', 'altitude',
                'date_days', 'date', 'time'
            ]
            df.insert(2, 'reserved', 0)
        else:
            return pd.DataFrame()

        # 3. 合并日期时间
        df['datetime'] = pd.to_datetime(
            df['date'] + ' ' + df['time'],
            format='%Y-%m-%d %H:%M:%S'
        )
        df = df.sort_values('datetime').reset_index(drop=True)

        # 4. 清洗无效坐标
        valid_mask = (
                (df['latitude'] >= -90) & (df['latitude'] <= 90) &
                (df['longitude'] >= -180) & (df['longitude'] <= 180)
        )
        df = df[valid_mask].reset_index(drop=True)

        if len(df) < 2:
            return pd.DataFrame()

        # 5. 向量化计算9维特征
        df = self._compute_trajectory_features(df)

        # 6. 只保留需要的列
        keep_cols = [
            'datetime', 'latitude', 'longitude',
            'speed', 'acceleration', 'bearing_change',
            'distance', 'time_diff',
            'total_distance', 'total_time'
        ]

        return df[keep_cols]

    def _compute_trajectory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """向量化计算轨迹特征"""

        # 1. 时间差
        df['time_diff'] = df['datetime'].diff().dt.total_seconds().fillna(0)

        # 2. 距离（Haversine公式 - 向量化）
        lat1 = df['latitude'].shift(1).fillna(df['latitude'].iloc[0])
        lon1 = df['longitude'].shift(1).fillna(df['longitude'].iloc[0])
        lat2 = df['latitude']
        lon2 = df['longitude']

        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        distances = 6371000 * c  # 地球半径（米）
        distances.iloc[0] = 0.0
        df['distance'] = distances

        # 3. 速度和加速度
        time_diff_safe = df['time_diff'].replace(0, 1e-6)
        df['speed'] = df['distance'] / time_diff_safe
        df['acceleration'] = df['speed'].diff() / time_diff_safe
        df['acceleration'] = df['acceleration'].fillna(0)

        # 4. 方向变化
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - \
            np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)

        bearing = np.degrees(np.arctan2(y, x))
        bearing = (bearing + 360) % 360
        bearing.iloc[0] = 0.0

        df['bearing'] = bearing
        df['bearing_change'] = df['bearing'].diff().abs().fillna(0)
        df['bearing_change'] = np.where(
            df['bearing_change'] > 180,
            360 - df['bearing_change'],
            df['bearing_change']
        )

        # 5. 累积特征
        df['total_distance'] = df['distance'].cumsum()
        df['total_time'] = df['time_diff'].cumsum()

        return df

    def _segment_trajectory(self, trajectory: pd.DataFrame,
                            labels: pd.DataFrame,
                            user_id: str,
                            trajectory_id: str) -> List[Dict]:
        """按标签分割轨迹"""
        segments = []

        for _, label_row in labels.iterrows():
            start_time = label_row['Start Time']
            end_time = label_row['End Time']
            raw_mode = str(label_row['Transportation Mode']).lower().strip()

            # 标签归一化
            mode = self.label_mapping.get(raw_mode, raw_mode.capitalize())

            # 时间范围过滤
            mask = (
                    (trajectory['datetime'] >= start_time) &
                    (trajectory['datetime'] <= end_time)
            )
            segment = trajectory[mask].copy()

            if len(segment) < 10:  # 最小长度过滤
                continue

            # 构建段信息
            segment_info = {
                'user_id': user_id,
                'trajectory_id': trajectory_id,
                'segment_id': f"{user_id}_{trajectory_id}_{start_time.strftime('%Y%m%d%H%M%S')}",
                'label': mode,
                'start_time': start_time,
                'end_time': end_time,
                'length': len(segment),
                'raw_points': segment.reset_index(drop=True),  # 包含9维特征
                'datetime_series': segment['datetime'].reset_index(drop=True)
            }

            segments.append(segment_info)

        return segments

    def _print_statistics(self, segments: List[Dict]):
        """打印统计信息"""
        from collections import Counter

        print("\n" + "=" * 80)
        print("数据统计")
        print("=" * 80)

        # 标签分布
        labels = [seg['label'] for seg in segments]
        label_counts = Counter(labels)

        print(f"\n总轨迹段数: {len(segments)}")
        print(f"\n标签分布:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label:15s}: {count:6d} ({count / len(segments) * 100:.2f}%)")

        # 长度统计
        lengths = [seg['length'] for seg in segments]
        print(f"\n轨迹段长度:")
        print(f"  最小: {min(lengths)}")
        print(f"  最大: {max(lengths)}")
        print(f"  平均: {np.mean(lengths):.1f}")
        print(f"  中位数: {np.median(lengths):.1f}")

    def save_to_cache(self, segments: List[Dict], cache_path: str):
        """保存到缓存"""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        with open(cache_path, 'wb') as f:
            pickle.dump(segments, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"\n✅ 数据已保存到: {cache_path}")
        print(f"   文件大小: {os.path.getsize(cache_path) / 1024 / 1024:.2f} MB")

    @staticmethod
    def load_from_cache(cache_path: str) -> List[Dict]:
        """从缓存加载"""
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"缓存文件不存在: {cache_path}")

        with open(cache_path, 'rb') as f:
            segments = pickle.load(f)

        print(f"\n✅ 从缓存加载: {len(segments)} 个轨迹段")
        return segments

    @staticmethod
    def compute_segment_stats(features: np.ndarray) -> np.ndarray:
        """
        计算轨迹段的统计特征（段级，与序列长度无关）。

        基于已计算好的9维特征矩阵，提取能区分交通方式的统计量。

        Args:
            features: (N, 9) 特征矩阵，列顺序：
                [0]latitude [1]longitude [2]speed [3]acceleration
                [4]bearing_change [5]distance [6]time_diff
                [7]total_distance [8]total_time

        Returns:
            stats: (18,) 统计特征向量

        特征说明（共18维）：
            [0]  speed_mean          速度均值
            [1]  speed_std           速度标准差
            [2]  speed_max           速度最大值
            [3]  speed_cv            速度变异系数 = std/mean，区分 Bus/Car 关键特征
            [4]  accel_mean          加速度均值（绝对值）
            [5]  accel_std           加速度标准差
            [6]  accel_max           加速度最大值（绝对值）
            [7]  bearing_change_mean 方向变化均值，区分直线/曲线行驶
            [8]  bearing_change_std  方向变化标准差
            [9]  stop_ratio          停止比例（速度<0.5m/s 的点占比），Bus特征
            [10] high_speed_ratio    高速比例（速度>15m/s 的点占比），Train/Airplane特征
            [11] linearity           直线度 = 直线距离/累计距离，Walk特征
            [12] total_distance      总行驶距离
            [13] total_time          总行驶时间
            [14] avg_segment_speed   平均段速度 = 总距离/总时间
            [15] speed_entropy       速度分布熵，衡量速度变化规律性
            [16] accel_sign_changes  加速度符号变化次数（归一化），Bus走走停停特征
            [17] max_sustained_speed 最大持续高速时长比例，区分 Subway/Train
        """
        eps = 1e-8
        N = len(features)

        if N == 0:
            return np.zeros(18, dtype=np.float32)

        speed = features[:, 2].clip(0)
        accel = np.abs(features[:, 3])
        bearing = np.abs(features[:, 4])
        distance = features[:, 5].clip(0)
        time_diff = features[:, 6].clip(0)
        total_dist = features[-1, 7] if features[-1, 7] > 0 else distance.sum()
        total_time = features[-1, 8] if features[-1, 8] > 0 else time_diff.sum()

        speed_mean = float(np.mean(speed))
        speed_std  = float(np.std(speed))
        speed_max  = float(np.max(speed))

        speed_cv = speed_std / (speed_mean + eps)

        accel_mean = float(np.mean(accel))
        accel_std  = float(np.std(accel))
        accel_max  = float(np.max(accel))

        bearing_mean = float(np.mean(bearing))
        bearing_std  = float(np.std(bearing))

        stop_ratio = float(np.mean(speed < 0.5))

        high_speed_ratio = float(np.mean(speed > 15.0))

        if total_dist > eps:
            lat_start, lon_start = features[0, 0], features[0, 1]
            lat_end,   lon_end   = features[-1, 0], features[-1, 1]
            straight_dist = np.sqrt(
                ((lat_end - lat_start) * 111300) ** 2 +
                ((lon_end - lon_start) * 111300 * np.cos(np.radians(lat_start))) ** 2
            )
            linearity = float(min(straight_dist / (total_dist + eps), 1.0))
        else:
            linearity = 0.0

        total_distance    = float(total_dist)
        total_time_val    = float(total_time)
        avg_segment_speed = float(total_dist / (total_time + eps))

        if speed_max > eps:
            hist, _ = np.histogram(speed, bins=10, range=(0, speed_max + eps))
            hist = hist / (hist.sum() + eps)
            hist = hist[hist > 0]
            speed_entropy = float(-np.sum(hist * np.log(hist + eps)))
        else:
            speed_entropy = 0.0

        if N > 1:
            raw_accel = features[:, 3]
            sign_changes = np.sum(np.diff(np.sign(raw_accel)) != 0)
            accel_sign_changes = float(sign_changes / (N - 1))
        else:
            accel_sign_changes = 0.0

        high_speed_mask = speed > 10.0
        max_sustained = 0
        current_run = 0
        for v in high_speed_mask:
            if v:
                current_run += 1
                max_sustained = max(max_sustained, current_run)
            else:
                current_run = 0
        max_sustained_speed = float(max_sustained / N)

        stats = np.array([
            speed_mean, speed_std, speed_max, speed_cv,
            accel_mean, accel_std, accel_max,
            bearing_mean, bearing_std,
            stop_ratio, high_speed_ratio,
            linearity,
            total_distance, total_time_val, avg_segment_speed,
            speed_entropy,
            accel_sign_changes,
            max_sustained_speed
        ], dtype=np.float32)

        stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

        return stats

    @staticmethod
    def augment_segment(features: np.ndarray,
                     label: str,
                     augment_types: list = None) -> list:
        """
        对单个轨迹段做数据增强，返回增强后的样本列表。

        用途：对少数类（Subway/Airplane）进行过采样，
              增强样本多样性而不改变语义。

        Args:
            features: (N, 27) 原始特征数组
            label: 类别标签字符串
            augment_types: 增强类型列表，可选：
                'time_warp'   : 时间轴随机拉伸/压缩（模拟不同速度）
                'noise'       : 添加高斯噪声（模拟传感器误差）
                'flip'        : 时间轴翻转（模拟反向行驶）
                'scale'       : 速度/加速度维度随机缩放

        Returns:
            augmented: List of (features_aug, label)，不含原始样本
        """
        if augment_types is None:
            augment_types = ['noise', 'scale']

        augmented = []
        N, D = features.shape

        for aug_type in augment_types:
            feat = features.copy()

            if aug_type == 'time_warp':
                # 在时间轴上随机插值，模拟不同采样率
                warp_factor = np.random.uniform(0.8, 1.2)
                new_len = max(int(N * warp_factor), 10)
                indices = np.linspace(0, N - 1, new_len)
                feat_warped = np.zeros((new_len, D), dtype=np.float32)
                for d in range(D):
                    feat_warped[:, d] = np.interp(indices, np.arange(N), feat[:, d])
                # 重新规范化到原始长度
                indices_back = np.linspace(0, new_len - 1, N)
                for d in range(D):
                    feat[:, d] = np.interp(indices_back, np.arange(new_len), feat_warped[:, d])

            elif aug_type == 'noise':
                # 只对运动特征维度（speed/accel/bearing等）加噪声，不动经纬度
                noise_cols = [2, 3, 4, 5, 6]  # speed, accel, bearing, distance, time_diff
                noise = np.random.normal(0, 0.05, (N, len(noise_cols))).astype(np.float32)
                feat[:, noise_cols] += noise

            elif aug_type == 'flip':
                # 时间轴翻转：模拟反向轨迹
                feat = feat[::-1].copy()

            elif aug_type == 'scale':
                # 速度和加速度维度随机缩放（模拟不同交通状况）
                scale_cols = [2, 3]  # speed, acceleration
                scale = np.random.uniform(0.85, 1.15)
                feat[:, scale_cols] *= scale

            augmented.append((feat.astype(np.float32), label))

        return augmented

    @staticmethod
    def oversample_minority_classes(
            segments: list,
            target_ratio: float = 0.3,
            minority_classes: list = None) -> list:
        """
        对少数类进行过采样，使其样本量达到多数类的 target_ratio 倍。

        Args:
            segments: List of (features, label) 或 (features, datetime, label) 或 (traj, weather, stats, label)
            target_ratio: 少数类目标样本量 = 多数类样本量 * target_ratio
            minority_classes: 指定需要过采样的类别，None 则自动识别

        Returns:
            augmented_segments: 原始 + 增强样本的混合列表（已随机打乱）
        """
        import random

        # 判断 segments 格式：(feat, label), (feat, dt, label), (traj, weather, stats, label)
        num_elements = len(segments[0])
        has_datetime = num_elements == 3
        has_weather = num_elements == 4

        if has_datetime:
            by_class = {}
            for traj, dt, label in segments:
                if label not in by_class:
                    by_class[label] = []
                by_class[label].append((traj, dt, label))
        elif has_weather:
            by_class = {}
            for traj, weather, stats, label in segments:
                if label not in by_class:
                    by_class[label] = []
                by_class[label].append((traj, weather, stats, label))
        else:
            by_class = {}
            for traj, label in segments:
                if label not in by_class:
                    by_class[label] = []
                by_class[label].append((traj, label))

        # 自动识别少数类
        if minority_classes is None:
            counts = {label: len(samples) for label, samples in by_class.items()}
            max_count = max(counts.values())
            minority_classes = [label for label, count in counts.items()
                            if count < max_count * target_ratio]

        # 对少数类进行过采样
        augmented_segments = list(segments)

        for label in minority_classes:
            samples = by_class[label]
            current_count = len(samples)
            target_count = int(max(len(s) for s in by_class.values()) * target_ratio)

            if current_count >= target_count:
                continue

            # 需要增强的样本数
            needed = target_count - current_count

            # 对每个样本进行多次增强
            for i in range(needed):
                original_sample = samples[i % current_count]

                if has_datetime:
                    traj, dt, lbl = original_sample
                    traj_aug = BaseGeoLifePreprocessor.augment_segment(traj, lbl)
                    for aug_traj, _ in traj_aug:
                        augmented_segments.append((aug_traj, dt, lbl))
                elif has_weather:
                    traj, weather, stats, lbl = original_sample
                    traj_aug = BaseGeoLifePreprocessor.augment_segment(traj, lbl)
                    for aug_traj, _ in traj_aug:
                        augmented_segments.append((aug_traj, weather, stats, lbl))
                else:
                    traj, lbl = original_sample
                    traj_aug = BaseGeoLifePreprocessor.augment_segment(traj, lbl)
                    for aug_traj, _ in traj_aug:
                        augmented_segments.append((aug_traj, lbl))

        # 随机打乱
        random.shuffle(augmented_segments)

        return augmented_segments


def normalize_datetime_series(datetime_series, target_length: int):
    """将时间序列统一到 target_length（供prepare_data.py使用）"""
    current_len = len(datetime_series)
    if current_len == target_length:
        return datetime_series.reset_index(drop=True)
    elif current_len > target_length:
        indices = np.linspace(0, current_len - 1, target_length, dtype=int)
        return datetime_series.iloc[indices].reset_index(drop=True)
    else:
        last_time = datetime_series.iloc[-1]
        padding = pd.Series([last_time] * (target_length - current_len))
        return pd.concat([datetime_series.reset_index(drop=True), padding], ignore_index=True)


def print_cleaned_stats(cleaned_path: str, data=None):
    """打印标签分布统计（供prepare_data.py使用）"""
    from collections import Counter
    if data is None:
        with open(cleaned_path, 'rb') as f:
            data = pickle.load(f)

    labels = [item[3] for item in data]
    counts = Counter(labels)
    total = len(data)
    print(f"\n📊 标签分布 (共 {total} 条):")
    for label in sorted(counts.keys()):
        pct = counts[label] / total * 100
        print(f"   {label:15s}: {counts[label]:5d} ({pct:5.1f}%)")

# ========== 加载带地理信息的原始段（用于生成预测CSV） ==========
def load_segments_with_geo(self, max_users=None, min_segment_length=10):
    """
    与 process_all_users 类似，但每个 segment 中增加 'geo_points' 字段，
    包含 timestamp, latitude, longitude（原始未归一化）
    """
    users = self._get_all_users()
    if max_users:
        users = users[:max_users]

    all_segments = []
    for user_id in tqdm(users, desc="[加载地理信息]"):
        labels = self._load_labels(user_id)
        if labels.empty:
            continue
        trajectory_dir = os.path.join(self.geolife_root, f"Data/{user_id}/Trajectory")
        if not os.path.exists(trajectory_dir):
            continue
        for traj_file in os.listdir(trajectory_dir):
            if not traj_file.endswith('.plt'):
                continue
            traj_path = os.path.join(trajectory_dir, traj_file)
            trajectory_id = traj_file.replace('.plt', '')
            try:
                trajectory = self._load_and_compute_features(traj_path)
                if trajectory.empty or len(trajectory) < min_segment_length:
                    continue
                segments = self._segment_trajectory_with_geo(
                    trajectory, labels, user_id, trajectory_id
                )
                all_segments.extend(segments)
            except Exception as e:
                warnings.warn(f"处理失败 {traj_path}: {e}")
                continue
    return all_segments

def _segment_trajectory_with_geo(self, trajectory, labels, user_id, trajectory_id):
    """与 _segment_trajectory 相同，但额外保存原始经纬度和时间"""
    segments = []
    for _, label_row in labels.iterrows():
        start_time = label_row['Start Time']
        end_time = label_row['End Time']
        raw_mode = str(label_row['Transportation Mode']).lower().strip()
        mode = self.label_mapping.get(raw_mode, raw_mode.capitalize())

        mask = (trajectory['datetime'] >= start_time) & (trajectory['datetime'] <= end_time)
        segment = trajectory[mask].copy()
        if len(segment) < 10:
            continue

        # 提取地理点信息（原始经纬度和时间）
        geo_points = segment[['datetime', 'latitude', 'longitude']].copy()
        geo_points.rename(columns={'datetime': 'timestamp'}, inplace=True)
        # 确保时间戳字符串格式
        geo_points['timestamp'] = geo_points['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        segment_info = {
            'user_id': user_id,
            'trajectory_id': trajectory_id,
            'segment_id': f"{user_id}_{trajectory_id}_{start_time.strftime('%Y%m%d%H%M%S')}",
            'label': mode,
            'start_time': start_time,
            'end_time': end_time,
            'length': len(segment),
            'raw_points': segment.reset_index(drop=True),   # 含9维特征
            'geo_points': geo_points,                       # 新增
            'datetime_series': segment['datetime'].reset_index(drop=True)
        }
        segments.append(segment_info)
    return segments


class BaseGeoLifeDataLoader:
    pass