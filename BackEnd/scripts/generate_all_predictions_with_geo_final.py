#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 Exp1~Exp4 生成带地理信息的预测 CSV（最终修正版）
支持 Exp2 的 21 维点级特征、Exp3/Exp4 的双输入（21维轨迹+空间 + 10维天气）
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from collections import Counter

# ==================== 配置（请修改为您的实际路径）====================
BACKEND_DIR = r"D:\TransportationSystem3.0\BackEnd"
sys.path.insert(0, BACKEND_DIR)

GEOLIFE_ROOT = r"D:\TransportationSystem3.0\BackEnd\data\Geolife Trajectories 1.3"
OSM_GEOJSON_PATH = r"D:\TransportationSystem3.0\BackEnd\data\exp2.geojson"
WEATHER_CSV_PATH = r"D:\TransportationSystem3.0\BackEnd\data\beijing_weather_daily_2007_2012.csv"

MODEL_PATHS = {
    "exp1": r"D:/TransportationSystem3.0/BackEnd/exp1/checkpoints/exp1_model.pth",
    "exp2": r"D:/TransportationSystem3.0/BackEnd/exp2/checkpoints/exp2_model.pth",
    "exp3": r"D:/TransportationSystem3.0/BackEnd/exp3/checkpoints/exp3_model.pth",
    "exp4": r"D:/TransportationSystem3.0/BackEnd/exp4/checkpoints/exp4_model.pth",
}

# Exp4 复用 Exp3 的模型类
MODEL_CLASS_MAP = {
    "exp1": ("exp1.src.model", "TransportationModeClassifier"),
    "exp2": ("exp2.src.model", "TransportationModeClassifier"),
    "exp3": ("exp3.src.model_weather", "TransportationModeClassifierWithWeather"),
    "exp4": ("exp3.src.model_weather", "TransportationModeClassifierWithWeather"),
}

OUTPUT_DIR = r"D:/TransportationSystem3.0/BackEnd/data/predictions_with_geo"
MIN_SEGMENT_LENGTH = 10
RANDOM_STATE = 42
TEST_SIZE = 0.2

LABEL_MAPPING = {
    'taxi': 'Car & taxi', 'car': 'Car & taxi', 'drive': 'Car & taxi',
    'bus': 'Bus', 'walk': 'Walk', 'bike': 'Bike', 'train': 'Train',
    'subway': 'Subway', 'railway': 'Train', 'airplane': 'Airplane'
}

FEATURE_COLUMNS = [
    'latitude', 'longitude', 'speed', 'acceleration',
    'bearing_change', 'distance', 'time_diff',
    'total_distance', 'total_time'
]

# ==================== 1. 从原始 GeoLife 加载所有段（含地理点）====================
class RawSegmentLoader:
    def __init__(self, root_path, label_mapping, min_len=10):
        self.root_path = root_path
        self.label_mapping = label_mapping
        self.min_len = min_len

    def get_all_users(self):
        data_path = os.path.join(self.root_path, "Data")
        users = []
        for item in os.listdir(data_path):
            if os.path.isdir(os.path.join(data_path, item)) and item.isdigit():
                users.append(item)
        return sorted(users)

    def load_labels(self, user_id):
        path = os.path.join(self.root_path, f"Data/{user_id}/labels.txt")
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path, sep='\t')
        df['Start Time'] = pd.to_datetime(df['Start Time'])
        df['End Time'] = pd.to_datetime(df['End Time'])
        return df

    def load_trajectory_features(self, file_path):
        try:
            df = pd.read_csv(file_path, skiprows=6, header=None)
        except:
            return pd.DataFrame()
        num_cols = df.shape[1]
        if num_cols == 7:
            df.columns = ['latitude', 'longitude', 'reserved', 'altitude', 'date_days', 'date', 'time']
        elif num_cols == 6:
            df.columns = ['latitude', 'longitude', 'altitude', 'date_days', 'date', 'time']
            df.insert(2, 'reserved', 0)
        else:
            return pd.DataFrame()
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S')
        df = df.sort_values('datetime').reset_index(drop=True)
        valid = (df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))
        df = df[valid].reset_index(drop=True)
        if len(df) < 2:
            return pd.DataFrame()
        # 计算9维特征
        df['time_diff'] = df['datetime'].diff().dt.total_seconds().fillna(0)
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
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad)*np.cos(lat2_rad)*np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distances = 6371000 * c
        distances.iloc[0] = 0.0
        df['distance'] = distances
        time_diff_safe = df['time_diff'].replace(0, 1e-6)
        df['speed'] = df['distance'] / time_diff_safe
        df['acceleration'] = df['speed'].diff() / time_diff_safe
        df['acceleration'] = df['acceleration'].fillna(0)
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = np.cos(lat1_rad)*np.sin(lat2_rad) - np.sin(lat1_rad)*np.cos(lat2_rad)*np.cos(dlon)
        bearing = np.degrees(np.arctan2(y, x))
        bearing = (bearing + 360) % 360
        bearing.iloc[0] = 0.0
        df['bearing'] = bearing
        df['bearing_change'] = df['bearing'].diff().abs().fillna(0)
        df['bearing_change'] = np.where(df['bearing_change'] > 180, 360 - df['bearing_change'], df['bearing_change'])
        df['total_distance'] = df['distance'].cumsum()
        df['total_time'] = df['time_diff'].cumsum()
        return df[['datetime', 'latitude', 'longitude', 'speed', 'acceleration',
                   'bearing_change', 'distance', 'time_diff', 'total_distance', 'total_time']]

    def compute_segment_stats(self, traj_9):
        eps = 1e-8
        N = len(traj_9)
        if N == 0:
            return np.zeros(18, dtype=np.float32)
        speed = traj_9[:, 2].clip(0)
        accel = np.abs(traj_9[:, 3])
        bearing = np.abs(traj_9[:, 4])
        distance = traj_9[:, 5].clip(0)
        time_diff = traj_9[:, 6].clip(0)
        total_dist = traj_9[-1, 7] if traj_9[-1, 7] > 0 else distance.sum()
        total_time = traj_9[-1, 8] if traj_9[-1, 8] > 0 else time_diff.sum()
        speed_mean = float(np.mean(speed))
        speed_std = float(np.std(speed))
        speed_max = float(np.max(speed))
        speed_cv = speed_std / (speed_mean + eps)
        accel_mean = float(np.mean(accel))
        accel_std = float(np.std(accel))
        accel_max = float(np.max(accel))
        bearing_mean = float(np.mean(bearing))
        bearing_std = float(np.std(bearing))
        stop_ratio = float(np.mean(speed < 0.5))
        high_speed_ratio = float(np.mean(speed > 15.0))
        if total_dist > eps:
            lat_start, lon_start = traj_9[0, 0], traj_9[0, 1]
            lat_end, lon_end = traj_9[-1, 0], traj_9[-1, 1]
            straight_dist = np.sqrt(((lat_end - lat_start)*111300)**2 + ((lon_end - lon_start)*111300*np.cos(np.radians(lat_start)))**2)
            linearity = float(min(straight_dist/(total_dist+eps), 1.0))
        else:
            linearity = 0.0
        total_distance = float(total_dist)
        total_time_val = float(total_time)
        avg_segment_speed = float(total_dist / (total_time + eps))
        if speed_max > eps:
            hist, _ = np.histogram(speed, bins=10, range=(0, speed_max+eps))
            hist = hist/(hist.sum()+eps)
            hist = hist[hist>0]
            speed_entropy = float(-np.sum(hist * np.log(hist+eps)))
        else:
            speed_entropy = 0.0
        if N > 1:
            raw_accel = traj_9[:, 3]
            sign_changes = np.sum(np.diff(np.sign(raw_accel)) != 0)
            accel_sign_changes = float(sign_changes/(N-1))
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
        return np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

    def load_all_segments(self):
        users = self.get_all_users()
        all_segments = []
        for user_id in tqdm(users, desc="加载原始轨迹段"):
            labels_df = self.load_labels(user_id)
            if labels_df.empty:
                continue
            traj_dir = os.path.join(self.root_path, f"Data/{user_id}/Trajectory")
            if not os.path.exists(traj_dir):
                continue
            for traj_file in os.listdir(traj_dir):
                if not traj_file.endswith('.plt'):
                    continue
                traj_path = os.path.join(traj_dir, traj_file)
                trajectory_id = traj_file.replace('.plt', '')
                traj_df = self.load_trajectory_features(traj_path)
                if traj_df.empty or len(traj_df) < self.min_len:
                    continue
                for _, row in labels_df.iterrows():
                    start = row['Start Time']
                    end = row['End Time']
                    raw_mode = str(row['Transportation Mode']).lower().strip()
                    mode = self.label_mapping.get(raw_mode, raw_mode.capitalize())
                    mask = (traj_df['datetime'] >= start) & (traj_df['datetime'] <= end)
                    seg_df = traj_df[mask].copy()
                    if len(seg_df) < self.min_len:
                        continue
                    traj_9 = seg_df[FEATURE_COLUMNS].values.astype(np.float32)
                    stats_18 = self.compute_segment_stats(traj_9)
                    geo_points = seg_df[['datetime', 'latitude', 'longitude']].copy()
                    geo_points.rename(columns={'datetime': 'timestamp'}, inplace=True)
                    geo_points['timestamp'] = geo_points['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    segment_id = f"{user_id}_{trajectory_id}_{start.strftime('%Y%m%d%H%M%S')}"
                    all_segments.append({
                        'segment_id': segment_id,
                        'traj_9': traj_9,
                        'stats_18': stats_18,
                        'label': mode,
                        'geo_points': geo_points,
                        'start_time': start,
                        'end_time': end,
                    })
        print(f"✅ 共加载 {len(all_segments)} 个轨迹段")
        return all_segments

# ==================== 2. 测试集划分 ====================
def split_test_set(segments, test_size=0.2, random_state=42):
    labels = [seg['label'] for seg in segments]
    label_counts = Counter(labels)
    rare_labels = [lab for lab, cnt in label_counts.items() if cnt < 2]
    if rare_labels:
        print(f"⚠️ 移除稀有类别: {rare_labels}")
        segments = [seg for seg in segments if seg['label'] not in rare_labels]
        labels = [seg['label'] for seg in segments]
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    indices = np.arange(len(segments))
    _, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=y_enc)
    test_segments = [segments[i] for i in test_indices]
    print(f"✅ 测试集大小: {len(test_segments)} / {len(segments)}")
    return test_segments, le

# ==================== 3. 特征构建器（支持 Exp2 和 Exp3/4）====================
class FeatureBuilder:
    def __init__(self, osm_extractor=None, weather_processor=None):
        self.osm_extractor = osm_extractor
        self.weather_processor = weather_processor

    def build_exp1(self, traj_9):
        return traj_9  # (N,9)

    def build_exp2(self, traj_9):
        spatial = self.osm_extractor.extract_spatial_features(traj_9)  # (N,12)
        return np.concatenate([traj_9, spatial], axis=1)  # (N,21)

    def build_exp3_features(self, traj_9, datetime_series):
        spatial = self.osm_extractor.extract_spatial_features(traj_9)  # (N,12)
        traj_spatial = np.concatenate([traj_9, spatial], axis=1)       # (N,21)
        weather = self.weather_processor.get_weather_features_for_trajectory(datetime_series)  # (N,10)
        return traj_spatial, weather

# ==================== 4. 加载模型（从 checkpoint 读取配置）====================
def load_model(exp_name, device='cpu'):
    model_path = MODEL_PATHS[exp_name]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # 从 state_dict 中推断输入维度
    state_dict = ckpt['model_state_dict']
    # 查找 'encoders.0.weight_ih_l0' 的形状，第二维即为输入维度
    for key in state_dict.keys():
        if 'encoders.0.weight_ih_l0' in key:
            input_dim = state_dict[key].shape[1]
            print(f"   从 checkpoint 推断输入维度: {input_dim}")
            break
    else:
        input_dim = 9  # fallback

    # 获取其他配置
    config = ckpt.get('model_config', {})
    segment_stats_dim = config.get('segment_stats_dim', 18)
    hidden_dim = config.get('hidden_dim', 128)
    num_layers = config.get('num_layers', 2)
    num_classes = config.get('num_classes', 7)
    dropout = config.get('dropout', 0.3)

    # 对于 Exp3/Exp4 还需要 weather 维度
    weather_dim = 10  # 默认
    if exp_name in ['exp3', 'exp4']:
        # 尝试从 state_dict 中推断 weather 输入维度（如果有第二个编码器）
        for key in state_dict.keys():
            if 'encoders.1.weight_ih_l0' in key:
                weather_dim = state_dict[key].shape[1]
                print(f"   从 checkpoint 推断天气维度: {weather_dim}")
                break

    # 导入模型类
    module_name, class_name = MODEL_CLASS_MAP[exp_name]
    mod = __import__(module_name, fromlist=[''])
    ModelClass = getattr(mod, class_name)

    # 实例化模型
    if exp_name == 'exp1':
        model = ModelClass(
            trajectory_feature_dim=input_dim,
            segment_stats_dim=segment_stats_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif exp_name == 'exp2':
        model = ModelClass(
            trajectory_feature_dim=input_dim,
            segment_stats_dim=segment_stats_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif exp_name in ['exp3', 'exp4']:
        model = ModelClass(
            trajectory_feature_dim=input_dim,
            weather_feature_dim=weather_dim,
            segment_stats_dim=segment_stats_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown exp: {exp_name}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    label_encoder = ckpt['label_encoder']
    norm_params = ckpt.get('norm_params', {})
    return model, label_encoder, norm_params

# ==================== 5. 预测并保存 ====================
def predict_and_save(exp_name, test_segments, model, label_encoder, norm_params, feature_builder, device, output_dir):
    # 准备存储结果
    y_true_str = [seg['label'] for seg in test_segments]
    pred_ids = []
    confs = []
    # 根据实验类型进行预测
    for seg in tqdm(test_segments, desc=f"{exp_name} 预测"):
        traj_9 = seg['traj_9'].copy()
        stats = seg['stats_18'].copy()
        if exp_name == 'exp1':
            point_feat = feature_builder.build_exp1(traj_9)  # (N,9)
            # 归一化
            traj_mean = norm_params.get('traj_mean')
            traj_std = norm_params.get('traj_std')
            if traj_mean is not None and point_feat.shape[1] == traj_mean.shape[0]:
                point_feat = (point_feat - traj_mean) / traj_std
            else:
                print(f"警告: Exp1 特征维度 {point_feat.shape[1]} 与归一化参数不匹配，跳过归一化")
            stats_mean = norm_params.get('stats_mean')
            stats_std = norm_params.get('stats_std')
            if stats_mean is not None and stats.shape[0] == stats_mean.shape[0]:
                stats = (stats - stats_mean) / stats_std
            # 转为 tensor
            point_t = torch.FloatTensor(point_feat).unsqueeze(0).to(device)  # (1, N, 9)
            stats_t = torch.FloatTensor(stats).unsqueeze(0).to(device)       # (1, 18)
            with torch.no_grad():
                logits = model(point_t, segment_stats=stats_t)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
        elif exp_name == 'exp2':
            point_feat = feature_builder.build_exp2(traj_9)  # (N,21)
            # 归一化
            traj_mean = norm_params.get('traj_mean')
            traj_std = norm_params.get('traj_std')
            if traj_mean is not None and point_feat.shape[1] == traj_mean.shape[0]:
                point_feat = (point_feat - traj_mean) / traj_std
            else:
                print(f"警告: Exp2 特征维度 {point_feat.shape[1]} 与归一化参数不匹配，跳过归一化")
            stats_mean = norm_params.get('stats_mean')
            stats_std = norm_params.get('stats_std')
            if stats_mean is not None and stats.shape[0] == stats_mean.shape[0]:
                stats = (stats - stats_mean) / stats_std
            point_t = torch.FloatTensor(point_feat).unsqueeze(0).to(device)
            stats_t = torch.FloatTensor(stats).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(point_t, segment_stats=stats_t)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
        elif exp_name in ['exp3', 'exp4']:
            datetime_series = pd.to_datetime(seg['geo_points']['timestamp'])
            traj_spatial, weather = feature_builder.build_exp3_features(traj_9, datetime_series)  # (N,21), (N,10)
            # 归一化（可能分别存储）
            traj_mean = norm_params.get('traj_mean')
            traj_std = norm_params.get('traj_std')
            if traj_mean is not None and traj_spatial.shape[1] == traj_mean.shape[0]:
                traj_spatial = (traj_spatial - traj_mean) / traj_std
            else:
                print(f"警告: Exp3 轨迹+空间特征维度 {traj_spatial.shape[1]} 与归一化参数不匹配，跳过归一化")
            weather_mean = norm_params.get('weather_mean')
            weather_std = norm_params.get('weather_std')
            if weather_mean is not None and weather.shape[1] == weather_mean.shape[0]:
                weather = (weather - weather_mean) / weather_std
            else:
                print(f"警告: Exp3 天气特征维度 {weather.shape[1]} 与归一化参数不匹配，跳过归一化")
            stats_mean = norm_params.get('stats_mean')
            stats_std = norm_params.get('stats_std')
            if stats_mean is not None and stats.shape[0] == stats_mean.shape[0]:
                stats = (stats - stats_mean) / stats_std
            traj_t = torch.FloatTensor(traj_spatial).unsqueeze(0).to(device)   # (1, N, 21)
            weather_t = torch.FloatTensor(weather).unsqueeze(0).to(device)      # (1, N, 10)
            stats_t = torch.FloatTensor(stats).unsqueeze(0).to(device)          # (1, 18)
            with torch.no_grad():
                logits = model(traj_t, weather_t, segment_stats=stats_t)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
        else:
            raise ValueError(f"Unknown exp: {exp_name}")
        pred_ids.append(pred.item())
        confs.append(conf.item())
    # 转换标签
    pred_labels = label_encoder.inverse_transform(pred_ids)
    # 保存样本级
    sample_df = pd.DataFrame({
        'segment_id': [seg['segment_id'] for seg in test_segments],
        'true_label': y_true_str,
        'pred_label': pred_labels,
        'confidence': confs,
    })
    sample_path = os.path.join(output_dir, f'predictions_with_geo_{exp_name}.csv')
    sample_df.to_csv(sample_path, index=False, encoding='utf-8-sig')
    print(f"✅ 样本级保存至: {sample_path}")
    # 保存点级
    points = []
    for seg, tl, pl, cf in zip(test_segments, y_true_str, pred_labels, confs):
        geo = seg['geo_points'].copy()
        geo['segment_id'] = seg['segment_id']
        geo['true_label'] = tl
        geo['pred_label'] = pl
        geo['confidence'] = cf
        points.append(geo)
    points_df = pd.concat(points, ignore_index=True)
    points_path = os.path.join(output_dir, f'points_with_geo_{exp_name}.csv')
    points_df.to_csv(points_path, index=False, encoding='utf-8-sig')
    print(f"✅ 点级保存至: {points_path} (共 {len(points_df)} 个点)")

# ==================== 6. 主函数 ====================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 加载所有原始段
    loader = RawSegmentLoader(GEOLIFE_ROOT, LABEL_MAPPING, MIN_SEGMENT_LENGTH)
    all_segments = loader.load_all_segments()
    if not all_segments:
        print("未加载到任何轨迹段，请检查 GeoLife 路径")
        return

    # 划分测试集
    test_segments, global_label_encoder = split_test_set(all_segments, TEST_SIZE, RANDOM_STATE)

    # 延迟初始化 OSM 和天气处理器
    osm_extractor = None
    weather_processor = None

    for exp_name in ['exp1', 'exp2', 'exp3', 'exp4']:
        print(f"\n{'='*50}\n处理 {exp_name.upper()}\n{'='*50}")
        if not os.path.exists(MODEL_PATHS[exp_name]):
            print(f"❌ 模型文件不存在: {MODEL_PATHS[exp_name]}，跳过")
            continue

        # 初始化特征构建器
        if exp_name == 'exp1':
            feature_builder = FeatureBuilder()
        elif exp_name == 'exp2':
            if osm_extractor is None:
                print("初始化 OSM 空间特征提取器...")
                from exp2.src.osm_feature_extractor import OsmSpatialExtractor
                from exp2.src.data_preprocessing import OSMDataLoader
                osm_loader = OSMDataLoader(OSM_GEOJSON_PATH)
                osm_data = osm_loader.load_osm_data()
                road_network = osm_loader.extract_road_network(osm_data)
                pois = osm_loader.extract_pois(osm_data)
                osm_extractor = OsmSpatialExtractor()
                osm_extractor.build_from_osm(road_network, pois)
                # 尝试加载缓存
                cache_dir = os.path.join(BACKEND_DIR, 'exp2', 'cache')
                cache_file = os.path.join(cache_dir, 'spatial_grid_cache.pkl')
                if os.path.exists(cache_file):
                    osm_extractor.load_cache(cache_file)
            feature_builder = FeatureBuilder(osm_extractor=osm_extractor)
        elif exp_name in ['exp3', 'exp4']:
            if osm_extractor is None:
                print("初始化 OSM 空间特征提取器...")
                from exp2.src.osm_feature_extractor import OsmSpatialExtractor
                from exp2.src.data_preprocessing import OSMDataLoader
                osm_loader = OSMDataLoader(OSM_GEOJSON_PATH)
                osm_data = osm_loader.load_osm_data()
                road_network = osm_loader.extract_road_network(osm_data)
                pois = osm_loader.extract_pois(osm_data)
                osm_extractor = OsmSpatialExtractor()
                osm_extractor.build_from_osm(road_network, pois)
                cache_dir = os.path.join(BACKEND_DIR, 'exp2', 'cache')
                cache_file = os.path.join(cache_dir, 'spatial_grid_cache.pkl')
                if os.path.exists(cache_file):
                    osm_extractor.load_cache(cache_file)
            if weather_processor is None:
                print("初始化天气处理器...")
                from exp3.src.weather_preprocessing import WeatherDataProcessor
                weather_processor = WeatherDataProcessor(WEATHER_CSV_PATH)
                weather_processor.load_and_process()
            feature_builder = FeatureBuilder(osm_extractor=osm_extractor, weather_processor=weather_processor)

        # 加载模型
        try:
            model, le, norm_params = load_model(exp_name, device)
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            continue

        # 预测并保存
        try:
            predict_and_save(exp_name, test_segments, model, le, norm_params, feature_builder, device, OUTPUT_DIR)
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n🎉 全部完成！")

if __name__ == "__main__":
    main()