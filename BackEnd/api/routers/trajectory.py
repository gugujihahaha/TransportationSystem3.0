from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import io
import sys
import os
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.schemas import (
    TrajectoryPrediction,
    TrajectoryPoint,
    TrajectoryStats,
    TransportMode
)

router = APIRouter()

TRANSPORT_MODES = [
    {"id": "walk", "name": "步行", "color": "#4A90E2", "icon": "walking"},
    {"id": "bike", "name": "自行车", "color": "#52C41A", "icon": "bicycle"},
    {"id": "bus", "name": "公交", "color": "#FA8C16", "icon": "bus"},
    {"id": "car", "name": "汽车/出租", "color": "#F5222D", "icon": "car"},
    {"id": "subway", "name": "地铁", "color": "#722ED1", "icon": "subway"},
    {"id": "train", "name": "火车", "color": "#13C2C2", "icon": "train"},
    {"id": "airplane", "name": "飞机", "color": "#EB2F96", "icon": "plane"},
]

predictors = {}

def load_predictors():
    """加载所有预测器"""
    try:
        from exp1.predict import TrajectoryPredictor
        exp1_path = Path(__file__).parent.parent.parent / "exp1" / "checkpoints" / "exp1_model.pth"
        if exp1_path.exists():
            predictors['exp1'] = TrajectoryPredictor(str(exp1_path))
        else:
            print(f"⚠️ exp1 模型文件不存在: {exp1_path}")
    except Exception as e:
        print(f"⚠️ 加载 exp1 预测器失败: {e}")

load_predictors()

def load_plt_file(content: bytes) -> pd.DataFrame:
    """加载 PLT 格式文件（Geolife 格式）"""
    try:
        lines = content.decode('utf-8').splitlines()
        
        if len(lines) < 7:
            raise ValueError("PLT 文件格式错误：文件过短")
        
        data_lines = lines[6:]
        
        data = []
        for line in data_lines:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    date_str = parts[-2]
                    time_str = parts[-1]
                    timestamp = f"{date_str} {time_str}"
                    data.append({
                        'latitude': lat,
                        'longitude': lon,
                        'timestamp': timestamp
                    })
                except (ValueError, IndexError):
                    continue
        
        if not data:
            raise ValueError("PLT 文件中没有有效的数据点")
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        raise ValueError(f"解析 PLT 文件失败: {str(e)}")

def compute_trajectory_features(df: pd.DataFrame) -> tuple:
    """计算轨迹特征（9维）和统计特征（18维）"""
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    
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
    
    distances = 6371000 * c
    distances.iloc[0] = 0.0
    df['distance'] = distances
    
    time_diff_safe = df['time_diff'].replace(0, 1e-6)
    df['speed'] = df['distance'] / time_diff_safe
    df['acceleration'] = df['speed'].diff() / time_diff_safe
    df['acceleration'] = df['acceleration'].fillna(0)
    
    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
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
    
    df['total_distance'] = df['distance'].cumsum()
    df['total_time'] = df['time_diff'].cumsum()
    
    features = df[[
        'latitude', 'longitude', 'speed', 'acceleration',
        'bearing_change', 'distance', 'time_diff',
        'total_distance', 'total_time'
    ]].values.astype(np.float32)
    
    segment_stats = compute_segment_stats(features)
    
    return features, df, segment_stats

def compute_segment_stats(features: np.ndarray) -> np.ndarray:
    """计算18维统计特征"""
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
        lat_start, lon_start = features[0, 0], features[0, 1]
        lat_end, lon_end = features[-1, 0], features[-1, 1]
        straight_dist = np.sqrt(
            ((lat_end - lat_start) * 111300) ** 2 +
            ((lon_end - lon_start) * 111300 * np.cos(np.radians(lat_start))) ** 2
        )
        linearity = float(min(straight_dist / (total_dist + eps), 1.0))
    else:
        linearity = 0.0
    
    total_distance_val = float(total_dist)
    total_time_val = float(total_time)
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
        total_distance_val, total_time_val, avg_segment_speed,
        speed_entropy,
        accel_sign_changes,
        max_sustained_speed
    ], dtype=np.float32)
    
    stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
    
    return stats

def normalize_sequence_length(features: np.ndarray, target_length: int = 50) -> np.ndarray:
    """统一序列长度"""
    if len(features) == target_length:
        return features
    elif len(features) > target_length:
        indices = np.linspace(0, len(features) - 1, target_length, dtype=int)
        return features[indices]
    else:
        padding = np.zeros((target_length - len(features), 9), dtype=np.float32)
        return np.vstack([features, padding])

def predict_with_model(traj_features: np.ndarray, segment_stats: np.ndarray, model_id: str) -> tuple:
    """使用指定模型进行预测"""
    if model_id not in predictors:
        raise HTTPException(status_code=400, detail=f"模型 {model_id} 未加载或不存在")
    
    predictor = predictors[model_id]
    
    if model_id == 'exp1':
        pred_labels, confidences = predictor.predict(traj_features, segment_stats)
        return pred_labels[0], float(confidences[0])
    
    return None, None

@router.post("/predict", response_model=TrajectoryPrediction)
async def predict_trajectory(file: UploadFile = File(...), model: str = Form('exp1')):
    """上传GPS文件并预测交通方式"""
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            data = pd.read_json(io.StringIO(content.decode('utf-8')))
            df = pd.DataFrame(data)
        elif file.filename.endswith('.plt'):
            df = load_plt_file(content)
        else:
            raise HTTPException(status_code=400, detail="不支持的文件格式，请上传CSV、JSON或PLT")
        
        required_columns = ['latitude', 'longitude', 'timestamp']
        for col in required_columns:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"缺少必需列: {col}")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="轨迹点太少，至少需要10个点")
        
        features, df_with_stats, segment_stats = compute_trajectory_features(df)
        
        features_normalized = normalize_sequence_length(features, 50)
        
        if model in predictors and model == 'exp1':
            predicted_mode, confidence = predict_with_model(features_normalized, segment_stats, model)
        else:
            speed = features_normalized[:, 2].mean()
            max_speed = features_normalized[:, 2].max()
            avg_speed = features_normalized[:, 2].mean()
            
            if max_speed > 50:
                predicted_mode, confidence = "airplane", 0.95
            elif max_speed > 30:
                predicted_mode, confidence = "train", 0.90
            elif avg_speed > 15:
                predicted_mode, confidence = "subway", 0.88
            elif avg_speed > 8:
                predicted_mode, confidence = "car", 0.85
            elif avg_speed > 4:
                predicted_mode, confidence = "bus", 0.82
            elif avg_speed > 2:
                predicted_mode, confidence = "bike", 0.80
            else:
                predicted_mode, confidence = "walk", 0.78
            
            if model != 'exp1':
                predicted_mode = f"{predicted_mode} (规则预测)"
        
        mode_mapping = {
            'Walk': 'walk',
            'Bike': 'bike',
            'Bus': 'bus',
            'Car & taxi': 'car',
            'Subway': 'subway',
            'Train': 'train',
            'Airplane': 'airplane'
        }
        predicted_mode = mode_mapping.get(predicted_mode, predicted_mode.lower())
        
        total_distance = df_with_stats['distance'].sum()
        total_time = df_with_stats['time_diff'].sum()
        avg_speed = total_distance / (total_time + 1e-6)
        max_speed = df_with_stats['speed'].max()
        
        trajectory_id = f"traj_{hash(file.filename) % 1000000}"
        
        points = [
            TrajectoryPoint(
                lat=row['latitude'],
                lng=row['longitude'],
                timestamp=row['timestamp'].isoformat(),
                speed=row['speed']
            )
            for _, row in df_with_stats.iterrows()
        ]
        
        stats = TrajectoryStats(
            distance=float(total_distance),
            duration=float(total_time),
            avg_speed=float(avg_speed),
            max_speed=float(max_speed)
        )
        
        return TrajectoryPrediction(
            trajectory_id=trajectory_id,
            predicted_mode=predicted_mode,
            confidence=confidence,
            points=points,
            stats=stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@router.get("/modes", response_model=List[TransportMode])
async def get_transport_modes():
    """获取所有交通方式的配置"""
    return [
        TransportMode(**mode) for mode in TRANSPORT_MODES
    ]
