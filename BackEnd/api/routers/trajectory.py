import httpx
import pandas as pd
import io
import sys
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel
import numpy as np
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from api.models import User
from api.security import get_current_user

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.schemas import (
    TrajectoryPrediction,
    TrajectoryPoint,
    TrajectoryStats,
    TransportMode
)

router = APIRouter()

# 加载 DeepSeek 配置
CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

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
weather_processor = None
osm_extractor = None

def load_osm_data():
    """加载真实OSM数据（优先使用缓存）"""
    global osm_extractor
    try:
        import pickle
        import sys
        from pathlib import Path
        
        exp2_path = str(Path(__file__).parent.parent.parent / "exp2")
        if exp2_path not in sys.path:
            sys.path.insert(0, exp2_path)
        
        from exp2.src.osm_feature_extractor import OsmSpatialExtractor
        
        spatial_cache_path = Path(__file__).parent.parent.parent / "exp2" / "cache" / "spatial_data.pkl"
        if spatial_cache_path.exists():
            print(f"📋 正在从缓存加载OSM数据: {spatial_cache_path}")
            with open(spatial_cache_path, 'rb') as f:
                osm_extractor = pickle.load(f)
            
            grid_cache_path = Path(__file__).parent.parent.parent / "exp2" / "cache" / "spatial_grid_cache.pkl"
            if grid_cache_path.exists():
                osm_extractor.load_cache(str(grid_cache_path))
            
            print(f"✅ OSM数据从缓存加载成功")
            print(f"   - 道路节点: {len(osm_extractor.road_coords) if osm_extractor.road_coords is not None else 0}")
            print(f"   - POI节点: {len(osm_extractor.poi_coords) if osm_extractor.poi_coords is not None else 0}")
            return
        
        osm_geojson_path = Path(__file__).parent.parent.parent / "data" / "exp2.geojson"
        if osm_geojson_path.exists():
            print(f"📋 正在加载OSM数据: {osm_geojson_path}")
            
            from exp2.src.data_preprocessing import OSMDataLoader
            osm_loader = OSMDataLoader(str(osm_geojson_path))
            osm_data = osm_loader.load_osm_data()
            
            road_network = osm_loader.extract_road_network(osm_data)
            pois = osm_loader.extract_pois(osm_data)
            
            print(f"   -> 加载了 {len(road_network)} 条道路, {len(pois)} 个POI")
            
            osm_extractor = OsmSpatialExtractor()
            osm_extractor.build_from_osm(road_network, pois)
            
            print(f"✅ OSM数据加载成功")
        else:
            print(f"⚠️ OSM数据文件不存在: {osm_geojson_path}")
    except Exception as e:
        print(f"⚠️ 加载OSM数据失败: {e}")
        import traceback
        traceback.print_exc()

def load_weather_data():
    """加载真实天气数据"""
    global weather_processor
    try:
        from exp3.src.weather_preprocessing import WeatherDataProcessor
        weather_csv_path = Path(__file__).parent.parent.parent / "data" / "beijing_weather_daily_2007_2012.csv"
        if weather_csv_path.exists():
            weather_processor = WeatherDataProcessor(str(weather_csv_path))
            weather_processor.load_and_process()
            print(f"✅ 天气数据加载成功")
        else:
            print(f"⚠️ 天气数据文件不存在: {weather_csv_path}")
    except Exception as e:
        print(f"⚠️ 加载天气数据失败: {e}")

def load_predictors():
    """加载所有预测器"""
    import sys
    from pathlib import Path
    
    base_dir = Path(__file__).parent.parent.parent
    
    exp1_path = str(base_dir / "exp1")
    exp2_path = str(base_dir / "exp2")
    exp3_path = str(base_dir / "exp3")
    exp4_path = str(base_dir / "exp4")
    
    for path in [exp1_path, exp2_path, exp3_path, exp4_path]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    try:
        from exp1.predict import TrajectoryPredictor
        model_path = base_dir / "exp1" / "checkpoints" / "exp1_model.pth"
        if model_path.exists():
            predictors['exp1'] = TrajectoryPredictor(str(model_path))
        else:
            print(f"⚠️ exp1 模型文件不存在: {model_path}")
    except Exception as e:
        print(f"⚠️ 加载 exp1 预测器失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        from exp2.predict import TransportationPredictorExp2
        model_path = base_dir / "exp2" / "checkpoints" / "exp2_model.pth"
        if model_path.exists():
            predictors['exp2'] = TransportationPredictorExp2(str(model_path))
        else:
            print(f"⚠️ exp2 模型文件不存在: {model_path}")
    except Exception as e:
        print(f"⚠️ 加载 exp2 预测器失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        from exp3.predict import TransportationPredictorExp3
        model_path = base_dir / "exp3" / "checkpoints" / "exp3_model.pth"
        if model_path.exists():
            predictors['exp3'] = TransportationPredictorExp3(str(model_path))
        else:
            print(f"⚠️ exp3 模型文件不存在: {model_path}")
    except Exception as e:
        print(f"⚠️ 加载 exp3 预测器失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        from exp4.predict import TransportationPredictorExp4
        model_path = base_dir / "exp4" / "checkpoints" / "exp4_model.pth"
        if model_path.exists():
            predictors['exp4'] = TransportationPredictorExp4(str(model_path))
        else:
            print(f"⚠️ exp4 模型文件不存在: {model_path}")
    except Exception as e:
        print(f"⚠️ 加载 exp4 预测器失败: {e}")
        import traceback
        traceback.print_exc()

# load_predictors()  # 在 main.py 的 startup 事件中调用

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

def compute_trajectory_features_21d(df: pd.DataFrame) -> np.ndarray:
    """计算21维轨迹特征（包含真实OSM空间特征）"""
    features_9d, df_with_stats, _ = compute_trajectory_features(df)
    
    if osm_extractor is not None and osm_extractor.road_kdtree is not None:
        try:
            osm_features = osm_extractor.extract_spatial_features(features_9d)
            features_21d = np.hstack([features_9d, osm_features])
            return features_21d
        except Exception as e:
            print(f"⚠️ OSM特征提取失败: {e}，使用备用特征")
    
    N = len(features_9d)
    
    lat = df_with_stats['latitude'].values
    lon = df_with_stats['longitude'].values
    speed = df_with_stats['speed'].values
    acceleration = df_with_stats['acceleration'].values
    
    hour_of_day = df_with_stats['timestamp'].dt.hour.values / 24.0
    day_of_week = df_with_stats['timestamp'].dt.dayofweek.values / 6.0
    
    osm_features = np.zeros((N, 12), dtype=np.float32)
    
    for i in range(N):
        osm_features[i, 0] = lat[i]
        osm_features[i, 1] = lon[i]
        osm_features[i, 2] = speed[i]
        osm_features[i, 3] = acceleration[i]
        osm_features[i, 4] = hour_of_day[i]
        osm_features[i, 5] = day_of_week[i]
        osm_features[i, 6] = np.sin(2 * np.pi * hour_of_day[i])
        osm_features[i, 7] = np.cos(2 * np.pi * hour_of_day[i])
        osm_features[i, 8] = np.sin(2 * np.pi * day_of_week[i])
        osm_features[i, 9] = np.cos(2 * np.pi * day_of_week[i])
        osm_features[i, 10] = speed[i] / (speed.max() + 1e-6)
        osm_features[i, 11] = min(1.0, speed[i] / 15.0)
    
    features_21d = np.hstack([features_9d, osm_features])
    
    return features_21d

def compute_weather_features(df: pd.DataFrame) -> np.ndarray:
    """计算10维天气特征（使用真实天气数据）"""
    if weather_processor is not None and weather_processor._load_successful:
        return weather_processor.get_weather_features_for_trajectory(df['timestamp'])
    
    N = len(df)
    weather_features = np.zeros((N, 10), dtype=np.float32)
    
    hour = df['timestamp'].dt.hour.values
    minute = df['timestamp'].dt.minute.values
    day_of_week = df['timestamp'].dt.dayofweek.values
    
    for i in range(N):
        h = hour[i]
        m = minute[i]
        dow = day_of_week[i]
        
        time_of_day = h + m / 60.0
        
        if 6 <= h < 18:
            weather_features[i, 0] = 1.0
            weather_features[i, 1] = 18.0 + 7.0 * np.sin((time_of_day - 6) * np.pi / 12)
        else:
            weather_features[i, 0] = 0.0
            weather_features[i, 1] = 12.0 + 5.0 * np.sin((time_of_day - 18) * np.pi / 12)
        
        weather_features[i, 2] = 0.4 + 0.4 * np.sin(time_of_day * np.pi / 12)
        weather_features[i, 3] = 1013.25 + 4.0 * np.sin(time_of_day * np.pi / 6)
        weather_features[i, 4] = 3.0 + 4.0 * np.sin(time_of_day * np.pi / 8)
        weather_features[i, 5] = 0.2 + 0.5 * np.sin(time_of_day * np.pi / 10)
        weather_features[i, 6] = 0.0
        weather_features[i, 7] = 1.0 if (7 <= h < 9) or (17 <= h < 19) else 0.0
        weather_features[i, 8] = np.sin(2 * np.pi * time_of_day / 24)
        weather_features[i, 9] = np.cos(2 * np.pi * time_of_day / 24)
    
    return weather_features

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
        feature_dim = features.shape[1] if features.ndim > 1 else 1
        padding = np.zeros((target_length - len(features), feature_dim), dtype=np.float32)
        return np.vstack([features, padding])

def predict_with_model(traj_features: np.ndarray, segment_stats: np.ndarray, 
                      traj_features_21d: np.ndarray, weather_features: np.ndarray,
                      model_id: str) -> tuple:
    """使用指定模型进行预测"""
    print(f"🔍 使用模型: {model_id}")
    print(f"📊 可用的模型: {list(predictors.keys())}")
    
    if model_id not in predictors:
        raise HTTPException(status_code=400, detail=f"模型 {model_id} 未加载或不存在")
    
    predictor = predictors[model_id]
    
    if model_id == 'exp1':
        print(f"🧪 Exp1: 使用9维特征")
        pred_labels, confidences = predictor.predict(traj_features, segment_stats)
        print(f"✅ Exp1 预测结果: {pred_labels[0]}, 置信度: {confidences[0]}")
        return pred_labels[0], float(confidences[0])
    elif model_id == 'exp2':
        print(f"🧪 Exp2: 使用21维特征")
        pred_label, confidence = predictor.predict(traj_features_21d, segment_stats)
        print(f"✅ Exp2 预测结果: {pred_label}, 置信度: {confidence}")
        return pred_label, float(confidence)
    elif model_id in ['exp3', 'exp4']:
        print(f"🧪 {model_id}: 使用21维轨迹 + 10维天气特征")
        pred_label, confidence = predictor.predict(traj_features_21d, weather_features, segment_stats)
        print(f"✅ {model_id} 预测结果: {pred_label}, 置信度: {confidence}")
        return pred_label, float(confidence)
    
    return None, None

@router.post("/predict", response_model=TrajectoryPrediction)
async def predict_trajectory(file: UploadFile = File(...), modelId: str = Form('exp1')):
    model = modelId  # 将接收到的 modelId 赋值给内部逻辑使用的 model 变量
    """上传GPS文件并预测交通方式"""
    if not predictors:
        print("⚠️ 正在紧急唤醒 PyTorch 真实模型...")
        load_predictors()
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
        
        features_21d = compute_trajectory_features_21d(df)
        features_21d_normalized = normalize_sequence_length(features_21d, 50)
        
        weather_features = compute_weather_features(df)
        weather_features_normalized = normalize_sequence_length(weather_features, 50)
        
        if model in predictors:
            predicted_mode, confidence = predict_with_model(
                features_normalized, segment_stats,
                features_21d_normalized, weather_features_normalized,
                model
            )
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
            
            # if model != 'exp1':
            #  # predicted_mode = f"{predicted_mode} (规则预测)"

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


import asyncio
import json
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


class ReportRequest(BaseModel):
    model_id: str
    mode: str
    confidence: str
    scene: str = "congestion"
    distance: str = "0"
    co2: str = "0"


async def generate_deepseek_stream(req: ReportRequest):
    """真正接入 DeepSeek 的流式生成器"""
    # 针对 TrafficRec 项目场景定制 Prompt
    if req.scene == "green":
        prompt = f"""你是一个低碳出行专家。系统利用多模态深度学习识别了用户的出行轨迹：
        - 判定模态：{req.mode} (置信度 {req.confidence}%)
        - 绿色里程：{req.distance} km
        - 累计减排：{req.co2} kg
        请基于以上数据生成一份个性化报告。要求包含：1.对本次识别结果的专业确认；2.对环保贡献的量化赞赏；3.一条关于该路段后续绿色出行的建议。字数150字左右。"""
    else:
        prompt = f"""你是一个城市交通研判专家。在北京市路网分析中，系统利用 {req.model_id} 模型得出以下结果：
        - 识别出的主要交通流：{req.mode}
        - 引擎置信度：{req.confidence}%
        请从“时空特征分析”和“路段拥堵治理方案”两个维度生成深度溯源报告。要求语调专业、具有科技感，字数200字左右。"""

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            async with client.stream(
                    "POST",
                    f"{config['deepseek']['base_url']}/chat/completions",
                    headers={"Authorization": f"Bearer {config['deepseek']['api_key']}"},
                    json={
                        "model": config['deepseek']['model'],
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": True
                    }
            ) as response:
                if response.status_code != 200:
                    yield f"data: {json.dumps({'status': 'generating', 'content': 'AI 服务暂时繁忙，请稍后再试。'})}\n\n"
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        if "[DONE]" in line: break
                        data = json.loads(line[6:])
                        chunk = data["choices"][0]["delta"].get("content", "")
                        if chunk:
                            yield f"data: {json.dumps({'status': 'generating', 'content': chunk})}\n\n"

                yield f"data: {json.dumps({'status': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'status': 'generating', 'content': f'连接模型失败: {str(e)}'})}\n\n"


@router.post("/generate_report_stream")
async def generate_report_stream(req: ReportRequest, current_user: User = Depends(get_current_user)):
    return StreamingResponse(generate_deepseek_stream(req), media_type="text/event-stream")