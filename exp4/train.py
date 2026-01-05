"""
训练脚本 (Exp4)
在 Exp3 基础上增加天气数据
✅ 已集成快速模式支持（保留时间序列用于天气特征提取）
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm
import pickle
import warnings
import hashlib
import json
from datetime import datetime
import numpy as np
import pandas as pd

# ===== ✅ 修改 1: 文件开头添加导入 =====
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import BaseGeoLifePreprocessor, Exp4DataAdapter
# ===== 新增结束 =====

# 导入 Exp3 的模块
from src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader, preprocess_trajectory_segments
from src.knowledge_graph import EnhancedTransportationKG

# 导入 Exp4 的新模块
from src.weather_preprocessing import WeatherDataProcessor
from src.feature_extraction_weather import FeatureExtractorWithWeather
from src.model_weather import TransportationModeClassifierWithWeather

# ========================== 特征维度常量 ==========================
TRAJECTORY_FEATURE_DIM = 9
KG_FEATURE_DIM = 15
WEATHER_FEATURE_DIM = 12  # 新增
# ==================================================================

# ========================== 缓存配置 ==========================
CACHE_VERSION = "v1"
CACHE_DIR = 'cache'
KG_CACHE_PATH = os.path.join(CACHE_DIR, f'kg_data_{CACHE_VERSION}.pkl')
GRID_CACHE_PATH = os.path.join(CACHE_DIR, f'grid_cache_{CACHE_VERSION}.pkl')
WEATHER_CACHE_PATH = os.path.join(CACHE_DIR, f'weather_data_{CACHE_VERSION}.pkl')
PROCESSED_FEATURE_CACHE_PATH = os.path.join(CACHE_DIR, f'processed_features_weather_{CACHE_VERSION}.pkl')
META_CACHE_PATH = os.path.join(CACHE_DIR, 'cache_meta_weather.json')
os.makedirs(CACHE_DIR, exist_ok=True)


def compute_file_hash(filepath: str) -> str:
    """计算文件 MD5 哈希"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_cache_metadata(osm_path: str, weather_path: str, geolife_root: str,
                        num_segments: int, label_encoder: LabelEncoder):
    """保存缓存元数据"""
    meta = {
        "version": CACHE_VERSION,
        "experiment": "exp4",
        "created_at": datetime.now().isoformat(),
        "osm_file": osm_path,
        "osm_file_hash": compute_file_hash(osm_path),
        "weather_file": weather_path,
        "weather_file_hash": compute_file_hash(weather_path),
        "geolife_root": geolife_root,
        "kg_feature_dim": KG_FEATURE_DIM,
        "trajectory_feature_dim": TRAJECTORY_FEATURE_DIM,
        "weather_feature_dim": WEATHER_FEATURE_DIM,
        "total_feature_dim": TRAJECTORY_FEATURE_DIM + KG_FEATURE_DIM + WEATHER_FEATURE_DIM,
        "num_segments": num_segments,
        "num_classes": len(label_encoder.classes_),
        "classes": label_encoder.classes_.tolist()
    }

    with open(META_CACHE_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"✓ 缓存元数据已保存: {META_CACHE_PATH}")


def validate_cache(osm_path: str, weather_path: str) -> bool:
    """验证缓存是否有效"""
    if not os.path.exists(META_CACHE_PATH):
        return False

    try:
        with open(META_CACHE_PATH, 'r') as f:
            meta = json.load(f)

        if meta.get('version') != CACHE_VERSION:
            print(f"⚠️  缓存版本不匹配")
            return False

        if meta.get('experiment') != 'exp4':
            print(f"⚠️  缓存实验类型不匹配")
            return False

        current_osm_hash = compute_file_hash(osm_path)
        if meta.get('osm_file_hash') != current_osm_hash:
            print(f"⚠️  OSM 文件已更改")
            return False

        current_weather_hash = compute_file_hash(weather_path)
        if meta.get('weather_file_hash') != current_weather_hash:
            print(f"⚠️  天气文件已更改")
            return False

        print(f"✓ 缓存验证通过 (版本: {CACHE_VERSION})")
        return True

    except Exception as e:
        print(f"⚠️  缓存验证失败: {e}")
        return False


class TrajectoryDatasetWithWeather(Dataset):
    """轨迹数据集（含天气）"""

    def __init__(self, all_features_and_labels):
        self.data = all_features_and_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trajectory_features, kg_features, weather_features, label_encoded = self.data[idx]

        trajectory_tensor = torch.FloatTensor(trajectory_features)
        kg_tensor = torch.FloatTensor(kg_features)
        weather_tensor = torch.FloatTensor(weather_features)
        label_tensor = torch.LongTensor([label_encoded])[0]

        return trajectory_tensor, kg_tensor, weather_tensor, label_tensor


# ============================================================
# Data loading (✅ 修改 2: 更新 load_data 函数集成快速模式)
# ============================================================
def load_data(geolife_root: str, osm_path: str, weather_path: str, max_users: int = None, use_base_data: bool = False):
    """加载所有数据 (支持快速模式与天气特征)"""

    BASE_DATA_PATH = os.path.join(
        os.path.dirname(geolife_root),
        'processed/base_segments.pkl'
    )

    geolife_loader = GeoLifeDataLoader(geolife_root)
    users = geolife_loader.get_all_users()
    if max_users and max_users < len(users):
        users = users[:max_users]

    # ================= 阶段 1: 知识图谱构建 ==================
    kg = None
    if os.path.exists(KG_CACHE_PATH):
        print(f"\n========== 阶段 1: 知识图谱加载 (从缓存) ==========")
        try:
            with open(KG_CACHE_PATH, 'rb') as f:
                kg = pickle.load(f)
            print("✅ 知识图谱从缓存加载完成")
            if os.path.exists(GRID_CACHE_PATH):
                kg.load_cache(GRID_CACHE_PATH)
        except Exception as e:
            warnings.warn(f"[WARN] KG 缓存加载失败 ({e})，将重新构建")
            kg = None

    if kg is None:
        print("\n========== 阶段 1: 知识图谱构建 (重建) ==========")
        osm_loader = OSMDataLoader(osm_path)
        osm_data = osm_loader.load_osm_data()
        road_network = osm_loader.extract_road_network(osm_data)
        pois = osm_loader.extract_pois(osm_data)
        transit_routes = osm_loader.extract_transit_routes(osm_data)
        kg = EnhancedTransportationKG()
        kg.build_from_osm(road_network, pois, transit_routes)
        with open(KG_CACHE_PATH, 'wb') as f:
            pickle.dump(kg, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✅ 知识图谱缓存完成")

    # ================= 阶段 2: 天气数据加载 ==================
    weather_processor = None
    if os.path.exists(WEATHER_CACHE_PATH):
        print(f"\n========== 阶段 2: 天气数据加载 (从缓存) ==========")
        try:
            with open(WEATHER_CACHE_PATH, 'rb') as f:
                weather_processor = pickle.load(f)
            print("✅ 天气数据从缓存加载完成")
        except Exception as e:
            warnings.warn(f"[WARN] 天气缓存加载失败 ({e})")
            weather_processor = None

    if weather_processor is None:
        print(f"\n========== 阶段 2: 天气数据处理 (重建) ==========")
        weather_processor = WeatherDataProcessor(weather_path)
        weather_processor.load_and_process()
        with open(WEATHER_CACHE_PATH, 'wb') as f:
            pickle.dump(weather_processor, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✅ 天气数据缓存完成")

    # ================= 阶段 3: 轨迹数据加载与特征提取 ==================
    all_features_and_labels = None
    label_encoder = None

    if os.path.exists(PROCESSED_FEATURE_CACHE_PATH):
        print(f"\n========== 阶段 3: 特征加载 (从缓存) ==========")
        try:
            with open(PROCESSED_FEATURE_CACHE_PATH, 'rb') as f:
                all_features_and_labels, label_encoder = pickle.load(f)
            print(f"✅ 特征从缓存加载完成: {len(all_features_and_labels)} 条")
            return all_features_and_labels, kg, weather_processor, label_encoder
        except Exception:
            pass

    processed_segments_with_time = None

    # ✅ 快速模式：使用基础数据 (保留时间序列用于天气)
    if use_base_data and os.path.exists(BASE_DATA_PATH):
        print(f"\n{'='*80}")
        print("阶段 3: 使用预处理的基础数据（快速模式 - 含时间序列）")
        print(f"{'='*80}\n")

        base_segments = BaseGeoLifePreprocessor.load_from_cache(BASE_DATA_PATH)
        adapter = Exp4DataAdapter(target_length=50)
        processed_segments_with_time = adapter.process_segments(base_segments)

    # 传统模式：从头处理
    else:
        if use_base_data:
            print(f"\n⚠️  基础数据不存在，使用传统模式")

        print("\n========== 阶段 3: 加载轨迹数据 (传统模式) ==========")
        all_segments_with_dates = []
        for user_id in tqdm(users, desc="[用户加载]"):
            labels = geolife_loader.load_labels(user_id)
            if labels.empty: continue
            trajectory_dir = os.path.join(geolife_root, f"Data/{user_id}/Trajectory")
            if not os.path.exists(trajectory_dir): continue

            for traj_file in os.listdir(trajectory_dir):
                if not traj_file.endswith('.plt'): continue
                try:
                    trajectory = geolife_loader.load_trajectory(os.path.join(trajectory_dir, traj_file))
                    segments = geolife_loader.segment_trajectory(trajectory, labels)
                    for seg, label in segments:
                        if 'datetime' in seg.columns and len(seg) >= 10:
                            all_segments_with_dates.append((seg, label, seg['datetime']))
                except: continue

        # 传统模式下的规范化处理
        processed_segments_with_time = []
        for segment, label, dates in tqdm(all_segments_with_dates, desc="[预处理]"):
            feature_cols = ['latitude', 'longitude', 'speed', 'acceleration', 'bearing_change', 'distance', 'time_diff', 'total_distance', 'total_time']
            features = segment[feature_cols].values
            FIXED_LENGTH = 50
            if len(features) > FIXED_LENGTH:
                indices = np.linspace(0, len(features) - 1, FIXED_LENGTH, dtype=int)
                features = features[indices]; dates_resampled = dates.iloc[indices]
            elif len(features) < FIXED_LENGTH:
                padding = np.zeros((FIXED_LENGTH - len(features), features.shape[1]))
                features = np.vstack([features, padding])
                dates_resampled = pd.concat([dates.reset_index(drop=True), pd.Series([dates.iloc[-1]] * (FIXED_LENGTH - len(dates)))], ignore_index=True)
            else:
                dates_resampled = dates.reset_index(drop=True)
            processed_segments_with_time.append((features, dates_resampled, label))

    # 过滤与编码
    valid_modes = {'Walk', 'Bike', 'Bus', 'Car & taxi', 'Train', 'Subway', 'Airplane'}
    processed_segments_with_time = [s for s in processed_segments_with_time if s[2] in valid_modes]
    label_encoder = LabelEncoder().fit([s[2] for s in processed_segments_with_time])

    # 特征提取
    print("\n3.1 正在进行【增强特征提取（含天气）】...")
    feature_extractor = FeatureExtractorWithWeather(kg, weather_processor)
    all_features_and_labels = []

    for trajectory, datetime_series, label_str in tqdm(processed_segments_with_time, desc="[Exp4 特征提取]"):
        try:
            trajectory_features, kg_features, weather_features = feature_extractor.extract_features(trajectory, datetime_series)
            label_encoded = label_encoder.transform([label_str])[0]
            all_features_and_labels.append((trajectory_features, kg_features, weather_features, label_encoded))
        except Exception as e:
            continue

    with open(PROCESSED_FEATURE_CACHE_PATH, 'wb') as f:
        pickle.dump((all_features_and_labels, label_encoder), f, protocol=pickle.HIGHEST_PROTOCOL)
    kg.save_cache(GRID_CACHE_PATH)
    save_cache_metadata(osm_path, weather_path, geolife_root, len(all_features_and_labels), label_encoder)

    return all_features_and_labels, kg, weather_processor, label_encoder


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for traj_f, kg_f, weather_f, labels in tqdm(dataloader, desc="   [训练]"):
        traj_f, kg_f, weather_f, labels = traj_f.to(device), kg_f.to(device), weather_f.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(traj_f, kg_f, weather_f)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device, label_encoder):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for traj_f, kg_f, weather_f, labels in tqdm(dataloader, desc="   [评估]"):
            traj_f, kg_f, weather_f, labels = traj_f.to(device), kg_f.to(device), weather_f.to(device), labels.to(device)
            logits = model(traj_f, kg_f, weather_f)
            total_loss += criterion(logits, labels).item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    return total_loss / len(dataloader), report, all_preds, all_labels


# ============================================================
# Main (✅ 修改 3: 更新 main 函数添加命令行参数)
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='训练交通方式识别模型 (Exp4 - 含天气)')
    parser.add_argument('--geolife_root', type=str, default='../data/Geolife Trajectories 1.3')
    parser.add_argument('--osm_path', type=str, default='../data/exp3.geojson')
    parser.add_argument('--weather_path', type=str, default='../data/beijing_weather_hourly_2007_2012.csv')

    # ===== ✅ 新增参数 =====
    parser.add_argument('--use_base_data', action='store_true', help='使用预处理的基础数据（推荐）')
    # ===== 新增结束 =====

    parser.add_argument('--max_users', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--clear_cache', action='store_true')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.clear_cache:
        for f in [KG_CACHE_PATH, GRID_CACHE_PATH, WEATHER_CACHE_PATH, PROCESSED_FEATURE_CACHE_PATH, META_CACHE_PATH]:
            if os.path.exists(f): os.remove(f)

    # ✅ 传递新参数
    all_features_and_labels, kg, weather_processor, label_encoder = load_data(
        args.geolife_root, args.osm_path, args.weather_path, args.max_users,
        use_base_data=args.use_base_data
    )

    # 训练逻辑 (与原代码一致)
    num_classes = len(label_encoder.classes_)
    dataset = TrajectoryDatasetWithWeather(all_features_and_labels)
    labels_stratify = [l for _, _, _, l in all_features_and_labels]
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42, stratify=labels_stratify)

    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TransportationModeClassifierWithWeather(
        TRAJECTORY_FEATURE_DIM, KG_FEATURE_DIM, WEATHER_FEATURE_DIM, args.hidden_dim, args.num_layers, num_classes, args.dropout
    ).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    best_test_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\n[EPOCH {epoch + 1}/{args.epochs}]")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)
        test_loss, report, test_preds, test_labels = evaluate(model, test_loader, criterion, args.device, label_encoder)

        print(f"   Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Acc: {report['accuracy']:.4f}")
        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'model_state_dict': model.state_dict(), 'label_encoder': label_encoder,
                'model_config': {
                    'trajectory_feature_dim': TRAJECTORY_FEATURE_DIM, 'kg_feature_dim': KG_FEATURE_DIM,
                    'weather_feature_dim': WEATHER_FEATURE_DIM, 'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers, 'num_classes': num_classes, 'dropout': args.dropout
                }
            }, os.path.join(args.save_dir, 'exp4_model.pth'))
            print("   ✓ 保存最佳模型")

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()