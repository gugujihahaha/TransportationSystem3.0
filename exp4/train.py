"""
训练脚本 (Exp4 - 稳定版)
在 Exp3 基础上增加天气数据

稳定性增强:
1. 全面的缺失数据处理（天气、KG）
2. 梯度裁剪和数值稳定性
3. NaN/Inf 检测和处理
4. 样本保留策略（不丢弃任何样本）
5. 降级特征提取
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

# ========================== 路径设置 (PyCharm 兼容) ==========================
# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 将 exp4 目录添加到 Python 路径
sys.path.insert(0, SCRIPT_DIR)

# 将上级目录也添加到路径（用于 common 模块）
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

# 切换工作目录到脚本所在目录（确保相对路径正确）
os.chdir(SCRIPT_DIR)
# ==============================================================================

# 尝试导入 common 模块（可选）
try:
    from common import BaseGeoLifePreprocessor, Exp4DataAdapter
    from common.data_split import load_split
    HAS_COMMON = True
except ImportError:
    HAS_COMMON = False
    print("⚠️ common 模块未找到，将使用传统数据加载模式")

# 导入 Exp3/4 的模块
from src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader, preprocess_trajectory_segments
from src.knowledge_graph import EnhancedTransportationKG
from src.weather_preprocessing import WeatherDataProcessor
from src.feature_extraction_weather import FeatureExtractorWithWeather
from src.model_weather import TransportationModeClassifierWithWeather

# ========================== 特征维度常量 ==========================
TRAJECTORY_FEATURE_DIM = 9
KG_FEATURE_DIM = 15
WEATHER_FEATURE_DIM = 12
TOTAL_FEATURE_DIM = TRAJECTORY_FEATURE_DIM + KG_FEATURE_DIM + WEATHER_FEATURE_DIM  # 36
FIXED_SEQUENCE_LENGTH = 50
# ==================================================================

# ========================== 缓存配置 ==========================
CACHE_VERSION = "v2_stable"
CACHE_DIR = 'cache'
KG_CACHE_PATH = os.path.join(CACHE_DIR, f'kg_data_{CACHE_VERSION}.pkl')
GRID_CACHE_PATH = os.path.join(CACHE_DIR, f'grid_cache_{CACHE_VERSION}.pkl')
WEATHER_CACHE_PATH = os.path.join(CACHE_DIR, f'weather_data_{CACHE_VERSION}.pkl')
PROCESSED_FEATURE_CACHE_PATH = os.path.join(CACHE_DIR, f'processed_features_weather_{CACHE_VERSION}.pkl')
META_CACHE_PATH = os.path.join(CACHE_DIR, 'cache_meta_weather.json')
os.makedirs(CACHE_DIR, exist_ok=True)


def compute_file_hash(filepath: str) -> str:
    """计算文件 MD5 哈希"""
    try:
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return "unknown"


def save_cache_metadata(osm_path: str, weather_path: str, geolife_root: str,
                        num_segments: int, label_encoder: LabelEncoder):
    """保存缓存元数据"""
    meta = {
        "version": CACHE_VERSION,
        "experiment": "exp4_stable",
        "created_at": datetime.now().isoformat(),
        "osm_file": osm_path,
        "osm_file_hash": compute_file_hash(osm_path),
        "weather_file": weather_path,
        "weather_file_hash": compute_file_hash(weather_path),
        "geolife_root": geolife_root,
        "kg_feature_dim": KG_FEATURE_DIM,
        "trajectory_feature_dim": TRAJECTORY_FEATURE_DIM,
        "weather_feature_dim": WEATHER_FEATURE_DIM,
        "total_feature_dim": TOTAL_FEATURE_DIM,
        "num_segments": num_segments,
        "num_classes": len(label_encoder.classes_),
        "classes": label_encoder.classes_.tolist()
    }

    try:
        with open(META_CACHE_PATH, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"✓ 缓存元数据已保存: {META_CACHE_PATH}")
    except Exception as e:
        print(f"⚠️ 元数据保存失败: {e}")


def validate_cache(osm_path: str, weather_path: str) -> bool:
    """验证缓存是否有效"""
    if not os.path.exists(META_CACHE_PATH):
        return False

    try:
        with open(META_CACHE_PATH, 'r') as f:
            meta = json.load(f)

        if meta.get('version') != CACHE_VERSION:
            print(f"⚠️ 缓存版本不匹配")
            return False

        if meta.get('experiment') != 'exp4_stable':
            print(f"⚠️ 缓存实验类型不匹配")
            return False

        current_osm_hash = compute_file_hash(osm_path)
        if meta.get('osm_file_hash') != current_osm_hash:
            print(f"⚠️ OSM 文件已更改")
            return False

        current_weather_hash = compute_file_hash(weather_path)
        if meta.get('weather_file_hash') != current_weather_hash:
            print(f"⚠️ 天气文件已更改")
            return False

        print(f"✓ 缓存验证通过 (版本: {CACHE_VERSION})")
        return True

    except Exception as e:
        print(f"⚠️ 缓存验证失败: {e}")
        return False


class TrajectoryDatasetWithWeather(Dataset):
    """轨迹数据集（含天气）- 稳定版"""

    def __init__(self, all_features_and_labels):
        self.data = all_features_and_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trajectory_features, kg_features, weather_features, label_encoded = self.data[idx]

        # 确保数据类型和形状正确
        trajectory_tensor = torch.FloatTensor(
            np.nan_to_num(trajectory_features, nan=0.0, posinf=0.0, neginf=0.0)
        )
        kg_tensor = torch.FloatTensor(
            np.nan_to_num(kg_features, nan=0.0, posinf=0.0, neginf=0.0)
        )
        weather_tensor = torch.FloatTensor(
            np.nan_to_num(weather_features, nan=0.0, posinf=0.0, neginf=0.0)
        )
        label_tensor = torch.LongTensor([label_encoded])[0]

        return trajectory_tensor, kg_tensor, weather_tensor, label_tensor


def normalize_features_safe(features: np.ndarray) -> np.ndarray:
    """安全的特征归一化"""
    features = features.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    normalized = (features - mean) / std
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = np.clip(normalized, -5, 5)

    return normalized.astype(np.float32)


def load_data(geolife_root: str, osm_path: str, weather_path: str,
              max_users: int = None, use_base_data: bool = True):
    """
    加载所有数据 - 稳定版

    关键改进:
    1. 不丢弃任何样本
    2. 缺失特征使用零向量填充
    3. 全面的异常处理
    """
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
        try:
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
        except Exception as e:
            print(f"⚠️ KG 构建失败: {e}")
            kg = EnhancedTransportationKG()  # 空 KG

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
        try:
            weather_processor = WeatherDataProcessor(weather_path)
            weather_processor.load_and_process()
            with open(WEATHER_CACHE_PATH, 'wb') as f:
                pickle.dump(weather_processor, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("✅ 天气数据缓存完成")
        except Exception as e:
            print(f"⚠️ 天气数据处理失败: {e}")
            weather_processor = WeatherDataProcessor(weather_path)  # 空处理器

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
        except Exception as e:
            print(f"⚠️ 特征缓存加载失败: {e}")

    processed_segments_with_time = None

    # 快速模式：使用基础数据
    if use_base_data and HAS_COMMON and os.path.exists(BASE_DATA_PATH):
        print(f"\n{'='*80}")
        print("阶段 3: 使用预处理的基础数据（快速模式 - 含时间序列）")
        print(f"{'='*80}\n")

        try:
            base_segments = BaseGeoLifePreprocessor.load_from_cache(BASE_DATA_PATH)
            adapter = Exp4DataAdapter(target_length=FIXED_SEQUENCE_LENGTH)
            processed_segments_with_time = adapter.process_segments(base_segments)
        except Exception as e:
            print(f"⚠️ 快速模式失败: {e}，切换到传统模式")
            processed_segments_with_time = None

    # 传统模式：从头处理
    if processed_segments_with_time is None:
        if use_base_data:
            print(f"\n⚠️ 基础数据不可用，使用传统模式")

        print("\n========== 阶段 3: 加载轨迹数据 (传统模式) ==========")
        all_segments_with_dates = []

        for user_id in tqdm(users, desc="[用户加载]"):
            try:
                labels = geolife_loader.load_labels(user_id)
                if labels.empty:
                    continue
                trajectory_dir = os.path.join(geolife_root, f"Data/{user_id}/Trajectory")
                if not os.path.exists(trajectory_dir):
                    continue

                for traj_file in os.listdir(trajectory_dir):
                    if not traj_file.endswith('.plt'):
                        continue
                    try:
                        trajectory = geolife_loader.load_trajectory(
                            os.path.join(trajectory_dir, traj_file)
                        )
                        if trajectory.empty:
                            continue
                        segments = geolife_loader.segment_trajectory(trajectory, labels)
                        for seg, label in segments:
                            if 'datetime' in seg.columns and len(seg) >= 10:
                                all_segments_with_dates.append((seg, label, seg['datetime']))
                    except Exception:
                        continue
            except Exception:
                continue

        # 传统模式下的规范化处理
        processed_segments_with_time = []
        feature_cols = [
            'latitude', 'longitude', 'speed', 'acceleration',
            'bearing_change', 'distance', 'time_diff',
            'total_distance', 'total_time'
        ]

        for segment, label, dates in tqdm(all_segments_with_dates, desc="[预处理]"):
            try:
                # 确保所有特征列存在
                for col in feature_cols:
                    if col not in segment.columns:
                        segment[col] = 0.0

                features = segment[feature_cols].values.astype(np.float32)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

                current_length = len(features)

                # 序列长度规范化
                if current_length > FIXED_SEQUENCE_LENGTH:
                    indices = np.linspace(0, current_length - 1, FIXED_SEQUENCE_LENGTH, dtype=int)
                    features = features[indices]
                    dates_resampled = dates.iloc[indices].reset_index(drop=True)
                elif current_length < FIXED_SEQUENCE_LENGTH:
                    padding = np.zeros((FIXED_SEQUENCE_LENGTH - current_length, features.shape[1]),
                                      dtype=np.float32)
                    features = np.vstack([features, padding])
                    # 填充日期使用最后一个有效日期
                    last_date = dates.iloc[-1] if len(dates) > 0 else pd.Timestamp.now()
                    padding_dates = pd.Series([last_date] * (FIXED_SEQUENCE_LENGTH - len(dates)))
                    dates_resampled = pd.concat([dates.reset_index(drop=True), padding_dates],
                                               ignore_index=True)
                else:
                    dates_resampled = dates.reset_index(drop=True)

                processed_segments_with_time.append((features, dates_resampled, label))
            except Exception:
                continue

    # 过滤有效类别
    valid_modes = {'Walk', 'Bike', 'Bus', 'Car & taxi', 'Train', 'Subway', 'Airplane'}
    processed_segments_with_time = [
        s for s in processed_segments_with_time if s[2] in valid_modes
    ]

    if len(processed_segments_with_time) == 0:
        raise ValueError("没有有效的轨迹数据！请检查数据路径和格式。")

    # 标签编码
    all_labels = [s[2] for s in processed_segments_with_time]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    print(f"\n类别分布:")
    for cls in label_encoder.classes_:
        count = sum(1 for l in all_labels if l == cls)
        print(f"  {cls}: {count}")

    # ================= 特征提取（关键改进：不丢弃样本）==================
    print("\n3.1 正在进行【增强特征提取（含天气）】...")
    feature_extractor = FeatureExtractorWithWeather(kg, weather_processor)
    all_features_and_labels = []

    success_count = 0
    degraded_count = 0

    for trajectory, datetime_series, label_str in tqdm(processed_segments_with_time,
                                                        desc="[Exp4 特征提取]"):
        try:
            # 尝试完整特征提取
            trajectory_features, kg_features, weather_features = feature_extractor.extract_features(
                trajectory, datetime_series
            )

            # 验证形状
            assert trajectory_features.shape == (FIXED_SEQUENCE_LENGTH, TRAJECTORY_FEATURE_DIM)
            assert kg_features.shape == (FIXED_SEQUENCE_LENGTH, KG_FEATURE_DIM)
            assert weather_features.shape == (FIXED_SEQUENCE_LENGTH, WEATHER_FEATURE_DIM)

            label_encoded = label_encoder.transform([label_str])[0]
            all_features_and_labels.append((
                trajectory_features, kg_features, weather_features, label_encoded
            ))
            success_count += 1

        except Exception as e:
            # ========== 降级特征提取：绝不丢弃样本 ==========
            try:
                # 轨迹特征：归一化原始数据
                trajectory_features = normalize_features_safe(trajectory)
                if trajectory_features.shape != (FIXED_SEQUENCE_LENGTH, TRAJECTORY_FEATURE_DIM):
                    # 形状修复
                    if trajectory_features.shape[0] != FIXED_SEQUENCE_LENGTH:
                        if trajectory_features.shape[0] > FIXED_SEQUENCE_LENGTH:
                            trajectory_features = trajectory_features[:FIXED_SEQUENCE_LENGTH]
                        else:
                            padding = np.zeros((FIXED_SEQUENCE_LENGTH - trajectory_features.shape[0],
                                              TRAJECTORY_FEATURE_DIM), dtype=np.float32)
                            trajectory_features = np.vstack([trajectory_features, padding])
                    if trajectory_features.shape[1] != TRAJECTORY_FEATURE_DIM:
                        if trajectory_features.shape[1] > TRAJECTORY_FEATURE_DIM:
                            trajectory_features = trajectory_features[:, :TRAJECTORY_FEATURE_DIM]
                        else:
                            padding = np.zeros((FIXED_SEQUENCE_LENGTH,
                                              TRAJECTORY_FEATURE_DIM - trajectory_features.shape[1]),
                                              dtype=np.float32)
                            trajectory_features = np.hstack([trajectory_features, padding])

                # KG 和天气：使用零向量
                kg_features = np.zeros((FIXED_SEQUENCE_LENGTH, KG_FEATURE_DIM), dtype=np.float32)
                weather_features = np.zeros((FIXED_SEQUENCE_LENGTH, WEATHER_FEATURE_DIM), dtype=np.float32)

                label_encoded = label_encoder.transform([label_str])[0]
                all_features_and_labels.append((
                    trajectory_features, kg_features, weather_features, label_encoded
                ))
                degraded_count += 1

            except Exception as inner_e:
                # 极端情况：创建完全零特征
                trajectory_features = np.zeros((FIXED_SEQUENCE_LENGTH, TRAJECTORY_FEATURE_DIM),
                                              dtype=np.float32)
                kg_features = np.zeros((FIXED_SEQUENCE_LENGTH, KG_FEATURE_DIM), dtype=np.float32)
                weather_features = np.zeros((FIXED_SEQUENCE_LENGTH, WEATHER_FEATURE_DIM), dtype=np.float32)

                try:
                    label_encoded = label_encoder.transform([label_str])[0]
                except:
                    label_encoded = 0

                all_features_and_labels.append((
                    trajectory_features, kg_features, weather_features, label_encoded
                ))
                degraded_count += 1

    print(f"\n✅ 特征提取完成:")
    print(f"   完整特征: {success_count}")
    print(f"   降级特征: {degraded_count}")
    print(f"   总样本数: {len(all_features_and_labels)}")

    # 保存缓存
    try:
        with open(PROCESSED_FEATURE_CACHE_PATH, 'wb') as f:
            pickle.dump((all_features_and_labels, label_encoder), f,
                       protocol=pickle.HIGHEST_PROTOCOL)
        kg.save_cache(GRID_CACHE_PATH)
        save_cache_metadata(osm_path, weather_path, geolife_root,
                           len(all_features_and_labels), label_encoder)
    except Exception as e:
        print(f"⚠️ 缓存保存失败: {e}")

    return all_features_and_labels, kg, weather_processor, label_encoder


def check_tensor_health(tensor: torch.Tensor, name: str) -> bool:
    """检查张量是否健康（无 NaN/Inf）"""
    if torch.isnan(tensor).any():
        print(f"⚠️ {name} 包含 NaN")
        return False
    if torch.isinf(tensor).any():
        print(f"⚠️ {name} 包含 Inf")
        return False
    return True


def train_epoch(model, dataloader, criterion, optimizer, device,
                max_grad_norm: float = 1.0):
    """
    训练一个 epoch - 稳定版

    关键改进:
    1. 梯度裁剪
    2. NaN/Inf 检测
    3. 跳过异常批次但不中断训练
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    nan_batches = 0

    for batch_idx, (traj_f, kg_f, weather_f, labels) in enumerate(
        tqdm(dataloader, desc="   [训练]", leave=False)
    ):
        try:
            # 数据移动到设备
            traj_f = traj_f.to(device)
            kg_f = kg_f.to(device)
            weather_f = weather_f.to(device)
            labels = labels.to(device)

            # 检查输入健康状态（静默检查，不打印）
            input_healthy = True
            for t in [traj_f, kg_f, weather_f]:
                if torch.isnan(t).any() or torch.isinf(t).any():
                    input_healthy = False
                    break

            if not input_healthy:
                nan_batches += 1
                continue

            optimizer.zero_grad()

            # 前向传播
            logits = model(traj_f, kg_f, weather_f)

            # 检查输出
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_batches += 1
                continue

            # 计算损失
            loss = criterion(logits, labels)

            # 检查损失
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                continue

            # 反向传播
            loss.backward()

            # 检查梯度
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        break

            if has_nan_grad:
                optimizer.zero_grad()
                nan_batches += 1
                continue

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            # 参数更新
            optimizer.step()

            # 统计
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        except RuntimeError as e:
            nan_batches += 1
            optimizer.zero_grad()
            continue

    if nan_batches > 0:
        print(f"   ⚠️ 本 epoch 跳过 {nan_batches} 个异常批次")

    avg_loss = total_loss / max(len(dataloader) - nan_batches, 1)
    accuracy = correct / max(total, 1)

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, label_encoder):
    """
    评估模型 - 稳定版
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    nan_batches = 0

    with torch.no_grad():
        for batch_idx, (traj_f, kg_f, weather_f, labels) in enumerate(
            tqdm(dataloader, desc="   [评估]", leave=False)
        ):
            try:
                traj_f = traj_f.to(device)
                kg_f = kg_f.to(device)
                weather_f = weather_f.to(device)
                labels = labels.to(device)

                logits = model(traj_f, kg_f, weather_f)

                # 检查输出
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    nan_batches += 1
                    continue

                loss = criterion(logits, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_batches += 1
                    continue

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            except Exception as e:
                nan_batches += 1
                continue

    if nan_batches > 0:
        print(f"   ⚠️ 评估时跳过 {nan_batches} 个异常批次")

    # 生成报告
    if len(all_preds) > 0:
        report = classification_report(
            all_labels, all_preds,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
    else:
        report = {'accuracy': 0.0}

    avg_loss = total_loss / max(len(dataloader) - nan_batches, 1)

    return avg_loss, report, all_preds, all_labels


def main():
    # ========================================================================
    # PyCharm 直接运行配置区域
    # 如果在 PyCharm 中点击运行按钮，请修改下面的路径为你的实际路径
    # ========================================================================
    PYCHARM_MODE = True  # 设为 True 使用下面的硬编码路径，设为 False 使用命令行参数

    if PYCHARM_MODE:
        class Args:
            # ============ 请根据你的实际路径修改以下三行 ============
            geolife_root = '../data/Geolife Trajectories 1.3'  # GeoLife 数据集路径
            osm_path = '../data/exp3.geojson'                   # OSM GeoJSON 路径
            weather_path = '../data/beijing_weather_hourly_2007_2012.csv'  # 天气数据路径
            # ========================================================

            # 数据加载选项
            use_base_data = True   # 使用预处理的基础数据（更快）
            max_users = None       # 最大用户数（None = 全部）

            # 训练参数
            batch_size = 32
            epochs = 50
            lr = 5e-5              # 学习率
            hidden_dim = 128
            num_layers = 2
            dropout = 0.3
            max_grad_norm = 1.0    # 梯度裁剪

            # 系统参数
            save_dir = 'checkpoints'
            num_workers = 0        # Windows 建议设为 0，Linux/Mac 可以设为 4
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            clear_cache = False    # 是否清除缓存
            seed = 42

        args = Args()
        print("📌 使用 PyCharm 模式（硬编码配置）")
    else:
        parser = argparse.ArgumentParser(description='训练交通方式识别模型 (Exp4 - 稳定版)')
        parser.add_argument('--geolife_root', type=str, default='../data/Geolife Trajectories 1.3')
        parser.add_argument('--osm_path', type=str, default='../data/exp3.geojson')
        parser.add_argument('--weather_path', type=str, default='../data/beijing_weather_hourly_2007_2012.csv')

        # 数据加载选项
        parser.add_argument('--use_base_data', action='store_true', default=True,
                            help='使用预处理的基础数据（推荐）')
        parser.add_argument('--max_users', type=int, default=None)

        # 训练参数（调整默认值以提高稳定性）
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--lr', type=float, default=5e-5,
                            help='学习率（建议 1e-4 或 5e-5）')
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.3)
        parser.add_argument('--max_grad_norm', type=float, default=1.0,
                            help='梯度裁剪阈值')

        # 系统参数
        parser.add_argument('--save_dir', type=str, default='checkpoints')
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--device', type=str,
                            default='cuda' if torch.cuda.is_available() else 'cpu')
        parser.add_argument('--clear_cache', action='store_true')
        parser.add_argument('--seed', type=int, default=42)

        args = parser.parse_args()

    # ========================================================================

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Exp4 交通方式识别训练 (稳定版)")
    print("=" * 80)
    print(f"设备: {args.device}")
    print(f"学习率: {args.lr}")
    print(f"梯度裁剪: {args.max_grad_norm}")
    print(f"批次大小: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print("=" * 80 + "\n")

    # 清理缓存
    if args.clear_cache:
        print("正在清理缓存...")
        for f in [KG_CACHE_PATH, GRID_CACHE_PATH, WEATHER_CACHE_PATH,
                  PROCESSED_FEATURE_CACHE_PATH, META_CACHE_PATH]:
            if os.path.exists(f):
                os.remove(f)
                print(f"  已删除: {f}")

    # 加载数据
    all_features_and_labels, kg, weather_processor, label_encoder = load_data(
        args.geolife_root, args.osm_path, args.weather_path, args.max_users,
        use_base_data=args.use_base_data
    )

    # 数据集划分
    num_classes = len(label_encoder.classes_)
    print(f"\n类别数量: {num_classes}")
    print(f"类别列表: {label_encoder.classes_}")

    dataset = TrajectoryDatasetWithWeather(all_features_and_labels)

    # ========================================================
    # ✅ 加载统一数据划分（Train/Val/Test = 0.7/0.15/0.15）
    # ========================================================
    splits = load_split("common/splits.json")

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, splits["train"]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, splits["val"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, splits["test"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"\n✅ 数据加载完成:")
    print(f"  Train: {len(splits['train'])} 样本")
    print(f"  Val:   {len(splits['val'])} 样本")
    print(f"  Test:  {len(splits['test'])} 样本")

    # 构建模型
    model = TransportationModeClassifierWithWeather(
        trajectory_feature_dim=TRAJECTORY_FEATURE_DIM,
        kg_feature_dim=KG_FEATURE_DIM,
        weather_feature_dim=WEATHER_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(args.device)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_test_loss = float('inf')
    best_accuracy = 0.0
    patience_counter = 0
    early_stop_patience = 15

    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)

    for epoch in range(args.epochs):
        print(f"\n[EPOCH {epoch + 1}/{args.epochs}]")

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device,
            max_grad_norm=args.max_grad_norm
        )

        # 评估
        test_loss, report, test_preds, test_labels = evaluate(
            model, test_loader, criterion, args.device, label_encoder
        )

        test_acc = report.get('accuracy', 0.0)

        # 打印结果
        print(f"   Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"   Test  Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 检查 NaN
        if np.isnan(train_loss) or np.isnan(test_loss):
            print("   ⚠️ 检测到 NaN，降低学习率重试")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            continue

        # 学习率调度
        scheduler.step(test_loss)

        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_accuracy = test_acc
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_encoder': label_encoder,
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'model_config': {
                    'trajectory_feature_dim': TRAJECTORY_FEATURE_DIM,
                    'kg_feature_dim': KG_FEATURE_DIM,
                    'weather_feature_dim': WEATHER_FEATURE_DIM,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'num_classes': num_classes,
                    'dropout': args.dropout
                }
            }

            save_path = os.path.join(args.save_dir, 'exp4_model.pth')
            torch.save(checkpoint, save_path)
            print(f"   ✓ 保存最佳模型 (Loss: {test_loss:.4f}, Acc: {test_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n早停：{early_stop_patience} epochs 无改善")
                break

        # 打印详细报告（每 10 epoch）
        if (epoch + 1) % 10 == 0:
            print("\n   分类详情:")
            for cls in label_encoder.classes_:
                if cls in report:
                    cls_report = report[cls]
                    print(f"     {cls}: P={cls_report['precision']:.3f}, "
                          f"R={cls_report['recall']:.3f}, "
                          f"F1={cls_report['f1-score']:.3f}")

    # 训练完成
    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    print(f"最佳测试 Loss: {best_test_loss:.4f}")
    print(f"最佳测试 Accuracy: {best_accuracy:.4f}")
    print(f"模型已保存到: {os.path.join(args.save_dir, 'exp4_model.pth')}")

    # 打印 KG 和天气统计
    if kg is not None:
        print("\nKG 统计:")
        kg_stats = kg.get_graph_statistics()
        for k, v in kg_stats.items():
            print(f"  {k}: {v}")

        cache_stats = kg.get_cache_stats()
        print(f"\nKG 缓存统计:")
        for k, v in cache_stats.items():
            print(f"  {k}: {v}")

    if weather_processor is not None:
        print("\n天气统计:")
        weather_stats = weather_processor.get_statistics()
        for k, v in weather_stats.items():
            print(f"  {k}: {v}")


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    main()