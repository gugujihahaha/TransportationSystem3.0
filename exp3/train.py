"""
训练脚本 (Exp3 - 轨迹+空间+天气)

与 exp2 的关系：
    - 轨迹特征完全复用 exp2 的点级融合特征 (21维)
    - 新增天气特征通道 (10维，日级别广播到序列)
    - 控制变量：模型结构、超参数、数据划分方式与 exp2 一致

已修复的Bug：
    1. oversample 移到 split 之后，只对训练集做，避免测试集数据泄露
    2. early stopping 改为监控 val_acc（与exp1/exp2一致）
    3. plateau_scheduler mode='max'（监控acc）
    4. plateau 在 warmup 结束时 reset 内部计数器
    5. labels_stratify 取正确的索引（-1）
    6. evaluate.py 的 split 方式与 train.py 保持一致（70/10/20）
    7. patience 使用 args.patience，不硬编码
"""
import os
import sys
import argparse
import csv
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ========================== 路径设置 ==========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
os.chdir(SCRIPT_DIR)
# ==============================================================

from common import (BaseGeoLifePreprocessor, Exp3DataAdapter,
                    train_epoch, evaluate, compute_class_weights)
from common.train_utils import compute_feature_stats

# exp3专用模块
from src.weather_preprocessing import WeatherDataProcessor
from src.model_weather import TransportationModeClassifierWithWeather

# exp2 模块（复用空间特征提取器）
from exp2.src.feature_extraction import FeatureExtractor
from exp2.src.osm_feature_extractor import OsmSpatialExtractor

# ========================== 特征维度常量 ==========================
TRAJECTORY_FEATURE_DIM = 21   # 复用exp2：9轨迹+12空间，点级融合
WEATHER_FEATURE_DIM    = 10   # 日级天气特征
SPATIAL_PLACEHOLDER    = 1    # exp2的占位符，exp3直接继承
FIXED_SEQUENCE_LENGTH  = 50
# ==================================================================

# ========================== 缓存配置 ==========================
CACHE_DIR = 'cache'
EXP2_SPATIAL_CACHE   = os.path.join(PARENT_DIR, 'exp2', 'cache', 'spatial_data.pkl')
EXP2_GRID_CACHE      = os.path.join(PARENT_DIR, 'exp2', 'cache', 'spatial_grid_cache.pkl')
WEATHER_CACHE_PATH    = os.path.join(CACHE_DIR, 'weather_processor.pkl')
PROCESSED_FEATURE_CACHE = os.path.join(CACHE_DIR, 'processed_features_exp3.pkl')
os.makedirs(CACHE_DIR, exist_ok=True)
# ==============================================================


# ============================================================
# Dataset
# ============================================================
class TrajectoryDatasetWithWeather(Dataset):
    """
    数据格式: (traj_21dim, weather_10dim, stats_18dim, label_encoded)
    traj_21dim 已经是 exp2 的点级融合特征（9轨迹+12空间），不需要再单独处理空间特征
    """
    def __init__(self, data,
                 traj_mean=None, traj_std=None,
                 weather_mean=None, weather_std=None,
                 stats_mean=None, stats_std=None):
        self.data = data
        self.traj_mean    = traj_mean
        self.traj_std     = traj_std
        self.weather_mean = weather_mean
        self.weather_std  = weather_std
        self.stats_mean   = stats_mean
        self.stats_std    = stats_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj, weather, stats, label = self.data[idx]
        traj    = traj.astype(np.float32).copy()
        weather = weather.astype(np.float32).copy()
        stats   = stats.astype(np.float32).copy()

        if self.traj_mean is not None:
            traj = (traj - self.traj_mean) / self.traj_std
        if self.weather_mean is not None:
            weather = (weather - self.weather_mean) / self.weather_std
        if self.stats_mean is not None:
            stats = (stats - self.stats_mean) / self.stats_std

        return (torch.FloatTensor(traj),
                torch.FloatTensor(weather),
                torch.FloatTensor(stats),
                torch.LongTensor([label])[0])


# ============================================================
# 数据加载
# ============================================================
def load_data(geolife_root: str, osm_path: str, weather_path: str,
              max_users=None, use_base_data=True, cleaning_mode='balanced'):
    """
    加载数据策略：
    1. 用 Exp3DataAdapter 处理 base_segments，得到带时间戳的 processed_with_time
    2. 加载 exp2 的空间特征提取器对象（复用网格缓存）
    3. 在同一个循环里处理每个segment，提取21维融合特征和天气特征
    4. NaN过滤后保存 exp3 完整缓存

    控制变量：exp3 = exp2的空间特征（完全一致）+ 新增天气特征
    """

    # ===== 优先检查 exp3 完整缓存 =====
    if os.path.exists(PROCESSED_FEATURE_CACHE):
        print(f"\n========== 加载 Exp3 特征缓存 ==========")
        try:
            with open(PROCESSED_FEATURE_CACHE, 'rb') as f:
                all_data, label_encoder, cleaning_stats = pickle.load(f)
            print(f"✅ 缓存加载完成: {len(all_data)} 个样本")
            return all_data, label_encoder, cleaning_stats
        except Exception as e:
            print(f"⚠️ 缓存加载失败 ({e})，重新构建")

    # ===== 加载天气处理器 =====
    weather_processor = None
    if os.path.exists(WEATHER_CACHE_PATH):
        print(f"\n========== 天气处理器加载 (从缓存) ==========")
        try:
            with open(WEATHER_CACHE_PATH, 'rb') as f:
                weather_processor = pickle.load(f)
            print("✅ 天气处理器从缓存加载完成")
        except Exception:
            weather_processor = None

    if weather_processor is None:
        print(f"\n========== 天气数据处理 ==========")
        weather_processor = WeatherDataProcessor(weather_path)
        weather_processor.load_and_process()
        with open(WEATHER_CACHE_PATH, 'wb') as f:
            pickle.dump(weather_processor, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ===== 加载 exp2 的空间特征提取器对象 =====
    print(f"\n========== 加载 Exp2 空间特征提取器 ==========")
    if not os.path.exists(EXP2_SPATIAL_CACHE):
        raise FileNotFoundError(
            f"找不到 exp2 空间特征提取器: {EXP2_SPATIAL_CACHE}\n"
            "请先确保 exp2/train.py 已经运行并生成缓存。"
        )
    
    with open(EXP2_SPATIAL_CACHE, 'rb') as f:
        spatial_extractor = pickle.load(f)
    
    # 加载 exp2 的网格缓存
    if os.path.exists(EXP2_GRID_CACHE):
        spatial_extractor.load_cache(EXP2_GRID_CACHE)
        print(f"✅ exp2 网格缓存加载完成: {EXP2_GRID_CACHE}")
    else:
        print(f"⚠️ exp2 网格缓存不存在: {EXP2_GRID_CACHE}")
    
    # 初始化 exp2 的特征提取器
    feature_extractor = FeatureExtractor(spatial_extractor)
    print("✅ exp2 特征提取器初始化完成")

    # ===== 用 Exp3DataAdapter 处理 base_segments =====
    BASE_DATA_PATH = os.path.join(PARENT_DIR, 'data', 'processed', 'base_segments.pkl')
    if not os.path.exists(BASE_DATA_PATH):
        raise FileNotFoundError(
            f"找不到基础数据: {BASE_DATA_PATH}\n"
            "请先运行 scripts/generate_base_data.py"
        )

    print(f"\n========== 处理基础数据 ==========")
    base_segments = BaseGeoLifePreprocessor.load_from_cache(BASE_DATA_PATH)
    adapter = Exp3DataAdapter(enable_cleaning=True, cleaning_mode=cleaning_mode)
    processed_with_time = adapter.process_segments(base_segments)
    cleaning_stats = adapter.get_cleaning_stats()

    # ===== 过滤类别 =====
    TARGET_MODES = {'Walk', 'Bike', 'Bus', 'Car & taxi', 'Train', 'Subway'}
    processed_filtered = [s for s in processed_with_time if s[3] in TARGET_MODES]
    print(f"✅ 过滤后样本数: {len(processed_filtered)}")

    # ===== 在同一个循环里处理每个segment =====
    print(f"\n========== 提取特征并拼接天气 ==========")
    
    # 初始化 label_encoder
    all_labels = [s[3] for s in processed_filtered]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    
    all_data = []
    zero_weather = np.zeros((FIXED_SEQUENCE_LENGTH, WEATHER_FEATURE_DIM), dtype=np.float32)
    
    for traj_9, stats, datetime_series, label_str in tqdm(
            processed_filtered, desc="[提取特征]",
            total=len(processed_filtered)):
        try:
            # 提取21维融合特征（复用exp2的空间查询缓存）
            traj_21, _ = feature_extractor.extract_features(traj_9)
            
            # NaN过滤
            if np.isnan(traj_21).any() or np.isinf(traj_21).any():
                continue
            
            # 获取天气特征
            if datetime_series is not None and len(datetime_series) > 0:
                weather_feat = weather_processor.get_weather_features_for_trajectory(
                    datetime_series
                )
                # 确保长度匹配
                T = traj_21.shape[0]
                if weather_feat.shape[0] != T:
                    if weather_feat.shape[0] > T:
                        weather_feat = weather_feat[:T]
                    else:
                        pad = np.zeros((T - weather_feat.shape[0], WEATHER_FEATURE_DIM), dtype=np.float32)
                        weather_feat = np.vstack([weather_feat, pad])
            else:
                weather_feat = zero_weather.copy()
            
            weather_feat = np.nan_to_num(weather_feat, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 编码标签
            label_encoded = label_encoder.transform([label_str])[0]
            
            # exp3 最终格式: (traj_21dim, weather_10dim, stats_18dim, label_encoded)
            all_data.append((traj_21, weather_feat, stats, label_encoded))
            
        except Exception as e:
            continue

    print(f"✅ 特征提取完成: {len(all_data)} 个样本")

    # 保存缓存
    with open(PROCESSED_FEATURE_CACHE, 'wb') as f:
        pickle.dump((all_data, label_encoder, cleaning_stats), f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Exp3 特征缓存已保存: {PROCESSED_FEATURE_CACHE}")

    return all_data, label_encoder, cleaning_stats


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='训练 Exp3 (轨迹+空间+天气)')
    parser.add_argument('--geolife_root', default='../data/Geolife Trajectories 1.3')
    parser.add_argument('--osm_path',     default='../data/exp2.geojson')  # 复用exp2的OSM
    parser.add_argument('--weather_path', default='../data/beijing_weather_daily_2007_2012.csv')
    parser.add_argument('--use_base_data', action='store_true', default=True)
    parser.add_argument('--cleaning_mode', default='balanced',
                        choices=['strict', 'balanced', 'gentle'])
    parser.add_argument('--max_users',  type=int,   default=None)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--epochs',     type=int,   default=150)
    parser.add_argument('--lr',         type=float, default=1e-4)   # 与exp1/exp2一致
    parser.add_argument('--hidden_dim', type=int,   default=128)
    parser.add_argument('--num_layers', type=int,   default=2)
    parser.add_argument('--dropout',    type=float, default=0.3)
    parser.add_argument('--patience',   type=int,   default=25)     # 与exp1/exp2一致
    parser.add_argument('--save_dir',   default='checkpoints')
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--clear_cache', action='store_true')
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Exp3 训练 (轨迹21维 + 天气10维)")
    print(f"设备: {args.device} | lr: {args.lr} | patience: {args.patience}")
    print("=" * 80)

    if args.clear_cache:
        for f in [WEATHER_CACHE_PATH, PROCESSED_FEATURE_CACHE]:
            if os.path.exists(f):
                os.remove(f)
                print(f"  已删除缓存: {f}")

    # ===== 加载数据 =====
    all_data, label_encoder, cleaning_stats = load_data(
        args.geolife_root, args.osm_path, args.weather_path,
        args.max_users, args.use_base_data, args.cleaning_mode
    )

    if not all_data:
        print("❌ 没有有效数据，退出")
        return

    # ===== ✅ 修复1: 先split，再oversample（只对训练集） =====
    all_indices = np.arange(len(all_data))
    labels_stratify = [item[-1] for item in all_data]   # label_encoded 在最后一位

    train_indices, temp_indices = train_test_split(
        all_indices, test_size=0.3, random_state=42, stratify=labels_stratify
    )
    temp_labels = [labels_stratify[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.6667, random_state=42, stratify=temp_labels
    )

    # oversample 只对训练集做，val/test 保持原样
    train_segments_raw = [all_data[i] for i in train_indices]
    # 转换格式以适配 oversample 接口：需要 (feat, feat, label_str, ...) 格式
    # oversample 需要字符串标签，临时转回
    train_with_str = [(t, w, s, label_encoder.inverse_transform([l])[0])
                      for t, w, s, l in train_segments_raw]
    train_oversampled = BaseGeoLifePreprocessor.oversample_minority_classes(
        train_with_str,
        target_ratio=0.3,
        minority_classes=['Subway', 'Train']
    )
    # 转回 encoded
    train_data_final = [
        (t, w, s, label_encoder.transform([l])[0])
        for t, w, s, l in train_oversampled
    ]
    val_data   = [all_data[i] for i in val_indices]
    test_data  = [all_data[i] for i in test_indices]

    print(f"\n✅ 数据划分完成 (oversample仅对训练集):")
    print(f"  Train (oversampled): {len(train_data_final)}")
    print(f"  Val:                 {len(val_data)}")
    print(f"  Test:                {len(test_data)}")

    # ===== 计算归一化统计量（仅基于训练集）=====
    traj_all    = np.vstack([s[0] for s in train_data_final])
    weather_all = np.vstack([s[1] for s in train_data_final])
    stats_all   = np.vstack([s[2] for s in train_data_final])

    def safe_stats(arr):
        mean = arr.mean(axis=0).astype(np.float32)
        std  = arr.std(axis=0).astype(np.float32)
        std  = np.where(std < 1e-6, 1.0, std)
        return mean, std

    traj_mean,    traj_std    = safe_stats(traj_all)
    weather_mean, weather_std = safe_stats(weather_all)
    stats_mean,   stats_std   = safe_stats(stats_all)

    norm_params = {
        'traj_mean': traj_mean, 'traj_std': traj_std,
        'weather_mean': weather_mean, 'weather_std': weather_std,
        'stats_mean': stats_mean, 'stats_std': stats_std
    }
    print(f"✅ 归一化统计量计算完成（基于 {len(train_data_final)} 个训练样本）")

    # ===== Dataset & DataLoader =====
    def make_loader(data, shuffle):
        ds = TrajectoryDatasetWithWeather(
            data,
            traj_mean=traj_mean, traj_std=traj_std,
            weather_mean=weather_mean, weather_std=weather_std,
            stats_mean=stats_mean, stats_std=stats_std
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_data_final, shuffle=True)
    val_loader   = make_loader(val_data,          shuffle=False)
    test_loader  = make_loader(test_data,         shuffle=False)

    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ===== 类别分布 =====
    print(f"\n类别分布 (训练集, oversampled):")
    for cls in label_encoder.classes_:
        enc = label_encoder.transform([cls])[0]
        cnt = sum(1 for _, _, _, l in train_data_final if l == enc)
        print(f"  {cls:15s}: {cnt}")

    # ===== 模型 =====
    model = TransportationModeClassifierWithWeather(
        trajectory_feature_dim=TRAJECTORY_FEATURE_DIM,
        weather_feature_dim=WEATHER_FEATURE_DIM,
        segment_stats_dim=18,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=len(label_encoder.classes_),
        dropout=args.dropout,
        num_segments=5,
        local_hidden=64,
        global_hidden=128,
    ).to(args.device)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ===== 损失函数（用训练集的标签分布）=====
    class_weights = compute_class_weights(
        label_encoder,
        [(t, w, s, l) for t, w, s, l in train_data_final],
        label_index=-1,
        mode='sqrt_inverse'
    ).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ===== 优化器 =====
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Warmup: 前10轮线性升温（与exp1/exp2一致）
    warmup_epochs = 10
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: (ep + 1) / warmup_epochs if ep < warmup_epochs else 1.0
    )

    # ===== ✅ 修复2: plateau 监控 val_acc (mode='max')，与exp1/exp2一致 =====
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-5
    )

    # ===== Early Stopping =====
    CHECKPOINT_PATH = os.path.join(args.save_dir, 'exp3_model.pth')
    best_val_acc = 0.0
    epochs_no_improve = 0
    patience = args.patience   # ✅ 修复3: 使用 args.patience，不硬编码

    if os.path.exists(CHECKPOINT_PATH):
        try:
            prev = torch.load(CHECKPOINT_PATH, map_location=args.device, weights_only=False)
            if prev.get('model_config', {}).get('num_classes') == len(label_encoder.classes_):
                model.load_state_dict(prev['model_state_dict'])
                best_val_acc = prev.get('val_acc', 0.0)
                print(f"✅ 加载历史最佳权重，val_acc={best_val_acc:.4f}")
            else:
                print("⚠️ 类别数不匹配，从零开始")
        except Exception as e:
            print(f"⚠️ 历史模型加载失败 ({e})，从零开始")

    # NaN 连续检测
    consecutive_nan = 0
    MAX_NAN = 3

    # 训练日志
    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/exp3_training_log.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])

    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )

        # NaN 检测
        if np.isnan(train_loss) or np.isinf(train_loss):
            consecutive_nan += 1
            print(f"⚠️ 训练损失异常 ({train_loss})，连续NaN: {consecutive_nan}/{MAX_NAN}")
            if consecutive_nan >= MAX_NAN:
                print("🛑 连续NaN过多，停止训练")
                break
            if os.path.exists(CHECKPOINT_PATH):
                prev = torch.load(CHECKPOINT_PATH, map_location=args.device, weights_only=False)
                model.load_state_dict(prev['model_state_dict'])
                print("✅ 已恢复到最佳模型")
            continue
        else:
            consecutive_nan = 0

        val_loss, val_report, _, _ = evaluate(
            model, val_loader, criterion, args.device, label_encoder.classes_
        )
        val_acc = val_report['accuracy']

        if np.isnan(val_loss) or np.isinf(val_loss):
            consecutive_nan += 1
            print(f"⚠️ 验证损失异常，连续NaN: {consecutive_nan}/{MAX_NAN}")
            if consecutive_nan >= MAX_NAN:
                print("🛑 连续NaN过多，停止训练")
                break
            continue
        else:
            consecutive_nan = 0

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # ===== ✅ 修复4: plateau 在 warmup 结束时 reset 计数器 =====
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            if epoch == warmup_epochs:
                plateau_scheduler.num_bad_epochs = 0  # reset warmup期间的历史
            plateau_scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"学习率: {current_lr:.6f}")

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch+1, f"{train_loss:.4f}", f"{train_acc:.4f}",
                f"{val_loss:.4f}", f"{val_acc:.4f}", f"{current_lr:.6f}"
            ])

        # ===== ✅ 修复5: 监控 val_acc =====
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_encoder': label_encoder,
                'val_acc': val_acc,
                'norm_params': norm_params,
                'model_config': {
                    'trajectory_feature_dim': TRAJECTORY_FEATURE_DIM,
                    'weather_feature_dim': WEATHER_FEATURE_DIM,
                    'segment_stats_dim': 18,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'num_classes': len(label_encoder.classes_),
                    'dropout': args.dropout,
                }
            }, CHECKPOINT_PATH)
            print("✓ 保存最佳模型 (val_acc)")
        else:
            epochs_no_improve += 1
            print(f"⏳ 未改善: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping (patience={patience})")
                break

    # ===== 最终测试集评估 =====
    print("\n" + "=" * 80)
    print("最终测试集评估")
    print("=" * 80)
    best_ckpt = torch.load(CHECKPOINT_PATH, map_location=args.device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    print("✅ 已加载最佳 checkpoint")

    test_loss, test_report, all_preds, all_labels = evaluate(
        model, test_loader, criterion, args.device, label_encoder.classes_
    )
    test_acc = test_report['accuracy']
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print("\n各类别指标:")
    for cls in label_encoder.classes_:
        if cls in test_report:
            m = test_report[cls]
            print(f"  {cls:15s}: P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1-score']:.4f}")

    print(f"\n最佳验证准确率: {best_val_acc:.4f}")
    print(f"模型保存路径:   {CHECKPOINT_PATH}")


if __name__ == '__main__':
    warnings.simplefilter('ignore', FutureWarning)
    warnings.simplefilter('ignore', UserWarning)
    main()