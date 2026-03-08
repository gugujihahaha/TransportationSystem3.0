"""
exp3/train.py（修复版）
======================
修复内容：
1. evaluate() 返回值从4个改为5个
2. 数据划分改为加载共享划分（shared_split.pkl）
3. 特征从 EXP2 缓存中加载21维，然后附加天气特征

训练顺序依赖：
    EXP2 → exp2/cache/processed_features.pkl
    EXP2 → data/processed/shared_split.pkl
    EXP2 → exp2/cache/spatial_data.pkl（EXP3 的天气提取不依赖此项）
"""

import argparse
import csv
import os
import pickle
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
os.chdir(SCRIPT_DIR)

from common import (BaseGeoLifePreprocessor, train_epoch, evaluate,
                    compute_class_weights)
from src.weather_preprocessing import WeatherDataProcessor
from src.model_weather import TransportationModeClassifierWithWeather

TRAJECTORY_FEATURE_DIM = 21
WEATHER_FEATURE_DIM    = 10
FIXED_SEQUENCE_LENGTH  = 50

CACHE_DIR               = os.path.join(SCRIPT_DIR, 'cache')
WEATHER_CACHE_PATH      = os.path.join(CACHE_DIR, 'weather_processor.pkl')
PROCESSED_FEATURE_CACHE = os.path.join(CACHE_DIR, 'processed_features_exp3.pkl')
EXP2_FEATURE_CACHE      = os.path.join(PARENT_DIR, 'exp2', 'cache', 'processed_features.pkl')
SHARED_SPLIT_PATH       = os.path.join(PARENT_DIR, 'data', 'processed', 'shared_split.pkl')
# EXP3 需要 cleaned_balanced.pkl 中的时间戳信息
CLEANED_DATA_PATH       = os.path.join(PARENT_DIR, 'data', 'processed', 'cleaned_balanced.pkl')
os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# Dataset
# ============================================================
class TrajectoryDatasetWithWeather(Dataset):
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
        # 格式: (traj_21, weather_10, stats_18, label_encoded)
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
def load_data(weather_path, cleaning_mode='balanced'):
    """
    加载策略：
    1. 优先加载 EXP3 完整特征缓存
    2. 若无缓存：从 EXP2 特征缓存获取21维特征 + 从 cleaned_balanced.pkl 获取时间戳
    """
    # 优先加载 EXP3 缓存
    if os.path.exists(PROCESSED_FEATURE_CACHE):
        print(f"\n========== 加载 EXP3 特征缓存 ==========")
        try:
            with open(PROCESSED_FEATURE_CACHE, 'rb') as f:
                all_data, label_encoder, cleaning_stats = pickle.load(f)
            print(f"✅ 缓存加载完成: {len(all_data)} 个样本")
            return all_data, label_encoder, cleaning_stats
        except Exception as e:
            print(f"⚠️ 缓存加载失败 ({e})，重新构建")

    # 检查依赖
    if not os.path.exists(EXP2_FEATURE_CACHE):
        raise FileNotFoundError(
            f"EXP2 特征缓存不存在: {EXP2_FEATURE_CACHE}\n"
            "请先运行 exp2/train.py"
        )

    print(f"\n========== 从 EXP2 缓存构建 EXP3 数据 ==========")
    with open(EXP2_FEATURE_CACHE, 'rb') as f:
        exp2_features, label_encoder, cleaning_stats = pickle.load(f)
    print(f"✅ EXP2 特征加载完成: {len(exp2_features)} 个样本")

    # 加载 cleaned_balanced.pkl 获取时间戳
    # cleaned_balanced 格式：(traj_9, stats_18, datetime_series, label_str)
    # 注意：两个缓存的样本数和顺序必须一致（都来自同一个 base_segments + Exp2DataAdapter）
    if not os.path.exists(CLEANED_DATA_PATH):
        raise FileNotFoundError(
            f"找不到 {CLEANED_DATA_PATH}\n"
            "天气特征需要时间戳，请确保此文件存在。"
        )

    print(f"\n========== 加载时间戳数据 ==========")
    with open(CLEANED_DATA_PATH, 'rb') as f:
        cleaned_data = pickle.load(f)
    print(f"✅ 加载完成: {len(cleaned_data)} 个样本")

    # 验证样本数一致
    if len(exp2_features) != len(cleaned_data):
        print(f"⚠️ 警告：EXP2特征={len(exp2_features)}, cleaned_data={len(cleaned_data)}")
        print("   将使用较小的样本数")
        n = min(len(exp2_features), len(cleaned_data))
        exp2_features = exp2_features[:n]
        cleaned_data  = cleaned_data[:n]

    # 加载天气处理器
    weather_processor = None
    if os.path.exists(WEATHER_CACHE_PATH):
        try:
            with open(WEATHER_CACHE_PATH, 'rb') as f:
                weather_processor = pickle.load(f)
            print(f"✅ 天气处理器从缓存加载")
        except Exception:
            weather_processor = None

    if weather_processor is None:
        print(f"\n========== 处理天气数据 ==========")
        weather_processor = WeatherDataProcessor(weather_path)
        weather_processor.load_and_process()
        with open(WEATHER_CACHE_PATH, 'wb') as f:
            pickle.dump(weather_processor, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 提取 EXP3 特征
    print(f"\n========== 提取天气特征 ==========")
    zero_weather = np.zeros((FIXED_SEQUENCE_LENGTH, WEATHER_FEATURE_DIM), dtype=np.float32)
    all_data = []

    for i, ((traj_21, _, stats, label_encoded), (_, _, datetime_series, _)) in enumerate(
            tqdm(zip(exp2_features, cleaned_data), total=len(exp2_features), desc="[附加天气]")):
        try:
            if datetime_series is not None and len(datetime_series) > 0:
                weather_feat = weather_processor.get_weather_features_for_trajectory(
                    datetime_series
                )
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
            all_data.append((traj_21, weather_feat, stats, label_encoded))
        except Exception:
            # 天气特征提取失败，用零填充（不丢弃样本，保持索引对齐）
            all_data.append((traj_21, zero_weather.copy(), stats, label_encoded))

    print(f"✅ EXP3 特征提取完成: {len(all_data)} 个样本")

    with open(PROCESSED_FEATURE_CACHE, 'wb') as f:
        pickle.dump((all_data, label_encoder, cleaning_stats),
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ EXP3 缓存已保存: {PROCESSED_FEATURE_CACHE}")

    return all_data, label_encoder, cleaning_stats


def load_shared_split():
    if not os.path.exists(SHARED_SPLIT_PATH):
        raise FileNotFoundError(
            f"共享划分不存在: {SHARED_SPLIT_PATH}\n"
            "请先运行 exp2/train.py（会自动生成）"
        )
    with open(SHARED_SPLIT_PATH, 'rb') as f:
        return pickle.load(f)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weather_path',  default='../data/beijing_weather_daily_2007_2012.csv')
    parser.add_argument('--cleaning_mode', default='balanced')
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--epochs',        type=int,   default=150)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--hidden_dim',    type=int,   default=128)
    parser.add_argument('--num_layers',    type=int,   default=2)
    parser.add_argument('--dropout',       type=float, default=0.3)
    parser.add_argument('--patience',      type=int,   default=25)
    parser.add_argument('--save_dir',      default='checkpoints')
    parser.add_argument('--device',        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--clear_cache',   action='store_true')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.clear_cache:
        for f in [WEATHER_CACHE_PATH, PROCESSED_FEATURE_CACHE]:
            if os.path.exists(f):
                os.remove(f)

    print("\n" + "=" * 60)
    print("Exp3 训练 (21维轨迹+空间 + 10维天气)")
    print("=" * 60)

    # 1. 加载数据
    all_data, label_encoder, _ = load_data(args.weather_path, args.cleaning_mode)

    # 2. 加载共享划分
    print(f"\n[共享划分] 加载...")
    split = load_shared_split()
    train_indices = split['train_indices']
    val_indices   = split['val_indices']
    test_indices  = split['test_indices']
    print(f"   Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    n = len(all_data)
    assert max(train_indices + val_indices + test_indices) < n, \
        f"索引越界：EXP3数据量={n}"

    # 3. 提取各集合样本
    train_raw  = [all_data[i] for i in train_indices]
    val_data   = [all_data[i] for i in val_indices]
    test_data  = [all_data[i] for i in test_indices]

    # 4. 仅对训练集过采样
    train_str = [(t, w, s, label_encoder.inverse_transform([l])[0])
                 for t, w, s, l in train_raw]
    train_oversampled_str = BaseGeoLifePreprocessor.oversample_minority_classes(
        train_str, target_ratio=0.3, minority_classes=['Subway', 'Train']
    )
    train_data = [(t, w, s, label_encoder.transform([l])[0])
                  for t, w, s, l in train_oversampled_str]

    print(f"\n训练集过采样: {len(train_raw)} → {len(train_data)}")

    # 5. 归一化统计量
    traj_all    = np.vstack([s[0] for s in train_data])
    weather_all = np.vstack([s[1] for s in train_data])
    stats_all   = np.vstack([s[2] for s in train_data])

    def safe_stats(arr):
        mean = arr.mean(0).astype(np.float32)
        std  = np.where(arr.std(0) < 1e-6, 1.0, arr.std(0)).astype(np.float32)
        return mean, std

    traj_mean,    traj_std    = safe_stats(traj_all)
    weather_mean, weather_std = safe_stats(weather_all)
    stats_mean,   stats_std   = safe_stats(stats_all)

    norm_params = dict(traj_mean=traj_mean, traj_std=traj_std,
                       weather_mean=weather_mean, weather_std=weather_std,
                       stats_mean=stats_mean, stats_std=stats_std)

    # 6. Dataset & DataLoader
    def make_loader(data, shuffle):
        ds = TrajectoryDatasetWithWeather(
            data,
            traj_mean=traj_mean, traj_std=traj_std,
            weather_mean=weather_mean, weather_std=weather_std,
            stats_mean=stats_mean, stats_std=stats_std
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_data, shuffle=True)
    val_loader   = make_loader(val_data,   shuffle=False)
    test_loader  = make_loader(test_data,  shuffle=False)

    print(f"\nTrain={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # 7. 模型
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

    class_weights = compute_class_weights(
        label_encoder, train_data, label_index=3, mode='sqrt_inverse'
    ).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: (ep+1)/10 if ep < 10 else 1.0
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-5
    )

    CHECKPOINT_PATH = os.path.join(args.save_dir, 'exp3_model.pth')
    best_val_acc = 0.0
    epochs_no_improve = 0
    consecutive_nan = 0

    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/exp3_training_log.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc',
                                 'val_loss', 'val_acc', 'lr'])

    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )

        if np.isnan(train_loss) or np.isinf(train_loss):
            consecutive_nan += 1
            if consecutive_nan >= 3:
                break
            if os.path.exists(CHECKPOINT_PATH):
                prev = torch.load(CHECKPOINT_PATH, map_location=args.device, weights_only=False)
                model.load_state_dict(prev['model_state_dict'])
            continue
        else:
            consecutive_nan = 0

        # ✅ 修复：evaluate 返回5个值
        val_loss, val_acc, val_report, _, _ = evaluate(
            model, val_loader, criterion, args.device, label_encoder.classes_
        )

        if np.isnan(val_loss) or np.isinf(val_loss):
            consecutive_nan += 1
            if consecutive_nan >= 3:
                break
            continue
        else:
            consecutive_nan = 0

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if epoch < 10:
            warmup_scheduler.step()
        else:
            if epoch == 10:
                plateau_scheduler.num_bad_epochs = 0
            plateau_scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"学习率: {current_lr:.6f}")

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, f"{train_loss:.4f}", f"{train_acc:.4f}",
                                     f"{val_loss:.4f}", f"{val_acc:.4f}", f"{current_lr:.6f}"])

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
            print(f"✓ 保存最佳模型 val_acc={val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"⏳ 未改善: {epochs_no_improve}/{args.patience}")
            if epochs_no_improve >= args.patience:
                print("🛑 Early stopping")
                break

    print("\n" + "=" * 60)
    print("最终测试集评估")
    print("=" * 60)
    best_ckpt = torch.load(CHECKPOINT_PATH, map_location=args.device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    # ✅ 修复：evaluate 返回5个值
    test_loss, test_acc, test_report, all_preds, all_labels = evaluate(
        model, test_loader, criterion, args.device, label_encoder.classes_
    )
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    for cls in label_encoder.classes_:
        if cls in test_report:
            m = test_report[cls]
            print(f"  {cls:15s}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1-score']:.4f}")


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    main()