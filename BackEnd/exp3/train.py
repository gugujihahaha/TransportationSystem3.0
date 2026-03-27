"""
exp3/train.py（简化版）
======================
1. 从exp2/cache/processed_features.pkl读取traj_21
2. 自己划分训练集、验证集、测试集（random_state=42）
3. 使用21维轨迹特征（traj_9 + OSM 12）+ 10维天气特征
"""

import argparse
import os
import pickle
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
EXP2_DIR = os.path.join(PARENT_DIR, 'exp2')
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
os.chdir(SCRIPT_DIR)

from common import (train_epoch, evaluate, compute_class_weights)
from src.model_weather import TransportationModeClassifierWithWeather
from src.weather_preprocessing import WeatherDataProcessor

TRAJECTORY_FEATURE_DIM = 21
WEATHER_FEATURE_DIM    = 10

CACHE_DIR          = os.path.join(SCRIPT_DIR, 'cache')
PROCESSED_FEATURE_CACHE = os.path.join(CACHE_DIR, 'processed_features.pkl')
EXP2_CACHE_PATH    = os.path.join(EXP2_DIR, 'cache', 'processed_features.pkl')
CLEANED_DATA_PATH  = os.path.join(PARENT_DIR, 'data', 'processed', 'cleaned_balanced.pkl')
WEATHER_CSV_PATH   = os.path.join(PARENT_DIR, 'data', 'beijing_weather_daily_2007_2012.csv')
os.makedirs(CACHE_DIR, exist_ok=True)


class TrajectoryDataset(Dataset):
    def __init__(self, all_features_and_labels,
                 traj_mean=None, traj_std=None,
                 weather_mean=None, weather_std=None,
                 stats_mean=None, stats_std=None):
        self.data = all_features_and_labels
        self.traj_mean  = traj_mean
        self.traj_std   = traj_std
        self.weather_mean = weather_mean
        self.weather_std  = weather_std
        self.stats_mean = stats_mean
        self.stats_std  = stats_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj, stats, label, weather = self.data[idx]
        traj    = traj.astype(np.float32)
        stats   = stats.astype(np.float32)
        weather = weather.astype(np.float32)

        if self.traj_mean is not None:
            traj    = (traj    - self.traj_mean)    / (self.traj_std    + 1e-8)
        if self.weather_mean is not None:
            weather = (weather - self.weather_mean) / (self.weather_std + 1e-8)
        if self.stats_mean is not None:
            stats   = (stats   - self.stats_mean)   / (self.stats_std   + 1e-8)

        return (torch.FloatTensor(traj),
                torch.FloatTensor(weather),
                torch.FloatTensor(stats),
                torch.LongTensor([label]).squeeze())


def load_data():
    """从exp2/cache读取traj_21，从cleaned_balanced.pkl读取时间戳提取天气特征"""
    if not os.path.exists(EXP2_CACHE_PATH):
        raise FileNotFoundError(f"找不到 {EXP2_CACHE_PATH}")
    if not os.path.exists(CLEANED_DATA_PATH):
        raise FileNotFoundError(f"找不到 {CLEANED_DATA_PATH}")

    print(f"\n[数据加载] 从exp2/cache读取traj_21...")
    with open(EXP2_CACHE_PATH, 'rb') as f:
        exp2_data = pickle.load(f)
    exp2_features = exp2_data[0]
    exp2_label_encoder = exp2_data[1]
    print(f"   EXP2样本数: {len(exp2_features)}")

    print(f"\n[数据加载] 从cleaned_balanced.pkl读取时间戳...")
    with open(CLEANED_DATA_PATH, 'rb') as f:
        cleaned_data = pickle.load(f)
    print(f"   原始样本数: {len(cleaned_data)}")

    # 按索引对齐，取最小值
    min_samples = min(len(exp2_features), len(cleaned_data))
    exp2_features = exp2_features[:min_samples]
    cleaned_data = cleaned_data[:min_samples]
    print(f"   对齐后样本数: {min_samples}")

    # 加载天气数据处理器
    weather_processor = WeatherDataProcessor(weather_csv_path=WEATHER_CSV_PATH)
    weather_processor.load_and_process()  # 加载天气数据

    # 提取21维轨迹特征+10维天气特征
    print(f"\n[特征提取] 提取21维轨迹特征+10维天气特征...")
    all_features = []
    for idx in tqdm(range(min_samples), desc="[特征提取]"):
        traj_21, stats, label_encoded = exp2_features[idx]
        _, _, datetime_series, _ = cleaned_data[idx]
        try:
            # 提取天气特征
            weather_features = weather_processor.get_weather_features_for_trajectory(datetime_series)
            all_features.append((traj_21, stats, label_encoded, weather_features))
        except Exception as e:
            continue

    # 使用exp2的标签编码器
    label_encoder = exp2_label_encoder

    # 过滤NaN和Inf
    all_features = [
        (traj, stats, label, weather)
        for traj, stats, label, weather in all_features
        if not (np.isnan(traj).any() or np.isinf(traj).any() or
                np.isnan(weather).any() or np.isinf(weather).any() or
                np.isnan(stats).any() or np.isinf(stats).any())
    ]

    print(f"   处理后样本数: {len(all_features)}")
    print(f"   类别数: {len(label_encoder.classes_)}")
    print(f"   类别: {list(label_encoder.classes_)}")

    # 保存缓存
    with open(PROCESSED_FEATURE_CACHE, 'wb') as f:
        pickle.dump((all_features, label_encoder, {}), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   缓存已保存: {PROCESSED_FEATURE_CACHE}")

    return all_features, label_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--clear_cache', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.clear_cache and os.path.exists(PROCESSED_FEATURE_CACHE):
        os.remove(PROCESSED_FEATURE_CACHE)

    # 1. 加载数据
    all_features, label_encoder = load_data()
    if not all_features:
        return

    # 2. 划分数据集（70/10/20，random_state=42）
    print(f"\n[数据划分] 70/10/20，random_state=42...")
    all_indices = np.arange(len(all_features))
    labels_encoded = [item[2] for item in all_features]

    train_indices, temp_indices = train_test_split(
        all_indices, test_size=0.3, random_state=42, stratify=labels_encoded
    )
    temp_labels = [labels_encoded[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.6667, random_state=42, stratify=temp_labels
    )

    train_data = [all_features[i] for i in train_indices]
    val_data   = [all_features[i] for i in val_indices]
    test_data  = [all_features[i] for i in test_indices]

    print(f"   训练集: {len(train_data)}")
    print(f"   验证集: {len(val_data)}")
    print(f"   测试集: {len(test_data)}")

    # 3. 计算归一化参数（仅基于训练集）
    traj_all    = np.vstack([s[0] for s in train_data])
    stats_all   = np.vstack([s[1] for s in train_data])
    weather_all = np.vstack([s[3] for s in train_data])

    print(f"\n[数据统计] 训练集特征统计:")
    print(f"   traj_all: shape={traj_all.shape}, min={traj_all.min():.4f}, max={traj_all.max():.4f}, mean={traj_all.mean():.4f}, std={traj_all.std():.4f}")
    print(f"   weather_all: shape={weather_all.shape}, min={weather_all.min():.4f}, max={weather_all.max():.4f}, mean={weather_all.mean():.4f}, std={weather_all.std():.4f}")
    print(f"   stats_all: shape={stats_all.shape}, min={stats_all.min():.4f}, max={stats_all.max():.4f}, mean={stats_all.mean():.4f}, std={stats_all.std():.4f}")
    print(f"   NaN检查: traj={np.isnan(traj_all).any()}, weather={np.isnan(weather_all).any()}, stats={np.isnan(stats_all).any()}")

    def safe_stats(arr):
        mean = arr.mean(0).astype(np.float32)
        std  = np.where(arr.std(0) < 1e-6, 1.0, arr.std(0)).astype(np.float32)
        return mean, std

    traj_mean,    traj_std    = safe_stats(traj_all)
    weather_mean, weather_std = safe_stats(weather_all)
    stats_mean,   stats_std   = safe_stats(stats_all)

    print(f"\n[归一化参数]")
    print(f"   traj_mean: {traj_mean[:5]}... (前5维)")
    print(f"   traj_std: {traj_std[:5]}... (前5维)")
    print(f"   weather_mean: {weather_mean}")
    print(f"   weather_std: {weather_std}")

    norm_params = dict(traj_mean=traj_mean, traj_std=traj_std,
                       weather_mean=weather_mean, weather_std=weather_std,
                       stats_mean=stats_mean, stats_std=stats_std)

    # 4. Dataset & DataLoader
    def make_loader(data, shuffle):
        ds = TrajectoryDataset(data,
                               traj_mean=traj_mean, traj_std=traj_std,
                               weather_mean=weather_mean, weather_std=weather_std,
                               stats_mean=stats_mean, stats_std=stats_std)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_data, shuffle=True)
    val_loader   = make_loader(val_data,   shuffle=False)
    test_loader  = make_loader(test_data,  shuffle=False)

    # 5. 模型
    model = TransportationModeClassifierWithWeather(
        TRAJECTORY_FEATURE_DIM, WEATHER_FEATURE_DIM, 18, args.hidden_dim, args.num_layers,
        len(label_encoder.classes_), args.dropout,
        num_segments=5, local_hidden=64, global_hidden=128,
    ).to(args.device)

    class_weights = compute_class_weights(
        label_encoder, train_data, label_index=2, mode='sqrt_inverse'
    )
    class_weights = torch.FloatTensor(class_weights).to(args.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 6. 训练
    # 加载历史最佳模型（如果存在）
    model_path = os.path.join(args.save_dir, 'exp3_model.pth')
    best_val_loss = float('inf')
    start_epoch = 0
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"\n[加载历史最佳模型] 历史最佳验证损失: {best_val_loss:.4f}")
            print(f"[继续训练] 从第 {start_epoch} 轮开始")
        except Exception as e:
            print(f"\n[警告] 无法加载历史最佳模型: {e}")
            best_val_loss = float('inf')
            start_epoch = 0
    else:
        print(f"\n[训练] 未找到历史最佳模型，从头开始训练")

    epochs_no_improve = 0
    patience = args.patience

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        val_loss, val_acc, val_report, _, _ = evaluate(
            model, val_loader, criterion, args.device, label_encoder.classes_
        )

        if val_loss < best_val_loss:
            old_best = best_val_loss
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_encoder': label_encoder,
                'val_loss': val_loss,
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
            }, os.path.join(args.save_dir, 'exp3_model.pth'))
            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f} "
                  f"[✅ 保存最佳模型: {val_loss:.4f} < {old_best:.4f}]")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f} "
                  f"[❌ 未超最佳: {val_loss:.4f} >= {best_val_loss:.4f}]")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 7. 测试
    test_loss, test_acc, test_report, _, _ = evaluate(
        model, test_loader, criterion, args.device, label_encoder.classes_
    )
    print(f"\nTest Acc: {test_acc:.4f}")
    print(test_report)


if __name__ == '__main__':
    main()
