"""
训练脚本 (混合优化版) - 最终稳定版 - **已集成过拟合修复**

核心优化:
1. ✅ 修正了标签过滤逻辑，确保使用 GeoLifeDataLoader 中定义的全部七个归一化类别。
2. ✅ 修正了数据预处理，强制轨迹段张量长度一致（[50, 9]），解决 DataLoader RuntimeError。
3. ✅ 新增中间数据缓存 (已加载和预处理的轨迹段列表)，避免重复 I/O。
4. ✅ 网格缓存 KG 特征 (二级缓存)。
5. ✅ 最终特征缓存 (三级缓存)。
6. 🌟 **新增早停 (Early Stopping) 机制**，解决实验二过拟合问题。
7. 🌟 **新增 L2 正则化 (Weight Decay)**，提升模型泛化能力。
"""
import os
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
import pandas as pd
import numpy as np
from typing import List, Tuple

# 假设这些模块已在 src 目录下正确定义
try:
    from src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader, preprocess_trajectory_segments
    from src.feature_extraction import FeatureExtractor
    from src.model import TransportationModeClassifier
    from src.knowledge_graph import TransportationKnowledgeGraph
except ImportError:
    # 仅作提示，如果实际运行报错，请确保 src 模块可导入
    pass


# ========================== 特征维度常量 ==========================
TRAJECTORY_FEATURE_DIM = 9
KG_FEATURE_DIM = 11
# 注意：FIXED_SEQUENCE_LENGTH 必须与 data_preprocessing.py 中的一致
FIXED_SEQUENCE_LENGTH = 50
# =================================================================


# ========================== 四级缓存配置 ==========================
CACHE_DIR = 'cache'
KG_CACHE_PATH = os.path.join(CACHE_DIR, 'kg_data.pkl')
GRID_CACHE_PATH = os.path.join(CACHE_DIR, 'grid_cache.pkl')
PROCESSED_SEGMENTS_CACHE_PATH = os.path.join(CACHE_DIR, 'processed_segments.pkl')
PROCESSED_FEATURE_CACHE_PATH = os.path.join(CACHE_DIR, 'processed_features.pkl')
os.makedirs(CACHE_DIR, exist_ok=True)
# ==================================================================


class TrajectoryDataset(Dataset):
    """轨迹数据集"""

    def __init__(self, all_features_and_labels: List[Tuple[np.ndarray, np.ndarray, int]]):
        self.data = all_features_and_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trajectory_features, kg_features, label_encoded = self.data[idx]

        trajectory_tensor = torch.FloatTensor(trajectory_features)
        kg_tensor = torch.FloatTensor(kg_features)
        label_tensor = torch.LongTensor([label_encoded])[0]

        # 检查张量大小，以防万一
        if trajectory_tensor.shape != (FIXED_SEQUENCE_LENGTH, TRAJECTORY_FEATURE_DIM):
             # 理论上不会发生，因为已经在预处理中强制了
             raise RuntimeError(f"加载的轨迹张量尺寸错误: {trajectory_tensor.shape}")

        return trajectory_tensor, kg_tensor, label_tensor


def load_data(geolife_root: str, osm_path: str, max_users: int = None):
    """加载所有数据，实现四级缓存"""

    geolife_loader = GeoLifeDataLoader(geolife_root)
    users = geolife_loader.get_all_users()
    if max_users and max_users < len(users):
        users = users[:max_users]

    # ================= 阶段 1: 知识图谱构建 (一级缓存: KG对象) ==================
    kg = None
    if os.path.exists(KG_CACHE_PATH):
        print(f"\n========== 阶段 1: 知识图谱加载 (从缓存) ==========")
        try:
            with open(KG_CACHE_PATH, 'rb') as f:
                kg = pickle.load(f)
            print("✅ 知识图谱从缓存加载完成。")
            print(f"   统计: {kg.get_graph_statistics()}")

            if os.path.exists(GRID_CACHE_PATH):
                kg.load_cache(GRID_CACHE_PATH)

        except Exception as e:
            warnings.warn(f"[WARN] KG 缓存加载失败 ({e})，将重新构建。")
            if os.path.exists(KG_CACHE_PATH): os.remove(KG_CACHE_PATH)
            kg = None

    if kg is None:
        print("\n========== 阶段 1: 知识图谱构建 (重建) ==========")
        print(f"找到 {len(users)} 个用户")

        osm_loader = OSMDataLoader(osm_path)
        osm_data = osm_loader.load_osm_data()
        road_network = osm_loader.extract_road_network(osm_data)
        pois = osm_loader.extract_pois(osm_data)
        print(f"OSM 数据: {len(road_network)} 条道路, {len(pois)} 个POI")

        kg = TransportationKnowledgeGraph()
        kg.build_from_osm(road_network, pois)
        print(f"统计: {kg.get_graph_statistics()}")

        print(f"正在保存到: {KG_CACHE_PATH}")
        with open(KG_CACHE_PATH, 'wb') as f:
            pickle.dump(kg, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✅ 知识图谱缓存完成。")

    # ================= 阶段 2: 数据准备 (中间缓存和最终缓存) ==================

    # A. 尝试加载最终特征 (最高优先级)
    if os.path.exists(PROCESSED_FEATURE_CACHE_PATH):
        print(f"\n========== 阶段 2: 最终特征加载 (从缓存) ==========")
        try:
            with open(PROCESSED_FEATURE_CACHE_PATH, 'rb') as f:
                all_features_and_labels, label_encoder = pickle.load(f)
            print(f"✅ 最终特征从缓存加载完成: {len(all_features_and_labels)} 条记录")

            cache_stats = kg.get_cache_stats()
            print(f"   网格缓存统计: {cache_stats}")

            return all_features_and_labels, kg, label_encoder
        except Exception as e:
            warnings.warn(f"[WARN] 最终特征缓存加载失败 ({e})，将重新提取。")
            if os.path.exists(PROCESSED_FEATURE_CACHE_PATH): os.remove(PROCESSED_FEATURE_CACHE_PATH)


    # B. 尝试加载中间数据：预处理后的轨迹段 (新增的缓存点)
    processed_segments = None
    label_encoder = None
    min_length = 10

    if os.path.exists(PROCESSED_SEGMENTS_CACHE_PATH):
        print(f"\n========== 阶段 2.1: 轨迹段加载 (从缓存) ==========")
        try:
            with open(PROCESSED_SEGMENTS_CACHE_PATH, 'rb') as f:
                processed_segments, label_encoder = pickle.load(f)
            print(f"✅ 轨迹段缓存加载完成: {len(processed_segments)} 个有效段。")
        except Exception as e:
            warnings.warn(f"[WARN] 轨迹段缓存加载失败 ({e})，将重新加载和预处理。")
            if os.path.exists(PROCESSED_SEGMENTS_CACHE_PATH): os.remove(PROCESSED_SEGMENTS_CACHE_PATH)

    if processed_segments is None:
        # C. 重新加载和预处理轨迹段 (耗时操作)
        print("\n========== 阶段 2.1: 轨迹段加载 (重建) ==========")

        # 2.1 加载轨迹段
        print("2.1 正在加载轨迹段...")
        all_segments = []
        for user_id in tqdm(users, desc="[用户加载]"):
            labels = geolife_loader.load_labels(user_id)
            if labels.empty:
                continue

            trajectory_dir = os.path.join(geolife_root, f"Data/{user_id}/Trajectory")
            if not os.path.exists(trajectory_dir):
                continue

            for traj_file in os.listdir(trajectory_dir):
                if not traj_file.endswith('.plt'):
                    continue

                traj_path = os.path.join(trajectory_dir, traj_file)
                try:
                    trajectory = geolife_loader.load_trajectory(traj_path)
                    # segment_trajectory 会进行标签归一化/合并
                    segments = geolife_loader.segment_trajectory(trajectory, labels)
                    all_segments.extend(segments)
                except Exception:
                    continue

        print(f"   共加载 {len(all_segments)} 个轨迹段")

        # 2.2 预处理 (包括 min_length 过滤 和 长度规范化到 FIXED_SEQUENCE_LENGTH)
        print("2.2 正在预处理轨迹段...")
        processed_segments = preprocess_trajectory_segments(all_segments, min_length=min_length)

        # --- 核心修复：使用全部七个归一化类别进行过滤 ---
        final_seven_modes = {
            'Walk', 'Bike', 'Bus', 'Car & taxi', 'Train', 'Airplane', 'Other'
        }

        processed_segments = [
            (traj, label) for traj, label in processed_segments
            if label in final_seven_modes
        ]

        # 检查是否还有有效数据
        if not processed_segments:
            print(f"   剩余 0 个有效轨迹段 (请检查 min_length={min_length} 是否过大)")
            print("❌ 错误: 没有可用数据")
            return [], kg, None

        print(f"   剩余 {len(processed_segments)} 个有效轨迹段 (按七大类过滤)")

        # 2.3 创建标签编码器
        all_labels_str = [label for _, label in processed_segments]
        label_encoder = LabelEncoder()
        label_encoder.fit(all_labels_str)

        print(f"   最终编码类别 (七大类): {label_encoder.classes_}")

        # --- 关键：保存中间数据缓存 ---
        print(f"   正在保存轨迹段到: {PROCESSED_SEGMENTS_CACHE_PATH}")
        data_to_save = (processed_segments, label_encoder)
        with open(PROCESSED_SEGMENTS_CACHE_PATH, 'wb') as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✅ 轨迹段缓存完成。")


    # D. 混合特征提取 (重建/继续)
    print("\n========== 阶段 2.2: 特征提取 (重建) ==========")
    print(f"2.4 正在提取特征 (共 {len(processed_segments)} 个轨迹段)...")
    print("    提示: 首次运行会慢，后续运行会从网格缓存加速")

    feature_extractor = FeatureExtractor(kg)
    all_features_and_labels = []

    for trajectory, label_str in tqdm(processed_segments, desc="[特征提取]"):
        try:
            trajectory_features, kg_features = feature_extractor.extract_features(trajectory)

            # 最终检查特征形状，确保符合预期
            if trajectory_features.shape != (FIXED_SEQUENCE_LENGTH, TRAJECTORY_FEATURE_DIM):
                warnings.warn(f"警告: 轨迹段 {label_str} 最终特征形状错误: {trajectory_features.shape}")
                continue

            label_encoded = label_encoder.transform([label_str])[0]
            all_features_and_labels.append((trajectory_features, kg_features, label_encoded))
        except Exception as e:
            warnings.warn(f"警告: 轨迹段 {label_str} 特征提取失败 ({e})")
            continue

    # 2.5 保存最终特征缓存
    print(f"2.5 正在保存最终特征到: {PROCESSED_FEATURE_CACHE_PATH}")
    data_to_save = (all_features_and_labels, label_encoder)
    with open(PROCESSED_FEATURE_CACHE_PATH, 'wb') as f:
        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("✅ 最终特征缓存完成。")

    # 2.6 保存网格缓存
    print(f"2.6 正在保存网格缓存到: {GRID_CACHE_PATH}")
    kg.save_cache(GRID_CACHE_PATH)

    # 打印缓存统计
    cache_stats = kg.get_cache_stats()
    print(f"\n缓存统计:")
    print(f"  - 网格数量: {cache_stats['cache_size']}")
    print(f"  - 缓存命中: {cache_stats['cache_hits']}")
    print(f"  - 缓存未命中: {cache_stats['cache_misses']}")
    print(f"  - 命中率: {cache_stats['hit_rate']}")
    print(f"  - 缓存内存: {cache_stats['cache_memory_mb']:.2f} MB")

    return all_features_and_labels, kg, label_encoder


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for trajectory_features, kg_features, labels in tqdm(dataloader, desc="   [训练]"):
        trajectory_features = trajectory_features.to(device)
        kg_features = kg_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(trajectory_features, kg_features)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device, label_encoder):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for trajectory_features, kg_features, labels in tqdm(dataloader, desc="   [评估]"):
            trajectory_features = trajectory_features.to(device)
            kg_features = kg_features.to(device)
            labels = labels.to(device)

            logits = model(trajectory_features, kg_features)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    report = classification_report(
        all_labels, all_preds,
        target_names=label_encoder.classes_,
        output_dict=True, zero_division=0
    )

    return avg_loss, report, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='训练交通方式识别模型 (混合优化版)')
    parser.add_argument('--geolife_root', type=str,
                        default='../data/Geolife Trajectories 1.3')
    parser.add_argument('--osm_path', type=str,
                        default='../data/exp2.geojson')
    parser.add_argument('--max_users', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-4, # 🌟 新增L2正则化参数
                        help='L2 regularization strength (weight decay)')
    parser.add_argument('--patience', type=int, default=10, # 🌟 新增早停耐心值
                        help='Number of epochs to wait for improvement before stopping.')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--clear_cache', action='store_true',
                        help='清空所有缓存并重新构建')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 可选：清空缓存
    if args.clear_cache:
        print("\n清空所有缓存...")
        for cache_file in [
            KG_CACHE_PATH,
            GRID_CACHE_PATH,
            PROCESSED_SEGMENTS_CACHE_PATH,
            PROCESSED_FEATURE_CACHE_PATH
        ]:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"  - 已删除: {cache_file}")

    # 加载数据
    all_features_and_labels, kg, label_encoder = load_data(
        args.geolife_root, args.osm_path, args.max_users
    )

    if len(all_features_and_labels) == 0:
        print("❌ 错误: 没有可用数据")
        return

    print("\n========== 阶段 3: 模型训练 ==========")
    num_classes = len(label_encoder.classes_)
    print(f"类别数: {num_classes}")
    print(f"类别: {label_encoder.classes_}")

    dataset = TrajectoryDataset(all_features_and_labels)

    # 划分数据集 (使用 LabelEncoder 确保分层抽样准确)
    labels_for_stratify = [label for _, _, label in all_features_and_labels]

    unique_labels, counts = np.unique(labels_for_stratify, return_counts=True)
    if any(c < 2 for c in counts):
        warnings.warn("警告: 某些类别样本数少于2个，分层抽样可能失败。将使用非分层抽样。")
        stratify_labels = None
    else:
        stratify_labels = labels_for_stratify

    train_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42,
        stratify=stratify_labels
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # 打印各类别样本数
    train_labels = np.array(labels_for_stratify)[train_indices]
    test_labels = np.array(labels_for_stratify)[test_indices]

    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    unique_test, counts_test = np.unique(test_labels, return_counts=True)

    print("\n各类别样本数 (训练集/测试集):")
    for cls in label_encoder.classes_:
        encoded_cls = label_encoder.transform([cls])[0]
        train_count = counts_train[unique_train == encoded_cls][0] if encoded_cls in unique_train else 0
        test_count = counts_test[unique_test == encoded_cls][0] if encoded_cls in unique_test else 0
        print(f"  {cls}: {train_count} / {test_count}")

    print(f"\n训练集总数: {len(train_dataset)}, 测试集总数: {len(test_dataset)}")


    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # 创建模型
    model = TransportationModeClassifier(
        trajectory_feature_dim=TRAJECTORY_FEATURE_DIM,
        kg_feature_dim=KG_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(args.device)

    print(f"模型参数: {sum(p.numel() for p in model.parameters())}")
    print(f"训练设备: {args.device}")

    # 使用交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # 🌟 修改点 1: 在优化器中加入 weight_decay 参数实现 L2 正则化
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 训练循环
    best_test_loss = float('inf')
    # 🌟 修改点 2: 早停机制初始化
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\n[EPOCH {epoch + 1}/{args.epochs}]")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        print(f"   训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")

        test_loss, test_report, test_preds, test_labels = evaluate(
            model, test_loader, criterion, args.device, label_encoder
        )
        test_acc = test_report['accuracy']
        print(f"   测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")

        scheduler.step(test_loss)

        # 🌟 修改点 3: 早停逻辑和最佳模型保存
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0 # 损失下降，重置计数器
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
                'test_acc': test_acc,
                'label_encoder': label_encoder,
            }, os.path.join(args.save_dir, 'exp2_model.pth'))
            print("   ✅ 保存最佳模型")
        else:
            patience_counter += 1
            print(f"   ⚠️ 测试损失未改善. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"   ❌ 早停触发! 连续 {args.patience} 个 Epoch 损失未下降。")
                break # 终止训练循环


        if (epoch + 1) % 10 == 0:
            print("\n   分类报告:")
            print(classification_report(
                test_labels, test_preds,
                target_names=label_encoder.classes_, zero_division=0
            ))

    print(f"\n========== 训练完成 ==========")
    print(f"最佳测试损失: {best_test_loss:.4f}")

    # 最终缓存统计
    cache_stats = kg.get_cache_stats()
    print(f"\n最终缓存统计:")
    print(f"  - 网格缓存命中率: {cache_stats['hit_rate']}")
    print(f"  - 缓存内存占用: {cache_stats['cache_memory_mb']:.2f} MB")


if __name__ == '__main__':
    # 隐藏来自 pandas 的未来警告
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # 运行主函数
    main()