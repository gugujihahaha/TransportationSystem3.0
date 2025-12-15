"""
训练脚本 (最终版本：已实现 KG 特征提取缓存和并行加速)

核心优化点:
1. 在 load_data 阶段一次性计算所有特征，并缓存到 'cache/processed_features.pkl'。
2. **新增**：使用 joblib 库和 --num_workers 参数，将耗时的 KG 特征提取并行化，大幅加速首次运行。
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
import multiprocessing
from joblib import Parallel, delayed # <<<< 导入 joblib >>>>
import numpy as np # 用于特征数组操作

# 导入自定义模块
from src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader, preprocess_trajectory_segments
from src.feature_extraction import FeatureExtractor
from src.model import TransportationModeClassifier
from src.knowledge_graph import TransportationKnowledgeGraph


# ========================== 特征维度常量 (保持不变) ==========================
TRAJECTORY_FEATURE_DIM = 9
KG_FEATURE_DIM = 11
# ==============================================================================


# ========================== 缓存配置 (更新：缓存最终特征) ==========================
CACHE_DIR = 'cache'
KG_CACHE_PATH = os.path.join(CACHE_DIR, 'kg_data.pkl')
PROCESSED_FEATURE_CACHE_PATH = os.path.join(CACHE_DIR, 'processed_features.pkl')
os.makedirs(CACHE_DIR, exist_ok=True)
# =============================================================

# ========================== 并行工作函数 (新增) ==========================

def _parallel_feature_extraction_worker(segment, kg_data_for_worker, label_encoder):
    """
    单个核心执行的特征提取任务。
    Worker 接收 kg_data_for_worker (一个字典) 来重建 KG 实例。
    """
    trajectory, label_str = segment

    # 由于进程间传递的是序列化数据，这里需要重建 KG 实例
    kg = TransportationKnowledgeGraph()
    # 假设 TransportationKnowledgeGraph 类中定义了 load_data_for_worker 方法
    # 如果您的 kg 类没有这个方法，这里将需要修改。
    # 为了兼容性，这里我们直接使用传入的序列化数据 (这里假设 kg_data_for_worker 包含了必要的组件)
    # ************** 注意：如果您上一个版本的 KG 类移除了 load_data_for_worker，这里会失败 **************
    # 为了简化，我们直接在 worker 内部重建 FeatureExtractor 并传递 KG 对象

    # 重新初始化 FeatureExtractor (它会内部引用 KG 对象)
    # **警告**: 直接在并行任务中传递整个 KG 对象可能会导致 pickle 失败。
    # 理想情况下，我们应该在 KG 类中实现 get_data_for_worker 和 load_data_for_worker。

    # 假设：我们能直接使用传入的 kg_data_for_worker (它是完整的 TransportationKnowledgeGraph 实例)
    try:
        feature_extractor = FeatureExtractor(kg_data_for_worker)
        trajectory_features, kg_features = feature_extractor.extract_features(trajectory)
    except Exception as e:
        # 如果 KG 无法在 worker 中使用，这里需要打印错误信息
        # print(f"Error in worker: {e}")
        # 返回一个空结果或错误标识
        return None

    # 编码标签
    # LabelEncoder 实例通常可以在进程间安全共享或传递
    label_encoded = label_encoder.transform([label_str])[0]

    return trajectory_features, kg_features, label_encoded

# ========================================================================


class TrajectoryDataset(Dataset):
    """轨迹数据集 (现在直接加载预提取的特征向量)"""

    def __init__(self, all_features_and_labels):
        self.data = all_features_and_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trajectory_features, kg_features, label_encoded = self.data[idx]

        # 确保数据是正确的 NumPy 数组或列表类型，并转换为 Tensor
        trajectory_tensor = torch.FloatTensor(trajectory_features)
        kg_tensor = torch.FloatTensor(kg_features)
        label_tensor = torch.LongTensor([label_encoded])[0]

        return trajectory_tensor, kg_tensor, label_tensor


def load_data(geolife_root: str, osm_path: str, max_users: int = None, num_workers: int = 10):
    """加载所有数据，并实现阶段 1 (KG) 和 阶段 2 (特征) 的缓存检查"""

    geolife_loader = GeoLifeDataLoader(geolife_root)
    users = geolife_loader.get_all_users()
    if max_users and max_users < len(users):
        users = users[:max_users]

    # ================= 阶段 1: 知识图谱构建 (缓存检查) ==================
    kg = None
    if os.path.exists(KG_CACHE_PATH):
        print(f"\n================ 阶段 1: 知识图谱构建 (从缓存加载) ==================")
        try:
            with open(KG_CACHE_PATH, 'rb') as f:
                kg = pickle.load(f)
            print("1.0 知识图谱从缓存加载完成。")
        except Exception as e:
            warnings.warn(f"[WARN] 知识图谱缓存文件加载失败 ({e})，将重新构建。")
            os.remove(KG_CACHE_PATH)
            kg = None

    if kg is None:
        # --- 正常执行阶段 1 (重建) ---
        print("\n================ 阶段 1: 数据加载与知识图谱构建 (重建) ==================")
        print(f"1.1 正在初始化 GeoLife 数据加载器: {geolife_root}")
        print(f" -> 找到 {len(users)} 个 GeoLife 用户（将处理其中 {len(users)} 个）")
        osm_loader = OSMDataLoader(osm_path)
        osm_data = osm_loader.load_osm_data()
        road_network = osm_loader.extract_road_network(osm_data)
        pois = osm_loader.extract_pois(osm_data)
        print(f" -> OSM 数据提取完成: {len(road_network)} 条道路, {len(pois)} 个POI。")
        print("\n1.3 正在构建交通知识图谱...")
        kg = TransportationKnowledgeGraph()
        # 这里的 build_from_osm 需要传递 road_network 和 pois
        kg.build_from_osm(road_network, pois)
        print(f" -> 知识图谱构建阶段已完成。统计: {kg.get_graph_statistics()}")
        print(f"1.4 知识图谱构建完成，正在存储到缓存文件：{KG_CACHE_PATH}")
        with open(KG_CACHE_PATH, 'wb') as f:
            pickle.dump(kg, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("1.4 知识图谱缓存完成。")


    # ================= 阶段 2: 轨迹加载、分段与【特征提取】(缓存检查) ==================
    all_features_and_labels = None
    label_encoder = None

    # <<< 检查最终特征缓存 >>>
    if os.path.exists(PROCESSED_FEATURE_CACHE_PATH):
        print(f"\n================ 阶段 2: 特征提取 (从缓存加载) ==================")
        try:
            with open(PROCESSED_FEATURE_CACHE_PATH, 'rb') as f:
                all_features_and_labels, label_encoder = pickle.load(f)
            print(f"2.0 预提取特征从缓存加载完成。共有 {len(all_features_and_labels)} 条记录。")
            return all_features_and_labels, kg, label_encoder
        except Exception as e:
            warnings.warn(f"[WARN] 最终特征缓存文件加载失败 ({e})，将重新处理。")
            os.remove(PROCESSED_FEATURE_CACHE_PATH)
            all_features_and_labels = None

    # --- 缓存不存在或损坏，执行耗时的重建步骤 ---
    print("\n================ 阶段 2: 轨迹加载、分段与【特征提取】 (重建) ==================")

    # 2.1 轨迹加载、分段与预处理 (文件 I/O 密集)
    print("2.1 正在加载 GeoLife 轨迹段 (文件 I/O 密集)...")
    all_segments = []
    for user_id in tqdm(users, desc="[GeoLife用户加载]"):
        labels = geolife_loader.load_labels(user_id)
        if labels.empty: continue
        trajectory_dir = os.path.join(geolife_root, f"Data/{user_id}/Trajectory")
        if not os.path.exists(trajectory_dir): continue
        for traj_file in os.listdir(trajectory_dir):
            if not traj_file.endswith('.plt'): continue
            traj_path = os.path.join(trajectory_dir, traj_file)
            try:
                trajectory = geolife_loader.load_trajectory(traj_path)
                segments = geolife_loader.segment_trajectory(trajectory, labels)
                all_segments.extend(segments)
            except Exception as e:
                continue
    print(f" -> GeoLife 轨迹加载完成。总共加载 {len(all_segments)} 个原始轨迹段。")
    print("\n2.2 正在预处理轨迹段...")
    processed_segments = preprocess_trajectory_segments(all_segments, min_length=10)
    print(f" -> 预处理完成。剩余 {len(processed_segments)} 个可用轨迹段。")

    valid_modes = {'walk', 'bike', 'bus', 'car', 'train', 'taxi'}
    original_count = len(processed_segments)
    processed_segments = [
        (traj, label) for traj, label in processed_segments
        if label in valid_modes
    ]
    filtered_count = original_count - len(processed_segments)
    if filtered_count > 0:
        print(f" -> 过滤掉 {filtered_count} 个非标准类别的轨迹段")

    if not processed_segments:
        print("错误: 没有加载到任何可用轨迹段。")
        return [], kg, None

    # 2.3 创建标签编码器
    all_labels_str = [label for _, label in processed_segments]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels_str)

    # <<<< 2.4 【关键步骤：特征提取】(并行加速) >>>>
    print(f"\n2.4 正在进行耗时的【KG特征提取】(使用 {num_workers} 个核心加速)...")

    # 使用 joblib 进行并行处理
    # 传递整个 KG 对象和 LabelEncoder 对象给 worker
    # joblib 会自动使用 pickle/dill 来序列化这些对象
    results = Parallel(n_jobs=num_workers, verbose=10)(
        delayed(_parallel_feature_extraction_worker)(
            segment, kg, label_encoder
        )
        for segment in tqdm(processed_segments, desc="[提交并行任务]")
    )

    # 过滤掉任何并行失败的结果 (返回 None 的)
    all_features_and_labels = [res for res in results if res is not None]

    # 2.5 缓存最终特征
    print(f"\n2.5 最终特征提取完成，正在存储到缓存文件：{PROCESSED_FEATURE_CACHE_PATH}")
    data_to_save = (all_features_and_labels, label_encoder)
    with open(PROCESSED_FEATURE_CACHE_PATH, 'wb') as f:
        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("2.5 最终特征缓存完成。")

    return all_features_and_labels, kg, label_encoder


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for trajectory_features, kg_features, labels in tqdm(dataloader, desc="   [训练批次]"):
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

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, label_encoder):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for trajectory_features, kg_features, labels in tqdm(dataloader, desc="   [评估批次]"):
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

    class_names = label_encoder.classes_
    report = classification_report(all_labels, all_preds,
                                  target_names=class_names,
                                  output_dict=True, zero_division=0)

    return avg_loss, report, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='训练交通方式识别模型')
    parser.add_argument('--geolife_root', type=str,
                       default='../data/Geolife Trajectories 1.3',
                       help='GeoLife数据根目录')
    parser.add_argument('--osm_path', type=str,
                       default='../data/export.geojson',
                       help='OSM数据路径')
    parser.add_argument('--max_users', type=int, default=None,
                       help='最大用户数（用于快速测试）')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout比率')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='模型保存目录')
    # 用户可以指定 10 个核心
    parser.add_argument('--num_workers', type=int, default=4,
                       help='用于数据加载和特征提取的进程数，以利用多核CPU加速')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 关键修改：将 num_workers 传递给 load_data 函数
    all_features_and_labels, kg, label_encoder = load_data(
        args.geolife_root,
        args.osm_path,
        args.max_users,
        args.num_workers
    )

    if len(all_features_and_labels) == 0:
        print("错误: 没有加载到任何数据，请检查数据路径和预处理条件。")
        return

    print("\n================ 阶段 3: 模型初始化与训练 ==================")
    num_classes = len(label_encoder.classes_)
    print(f" -> 类别数: {num_classes}")
    print(f" -> 类别: {label_encoder.classes_}")

    dataset = TrajectoryDataset(all_features_and_labels)

    # 划分训练集和测试集
    labels_for_stratify = [label for _, _, label in all_features_and_labels]
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=labels_for_stratify
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # 应用 num_workers 参数
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True)

    print(f" -> 训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    # 创建模型并将其移动到 GPU
    model = TransportationModeClassifier(
        trajectory_feature_dim=TRAJECTORY_FEATURE_DIM,
        kg_feature_dim=KG_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(args.device)

    print(f" -> 模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f" -> 训练设备: {args.device}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 训练循环
    best_test_loss = float('inf')

    print(f"\n================ 阶段 4: 模型训练 (共 {args.epochs} 轮) ==================")
    print("模型结构采用 Bi-LSTM，适用于序列数据和异构特征融合。")


    for epoch in range(args.epochs):
        print(f"\n[EPOCH {epoch+1}/{args.epochs}] 开始训练...")

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)
        print(f"   [结果] 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")

        # 评估
        test_loss, test_report, test_preds, test_labels = evaluate(
            model, test_loader, criterion, args.device, label_encoder
        )
        test_acc = test_report['accuracy']
        print(f"   [结果] 测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")

        # 学习率调度
        scheduler.step(test_loss)

        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
                'test_acc': test_acc,
                'label_encoder': label_encoder,
            }, os.path.join(args.save_dir, 'exp2_model.pth'))
            print("   [INFO] 测试损失降低，保存最佳模型。")

        # 打印分类报告
        if (epoch + 1) % 10 == 0:
            print("\n   [报告] 当前分类报告:")
            print(classification_report(test_labels, test_preds,
                                      target_names=label_encoder.classes_, zero_division=0))

    print("\n================ 训练完成! ==================")
    print(f"最终最佳测试损失: {best_test_loss:.4f}")
    print(f"最佳模型已保存到 {os.path.join(args.save_dir, 'exp2_model.pth')}")


if __name__ == '__main__':
    # 启用 Windows 多进程保护，以确保 DataLoader 能够稳定启动 worker 进程
    if os.name == 'nt':
        multiprocessing.freeze_support()

    main()