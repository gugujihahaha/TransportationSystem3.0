"""
完整的交通方式识别项目主脚本 (已修复变长序列的 RuntimeError)
-----------------------------------------------------
该脚本集成了：
1. 阶段 1: 知识图谱 (KG) 构建与缓存
2. 阶段 2: 轨迹加载、特征提取与缓存 (使用占位符数据)
3. 阶段 3: 基于 Bi-LSTM 和 KG 的模型定义 (使用序列打包确保精度)
4. 阶段 4: 模型训练与评估
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm
import pickle
from typing import Tuple, List

# =============================================================
# I. 配置与缓存路径
# =============================================================
CACHE_DIR = 'cache'
KG_CACHE_PATH = os.path.join(CACHE_DIR, 'kg_data.pkl')
PROCESSED_DATA_CACHE_PATH = os.path.join(CACHE_DIR, 'processed_data.pkl')
os.makedirs(CACHE_DIR, exist_ok=True)
# =============================================================


# =============================================================
# II. 模块定义 (为了可执行性，所有类定义放在此处)
# =============================================================

# --- A. 知识图谱模块 (占位符实现) ---
class TransportationKnowledgeGraph:
    """知识图谱占位类：模拟知识图谱的构建和特征提取功能"""
    def __init__(self, *args, **kwargs):
        self.road_network = None
        self.poi_df = None
        self.graph_stats = {'num_nodes': 0, 'num_edges': 0}

    def build_from_osm(self, road_network, pois):
        # 模拟构建逻辑
        self.road_network = road_network
        self.poi_df = pois
        self.graph_stats = {'num_nodes': 1113602, 'num_edges': 1007661}

    def get_graph_statistics(self):
        return self.graph_stats

    def extract_kg_features(self, trajectory_df: pd.DataFrame) -> np.ndarray:
        """KG 特征提取占位：返回 N x 11 的零矩阵"""
        # 修复：接受 DataFrame，但使用其形状生成随机特征，避免 'iterrows' 错误
        N = trajectory_df.shape[0]
        # 返回 N x 11 的模拟特征
        return np.zeros((N, 11), dtype=np.float32) + np.random.rand(N, 11) * 0.1


# --- B. 特征提取模块 ---
class FeatureExtractor:
    """特征提取器"""

    def __init__(self, kg: TransportationKnowledgeGraph):
        self.kg = kg

    def extract_features(self, trajectory_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取轨迹特征和知识图谱特征

        Args:
            trajectory_df: 预处理后的轨迹 DataFrame (N, 7)

        Returns:
            trajectory_features: 轨迹特征 (N, 7)
            kg_features: 知识图谱特征 (N, 11)
        """
        trajectory_array = trajectory_df.values # 转换为 NumPy 数组

        # 提取轨迹特征 (直接归一化)
        trajectory_features = self._normalize_features(trajectory_array)

        # 提取知识图谱特征 (传入 DataFrame)
        kg_features = self.kg.extract_kg_features(trajectory_df)

        return trajectory_features, kg_features

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """归一化特征 (Z-score 归一化)"""
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8

        normalized = (features - mean) / std
        normalized = np.clip(normalized, -5, 5)

        return normalized


# --- C. 模型模块 ---
class TransportationModeClassifier(nn.Module):
    """交通方式分类器：Bi-LSTM 融合模型 (已修改 forward 方法以处理填充序列)"""

    def __init__(self,
                 trajectory_feature_dim: int = 7,
                 kg_feature_dim: int = 11,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 11,
                 dropout: float = 0.3):
        super(TransportationModeClassifier, self).__init__()

        self.trajectory_lstm = nn.LSTM(
            input_size=trajectory_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.kg_lstm = nn.LSTM(
            input_size=kg_feature_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        fusion_input_dim = hidden_dim * 3 # 2*hidden_dim + 1*hidden_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, trajectory_features: torch.Tensor,
                kg_features: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory_features: (batch_size, max_len, traj_dim)
            kg_features: (batch_size, max_len, kg_dim)
            lengths: 原始序列长度 (batch_size,)
        """

        # 1. 轨迹特征编码 (Pack -> LSTM -> Unpack)
        # 将填充序列打包，LSTM 将只对非填充数据进行计算
        packed_traj = nn.utils.rnn.pack_padded_sequence(
            trajectory_features, lengths.cpu(), batch_first=True
        )
        trajectory_out_packed, _ = self.trajectory_lstm(packed_traj)
        trajectory_out, _ = nn.utils.rnn.pad_packed_sequence(
            trajectory_out_packed, batch_first=True
        )

        # 提取最后一个有效时间步的输出作为序列表示
        trajectory_repr_list = []
        for i in range(trajectory_out.size(0)):
            # lengths 已经按降序排列，lengths[i] 就是第 i 个序列的长度
            trajectory_repr_list.append(trajectory_out[i, lengths[i].item() - 1, :])
        trajectory_repr = torch.stack(trajectory_repr_list, dim=0) # (batch_size, hidden_dim * 2)

        # 2. 知识图谱特征编码 (Pack -> LSTM -> Unpack)
        packed_kg = nn.utils.rnn.pack_padded_sequence(
            kg_features, lengths.cpu(), batch_first=True
        )
        kg_out_packed, _ = self.kg_lstm(packed_kg)
        kg_out, _ = nn.utils.rnn.pad_packed_sequence(
            kg_out_packed, batch_first=True
        )

        # 提取最后一个有效时间步的输出
        kg_repr_list = []
        for i in range(kg_out.size(0)):
            kg_repr_list.append(kg_out[i, lengths[i].item() - 1, :])
        kg_repr = torch.stack(kg_repr_list, dim=0) # (batch_size, hidden_dim)

        # 3. 特征融合
        combined = torch.cat([trajectory_repr, kg_repr], dim=1)
        fused = self.fusion_layer(combined)

        # 4. 分类
        logits = self.classifier(fused)

        return logits


# --- D. 数据集模块 ---
class TrajectoryDataset(Dataset):
    """轨迹数据集"""

    def __init__(self, segments: List[Tuple[pd.DataFrame, str]], feature_extractor: FeatureExtractor, label_encoder: LabelEncoder):
        self.segments = segments
        self.feature_extractor = feature_extractor
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        # trajectory 现在是 DataFrame
        trajectory_df, label = self.segments[idx]

        # 提取特征
        trajectory_features, kg_features = self.feature_extractor.extract_features(trajectory_df)

        # 转换为tensor
        trajectory_tensor = torch.FloatTensor(trajectory_features)
        kg_tensor = torch.FloatTensor(kg_features)

        # 编码标签
        label_encoded = self.label_encoder.transform([label])[0]
        label_tensor = torch.LongTensor([label_encoded])[0]

        return trajectory_tensor, kg_tensor, label_tensor

# --- E. 预处理占位符 ---
class GeoLifeDataLoader:
    def __init__(self, root): pass
    def get_all_users(self): return [f'{i:03}' for i in range(100)]
    def load_labels(self, user): return pd.DataFrame()
    def load_trajectory(self, path): return pd.DataFrame()
    def segment_trajectory(self, traj, labels): return []

class OSMDataLoader:
    def __init__(self, path): pass
    def load_osm_data(self): return None
    def extract_road_network(self, data): return {'dummy': 1}
    def extract_pois(self, data): return pd.DataFrame()

def preprocess_trajectory_segments(all_segments: list, min_length: int) -> List[Tuple[pd.DataFrame, str]]:
    """模拟预处理：创建随机的轨迹段数据 (返回 DataFrame)"""
    print(f"   [INFO] 模拟生成随机轨迹段...")

    mode_list = ['walk', 'bike', 'car', 'bus', 'train', 'taxi', 'subway', 'run', 'motorcycle', 'airplane', 'boat']
    N_segments = 1000 # 增加到 1000 段以提高训练稳定性

    segments_data = []
    columns=['latitude', 'longitude', 'speed', 'accel', 'bearing', 'distance', 'time_diff']

    for _ in range(N_segments):
        # 序列长度随机 10-200
        seq_len = np.random.randint(min_length, 200)
        features = np.random.randn(seq_len, 7)

        # 将 NumPy 数组转换为 DataFrame
        trajectory_df = pd.DataFrame(features, columns=columns)
        label = np.random.choice(mode_list)
        segments_data.append((trajectory_df, label))

    return segments_data

# --- F. 自定义 Collate 函数 (核心修复) ---
def custom_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    用于处理变长轨迹序列的自定义 collate_fn，进行填充和排序。
    """
    # 1. 分离数据
    traj_features = [item[0] for item in batch]
    kg_features = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch])

    # 2. 计算原始长度并找到最大长度
    lengths = torch.tensor([f.size(0) for f in traj_features], dtype=torch.long)
    max_len = lengths.max().item()

    # 3. 对轨迹特征和 KG 特征进行填充 (Padding)
    # 使用 torch.zeros 初始化填充张量
    padded_traj_features = torch.zeros((len(batch), max_len, traj_features[0].size(1)), dtype=torch.float)
    padded_kg_features = torch.zeros((len(batch), max_len, kg_features[0].size(1)), dtype=torch.float)

    for i, seq in enumerate(traj_features):
        padded_traj_features[i, :seq.size(0)] = seq

    for i, seq in enumerate(kg_features):
        padded_kg_features[i, :seq.size(0)] = seq

    # 4. 排序：PackPaddedSequence 要求输入数据必须按长度降序排列
    lengths, perm_idx = lengths.sort(0, descending=True)

    # 重新排列所有 Tensor
    padded_traj_features = padded_traj_features[perm_idx]
    padded_kg_features = padded_kg_features[perm_idx]
    labels = labels[perm_idx]

    return padded_traj_features, padded_kg_features, lengths, labels


# =============================================================
# III. 主函数与训练流程
# =============================================================

def load_data(geolife_root: str, osm_path: str, max_users: int = None):
    """加载所有数据，并实现阶段 1 和阶段 2 的缓存检查"""

    geolife_loader = GeoLifeDataLoader(geolife_root)
    users = geolife_loader.get_all_users()
    if max_users:
        users = users[:max_users]

    # ================= 阶段 1: 知识图谱构建 (缓存检查) ==================
    if os.path.exists(KG_CACHE_PATH):
        print(f"\n================ 阶段 1: 知识图谱构建 (从缓存加载) ==================")
        with open(KG_CACHE_PATH, 'rb') as f:
            kg = pickle.load(f)
        print("1.0 知识图谱从缓存加载完成。")
    else:
        print("\n================ 阶段 1: 数据加载与知识图谱构建 ==================")
        # ... (执行构建逻辑)
        osm_loader = OSMDataLoader(osm_path)
        osm_data = osm_loader.load_osm_data()
        road_network = osm_loader.extract_road_network(osm_data)
        pois = osm_loader.extract_pois(osm_data)
        kg = TransportationKnowledgeGraph()
        kg.build_from_osm(road_network, pois)
        print(f" -> 知识图谱构建阶段已完成。统计: {kg.get_graph_statistics()}")

        with open(KG_CACHE_PATH, 'wb') as f:
            pickle.dump(kg, f)
        print("1.4 知识图谱缓存完成。")


    # ================= 阶段 2: 轨迹加载、分段与特征提取 (缓存检查) ==================
    if os.path.exists(PROCESSED_DATA_CACHE_PATH):
        print("\n================ 阶段 2: 轨迹加载与特征提取 (从缓存加载) ==================")
        with open(PROCESSED_DATA_CACHE_PATH, 'rb') as f:
            processed_segments, label_encoder = pickle.load(f)
        print(f"2.0 预处理数据从缓存加载完成。剩余 {len(processed_segments)} 个可用轨迹段。")
    else:
        print("\n================ 阶段 2: 轨迹加载、分段与特征提取 (使用占位符数据) ==================")
        # 使用占位符函数生成 DataFrame 轨迹数据
        processed_segments = preprocess_trajectory_segments([], min_length=10)
        print(f" -> 预处理完成。剩余 {len(processed_segments)} 个可用轨迹段。")

        all_labels = [label for _, label in processed_segments]
        label_encoder = LabelEncoder()
        label_encoder.fit(all_labels)

        with open(PROCESSED_DATA_CACHE_PATH, 'wb') as f:
            pickle.dump((processed_segments, label_encoder), f)
        print("2.3 预处理数据缓存完成。")

    return processed_segments, kg, label_encoder


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch (已适配新的 DataLoader 输出)"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # 接收四个参数: features, kg_features, lengths, labels
    for trajectory_features, kg_features, lengths, labels in tqdm(dataloader, desc="   [训练批次]"):

        trajectory_features = trajectory_features.to(device)
        kg_features = kg_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 调用模型时，传入 lengths
        logits = model(trajectory_features, kg_features, lengths)
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
    """评估模型 (已适配新的 DataLoader 输出)"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # 接收四个参数: features, kg_features, lengths, labels
        for trajectory_features, kg_features, lengths, labels in tqdm(dataloader, desc="   [评估批次]"):

            trajectory_features = trajectory_features.to(device)
            kg_features = kg_features.to(device)
            labels = labels.to(device)

            # 调用模型时，传入 lengths
            logits = model(trajectory_features, kg_features, lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    class_names = label_encoder.classes_
    report = classification_report(all_labels, all_preds,
                                  target_names=class_names,
                                  output_dict=True,
                                  zero_division=0)

    return avg_loss, report, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='训练交通方式识别模型')
    parser.add_argument('--geolife_root', type=str,
                       default='data/Geolife Trajectories 1.3',
                       help='GeoLife数据根目录')
    parser.add_argument('--osm_path', type=str,
                       default='data/export.geojson',
                       help='OSM数据路径')
    parser.add_argument('--max_users', type=int, default=100,
                       help='最大用户数（用于快速测试）')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=10,
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
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 加载数据
    segments, kg, label_encoder = load_data(args.geolife_root, args.osm_path, args.max_users)

    if len(segments) == 0:
        print("错误: 没有加载到任何数据，请检查数据路径和预处理条件。")
        return

    # 创建特征提取器
    print("\n================ 阶段 3: 模型初始化与训练 ==================")
    feature_extractor = FeatureExtractor(kg)

    num_classes = len(label_encoder.classes_)
    print(f" -> 类别数: {num_classes}")
    print(f" -> 类别: {label_encoder.classes_}")

    # 创建数据集
    dataset = TrajectoryDataset(segments, feature_extractor, label_encoder)

    # 划分训练集和测试集
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=label_encoder.transform([segments[i][1] for i in range(len(segments))])
    )

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # --- 关键修改：使用 custom_collate_fn 修复 RuntimeError ---
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    # --------------------------------------------------------

    print(f" -> 训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    # 创建模型
    model = TransportationModeClassifier(
        trajectory_feature_dim=7,
        kg_feature_dim=11,
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
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print("   [INFO] 测试损失降低，保存最佳模型。")

        # 打印分类报告
        if (epoch + 1) % 5 == 0:
            print("\n   [报告] 当前分类报告:")
            print(classification_report(test_labels, test_preds,
                                      target_names=label_encoder.classes_, zero_division=0))

    print("\n================ 训练完成! ==================")
    print(f"最终最佳测试损失: {best_test_loss:.4f}")
    print(f"最佳模型已保存到 {os.path.join(args.save_dir, 'best_model.pth')}")


if __name__ == '__main__':
    main()