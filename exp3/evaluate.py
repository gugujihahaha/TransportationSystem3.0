# ============================================================
# exp3/evaluate.py - 评估脚本 (双输入：轨迹特征 + 增强知识图谱特征)
# ============================================================
"""
评估脚本 - 结合轨迹特征和增强知识图谱特征 (Exp3)

对训练好的双输入模型进行离线整体评估：
- Accuracy / Precision / Recall / F1
- Confusion Matrix
"""
import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 导入 Exp3 模块
from src.model import TransportationModeClassifier
from src.knowledge_graph import EnhancedTransportationKG  # 🔥 Exp3 核心修改
from src.feature_extraction import FeatureExtractor
from src.data_preprocessing import GeoLifeDataLoader, preprocess_trajectory_segments


# ------------------------------------------------------------
# 数据集类 (与 Exp2 相同，但输入维度变为 9 维和 15 维)
# ------------------------------------------------------------
class DualFeatureDataset(Dataset):
    """用于 Exp3 的双输入数据集 (轨迹特征 (N, 9) + 增强 KG特征 (N, 15))"""

    def __init__(self, segments, label_encoder):
        # segments: List[Tuple[traj_features_np, kg_features_np, label_str]]
        self.segments = segments
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        traj_features, kg_features, label = self.segments[idx]

        # 转换为tensor
        traj_tensor = torch.FloatTensor(traj_features)  # (seq_len, 9)
        kg_tensor = torch.FloatTensor(kg_features)  # (seq_len, 15) 🔥 维度变化

        # 编码标签
        label_encoded = self.label_encoder.transform([label])[0]
        label_tensor = torch.LongTensor([label_encoded])

        return traj_tensor, kg_tensor, label_tensor

    # ------------------------------------------------------------


# 数据加载和特征提取（适配 Exp3）
# ------------------------------------------------------------
def load_and_extract_data(geolife_root: str, kg_data_path: str):
    """加载 GeoLife 数据，初始化增强 KG，并提取双特征"""

    # 1. 初始化数据加载器
    geolife_loader = GeoLifeDataLoader(geolife_root)

    # 2. 加载所有轨迹段 (返回原始 DataFrame 和标签) - 逻辑与 Exp2 相同
    print("=" * 60)
    print("加载 GeoLife 原始轨迹数据...")
    all_raw_segments = []

    users = geolife_loader.get_all_users()
    for user_id in tqdm(users, desc="加载用户数据"):
        labels = geolife_loader.load_labels(user_id)
        if labels.empty:
            continue

        trajectory_dir = os.path.join(geolife_root, f"Data/{user_id}/Trajectory")
        if not os.path.exists(trajectory_dir):
            continue

        for traj_file in os.listdir(trajectory_dir):
            if traj_file.endswith('.plt'):
                traj_path = os.path.join(trajectory_dir, traj_file)
                trajectory = geolife_loader.load_trajectory(traj_path)
                segments = geolife_loader.segment_trajectory(trajectory, labels)
                all_raw_segments.extend(segments)

    print(f"\n总共加载 {len(all_raw_segments)} 个原始轨迹段")

    # 3. 预处理 (提取 9 维轨迹特征和长度规范化) - 逻辑与 Exp2 相同
    print("\n预处理轨迹段 (提取 9 维轨迹特征, 长度规范化)...")
    processed_segments = preprocess_trajectory_segments(all_raw_segments, min_length=10)

    # 4. 初始化知识图谱和特征提取器
    print("\n初始化增强知识图谱 (Exp3)...")
    # 🔥 核心修改：使用 EnhancedTransportationKG
    kg = EnhancedTransportationKG()
    # 假设 kg_data_path 包含 road_network.pkl/pois.pkl/transit_routes.pkl 等
    kg.load_data(data_root=kg_data_path)
    feature_extractor = FeatureExtractor(kg)

    # 5. 提取 KG 特征并组合
    final_segments = []
    print("\n提取知识图谱特征并组合 (特征维度: 9 + 15)...")
    for traj_features_raw, label in tqdm(processed_segments, desc="[特征组合]"):

        # feature_extractor.extract_features 返回 (归一化轨迹特征 (N, 9), 增强 KG 特征 (N, 15))
        normalized_traj_features, kg_features = feature_extractor.extract_features(traj_features_raw)

        # 验证维度 (可选，但推荐)
        if kg_features.shape[1] != 15:
            raise ValueError(f"KG 特征维度错误：预期 15 维，实际 {kg_features.shape[1]} 维。")

        # 最终结构: (归一化后的轨迹特征, KG 特征, 标签)
        final_segments.append((normalized_traj_features, kg_features, label))

    print(f"最终用于评估的有效轨迹段: {len(final_segments)} 个")
    return final_segments


# ------------------------------------------------------------
# 评估函数 (与 Exp2 相同)
# ------------------------------------------------------------
def evaluate_model(model, dataloader, device):
    """在整个数据集上运行模型并收集结果 (双输入)"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs_raw = []

    with torch.no_grad():
        for traj_features, kg_features, labels in tqdm(dataloader, desc="评估模型中"):
            traj_features = traj_features.to(device)
            kg_features = kg_features.to(device)
            labels = labels.squeeze().to(device)

            # 双输入模型调用
            logits = model(traj_features, kg_features)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs_raw.extend(probs.cpu().numpy())

    return all_preds, all_labels, all_probs_raw


# ------------------------------------------------------------
# 可视化函数 (与 Exp2 相同)
# ------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix (Exp3: Trajectory + Enhanced KG)', fontsize=16)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_per_class_metrics(report, class_names, save_path):
    metrics_df = pd.DataFrame({
        'Precision': [report[c]['precision'] for c in class_names],
        'Recall': [report[c]['recall'] for c in class_names],
        'F1-Score': [report[c]['f1-score'] for c in class_names],
    }, index=class_names)

    metrics_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Per-Class Performance Metrics (Exp3)', fontsize=16)
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='评估 Exp3 交通方式识别模型 (轨迹 + 增强KG)')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints/exp3_model.pth',
                        help='已训练模型的路径')
    parser.add_argument('--geolife_root', type=str,
                        default='../data/Geolife Trajectories 1.3',
                        help='GeoLife数据根目录')
    parser.add_argument('--kg_data_path', type=str,
                        default='../data/kg_data',
                        help='知识图谱数据目录')
    parser.add_argument('--output_dir', type=str,
                        default='evaluation_results_exp3',
                        help='评估结果保存目录')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载模型和 LabelEncoder
    print("=" * 60)
    print("加载模型和LabelEncoder...")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    label_encoder = checkpoint['label_encoder']
    config = checkpoint['model_config']

    # 验证模型配置
    if config.get('kg_feature_dim') != 15:
        print(f"警告: 模型配置中的 KG 维度为 {config.get('kg_feature_dim')}，但 Exp3 预期为 15 维。")

    # 实例化模型 (使用 Exp3 的参数)
    model = TransportationModeClassifier(
        trajectory_feature_dim=config.get('trajectory_feature_dim', 9),
        # 🔥 核心修改：确保 KG 维度为 15
        kg_feature_dim=config.get('kg_feature_dim', 15),
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.3)
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 模型加载成功 (Epoch: {checkpoint['epoch']})")
    print("=" * 60)

    # 2. 加载、预处理和特征提取
    all_segments = load_and_extract_data(args.geolife_root, args.kg_data_path)

    # 3. 过滤和创建 DataLoader
    # 目标类别列表（已合并 'car' & 'taxi'）
    TARGET_MODES_FINAL = ['walk', 'bike', 'car & taxi', 'bus', 'train', 'subway']

    # 过滤掉非目标类别
    all_segments = [seg for seg in all_segments if seg[2] in TARGET_MODES_FINAL]

    # 确保只保留 LabelEncoder 中存在的类别
    valid_segments = [seg for seg in all_segments if seg[2] in label_encoder.classes_]

    dataset = DualFeatureDataset(valid_segments, label_encoder)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    class_names = label_encoder.classes_
    num_classes = len(class_names)

    print(f"最终评估样本数: {len(dataset)}")
    print(f"评估类别: {list(class_names)}")
    print("=" * 60)

    # 4. 运行评估
    all_preds, all_labels, all_probs_raw = evaluate_model(model, dataloader, args.device)

    # 提取预测的置信度
    all_probs = [probs_array[pred] for probs_array, pred in zip(all_probs_raw, all_preds)]

    # 5. 生成报告
    target_labels = np.arange(num_classes)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        labels=target_labels,
        output_dict=True,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("详细分类报告")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names, labels=target_labels, zero_division=0))

    # 6. 保存结果
    report_path = os.path.join(args.output_dir, 'evaluation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✓ 评估报告已保存: {report_path}")

    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names,
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )

    plot_per_class_metrics(
        report,
        class_names,
        os.path.join(args.output_dir, 'per_class_metrics.png')
    )

    results_df = pd.DataFrame({
        'true_label': [class_names[i] for i in all_labels],
        'pred_label': [class_names[i] for i in all_preds],
        'confidence': [probs for probs in all_probs]
    })

    csv_path = os.path.join(args.output_dir, 'predictions.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ 预测结果已保存: {csv_path}")


if __name__ == '__main__':
    main()