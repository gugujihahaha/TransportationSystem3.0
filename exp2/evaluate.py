"""
评估脚本 (Exp2) - 基础知识图谱版 (9+11维)
功能：
1. 自动适配 11 维 KG 特征
2. 修复标签映射逻辑，合并 Car & Taxi
3. 支持从缓存快速加载，避免冗长的路网匹配过程
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

# 假设模型类定义在 src.model 中
from src.model import TransportationModeClassifier


# ------------------------------------------------------------
# 1. 数据集定义
# ------------------------------------------------------------
class DualFeatureDataset(Dataset):
    def __init__(self, segments, label_encoder):
        self.segments = segments
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        # 适配训练脚本保存的格式: (traj_feat, kg_feat, label)
        traj_features, kg_features, label = self.segments[idx]

        traj_tensor = torch.FloatTensor(traj_features)
        kg_tensor = torch.FloatTensor(kg_features)

        # 标签处理：如果是字符串则转换，如果是数字则直接读取
        if isinstance(label, (int, np.integer)):
            label_tensor = torch.LongTensor([label])[0]
        else:
            label_encoded = self.label_encoder.transform([label])[0]
            label_tensor = torch.LongTensor([label_encoded])[0]

        return traj_tensor, kg_tensor, label_tensor


# ------------------------------------------------------------
# 2. 推理逻辑
# ------------------------------------------------------------
def run_inference(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for traj, kg, labels in tqdm(dataloader, desc="推理中"):
            traj, kg, labels = traj.to(device), kg.to(device), labels.to(device)

            logits = model(traj, kg)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 记录预测概率
            p_np = probs.cpu().numpy()
            all_confidences.extend([p_np[i, p] for i, p in enumerate(preds.cpu().numpy())])

    return all_preds, all_labels, all_confidences


# ------------------------------------------------------------
# 3. 主函数
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Exp2 评估脚本')
    parser.add_argument('--model_path', type=str, default='checkpoints/exp2_model.pth')
    parser.add_argument('--cache_path', type=str, default='cache/processed_features.pkl')
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载模型及配置
    print(f"正在加载模型: {args.model_path}")
    # 使用 weights_only=False 因为包含 LabelEncoder 对象
    checkpoint = torch.load(args.model_path, map_location=args.device)
    label_encoder = checkpoint['label_encoder']
    config = checkpoint['model_config']

    # 动态初始化模型 (Exp2: 9 + 11 维)
    model = TransportationModeClassifier(
        trajectory_feature_dim=config['trajectory_feature_dim'],
        kg_feature_dim=config['kg_feature_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.3)
    ).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 2. 加载数据
    if not os.path.exists(args.cache_path):
        print(f"❌ 错误: 未找到缓存文件 {args.cache_path}。请确保 train.py 已运行并生成了特征缓存。")
        return

    print(f"✓ 正在从缓存加载特征: {args.cache_path}")
    with open(args.cache_path, 'rb') as f:
        # 训练脚本保存的是 (processed_segments, label_encoder)
        cached_data, _ = pickle.load(f)

    dataset = DualFeatureDataset(cached_data, label_encoder)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # 3. 执行评估
    preds, labels, confs = run_inference(model, dataloader, args.device)

    # 4. 生成报告
    class_names = label_encoder.classes_
    print("\n" + "=" * 40)
    print("      Exp2 分类报告 (9轨迹 + 11知识图谱)")
    print("=" * 40)
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))

    # 保存 JSON 报告
    report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
    with open(os.path.join(args.output_dir, 'evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=4)

    # 5. 可视化混淆矩阵
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Exp2 (Base KG)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))

    # 6. 保存详细预测 CSV
    df = pd.DataFrame({
        'true_label': [class_names[i] for i in labels],
        'pred_label': [class_names[i] for i in preds],
        'confidence': confs
    })
    df.to_csv(os.path.join(args.output_dir, 'predictions_exp2.csv'), index=False)
    print(f"\n✅ 评估完成！结果已保存至: {args.output_dir}")


if __name__ == '__main__':
    main()