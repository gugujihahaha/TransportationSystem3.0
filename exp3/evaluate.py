"""
评估脚本 (Exp3) - 增强知识图谱版
功能：
1. 支持从缓存直接加载特征（快速模式）
2. 自动处理标签映射（修复 Exp2 中的标签不匹配问题）
3. 生成详细的分类报告、混淆矩阵和预测详情
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

# 导入与训练一致的模块
from src.model import TransportationModeClassifier


# ------------------------------------------------------------
# 1. 数据集定义 (适配 Exp3 15维特征)
# ------------------------------------------------------------
class DualFeatureDataset(Dataset):
    def __init__(self, segments, label_encoder):
        self.segments = segments
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        # 兼容性处理：根据训练脚本，特征可能是 (traj, kg, label)
        traj_features, kg_features, label = self.segments[idx]

        traj_tensor = torch.FloatTensor(traj_features)
        kg_tensor = torch.FloatTensor(kg_features)

        # 标签编码
        if isinstance(label, (int, np.integer)):
            label_tensor = torch.LongTensor([label])[0]
        else:
            label_encoded = self.label_encoder.transform([label])[0]
            label_tensor = torch.LongTensor([label_encoded])[0]

        return traj_tensor, kg_tensor, label_tensor


# ------------------------------------------------------------
# 2. 评估核心逻辑
# ------------------------------------------------------------
def run_evaluation(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for traj_f, kg_f, labels in tqdm(dataloader, desc="正在进行模型推理"):
            traj_f, kg_f, labels = traj_f.to(device), kg_f.to(device), labels.to(device)

            logits = model(traj_f, kg_f)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # 记录预测对应类别的置信度
            batch_probs = probs.cpu().numpy()
            all_probs.extend([batch_probs[i, p] for i, p in enumerate(preds.cpu().numpy())])

    return all_preds, all_labels, all_probs


# ------------------------------------------------------------
# 3. 可视化函数
# ------------------------------------------------------------
def save_visualizations(y_true, y_pred, class_names, output_dir):
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Exp3 (Enhanced KG)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


# ------------------------------------------------------------
# 4. 主函数
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Exp3 评估脚本')
    parser.add_argument('--model_path', type=str, default='checkpoints/exp3_model.pth')
    parser.add_argument('--feature_cache', type=str, default='cache/processed_features_v1.pkl')
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载模型存档
    print(f"正在加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
    label_encoder = checkpoint['label_encoder']
    config = checkpoint['model_config']

    model = TransportationModeClassifier(
        trajectory_feature_dim=config['trajectory_feature_dim'],
        kg_feature_dim=config['kg_feature_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    ).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 2. 加载评估数据 (优先从 Exp3 训练产生的缓存加载)
    if os.path.exists(args.feature_cache):
        print(f"✓ 发现特征缓存，正在快速加载: {args.feature_cache}")
        with open(args.feature_cache, 'rb') as f:
            # 训练脚本保存格式为 (all_features, label_encoder)
            all_data, _ = pickle.load(f)
    else:
        print(f"❌ 未找到缓存文件 {args.feature_cache}。请先运行 train.py 生成缓存。")
        return

    dataset = DualFeatureDataset(all_data, label_encoder)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # 3. 执行推理
    preds, labels, confidences = run_evaluation(model, dataloader, args.device)

    # 4. 生成报告
    class_names = label_encoder.classes_
    report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
    print("\n评估完成！分类报告如下:")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))

    # 5. 保存结果文件
    with open(os.path.join(args.output_dir, 'evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=4)

    save_visualizations(labels, preds, class_names, args.output_dir)

    df_results = pd.DataFrame({
        'true_label': [class_names[i] for i in labels],
        'pred_label': [class_names[i] for i in preds],
        'confidence': confidences
    })
    df_results.to_csv(os.path.join(args.output_dir, 'predictions_exp3.csv'), index=False)
    print(f"✅ 结果已保存至: {args.output_dir}")


if __name__ == '__main__':
    main()