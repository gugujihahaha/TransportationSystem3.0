# ============================================================
# exp1/evaluate.py - 评估脚本
# ============================================================
"""
评估脚本 - 仅使用GeoLife轨迹数据 (Exp1)

对训练好的模型进行离线整体评估：
- Accuracy / Precision / Recall / F1
- Confusion Matrix
- 每类别性能对比图
- 预测结果 CSV
"""

import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import TransportationModeClassifier
# 从 train.py 导入 TrajectoryDataset 和 load_data
from train import TrajectoryDataset, load_data

# 🔥 核心修改：更新最终类别列表
# 注意：'taxi' 已在 data_loader.py 中合并为 'car & taxi'
TARGET_MODES_FINAL = ['walk', 'bike', 'car & taxi', 'bus', 'train', 'subway']


# ------------------------------------------------------------
# 可视化函数
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
    plt.title('Confusion Matrix (Exp1)', fontsize=16)
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
    plt.title('Per-Class Performance Metrics', fontsize=16)
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ------------------------------------------------------------
# 评估主函数
# ------------------------------------------------------------
def evaluate_model(model, dataloader, device, label_encoder):
    """在整个数据集上运行模型并收集结果"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="评估模型中"):
            features = features.to(device)
            labels = labels.squeeze().to(device)

            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_preds, all_labels, all_probs


def main():
    parser = argparse.ArgumentParser(description='评估交通方式识别模型')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints/exp1_model.pth',
                        help='已训练模型的路径')
    parser.add_argument('--geolife_root', type=str,
                        default='../data/Geolife Trajectories 1.3',
                        help='GeoLife数据根目录')
    parser.add_argument('--output_dir', type=str,
                        default='evaluation_results',
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

    # 实例化模型
    model = TransportationModeClassifier(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.3)
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 模型加载成功 (Epoch: {checkpoint['epoch']})")
    print("=" * 60)

    # 2. 加载和预处理数据
    # segments 标签已包含 'car & taxi'
    print("\n加载和预处理评估数据...")
    all_segments = load_data(args.geolife_root, max_users=None)  # 加载全部数据

    # 3. 过滤掉非目标类别 (使用更新后的最终类别列表)
    original_count = len(all_segments)
    all_segments = [seg for seg in all_segments if seg[1] in TARGET_MODES_FINAL]
    removed_count = original_count - len(all_segments)

    print(f"原始轨迹段总数: {original_count}")
    print(f"保留轨迹段总数: {len(all_segments)} (移除 {removed_count} 个稀疏类别)")

    # 4. 创建 DataLoader
    # 确保只保留 LabelEncoder 中存在的类别
    valid_segments = [seg for seg in all_segments if seg[1] in label_encoder.classes_]

    dataset = TrajectoryDataset(valid_segments, label_encoder)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 确保类别名称列表与 LabelEncoder 保持一致
    class_names = label_encoder.classes_
    num_classes = len(class_names)

    print(f"最终评估样本数: {len(dataset)}")
    print(f"评估类别: {list(class_names)}")
    print("=" * 60)

    # 5. 运行评估
    all_preds, all_labels, all_probs_raw = evaluate_model(model, dataloader, args.device, label_encoder)

    # 将 all_probs_raw (numpy array) 转换为 list of lists/array for easier processing
    all_probs = [probs_array[pred] for probs_array, pred in zip(all_probs_raw, all_preds)]

    # 6. 生成报告
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

    # 7. 保存结果

    # 保存 JSON
    report_path = os.path.join(args.output_dir, 'evaluation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✓ 评估报告已保存: {report_path}")

    # 可视化
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

    # 保存预测明细
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