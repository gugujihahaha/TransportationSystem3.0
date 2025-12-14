"""
评估脚本 - 仅使用GeoLife轨迹数据
"""
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import GeoLifeDataLoader, preprocess_segments
from src.model import TransportationModeClassifier
from train import TrajectoryDataset, load_data


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵 - 仅轨迹特征模型', fontsize=14, fontweight='bold')
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"混淆矩阵已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='评估交通方式识别模型（仅轨迹特征）')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--geolife_root', type=str, 
                       default='../data/Geolife Trajectories 1.3',
                       help='GeoLife数据根目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print("=" * 60)
    print("加载模型...")
    print("=" * 60)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    label_encoder = checkpoint['label_encoder']
    model_config = checkpoint.get('model_config', {})
    
    print(f"模型配置: {model_config}")
    
    # 加载数据
    print("\n加载数据...")
    segments = load_data(args.geolife_root)
    
    # 预处理
    processed_segments = preprocess_segments(segments, min_length=10)
    
    # 创建数据集
    dataset = TrajectoryDataset(processed_segments, label_encoder)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    num_classes = len(label_encoder.classes_)
    model = TransportationModeClassifier(
        input_dim=model_config.get('input_dim', 9),
        hidden_dim=model_config.get('hidden_dim', 128),
        num_layers=model_config.get('num_layers', 2),
        num_classes=num_classes,
        dropout=model_config.get('dropout', 0.3)
    ).to(args.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"使用设备: {args.device}")
    
    # 评估
    print("\n" + "=" * 60)
    print("评估模型...")
    print("=" * 60)
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(args.device)
            labels = labels.squeeze()
            
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 分类报告
    class_names = label_encoder.classes_
    report = classification_report(all_labels, all_preds, 
                                  target_names=class_names,
                                  output_dict=True)
    
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, 
                              target_names=class_names))
    
    # 保存结果
    import json
    with open(os.path.join(args.output_dir, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, class_names,
                         os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # 保存预测结果
    results_df = {
        'true_label': [class_names[label] for label in all_labels],
        'pred_label': [class_names[pred] for pred in all_preds],
        'confidence': [probs[pred] for probs, pred in zip(all_probs, all_preds)]
    }
    import pandas as pd
    pd.DataFrame(results_df).to_csv(
        os.path.join(args.output_dir, 'predictions.csv'), 
        index=False, 
        encoding='utf-8-sig'
    )
    
    print(f"\n结果已保存到: {args.output_dir}")
    print(f"  - evaluation_report.json: 详细评估报告")
    print(f"  - confusion_matrix.png: 混淆矩阵图")
    print(f"  - predictions.csv: 预测结果")


if __name__ == '__main__':
    main()

