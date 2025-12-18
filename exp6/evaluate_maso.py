"""
MASO-MSF 模型评估脚本 (Exp6) - 修复版
"""

import os
import json
import pickle
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import GeoLifeDataLoader, preprocess_segments
from src.maso import MASOConfig, MASOFeatureOrganizer
from src.model_msf import SimpleMSFModel
from train_maso import MASOTrajectoryDataset, load_geolife_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="checkpoints/exp6_maso_model.pth")
    parser.add_argument("--geolife_root", default="../data/Geolife Trajectories 1.3")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="results/exp6")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ========================================================
    # 1. 加载模型和配置
    # ========================================================
    print("=" * 60)
    print("加载模型和配置...")
    print("=" * 60)

    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 找不到模型文件 {args.model_path}")
        return

    # 加载检查点
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)

    label_encoder = checkpoint['label_encoder']
    maso_config = checkpoint['maso_config']
    model_config = checkpoint['model_config']

    print(f"\nMASO配置:")
    print(f"  K (对象数): {maso_config.K}")
    print(f"  N (空间范围数): {maso_config.N}")
    print(f"  M (图像尺寸数): {maso_config.M}")
    print(f"  L (属性维度): {maso_config.L}")
    print(f"  总特征维度: {maso_config.L * maso_config.N * maso_config.M}")

    # 创建模型
    model = SimpleMSFModel(
        input_channels=model_config['input_channels'],
        num_objects=model_config['num_objects'],
        num_classes=model_config['num_classes'],
        img_size=model_config['img_size']
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\n✅ 模型加载成功!")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========================================================
    # 2. 加载测试数据
    # ========================================================
    print("\n加载测试数据...")
    segments = load_geolife_data(args.geolife_root, max_users=None)

    TARGET_MODES = ['walk', 'bike', 'car & taxi', 'bus', 'train', 'subway']
    segments = [s for s in segments if s[1] in TARGET_MODES]

    print(f"总轨迹段数: {len(segments)}")

    # ========================================================
    # 3. 创建数据集和数据加载器
    # ========================================================
    dataset = MASOTrajectoryDataset(segments, label_encoder, maso_config)

    # 为了简化,使用全部数据进行评估
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # ========================================================
    # 4. 推理评估
    # ========================================================
    print("\n开始推理...")
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(args.device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.squeeze().cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ========================================================
    # 5. 生成评估报告
    # ========================================================
    print("\n生成评估报告...")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    # 打印控制台报告
    print("\n" + "=" * 60)
    print("      Exp6 评估报告 (MASO-MSF 模型)")
    print("=" * 60)
    print(classification_report(
        all_labels,
        all_preds,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    # 分析最佳和最差类别
    metrics_only = {k: v for k, v in report.items() if k in label_encoder.classes_}
    best_cls = max(metrics_only, key=lambda x: metrics_only[x]['f1-score'])
    worst_cls = min(metrics_only, key=lambda x: metrics_only[x]['f1-score'])

    print("-" * 60)
    print(f"🥇 表现最佳: {best_cls} (F1: {metrics_only[best_cls]['f1-score']:.4f})")
    print(f"⚠️  识别难点: {worst_cls} (F1: {metrics_only[worst_cls]['f1-score']:.4f})")
    print(f"\n📊 整体准确率: {report['accuracy']:.2%}")
    print("=" * 60)

    # ========================================================
    # 6. 保存结果
    # ========================================================
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存JSON报告
    with open(os.path.join(args.output_dir, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # ========================================================
    # 7. 绘制可视化
    # ========================================================
    print("\n生成可视化图表...")

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title('Exp6: 混淆矩阵 (MASO-MSF模型)', fontsize=14, fontweight='bold')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # 各类别性能指标
    metrics = ['precision', 'recall', 'f1-score']
    plot_data = []
    for cls in label_encoder.classes_:
        for m in metrics:
            plot_data.append({
                'Class': cls,
                'Metric': m,
                'Value': report[cls][m]
            })
    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Class', y='Value', hue='Metric', data=df_plot, palette='viridis')
    plt.title('Exp6: 各类别性能指标 (MASO-MSF模型)', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.ylabel('得分')
    plt.xlabel('交通方式')
    plt.xticks(rotation=30)
    plt.legend(title='指标', bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "per_class_metrics.png"), dpi=300)
    plt.close()

    # 保存预测详情
    results_df = pd.DataFrame({
        'true_label': [label_encoder.classes_[i] for i in all_labels],
        'pred_label': [label_encoder.classes_[i] for i in all_preds],
        'confidence': all_probs.max(axis=1)
    })
    results_df['is_correct'] = results_df['true_label'] == results_df['pred_label']
    results_df.to_csv(
        os.path.join(args.output_dir, "predictions.csv"),
        index=False,
        encoding='utf-8-sig'
    )

    print(f"✅ 评估完成! 结果已保存至: {args.output_dir}")

    # ========================================================
    # 8. Exp1 vs Exp6 对比总结
    # ========================================================
    print("\n" + "=" * 60)
    print("Exp1 vs Exp6 性能对比")
    print("=" * 60)
    print(f"Exp1 准确率 (轨迹特征):     ~78%")
    print(f"Exp6 准确率 (MASO特征):     {report['accuracy']:.2%}")
    print(f"性能提升:                    {(report['accuracy'] - 0.78) * 100:+.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()