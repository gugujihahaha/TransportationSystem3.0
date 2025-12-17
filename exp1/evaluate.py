import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 假设你的模型类和 Dataset 定义在这些文件中
from src.model import TransportationModeClassifier
from train import TrajectoryDataset


def main():
    # 基础配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)

    # ========================================================
    # 1. 加载缓存与模型
    # ========================================================
    print("正在加载缓存数据...")
    try:
        with open("cache/exp1_processed_features.pkl", "rb") as f:
            cache = pickle.load(f)
    except FileNotFoundError:
        print("❌ 错误：未找到 cache/exp1_processed_features.pkl，请确保 Exp1 预处理已完成。")
        return

    segments = cache["segments"]
    label_encoder = cache["label_encoder"]
    class_names = label_encoder.classes_

    print(f"正在加载模型支架 (Device: {device})...")
    # 使用 weights_only=False 确保安全加载包含自定义对象的模型
    ckpt = torch.load("checkpoints/exp1_model.pth", map_location=device, weights_only=False)
    config = ckpt["model_config"]

    model = TransportationModeClassifier(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ========================================================
    # 2. 推理循环
    # ========================================================
    dataset = TrajectoryDataset(segments, label_encoder)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    all_preds, all_labels, all_conf = [], [], []

    print("开始全量推理评估...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.squeeze().numpy())
            all_conf.extend(probs.max(1).values.cpu().numpy())

    # ========================================================
    # 3. 生成报告与保存数据
    # ========================================================
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # 打印控制台报告
    print("\n" + "=" * 50)
    print("      Exp1 评估报告 (Base: 纯轨迹流)")
    print("=" * 50)
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    # 自动提取结论
    metrics_only = {k: v for k, v in report.items() if k in class_names}
    best_cls = max(metrics_only, key=lambda x: metrics_only[x]['f1-score'])
    worst_cls = min(metrics_only, key=lambda x: metrics_only[x]['f1-score'])

    print("-" * 50)
    print(f"🥇 表现最佳类别: {best_cls} (F1: {metrics_only[best_cls]['f1-score']:.4f})")
    print(f"⚠️ 识别难点类别: {worst_cls} (F1: {metrics_only[worst_cls]['f1-score']:.4f})")
    print("\n[Exp1 核心结论]")
    print(f"模型整体准确率为 {report['accuracy']:.2%}。")
    print(f"在仅使用运动学特征的情况下，模型难以区分特征相近的类别。")
    print("=" * 50)

    # 保存 JSON 报告
    with open(os.path.join(output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # ========================================================
    # 4. 可视化绘制
    # ========================================================
    print("\n正在生成可视化图表...")

    # --- 绘制柱状图 (Precision, Recall, F1) ---
    metrics = ['precision', 'recall', 'f1-score']
    plot_data = []
    for cls in class_names:
        for m in metrics:
            plot_data.append({
                'Class': cls,
                'Metric': m,
                'Value': report[cls][m]
            })
    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(x='Class', y='Value', hue='Metric', data=df_plot, palette='viridis')

    plt.title('Exp1: Per-class Performance Metrics', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.ylabel('Score')
    plt.xlabel('Transportation Mode')
    plt.xticks(rotation=30)
    plt.legend(title='Metrics', bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_metrics.png"), dpi=300)
    plt.close()

    # --- 绘制混淆矩阵 (Confusion Matrix) ---
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title('Exp1: Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # ========================================================
    # 5. 保存详细预测结果到 CSV (新增)
    # ========================================================
    print("正在保存预测详情到 CSV...")

    results_df = pd.DataFrame({
        'true_label': [class_names[i] for i in all_labels],
        'pred_label': [class_names[i] for i in all_preds],
        'confidence': all_conf
    })

    # 添加正确性判断列，方便后续筛选分析
    results_df['is_correct'] = results_df['true_label'] == results_df['pred_label']

    csv_path = os.path.join(output_dir, "predictions_exp1.csv")
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"✅ 评估完成！所有结果已保存至: {output_dir}")


if __name__ == "__main__":
    # 处理可能的中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    main()