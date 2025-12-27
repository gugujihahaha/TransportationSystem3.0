"""
评估脚本 (Exp4)
单独评估 Exp4 模型性能
"""
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from torch.utils.data import DataLoader
import pickle

# 导入模型
from src.model_weather import TransportationModeClassifierWithWeather


class Exp4Evaluator:
    """Exp4 模型评估器"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        初始化评估器

        Args:
            model_path: 模型检查点路径
            device: 设备
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.label_encoder = None
        self.config = None

        self.load_model()

    def load_model(self):
        """加载模型"""
        print(f"\n{'=' * 80}")
        print(f"加载 Exp4 模型: {self.model_path}")
        print(f"{'=' * 80}\n")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.label_encoder = checkpoint['label_encoder']
        self.config = checkpoint['model_config']

        # 创建模型
        self.model = TransportationModeClassifierWithWeather(
            trajectory_feature_dim=self.config['trajectory_feature_dim'],
            kg_feature_dim=self.config['kg_feature_dim'],
            weather_feature_dim=self.config['weather_feature_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_classes=self.config['num_classes'],
            dropout=self.config.get('dropout', 0.3)
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ 模型加载成功")
        print(f"  设备: {self.device}")
        print(f"  类别: {self.label_encoder.classes_}")
        print(f"  轨迹特征维度: {self.config['trajectory_feature_dim']}")
        print(f"  KG特征维度: {self.config['kg_feature_dim']}")
        print(f"  天气特征维度: {self.config['weather_feature_dim']}")
        print(
            f"  总输入维度: {self.config['trajectory_feature_dim'] + self.config['kg_feature_dim'] + self.config['weather_feature_dim']}")

    def evaluate(self, test_loader: DataLoader) -> dict:
        """
        评估模型

        Args:
            test_loader: 测试数据加载器

        Returns:
            results: 评估结果字典
        """
        print(f"\n{'=' * 80}")
        print("开始评估...")
        print(f"{'=' * 80}\n")

        all_preds = []
        all_labels = []
        all_probs = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                trajectory_features, kg_features, weather_features, labels = batch

                trajectory_features = trajectory_features.to(self.device)
                kg_features = kg_features.to(self.device)
                weather_features = weather_features.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                logits = self.model(trajectory_features, kg_features, weather_features)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                if (batch_idx + 1) % 50 == 0:
                    print(f"  已处理 {batch_idx + 1}/{len(test_loader)} 批次")

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

        # 分类报告
        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)

        print(f"\n{'=' * 80}")
        print("评估完成！")
        print(f"{'=' * 80}\n")
        print(f"总体准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"宏平均 F1: {macro_f1:.4f}")
        print(f"加权 F1: {weighted_f1:.4f}")

        results = {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'report': report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'true_labels': all_labels,
            'probabilities': all_probs
        }

        return results

    def save_results(self, results: dict, output_dir: str):
        """
        保存评估结果

        Args:
            results: 评估结果
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存分类报告（JSON）
        report_data = {
            'evaluation_time': datetime.now().isoformat(),
            'model_path': self.model_path,
            'device': self.device,
            'accuracy': results['accuracy'],
            'macro_f1': results['macro_f1'],
            'weighted_f1': results['weighted_f1'],
            'per_class_metrics': {}
        }

        classes = self.label_encoder.classes_
        for cls in classes:
            report_data['per_class_metrics'][cls] = {
                'precision': float(results['report'][cls]['precision']),
                'recall': float(results['report'][cls]['recall']),
                'f1-score': float(results['report'][cls]['f1-score']),
                'support': int(results['report'][cls]['support'])
            }

        with open(os.path.join(output_dir, 'evaluation_report.json'), 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"✓ 保存评估报告: {output_dir}/evaluation_report.json")

        # 2. 保存分类报告（文本）
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Exp4 模型评估报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型路径: {self.model_path}\n")
            f.write(f"设备: {self.device}\n\n")

            f.write("=" * 80 + "\n")
            f.write("整体指标\n")
            f.write("=" * 80 + "\n")
            f.write(f"准确率 (Accuracy):      {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)\n")
            f.write(f"宏平均 F1 (Macro F1):   {results['macro_f1']:.4f}\n")
            f.write(f"加权 F1 (Weighted F1):  {results['weighted_f1']:.4f}\n\n")

            f.write("=" * 80 + "\n")
            f.write("各类别详细指标\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'类别':<15} {'精确率':<12} {'召回率':<12} {'F1分数':<12} {'样本数':<10}\n")
            f.write("-" * 80 + "\n")

            for cls in classes:
                metrics = results['report'][cls]
                f.write(f"{cls:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                        f"{metrics['f1-score']:<12.4f} {metrics['support']:<10}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("详细分类报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(classification_report(
                results['true_labels'],
                results['predictions'],
                target_names=classes,
                zero_division=0
            ))

        print(f"✓ 保存文本报告: {output_dir}/classification_report.txt")

        # 3. 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Exp4 (轨迹 + KG + 天气)', fontsize=14, pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存混淆矩阵: {output_dir}/confusion_matrix.png")

        # 4. 绘制各类别指标条形图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        metrics_names = ['precision', 'recall', 'f1-score']
        metrics_labels = ['Precision', 'Recall', 'F1-Score']

        for idx, (metric_name, metric_label) in enumerate(zip(metrics_names, metrics_labels)):
            values = [results['report'][cls][metric_name] for cls in classes]

            bars = axes[idx].bar(range(len(classes)), values, color='skyblue', edgecolor='navy', alpha=0.7)
            axes[idx].set_xlabel('Transportation Mode', fontsize=11)
            axes[idx].set_ylabel(metric_label, fontsize=11)
            axes[idx].set_title(f'Per-Class {metric_label}', fontsize=12, pad=10)
            axes[idx].set_xticks(range(len(classes)))
            axes[idx].set_xticklabels(classes, rotation=45, ha='right')
            axes[idx].set_ylim([0, 1.0])
            axes[idx].grid(axis='y', alpha=0.3, linestyle='--')

            # 在柱子上显示数值
            for i, (bar, val) in enumerate(zip(bars, values)):
                axes[idx].text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存各类别指标: {output_dir}/per_class_metrics.png")

        # 5. 保存预测结果
        predictions_df = pd.DataFrame({
            'true_label': [classes[i] for i in results['true_labels']],
            'predicted_label': [classes[i] for i in results['predictions']],
            'correct': results['true_labels'] == results['predictions']
        })

        # 添加每个类别的预测概率
        for i, cls in enumerate(classes):
            predictions_df[f'prob_{cls}'] = results['probabilities'][:, i]

        predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
        print(f"✓ 保存预测结果: {output_dir}/predictions.csv")

        print(f"\n{'=' * 80}")
        print(f"所有结果已保存到: {output_dir}")
        print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description='评估 Exp4 模型')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints/exp4_model.pth',
                        help='Exp4 模型路径')
    parser.add_argument('--test_data_path', type=str,
                        default='cache/processed_features_weather_v1.pkl',
                        help='测试数据路径（缓存文件）')
    parser.add_argument('--output_dir', type=str,
                        default='results/exp4',
                        help='输出目录')
    parser.add_argument('--batch_size', type=int,
                        default=32,
                        help='批次大小')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')

    args = parser.parse_args()

    # 创建评估器
    evaluator = Exp4Evaluator(args.model_path, args.device)

    # 加载测试数据
    print(f"\n{'=' * 80}")
    print(f"加载测试数据: {args.test_data_path}")
    print(f"{'=' * 80}\n")

    if not os.path.exists(args.test_data_path):
        print(f"❌ 错误: 测试数据文件不存在: {args.test_data_path}")
        print("\n提示: 请先运行训练脚本生成缓存数据，或提供正确的测试数据路径")
        return

    with open(args.test_data_path, 'rb') as f:
        all_features_and_labels, label_encoder = pickle.load(f)

    print(f"✓ 数据加载成功: {len(all_features_and_labels)} 条样本")

    # 创建数据加载器
    from train import TrajectoryDatasetWithWeather
    from sklearn.model_selection import train_test_split

    dataset = TrajectoryDatasetWithWeather(all_features_and_labels)

    # 划分测试集（使用相同的随机种子确保一致性）
    labels_for_stratify = [label for _, _, _, label in all_features_and_labels]
    _, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=labels_for_stratify
    )

    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"✓ 测试集大小: {len(test_dataset)}")

    # 评估
    results = evaluator.evaluate(test_loader)

    # 保存结果
    evaluator.save_results(results, args.output_dir)

    # 打印总结
    print("\n" + "=" * 80)
    print("评估总结")
    print("=" * 80)
    print(f"准确率:     {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)")
    print(f"宏平均 F1:  {results['macro_f1']:.4f}")
    print(f"加权 F1:    {results['weighted_f1']:.4f}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()