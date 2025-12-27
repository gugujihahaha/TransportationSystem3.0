"""
评估与对比脚本 (Exp3 vs Exp4)
对比轨迹+KG 与 轨迹+KG+天气 的性能差异
"""
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# 导入 Exp3 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'exp3'))
from src.model import TransportationModeClassifier

# 导入 Exp4 模块
from src.model_weather import TransportationModeClassifierWithWeather


class ModelEvaluator:
    """模型评估器"""

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

    def load_model(self):
        """加载模型"""
        print(f"加载模型: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.label_encoder = checkpoint['label_encoder']
        self.config = checkpoint['model_config']

        # 判断是 Exp3 还是 Exp4
        if 'weather_feature_dim' in self.config:
            # Exp4 模型
            self.model = TransportationModeClassifierWithWeather(
                trajectory_feature_dim=self.config['trajectory_feature_dim'],
                kg_feature_dim=self.config['kg_feature_dim'],
                weather_feature_dim=self.config['weather_feature_dim'],
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_layers'],
                num_classes=self.config['num_classes'],
                dropout=self.config.get('dropout', 0.3)
            ).to(self.device)
            self.model_type = 'exp4'
        else:
            # Exp3 模型
            self.model = TransportationModeClassifier(
                trajectory_feature_dim=self.config['trajectory_feature_dim'],
                kg_feature_dim=self.config['kg_feature_dim'],
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_layers'],
                num_classes=self.config['num_classes'],
                dropout=self.config.get('dropout', 0.3)
            ).to(self.device)
            self.model_type = 'exp3'

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ 模型加载成功 (类型: {self.model_type})")
        print(f"  类别: {self.label_encoder.classes_}")

    def evaluate(self, dataloader) -> dict:
        """
        评估模型

        Args:
            dataloader: 数据加载器

        Returns:
            results: 评估结果字典
        """
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                if self.model_type == 'exp4':
                    trajectory_features, kg_features, weather_features, labels = batch
                    trajectory_features = trajectory_features.to(self.device)
                    kg_features = kg_features.to(self.device)
                    weather_features = weather_features.to(self.device)
                    labels = labels.to(self.device)

                    logits = self.model(trajectory_features, kg_features, weather_features)
                else:
                    trajectory_features, kg_features, labels = batch
                    trajectory_features = trajectory_features.to(self.device)
                    kg_features = kg_features.to(self.device)
                    labels = labels.to(self.device)

                    logits = self.model(trajectory_features, kg_features)

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

        # 分类报告
        report = classification_report(
            all_labels, all_preds,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)

        results = {
            'model_type': self.model_type,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'report': report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'true_labels': all_labels
        }

        return results


def plot_comparison(exp3_results: dict, exp4_results: dict, output_dir: str):
    """
    绘制对比图表

    Args:
        exp3_results: Exp3 评估结果
        exp4_results: Exp4 评估结果
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 整体指标对比
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
    exp3_scores = [exp3_results['accuracy'], exp3_results['macro_f1'], exp3_results['weighted_f1']]
    exp4_scores = [exp4_results['accuracy'], exp4_results['macro_f1'], exp4_results['weighted_f1']]

    x = np.arange(len(metrics))
    width = 0.35

    for i, (metric, exp3, exp4) in enumerate(zip(metrics, exp3_scores, exp4_scores)):
        ax[i].bar([0], [exp3], width, label='Exp3 (无天气)', color='skyblue')
        ax[i].bar([1], [exp4], width, label='Exp4 (含天气)', color='orange')
        ax[i].set_ylabel('Score')
        ax[i].set_title(metric)
        ax[i].set_xticks([0, 1])
        ax[i].set_xticklabels(['Exp3', 'Exp4'])
        ax[i].set_ylim([0, 1])
        ax[i].legend()

        # 显示数值
        ax[i].text(0, exp3 + 0.02, f'{exp3:.4f}', ha='center')
        ax[i].text(1, exp4 + 0.02, f'{exp4:.4f}', ha='center')

        # 显示提升百分比
        improvement = ((exp4 - exp3) / exp3) * 100
        color = 'green' if improvement > 0 else 'red'
        ax[i].text(0.5, 0.9, f'{improvement:+.2f}%',
                   ha='center', transform=ax[i].transAxes,
                   fontsize=12, color=color, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存整体对比图: {output_dir}/overall_comparison.png")

    # 2. 各类别 F1 对比
    classes = list(exp3_results['report'].keys())[:-3]  # 排除 accuracy, macro avg, weighted avg

    exp3_f1_scores = [exp3_results['report'][cls]['f1-score'] for cls in classes]
    exp4_f1_scores = [exp4_results['report'][cls]['f1-score'] for cls in classes]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.35

    ax.bar(x - width / 2, exp3_f1_scores, width, label='Exp3 (无天气)', color='skyblue')
    ax.bar(x + width / 2, exp4_f1_scores, width, label='Exp4 (含天气)', color='orange')

    ax.set_xlabel('Transportation Mode')
    ax.set_ylabel('F1-Score')
    ax.set_title('Per-Class F1-Score Comparison (Exp3 vs Exp4)')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 显示数值
    for i, (e3, e4) in enumerate(zip(exp3_f1_scores, exp4_f1_scores)):
        ax.text(i - width / 2, e3 + 0.02, f'{e3:.3f}', ha='center', fontsize=8)
        ax.text(i + width / 2, e4 + 0.02, f'{e4:.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_f1_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存各类别F1对比图: {output_dir}/per_class_f1_comparison.png")

    # 3. 混淆矩阵对比
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Exp3 混淆矩阵
    sns.heatmap(exp3_results['confusion_matrix'], annot=True, fmt='d',
                cmap='Blues', xticklabels=classes, yticklabels=classes,
                ax=axes[0])
    axes[0].set_title('Exp3 Confusion Matrix (无天气)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # Exp4 混淆矩阵
    sns.heatmap(exp4_results['confusion_matrix'], annot=True, fmt='d',
                cmap='Oranges', xticklabels=classes, yticklabels=classes,
                ax=axes[1])
    axes[1].set_title('Exp4 Confusion Matrix (含天气)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存混淆矩阵对比图: {output_dir}/confusion_matrices.png")


def save_comparison_report(exp3_results: dict, exp4_results: dict, output_dir: str):
    """
    保存对比报告

    Args:
        exp3_results: Exp3 评估结果
        exp4_results: Exp4 评估结果
        output_dir: 输出目录
    """
    report = {
        'evaluation_time': datetime.now().isoformat(),
        'exp3': {
            'accuracy': float(exp3_results['accuracy']),
            'macro_f1': float(exp3_results['macro_f1']),
            'weighted_f1': float(exp3_results['weighted_f1']),
            'per_class_metrics': {}
        },
        'exp4': {
            'accuracy': float(exp4_results['accuracy']),
            'macro_f1': float(exp4_results['macro_f1']),
            'weighted_f1': float(exp4_results['weighted_f1']),
            'per_class_metrics': {}
        },
        'improvements': {}
    }

    # 各类别指标
    classes = list(exp3_results['report'].keys())[:-3]
    for cls in classes:
        report['exp3']['per_class_metrics'][cls] = {
            'precision': float(exp3_results['report'][cls]['precision']),
            'recall': float(exp3_results['report'][cls]['recall']),
            'f1-score': float(exp3_results['report'][cls]['f1-score']),
            'support': int(exp3_results['report'][cls]['support'])
        }
        report['exp4']['per_class_metrics'][cls] = {
            'precision': float(exp4_results['report'][cls]['precision']),
            'recall': float(exp4_results['report'][cls]['recall']),
            'f1-score': float(exp4_results['report'][cls]['f1-score']),
            'support': int(exp4_results['report'][cls]['support'])
        }

    # 计算提升
    report['improvements']['accuracy'] = float(exp4_results['accuracy'] - exp3_results['accuracy'])
    report['improvements']['macro_f1'] = float(exp4_results['macro_f1'] - exp3_results['macro_f1'])
    report['improvements']['weighted_f1'] = float(exp4_results['weighted_f1'] - exp3_results['weighted_f1'])

    report['improvements']['accuracy_percent'] = float(
        ((exp4_results['accuracy'] - exp3_results['accuracy']) / exp3_results['accuracy']) * 100
    )
    report['improvements']['macro_f1_percent'] = float(
        ((exp4_results['macro_f1'] - exp3_results['macro_f1']) / exp3_results['macro_f1']) * 100
    )

    # 保存 JSON
    with open(os.path.join(output_dir, 'comparison_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ 保存对比报告: {output_dir}/comparison_report.json")

    # 保存文本报告
    with open(os.path.join(output_dir, 'comparison_report.txt'), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("实验对比报告: Exp3 (轨迹+KG) vs Exp4 (轨迹+KG+天气)\n")
        f.write("=" * 80 + "\n\n")

        f.write("【整体指标】\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'指标':<20} {'Exp3':<15} {'Exp4':<15} {'提升':<15} {'提升%':<15}\n")
        f.write("-" * 80 + "\n")

        metrics = [
            ('Accuracy', 'accuracy'),
            ('Macro F1', 'macro_f1'),
            ('Weighted F1', 'weighted_f1')
        ]

        for name, key in metrics:
            exp3_val = report['exp3'][key]
            exp4_val = report['exp4'][key]
            improvement = report['improvements'][key]
            improvement_pct = ((exp4_val - exp3_val) / exp3_val) * 100

            f.write(f"{name:<20} {exp3_val:<15.4f} {exp4_val:<15.4f} "
                    f"{improvement:+<15.4f} {improvement_pct:+<15.2f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("【各类别 F1-Score】\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'类别':<20} {'Exp3 F1':<15} {'Exp4 F1':<15} {'提升':<15}\n")
        f.write("-" * 80 + "\n")

        for cls in classes:
            exp3_f1 = report['exp3']['per_class_metrics'][cls]['f1-score']
            exp4_f1 = report['exp4']['per_class_metrics'][cls]['f1-score']
            improvement = exp4_f1 - exp3_f1

            f.write(f"{cls:<20} {exp3_f1:<15.4f} {exp4_f1:<15.4f} {improvement:+<15.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("【结论】\n")
        f.write("-" * 80 + "\n")

        if report['improvements']['accuracy'] > 0:
            f.write(f"✓ 天气数据的引入使准确率提升了 {report['improvements']['accuracy_percent']:.2f}%\n")
        else:
            f.write(f"✗ 天气数据对准确率的影响为 {report['improvements']['accuracy_percent']:.2f}%\n")

        if report['improvements']['macro_f1'] > 0:
            f.write(f"✓ 宏平均F1提升了 {report['improvements']['macro_f1_percent']:.2f}%\n")
        else:
            f.write(f"✗ 宏平均F1的变化为 {report['improvements']['macro_f1_percent']:.2f}%\n")

    print(f"✓ 保存文本报告: {output_dir}/comparison_report.txt")


def main():
    parser = argparse.ArgumentParser(description='评估并对比 Exp3 和 Exp4')
    parser.add_argument('--exp3_model', type=str,
                        default='../exp3/checkpoints/exp3_model.pth',
                        help='Exp3 模型路径')
    parser.add_argument('--exp4_model', type=str,
                        default='checkpoints/exp4_model.pth',
                        help='Exp4 模型路径')
    parser.add_argument('--test_data_exp3', type=str,
                        help='Exp3 测试数据路径 (可选)')
    parser.add_argument('--test_data_exp4', type=str,
                        help='Exp4 测试数据路径 (可选)')
    parser.add_argument('--output_dir', type=str,
                        default='results/comparison',
                        help='输出目录')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("实验对比评估: Exp3 (轨迹+KG) vs Exp4 (轨迹+KG+天气)")
    print("=" * 80)

    # 注意：这里需要根据实际情况加载测试数据
    # 由于完整的数据加载逻辑较复杂，这里提供框架
    # 实际使用时需要从缓存或重新生成测试集

    print("\n⚠️  注意：完整评估需要测试数据集")
    print("建议在训练脚本中保存测试集，或使用缓存的数据")
    print("\n示例输出格式已生成，请根据实际数据完善评估逻辑")

    # 创建示例报告
    os.makedirs(args.output_dir, exist_ok=True)

    example_report = {
        'evaluation_time': datetime.now().isoformat(),
        'note': '这是一个示例报告模板，实际评估需要加载完整的测试数据',
        'exp3': {
            'accuracy': 0.88,
            'macro_f1': 0.86,
            'weighted_f1': 0.88
        },
        'exp4': {
            'accuracy': 0.90,
            'macro_f1': 0.88,
            'weighted_f1': 0.90
        },
        'improvements': {
            'accuracy': 0.02,
            'macro_f1': 0.02,
            'accuracy_percent': 2.27,
            'macro_f1_percent': 2.33
        }
    }

    with open(os.path.join(args.output_dir, 'example_comparison_report.json'), 'w') as f:
        json.dump(example_report, f, indent=2)

    print(f"\n✓ 示例报告已保存至: {args.output_dir}/example_comparison_report.json")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()