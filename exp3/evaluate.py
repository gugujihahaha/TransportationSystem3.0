"""
Exp3 评估脚本
与 train.py 的数据划分方式完全一致（70/10/20，相同 random_state）
"""
import os
import sys
import json
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
os.chdir(SCRIPT_DIR)

from src.model_weather import TransportationModeClassifierWithWeather
from train import TrajectoryDatasetWithWeather

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def main():
    MODEL_PATH  = 'checkpoints/exp3_model.pth'
    CACHE_PATH  = 'cache/processed_features_exp3.pkl'
    OUTPUT_DIR  = 'evaluation_results'
    DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "=" * 60)
    print("Exp3 模型评估 (轨迹+空间+天气)")
    print("=" * 60)
    print(f"设备: {DEVICE}")

    # 1. 加载模型
    print(f"\n[1/5] 正在加载模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型: {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    le     = checkpoint['label_encoder']
    config = checkpoint['model_config']
    class_names = le.classes_

    norm_params  = checkpoint.get('norm_params', {})
    traj_mean    = norm_params.get('traj_mean', None)
    traj_std     = norm_params.get('traj_std', None)
    weather_mean = norm_params.get('weather_mean', None)
    weather_std  = norm_params.get('weather_std', None)
    stats_mean   = norm_params.get('stats_mean', None)
    stats_std    = norm_params.get('stats_std', None)

    print(f"   模型配置:")
    print(f"     - 轨迹维度: {config['trajectory_feature_dim']}")
    print(f"     - 天气维度: {config['weather_feature_dim']}")
    print(f"     - 类别数:   {config['num_classes']}")
    print(f"     - 类别:     {list(class_names)}")

    model = TransportationModeClassifierWithWeather(
        trajectory_feature_dim=config['trajectory_feature_dim'],
        weather_feature_dim=config['weather_feature_dim'],
        segment_stats_dim=config.get('segment_stats_dim', 18),
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.3)
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("   ✓ 模型加载完成")

    # 2. 加载特征缓存
    print(f"\n[2/5] 正在加载特征缓存...")
    if not os.path.exists(CACHE_PATH):
        print(f"❌ 找不到缓存: {CACHE_PATH}")
        return
    with open(CACHE_PATH, 'rb') as f:
        all_features, cached_le, cleaning_stats = pickle.load(f)
    print(f"   ✓ 加载完成: {len(all_features)} 个样本")

    # 3. 准备测试数据
    # ✅ 与 train.py 完全一致的划分方式
    print(f"\n[3/5] 正在准备测试数据...")
    all_indices      = np.arange(len(all_features))
    labels_stratify  = [item[-1] for item in all_features]

    train_indices, temp_indices = train_test_split(
        all_indices, test_size=0.3, random_state=42, stratify=labels_stratify
    )
    temp_labels = [labels_stratify[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.6667, random_state=42, stratify=temp_labels
    )

    # 注意：evaluate 只用 test_indices，不做 oversample（测试集不应被增强）
    dataset = TrajectoryDatasetWithWeather(
        all_features,
        traj_mean=traj_mean, traj_std=traj_std,
        weather_mean=weather_mean, weather_std=weather_std,
        stats_mean=stats_mean, stats_std=stats_std
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=64, shuffle=False, num_workers=0
    )
    print(f"   ✓ 测试集大小: {len(test_indices)} 个样本")

    # 4. 推理
    print(f"\n[4/5] 正在进行模型推理...")
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for traj, weather, stats, labels in tqdm(test_loader, desc="Evaluation Progress"):
            traj    = traj.to(DEVICE)
            weather = weather.to(DEVICE)
            stats   = stats.to(DEVICE)

            logits = model(traj, weather, segment_stats=stats)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_probs = np.array(y_probs)

    # 5. 生成报告
    print(f"\n[5/5] 正在生成评估报告...")
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names,
                                 zero_division=0, digits=4))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_dict = classification_report(y_true, y_pred, target_names=class_names,
                                         output_dict=True, zero_division=0)

    # JSON
    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)
    print(f"   ✓ 保存: evaluation_results/evaluation_report.json")

    # CSV 预测
    conf_list = [float(y_probs[i, p]) for i, p in enumerate(y_pred)]
    import pandas as pd
    pd.DataFrame({
        'true_label': [class_names[i] for i in y_true],
        'pred_label': [class_names[i] for i in y_pred],
        'confidence': conf_list,
        'correct':    y_true == y_pred
    }).to_csv(os.path.join(OUTPUT_DIR, 'predictions_exp3.csv'), index=False, encoding='utf-8-sig')
    print(f"   ✓ 保存: evaluation_results/predictions_exp3.csv")

    # 混淆矩阵
    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Exp3 Confusion Matrix (Trajectory + Spatial + Weather)', fontsize=14)
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
        plt.close()
        print(f"   ✓ 保存: evaluation_results/confusion_matrix.png")
    except Exception as e:
        print(f"   ⚠️ 混淆矩阵生成失败: {e}")

    # F1图
    try:
        f1_scores = [report_dict[cls]['f1-score'] for cls in class_names]
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(class_names), y=f1_scores, color='darkorange')
        plt.title('Exp3 F1-Score by Transportation Mode', fontsize=14)
        plt.xlabel('Transportation Mode'); plt.ylabel('F1-Score')
        plt.ylim(0, 1.0)
        for i, v in enumerate(f1_scores):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=10)
        plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'per_class_f1_scores.png'), dpi=300)
        plt.close()
        print(f"   ✓ 保存: evaluation_results/per_class_f1_scores.png")
    except Exception as e:
        print(f"   ⚠️ F1图生成失败: {e}")

    # 错误分析
    errors_df = pd.DataFrame({
        'true_label': [class_names[i] for i in y_true],
        'pred_label': [class_names[i] for i in y_pred],
        'confidence': conf_list
    })
    errors_df = errors_df[errors_df['true_label'] != errors_df['pred_label']]
    errors_df.to_csv(os.path.join(OUTPUT_DIR, 'error_analysis.csv'), index=False, encoding='utf-8-sig')
    print(f"   ✓ 保存: evaluation_results/error_analysis.csv ({len(errors_df)} 个错误样本)")

    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)
    print(f"总样本数: {len(y_true)}")
    print(f"正确预测: {(y_true == y_pred).sum()}")
    print(f"错误预测: {(y_true != y_pred).sum()}")
    print(f"准确率:   {report_dict['accuracy']:.4f}")
    print(f"加权 F1:  {report_dict['weighted avg']['f1-score']:.4f}")
    print(f"宏平均 F1:{report_dict['macro avg']['f1-score']:.4f}")
    print(f"\n✅ 结果保存至: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()