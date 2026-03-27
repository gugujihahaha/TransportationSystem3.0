"""
exp1/evaluate.py（修复版）
======================
修复内容：
1. 从 EXP1 缓存加载数据（格式：traj_9, stats_18, label_encoded）
2. 使用共享测试集索引（shared_split.pkl）
3. 模型调用：model(traj, segment_stats=stats)
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
os.chdir(SCRIPT_DIR)

from src.model import TransportationModeClassifier
from train import TrajectoryDataset

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

MODEL_PATH         = 'checkpoints/exp1_model.pth'
EXP1_CACHE_PATH    = os.path.join(SCRIPT_DIR, 'cache', 'exp1_processed_features.pkl')
OUTPUT_DIR         = 'evaluation_results'
DEVICE             = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print("\n" + "=" * 60)
    print("Exp1 模型评估 (9维轨迹特征)")
    print("=" * 60)

    # 1. 加载模型
    print(f"\n[1/4] 加载模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    le = checkpoint['label_encoder']
    config = checkpoint['model_config']
    norm_params = checkpoint.get('norm_params', {})
    class_names = le.classes_

    traj_mean  = norm_params.get('traj_mean', None)
    traj_std   = norm_params.get('traj_std', None)
    stats_mean = norm_params.get('stats_mean', None)
    stats_std  = norm_params.get('stats_std', None)

    if traj_mean is None:
        raise ValueError("checkpoint 中缺少 norm_params，请重新训练")

    model = TransportationModeClassifier(
        trajectory_feature_dim=9,
        segment_stats_dim=18,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.3)
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   ✓ 模型加载完成")

    # 2. 加载测试数据
    print(f"\n[2/4] 加载测试数据...")
    if not os.path.exists(EXP1_CACHE_PATH):
        raise FileNotFoundError(f"EXP1 缓存不存在: {EXP1_CACHE_PATH}\n请先运行 exp1/train.py")

    with open(EXP1_CACHE_PATH, 'rb') as f:
        all_data, label_encoder, _ = pickle.load(f)

    # 使用与train.py相同的划分逻辑（70/10/20，random_state=42）
    from sklearn.model_selection import train_test_split
    all_indices = np.arange(len(all_data))
    labels_encoded = [item[2] for item in all_data]

    train_indices, temp_indices = train_test_split(
        all_indices, test_size=0.3, random_state=42, stratify=labels_encoded
    )
    temp_labels = [labels_encoded[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.6667, random_state=42, stratify=temp_labels
    )

    test_data = [all_data[i] for i in test_indices]
    print(f"   ✓ 测试样本: {len(test_data)}")

    # 3. 推理
    print(f"\n[3/4] 推理...")
    dataset = TrajectoryDataset(
        test_data, le,
        traj_mean=traj_mean, traj_std=traj_std,
        stats_mean=stats_mean, stats_std=stats_std
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for x, stats, labels in tqdm(loader, desc="Evaluating"):
            x      = x.to(DEVICE)
            stats  = stats.to(DEVICE)
            logits = model(x, segment_stats=stats)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(logits, dim=1)
            y_true.extend(labels.squeeze().cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_probs = np.array(y_probs)

    # 4. 报告
    print(f"\n[4/4] 生成报告...")
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names,
                                 zero_division=0, digits=4))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_dict = classification_report(y_true, y_pred, target_names=class_names,
                                         output_dict=True, zero_division=0)

    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)

    conf_list = [float(y_probs[i, p]) for i, p in enumerate(y_pred)]
    pd.DataFrame({
        'true_label': [class_names[i] for i in y_true],
        'pred_label': [class_names[i] for i in y_pred],
        'confidence': conf_list,
        'correct': y_true == y_pred
    }).to_csv(os.path.join(OUTPUT_DIR, 'predictions_exp1.csv'), index=False, encoding='utf-8-sig')

    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Exp1 Confusion Matrix (Trajectory Features Only)', fontsize=14)
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"⚠️ 混淆矩阵生成失败: {e}")

    errors_df = pd.DataFrame({
        'true_label': [class_names[i] for i in y_true],
        'pred_label': [class_names[i] for i in y_pred],
        'confidence': conf_list
    })
    errors_df[errors_df['true_label'] != errors_df['pred_label']].to_csv(
        os.path.join(OUTPUT_DIR, 'error_analysis.csv'), index=False, encoding='utf-8-sig'
    )

    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)
    print(f"总样本数: {len(y_true)}")
    print(f"准确率:   {report_dict['accuracy']:.4f}")
    print(f"加权 F1:  {report_dict['weighted avg']['f1-score']:.4f}")
    print(f"宏平均 F1:{report_dict['macro avg']['f1-score']:.4f}")
    print(f"\n✅ 结果保存至: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()