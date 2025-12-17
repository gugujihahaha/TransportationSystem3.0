# ============================================================
# exp2/evaluate.py - 评估脚本 (双输入：轨迹特征 + 知识图谱特征)
# ============================================================
"""
评估脚本 - 结合轨迹特征和知识图谱特征 (Exp2)

功能：
- Accuracy / Precision / Recall / F1
- Confusion Matrix
- Per-class metrics
- 预测结果 CSV

兼容：
- PyTorch >= 2.6
- 安全加载 LabelEncoder（safe_globals）
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
from sklearn.preprocessing import LabelEncoder
import torch.serialization
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 模型与数据处理模块
from src.model import TransportationModeClassifier

# ============================================================
# 双输入数据集（与 Exp2 train.py 完全一致）
# ============================================================
class DualFeatureDataset(Dataset):
    """
    segments: List[
        (traj_features_np, kg_features_np, label_encoded)
        or
        (traj_features_np, kg_features_np, label_str)
    ]
    """

    def __init__(self, segments, label_encoder):
        self.segments = segments
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        traj_features, kg_features, label = self.segments[idx]

        # 兼容字符串或已编码标签
        if isinstance(label, str):
            label = self.label_encoder.transform([label])[0]

        traj_tensor = torch.FloatTensor(traj_features)
        kg_tensor = torch.FloatTensor(kg_features)
        label_tensor = torch.LongTensor([label])

        return traj_tensor, kg_tensor, label_tensor


# ============================================================
# 加载评估数据（从缓存，保证与训练完全一致）
# ============================================================
CACHE_DIR = 'cache'
PROCESSED_FEATURE_CACHE_PATH = os.path.join(CACHE_DIR, 'processed_features.pkl')


def load_evaluation_data():
    if not os.path.exists(PROCESSED_FEATURE_CACHE_PATH):
        raise FileNotFoundError(
            f"❌ 未找到特征缓存文件: {PROCESSED_FEATURE_CACHE_PATH}\n"
            f"请先运行 Exp2 的 train.py 生成缓存。"
        )

    print("\n========== 加载评估特征（缓存） ==========")

    with open(PROCESSED_FEATURE_CACHE_PATH, 'rb') as f:
        # ✅ PyTorch 2.6 正确用法：只传一个 dict
        with torch.serialization.safe_globals({
            LabelEncoder: LabelEncoder
        }):
            all_features_and_labels, label_encoder = pickle.load(f)

    print(f"✓ 加载完成，共 {len(all_features_and_labels)} 条样本")

    # 最终类别（car & taxi 已合并）
    TARGET_MODES_FINAL = [
        'Walk', 'Bike', 'Bus',
        'Car & taxi', 'Train',
        'Airplane', 'Other'
    ]

    # 只保留目标类别
    valid_segments = [
        seg for seg in all_features_and_labels
        if label_encoder.classes_[seg[2]] in TARGET_MODES_FINAL
    ]

    print(f"✓ 有效评估样本数: {len(valid_segments)}")
    return valid_segments, label_encoder


# ============================================================
# 模型评估
# ============================================================
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for traj_features, kg_features, labels in tqdm(dataloader, desc="评估中"):
            traj_features = traj_features.to(device)
            kg_features = kg_features.to(device)
            labels = labels.squeeze().to(device)

            logits = model(traj_features, kg_features)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_preds, all_labels, all_probs


# ============================================================
# 可视化
# ============================================================
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
    plt.title('Confusion Matrix (Exp2)', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_per_class_metrics(report, class_names, save_path):
    metrics = ['precision', 'recall', 'f1-score']
    data = {m: [] for m in metrics}

    for cls in class_names:
        for m in metrics:
            data[m].append(report[cls][m])

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(12, 6))
    for i, m in enumerate(metrics):
        plt.bar(x + i * width, data[m], width, label=m)

    plt.xticks(x + width, class_names, rotation=30)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='评估 Exp2 双输入模型')
    parser.add_argument('--model_path', type=str, default='checkpoints/exp2_model.pth')
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================================
    # 加载模型
    # ========================================================
    print("=" * 60)
    print("加载模型和 LabelEncoder")

    with torch.serialization.safe_globals({
        LabelEncoder: LabelEncoder
    }):
        checkpoint = torch.load(
            args.model_path,
            map_location=args.device,
            weights_only=False
        )

    label_encoder = checkpoint['label_encoder']
    config = checkpoint['model_config']

    model = TransportationModeClassifier(
        trajectory_feature_dim=config['trajectory_feature_dim'],
        kg_feature_dim=config['kg_feature_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.3)
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ 模型加载成功（Epoch {checkpoint.get('epoch', '?')}）")
    print(f"类别: {list(label_encoder.classes_)}")

    # ========================================================
    # 加载评估数据
    # ========================================================
    segments, label_encoder = load_evaluation_data()
    dataset = DualFeatureDataset(segments, label_encoder)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    class_names = label_encoder.classes_
    num_classes = len(class_names)

    # ========================================================
    # 评估
    # ========================================================
    preds, labels, probs_raw = evaluate_model(model, dataloader, args.device)

    report = classification_report(
        labels, preds,
        target_names=class_names,
        labels=np.arange(num_classes),
        output_dict=True,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    print(classification_report(
        labels, preds,
        target_names=class_names,
        labels=np.arange(num_classes),
        zero_division=0
    ))

    # ========================================================
    # 保存结果
    # ========================================================
    with open(os.path.join(args.output_dir, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    plot_confusion_matrix(
        labels, preds, class_names,
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )

    plot_per_class_metrics(
        report, class_names,
        os.path.join(args.output_dir, 'per_class_metrics.png')
    )

    confidences = [p[pred] for p, pred in zip(probs_raw, preds)]
    df = pd.DataFrame({
        'true_label': [class_names[i] for i in labels],
        'pred_label': [class_names[i] for i in preds],
        'confidence': confidences
    })

    df.to_csv(
        os.path.join(args.output_dir, 'predictions.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    print(f"\n✓ 所有评估结果已保存至: {args.output_dir}")


if __name__ == '__main__':
    main()
