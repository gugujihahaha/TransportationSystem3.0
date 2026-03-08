"""
exp4/evaluate.py（修复版）
======================
修复内容：
1. 去掉 placeholder，模型是单编码器不需要
2. Dataset 格式改为 (traj_21, stats_18, label_encoded)
3. 推理时调用 model(traj, segment_stats=stats)
4. 从 EXP4 缓存加载数据，使用共享测试集索引
"""
import os
import sys
import json
import pickle
import argparse
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

from exp2.src.model import TransportationModeClassifier
from src.focal_loss import LabelSmoothingFocalLoss

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

EXP4_FEATURE_CACHE = os.path.join(SCRIPT_DIR, 'cache', 'processed_features_exp4.pkl')
SHARED_TEST_PATH  = os.path.join(PARENT_DIR, 'data', 'processed', 'shared_test_indices.pkl')
OUTPUT_DIR         = 'evaluation_results'


# ✅ 修复：去掉 placeholder
class TrajectoryDatasetExp4(Dataset):
    def __init__(self, all_features,
                 traj_mean=None, traj_std=None,
                 stats_mean=None, stats_std=None):
        self.data = all_features
        self.traj_mean  = traj_mean
        self.traj_std   = traj_std
        self.stats_mean = stats_mean
        self.stats_std  = stats_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj_21, stats, label = self.data[idx]
        traj  = traj_21.astype(np.float32)
        stats = stats.astype(np.float32)
        if self.traj_mean is not None:
            traj = (traj - self.traj_mean) / self.traj_std
        if self.stats_mean is not None:
            stats = (stats - self.stats_mean) / self.stats_std
        return (torch.FloatTensor(traj),
                torch.FloatTensor(stats),
                torch.LongTensor([label])[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/exp4_model.pth')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device',     default='cpu')
    args = parser.parse_args()

    DEVICE = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("Exp4 模型评估 (LabelSmoothing + Focal Loss)")
    print("=" * 60)

    # 1. 加载模型
    print(f"\n[1/4] 加载模型: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"找不到模型: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    label_encoder = checkpoint['label_encoder']
    config = checkpoint['model_config']
    norm_params = checkpoint.get('norm_params', {})

    traj_mean  = norm_params.get('traj_mean', None)
    traj_std   = norm_params.get('traj_std', None)
    stats_mean = norm_params.get('stats_mean', None)
    stats_std  = norm_params.get('stats_std', None)

    model = TransportationModeClassifier(
        trajectory_feature_dim=config.get('input_dim', 21),
        segment_stats_dim=config.get('segment_stats_dim', 18),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 2),
        num_classes=config.get('num_classes', len(label_encoder.classes_)),
        dropout=config.get('dropout', 0.3)
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   ✓ 模型加载完成，类别: {list(label_encoder.classes_)}")

    # 2. 加载测试数据
    print(f"\n[2/4] 加载测试数据...")
    if not os.path.exists(SHARED_TEST_PATH):
        raise FileNotFoundError(f"共享测试集不存在: {SHARED_TEST_PATH}\n请先运行 create_shared_test_set.py")

    with open(SHARED_TEST_PATH, 'rb') as f:
        shared_data = pickle.load(f)
    
    valid_indices = shared_data['valid_indices']
    test_indices = shared_data['test_indices']
    cleaned_data = shared_data['cleaned_data']
    # 使用模型checkpoint中的label_encoder，而不是共享测试集中的
    # shared_label_encoder = shared_data['label_encoder']
    
    # 从共享测试集提取21维特征
    from exp2.src.feature_extraction import FeatureExtractor
    from exp2.src.osm_feature_extractor import OsmSpatialExtractor
    
    spatial_extractor = OsmSpatialExtractor()
    feature_extractor = FeatureExtractor(spatial_extractor)
    
    test_data = []
    for idx in test_indices:
        cleaned_idx = valid_indices[idx]
        traj, stats, datetime_series, label = cleaned_data[cleaned_idx]
        
        try:
            # 提取21维融合特征（9轨迹 + 12空间）
            traj_21, spatial_features = feature_extractor.extract_features(traj)
            # 使用模型checkpoint中的label_encoder进行编码
            if label in label_encoder.classes_:
                label_encoded = label_encoder.transform([label])[0]
                # EXP4格式: (traj_21, stats_18, label_encoded)
                test_data.append((traj_21, stats, label_encoded))
        except Exception as e:
            print(f"  警告: 样本 {idx} 特征提取失败: {e}")
            continue
    
    print(f"   ✓ 测试样本: {len(test_data)}（共享测试集，random_state=42）")

    # 3. 推理
    print(f"\n[3/4] 推理...")
    dataset = TrajectoryDatasetExp4(
        test_data,
        traj_mean=traj_mean, traj_std=traj_std,
        stats_mean=stats_mean, stats_std=stats_std
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    class_names = label_encoder.classes_
    class_weights = checkpoint.get('class_weights', None)
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    criterion = LabelSmoothingFocalLoss(
        num_classes=len(class_names), gamma=2.0, smoothing=0.1, weight=class_weights
    )

    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for traj, stats, labels in tqdm(loader, desc="Evaluating"):
            traj   = traj.to(DEVICE)
            stats  = stats.to(DEVICE)
            labels = labels.to(DEVICE)

            # ✅ 修复：单编码器，不传 placeholder
            logits = model(traj, segment_stats=stats)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
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
    }).to_csv(os.path.join(OUTPUT_DIR, 'predictions_exp4.csv'), index=False, encoding='utf-8-sig')

    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Exp4 Confusion Matrix (LabelSmoothing + Focal Loss)', fontsize=14)
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