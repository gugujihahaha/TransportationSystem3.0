"""
评估脚本 (Exp4 - 标签平滑 + Focal Loss)

使用训练好的模型在测试集上进行评估
"""
import os
import sys
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ========================== 路径设置 ==========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
os.chdir(SCRIPT_DIR)
# ==============================================================

from common import evaluate as common_evaluate

# exp2 专用模块（复用模型）
from exp2.src.model import TransportationModeClassifier
from exp2.src.feature_extraction import FeatureExtractor
from exp2.src.osm_feature_extractor import OsmSpatialExtractor

# ========================== 缓存配置 ==========================
CACHE_DIR = 'cache'
PROCESSED_FEATURE_CACHE = os.path.join(CACHE_DIR, 'processed_features_exp4.pkl')
EXP2_FEATURE_CACHE = os.path.join('..', 'exp2', 'cache', 'processed_features.pkl')
OUTPUT_DIR = 'evaluation_results'
# ==================================================================


def compute_feature_stats(segments):
    """
    计算特征的均值和标准差，用于归一化

    Args:
        segments: List of (traj_features, stats, label)

    Returns:
        mean: 均值
        std: 标准差
    """
    traj_list = []
    stats_list = []

    for item in segments:
        traj = item[0]    # (50, 21)
        stats = item[1]   # (18,)
        traj_list.append(traj)
        stats_list.append(stats)

    traj_all = np.vstack(traj_list)       # (N*50, 21)
    stats_all = np.vstack(stats_list)     # (N, 18)

    traj_mean = traj_all.mean(axis=0).astype(np.float32)
    traj_std  = traj_all.std(axis=0).astype(np.float32)
    traj_std  = np.where(traj_std < 1e-6, 1.0, traj_std)  # 防止除零

    stats_mean = stats_all.mean(axis=0).astype(np.float32)
    stats_std  = stats_all.std(axis=0).astype(np.float32)
    stats_std  = np.where(stats_std < 1e-6, 1.0, stats_std)

    return traj_mean, traj_std, stats_mean, stats_std


class TrajectoryDatasetExp4(Dataset):
    """轨迹数据集（Exp4 版本）"""

    def __init__(self, all_features,
                 traj_mean=None, traj_std=None,
                 stats_mean=None, stats_std=None):
        self.data = all_features
        self.traj_mean = traj_mean
        self.traj_std = traj_std
        self.stats_mean = stats_mean
        self.stats_std = stats_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj_21, stats, label = self.data[idx]

        # 归一化
        if self.traj_mean is not None:
            traj = (traj_21 - self.traj_mean) / self.traj_std
        else:
            traj = traj_21

        if self.stats_mean is not None:
            stats_norm = (stats - self.stats_mean) / self.stats_std
        else:
            stats_norm = stats

        # ✅ 关键：加placeholder，与exp2的Dataset格式完全一致
        placeholder = np.zeros((traj.shape[0], 1), dtype=np.float32)

        return (torch.FloatTensor(traj),
                torch.FloatTensor(placeholder),
                torch.FloatTensor(stats_norm),
                torch.LongTensor([label])[0])


def load_data():
    """
    加载共享测试集数据

    Returns:
        all_data: List of (traj_21, stats, label)
        label_encoder: LabelEncoder
    """
    print(f"✅ 加载共享测试集")
    
    SHARED_TEST_PATH = '../../data/processed/shared_test_indices.pkl'
    if not os.path.exists(SHARED_TEST_PATH):
        raise FileNotFoundError(
            f"找不到共享测试集: {SHARED_TEST_PATH}\n"
            "请先运行 create_shared_test_set.py 生成共享测试集。"
        )
    
    with open(SHARED_TEST_PATH, 'rb') as f:
        shared_data = pickle.load(f)
    
    valid_indices = shared_data['valid_indices']
    test_indices = shared_data['test_indices']
    cleaned_data = shared_data['cleaned_data']
    label_encoder = shared_data['label_encoder']
    
    print(f"   共享测试集加载完成: {len(test_indices)} 个样本")
    
    # 初始化特征提取器
    spatial_extractor = OsmSpatialExtractor()
    feature_extractor = FeatureExtractor(spatial_extractor)
    
    # 提取21维融合特征
    all_data = []
    for idx in tqdm(test_indices, desc="提取特征"):
        cleaned_idx = valid_indices[idx]
        traj, stats, datetime_series, label = cleaned_data[cleaned_idx]
        
        try:
            # 提取21维融合特征（9轨迹 + 12空间）
            traj_21, spatial_features = feature_extractor.extract_features(traj)
            label_encoded = label_encoder.transform([label])[0]
            
            # exp4 格式: (traj_21dim, stats_18dim, label_encoded)
            all_data.append((traj_21, stats, label_encoded))
        except Exception as e:
            print(f"  警告: 样本 {idx} 特征提取失败: {e}")
            continue
    
    print(f"   特征提取完成: {len(all_data)} 个样本")
    
    return all_data, label_encoder


def main():
    parser = argparse.ArgumentParser(description='Exp4: 模型评估')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/exp4_model.pth',
                       help='模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='设备')

    args = parser.parse_args()

    DEVICE = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("Exp4 模型评估")
    print(f"设备: {DEVICE}")
    print("=" * 80)

    # 加载数据
    all_data, label_encoder = load_data()

    print(f"✅ 测试集加载完成: {len(all_data)} 个样本")

    # 创建测试集 Dataset
    test_dataset = TrajectoryDatasetExp4(
        all_data,
        traj_mean=None, traj_std=None,  # 归一化参数将从checkpoint读取
        stats_mean=None, stats_std=None
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=0)

    print(f"  Test batches: {len(test_loader)}")

    # 加载模型
    print(f"\n========== 加载模型 ==========")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"找不到模型文件: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)

    # 检查 checkpoint 格式
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        print(f"✅ 检测到新格式 checkpoint")
        print(f"   模型配置: {model_config}")
    else:
        # 旧格式兼容
        model_config = {
            'input_dim': 21,
            'segment_stats_dim': 18,
            'hidden_dim': 128,
            'num_layers': 2,
            'num_classes': len(label_encoder.classes_),
            'dropout': 0.3,
        }
        print(f"⚠️ 检测到旧格式 checkpoint，使用默认配置")

    # 创建模型
    model = TransportationModeClassifier(
        trajectory_feature_dim=model_config.get('input_dim', 21),
        segment_stats_dim=model_config.get('segment_stats_dim', 18),
        hidden_dim=model_config.get('hidden_dim', 128),
        num_layers=model_config.get('num_layers', 2),
        num_classes=model_config.get('num_classes', len(label_encoder.classes_)),
        dropout=model_config.get('dropout', 0.3)
    ).to(DEVICE)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 读取归一化参数
    norm_params = checkpoint.get('norm_params', {})
    traj_mean = norm_params.get('traj_mean', None)
    traj_std = norm_params.get('traj_std', None)
    stats_mean = norm_params.get('stats_mean', None)
    stats_std = norm_params.get('stats_std', None)
    
    # 更新Dataset的归一化参数
    test_dataset.traj_mean = traj_mean
    test_dataset.traj_std = traj_std
    test_dataset.stats_mean = stats_mean
    test_dataset.stats_std = stats_std
    
    print(f"✅ 模型加载完成")
    print(f"   归一化参数: traj_mean={traj_mean is not None}, stats_mean={stats_mean is not None}")
    # 损失函数（标签平滑 + Focal Loss）
    from src.focal_loss import LabelSmoothingFocalLoss
    class_weights = checkpoint.get('class_weights', None)
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)

    criterion = LabelSmoothingFocalLoss(
        num_classes=len(label_encoder.classes_),
        gamma=2.0,
        smoothing=0.1,
        weight=class_weights
    )

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 执行推理
    print(f"\n========== 测试集评估 ==========")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluation Progress", leave=True):
            traj, placeholder, stats, labels = batch
            traj = traj.to(DEVICE)
            placeholder = placeholder.to(DEVICE)
            stats = stats.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(traj, placeholder, segment_stats=stats)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 转换为 Numpy 数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 生成评估报告
    print(f"\n========== 评估报告 ==========")
    class_names = label_encoder.classes_

    # 打印分类报告
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    report_text = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0,
        digits=4
    )
    print(report_text)

    # 保存 JSON 报告
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    json_path = os.path.join(OUTPUT_DIR, 'evaluation_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)
    print(f"   ✓ 保存: {json_path}")

    # 保存 CSV 预测结果
    conf_list = [float(all_probs[i, p]) for i, p in enumerate(all_preds)]
    csv_path = os.path.join(OUTPUT_DIR, 'predictions_exp4.csv')
    pd.DataFrame({
        'true_label': [class_names[i] for i in all_labels],
        'pred_label': [class_names[i] for i in all_preds],
        'confidence': conf_list,
        'correct': all_labels == all_preds
    }).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"   ✓ 保存: {csv_path}")

    # 保存混淆矩阵图
    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names
        )
        plt.title('Exp4 Confusion Matrix (Label Smoothing + Focal Loss)', fontsize=14)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"   ✓ 保存: {cm_path}")
    except Exception as e:
        print(f"   ⚠️ 混淆矩阵图生成失败: {e}")

    # 保存各类别 F1-Score 图
    try:
        f1_scores = [report_dict[cls]['f1-score'] for cls in class_names]
        plt.figure(figsize=(12, 6))
        bars = sns.barplot(x=list(class_names), y=f1_scores, palette='viridis')
        plt.title('Exp4 F1-Score by Transportation Mode', fontsize=14)
        plt.xlabel('Transportation Mode')
        plt.ylabel('F1-Score')
        plt.ylim(0, 1.0)
        for i, v in enumerate(f1_scores):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        f1_path = os.path.join(OUTPUT_DIR, 'per_class_f1_scores.png')
        plt.savefig(f1_path, dpi=300)
        plt.close()
        print(f"   ✓ 保存: {f1_path}")
    except Exception as e:
        print(f"   ⚠️ F1-Score 图生成失败: {e}")

    # 保存错误分析
    try:
        errors_df = pd.DataFrame({
            'true_label': [class_names[i] for i in all_labels],
            'pred_label': [class_names[i] for i in all_preds],
            'confidence': conf_list
        })
        errors_df = errors_df[errors_df['true_label'] != errors_df['pred_label']]
        errors_path = os.path.join(OUTPUT_DIR, 'error_analysis.csv')
        errors_df.to_csv(errors_path, index=False, encoding='utf-8-sig')
        print(f"   ✓ 保存: {errors_path} ({len(errors_df)} 个错误样本)")
    except Exception as e:
        print(f"   ⚠️ 错误分析保存失败: {e}")

    # 汇总统计
    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)
    print(f"总样本数: {len(all_labels)}")
    print(f"正确预测: {(all_labels == all_preds).sum()}")
    print(f"错误预测: {(all_labels != all_preds).sum()}")
    print(f"准确率: {report_dict['accuracy']:.4f}")
    print(f"加权 F1: {report_dict['weighted avg']['f1-score']:.4f}")
    print(f"宏平均 F1: {report_dict['macro avg']['f1-score']:.4f}")

    print(f"\n✅ 所有评估结果已保存至: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
