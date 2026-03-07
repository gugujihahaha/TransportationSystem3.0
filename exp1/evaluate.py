"""
Exp1 评估脚本 (标准版 - 与 Exp4 一致)
功能：
1. 自动适配轨迹特征（9维）
2. 支持多路径缓存加载
3. 生成详细的分类报告、混淆矩阵和预测详情
4. 输出所有文件至 evaluation_results/ 子目录
"""
import os
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

# 导入模型和数据集
from src.model import TransportationModeClassifier
from train import TrajectoryDataset

# 设置中文字体 (防止图片乱码)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def main():
    # 配置参数
    MODEL_PATH = 'checkpoints/exp1_model.pth'
    CACHE_PATH = 'cache/exp1_processed_features.pkl'
    OUTPUT_DIR = 'evaluation_results'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 切换到exp1目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 备用缓存路径列表（按优先级）
    ALTERNATIVE_CACHE_PATHS = [
        'cache/processed_features.pkl',
        'cache/exp1_features.pkl',
    ]

    print("\n" + "=" * 60)
    print("Exp1 模型评估 (仅轨迹特征)")
    print("=" * 60)
    print(f"设备: {DEVICE}")

    # 1. 加载模型
    print(f"\n[1/5] 正在加载模型: {MODEL_PATH}")

    # 尝试多个模型路径
    model_paths_to_try = [
        MODEL_PATH,
        'checkpoints/exp1_model.pth',
    ]

    model_loaded = False
    for mp in model_paths_to_try:
        if os.path.exists(mp):
            MODEL_PATH = mp
            model_loaded = True
            break

    if not model_loaded:
        print(f"❌ 错误: 找不到模型文件，尝试过以下路径:")
        for mp in model_paths_to_try:
            print(f"   - {mp}")
        return

    print(f"   ✓ 找到模型: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    le = checkpoint['label_encoder']
    config = checkpoint['model_config']
    class_names = le.classes_

    print(f"   模型配置:")
    print(f"     - 输入维度: {config.get('trajectory_feature_dim', config.get('input_dim', 9))}")
    print(f"     - 隐藏层维度: {config['hidden_dim']}")
    print(f"     - 层数: {config['num_layers']}")
    print(f"     - 类别数: {config['num_classes']}")
    print(f"     - 类别: {list(class_names)}")

    model = TransportationModeClassifier(
        trajectory_feature_dim=9,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.3)
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("   ✓ 模型加载完成")

    norm_params = checkpoint.get('norm_params', {})
    traj_mean = norm_params.get('traj_mean', None)
    traj_std = norm_params.get('traj_std', None)
    stats_mean = norm_params.get('stats_mean', None)
    stats_std = norm_params.get('stats_std', None)

    # 检查归一化参数是否存在
    if traj_mean is None or traj_std is None:
        print("❌ 错误: checkpoint 中没有 norm_params，评估结果不可信！")
        print("   请使用新版本训练的 checkpoint（包含归一化参数）。")
        return

    # 2. 加载共享测试集
    print(f"\n[2/5] 正在加载共享测试集...")
    
    SHARED_TEST_PATH = '../../data/processed/shared_test_indices.pkl'
    if not os.path.exists(SHARED_TEST_PATH):
        print(f"❌ 找不到共享测试集: {SHARED_TEST_PATH}")
        print(f"   请先运行 create_shared_test_set.py 生成共享测试集")
        return
    
    with open(SHARED_TEST_PATH, 'rb') as f:
        shared_data = pickle.load(f)
    
    valid_indices = shared_data['valid_indices']
    test_indices = shared_data['test_indices']
    cleaned_data = shared_data['cleaned_data']
    shared_label_encoder = shared_data['label_encoder']
    
    # 提取测试集样本
    segments = []
    for idx in test_indices:
        cleaned_idx = valid_indices[idx]
        traj, stats, datetime_series, label = cleaned_data[cleaned_idx]
        # EXP1只需要前9维轨迹特征
        segments.append((traj, stats, label))
    
    print(f"   ✓ 加载完成: {len(segments)} 个测试样本 (使用共享测试集)")
    print(f"   共享测试集索引: {len(test_indices)} 个")

    # 3. 准备测试数据
    print(f"\n[3/5] 正在准备测试数据...")
    dataset = TrajectoryDataset(
        segments, shared_label_encoder,
        traj_mean=traj_mean, traj_std=traj_std,
        stats_mean=stats_mean, stats_std=stats_std
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0  # Windows 兼容
    )
    print(f"   ✓ 测试集大小: {len(segments)} 个样本")

    # 4. 执行推理
    print(f"\n[4/5] 正在进行模型推理...")
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for x, stats, y in tqdm(test_loader, desc="Evaluation Progress", leave=True):
            try:
                x = x.to(DEVICE)
                stats = stats.to(DEVICE)
                logits = model(x, segment_stats=stats)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                y_true.extend(y.squeeze().cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_probs.extend(probs.cpu().numpy())
            except Exception as e:
                print(f"   ⚠️ 推理异常: {e}")
                continue

    # 转换为 Numpy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # 5. 生成评估报告
    print(f"\n[5/5] 正在生成评估报告...")

    # 打印分类报告
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    report_text = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        digits=4
    )
    print(report_text)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 文件 1: JSON 报告
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    json_path = os.path.join(OUTPUT_DIR, 'evaluation_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)
    print(f"   ✓ 保存: {json_path}")

    # 文件 2: CSV 预测结果
    conf_list = [float(y_probs[i, p]) for i, p in enumerate(y_pred)]
    csv_path = os.path.join(OUTPUT_DIR, 'predictions_exp1.csv')
    pd.DataFrame({
        'true_label': [class_names[i] for i in y_true],
        'pred_label': [class_names[i] for i in y_pred],
        'confidence': conf_list,
        'correct': y_true == y_pred
    }).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"   ✓ 保存: {csv_path}")

    # 文件 3: 混淆矩阵图
    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names
        )
        plt.title('Exp1 Confusion Matrix (Trajectory Features Only)', fontsize=14)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"   ✓ 保存: {cm_path}")
    except Exception as e:
        print(f"   ⚠️ 混淆矩阵图生成失败: {e}")

    # 文件 4: 各类别 F1-Score 图
    try:
        f1_scores = [report_dict[cls]['f1-score'] for cls in class_names]
        plt.figure(figsize=(12, 6))
        bars = sns.barplot(x=list(class_names), y=f1_scores, palette='viridis')
        plt.title('Exp1 F1-Score by Transportation Mode', fontsize=14)
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

    # 文件 5: 错误分析
    try:
        errors_df = pd.DataFrame({
            'true_label': [class_names[i] for i in y_true],
            'pred_label': [class_names[i] for i in y_pred],
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
    print(f"总样本数: {len(y_true)}")
    print(f"正确预测: {(y_true == y_pred).sum()}")
    print(f"错误预测: {(y_true != y_pred).sum()}")
    print(f"准确率: {report_dict['accuracy']:.4f}")
    print(f"加权 F1: {report_dict['weighted avg']['f1-score']:.4f}")
    print(f"宏平均 F1: {report_dict['macro avg']['f1-score']:.4f}")

    print(f"\n✅ 所有评估结果已保存至: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()