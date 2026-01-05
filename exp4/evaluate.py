import os
import json
import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 导入 Exp4 专用模型与 Dataset
from src.model_weather import TransportationModeClassifierWithWeather
from train import TrajectoryDatasetWithWeather

# 设置中文字体 (防止图片乱码)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    # --- 配置区 ---
    MODEL_PATH = 'checkpoints/exp4_model.pth'
    CACHE_PATH = 'cache/processed_features_weather_v1.pkl'
    OUTPUT_DIR = 'evaluation_results'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 加载模型
    print(f"正在加载模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    le = checkpoint['label_encoder']
    config = checkpoint['model_config']
    class_names = le.classes_

    model = TransportationModeClassifierWithWeather(
        trajectory_feature_dim=config['trajectory_feature_dim'],
        kg_feature_dim=config['kg_feature_dim'],
        weather_feature_dim=config['weather_feature_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.3)
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. 加载特征缓存
    if not os.path.exists(CACHE_PATH):
        alt_path = 'cache/processed_features_exp4.pkl'
        if os.path.exists(alt_path):
            CACHE_PATH = alt_path
        else:
            print(f"❌ 错误: 找不到特征缓存 {CACHE_PATH}")
            return

    print(f"✓ 发现特征缓存，正在快速加载: {CACHE_PATH}")
    with open(CACHE_PATH, 'rb') as f:
        all_features, _ = pickle.load(f)

    # 3. 准备测试数据
    dataset = TrajectoryDatasetWithWeather(all_features)
    labels_for_stratify = [feat[3] for feat in all_features]
    _, test_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42, stratify=labels_for_stratify
    )
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=64, shuffle=False)

    # 4. 执行推理
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for traj, kg, weather, labels in tqdm(test_loader, desc="正在进行模型推理", unit="it/s"):
            traj, kg, weather = traj.to(DEVICE), kg.to(DEVICE), weather.to(DEVICE)
            logits = model(traj, kg, weather)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    # 💡 核心修复：将结果转换为 Numpy 数组，方便矩阵索引和绘图
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # 5. 打印分类报告
    print("\n评估完成！分类报告如下:")
    report_text = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, digits=2)
    print(report_text)

    # 6. 生成并保存四个核心文件
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 文件 1: JSON 报告
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)

    # 文件 2: CSV 预测 (修复了此处的 confidence 获取逻辑)
    conf_list = [float(y_probs[i, p]) for i, p in enumerate(y_pred)]
    pd.DataFrame({
        'true_label': [class_names[i] for i in y_true],
        'pred_label': [class_names[i] for i in y_pred],
        'confidence': conf_list
    }).to_csv(os.path.join(OUTPUT_DIR, 'predictions_exp4.csv'), index=False, encoding='utf-8-sig')

    # 文件 3: 混淆矩阵图
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Exp4 混淆矩阵 (轨迹+KG+天气)', fontsize=14)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # 文件 4: 各类别指标图
    f1_scores = [report_dict[cls]['f1-score'] for cls in class_names]
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(class_names), y=f1_scores, palette='viridis')
    plt.title('Exp4 各交通方式 F1-Score 分布', fontsize=14)
    plt.ylim(0, 1.0)
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    plt.savefig(os.path.join(OUTPUT_DIR, 'per_class_metrics.png'), dpi=300)
    plt.close()

    print(f"\n✅ 所有四个结果文件已保存至: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()