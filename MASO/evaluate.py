"""
MASO-MSF 评估脚本
复现论文: 《基于GPS轨迹多尺度表达的交通出行方式识别方法》- 马妍莉
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# 将 src 目录加入路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import GeoLifeDataLoader, preprocess_segments
from src.model import MSFModel
from train import MASODataset

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "../checkpoints/maso_msf_best.pth"

    if not os.path.exists(checkpoint_path):
        print(f"[!] 错误: 未找到模型权重文件 {checkpoint_path}，请先运行 train.py")
        return

    # 1. 加载模型和标签编码器
    print(f"[*] 正在加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    label_encoder = checkpoint['label_encoder']
    num_classes = len(label_encoder.classes_)

    model = MSFModel(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. 加载测试数据 (为了演示，这里重新加载部分数据作为评估)
    # 实际项目中，建议在 train.py 中保存 test_data 路径
    data_root = "../data/Geolife Trajectories 1.3"
    loader = GeoLifeDataLoader(data_root)
    # 选最后20个用户作为评估示例，加快速度
    all_users = loader.get_all_users()
    eval_users = all_users[-20:] if len(all_users) > 20 else all_users

    print("[*] 正在预处理评估数据...")
    raw_data = preprocess_segments(loader, eval_users)
    eval_loader = DataLoader(MASODataset(raw_data, label_encoder), batch_size=32, shuffle=False)

    # 3. 执行推理
    all_preds = []
    all_trues = []

    print("[*] 开始验证集推理...")
    with torch.no_grad():
        for x32, x64, y in eval_loader:
            x32, x64, y = x32.to(device), x64.to(device), y.to(device)
            outputs = model(x32, x64)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(y.cpu().numpy())

    # 4. 生成性能报告 (复现论文表 5-4)
    target_names = label_encoder.classes_
    print("\n" + "="*50)
    print("MASO-MSF 交通方式识别性能报告")
    print("="*50)
    print(classification_report(all_trues, all_preds, target_names=target_names))

    # 5. 绘制混淆矩阵 (复现论文图 5-5)
    cm = confusion_matrix(all_trues, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - MASO-MSF Model')

    save_path = "evaluation_result.png"
    plt.savefig(save_path)
    print(f"[*] 混淆矩阵已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_model()