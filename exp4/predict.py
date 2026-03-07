"""
预测脚本 (Exp4 - 标签平滑 + Focal Loss)

使用训练好的模型对单个样本进行预测
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

# ========================== 路径设置 ==========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
os.chdir(SCRIPT_DIR)
# ==============================================================

# exp2 专用模块（复用模型）
from exp2.src.model import TransportationModeClassifier

# 默认模型路径
DEFAULT_CHECKPOINT_PATH = os.path.join('checkpoints', 'exp4_model.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(traj_features, segment_stats, checkpoint_path=DEFAULT_CHECKPOINT_PATH, device=DEVICE):
    """
    对单个样本进行预测

    Args:
        traj_features: (T, 21) 轨迹特征（21维点级融合）
        segment_stats: (18,) 段级统计特征
        checkpoint_path: 模型检查点路径
        device: 设备

    Returns:
        predicted_class: 预测的类别名称
        confidence: 预测置信度
        all_probs: 所有类别的概率
    """
    # 加载模型
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get('model_config', {})
    label_encoder = checkpoint['label_encoder']
    norm_params = checkpoint['norm_params']

    # 创建模型
    model = TransportationModeClassifier(
        trajectory_feature_dim=model_config.get('input_dim', 21),
        segment_stats_dim=model_config.get('segment_stats_dim', 18),
        hidden_dim=model_config.get('hidden_dim', 128),
        num_layers=model_config.get('num_layers', 2),
        num_classes=model_config.get('num_classes', 6),
        dropout=model_config.get('dropout', 0.3)
    ).to(device)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 归一化
    traj_norm = (traj_features - norm_params['traj_mean']) / norm_params['traj_std']
    stats_norm = (segment_stats - norm_params['stats_mean']) / norm_params['stats_std']

    # 转换为张量
    traj_tensor = torch.FloatTensor(traj_norm).unsqueeze(0).to(device)  # (1, T, 21)
    stats_tensor = torch.FloatTensor(stats_norm).unsqueeze(0).to(device)  # (1, 18)

    # 预测
    with torch.no_grad():
        logits = model(traj_tensor, segment_stats=stats_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (num_classes,)

    # 获取预测结果
    pred_idx = np.argmax(probs)
    predicted_class = label_encoder.inverse_transform([pred_idx])[0]
    confidence = probs[pred_idx]

    return predicted_class, confidence, probs


if __name__ == "__main__":
    # 示例：加载一个测试样本进行预测
    import pickle

    # 加载测试数据
    test_data_path = os.path.join('cache', 'processed_features_exp4.pkl')
    if not os.path.exists(test_data_path):
        print(f"错误：找不到测试数据文件 {test_data_path}")
        print("请先运行 train.py 生成数据缓存")
        sys.exit(1)

    with open(test_data_path, 'rb') as f:
        all_data, label_encoder, _ = pickle.load(f)

    # 取第一个样本作为示例
    traj_features, segment_stats, label = all_data[0]

    # 预测
    predicted_class, confidence, probs = predict(traj_features, segment_stats)

    # 打印结果
    print("=" * 60)
    print("预测结果")
    print("=" * 60)
    print(f"真实标签: {label_encoder.inverse_transform([label])[0]}")
    print(f"预测标签: {predicted_class}")
    print(f"置信度:   {confidence:.4f}")
    print("\n各类别概率:")
    for cls, prob in zip(label_encoder.classes_, probs):
        print(f"  {cls:12s}: {prob:.4f}")
    print("=" * 60)
