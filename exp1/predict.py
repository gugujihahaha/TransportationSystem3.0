"""
预测脚本 - 仅使用GeoLife轨迹数据
"""
import os
import argparse
import torch
import numpy as np

from src.data_loader import GeoLifeDataLoader, preprocess_segments
from src.model import TransportationModeClassifier
from sklearn.preprocessing import LabelEncoder


def predict_trajectory(model, trajectory, device, label_encoder, target_length=100):
    """预测单个轨迹的交通方式"""
    # 预处理轨迹
    segment = (trajectory, 'unknown')  # 临时标签，不会被使用
    processed = preprocess_segments([segment], min_length=10, target_length=target_length)
    
    if len(processed) == 0:
        return None, None, None
    
    features, _ = processed[0]
    
    # 转换为tensor
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        logits = model(features_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0][pred_idx].item()
    
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    all_probs = probs[0].cpu().numpy()
    
    return pred_label, pred_prob, all_probs


def main():
    parser = argparse.ArgumentParser(description='预测轨迹的交通方式（仅轨迹特征）')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--trajectory_path', type=str, required=True,
                       help='轨迹文件路径（.plt格式）')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    
    args = parser.parse_args()
    
    # 加载模型
    print("=" * 60)
    print("加载模型...")
    print("=" * 60)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    label_encoder = checkpoint['label_encoder']
    model_config = checkpoint.get('model_config', {})
    
    # 创建模型
    num_classes = len(label_encoder.classes_)
    model = TransportationModeClassifier(
        input_dim=model_config.get('input_dim', 9),
        hidden_dim=model_config.get('hidden_dim', 128),
        num_layers=model_config.get('num_layers', 2),
        num_classes=num_classes,
        dropout=model_config.get('dropout', 0.3)
    ).to(args.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载轨迹
    print("\n加载轨迹...")
    geolife_loader = GeoLifeDataLoader('')
    trajectory = geolife_loader.load_trajectory(args.trajectory_path)
    print(f"轨迹点数: {len(trajectory)}")
    
    # 预测
    print("\n预测中...")
    pred_label, pred_prob, all_probs = predict_trajectory(
        model, trajectory, args.device, label_encoder
    )
    
    if pred_label is None:
        print("错误: 轨迹太短，无法预测")
        return
    
    print("\n" + "=" * 60)
    print("预测结果")
    print("=" * 60)
    print(f"交通方式: {pred_label}")
    print(f"置信度: {pred_prob:.4f} ({pred_prob*100:.2f}%)")
    print(f"\n所有类别的概率:")
    for i, class_name in enumerate(label_encoder.classes_):
        prob = all_probs[i]
        bar = '█' * int(prob * 50)
        print(f"  {class_name:10s}: {prob:.4f} ({prob*100:5.2f}%) {bar}")


if __name__ == '__main__':
    main()

