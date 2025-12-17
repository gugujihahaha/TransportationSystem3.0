# ============================================================
# exp1/predict.py - 预测脚本
# ============================================================
"""
预测脚本 - 仅使用GeoLife轨迹数据 (Exp1)

使用方法:
    python predict.py \\
        --model_path checkpoints/exp1_model.pth \\
        --trajectory_path "../data/Geolife Trajectories 1.3/Data/080/Trajectory/20070627094922.plt"
"""
import os
import argparse
import torch
import numpy as np

from src.data_loader import GeoLifeDataLoader, preprocess_segments
from src.model import TransportationModeClassifier


def predict_trajectory(model, trajectory, device, label_encoder, target_length=100):
    """
    预测单个轨迹的交通方式

    Args:
        model: 训练好的模型
        trajectory: DataFrame 格式的轨迹数据
        device: 计算设备
        label_encoder: 标签编码器
        target_length: 目标序列长度

    Returns:
        pred_label: 预测的交通方式
        pred_prob: 预测置信度
        all_probs: 所有类别的概率分布
    """
    # 预处理轨迹
    # data_loader.py 中的 preprocess_segments 会在这里运行
    segment = (trajectory, 'unknown')  # 临时标签。'unknown' 标签在 preprocess_segments 中会被保留。
    processed = preprocess_segments([segment], min_length=10, target_length=target_length)

    if len(processed) == 0:
        return None, None, None

    features, _ = processed[0]

    # 全局归一化（预测时使用训练集统计量）
    # 通常在这里需要加载训练时的均值和标准差对 features 进行归一化
    # 假设均值和标准差也保存在 model_path 的 checkpoint 中 (但这里没有写加载逻辑，因此暂时省略)

    # 转换为 Tensor
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)  # [1, seq_len, input_dim]

    # 预测
    model.eval()
    with torch.no_grad():
        logits = model(features_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    # 获取预测结果
    pred_index = np.argmax(probs)
    pred_prob = probs[pred_index]
    pred_label = label_encoder.classes_[pred_index]

    return pred_label, pred_prob, probs


def main():
    parser = argparse.ArgumentParser(description='使用已训练模型预测单个轨迹')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints/exp1_model.pth',
                        help='已训练模型的路径')
    parser.add_argument('--trajectory_path', type=str,
                        required=True,
                        help='待预测的 GeoLife .plt 轨迹文件路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')

    args = parser.parse_args()

    # 1. 加载模型和 LabelEncoder
    print("=" * 60)
    print("加载模型和LabelEncoder...")
    if not os.path.exists(args.model_path):
        print(f"错误: 未找到模型文件 {args.model_path}")
        return

    checkpoint = torch.load(args.model_path, map_location=args.device)
    label_encoder = checkpoint['label_encoder']
    config = checkpoint['model_config']

    # 实例化模型
    model = TransportationModeClassifier(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.3)
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 模型加载成功")
    print("=" * 60)

    # 2. 加载和预处理轨迹
    print(f"\n加载轨迹: {args.trajectory_path}")
    data_loader = GeoLifeDataLoader(data_root=os.path.dirname(os.path.dirname(args.trajectory_path)))

    # 尝试加载轨迹
    try:
        trajectory = data_loader.load_trajectory(args.trajectory_path)
    except Exception as e:
        print(f"错误: 加载轨迹失败 {e}")
        return

    if trajectory.empty:
        print("错误: 轨迹文件为空或无效")
        return

    # 3. 预测
    print("\n预测中...")
    pred_label, pred_prob, all_probs = predict_trajectory(
        model, trajectory, args.device, label_encoder
    )

    if pred_label is None:
        print("错误: 轨迹太短，无法预测")
        return

    # ========== 输出结果 ==========
    print("\n" + "=" * 60)
    print("预测结果")
    print("=" * 60)
    # pred_label 可能为 'car & taxi'
    print(f"\n交通方式: {pred_label}")
    print(f"置信度: {pred_prob:.4f} ({pred_prob * 100:.2f}%)")

    print(f"\n所有类别的概率:")
    print("-" * 60)

    # 按概率排序
    sorted_indices = np.argsort(all_probs)[::-1]

    for rank, i in enumerate(sorted_indices, 1):
        class_name = label_encoder.classes_[i]
        prob = all_probs[i]
        bar = '█' * int(prob * 50)

        if rank == 1:
            marker = '🥇'
        elif rank == 2:
            marker = '🥈'
        elif rank == 3:
            marker = '🥉'
        else:
            marker = f'{rank}.'

        print(f"{marker} {class_name:10s}: {prob:.4f} ({prob * 100:.2f}%) {bar}")

    print("-" * 60)


if __name__ == '__main__':
    main()