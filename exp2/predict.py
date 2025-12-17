# ============================================================
# exp2/predict.py - 预测脚本 (双输入：轨迹特征 + 知识图谱特征)
# ============================================================
"""
预测脚本 - 结合轨迹特征和知识图谱特征 (Exp2)

使用方法:
    python predict.py \\
        --model_path checkpoints/exp2_model.pth \\
        --kg_data_path "../data/kg_data" \\
        --trajectory_path "../data/Geolife Trajectories 1.3/Data/080/Trajectory/20070627094922.plt"
"""
import os
import argparse
import torch
import numpy as np
from typing import Tuple, Optional

# 假设模块路径
from src.model import TransportationModeClassifier
from src.knowledge_graph import TransportationKnowledgeGraph
from src.feature_extraction import FeatureExtractor
from src.data_preprocessing import GeoLifeDataLoader, preprocess_trajectory_segments


def predict_trajectory(model, trajectory, device, label_encoder, feature_extractor: FeatureExtractor):
    """
    预测单个轨迹的交通方式 (双输入)

    Args:
        model: 训练好的模型
        trajectory: DataFrame 格式的轨迹数据
        device: 计算设备
        label_encoder: 标签编码器
        feature_extractor: 初始化好的特征提取器 (包含 KG)

    Returns:
        pred_label: 预测的交通方式
        pred_prob: 预测置信度
        all_probs: 所有类别的概率分布
    """

    # 1. 预处理轨迹 (提取 9 维轨迹特征，长度规范化)
    # 临时标签 'unknown' 不影响特征提取
    segment = (trajectory, 'unknown')

    # preprocess_trajectory_segments 会将轨迹 DataFrame 转换为 (N, 9) NumPy 数组
    # 并进行长度规范化 (例如 (50, 9))
    processed = preprocess_trajectory_segments([segment], min_length=10)

    if len(processed) == 0:
        print("警告: 轨迹太短，无法预测。")
        return None, None, None

    traj_features_raw, _ = processed[0]

    # 2. 提取双特征 (包括轨迹归一化和 KG 特征提取)
    # feature_extractor.extract_features 返回 (归一化轨迹特征, KG 特征)
    normalized_traj_features, kg_features = feature_extractor.extract_features(traj_features_raw)

    # 3. 转换为 Tensor
    # 增加批次维度 (1, seq_len, dim)
    traj_tensor = torch.FloatTensor(normalized_traj_features).unsqueeze(0).to(device)
    kg_tensor = torch.FloatTensor(kg_features).unsqueeze(0).to(device)

    # 4. 预测
    model.eval()
    with torch.no_grad():
        # 🔥 双输入模型调用
        logits = model(traj_tensor, kg_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    # 5. 获取预测结果
    pred_index = np.argmax(probs)
    pred_prob = probs[pred_index]
    pred_label = label_encoder.classes_[pred_index]

    return pred_label, pred_prob, probs


def main():
    parser = argparse.ArgumentParser(description='使用已训练的 Exp2 模型预测单个轨迹')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints/exp2_model.pth',
                        help='已训练模型的路径')
    parser.add_argument('--kg_data_path', type=str,
                        default='../data/kg_data',
                        help='知识图谱数据目录 (包含 road_network.pkl/pois.pkl 等)')
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

    # 实例化模型 (使用 Exp2 的参数)
    model = TransportationModeClassifier(
        trajectory_feature_dim=config.get('trajectory_feature_dim', 9),
        kg_feature_dim=config.get('kg_feature_dim', 11),
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.3)
    ).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 模型加载成功")
    print("=" * 60)

    # 2. 初始化知识图谱和特征提取器
    print("初始化知识图谱...")
    kg = TransportationKnowledgeGraph()
    kg.load_data(data_root=args.kg_data_path)
    feature_extractor = FeatureExtractor(kg)
    print("✓ 知识图谱和特征提取器初始化完成")

    # 3. 加载轨迹
    print(f"\n加载轨迹: {args.trajectory_path}")
    data_loader = GeoLifeDataLoader(data_root='.')  # 根目录不重要，因为我们只加载单个文件

    try:
        trajectory = data_loader.load_trajectory(args.trajectory_path)
    except Exception as e:
        print(f"错误: 加载轨迹失败 {e}")
        return

    if trajectory.empty:
        print("错误: 轨迹文件为空或无效")
        return

    # 4. 预测
    print("\n预测中...")
    pred_label, pred_prob, all_probs = predict_trajectory(
        model, trajectory, args.device, label_encoder, feature_extractor
    )

    if pred_label is None:
        return

    # ========== 输出结果 ==========
    print("\n" + "=" * 60)
    print("预测结果 (Exp2: 轨迹 + KG)")
    print("=" * 60)
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