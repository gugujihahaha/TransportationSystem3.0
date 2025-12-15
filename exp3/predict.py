"""
预测脚本 (Exp3)
用于对新轨迹进行交通方式识别
"""
import os
import argparse
import torch
import numpy as np

from src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader, preprocess_trajectory_segments
from src.knowledge_graph_enhanced import EnhancedTransportationKG
from src.feature_extraction import FeatureExtractor
from src.model import TransportationModeClassifier


def predict_trajectory(model, trajectory_features, kg_features, device, label_encoder):
    """
    预测单个轨迹的交通方式

    Args:
        model: 训练好的模型
        trajectory_features: (N, 9) 轨迹特征
        kg_features: (N, 15) KG特征
        device: 设备
        label_encoder: 标签编码器

    Returns:
        pred_label: 预测的交通方式
        pred_prob: 预测概率
        all_probs: 所有类别的概率
    """
    # 转换为 tensor
    trajectory_tensor = torch.FloatTensor(trajectory_features).unsqueeze(0).to(device)
    kg_tensor = torch.FloatTensor(kg_features).unsqueeze(0).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        logits = model(trajectory_tensor, kg_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0][pred_idx].item()

    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    all_probs = probs[0].cpu().numpy()

    return pred_label, pred_prob, all_probs


def main():
    parser = argparse.ArgumentParser(description='预测轨迹的交通方式 (Exp3)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--trajectory_path', type=str, required=True,
                        help='轨迹文件路径（.plt格式）')
    parser.add_argument('--osm_path', type=str,
                        default='../data/beijing_complete.geojson',
                        help='OSM数据路径')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')

    args = parser.parse_args()

    # 加载模型
    print("\n" + "=" * 60)
    print("加载模型")
    print("=" * 60)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    label_encoder = checkpoint['label_encoder']
    model_config = checkpoint.get('model_config', {})

    print(f"模型配置:")
    print(f"  - 轨迹特征维度: {model_config.get('trajectory_feature_dim', 9)}")
    print(f"  - KG特征维度: {model_config.get('kg_feature_dim', 15)}")
    print(f"  - 分类类别: {label_encoder.classes_}")

    # 加载 OSM 数据并构建知识图谱
    print("\n" + "=" * 60)
    print("加载 OSM 数据并构建知识图谱")
    print("=" * 60)
    osm_loader = OSMDataLoader(args.osm_path)
    osm_data = osm_loader.load_osm_data()
    road_network = osm_loader.extract_road_network(osm_data)
    pois = osm_loader.extract_pois(osm_data)
    transit_routes = osm_loader.extract_transit_routes(osm_data)

    kg = EnhancedTransportationKG()
    kg.build_from_osm(road_network, pois, transit_routes)

    stats = kg.get_graph_statistics()
    print(f"知识图谱统计:")
    print(f"  - 节点数: {stats['num_nodes']}")
    print(f"  - 边数: {stats['num_edges']}")
    print(f"  - 道路节点: {stats['road_nodes']}")
    print(f"  - POI节点: {stats['poi_nodes']}")
    print(f"  - 速度限制记录: {stats['speed_limits']}")
    print(f"  - 公交线路: {stats['bus_routes']}")
    print(f"  - 地铁线路: {stats['subway_routes']}")

    # 创建特征提取器
    feature_extractor = FeatureExtractor(kg)

    # 加载轨迹
    print("\n" + "=" * 60)
    print("加载轨迹")
    print("=" * 60)
    geolife_loader = GeoLifeDataLoader('')
    trajectory = geolife_loader.load_trajectory(args.trajectory_path)

    if trajectory.empty:
        print("错误: 轨迹文件为空或无效")
        return

    print(f"轨迹点数: {len(trajectory)}")
    print(f"时间范围: {trajectory['datetime'].min()} 到 {trajectory['datetime'].max()}")
    print(f"总距离: {trajectory['total_distance'].max():.2f} 米")
    print(f"总时长: {trajectory['total_time'].max():.2f} 秒")

    # 预处理轨迹
    print("\n预处理轨迹...")
    segment = (trajectory, 'unknown')  # 临时标签
    processed = preprocess_trajectory_segments([segment], min_length=10)

    if len(processed) == 0:
        print("错误: 轨迹太短，无法预测")
        return

    trajectory_array, _ = processed[0]
    print(f"预处理后序列长度: {trajectory_array.shape[0]}")

    # 提取特征
    print("\n" + "=" * 60)
    print("提取特征")
    print("=" * 60)
    print("正在提取轨迹特征和增强KG特征...")
    trajectory_features, kg_features = feature_extractor.extract_features(trajectory_array)

    print(f"轨迹特征维度: {trajectory_features.shape}")
    print(f"KG特征维度: {kg_features.shape}")

    # 创建模型
    print("\n创建模型...")
    num_classes = len(label_encoder.classes_)
    model = TransportationModeClassifier(
        trajectory_feature_dim=model_config.get('trajectory_feature_dim', 9),
        kg_feature_dim=model_config.get('kg_feature_dim', 15),
        hidden_dim=model_config.get('hidden_dim', 128),
        num_layers=model_config.get('num_layers', 2),
        num_classes=num_classes,
        dropout=model_config.get('dropout', 0.3)
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])

    # 预测
    print("\n" + "=" * 60)
    print("预测中...")
    print("=" * 60)
    pred_label, pred_prob, all_probs = predict_trajectory(
        model, trajectory_features, kg_features, args.device, label_encoder
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("预测结果")
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

        # 绘制概率条
        bar_length = int(prob * 50)
        bar = '█' * bar_length

        # 添加排名标记
        if rank == 1:
            marker = '🥇'
        elif rank == 2:
            marker = '🥈'
        elif rank == 3:
            marker = '🥉'
        else:
            marker = f'{rank}.'

        print(f"{marker} {class_name:10s}: {prob:.4f} ({prob * 100:5.2f}%) {bar}")

    print("=" * 60)

    # 输出轨迹统计信息
    print("\n轨迹统计:")
    print(f"  - 平均速度: {trajectory['speed'].mean():.2f} m/s ({trajectory['speed'].mean() * 3.6:.2f} km/h)")
    print(f"  - 最大速度: {trajectory['speed'].max():.2f} m/s ({trajectory['speed'].max() * 3.6:.2f} km/h)")
    print(f"  - 平均加速度: {trajectory['acceleration'].mean():.4f} m/s²")
    print(f"  - 平均方向变化: {trajectory['bearing_change'].mean():.2f} 度")

    # 根据预测结果给出解释
    print("\n预测依据分析:")
    speed_kmh = trajectory['speed'].mean() * 3.6

    if pred_label == 'walk':
        print("  ✓ 速度较低，符合步行特征")
    elif pred_label == 'bike':
        print("  ✓ 速度适中，符合骑行特征")
    elif pred_label in ['car', 'taxi']:
        print("  ✓ 速度较高，符合汽车特征")
    elif pred_label == 'bus':
        print("  ✓ 速度适中且可能经过公交站点")
    elif pred_label == 'train':
        print("  ✓ 速度较快且可能沿铁路线路运行")

    # 提示可能的误判
    if pred_prob < 0.5:
        print("\n⚠️  注意: 预测置信度较低，建议人工核查")
    elif pred_prob < 0.7:
        print("\n⚠️  注意: 预测置信度中等，存在一定不确定性")


if __name__ == '__main__':
    main()