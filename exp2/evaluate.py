"""
评估脚本
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader, preprocess_trajectory_segments
from src.knowledge_graph import TransportationKnowledgeGraph
from train import TrajectoryDataset


def load_data(geolife_root: str, osm_path: str, max_users: int = None):
    """加载所有数据"""
    print("加载GeoLife数据...")
    geolife_loader = GeoLifeDataLoader(geolife_root)
    
    # 获取用户列表
    users = geolife_loader.get_all_users()
    if max_users:
        users = users[:max_users]
    
    print(f"找到 {len(users)} 个用户")
    
    # 加载OSM数据并构建知识图谱
    print("加载OSM数据并构建知识图谱...")
    osm_loader = OSMDataLoader(osm_path)
    osm_data = osm_loader.load_osm_data()
    road_network = osm_loader.extract_road_network(osm_data)
    pois = osm_loader.extract_pois(osm_data)
    
    kg = TransportationKnowledgeGraph()
    kg.build_from_osm(road_network, pois)
    print(f"知识图谱构建完成: {kg.get_graph_statistics()}")
    
    # 加载所有轨迹段
    all_segments = []
    for user_id in tqdm(users, desc="加载轨迹数据"):
        labels = geolife_loader.load_labels(user_id)
        if labels.empty:
            continue
        
        # 加载该用户的所有轨迹文件
        trajectory_dir = os.path.join(geolife_root, f"Data/{user_id}/Trajectory")
        if not os.path.exists(trajectory_dir):
            continue
        
        for traj_file in os.listdir(trajectory_dir):
            if not traj_file.endswith('.plt'):
                continue
            
            traj_path = os.path.join(trajectory_dir, traj_file)
            try:
                trajectory = geolife_loader.load_trajectory(traj_path)
                segments = geolife_loader.segment_trajectory(trajectory, labels)
                all_segments.extend(segments)
            except Exception as e:
                print(f"加载轨迹文件失败 {traj_path}: {e}")
                continue
    
    print(f"总共加载 {len(all_segments)} 个轨迹段")
    
    # 预处理轨迹段
    print("预处理轨迹段...")
    processed_segments = preprocess_trajectory_segments(all_segments, min_length=10)
    
    return processed_segments, kg
from src.feature_extraction import FeatureExtractor
from src.model import TransportationModeClassifier


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='评估交通方式识别模型')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--geolife_root', type=str, 
                       default='../data/Geolife Trajectories 1.3',
                       help='GeoLife数据根目录')
    parser.add_argument('--osm_path', type=str, 
                       default='../data/exp2.geojson',
                       help='OSM数据路径')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print("加载模型...")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    label_encoder = checkpoint['label_encoder']
    
    # 加载数据
    print("加载数据...")
    segments, kg = load_data(args.geolife_root, args.osm_path)
    
    # 创建特征提取器
    feature_extractor = FeatureExtractor(kg)
    
    # 创建数据集
    dataset = TrajectoryDataset(segments, feature_extractor, label_encoder)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    num_classes = len(label_encoder.classes_)
    model = TransportationModeClassifier(
        trajectory_feature_dim=9,
        kg_feature_dim=11,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3
    ).to(args.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 评估
    print("评估模型...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for trajectory_features, kg_features, labels in dataloader:
            trajectory_features = trajectory_features.to(args.device)
            kg_features = kg_features.to(args.device)
            
            logits = model(trajectory_features, kg_features)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 分类报告
    class_names = label_encoder.classes_
    report = classification_report(all_labels, all_preds, 
                                  target_names=class_names,
                                  output_dict=True)
    
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, 
                              target_names=class_names))
    
    # 保存结果
    import json
    with open(os.path.join(args.output_dir, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, class_names,
                         os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    print(f"\n结果已保存到 {args.output_dir}")


if __name__ == '__main__':
    main()

# python evaluate.py --model_path checkpoints/exp2_model.pth