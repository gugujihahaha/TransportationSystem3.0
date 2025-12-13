"""
预测脚本
用于对新轨迹进行交通方式识别
"""
import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader
from src.knowledge_graph import TransportationKnowledgeGraph
from src.feature_extraction import FeatureExtractor
from src.model import TransportationModeClassifier
from sklearn.preprocessing import LabelEncoder


def predict_trajectory(model, trajectory, feature_extractor, device, label_encoder):
    """预测单个轨迹的交通方式"""
    # 提取特征
    trajectory_features, kg_features = feature_extractor.extract_features(trajectory)
    
    # 转换为tensor
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
    
    return pred_label, pred_prob, probs[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='预测轨迹的交通方式')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--trajectory_path', type=str, required=True,
                       help='轨迹文件路径（.plt格式）')
    parser.add_argument('--osm_path', type=str, 
                       default='data/export.geojson',
                       help='OSM数据路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    
    args = parser.parse_args()
    
    # 加载模型
    print("加载模型...")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    label_encoder = checkpoint['label_encoder']
    
    # 加载OSM数据并构建知识图谱
    print("加载OSM数据...")
    osm_loader = OSMDataLoader(args.osm_path)
    osm_data = osm_loader.load_osm_data()
    road_network = osm_loader.extract_road_network(osm_data)
    pois = osm_loader.extract_pois(osm_data)
    
    kg = TransportationKnowledgeGraph()
    kg.build_from_osm(road_network, pois)
    
    # 创建特征提取器
    feature_extractor = FeatureExtractor(kg)
    
    # 加载轨迹
    print("加载轨迹...")
    geolife_loader = GeoLifeDataLoader('')
    trajectory = geolife_loader.load_trajectory(args.trajectory_path)
    
    # 创建模型
    num_classes = len(label_encoder.classes_)
    model = TransportationModeClassifier(
        trajectory_feature_dim=7,
        kg_feature_dim=11,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3
    ).to(args.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 预测
    print("预测中...")
    pred_label, pred_prob, all_probs = predict_trajectory(
        model, trajectory, feature_extractor, args.device, label_encoder
    )
    
    print(f"\n预测结果:")
    print(f"交通方式: {pred_label}")
    print(f"置信度: {pred_prob:.4f}")
    print(f"\n所有类别的概率:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  {class_name}: {all_probs[i]:.4f}")


if __name__ == '__main__':
    main()

# python predict.py --model_path checkpoints/best_model.pth --trajectory_path <轨迹文件路径>