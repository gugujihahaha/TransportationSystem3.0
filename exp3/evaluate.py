"""
评估脚本 (Exp3)
评估训练好的模型性能
"""
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm

from src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader, preprocess_trajectory_segments
from src.knowledge_graph import EnhancedTransportationKG
from src.feature_extraction import FeatureExtractor
from src.model import TransportationModeClassifier
from train import TrajectoryDataset


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵 - Exp3 (增强KG模型)', fontsize=14, fontweight='bold')
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ 混淆矩阵已保存: {save_path}")


def plot_per_class_metrics(report, class_names, save_path):
    """绘制每个类别的精确率、召回率和F1分数"""
    metrics = ['precision', 'recall', 'f1-score']
    values = {metric: [] for metric in metrics}

    for class_name in class_names:
        if class_name in report:
            for metric in metrics:
                values[metric].append(report[class_name][metric])

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, values[metric], width,
               label=metric.capitalize(), alpha=0.8)

    ax.set_xlabel('交通方式', fontsize=12)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('各类别性能指标 (Exp3)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ 性能指标图已保存: {save_path}")


def load_data(geolife_root: str, osm_path: str, max_users: int = None):
    """加载评估数据"""
    print("\n" + "=" * 60)
    print("加载评估数据")
    print("=" * 60)

    # 加载 GeoLife 数据
    geolife_loader = GeoLifeDataLoader(geolife_root)
    users = geolife_loader.get_all_users()
    if max_users:
        users = users[:max_users]

    print(f"找到 {len(users)} 个用户")

    # 加载 OSM 数据并构建知识图谱
    print("加载 OSM 数据并构建知识图谱...")
    osm_loader = OSMDataLoader(osm_path)
    osm_data = osm_loader.load_osm_data()
    road_network = osm_loader.extract_road_network(osm_data)
    pois = osm_loader.extract_pois(osm_data)
    transit_routes = osm_loader.extract_transit_routes(osm_data)

    kg = EnhancedTransportationKG()
    kg.build_from_osm(road_network, pois, transit_routes)
    print(f"知识图谱构建完成: {kg.get_graph_statistics()}")

    # 加载所有轨迹段
    print("\n加载轨迹数据...")
    all_segments = []
    for user_id in tqdm(users, desc="加载用户轨迹"):
        labels = geolife_loader.load_labels(user_id)
        if labels.empty:
            continue