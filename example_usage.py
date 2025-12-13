"""
快速开始示例
演示如何使用交通方式识别系统
"""
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader, preprocess_trajectory_segments
from src.knowledge_graph import TransportationKnowledgeGraph
from src.feature_extraction import FeatureExtractor


def example_load_and_preprocess():
    """示例：加载和预处理数据"""
    print("=" * 50)
    print("示例1: 加载和预处理数据")
    print("=" * 50)
    
    # 数据路径
    geolife_root = "data/Geolife Trajectories 1.3"
    osm_path = "data/beijing_osm_full_enhanced_verified.geojson"
    
    # 加载GeoLife数据
    print("\n1. 加载GeoLife数据...")
    geolife_loader = GeoLifeDataLoader(geolife_root)
    users = geolife_loader.get_all_users()
    print(f"   找到 {len(users)} 个用户")
    
    if len(users) > 0:
        # 加载第一个用户的标签
        user_id = users[0]
        print(f"\n2. 加载用户 {user_id} 的标签...")
        labels = geolife_loader.load_labels(user_id)
        print(f"   找到 {len(labels)} 个标签段")
        
        if len(labels) > 0:
            # 加载一个轨迹文件
            trajectory_dir = os.path.join(geolife_root, f"Data/{user_id}/Trajectory")
            if os.path.exists(trajectory_dir):
                traj_files = [f for f in os.listdir(trajectory_dir) if f.endswith('.plt')]
                if traj_files:
                    traj_path = os.path.join(trajectory_dir, traj_files[0])
                    print(f"\n3. 加载轨迹文件: {traj_files[0]}")
                    trajectory = geolife_loader.load_trajectory(traj_path)
                    print(f"   轨迹点数: {len(trajectory)}")
                    print(f"   轨迹特征: {list(trajectory.columns)}")
                    
                    # 分割轨迹
                    segments = geolife_loader.segment_trajectory(trajectory, labels)
                    print(f"\n4. 分割后的轨迹段数: {len(segments)}")
                    if segments:
                        segment, label = segments[0]
                        print(f"   第一个段标签: {label}, 点数: {len(segment)}")


def example_build_knowledge_graph():
    """示例：构建知识图谱"""
    print("\n" + "=" * 50)
    print("示例2: 构建知识图谱")
    print("=" * 50)
    
    osm_path = "data/export.geojson"
    
    if not os.path.exists(osm_path):
        print(f"   OSM数据文件不存在: {osm_path}")
        print("   请先下载OSM数据")
        return
    
    print("\n1. 加载OSM数据...")
    osm_loader = OSMDataLoader(osm_path)
    osm_data = osm_loader.load_osm_data()
    
    print("\n2. 提取道路网络和POI...")
    road_network = osm_loader.extract_road_network(osm_data)
    pois = osm_loader.extract_pois(osm_data)
    
    print(f"   道路数量: {len(road_network)}")
    print(f"   POI数量: {len(pois)}")
    
    print("\n3. 构建知识图谱...")
    kg = TransportationKnowledgeGraph()
    kg.build_from_osm(road_network, pois)
    
    stats = kg.get_graph_statistics()
    print(f"   节点数: {stats['num_nodes']}")
    print(f"   边数: {stats['num_edges']}")
    print(f"   道路节点: {stats['road_nodes']}")
    print(f"   POI节点: {stats['poi_nodes']}")


def example_extract_features():
    """示例：提取特征"""
    print("\n" + "=" * 50)
    print("示例3: 提取特征")
    print("=" * 50)
    
    geolife_root = "data/Geolife Trajectories 1.3"
    osm_path = "data/export.geojson"
    
    # 加载数据
    print("\n1. 加载数据...")
    geolife_loader = GeoLifeDataLoader(geolife_root)
    users = geolife_loader.get_all_users()
    
    if len(users) == 0:
        print("   没有找到用户数据")
        return
    
    user_id = users[0]
    labels = geolife_loader.load_labels(user_id)
    
    if labels.empty:
        print("   没有找到标签数据")
        return
    
    # 加载轨迹
    trajectory_dir = os.path.join(geolife_root, f"Data/{user_id}/Trajectory")
    if not os.path.exists(trajectory_dir):
        print("   轨迹目录不存在")
        return
    
    traj_files = [f for f in os.listdir(trajectory_dir) if f.endswith('.plt')]
    if not traj_files:
        print("   没有找到轨迹文件")
        return
    
    traj_path = os.path.join(trajectory_dir, traj_files[0])
    trajectory = geolife_loader.load_trajectory(traj_path)
    segments = geolife_loader.segment_trajectory(trajectory, labels)
    
    if not segments:
        print("   没有找到匹配的轨迹段")
        return
    
    # 构建知识图谱
    print("\n2. 构建知识图谱...")
    if not os.path.exists(osm_path):
        print(f"   OSM数据文件不存在: {osm_path}")
        return
    
    osm_loader = OSMDataLoader(osm_path)
    osm_data = osm_loader.load_osm_data()
    road_network = osm_loader.extract_road_network(osm_data)
    pois = osm_loader.extract_pois(osm_data)
    
    kg = TransportationKnowledgeGraph()
    kg.build_from_osm(road_network, pois)
    
    # 提取特征
    print("\n3. 提取特征...")
    feature_extractor = FeatureExtractor(kg)
    segment, label = segments[0]
    
    trajectory_features, kg_features = feature_extractor.extract_features(segment)
    
    print(f"   轨迹特征形状: {trajectory_features.shape}")
    print(f"   知识图谱特征形状: {kg_features.shape}")
    print(f"   标签: {label}")
    print(f"   轨迹特征维度: {trajectory_features.shape[1]}")
    print(f"   知识图谱特征维度: {kg_features.shape[1]}")


def main():
    """运行所有示例"""
    print("\n" + "=" * 50)
    print("交通方式识别系统 - 快速开始示例")
    print("=" * 50)
    
    try:
        example_load_and_preprocess()
    except Exception as e:
        print(f"\n示例1出错: {e}")
    
    try:
        example_build_knowledge_graph()
    except Exception as e:
        print(f"\n示例2出错: {e}")
    
    try:
        example_extract_features()
    except Exception as e:
        print(f"\n示例3出错: {e}")
    
    print("\n" + "=" * 50)
    print("示例运行完成!")
    print("=" * 50)
    print("\n下一步:")
    print("1. 运行 python train.py 训练模型")
    print("2. 运行 python evaluate.py --model_path checkpoints/best_model.pth 评估模型")
    print("3. 运行 python predict.py --model_path checkpoints/best_model.pth --trajectory_path <轨迹文件> 进行预测")


if __name__ == '__main__':
    main()



