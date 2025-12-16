"""
KG 特征提取速度诊断脚本
用于测试是否还有嵌套循环问题
"""
import numpy as np
import time
from src.knowledge_graph import TransportationKnowledgeGraph
from src.data_preprocessing import OSMDataLoader

def test_kg_speed():
    """测试 KG 特征提取速度"""

    print("=" * 60)
    print("KG 特征提取速度诊断")
    print("=" * 60)

    # 1. 构建 KG
    print("\n[1] 加载 OSM 数据...")
    osm_loader = OSMDataLoader("../data/exp2.geojson")
    osm_data = osm_loader.load_osm_data()
    roads = osm_loader.extract_road_network(osm_data)
    pois = osm_loader.extract_pois(osm_data)

    print("\n[2] 构建知识图谱...")
    kg = TransportationKnowledgeGraph()
    kg.build_from_osm(roads, pois)
    stats = kg.get_graph_statistics()
    print(f"    图统计: {stats}")

    # 2. 测试单个轨迹段（50 个点）
    print("\n[3] 测试单个轨迹段 (50 个点)...")
    trajectory_50 = np.random.randn(50, 9)
    trajectory_50[:, 0] = 39.9 + np.random.randn(50) * 0.01  # 北京纬度
    trajectory_50[:, 1] = 116.4 + np.random.randn(50) * 0.01  # 北京经度

    start_time = time.time()
    kg_features_50 = kg.extract_kg_features(trajectory_50)
    elapsed_50 = time.time() - start_time

    print(f"    输入: {trajectory_50.shape}")
    print(f"    输出: {kg_features_50.shape}")
    print(f"    耗时: {elapsed_50:.3f} 秒")
    print(f"    每点耗时: {elapsed_50 / 50 * 1000:.2f} 毫秒")

    if elapsed_50 > 5:
        print("    ⚠️  警告: 速度过慢！可能存在嵌套循环问题")
    else:
        print("    ✅ 速度正常")

    # 3. 测试相同位置（测试缓存）
    print("\n[4] 测试缓存效果（相同位置）...")
    start_time = time.time()
    kg_features_cached = kg.extract_kg_features(trajectory_50)
    elapsed_cached = time.time() - start_time

    print(f"    耗时: {elapsed_cached:.3f} 秒")
    print(f"    加速比: {elapsed_50 / elapsed_cached:.1f}x")

    cache_stats = kg.get_cache_stats()
    print(f"    缓存统计: {cache_stats}")

    # 4. 测试批量轨迹段
    print("\n[5] 测试批量处理 (10 个轨迹段)...")
    trajectories = []
    for _ in range(10):
        traj = np.random.randn(50, 9)
        traj[:, 0] = 39.9 + np.random.randn(50) * 0.01
        traj[:, 1] = 116.4 + np.random.randn(50) * 0.01
        trajectories.append(traj)

    start_time = time.time()
    for traj in trajectories:
        kg.extract_kg_features(traj)
    elapsed_batch = time.time() - start_time

    print(f"    总耗时: {elapsed_batch:.3f} 秒")
    print(f"    平均每个轨迹段: {elapsed_batch / 10:.3f} 秒")

    cache_stats = kg.get_cache_stats()
    print(f"    最终缓存统计: {cache_stats}")

    # 5. 预期性能判断
    print("\n" + "=" * 60)
    print("性能评估:")
    print("=" * 60)

    avg_time_per_segment = elapsed_batch / 10

    if avg_time_per_segment < 0.1:
        print("✅ 性能优秀: 平均每个轨迹段 < 0.1 秒")
    elif avg_time_per_segment < 1.0:
        print("✅ 性能良好: 平均每个轨迹段 < 1 秒")
    elif avg_time_per_segment < 10:
        print("⚠️  性能一般: 平均每个轨迹段 < 10 秒")
    else:
        print("❌ 性能异常: 平均每个轨迹段 > 10 秒")
        print("   可能存在嵌套循环问题，请检查代码！")

    # 6. 预估总训练时间
    num_segments = 8912
    estimated_time = num_segments * avg_time_per_segment

    print(f"\n预估处理 {num_segments} 个轨迹段的时间:")
    print(f"  - 无缓存: {estimated_time / 60:.1f} 分钟")
    print(f"  - 有缓存 (75% 命中): {estimated_time * 0.25 / 60:.1f} 分钟")
    print(f"  - 有缓存 (95% 命中): {estimated_time * 0.05 / 60:.1f} 分钟")

    return avg_time_per_segment < 1.0


if __name__ == "__main__":
    success = test_kg_speed()

    if success:
        print("\n✅ 诊断通过！可以开始训练")
    else:
        print("\n❌ 诊断失败！请检查代码中是否有嵌套循环")