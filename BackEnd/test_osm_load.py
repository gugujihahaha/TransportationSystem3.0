import sys
import pickle
from pathlib import Path

exp2_path = str(Path(__file__).parent / "exp2")
if exp2_path not in sys.path:
    sys.path.insert(0, exp2_path)

spatial_cache_path = Path(__file__).parent / "exp2" / "cache" / "spatial_data.pkl"
print(f"正在加载: {spatial_cache_path}")

with open(spatial_cache_path, 'rb') as f:
    osm_extractor = pickle.load(f)

print(f"✅ 加载成功！")
print(f"  - road_kdtree: {osm_extractor.road_kdtree is not None}")
print(f"  - poi_kdtree: {osm_extractor.poi_kdtree is not None}")
print(f"  - road_coords: {len(osm_extractor.road_coords) if osm_extractor.road_coords is not None else 0}")
print(f"  - poi_coords: {len(osm_extractor.poi_coords) if osm_extractor.poi_coords is not None else 0}")

grid_cache_path = Path(__file__).parent / "exp2" / "cache" / "spatial_grid_cache.pkl"
if grid_cache_path.exists():
    osm_extractor.load_cache(str(grid_cache_path))
    print(f"✅ 网格缓存加载成功！")

print("测试完成！")
