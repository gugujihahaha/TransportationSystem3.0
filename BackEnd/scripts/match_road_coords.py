import json
import math
from pathlib import Path

GEOJSON_PATH = Path("D:/TransportationSystem3.0/BackEnd/data/exp2.geojson")
REGION_JSON_PATH = Path("D:/TransportationSystem3.0/FrontEnd/public/region_data.json")
OUTPUT_PATH = Path("D:/TransportationSystem3.0/FrontEnd/public/road_coords.json")

def geojson_centroid(coords):
    """计算 LineString 的中心点（取中点）"""
    if not coords:
        return None
    # 取几何的中点（按距离，简化取索引中位数）
    mid_idx = len(coords) // 2
    lon, lat = coords[mid_idx]
    return (lon, lat)

def main():
    # 1. 读取 region_data.json，获取需要匹配的路段名称
    with open(REGION_JSON_PATH, 'r', encoding='utf-8') as f:
        region_data = json.load(f)

    road_names = set()
    for district in region_data.values():
        for road in district.get('topRoads', []):
            road_names.add(road['name'])
    print(f"需要匹配的路段: {road_names}")

    # 2. 读取 exp2.geojson，提取道路名称和中心点
    with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
        geojson = json.load(f)

    # 建立名称到坐标的映射
    name_to_coords = {}

    for feature in geojson.get('features', []):
        props = feature.get('properties', {})
        geom = feature.get('geometry', {})
        # 获取道路名称
        name = props.get('name') or props.get('NAME') or props.get('road_name')
        if not name:
            continue
        # 只处理 LineString 类型
        if geom.get('type') != 'LineString':
            continue
        coords = geom.get('coordinates', [])
        if not coords:
            continue
        center = geojson_centroid(coords)
        if center:
            # 多个 feature 可能同名，我们只取第一个遇到的
            if name not in name_to_coords:
                name_to_coords[name] = center
            # 可选：如果有多个，可以合并或取平均，这里简单忽略后续

    # 3. 匹配
    road_coords = []
    for road_name in road_names:
        # 尝试精确匹配
        if road_name in name_to_coords:
            lng, lat = name_to_coords[road_name]
            road_coords.append({"name": road_name, "lng": lng, "lat": lat})
            print(f"✅ 匹配: {road_name}")
        else:
            # 尝试模糊匹配（包含关系）
            matched = False
            for osm_name, (lng, lat) in name_to_coords.items():
                if road_name in osm_name or osm_name in road_name:
                    road_coords.append({"name": road_name, "lng": lng, "lat": lat})
                    print(f"⚠️ 模糊匹配: {road_name} -> {osm_name}")
                    matched = True
                    break
            if not matched:
                print(f"❌ 未找到: {road_name}")

    # 4. 保存
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(road_coords, f, ensure_ascii=False, indent=2)

    print(f"✅ 已生成 {len(road_coords)} 个路段坐标，保存至 {OUTPUT_PATH}")

if __name__ == "__main__":
    main()