import json
from collections import Counter

OSM_PATH = "../data/exp3.geojson"

# OSM 真正的 tag 一般都是字符串 → 字符串
def is_valid_tag_value(v):
    return isinstance(v, (str, int, float))

def load_osm(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_osm(osm):
    features = osm.get("features", [])

    tag_counter = Counter()
    type_counter = Counter()
    maxspeed_count = 0

    for feat in features:
        if feat.get("type") != "Feature":
            continue

        props = feat.get("properties", {})
        if not isinstance(props, dict):
            continue

        # 从 @id 解析 OSM 类型
        osm_id = props.get("@id", "")
        if isinstance(osm_id, str) and "/" in osm_id:
            osm_type = osm_id.split("/")[0]
            if osm_type in ["node", "way", "relation"]:
                type_counter[osm_type] += 1

        # 统计 tag（只统计合法 tag）
        for k, v in props.items():
            # 跳过元字段
            if k.startswith("@"):
                continue
            if not is_valid_tag_value(v):
                continue

            tag_counter[(k, str(v))] += 1

            if k == "maxspeed":
                maxspeed_count += 1

    return {
        "type": type_counter,
        "tags": tag_counter,
        "maxspeed_count": maxspeed_count
    }

def print_key_findings(stats):
    print("\n========== 实验三 OSM 数据关键特征检查 ==========\n")

    def show(tag, value):
        count = stats["tags"].get((tag, value), 0)
        print(f"{tag}={value:<25} : {count}")

    print("【OSM 元素类型】")
    for k, v in stats["type"].items():
        print(f"{k:<10} : {v}")

    print("\n【公共交通站点】")
    show("highway", "bus_stop")
    show("railway", "station")
    show("public_transport", "station")
    show("railway", "subway_entrance")

    print("\n【交通辅助 POI】")
    show("amenity", "parking")
    show("amenity", "bicycle_rental")
    show("amenity", "taxi")

    print("\n【道路类型】")
    for hw in ["footway", "cycleway", "primary", "secondary", "tertiary", "residential"]:
        show("highway", hw)

    print("\n【速度信息】")
    print(f"带 maxspeed 的道路数量: {stats['maxspeed_count']}")

    print("\n【线路 Relation】")
    show("route", "bus")
    show("route", "subway")

    print("\n===============================================\n")

if __name__ == "__main__":
    osm = load_osm(OSM_PATH)
    stats = analyze_osm(osm)
    print_key_findings(stats)
