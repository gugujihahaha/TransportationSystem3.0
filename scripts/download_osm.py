import requests
import json
import os
import time

def download_osm_data(output_path: str):
    """
    通过 Overpass API 直接下载北京OSM数据
    自动处理超时重试和断点续传
    """
    
    query = """
[out:json][timeout:180];
area[name="北京市"]->.a;
(
  node(area.a)[highway=bus_stop];
  node(area.a)[railway=station];
  node(area.a)[railway=subway_entrance];
  node(area.a)[public_transport=station];
  node(area.a)[amenity=parking];
  node(area.a)[amenity=taxi];
  way(area.a)[highway=footway];
  way(area.a)[highway=cycleway];
  way(area.a)[highway=primary];
  way(area.a)[highway=secondary];
  way(area.a)[highway=tertiary];
  way(area.a)[highway=residential];
  way(area.a)[railway=rail];
  way(area.a)[railway=subway];
);
out body;
>;
out skel qt;
"""

    # 多个备用服务器，哪个快用哪个
    servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]

    for i, server in enumerate(servers):
        print(f"\n尝试服务器 {i+1}/{len(servers)}: {server}")
        try:
            print("正在发送请求...")
            response = requests.post(
                server,
                data={"data": query},
                timeout=300,  # 5分钟超时
                stream=True   # 流式下载，避免内存爆炸
            )
            response.raise_for_status()

            # 流式写入文件
            total_size = 0
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            temp_path = output_path + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
                        print(f"\r已下载: {total_size / 1024 / 1024:.1f} MB", end="", flush=True)

            print(f"\n✅ 下载完成，总大小: {total_size / 1024 / 1024:.1f} MB")

            # 验证是否是合法JSON
            print("正在验证数据格式...")
            with open(temp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            element_count = len(data.get('elements', []))
            print(f"✅ 数据验证通过，共 {element_count} 个元素")

            # 转换为 GeoJSON 格式
            print("正在转换为 GeoJSON 格式...")
            geojson = convert_to_geojson(data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, ensure_ascii=False)
            
            os.remove(temp_path)
            file_size = os.path.getsize(output_path) / 1024 / 1024
            print(f"✅ GeoJSON 已保存: {output_path} ({file_size:.1f} MB)")
            return True

        except requests.exceptions.Timeout:
            print(f"\n⚠️ 服务器 {i+1} 超时，尝试下一个...")
            continue
        except requests.exceptions.RequestException as e:
            print(f"\n⚠️ 服务器 {i+1} 请求失败: {e}，尝试下一个...")
            continue
        except json.JSONDecodeError as e:
            print(f"\n⚠️ 数据格式错误: {e}，尝试下一个...")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            continue
        except Exception as e:
            print(f"\n⚠️ 未知错误: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            continue

    print("❌ 所有服务器均失败")
    return False


def convert_to_geojson(osm_data: dict) -> dict:
    """
    将 Overpass JSON 格式转换为 GeoJSON 格式
    供 exp2/src/data_preprocessing.py 的 OSMDataLoader 使用
    """
    features = []
    
    # 建立节点坐标索引（way 需要引用 node 坐标）
    node_coords = {}
    for element in osm_data.get('elements', []):
        if element['type'] == 'node':
            node_coords[element['id']] = (element.get('lon', 0), element.get('lat', 0))
    
    for element in osm_data.get('elements', []):
        props = element.get('tags', {})
        props['@id'] = str(element['id'])
        
        if element['type'] == 'node':
            lon = element.get('lon', 0)
            lat = element.get('lat', 0)
            feature = {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                }
            }
            features.append(feature)
            
        elif element['type'] == 'way':
            coords = []
            for node_id in element.get('nodes', []):
                if node_id in node_coords:
                    coords.append(list(node_coords[node_id]))
            
            if len(coords) >= 2:
                feature = {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords
                    }
                }
                features.append(feature)
    
    print(f"  转换完成：{len(features)} 个 GeoJSON Feature")
    
    return {
        "type": "FeatureCollection",
        "features": features
    }


if __name__ == "__main__":
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data/exp2.geojson"
    )
    
    print("=" * 60)
    print("北京 OSM 数据下载工具")
    print("=" * 60)
    print(f"输出路径: {output_path}")
    
    success = download_osm_data(output_path)
    
    if success:
        print("\n✅ 全部完成！可以运行 exp2/train.py 了")
    else:
        print("\n❌ 下载失败，请检查网络连接后重试")
        print("备选方案：使用 VPN 后重试，或手动从 `https://overpass-turbo.eu` 下载")
