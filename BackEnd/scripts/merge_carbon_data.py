import pickle
import json
import os
import numpy as np

pkl_path = r'D:\TransportationSystem3.0\BackEnd\data\processed\cleaned_balanced.pkl'
output_json = r'D:\TransportationSystem3.0\FrontEnd\public\heatmap_grid.json'

# 北京范围
lat_min, lat_max = 39.7, 40.2
lon_min, lon_max = 116.1, 116.7
grid_size = 0.005  # 约500米

# 加载数据
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

green_labels = {'Walk', 'Bike'}
grids = {}

for item in data:
    traj_9 = item[0]          # (50, 9)
    label = item[3]
    is_green = label in green_labels
    for point in traj_9:
        lat, lng = point[0], point[1]
        if not (lat_min <= lat <= lat_max and lon_min <= lng <= lon_max):
            continue
        gx = int((lat - lat_min) / grid_size)
        gy = int((lng - lon_min) / grid_size)
        key = f"{gx},{gy}"
        if key not in grids:
            grids[key] = {
                "count": 0,
                "green_count": 0,
                "lat": lat_min + (gx + 0.5) * grid_size,
                "lng": lon_min + (gy + 0.5) * grid_size
            }
        grids[key]["count"] += 1
        if is_green:
            grids[key]["green_count"] += 1

# 计算绿色比例并输出
heatmap_data = []
for key, val in grids.items():
    green_ratio = val["green_count"] / val["count"] if val["count"] > 0 else 0
    heatmap_data.append({
        "lat": val["lat"],
        "lng": val["lng"],
        "count": val["count"],
        "green_ratio": green_ratio
    })

with open(output_json, 'w') as f:
    json.dump(heatmap_data, f)

print(f"生成 {len(heatmap_data)} 个网格，保存至 {output_json}")