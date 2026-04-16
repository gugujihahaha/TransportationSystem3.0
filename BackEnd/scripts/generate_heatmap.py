import pickle
import json
import os

pkl_path = r'D:\TransportationSystem3.0\BackEnd\data\processed\cleaned_balanced.pkl'
output_json = r'D:\TransportationSystem3.0\FrontEnd\public\heatmap_grid.json'

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

green_labels = {'Walk', 'Bike'}
total_green_points = 0
total_points = 0

for item in data:
    traj_9 = item[0]          # (50,9)
    label = item[3]           # 字符串标签
    is_green = label in green_labels
    total_points += len(traj_9)
    if is_green:
        total_green_points += len(traj_9)

print(f"总轨迹点: {total_points}, 绿色点: {total_green_points}, 绿色比例: {total_green_points/total_points:.2%}")

if total_green_points == 0:
    print("警告：绿色点为0，检查前10条标签：")
    for i, item in enumerate(data[:10]):
        print(f"样本{i}: label={item[3]}, type={type(item[3])}")
else:
    # 继续生成网格数据...
    lat_min, lat_max = 39.7, 40.2
    lon_min, lon_max = 116.1, 116.7
    grid_size = 0.005

    grids = {}
    for item in data:
        traj_9 = item[0]
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
                grids[key] = {"count": 0, "green_count": 0, "lat": lat_min + (gx + 0.5) * grid_size, "lng": lon_min + (gy + 0.5) * grid_size}
            grids[key]["count"] += 1
            if is_green:
                grids[key]["green_count"] += 1

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