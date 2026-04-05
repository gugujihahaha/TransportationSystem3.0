import pandas as pd
import numpy as np
import json
import os

POINTS_CSV = r"D:/TransportationSystem3.0/BackEnd/data/predictions_with_geo/points_with_geo_exp1.csv"
OUTPUT_JSON = r"D:/TransportationSystem3.0/FrontEnd/public/heatmap_grid.json"

LON_MIN, LON_MAX = 115.4, 117.0
LAT_MIN, LAT_MAX = 39.4, 41.2
GRID_SIZE = 0.01

df = pd.read_csv(POINTS_CSV)
car_df = df[df['pred_label'] == 'Car & taxi']
print(f"总点数: {len(df)}, 小汽车点数: {len(car_df)}")

lon_bins = np.arange(LON_MIN, LON_MAX, GRID_SIZE)
lat_bins = np.arange(LAT_MIN, LAT_MAX, GRID_SIZE)

# 注意：histogram2d 的返回矩阵形状为 (len(lat_bins)-1, len(lon_bins)-1)
heatmap, _, _ = np.histogram2d(car_df['longitude'], car_df['latitude'], bins=[lon_bins, lat_bins])
# 修正：heatmap 的 shape 是 (经度网格数, 纬度网格数)？实际上根据文档，第一个维度对应 x (经度)，第二个对应 y (纬度)
# 但为了清晰，我们统一转置为 (纬度, 经度) 以便按 (lat, lon) 索引
heatmap = heatmap.T   # 现在 shape = (纬度网格数, 经度网格数)
heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)

grid_data = []
# 正确顺序：外层循环纬度（行），内层循环经度（列）
for i in range(len(lat_bins)-1):      # 纬度索引
    for j in range(len(lon_bins)-1):  # 经度索引
        val = float(heatmap_norm[i, j])   # 注意顺序 [i, j]
        if val > 0.01:
            grid_data.append({
                'lng': (lon_bins[j] + lon_bins[j+1]) / 2,
                'lat': (lat_bins[i] + lat_bins[i+1]) / 2,
                'value': val
            })

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(grid_data, f)
print(f"✅ 生成 {len(grid_data)} 个网格点，保存至 {OUTPUT_JSON}")