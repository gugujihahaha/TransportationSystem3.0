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

heatmap, _, _ = np.histogram2d(car_df['longitude'], car_df['latitude'], bins=[lon_bins, lat_bins])
heatmap = heatmap.T
heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)

grid_data = []
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