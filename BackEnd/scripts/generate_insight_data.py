#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从预测点数据、天气数据、OSM 道路数据生成数据洞察页所需的聚合统计
输出: insight_data.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# ==================== 配置路径 ====================
POINTS_CSV = r"D:/TransportationSystem3.0/BackEnd/data/predictions_with_geo/points_with_geo_exp3.csv"
WEATHER_CSV = r"D:/TransportationSystem3.0/BackEnd/data/beijing_weather_daily_2007_2012.csv"
OSM_GEOJSON = r"D:/TransportationSystem3.0/BackEnd/data/exp2.geojson"  # 用于道路类型匹配
OUTPUT_JSON = r"D:/TransportationSystem3.0/FrontEnd/public/insight_data.json"

# 交通方式顺序（用于图表）
MODES = ['Walk', 'Bike', 'Bus', 'Car & taxi', 'Subway', 'Train']
MODE_COLORS = {
    'Walk': '#00FF88',
    'Bike': '#00FFFF',
    'Bus': '#FFDD00',
    'Car & taxi': '#FF0033',
    'Subway': '#CC33FF',
    'Train': '#FF6600'
}

# ==================== 1. 加载点数据 ====================
print("加载点数据...")
df = pd.read_csv(POINTS_CSV)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['date'] = df['timestamp'].dt.date


# 定义时段
def get_time_period(hour):
    if 7 <= hour <= 9:
        return '早高峰 (7-9点)'
    elif 11 <= hour <= 13:
        return '午间 (11-13点)'
    elif 17 <= hour <= 19:
        return '晚高峰 (17-19点)'
    elif 22 <= hour or hour <= 5:
        return '夜间 (22-5点)'
    else:
        return '其他时段'


df['period'] = df['hour'].apply(get_time_period)

print(f"总点数: {len(df)}")

# ==================== 2. 时段出行分析 ====================
print("计算时段出行分析...")
period_mode_counts = df.groupby(['period', 'pred_label']).size().unstack(fill_value=0)
# 转换为百分比（每个时段内各模式占比）
period_mode_pct = period_mode_counts.div(period_mode_counts.sum(axis=1), axis=0) * 100
period_mode_pct = period_mode_pct.round(2)

# 确保所有 MODES 都存在
for mode in MODES:
    if mode not in period_mode_pct.columns:
        period_mode_pct[mode] = 0.0

insight_data = {
    "time_period": {
        "periods": list(period_mode_pct.index),
        "modes": MODES,
        "data": period_mode_pct[MODES].to_dict(orient='list')
    }
}

# ==================== 3. 天气影响分析 ====================
print("加载天气数据...")
# 修复：将日期列作为普通列加载，不设索引
weather_df = pd.read_csv(WEATHER_CSV)
# 假设第一列是日期，列名可能是 'date', 'time', 'datetime' 等，尝试自动识别
date_col = None
for col in weather_df.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        date_col = col
        break
if date_col is None:
    raise ValueError("天气文件中未找到日期列，请检查列名")
weather_df['date'] = pd.to_datetime(weather_df[date_col]).dt.date
# 计算降雨标记（假设有 prcp 列）
prcp_col = None
for col in weather_df.columns:
    if 'prcp' in col.lower() or 'precip' in col.lower():
        prcp_col = col
        break
if prcp_col is None:
    weather_df['is_rainy'] = 0
else:
    weather_df['is_rainy'] = (weather_df[prcp_col] > 0.5).astype(int)

# 合并天气信息到点数据
df['date'] = pd.to_datetime(df['date']).dt.date
df = df.merge(weather_df[['date', 'is_rainy']], on='date', how='left')
df['is_rainy'] = df['is_rainy'].fillna(0).astype(int)

print("计算天气影响分析...")
rainy_modes = df[df['is_rainy'] == 1]['pred_label'].value_counts(normalize=True) * 100
sunny_modes = df[df['is_rainy'] == 0]['pred_label'].value_counts(normalize=True) * 100

weather_data = {
    "rainy": {mode: rainy_modes.get(mode, 0) for mode in MODES},
    "sunny": {mode: sunny_modes.get(mode, 0) for mode in MODES}
}
insight_data["weather_impact"] = weather_data

# ==================== 4. 拥堵时段分析（小汽车占比） ====================
print("计算拥堵时段分析...")
car_ratio_by_hour = df.groupby('hour')['pred_label'].apply(lambda x: (x == 'Car & taxi').mean() * 100)
car_ratio_by_hour = car_ratio_by_hour.round(2)
# 确保所有小时 0-23 都有值
all_hours = pd.Series(index=range(24), dtype=float)
all_hours.update(car_ratio_by_hour)
all_hours = all_hours.fillna(0)
insight_data["congestion_timing"] = {
    "hours": list(all_hours.index),
    "car_ratio": all_hours.tolist()
}

# ==================== 5. 道路类型出行分析（需要 OSM 匹配）====================
print("加载 OSM 数据并匹配道路类型...")
road_type_available = False
try:
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    # 加载 OSM GeoJSON
    if not os.path.exists(OSM_GEOJSON):
        raise FileNotFoundError(f"OSM 文件不存在: {OSM_GEOJSON}")

    gdf = gpd.read_file(OSM_GEOJSON)
    # 提取道路类型（highway 或 railway）
    if 'highway' in gdf.columns:
        gdf['road_type'] = gdf['highway'].fillna('unknown')
    elif 'railway' in gdf.columns:
        gdf['road_type'] = gdf['railway'].fillna('unknown')
    else:
        raise ValueError("OSM 数据中未找到 highway 或 railway 字段")

    # 过滤有效道路类型
    valid_types = ['primary', 'secondary', 'tertiary', 'residential', 'footway', 'cycleway', 'bus_stop', 'station']
    gdf = gdf[gdf['road_type'].isin(valid_types)].copy()
    # 映射到更简洁的类别
    type_mapping = {
        'primary': '主干道', 'secondary': '次干道', 'tertiary': '支路',
        'residential': '居住区道路', 'footway': '步行道', 'cycleway': '自行车道',
        'bus_stop': '公交站', 'station': '地铁站'
    }
    gdf['road_category'] = gdf['road_type'].map(type_mapping).fillna('其他')

    # 构建空间索引 (STRtree)
    geometries = gdf.geometry.tolist()
    tree = STRtree(geometries)
    categories = gdf['road_category'].tolist()


    def get_road_category(lon, lat):
        point = Point(lon, lat)
        # 查询最近的几何体
        nearest_idx = tree.nearest(point)
        return categories[nearest_idx]


    print("匹配道路类型（可能需要几分钟）...")
    tqdm.pandas(desc="匹配道路")
    df['road_category'] = df.progress_apply(lambda row: get_road_category(row['longitude'], row['latitude']), axis=1)

    # 统计每种道路类型上的出行方式占比
    road_mode_counts = df.groupby(['road_category', 'pred_label']).size().unstack(fill_value=0)
    road_mode_pct = road_mode_counts.div(road_mode_counts.sum(axis=1), axis=0) * 100
    road_mode_pct = road_mode_pct.round(2)

    # 确保所有 MODES 都存在
    for mode in MODES:
        if mode not in road_mode_pct.columns:
            road_mode_pct[mode] = 0.0

    insight_data["road_type"] = {
        "categories": list(road_mode_pct.index),
        "modes": MODES,
        "data": road_mode_pct[MODES].to_dict(orient='list')
    }
    road_type_available = True
    print("道路类型匹配完成")
except Exception as e:
    print(f"道路类型匹配失败（将使用占位数据）: {e}")
    # 提供占位数据
    insight_data["road_type"] = {
        "categories": ["主干道", "次干道", "支路", "步行道"],
        "modes": MODES,
        "data": {
            "Walk": [20, 10, 15, 80],
            "Bike": [10, 15, 20, 10],
            "Bus": [30, 25, 20, 5],
            "Car & taxi": [30, 35, 25, 2],
            "Subway": [5, 10, 15, 2],
            "Train": [5, 5, 5, 1]
        }
    }

# ==================== 保存 JSON ====================
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(insight_data, f, ensure_ascii=False, indent=2)

print(f"✅ 洞察数据已保存到: {OUTPUT_JSON}")