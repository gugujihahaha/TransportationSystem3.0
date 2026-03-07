"""
天气数据预处理模块 (Exp3 - Daily版)
适配 beijing_weather_daily_2007_2012.csv 的列结构

特征维度: 10维
    [0]  temp        - 平均气温 (°C)
    [1]  tmin        - 最低气温 (°C)
    [2]  tmax        - 最高气温 (°C)
    [3]  prcp        - 降水量 (mm)
    [4]  wspd        - 风速 (km/h，meteostat单位)
    [5]  is_rainy    - 是否降水 (prcp > 0.5mm)
    [6]  is_heavy_rain - 是否大雨 (prcp > 10mm)
    [7]  is_snowy    - 是否降雪 (prcp > 0 AND temp < 2°C)
    [8]  is_cold     - 是否寒冷 (temp < 0°C)
    [9]  is_hot      - 是否炎热 (temp > 30°C)
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings


class WeatherDataProcessor:
    """天气数据处理器 (Daily版 - 适配高质量meteostat数据)"""

    WEATHER_FEATURE_DIM = 10  # 修正：daily数据10维

    def __init__(self, weather_csv_path: str):
        self.weather_csv_path = weather_csv_path
        self.daily_weather = None
        self.weather_features_cache = {}
        self._default_features = np.zeros(self.WEATHER_FEATURE_DIM, dtype=np.float32)
        self._load_successful = False

    def load_and_process(self) -> pd.DataFrame:
        print("\n========== 天气数据加载与处理 ==========")
        try:
            print(f"1. 正在加载: {self.weather_csv_path}")
            df = pd.read_csv(self.weather_csv_path, index_col=0, parse_dates=True)
            print(f"   加载完成: {len(df)} 条日记录")

            if len(df) == 0:
                print("   ⚠️ 天气数据为空")
                self._init_empty()
                return self.daily_weather

            # 只保留有效列
            keep_cols = ['temp', 'tmin', 'tmax', 'prcp', 'wspd']
            df = df[[c for c in keep_cols if c in df.columns]]

            # 填充剩余缺失值
            defaults = {'temp': 13.5, 'tmin': 8.0, 'tmax': 19.0, 'prcp': 0.0, 'wspd': 10.0}
            for col, val in defaults.items():
                if col in df.columns:
                    df[col] = df[col].fillna(val)

            # 构造二值特征
            # is_windy阈值用36km/h(10m/s)，适配meteostat的km/h单位
            df['is_rainy']      = (df['prcp'] > 0.5).astype(np.float32)
            df['is_heavy_rain'] = (df['prcp'] > 10.0).astype(np.float32)
            # 降雪：降水且气温<2°C（snwd字段全缺，用此推断）
            df['is_snowy']      = ((df['prcp'] > 0.5) & (df['temp'] < 2.0)).astype(np.float32)
            df['is_cold']       = (df['temp'] < 0.0).astype(np.float32)
            df['is_hot']        = (df['temp'] > 30.0).astype(np.float32)

            # 最终清理
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            self.daily_weather = df
            self._load_successful = True

            print(f"✅ 天气处理完成 | {len(df)} 天 | "
                  f"{df.index.min().date()} ~ {df.index.max().date()}")
            # 简单统计验证
            print(f"   降水天: {int(df['is_rainy'].sum())} | "
                  f"降雪天: {int(df['is_snowy'].sum())} | "
                  f"寒冷天: {int(df['is_cold'].sum())} | "
                  f"炎热天: {int(df['is_hot'].sum())}")

        except FileNotFoundError:
            print(f"   ❌ 文件不存在: {self.weather_csv_path}")
            self._init_empty()
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            self._init_empty()

        return self.daily_weather

    def _init_empty(self):
        self.daily_weather = pd.DataFrame()
        self._load_successful = False

    def get_weather_features_for_date(self, date) -> np.ndarray:
        """获取指定日期的10维天气特征"""
        if date is None or (hasattr(date, '__class__') and
                            date.__class__.__name__ == 'NaTType'):
            return self._default_features.copy()
        try:
            date_key = pd.Timestamp(date).normalize()
        except Exception:
            return self._default_features.copy()

        if date_key in self.weather_features_cache:
            return self.weather_features_cache[date_key].copy()

        if not self._load_successful or self.daily_weather is None or len(self.daily_weather) == 0:
            self.weather_features_cache[date_key] = self._default_features.copy()
            return self._default_features.copy()

        # 找最近日期（最多容忍7天偏差）
        target = date_key
        if target not in self.daily_weather.index:
            diffs = (self.daily_weather.index - target).days
            abs_diffs = np.abs(diffs)
            min_idx = abs_diffs.argmin()
            if abs_diffs[min_idx] > 7:
                self.weather_features_cache[date_key] = self._default_features.copy()
                return self._default_features.copy()
            target = self.daily_weather.index[min_idx]

        try:
            row = self.daily_weather.loc[target]
            features = np.array([
                float(row.get('temp',  13.5)),   # 0: 平均温度
                float(row.get('tmin',  8.0)),    # 1: 最低温
                float(row.get('tmax',  19.0)),   # 2: 最高温
                float(row.get('prcp',  0.0)),    # 3: 降水量
                float(row.get('wspd',  10.0)),   # 4: 风速(km/h)
                float(row.get('is_rainy',      0)),  # 5
                float(row.get('is_heavy_rain', 0)),  # 6
                float(row.get('is_snowy',      0)),  # 7
                float(row.get('is_cold',       0)),  # 8
                float(row.get('is_hot',        0)),  # 9
            ], dtype=np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            features = self._default_features.copy()

        self.weather_features_cache[date_key] = features
        return features.copy()

    def get_weather_features_for_trajectory(self, trajectory_dates: pd.Series) -> np.ndarray:
        """为轨迹序列的每个时间点获取天气特征 (N, 10)"""
        if trajectory_dates is None or len(trajectory_dates) == 0:
            return np.zeros((0, self.WEATHER_FEATURE_DIM), dtype=np.float32)

        N = len(trajectory_dates)
        features = np.zeros((N, self.WEATHER_FEATURE_DIM), dtype=np.float32)
        for i, date in enumerate(trajectory_dates):
            try:
                features[i] = self.get_weather_features_for_date(date)
            except Exception:
                features[i] = self._default_features.copy()

        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def get_statistics(self) -> Dict:
        if not self._load_successful or self.daily_weather is None or len(self.daily_weather) == 0:
            return {'total_days': 0, 'load_successful': False}
        df = self.daily_weather
        return {
            'total_days': len(df),
            'date_range': f"{df.index.min().date()} ~ {df.index.max().date()}",
            'avg_temp': round(float(df['temp'].mean()), 1),
            'rainy_days': int(df['is_rainy'].sum()),
            'snowy_days': int(df['is_snowy'].sum()),
            'cold_days':  int(df['is_cold'].sum()),
            'hot_days':   int(df['is_hot'].sum()),
            'load_successful': True
        }