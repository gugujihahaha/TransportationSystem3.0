"""
天气数据预处理模块 (Exp4)
功能：加载、处理和特征化天气数据
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import warnings


class WeatherDataProcessor:
    """天气数据处理器"""

    def __init__(self, weather_csv_path: str):
        """
        初始化天气数据处理器

        Args:
            weather_csv_path: 天气CSV文件路径
        """
        self.weather_csv_path = weather_csv_path
        self.daily_weather = None
        self.weather_features_cache = {}

    def load_and_process(self) -> pd.DataFrame:
        """
        加载并处理天气数据

        Returns:
            daily_weather: 日级别聚合的天气数据
        """
        print("\n========== 天气数据加载与处理 ==========")

        # 1. 加载小时级数据
        print(f"1. 正在加载天气数据: {self.weather_csv_path}")
        hourly_data = pd.read_csv(self.weather_csv_path, index_col=0, parse_dates=True)
        print(f"   加载完成: {len(hourly_data)} 条小时记录")

        # 2. 数据清洗
        print("2. 正在清洗数据...")
        # 将 '<NA>' 字符串替换为 NaN
        hourly_data = hourly_data.replace('<NA>', np.nan)

        # 确保数值列为 float 类型
        numeric_columns = ['temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir',
                           'wspd', 'wpgt', 'pres', 'tsun']
        for col in numeric_columns:
            if col in hourly_data.columns:
                hourly_data[col] = pd.to_numeric(hourly_data[col], errors='coerce')

        # 3. 聚合为日级数据
        print("3. 正在聚合为日级数据...")
        self.daily_weather = self._aggregate_to_daily(hourly_data)
        print(f"   聚合完成: {len(self.daily_weather)} 条日记录")

        # 4. 构造天气特征
        print("4. 正在构造天气特征...")
        self.daily_weather = self._construct_weather_features(self.daily_weather)

        print("✅ 天气数据处理完成")
        print(f"   日期范围: {self.daily_weather.index.min()} 至 {self.daily_weather.index.max()}")
        print(f"   特征列: {list(self.daily_weather.columns)}")

        return self.daily_weather

    def _aggregate_to_daily(self, hourly_data: pd.DataFrame) -> pd.DataFrame:
        """
        将小时数据聚合为日数据

        聚合规则:
        - temp: 平均温度
        - prcp: 总降水量
        - snow: 总降雪量
        - wspd: 平均风速
        - rhum: 平均相对湿度
        """
        # 按日期分组
        daily_agg = hourly_data.groupby(hourly_data.index.date).agg({
            'temp': 'mean',  # 平均温度
            'prcp': 'sum',  # 总降水量
            'snow': 'sum',  # 总降雪量
            'wspd': 'mean',  # 平均风速
            'rhum': 'mean',  # 平均湿度
        })

        # 将索引转换为 datetime
        daily_agg.index = pd.to_datetime(daily_agg.index)

        # 填充缺失值（前向填充）
        daily_agg = daily_agg.fillna(method='ffill')

        # 剩余的NaN用0填充
        daily_agg = daily_agg.fillna(0)

        return daily_agg

    def _construct_weather_features(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """
        构造二值天气特征

        特征:
        - is_rainy: 是否有降水 (prcp > 0)
        - is_heavy_rain: 是否大雨 (prcp > 10)
        - is_snowy: 是否下雪 (snow > 0)
        - is_cold: 是否寒冷 (temp < 0)
        - is_hot: 是否炎热 (temp > 30)
        - is_windy: 是否风大 (wspd > 6)
        """
        df = daily_data.copy()

        # 二值特征
        df['is_rainy'] = (df['prcp'] > 0).astype(int)
        df['is_heavy_rain'] = (df['prcp'] > 10).astype(int)
        df['is_snowy'] = (df['snow'] > 0).astype(int)
        df['is_cold'] = (df['temp'] < 0).astype(int)
        df['is_hot'] = (df['temp'] > 30).astype(int)
        df['is_windy'] = (df['wspd'] > 6).astype(int)

        return df

    def get_weather_features_for_date(self, date: pd.Timestamp) -> np.ndarray:
        """
        获取指定日期的天气特征 (12维)

        Args:
            date: 日期时间戳

        Returns:
            weather_features: (12,) 天气特征向量
            [temp, prcp, snow, wspd, rhum,
             is_rainy, is_heavy_rain, is_snowy, is_cold, is_hot, is_windy, 归一化temp]
        """
        # 提取日期部分
        date_only = pd.Timestamp(date.date())

        # 检查缓存
        if date_only in self.weather_features_cache:
            return self.weather_features_cache[date_only]

        # 如果没有这一天的数据，返回全零
        if date_only not in self.daily_weather.index:
            # 尝试前向填充：找最近的日期
            if len(self.daily_weather) > 0:
                closest_date = self.daily_weather.index[
                    (self.daily_weather.index - date_only).abs().argmin()
                ]
                if abs((closest_date - date_only).days) <= 3:  # 3天内
                    date_only = closest_date
                else:
                    features = np.zeros(12, dtype=np.float32)
                    self.weather_features_cache[date_only] = features
                    return features
            else:
                features = np.zeros(12, dtype=np.float32)
                self.weather_features_cache[date_only] = features
                return features

        # 获取天气数据
        weather = self.daily_weather.loc[date_only]

        # 构造特征向量 (12维)
        features = np.array([
            weather['temp'],  # 0: 温度
            weather['prcp'],  # 1: 降水量
            weather['snow'],  # 2: 降雪量
            weather['wspd'],  # 3: 风速
            weather['rhum'],  # 4: 湿度
            weather['is_rainy'],  # 5: 是否降水
            weather['is_heavy_rain'],  # 6: 是否大雨
            weather['is_snowy'],  # 7: 是否下雪
            weather['is_cold'],  # 8: 是否寒冷
            weather['is_hot'],  # 9: 是否炎热
            weather['is_windy'],  # 10: 是否风大
            (weather['temp'] + 20) / 50  # 11: 归一化温度 (假设范围 -20~30)
        ], dtype=np.float32)

        # 缓存
        self.weather_features_cache[date_only] = features

        return features

    def get_weather_features_for_trajectory(self, trajectory_dates: pd.Series) -> np.ndarray:
        """
        获取轨迹的天气特征 (批量)

        Args:
            trajectory_dates: 轨迹的日期时间序列

        Returns:
            weather_features: (N, 12) 天气特征矩阵
        """
        N = len(trajectory_dates)
        weather_features = np.zeros((N, 12), dtype=np.float32)

        for i, date in enumerate(trajectory_dates):
            try:
                weather_features[i] = self.get_weather_features_for_date(date)
            except Exception as e:
                # Soft modality: missing weather data defaults to zero vector
                weather_features[i] = np.zeros(12, dtype=np.float32)

        return weather_features

    def get_statistics(self) -> Dict:
        """获取天气数据统计信息"""
        if self.daily_weather is None:
            return {}

        stats = {
            'total_days': len(self.daily_weather),
            'date_range': f"{self.daily_weather.index.min()} ~ {self.daily_weather.index.max()}",
            'avg_temp': self.daily_weather['temp'].mean(),
            'avg_prcp': self.daily_weather['prcp'].mean(),
            'rainy_days': self.daily_weather['is_rainy'].sum(),
            'snowy_days': self.daily_weather['is_snowy'].sum(),
            'cold_days': self.daily_weather['is_cold'].sum(),
            'hot_days': self.daily_weather['is_hot'].sum(),
        }

        return stats