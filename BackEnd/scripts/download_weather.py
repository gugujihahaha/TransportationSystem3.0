"""
北京天气数据下载脚本 (高质量版)
数据源: meteostat Daily (日级别，缺失率远低于Hourly)
站点: 54511 (北京首都国际机场气象站)
时间: 2007-2012 (覆盖GeoLife数据集完整时间范围)

相比Hourly的优势:
1. 缺失率从~50%降到<10%
2. 降水/降雪字段更可靠
3. 不存在小时级别的随机噪声

使用方法:
    pip install meteostat
    python download_weather.py
"""

import meteostat
from datetime import datetime
import pandas as pd
import numpy as np

# 使用明确的导入方式避免 IDE 静态检查警告
Daily = meteostat.daily

# ===== 配置 =====
STATION_ID = '54511'   # 北京首都国际机场 (数据最全的北京站)
START = datetime(2007, 1, 1)
END   = datetime(2012, 12, 31)
OUTPUT_PATH = '../data/beijing_weather_daily_2007_2012.csv'


def download_and_validate():
    print("=" * 60)
    print("北京天气数据下载 (Daily, 高质量版)")
    print("=" * 60)

    # 1. 下载数据
    print(f"\n正在从 meteostat 下载站点 {STATION_ID} 的数据...")
    print(f"时间范围: {START.date()} ~ {END.date()}")

    data = Daily(STATION_ID, START, END)
    df = data.fetch()

    print(f"\n原始数据: {len(df)} 条记录")
    print(f"字段: {list(df.columns)}")

    # 2. 数据质量报告
    print("\n" + "=" * 40)
    print("数据质量报告")
    print("=" * 40)
    total_days = (END - START).days + 1
    print(f"预期天数: {total_days}")
    print(f"实际天数: {len(df)}")
    print(f"覆盖率: {len(df)/total_days:.1%}")
    print("\n各字段缺失率:")
    for col in df.columns:
        missing_rate = df[col].isna().mean()
        status = "✅" if missing_rate < 0.1 else ("⚠️" if missing_rate < 0.3 else "❌")
        print(f"  {status} {col:8s}: {missing_rate:.1%} 缺失")

    # 3. 补全缺失日期（确保每一天都有记录）
    print("\n正在补全缺失日期...")
    full_date_range = pd.date_range(START, END, freq='D')
    df = df.reindex(full_date_range)

    # 4. 填充缺失值
    # 先用前后插值，再用历史同期均值填充，最后用合理默认值兜底
    print("正在填充缺失值...")

    # 线性插值（适合温度、湿度等连续量）
    continuous_cols = ['tavg', 'tmin', 'tmax', 'wspd', 'pres']
    for col in continuous_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=7)  # 最多插值7天

    # 降水/降雪：缺失视为0（保守但合理）
    for col in ['prcp', 'snwd']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # 剩余NaN：用合理默认值
    defaults = {
        'tavg': 12.0,   # 北京年均气温约12°C
        'tmin': 7.0,
        'tmax': 18.0,
        'prcp': 0.0,
        'snwd': 0.0,    # 积雪深度
        'wspd': 2.5,    # 北京平均风速约2.5 m/s
        'pres': 1010.0,
        'wdir': 180.0,
        'wpgt': 0.0,
        'tsun': 0.0,
        'cldc': 50.0,    # 云量（50%表示多云）
    }
    for col, default in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    # 5. 构造额外特征列（供exp3直接使用）
    print("正在构造衍生特征...")

    # 用tavg作为主温度（如果没有则用tmin/tmax均值）
    if 'tavg' in df.columns:
        df['temp'] = df['tavg']
    elif 'tmin' in df.columns and 'tmax' in df.columns:
        df['temp'] = (df['tmin'] + df['tmax']) / 2

    # 降水量（已有prcp）
    # 风速（已有wspd）

    # 二值天气标记
    df['is_rainy']      = (df['prcp'] > 0.5).astype(int)    # 降水>0.5mm
    df['is_heavy_rain'] = (df['prcp'] > 10.0).astype(int)   # 降水>10mm
    df['is_snowy']      = (df['snwd'] > 0.0).astype(int)    # 有积雪（使用snwd字段）
    df['is_cold']       = (df['temp'] < 0.0).astype(int)    # 气温<0°C
    df['is_hot']        = (df['temp'] > 30.0).astype(int)   # 气温>30°C
    df['is_windy']      = (df['wspd'] > 6.0).astype(int)    # 风速>6 m/s

    # 归一化温度 ([-20, 40] -> [0, 1])
    df['temp_norm'] = (df['temp'] + 20) / 60.0
    df['temp_norm'] = df['temp_norm'].clip(0, 1)

    # 月份和季节（时间特征，帮助模型感知季节性）
    df['month'] = df.index.month
    df['season'] = df['month'].map({
        12: 0, 1: 0, 2: 0,   # 冬
        3: 1,  4: 1, 5: 1,   # 春
        6: 2,  7: 2, 8: 2,   # 夏
        9: 3, 10: 3, 11: 3   # 秋
    })

    # 6. 最终质量检查
    print("\n最终数据质量检查:")
    nan_count = df.isna().sum().sum()
    if nan_count == 0:
        print("  ✅ 无缺失值")
    else:
        print(f"  ⚠️ 剩余缺失值: {nan_count} 个")
        # 强制填0兜底
        df = df.fillna(0)

    # 7. 保存
    df.index.name = 'date'
    df.to_csv(OUTPUT_PATH)
    print(f"\n✅ 数据已保存: {OUTPUT_PATH}")
    print(f"   总天数: {len(df)}")
    print(f"   时间范围: {df.index[0].date()} ~ {df.index[-1].date()}")

    # 8. 简单统计
    print("\n" + "=" * 40)
    print("气候统计摘要")
    print("=" * 40)
    print(f"  年均温: {df['temp'].mean():.1f}°C")
    print(f"  最高温: {df['temp'].max():.1f}°C")
    print(f"  最低温: {df['temp'].min():.1f}°C")
    print(f"  年均降水: {df['prcp'].mean()*365:.0f}mm/年")
    print(f"  降水天数: {df['is_rainy'].sum()} 天 ({df['is_rainy'].mean():.1%})")
    print(f"  下雪天数: {df['is_snowy'].sum()} 天")
    print(f"  寒冷天数: {df['is_cold'].sum()} 天")
    print(f"  炎热天数: {df['is_hot'].sum()} 天")
    print(f"  大风天数: {df['is_windy'].sum()} 天")

    return df


if __name__ == '__main__':
    df = download_and_validate()
    print("\n数据样例:")
    print(df[['temp', 'prcp', 'snwd', 'wspd',
              'is_rainy', 'is_cold', 'is_hot', 'temp_norm']].head(10).to_string())