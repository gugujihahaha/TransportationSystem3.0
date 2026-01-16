"""
通用数据加载器
加载清洗后的基础数据，供所有实验使用
"""
import os
import pickle
from typing import List, Dict, Tuple
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class CleanedDataLoader:
    """
    清洗后数据加载器（所有实验共用）

    功能:
    1. 加载清洗后的基础数据
    2. 提取实验所需的特征组合
    3. 提供统一的数据接口
    """

    def __init__(self, cleaned_data_path: str):
        """
        初始化数据加载器

        Args:
            cleaned_data_path: 清洗后数据路径
        """
        self.cleaned_data_path = cleaned_data_path
        self.cleaned_segments = None
        self.cleaning_stats = None
        self.cleaning_mode = None
        self.label_encoder = None

    def load(self) -> bool:
        """
        加载清洗后的数据

        Returns:
            是否加载成功
        """
        if not os.path.exists(self.cleaned_data_path):
            print(f"❌ 错误: 找不到清洗后数据: {self.cleaned_data_path}")
            print("\n请先运行以下命令生成清洗后的数据:")
            print("  python scripts/generate_cleaned_base_data.py")
            return False

        print(f"正在加载清洗后的数据: {self.cleaned_data_path}")

        with open(self.cleaned_data_path, 'rb') as f:
            data = pickle.load(f)
            
            if len(data) == 3:
                self.cleaned_segments, self.cleaning_stats, self.cleaning_mode = data
            else:
                self.cleaned_segments, self.cleaning_stats = data
                self.cleaning_mode = 'unknown'

        print(f"   ✓ 加载完成: {len(self.cleaned_segments)} 个轨迹段")
        print(f"   ✓ 清洗模式: {self.cleaning_mode}")

        return True

    def print_cleaning_stats(self):
        """打印清洗统计信息"""
        if self.cleaning_stats is None:
            print("⚠️ 没有清洗统计信息")
            return

        print("\n" + "=" * 60)
        print("数据清洗统计")
        print("=" * 60)

        print(f"\n清洗模式: {self.cleaning_mode}")

        print(f"\n数据统计:")
        print(f"  - 原始轨迹段: {self.cleaning_stats['total_segments']}")
        print(f"  - 保留轨迹段: {self.cleaning_stats['segments_kept']}")
        print(f"  - 丢弃轨迹段: {self.cleaning_stats['segments_discarded']}")
        print(f"  - 保留率: {self.cleaning_stats['segments_kept'] / self.cleaning_stats['total_segments'] * 100:.2f}%")

        print(f"\n清洗操作:")
        print(f"  - 剔除异常点: {self.cleaning_stats['outliers_removed']}")
        print(f"  - 插值点数: {self.cleaning_stats['points_interpolated']}")
        print(f"  - 平滑点数: {self.cleaning_stats['points_smoothed']}")

        print(f"\n丢弃原因:")
        for reason, count in self.cleaning_stats['discard_reasons'].items():
            if count > 0:
                print(f"  - {reason}: {count}")

    def get_label_encoder(self) -> LabelEncoder:
        """
        获取标签编码器

        Returns:
            LabelEncoder 实例
        """
        if self.label_encoder is None:
            labels = [seg['label'] for seg in self.cleaned_segments]
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(labels)
        
        return self.label_encoder

    def get_exp1_features(self) -> List[Tuple]:
        """
        获取 Exp1 所需的特征（仅轨迹）

        Returns:
            特征列表，每个元素为 (trajectory, label)
        """
        features = []
        
        for seg in self.cleaned_segments:
            trajectory = seg['cleaned_trajectory']
            label = seg['label']
            features.append((trajectory, label))
        
        return features

    def get_exp2_features(self) -> List[Tuple]:
        """
        获取 Exp2 所需的特征（轨迹 + 基础KG）

        Returns:
            特征列表，每个元素为 (trajectory, kg, label)
        """
        features = []
        
        for seg in self.cleaned_segments:
            trajectory = seg['cleaned_trajectory']
            kg = seg.get('kg_features', np.zeros((50, 11)))
            label = seg['label']
            features.append((trajectory, kg, label))
        
        return features

    def get_exp3_features(self) -> List[Tuple]:
        """
        获取 Exp3 所需的特征（轨迹 + 增强KG）

        Returns:
            特征列表，每个元素为 (trajectory, kg, label)
        """
        features = []
        
        for seg in self.cleaned_segments:
            trajectory = seg['cleaned_trajectory']
            kg = seg.get('kg_features', np.zeros((50, 15)))
            label = seg['label']
            features.append((trajectory, kg, label))
        
        return features

    def get_exp4_features(self) -> List[Tuple]:
        """
        获取 Exp4 所需的特征（轨迹 + KG + 天气）

        Returns:
            特征列表，每个元素为 (trajectory, kg, weather, label)
        """
        features = []
        
        for seg in self.cleaned_segments:
            trajectory = seg['cleaned_trajectory']
            kg = seg.get('kg_features', np.zeros((50, 15)))
            weather = seg.get('weather_features', np.zeros((50, 12)))
            label = seg['label']
            features.append((trajectory, kg, weather, label))
        
        return features

    def get_exp5_features(self) -> List[Tuple]:
        """
        获取 Exp5 所需的特征（轨迹 + KG + 天气，与Exp4相同）

        Returns:
            特征列表，每个元素为 (trajectory, kg, weather, label)
        """
        return self.get_exp4_features()

    def get_statistics(self) -> Dict:
        """
        获取数据统计信息

        Returns:
            统计信息字典
        """
        labels = [seg['label'] for seg in self.cleaned_segments]
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        return {
            'total_segments': len(self.cleaned_segments),
            'unique_labels': unique_labels.tolist(),
            'label_counts': counts.tolist(),
            'cleaning_mode': self.cleaning_mode,
            'cleaning_stats': self.cleaning_stats
        }


def get_cleaned_data_path(cleaning_mode: str = 'balanced') -> str:
    """
    获取清洗后数据路径

    Args:
        cleaning_mode: 清洗模式

    Returns:
        清洗后数据路径
    """
    return f'../data/processed/cleaned_segments_{cleaning_mode}.pkl'
