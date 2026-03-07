"""
各实验数据适配器。

每个适配器继承 BaseDataAdapter，只覆盖 experiment_name 和 _format_output，
差异由输出格式决定：
  - Exp1/2/3: 返回 (features, label)，丢弃时间序列
  - Exp4/5:   返回 (features, datetime_series, label)，保留时间序列（天气特征需要）
"""
import numpy as np
import pandas as pd
from typing import List, Tuple

from common.base_adapter import BaseDataAdapter


class Exp1DataAdapter(BaseDataAdapter):
    """Exp1 适配器：返回 (features, label)，仅轨迹特征。"""

    def __init__(self, enable_cleaning=True, cleaning_mode='balanced',
                 cache_dir='../data/processed'):
        super().__init__(enable_cleaning, cleaning_mode, cache_dir)

    @property
    def experiment_name(self) -> str:
        return "Exp1"

    def _format_output(self, cleaned_segments: List[Tuple]) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """丢弃时间序列，返回 (traj_features, segment_stats, label)。"""
        return [(traj, stats, label) for traj, stats, _, label in cleaned_segments]


class Exp2DataAdapter(BaseDataAdapter):
    """Exp2 适配器：返回 (features, label)，轨迹 + OSM 空间特征。"""

    def __init__(self, enable_cleaning=True, cleaning_mode='balanced',
                 cache_dir='./data/processed'):
        super().__init__(enable_cleaning, cleaning_mode, cache_dir)

    @property
    def experiment_name(self) -> str:
        return "Exp2"

    def _format_output(self, cleaned_segments: List[Tuple]) -> List[Tuple[np.ndarray, str]]:
        """丢弃时间序列，合并 traj 和 stats，返回 (features, label)。"""
        result = []
        for traj, stats, _, label in cleaned_segments:
            N = traj.shape[0]
            stats_expanded = np.tile(stats, (N, 1))
            features = np.concatenate([traj, stats_expanded], axis=1)
            result.append((features, label))
        return result


class Exp3DataAdapter(Exp2DataAdapter):
    """Exp3 适配器：与 Exp2 输出格式相同，仅实验名不同。"""

    @property
    def experiment_name(self) -> str:
        return "Exp3"


class Exp4DataAdapter(BaseDataAdapter):
    """Exp4 适配器：返回 (features, datetime_series, label)，保留时间序列供天气特征使用。"""

    def __init__(self, enable_cleaning=True, cleaning_mode='balanced',
                 cache_dir='./data/processed'):
        super().__init__(enable_cleaning, cleaning_mode, cache_dir)

    @property
    def experiment_name(self) -> str:
        return "Exp4"

    def _format_output(self, cleaned_segments: List[Tuple]) -> List[Tuple[np.ndarray, np.ndarray, pd.Series, str]]:
        """保留时间序列，返回完整四元组。"""
        return cleaned_segments


class Exp5DataAdapter(Exp4DataAdapter):
    """Exp5 适配器：与 Exp4 输出格式相同，仅实验名不同。"""

    @property
    def experiment_name(self) -> str:
        return "Exp5"
