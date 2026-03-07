"""
通用数据处理模块
提供所有实验共用的基础数据预处理、模型基类和训练工具
"""

from .base_preprocessor import BaseGeoLifePreprocessor, BaseGeoLifeDataLoader
from .base_adapter import BaseDataAdapter
from .adapters import (
    Exp1DataAdapter,
    Exp2DataAdapter,
    Exp3DataAdapter,
    Exp4DataAdapter,
)
from .trajectory_cleaner import TrajectoryCleaner
from .base_model import BaseTransportationClassifier
from .train_utils import train_epoch, evaluate, compute_class_weights

__all__ = [
    'BaseGeoLifePreprocessor',
    'BaseGeoLifeDataLoader',
    'BaseDataAdapter',
    'Exp1DataAdapter',
    'Exp2DataAdapter',
    'Exp3DataAdapter',
    'Exp4DataAdapter',
    'TrajectoryCleaner',
    'BaseTransportationClassifier',
    'train_epoch',
    'evaluate',
    'compute_class_weights',
]
