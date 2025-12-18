"""
exp6-MSF 模块: 辅助工具类 (坐标转换、归一化与指标计算)
复现论文: 《基于GPS轨迹多尺度表达的交通出行方式识别方法》- 马妍莉
"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


class Normalizer:
    """
    [复现论文 3.1.2] 全局特征归一化工具
    针对速度、加速度、转角变化率进行标准化处理
    """

    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, features: np.ndarray):
        """
        计算训练集的均值和标准差
        features: (N, 3) 包含 speed, accel, bearing_rate
        """
        self.means = np.mean(features, axis=0)
        self.stds = np.std(features, axis=0)
        # 防止除零
        self.stds[self.stds == 0] = 1.0

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.means is None:
            return features
        return (features - self.means) / self.stds


def calculate_metrics(y_true, y_pred, labels):
    """
    [复现论文 5.2.2] 评估指标计算
    计算 Accuracy, Precision, Recall, F1
    """
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # 混淆矩阵用于分析特定类别的识别效果 (如 car vs bus)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': cm
    }


def get_grid_indices(norm_coords: np.ndarray, grid_size: int):
    """
    将归一化后的 [0, 1] 坐标映射到网格索引 [0, grid_size-1]
    """
    indices = (norm_coords * (grid_size - 1)).astype(int)
    return np.clip(indices, 0, grid_size - 1)


def handle_imbalance(labels):
    """
    针对 GeoLife 类别不平衡计算类别权重
    用于 Loss Function 的 weight 参数
    """
    from collections import Counter
    counts = Counter(labels)
    total = sum(counts.values())
    num_classes = len(counts)

    # 权重计算公式: weight = total / (num_classes * count)
    weights = {label: total / (num_classes * count) for label, count in counts.items()}
    return weights