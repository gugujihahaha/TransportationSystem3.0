"""
标签平滑 + Focal Loss

结合标签平滑防止过拟合和 Focal Loss 针对难分类样本加大惩罚
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingFocalLoss(nn.Module):
    """
    标签平滑 + Focal Loss

    Args:
        num_classes: 类别数量
        gamma: Focal Loss 的 gamma 参数（默认 2.0）
        smoothing: 标签平滑系数（默认 0.1）
        weight: 类别权重 (num_classes,)
    """

    def __init__(self, num_classes, gamma=2.0, smoothing=0.1, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits, labels):
        """
        前向传播

        Args:
            logits: (B, num_classes) 模型输出
            labels: (B,) 真实标签

        Returns:
            loss: 标量损失值
        """
        # 标签平滑
        with torch.no_grad():
            smooth_labels = torch.full_like(logits, self.smoothing / (self.num_classes - 1))
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)

        # 计算对数概率
        log_prob = F.log_softmax(logits, dim=1)
        prob = torch.exp(log_prob)

        # Focal 权重：基于真实类别的预测概率
        pt = prob.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma

        # 交叉熵（对平滑标签）
        ce = -(smooth_labels * log_prob).sum(dim=1)

        # 应用类别权重
        if self.weight is not None:
            cls_weight = self.weight[labels]
            loss = (focal_weight * ce * cls_weight).mean()
        else:
            loss = (focal_weight * ce).mean()

        return loss
