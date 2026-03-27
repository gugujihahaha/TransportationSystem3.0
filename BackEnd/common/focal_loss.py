import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingFocalLoss(nn.Module):
    """
    标签平滑 + Focal Loss
    
    参数:
        num_classes: 类别数
        gamma: Focal Loss的focusing参数 (默认2.0)
        smoothing: 标签平滑系数 (默认0.1)
        weight: 类别权重 (可选)
    """
    
    def __init__(self, num_classes, gamma=2.0, smoothing=0.1, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smoothing = smoothing
        self.weight = weight
        
    def forward(self, logits, targets):
        """
        参数:
            logits: (B, num_classes) 模型输出
            targets: (B,) 目标标签
        """
        # 转换为one-hot
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # 标签平滑
        if self.smoothing > 0:
            targets_one_hot = (1 - self.smoothing) * targets_one_hot + \
                           self.smoothing / self.num_classes
        
        # 计算概率
        probs = F.softmax(logits, dim=1)
        
        # Focal Loss
        focal_weight = (1 - probs) ** self.gamma
        focal_loss = -focal_weight * targets_one_hot * torch.log(probs + 1e-8)
        
        # 应用类别权重
        if self.weight is not None:
            focal_loss = focal_loss * self.weight.unsqueeze(0)
        
        # 平均
        loss = focal_loss.sum(dim=1).mean()
        
        return loss