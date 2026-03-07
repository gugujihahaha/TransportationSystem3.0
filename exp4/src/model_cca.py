"""
CCA 对比学习模型 (Exp4)

核心思想：
- 使用 exp2 的 21 维点级融合特征（9 轨迹 + 12 空间）
- 将 21 维特征拆分为轨迹表示（9 维）和空间上下文表示（12 维）
- 通过 InfoNCE 对比损失对齐两个表示
- 主路径用于分类，辅路径仅用于对比学习

与 exp2 的关系：
- 特征维度完全一致（21 维）
- 数据划分完全一致（70/10/20）
- 超参数完全一致
- 唯一区别：新增 CCA 对比学习损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """注意力池化层"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, D) 序列特征
            mask: (B, T) 可选的掩码
        Returns:
            pooled: (B, D) 池化后的特征
        """
        attn_weights = self.attention(x).squeeze(-1)  # (B, T)
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_weights, dim=1)  # (B, T)
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (B, D)
        
        return pooled


class CCATransportationClassifier(nn.Module):
    """
    CCA 对比学习分类器

    架构：
    - 输入：traj_21dim (B, T, 21) + stats_18dim (B, 18)
    - 拆分：traj_9 = traj_21[:,:,:9], spatial_12 = traj_21[:,:,9:]
    - 主路径：traj_9 → BiLSTM(256) → Attention → Classifier
    - 辅路径：spatial_12 → BiLSTM(128) → Attention
    - CCA：InfoNCE 对比损失对齐两个表示
    """

    def __init__(
        self,
        trajectory_feature_dim: int = 21,
        segment_stats_dim: int = 18,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
        context_loss_weight: float = 0.1,
        temperature: float = 0.07
    ):
        super().__init__()

        self.trajectory_feature_dim = trajectory_feature_dim
        self.segment_stats_dim = segment_stats_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.context_loss_weight = context_loss_weight
        self.temperature = temperature

        # 主路径：轨迹特征编码器
        self.traj_lstm = nn.LSTM(
            input_size=9,  # 前 9 维是轨迹特征
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.traj_attn = AttentionPooling(hidden_dim * 2)

        # 辅路径：空间上下文编码器
        self.ctx_lstm = nn.LSTM(
            input_size=12,  # 后 12 维是空间特征
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.ctx_attn = AttentionPooling(hidden_dim)

        # 投影层（用于 CCA）
        self.traj_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # 分类器
        classifier_input_dim = (hidden_dim * 2) + segment_stats_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, traj_21, segment_stats=None, return_context=False):
        """
        前向传播

        Args:
            traj_21: (B, T, 21) 点级融合特征
            segment_stats: (B, 18) 段级统计特征
            return_context: 是否返回上下文表示（用于对比损失）

        Returns:
            如果 return_context=False: logits (B, num_classes)
            如果 return_context=True: (logits, traj_z, ctx_z)
        """
        # 拆分特征
        traj_9 = traj_21[:, :, :9]  # (B, T, 9) 轨迹特征
        spatial_12 = traj_21[:, :, 9:]  # (B, T, 12) 空间特征

        # 主路径：轨迹编码
        traj_out, _ = self.traj_lstm(traj_9)  # (B, T, hidden_dim * 2)
        traj_repr = self.traj_attn(traj_out)  # (B, hidden_dim * 2)

        # 辅路径：空间上下文编码
        ctx_out, _ = self.ctx_lstm(spatial_12)  # (B, T, hidden_dim)
        ctx_repr = self.ctx_attn(ctx_out)  # (B, hidden_dim)

        # 拼接 segment_stats 后分类
        if segment_stats is not None:
            cls_input = torch.cat([traj_repr, segment_stats], dim=1)  # (B, hidden_dim * 2 + 18)
        else:
            cls_input = traj_repr

        logits = self.classifier(cls_input)  # (B, num_classes)

        if return_context:
            # 返回归一化的表示用于 InfoNCE 损失
            traj_z = F.normalize(self.traj_proj(traj_repr), dim=1)  # (B, hidden_dim)
            ctx_z = F.normalize(self.ctx_proj(ctx_repr), dim=1)  # (B, hidden_dim)
            return logits, traj_z, ctx_z

        return logits

    def compute_infonce_loss(self, traj_z, ctx_z):
        """
        计算 InfoNCE 对比损失（双向）

        Args:
            traj_z: (B, D) 轨迹表示（已归一化）
            ctx_z: (B, D) 上下文表示（已归一化）

        Returns:
            loss: InfoNCE 损失值（双向平均）
        """
        batch_size = traj_z.shape[0]

        # 计算相似度矩阵
        logits = torch.matmul(traj_z, ctx_z.T) / self.temperature  # (B, B)

        # 标签：对角线为正样本
        labels = torch.arange(batch_size, device=traj_z.device)

        # 双向 InfoNCE 损失
        loss_t2c = F.cross_entropy(logits, labels)      # traj → ctx
        loss_c2t = F.cross_entropy(logits.T, labels)    # ctx → traj

        loss = (loss_t2c + loss_c2t) / 2.0

        return loss
