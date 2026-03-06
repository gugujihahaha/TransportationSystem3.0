"""
深度学习模型 (Exp5 - 弱监督上下文表示增强)
核心思想：GTA-Seg - 上下文特征仅用于改善轨迹编码器表示，不参与分类决策

与Exp4的关键区别：
- Exp4: trajectory + spatial + weather 硬拼接 → classifier
- Exp5: trajectory → classifier，spatial+weather 仅作为 context encoder 约束
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class WeaklySupervisedContextModel(nn.Module):
    """
    跨模态对比学习模型 (Exp5 - 论文核心创新)

    与 Exp4 的关键区别：
        Exp4: traj + spatial + weather 硬拼接 → 分类器
        Exp5: traj → 分类器（主路径）
              spatial + weather → 上下文编码器（辅助路径）
              两路径用 InfoNCE loss 在表示空间对齐

    理论依据：
        同一轨迹段的轨迹特征和上下文特征描述同一物理事件，
        在表示空间中应该比其他轨迹段的表示更相似（对比学习假设）。
        这种软约束比 MSE 更合理，因为不要求两个表示数值相等，
        只要求相对距离关系正确。

    论文中的名称：
        Cross-modal Contrastive Alignment (CCA) loss
    """

    def __init__(self,
                 trajectory_feature_dim: int = 9,
                 spatial_feature_dim: int = 15,
                 weather_feature_dim: int = 12,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 7,
                 dropout: float = 0.3,
                 context_loss_type: str = 'infonce',
                 context_loss_weight: float = 0.05,
                 temperature: float = 0.07):
        super().__init__()

        self.trajectory_feature_dim = trajectory_feature_dim
        self.spatial_feature_dim    = spatial_feature_dim
        self.weather_feature_dim    = weather_feature_dim
        self.hidden_dim             = hidden_dim
        self.num_classes            = num_classes
        self.context_loss_type      = context_loss_type
        self.context_loss_weight    = context_loss_weight
        self.temperature            = temperature

        self.trajectory_lstm = nn.LSTM(
            input_size=trajectory_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.traj_attn = nn.Linear(hidden_dim * 2, 1)

        self.traj_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.spatial_lstm = nn.LSTM(
            input_size=spatial_feature_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.spatial_attn = nn.Linear(hidden_dim, 1)

        self.weather_lstm = nn.LSTM(
            input_size=weather_feature_dim,
            hidden_size=hidden_dim // 4,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.weather_attn = nn.Linear(hidden_dim // 2, 1)

        context_dim = hidden_dim + hidden_dim // 2
        self.context_fusion = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def _attention_pool(self, lstm_out: torch.Tensor,
                        attn_layer: nn.Linear) -> torch.Tensor:
        attn_weights = torch.softmax(attn_layer(lstm_out), dim=1)
        return (attn_weights * lstm_out).sum(dim=1)

    def forward(self, trajectory_features: torch.Tensor,
                spatial_features: torch.Tensor,
                weather_features: torch.Tensor,
                return_context: bool = False):
        traj_out, _   = self.trajectory_lstm(trajectory_features)
        traj_repr     = self._attention_pool(traj_out, self.traj_attn)

        spatial_out, _ = self.spatial_lstm(spatial_features)
        spatial_repr   = self._attention_pool(spatial_out, self.spatial_attn)

        weather_out, _ = self.weather_lstm(weather_features)
        weather_repr   = self._attention_pool(weather_out, self.weather_attn)

        context_combined = torch.cat([spatial_repr, weather_repr], dim=1)
        context_repr     = self.context_fusion(context_combined)

        logits = self.classifier(traj_repr)

        if return_context:
            traj_z    = F.normalize(self.traj_proj(traj_repr), dim=1)
            context_z = F.normalize(self.context_proj(context_repr), dim=1)
            return logits, traj_z, context_z
        else:
            return logits

    def compute_context_loss(self, traj_z: torch.Tensor,
                             context_z: torch.Tensor) -> torch.Tensor:
        B = traj_z.size(0)
        if B < 2:
            return F.mse_loss(traj_z, context_z)

        sim_matrix = torch.matmul(traj_z, context_z.T) / self.temperature

        labels = torch.arange(B, device=traj_z.device)

        loss_t2c = F.cross_entropy(sim_matrix, labels)
        loss_c2t = F.cross_entropy(sim_matrix.T, labels)

        return (loss_t2c + loss_c2t) / 2.0

    def predict(self, trajectory_features, spatial_features, weather_features):
        self.eval()
        with torch.no_grad():
            logits = self.forward(trajectory_features, spatial_features,
                                  weather_features, return_context=False)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs


class TransportationModeClassifierExp5(nn.Module):
    """Exp5兼容接口（保持与Exp4相同的接口）"""
    def __init__(self, *args, **kwargs):
        super(TransportationModeClassifierExp5, self).__init__()
        self.model = WeaklySupervisedContextModel(*args, **kwargs)

    def forward(self, trajectory_features: torch.Tensor,
                spatial_features: torch.Tensor,
                weather_features: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.model.forward(trajectory_features, spatial_features, weather_features, return_context=True)
        return logits

    def predict_proba(self, trajectory_features: torch.Tensor,
                     spatial_features: torch.Tensor,
                     weather_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.predict_proba(trajectory_features, spatial_features, weather_features)
