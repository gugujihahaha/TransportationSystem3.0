"""
交通方式分类器基类。

提供通用的多输入 Bi-LSTM 编码器架构，
子类只需传入各模态的特征维度即可，无需重复编写 LSTM 和融合层代码。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class BaseTransportationClassifier(nn.Module):
    """
    交通方式识别基类。

    设计思路：
        每个输入模态对应一个独立的 Bi-LSTM 编码器。
        所有编码器的最后时间步输出拼接后经过融合层，最终输出分类 logits。

    参数：
        input_dims  (List[int]): 各模态的输入特征维度，例如 [9, 15, 12]。
        hidden_dims (List[int]): 各模态 LSTM 的隐藏层维度，例如 [128, 64, 32]。
        num_layers  (int):       LSTM 层数，默认 2。
        num_classes (int):       分类类别数，默认 7（7 种交通方式）。
        dropout     (float):     Dropout 比率，默认 0.3。
    """

    def __init__(
        self,
        input_dims: List[int],
        hidden_dims: List[int],
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert len(input_dims) == len(hidden_dims), \
            "input_dims 与 hidden_dims 长度必须一致"

        self.num_classes = num_classes

        # 为每个输入模态创建独立的 Bi-LSTM 编码器
        self.encoders = nn.ModuleList([
            nn.LSTM(
                input_size=in_dim,
                hidden_size=h_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )
            for in_dim, h_dim in zip(input_dims, hidden_dims)
        ])

        # 融合层输入维度 = 各编码器输出维度之和（双向 × 2）
        fusion_in = sum(h * 2 for h in hidden_dims)
        fusion_hidden = hidden_dims[0] * 2

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(fusion_hidden // 2, num_classes)

        # Attention Pooling：对所有时间步加权求和，替代只取最后时间步
        # 每个 encoder 对应一个注意力投影
        self.attn_projections = nn.ModuleList([
            nn.Linear(hd * 2, 1)   # Bi-LSTM 输出维度 = hidden_dim * 2
            for hd in hidden_dims
        ])

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数：
            *inputs: 各模态输入张量，形状均为 (batch, seq_len, feature_dim)。
                     顺序与构造函数中 input_dims 的顺序一致。
        返回：
            logits: (batch, num_classes)
        """
        reprs = []
        for i, (encoder, x) in enumerate(zip(self.encoders, inputs)):
            out, _ = encoder(x)
            # Attention Pooling
            attn_weights = torch.softmax(
                self.attn_projections[i](out), dim=1
            )  # (batch, seq_len, 1)
            repr_i = (attn_weights * out).sum(dim=1)
            # (batch, hidden_dim * 2)
            reprs.append(repr_i)

        combined = torch.cat(reprs, dim=1)
        fused = self.fusion_layer(combined)
        return self.classifier(fused)

    def predict_proba(self, *inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推断类别和置信度（推理模式，不计算梯度）。

        返回：
            preds: (batch,) 预测类别索引
            probs: (batch, num_classes) 各类别概率
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(*inputs)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs


class HierarchicalTransportationClassifier(BaseTransportationClassifier):
    """
    层次化时序编码器

    核心思想：
        局部编码器捕捉短时运动模式（如加速、转弯），
        全局编码器捕捉运动模式的变化趋势（如 Bus 的周期性停靠）。

    架构：
        输入序列 (B, T, D)
        → 分段为 num_segments 个子序列 (B, num_segments, seg_len, D)
        → 局部 Bi-LSTM 编码每段 (B, num_segments, local_hidden*2)
        → 全局 Bi-LSTM 编码段序列 (B, global_hidden*2)
        → Attention Pooling
        → 分类器
    """

    def __init__(
        self,
        input_dims: list,
        hidden_dims: list,
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.3,
        num_segments: int = 5,
        local_hidden: int = 64,
        global_hidden: int = 128,
        segment_stats_dim: int = 0,
    ):
        super().__init__(
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.num_segments = num_segments
        self.local_hidden = local_hidden
        self.global_hidden = global_hidden
        self.segment_stats_dim = segment_stats_dim

        # 为每个输入模态建立局部编码器
        self.local_encoders = nn.ModuleList([
            nn.LSTM(
                input_size=input_dim,
                hidden_size=local_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            for input_dim in input_dims
        ])

        # 全局编码器：输入是所有模态局部表示的拼接
        global_input_dim = local_hidden * 2 * len(input_dims)
        self.global_encoder = nn.LSTM(
            input_size=global_input_dim,
            hidden_size=global_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 全局 Attention
        self.global_attn = nn.Linear(global_hidden * 2, 1)

        # 分类器输入维度 = 全局编码维度 + 静态特征维度
        classifier_input_dim = global_hidden * 2 + segment_stats_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, global_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden, num_classes)
        )

    def forward(self, *inputs: torch.Tensor,
                segment_stats: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            inputs: 每个元素形状 (batch, seq_len, input_dim_i)
            segment_stats: (batch, segment_stats_dim) 静态统计特征

        Returns:
            logits: (batch, num_classes)
        """
        batch_size = inputs[0].size(0)
        seq_len    = inputs[0].size(1)

        # 将序列分段
        seg_len = seq_len // self.num_segments
        if seg_len == 0:
            seg_len = 1

        # 局部编码：对每个模态的每个子段编码
        local_reprs = []
        for i, (inp, local_enc) in enumerate(zip(inputs, self.local_encoders)):
            # inp: (batch, seq_len, dim)
            seg_reprs = []
            for s in range(self.num_segments):
                start = s * seg_len
                end   = start + seg_len if s < self.num_segments - 1 else seq_len
                seg   = inp[:, start:end, :]
                out, _ = local_enc(seg)
                # 取每段的最后时间步
                seg_repr = out[:, -1, :]
                seg_reprs.append(seg_repr.unsqueeze(1))

            modal_repr = torch.cat(seg_reprs, dim=1)
            local_reprs.append(modal_repr)

        # 拼接所有模态的局部表示
        combined = torch.cat(local_reprs, dim=2)

        # 全局编码
        global_out, _ = self.global_encoder(combined)

        # 全局 Attention Pooling
        attn_weights = torch.softmax(self.global_attn(global_out), dim=1)
        global_repr  = (attn_weights * global_out).sum(dim=1)

        # 拼接静态统计特征
        if segment_stats is not None and self.segment_stats_dim > 0:
            global_repr = torch.cat([global_repr, segment_stats], dim=1)

        # 分类
        logits = self.classifier(global_repr)
        return logits

    def predict_proba(self, *inputs: torch.Tensor,
                     segment_stats: torch.Tensor = None):
        """预测类别和概率"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(*inputs, segment_stats=segment_stats)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(probs, dim=1)
        return preds, probs
