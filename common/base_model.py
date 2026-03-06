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
        for encoder, x in zip(self.encoders, inputs):
            out, _ = encoder(x)
            reprs.append(out[:, -1, :])  # 取最后时间步，shape: (batch, hidden*2)

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
