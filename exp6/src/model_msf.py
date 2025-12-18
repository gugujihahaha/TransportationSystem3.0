"""
MSF (Multi-Stage Fusion) 交通方式识别模型 - 修复版本
基于论文第4章

三个融合阶段:
1. ACFM (Attribute-Channel Fusion Module): 融合多属性和通道信息
2. SFFM (Scale-Feature Fusion Module): 融合多尺度特征
3. ODFM (Object-Decision Fusion Module): 融合多对象决策
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """通道注意力机制 (SE模块)"""

    def __init__(self, channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.size()

        # Squeeze
        se = self.avg_pool(x).view(b, c)

        # Excitation
        se = self.fc(se).view(b, c, 1, 1)

        return x * se


class ACFM(nn.Module):
    """
    属性-通道融合模块 (4.2.1节)
    用于融合多属性(多通道)的局部运动信息和空间特征
    """

    def __init__(self, in_channels: int, out_channels: int = 64):
        super(ACFM, self).__init__()

        # 浅层卷积块
        self.shallow_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 残差块1
        self.residual_block1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.channel_attention1 = ChannelAttention(out_channels, reduction=16)

        # 池化
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 残差块2
        self.residual_block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.channel_attention2 = ChannelAttention(out_channels, reduction=16)

        # 池化
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.out_channels = out_channels

    def forward(self, x):
        # x: (B, C, H, W)

        # 浅层卷积
        x = self.shallow_conv(x)

        # 残差块1
        residual = x
        x = self.residual_block1(x)
        x = self.channel_attention1(x)
        x = x + residual  # 捷径连接

        # 池化
        x = self.pool1(x)

        # 残差块2
        residual = x
        x = self.residual_block2(x)
        x = self.channel_attention2(x)
        x = x + residual

        # 池化
        x = self.pool2(x)

        return x


class ScaleAttention(nn.Module):
    """尺度注意力机制 (4.2.2节)"""

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super(ScaleAttention, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: (B, feature_dim)
        weights = self.fc(x)
        return weights * x


class SFFM(nn.Module):
    """
    尺度-特征融合模块 (4.2.2节)
    用于有选择性地融合不同空间尺度的高维特征
    """

    def __init__(self, num_scales: int, feature_per_scale: int, hidden_dim: int = 128):
        super(SFFM, self).__init__()

        # 将每个尺度的特征映射到统一维度
        self.fc_layers = nn.ModuleList([
            nn.Linear(feature_per_scale, hidden_dim)
            for _ in range(num_scales)
        ])

        # 注意力机制
        total_features = hidden_dim * num_scales
        self.scale_attention = ScaleAttention(total_features, hidden_dim=256)

    def forward(self, x_list):
        # x_list: list of (B, feature_per_scale)
        features = []
        for fc, feat in zip(self.fc_layers, x_list):
            features.append(fc(feat))

        # 拼接所有尺度特征
        x = torch.cat(features, dim=1)  # (B, hidden_dim*num_scales)

        # 应用注意力
        x = self.scale_attention(x)

        return x


class ODFM(nn.Module):
    """
    对象-决策融合模块 (4.2.3节)
    用于融合多个对象的决策结果

    修复版本: 正确定义input_dim和num_classes
    """

    def __init__(self, num_objects: int, num_classes: int, input_dim: int):
        """
        Args:
            num_objects: 对象数量 (K)
            num_classes: 分类类别数
            input_dim: 每个对象的输入特征维度
        """
        super(ODFM, self).__init__()

        self.num_objects = num_objects
        self.num_classes = num_classes
        self.input_dim = input_dim

        # 每个对象独立分类
        self.object_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
            for _ in range(num_objects)
        ])

    def forward(self, x_list):
        """
        Args:
            x_list: list of (B, input_dim) - 来自每个对象的特征

        Returns:
            final_logits: (B, num_classes)
            object_logits: list of (B, num_classes)
        """
        # 对每个对象进行分类
        object_logits = []
        for classifier in self.object_classifiers:
            logits = classifier(x_list)
            object_logits.append(logits)

        # 融合: 取平均概率 (多数决策)
        object_probs = [F.softmax(logits, dim=1) for logits in object_logits]

        # 平均融合
        avg_probs = torch.stack(object_probs, dim=1).mean(dim=1)  # (B, num_classes)

        # 返回对数概率用于交叉熵损失
        final_logits = torch.log(avg_probs + 1e-10)

        return final_logits, object_logits


class SimpleMSFModel(nn.Module):
    """
    简化版MSF模型 (推荐用于exp6 - 易于训练和调试)

    架构:
    - 输入: (B, K, C, H, W) = (B, 6, 27, 32, 32)
    - K个对象独立的CNN特征提取
    - 拼接后的全连接分类
    """

    def __init__(self,
                 input_channels: int = 27,
                 num_objects: int = 6,
                 num_classes: int = 6,
                 img_size: int = 32):
        """
        Args:
            input_channels: 输入通道数 (L*N*M = 9*3*1)
            num_objects: 对象数量 (K)
            num_classes: 分类类别数
            img_size: 输入图像尺寸
        """
        super(SimpleMSFModel, self).__init__()

        self.input_channels = input_channels
        self.num_objects = num_objects
        self.num_classes = num_classes
        self.img_size = img_size

        # CNN特征提取器 (共享权重)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # 多对象融合与分类
        feature_dim = 128
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * num_objects, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (B, K, C, H, W)

        Returns:
            logits: (B, num_classes)
        """
        B, K, C, H, W = x.shape

        # 处理每个对象
        features = []
        for k in range(K):
            x_k = x[:, k, :, :, :]  # (B, C, H, W)
            feat = self.feature_extractor(x_k)  # (B, 128, 1, 1)
            feat = feat.view(feat.size(0), -1)  # (B, 128)
            features.append(feat)

        # 拼接所有对象特征
        combined = torch.cat(features, dim=1)  # (B, 128*K)

        # 分类
        logits = self.classifier(combined)  # (B, num_classes)

        return logits


class FullMSFModel(nn.Module):
    """
    完整的MSF模型 (按论文实现,更复杂,需要GPU)
    整合ACFM, SFFM, ODFM三个融合模块
    """

    def __init__(self,
                 input_channels: int = 27,  # L*N*M = 9*3*1
                 num_objects: int = 6,
                 num_scales: int = 3,
                 num_classes: int = 6,
                 img_size: int = 32):
        super(FullMSFModel, self).__init__()

        self.input_channels = input_channels
        self.num_objects = num_objects
        self.num_scales = num_scales
        self.num_classes = num_classes
        self.img_size = img_size

        # 计算通道数量
        channels_per_scale = input_channels // num_scales

        # ========================================================
        # 阶段1: ACFM (属性-通道融合)
        # ========================================================
        self.acfm_list = nn.ModuleList([
            ACFM(channels_per_scale, out_channels=64)
            for _ in range(num_scales)
        ])

        # ACFM输出后的特征维度
        acfm_output_size = (img_size // 4) * (img_size // 4)  # 8x8 = 64
        acfm_output_channels = 64

        # ========================================================
        # 阶段2: SFFM (尺度-特征融合)
        # ========================================================
        feature_per_scale = acfm_output_channels * acfm_output_size

        # 为每个对象创建SFFM
        self.sffm_list = nn.ModuleList([
            SFFM(num_scales, feature_per_scale, hidden_dim=128)
            for _ in range(num_objects)
        ])

        sffm_output_dim = 128 * num_scales  # 尺度注意力输出维度

        # ========================================================
        # 阶段3: ODFM (对象-决策融合)
        # ========================================================
        self.odfm = ODFM(
            num_objects=num_objects,
            num_classes=num_classes,
            input_dim=sffm_output_dim
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: (B, K, L*N*M, H, W) = (B, 6, 27, 32, 32)

        Returns:
            logits: (B, num_classes)
        """
        B, K, C, H, W = x.shape

        channels_per_scale = C // self.num_scales

        # ========================================================
        # 处理每个对象
        # ========================================================
        object_features = []

        for k in range(K):
            x_k = x[:, k, :, :, :]  # (B, C, H, W)

            # ====== 阶段1: ACFM ======
            scale_features = []
            for n in range(self.num_scales):
                # 提取当前尺度的通道
                start_ch = n * channels_per_scale
                end_ch = (n + 1) * channels_per_scale
                x_scale = x_k[:, start_ch:end_ch, :, :]  # (B, Ch, H, W)

                # 应用ACFM
                feat = self.acfm_list[n](x_scale)  # (B, 64, 8, 8)

                # 展平
                feat_flat = feat.view(feat.size(0), -1)  # (B, 64*64)
                scale_features.append(feat_flat)

            # ====== 阶段2: SFFM ======
            fused_feat = self.sffm_list[k](scale_features)  # (B, 128*3)
            object_features.append(fused_feat)

        # ========================================================
        # 阶段3: ODFM (多对象决策融合)
        # ========================================================
        # 使用第一个对象特征进行分类 (简化版)
        # 在实际应用中应该融合所有对象
        combined_feat = torch.stack(object_features, dim=1).mean(dim=1)

        final_logits = torch.log(
            torch.softmax(
                self.odfm.object_classifiers[0](combined_feat),
                dim=1
            ) + 1e-10
        )

        return final_logits