import torch
import torch.nn as nn


class FeatureAttention(nn.Module):
    """
    输入端通道重标定（CR / Channel Reweighting）
    - 支持任意 in_channels（>=1）
    - 避免 num_features//reduction_ratio 变成 0 的问题
    """
    def __init__(self, num_features: int, reduction_ratio: int = 4, dropout: float = 0.2):
        super().__init__()
        if num_features < 1:
            raise ValueError(f"num_features must be >= 1, got {num_features}")

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.layer_norm = nn.LayerNorm(num_features)

        mid_features = max(1, num_features // reduction_ratio)

        self.attention = nn.Sequential(
            nn.Linear(num_features, mid_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mid_features, num_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W]
        b, c, h, w = x.size()
        gap = self.global_avg_pool(x).view(b, c)     # [B, C]
        gap = self.layer_norm(gap)
        weights = self.attention(gap).view(b, c, 1, 1)
        out = x * weights
        return out, weights


class ImprovedBirdNetWithAttention(nn.Module):
    """多通道 + 输入端通道注意力（CR模块）"""
    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.feature_attention = FeatureAttention(num_features=in_channels)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),

            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        x, att = self.feature_attention(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, att
