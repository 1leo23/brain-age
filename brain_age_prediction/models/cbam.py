import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    1) Channel Attention
    2) Spatial Attention
    """
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.channel_attention = ChannelAttention(in_channels, reduction)
        # 空間注意力
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # 通道注意力
        x_ca = self.channel_attention(x)
        x = x * x_ca
        # 空間注意力
        x_sa = self.spatial_attention(x)
        x = x * x_sa
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # 平均池化
        avg = self.avg_pool(x).view(b, c)
        avg = self.mlp(avg)
        # 最大池化
        mx  = self.max_pool(x).view(b, c)
        mx  = self.mlp(mx)
        # 相加後通過 sigmoid
        ca = self.sigmoid(avg + mx).view(b, c, 1, 1)
        return ca

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 通過 7x7 卷積
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size//2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # 在通道維度做平均 & 最大池化，拼接
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        sa = self.sigmoid(self.conv(cat))
        return sa
