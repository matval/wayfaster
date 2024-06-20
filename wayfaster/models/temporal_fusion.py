import torch
import torch.nn as nn
import torch.nn.functional as F
    
class SpatioTemporalPooling(nn.Module):
    def __init__(self, in_channels, reduction_channels, pool_size):
        """
        Initialize the SpatioTemporalPooling layer.

        Args:
            in_channels (int): Number of input channels.
            reduction_channels (int): Number of output channels after reduction.
            pool_size (tuple): Pooling kernel size.
        """
        super().__init__()
        self.features = []

        stride = (1, *pool_size[1:])
        padding = (pool_size[0]-1, 0, 0)
        self.feature = nn.Sequential(
            torch.nn.AvgPool3d(kernel_size=pool_size, stride=stride, padding=padding, count_include_pad=False),
            nn.Conv3d(in_channels, reduction_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(reduction_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b, _, t, h, w = x.shape
        x_pool = self.feature(x)[:, :, :-1].contiguous()
        c = x_pool.shape[1]
        x_pool = F.interpolate(x_pool.view(b * t, c, *x_pool.shape[-2:]), (h, w), mode='bilinear', align_corners=False)
        x_pool = x_pool.view(b, c, t, h, w)
        return x_pool

class TemporalBlock(nn.Module):
    def __init__(self, channels, pool_size):
        """
        Initialize the TemporalBlock layer.

        Args:
            channels (int): Number of input channels.
            pool_size (tuple): Pooling kernel size.
        """
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(channels, channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels // 2),
            nn.ReLU(inplace=True),
            nn.ConstantPad3d(padding=(1, 1, 1, 1, 1, 0), value=0),
            nn.Conv3d(channels // 2, channels // 2, kernel_size=(2, 3, 3), bias=False),
            nn.BatchNorm3d(channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
                nn.Conv3d(channels, channels // 2, kernel_size=1, bias=False),
                nn.BatchNorm3d(channels // 2),
                nn.ReLU(inplace=True))

        reduction_channels = channels // 3
        self.pyramid_pooling = SpatioTemporalPooling(channels, reduction_channels, pool_size)
        agg_channels = 2 * (channels // 2) + reduction_channels

        # Feature aggregation
        self.aggregation = nn.Sequential(
                    nn.Conv3d(agg_channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm3d(channels),
                    nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x_residual = torch.cat([x1, x2], dim=1)
        x_pool = self.pyramid_pooling(x)
        x_residual = torch.cat([x_residual, x_pool], dim=1)
        x_residual = self.aggregation(x_residual)

        x = x + x_residual
        return x

class TemporalModel(nn.Module):
    def __init__(self, channels, temporal_length, input_shape):
        """
        Initialize the TemporalModel layer.

        Args:
            channels (int): Number of input channels.
            temporal_length (int): Length of the temporal dimension.
            input_shape (tuple): Shape of the input tensor (height, width).
        """
        super().__init__()
        h, w = input_shape
        modules = []
        for _ in range(temporal_length - 1):
            temporal = TemporalBlock(channels, pool_size=(2, h, w))
            modules.extend(nn.Sequential(temporal))

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x[:, -1, None]