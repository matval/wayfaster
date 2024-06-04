import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 3, 3), dilation=(1, 1, 1), bias=False):
        super().__init__()
        assert len(kernel_size) == 3, 'kernel_size must be a 3-tuple.'
        time_pad = (kernel_size[0] - 1) * dilation[0]
        height_pad = ((kernel_size[1] - 1) * dilation[1]) // 2
        width_pad = ((kernel_size[2] - 1) * dilation[2]) // 2

        # Pad temporally on the left
        self.conv = nn.Sequential(
            nn.ConstantPad3d(padding=(width_pad, width_pad, height_pad, height_pad, time_pad, 0), value=0),
            nn.Conv3d(in_channels, out_channels, kernel_size, dilation=dilation, stride=1, padding=0, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x
    
class SpatioTemporalPooling(nn.Module):
    """ Spatio-temporal pyramid pooling.
        Performs 3D average pooling followed by 1x1x1 convolution to reduce the number of channels and upsampling.
        Setting contains a list of kernel_size: usually it is [(2, h, w), (2, h//2, w//2), (2, h//4, w//4)]
    """

    def __init__(self, in_channels, reduction_channels, pool_size):
        super().__init__()
        self.features = []
        assert pool_size[0] == 2, ("Time kernel should be 2 as PyTorch raises an error when" "padding with more than half the kernel size")

        stride = (1, *pool_size[1:])
        padding = (pool_size[0]-1, 0, 0)
        self.feature = nn.Sequential(
            torch.nn.AvgPool3d(kernel_size=pool_size, stride=stride, padding=padding, count_include_pad=False),
            nn.Conv3d(in_channels, reduction_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(reduction_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b, _, t, h, w = x.shape
        # Remove unnecessary padded values (time dimension) on the right
        x_pool = self.feature(x)[:, :, :-1].contiguous()
        c = x_pool.shape[1]
        x_pool = F.interpolate(x_pool.view(b * t, c, *x_pool.shape[-2:]), (h, w), mode='bilinear', align_corners=False)
        x_pool = x_pool.view(b, c, t, h, w)
        return x_pool

class TemporalBlock(nn.Module):
    """ Temporal block with the following layers:
        - 2x3x3, 1x3x3, spatio-temporal pyramid pooling
        - skip connection.
    """

    def __init__(self, channels, pool_size):
        super().__init__()
        self.channels = channels
        self.half_channels = channels // 2

        self.conv1 = nn.Sequential(
            nn.Conv3d(self.channels, self.half_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.half_channels),
            nn.ReLU(inplace=True),
            CausalConv3d(self.half_channels, self.half_channels, kernel_size=(2, 3, 3)))
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(self.channels, self.half_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.half_channels),
            nn.ReLU(inplace=True),
            CausalConv3d(self.half_channels, self.half_channels, kernel_size=(1, 3, 3)))
        
        self.conv3 = nn.Sequential(
                nn.Conv3d(self.channels, self.half_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(self.half_channels),
                nn.ReLU(inplace=True))

        reduction_channels = self.channels // 3
        self.pyramid_pooling = SpatioTemporalPooling(self.channels, reduction_channels, pool_size)
        agg_channels = 3*self.half_channels + reduction_channels

        # Feature aggregation
        self.aggregation = nn.Sequential(
                    nn.Conv3d(agg_channels, self.channels, kernel_size=1, bias=False),
                    nn.BatchNorm3d(self.channels),
                    nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x_residual = torch.cat([x1, x2, x3], dim=1)
        x_pool = self.pyramid_pooling(x)
        x_residual = torch.cat([x_residual, x_pool], dim=1)
        x_residual = self.aggregation(x_residual)

        x = x + x_residual
        return x

class TemporalModel(nn.Module):
    def __init__(self, channels, temporal_length, input_shape):
        super().__init__()
        self.temporal_length = temporal_length

        h, w = input_shape
        modules = []
        for _ in range(temporal_length - 1):
            temporal = TemporalBlock(channels, pool_size=(2, h, w))
            modules.extend(nn.Sequential(temporal))

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        # Reshape input tensor to (batch, C, time, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x[:, -1, None]