import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet18, resnet34

class UpsamplingConcat(nn.Module):
    """
    Module for upsampling and concatenating feature maps.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int, optional): Scaling factor for upsampling. Defaults to 2.
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x_to_upsample, x):
        """
        Forward pass of the UpsamplingConcat module.

        Args:
            x_to_upsample (torch.Tensor): Tensor to be upsampled.
            x (torch.Tensor): Tensor to be concatenated with the upsampled tensor.

        Returns:
            torch.Tensor: The resulting tensor after upsampling, concatenation, and convolution.
        """
        x_to_upsample = self.upsample(x_to_upsample)
        diffY = x.size()[2] - x_to_upsample.size()[2]
        diffX = x.size()[3] - x_to_upsample.size()[3]
        x_to_upsample = F.pad(x_to_upsample, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

class UpsamplingAdd(nn.Module):
    """
    Module for upsampling and adding feature maps.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int, optional): Scaling factor for upsampling. Defaults to 2.
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x, x_skip):
        """
        Forward pass of the UpsamplingAdd module.

        Args:
            x (torch.Tensor): Tensor to be upsampled.
            x_skip (torch.Tensor): Tensor to be added to the upsampled tensor.

        Returns:
            torch.Tensor: The resulting tensor after upsampling and addition.
        """
        x = self.upsample_layer(x)
        diffY = x_skip.size()[2] - x.size()[2]
        diffX = x_skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x + x_skip

class Decoder(nn.Module):
    """
    Decoder module for feature extraction and upsampling.

    Args:
        in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels):
        super().__init__()
        backbone = resnet18(pretrained=False, zero_init_residual=True)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, in_channels, scale_factor=2)

    def forward(self, x):
        """
        Forward pass of the Decoder module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The resulting tensor after decoding and upsampling.
        """
        # (H, W)
        skip_1 = x
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        # (H/4, W/4)
        skip_2 = self.layer1(x)
        x = self.layer2(skip_2)
        # (H/8, W/8)
        skip_3 = x
        x = self.layer3(skip_3)
        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_3)            

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_2)
        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_1)

        return x

class Encoder(nn.Module):
    """
    Encoder module for feature extraction and downsampling.

    Args:
        C (int): Number of output channels.
        downsample (int, optional): Downsampling factor. Defaults to 8.
    """
    def __init__(self, C, downsample=8):
        super().__init__()
        self.C = C
        self.downsample = downsample

        print('Using Resnet34')
        resnet = resnet34(pretrained=True)
        c0 = 64
        c1 = 128
        c2 = 256

        if downsample == 8:
            self.backbone = nn.Sequential(*list(resnet.children())[:-4])
            self.layer = resnet.layer3
            self.upsampling_layer = UpsamplingConcat(c2+c1, c1)
            self.depth_layer = nn.Conv2d(c1, self.C, kernel_size=1, padding=0)
        elif downsample == 4:
            self.backbone = nn.Sequential(*list(resnet.children())[:-5])
            self.layer1 = resnet.layer2
            self.layer2 = resnet.layer3
            self.upsampling_layer1 = UpsamplingConcat(c2+c1, c1)
            self.upsampling_layer2 = UpsamplingConcat(c1+c0, c0)
            self.depth_layer = nn.Conv2d(c0, self.C, kernel_size=1, padding=0)
        else:
            print('Downsample {} not implemented'.format(downsample))
            sys.exit(1)

    def forward(self, x):
        """
        Forward pass of the Encoder module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The resulting tensor after encoding and upsampling.
        """
        x1 = self.backbone(x)
        if self.downsample == 8:
            x = self.layer(x1)
            x = self.upsampling_layer(x, x1)
        elif self.downsample == 4:
            x2 = self.layer1(x1)
            x = self.layer2(x2)
            x = self.upsampling_layer1(x, x2)
            x = self.upsampling_layer2(x, x1)
        x = self.depth_layer(x)

        return x