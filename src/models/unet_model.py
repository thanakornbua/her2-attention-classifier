"""
U-Net for semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block (Conv + ReLU + Conv + ReLU)."""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block (MaxPool + DoubleConv)."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block (Upsample + DoubleConv)."""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        # After concatenation, we'll have in_channels + skip_channels
        # For standard U-Net, skip has in_channels // 2
        # So total after concat = in_channels // 2 (upsampled) + in_channels // 2 (skip) = in_channels
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After upsampling, channels remain in_channels
            # After concat with skip (in_channels // 2), total = in_channels + in_channels // 2
            self.conv = DoubleConv(in_channels + in_channels // 2, out_channels)
        else:
            # ConvTranspose2d reduces channels to in_channels // 2
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # After concat with skip (in_channels // 2), total = in_channels
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad x1 if necessary
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate and convolve
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.
    
    References:
        Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        bilinear: bool = True,
        base_channels: int = 64
    ):
        """
        Args:
            in_channels: Number of input channels
            num_classes: Number of segmentation classes
            bilinear: Use bilinear upsampling
            base_channels: Number of channels at first layer
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        
        # Output
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits
