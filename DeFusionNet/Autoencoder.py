import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from DeFusionNet.diversebranchblock import DiverseBranchBlock
from DeFusionNet.fusion_strategy import Fusion_strategy

EPSILON = 1e-5

# NestFuse network - light, no desnse
class NestFuse_autoencoder(nn.Module):
    def __init__(self, nb_filter, input_nc=1, output_nc=1, bilinear=False, fs_type='spa', deploy=False):
        super(NestFuse_autoencoder, self).__init__()
        self.n_channels = input_nc
        self.n_classes = output_nc
        self.bilinear = bilinear
        self.fs_type = fs_type
        self.fusion = Fusion_strategy(fs_type)
        self.deploy = deploy
        #
        self.inc = VanDoubleConv(in_channels=self.n_channels, out_channels=nb_filter[0])
        self.down1 = Down(nb_filter[0], nb_filter[1], deploy=self.deploy)
        self.down2 = Down(nb_filter[1], nb_filter[2], deploy=self.deploy)
        self.down3 = Down(nb_filter[2], nb_filter[3], deploy=self.deploy)
        self.down4 = Down(nb_filter[3], nb_filter[4], deploy=self.deploy)
        self.down5 = Down(nb_filter[4], nb_filter[5], deploy=self.deploy)

        self.up0 = VanUp(nb_filter[5], nb_filter[4], bilinear=bilinear, deploy=self.deploy)
        self.up1 = VanUp(nb_filter[4], nb_filter[3], bilinear=bilinear, deploy=self.deploy)
        self.up2 = VanUp(nb_filter[3], nb_filter[2], bilinear=bilinear, deploy=self.deploy)
        self.up3 = VanUp(nb_filter[2], nb_filter[1], bilinear=bilinear, deploy=self.deploy)
        self.up4 = VanUp(nb_filter[1], nb_filter[0], bilinear=bilinear, deploy=self.deploy)
        self.outc = OutConv(nb_filter[0], self.n_classes)

    def forward(self, input1, input2):
        x = input1[:, :1]
        y = input2

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        y1 = self.inc(y)
        y2 = self.down1(y1)
        y3 = self.down2(y2)
        y4 = self.down3(y3)
        y5 = self.down4(y4)
        y6 = self.down5(y5)

        en_v = [x1, x2, x3, x4, x5, x6]
        en_r = [y1, y2, y3, y4, y5, y6]
        f = self.fusion(en_v, en_r)

        I0 = self.up0(f[5], f[4])
        I1 = self.up1(I0, f[3])
        I2 = self.up2(I1, f[2])
        I3 = self.up3(I2, f[1])
        I4 = self.up4(I3, f[0])

        logits = self.outc(I4)

        return logits

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, deploy=False):
        super().__init__()
        self.deploy = deploy
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=in_channels // 2, deploy=self.deploy)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, deploy=self.deploy)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class VanUp(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, deploy=False):
        super().__init__()
        self.deploy = deploy

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = VanDoubleConv(in_channels=in_channels, out_channels=out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = VanDoubleConv(in_channels=in_channels, out_channels=out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, deploy=False):
        super().__init__()
        self.deploy = deploy
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DiverseBranchBlock(in_channels, mid_channels, kernel_size=3, padding=1, groups=1, deploy=self.deploy, nonlinear=nn.ReLU(inplace=True)),
            DiverseBranchBlock(mid_channels, out_channels, kernel_size=3, padding=1, groups=1, deploy=self.deploy, nonlinear=nn.ReLU(inplace=True)),
        )
    def forward(self, x):
        return self.double_conv(x)


class VanDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
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
        y = self.double_conv(x)
        return y


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, deploy=self.deploy)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class VanDown(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            VanDoubleConv(in_channels=in_channels, out_channels=out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.conv(x)