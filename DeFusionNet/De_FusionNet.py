# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
import numpy as np
from DeFusionNet.Autoencoder import NestFuse_autoencoder, DoubleConv
from DeFusionNet.util import get_experiment_noise

class DeFusionNet(nn.Module):
    def __init__(self, deploy):
        super(DeFusionNet, self).__init__()
        nb_filter = [32, 64, 128, 256, 512, 1024]

        input_nc = 1
        output_nc = 1
        self.deploy = deploy
        self.autoencoder = NestFuse_autoencoder(nb_filter, input_nc, output_nc, False, 'spa', deploy=self.deploy)

        self.conva = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.convb = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.convc = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.convd = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.ReLu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.conve = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.convf = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.convg = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.convh = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, image_vis, image_ir, it):
        x_vis_origin = image_vis[:, :1]
        x_inf_origin = image_ir
        output = self.autoencoder(image_vis, image_ir)
        noisy_output = output
        a_feature_maps = self.conva(noisy_output)
        a_feature_maps = self.ReLu(a_feature_maps)
        a_feature_maps = self.convb(a_feature_maps)
        a_feature_maps = self.ReLu(a_feature_maps)
        a_feature_maps = self.convc(a_feature_maps)
        a_feature_maps = self.ReLu(a_feature_maps)
        de_vi = self.convd(a_feature_maps)
        de_vi = self.tanh(de_vi)

        b_feature_maps = self.conve(noisy_output)
        b_feature_maps = self.ReLu(b_feature_maps)
        b_feature_maps = self.convf(b_feature_maps)
        b_feature_maps = self.ReLu(b_feature_maps)
        b_feature_maps = self.convg(b_feature_maps)
        b_feature_maps = self.ReLu(b_feature_maps)
        de_ir = self.convh(b_feature_maps)
        de_ir = self.tanh(de_ir)

        return output, de_vi, de_ir