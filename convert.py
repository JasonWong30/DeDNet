import argparse
import os
import torch
from convnet_utils import switch_conv_bn_impl, switch_deploy_flag
from DeFusionNet.De_FusionNet import DeFusionNet

parser = argparse.ArgumentParser(description='DBB Conversion')
parser.add_argument('--load', default='./model/Fusion/fusion_model30.pth', help='path to the weights file')
parser.add_argument('--save', default='./model/Fusion/fusion_model_final.pth', help='path to the weights file')
parser.add_argument('--gpu', '-G', type=int, default=3)
def convert():
    args = parser.parse_args()

    switch_conv_bn_impl('DBB')
    switch_deploy_flag(False)
    fusionmodel = torch.nn.DataParallel(DeFusionNet(False), device_ids=[args.gpu])
    fusionmodel.load_state_dict(torch.load(args.load))

    for m in fusionmodel.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()

    torch.save(fusionmodel.state_dict(), args.save)

if __name__ == '__main__':
    convert()

