# coding:utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from util import *
import torch
from torch.utils.data import DataLoader
from datasets import Fusion_dataset
from DeFusionNet.De_FusionNet import DeFusionNet
from convnet_utils import switch_deploy_flag, switch_conv_bn_impl
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
import torch.nn.functional
import cv2
import time
def guidedfilter(srcImg, guidedImg, rad=9,eps=0.01):

    srcImg=srcImg/255.0
    guidedImg=guidedImg/255.0
    img_shape=np.shape(srcImg)

    P_mean = cv2.boxFilter(srcImg, -1, (rad, rad), normalize=True)
    I_mean = cv2.boxFilter(guidedImg, -1, (rad, rad), normalize=True)

    I_square_mean = cv2.boxFilter(np.multiply(guidedImg, guidedImg), -1, (rad, rad), normalize=True)
    I_mul_P_mean = cv2.boxFilter(np.multiply(srcImg, guidedImg), -1, (rad, rad), normalize=True)

    var_I = I_square_mean - np.multiply(I_mean, I_mean)
    cov_I_P = I_mul_P_mean - np.multiply(I_mean, P_mean)

    a = cov_I_P / (var_I + eps)
    b = P_mean - np.multiply(a, I_mean)

    a_mean = cv2.boxFilter(a, -1, (rad, rad), normalize=True)
    b_mean = cv2.boxFilter(b, -1, (rad, rad), normalize=True)

    dstImg = np.multiply(a_mean, guidedImg) + b_mean
    # h = srcImg-dstImg

    return dstImg * 255.0
    
def main(ir_dir='./test_imgs/noiseir30', vi_dir='./test_imgs/noisevi30', save_dir='./SeAFusion',
         fusion_model_path='./model/Fusion/fusion_model1.pth'):
    print(fusion_model_path)
    save_dir1='./Inf'
    save_dir2 = './Vis'
    fusionmodel = torch.nn.DataParallel(DeFusionNet(True), device_ids=[args.gpu])
    # fusionmodel = model()
    # fusionmodel.eval()
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    # fusionmodel.load_state_dict(
    #     {k.replace('module.', ''): v for k, v in torch.load(fusion_model_path, map_location='cpu').items()})
    fusionmodel = fusionmodel.to(device)
    print('fusionmodel load done!')
    test_dataset = Fusion_dataset('val', ir_path=ir_dir, vi_path=vi_dir)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    time_list = []
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for it, (img_vis, img_ir, name) in enumerate(test_bar):
            start = time.time()
            img_vis = img_vis.to(device)
            img_ir = img_ir.to(device)
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vis)
            fused_image_ycrcb = torch.cat((vi_Y, vi_Cr,vi_Cb),dim=1)
            fused_img, _, _ = fusionmodel(fused_image_ycrcb, img_ir, it)
            fused_img = fused_img.squeeze(dim=0).squeeze(dim=0)
            vi_Cb = vi_Cb.squeeze(dim=0).squeeze(dim=0)
            vi_Cr = vi_Cr.squeeze(dim=0).squeeze(dim=0)
            vi_Cb = vi_Cb.cpu().numpy()
            vi_Cr = vi_Cr.cpu().numpy()
            fused_img = fused_img.cpu().numpy()

            denoising_vi_cb = guidedfilter(vi_Cb, fused_img, 9, 0.01)
            denoising_vi_cr = guidedfilter(vi_Cr, fused_img, 9, 0.01)
            fused_img = torch.from_numpy(fused_img).float()
            denoising_vi_cb = torch.from_numpy(denoising_vi_cb).float()
            denoising_vi_cr = torch.from_numpy(denoising_vi_cr).float()
            fused_img = fused_img.unsqueeze(dim=0).unsqueeze(dim=0)
            denoising_vi_cb = denoising_vi_cb.unsqueeze(dim=0).unsqueeze(dim=0)
            denoising_vi_cr = denoising_vi_cr.unsqueeze(dim=0).unsqueeze(dim=0)


            fused_img = YCbCr2RGB(fused_img, denoising_vi_cb, denoising_vi_cr)
            end = time.time()
            time_list.append(end - start)
            for k in range(len(name)):
                img_name = name[k]
                save_path = os.path.join(save_dir, img_name)
                save_img_single(fused_img[k, ::], save_path)
                test_bar.set_description('Fusion {0} Sucessfully!'.format(name[k]))
    print(time_list)
def convert(args):
    switch_conv_bn_impl('DBB')
    switch_deploy_flag(False)
    fusionmodel = torch.nn.DataParallel(DeFusionNet(False), device_ids=[args.gpu])
    fusionmodel.load_state_dict(torch.load(args.model_path))
    for m in fusionmodel.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()
    torch.save(fusionmodel.state_dict(), args.model_path_final)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
    ## model
    parser.add_argument('--model_path', '-M', type=str, default='./model/Fusion/fusion_model.pth')
    parser.add_argument('--model_path_final', type=str, default='./model/Fusion/fusion_model_final.pth')
    ## dataset
    parser.add_argument('--blocktype', type=str, default='DBB')
    parser.add_argument('--ir_dir', '-ir_dir', type=str, default='./Impulse_MSRS/Noise_ir')
    parser.add_argument('--vi_dir', '-vi_dir', type=str, default='./Impulse_MSRS/Noise_vi')
    parser.add_argument('--save_dir', '-save_dir', type=str, default='./SeAFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--Train', type=bool, default=False)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    parser.add_argument('--mode', type=str,  default='deploy', help='train or deploy')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print('| testing %s on GPU #%d with pytorch' % ('SeAFusion', args.gpu))
    convert(args)
    main(ir_dir=args.ir_dir, vi_dir=args.vi_dir, save_dir=args.save_dir, fusion_model_path=args.model_path_final)


