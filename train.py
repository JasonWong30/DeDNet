#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0
from PIL import Image
import numpy as np
from torch.autograd import Variable
from DeFusionNet.De_FusionNet import DeFusionNet
from PPRNet.net2 import MODEL as model
from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
from Decompose.decomNet import DecomNet as DecomNet
from decomposedataset import Decom_dataset
from logger import setup_logger
from model_TII import BiSeNet
from cityscapes import CityScapes
from loss2 import OhemCELoss, Fusionloss
from optimizer import Optimizer
import torch
from torch.utils.data import DataLoader
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc

warnings.filterwarnings('ignore')


def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def train_fusion(logger=None):
    # num: control the segmodel
    train_de_losses = []
    train_seg_losses = []
    train_fusion_losses = []

    lr_start = 0.001  # 0.001
    modelpth = './model'
    Method = 'Fusion'
    modelpth = os.path.join(modelpth, Method)
    fusionmodel = eval('DeFusionNet')(False)
    
    fusionmodel.cuda()
    fusionmodel = torch.nn.DataParallel(fusionmodel, device_ids=[0, 1, 2, 3])

    fusionmodel.train()
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)

    n_classes = 9
    segmodel = BiSeNet(n_classes=n_classes)
    save_pth = osp.join(modelpth, 'model_final.pth')
    segmodel.load_state_dict(torch.load(save_pth))
    segmodel.cuda()
    segmodel = torch.nn.DataParallel(segmodel, device_ids=[0, 1, 2, 3])
    segmodel.eval()
    for p in segmodel.parameters():
        p.requires_grad = False
    print('Load Segmentation Model {} Sucessfully~'.format(save_pth))

    train_dataset = Fusion_dataset('train')
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,  # 8 a
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)

    score_thres = 0.7
    ignore_idx = 255
    n_min = 8 * 640 * 480 // 8
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    criteria_fusion = Fusionloss()

    epoch = 300  # 10
    print(epoch)
    st = glob_st = time.time()
    logger.info('Training Fusion Model start~')
    for epo in range(0, epoch):
        # print('\n| epo #%s begin...' % epo)
        loss_seg = 0
        loss_decom = 0
        loss_fus = 0

        lr_start = 0.001  # 0.001
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo

        for it, (image_vis, image_noisevis, image_ir, image_noiseir, label, name) in tqdm(enumerate(train_loader), total=len(train_loader)):

            fusionmodel.train()
            image_vis = Variable(image_vis).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_noisevis = Variable(image_noisevis).cuda()
            image_noisevis_ycrcb = RGB2YCrCb(image_noisevis)
            image_ir = Variable(image_ir).cuda()
            image_noiseir = Variable(image_noiseir).cuda()
            label = Variable(label).cuda()

           
            logits, de_vi, de_ir = fusionmodel(image_noisevis_ycrcb, image_noiseir, it)
            
            fusion_ycrcb = torch.cat((logits, image_vis_ycrcb[:, 1:2, :, :],
            image_vis_ycrcb[:, 2:, :, :]), dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
           
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)

            lb = torch.squeeze(label, 1)
            optimizer.zero_grad()
            # seg loss

            de_vi_ycrcb = torch.cat(
                (de_vi, image_vis_ycrcb[:, 1:2, :, :],
                 image_vis_ycrcb[:, 2:, :, :]), dim=1,
            )
            de_vi_rgb = YCrCb2RGB(de_vi_ycrcb)
            de_ir_rgb = torch.cat((de_ir, de_ir, de_ir), dim=1)
            ir_rgb = torch.cat((image_ir, image_ir, image_ir), dim=1)

            out_1, mid_1 = segmodel(de_vi_rgb)
            lossp_1 = criteria_p(out_1, lb)
            loss2_1 = criteria_16(mid_1, lb)
            de_vi_seg_loss = lossp_1 + 0.1 * loss2_1

            out_2, mid_2 = segmodel(image_vis)
            lossp_2 = criteria_p(out_2, lb)
            loss2_2 = criteria_16(mid_2, lb)
            vi_seg_loss = lossp_2 + 0.1 * loss2_2

            out_3, mid_3 = segmodel(de_ir_rgb)
            lossp_3 = criteria_p(out_3, lb)
            loss2_3 = criteria_16(mid_3, lb)
            de_ir_seg_loss = lossp_3 + 0.1 * loss2_3

            out_4, mid_4 = segmodel(ir_rgb)
            lossp_4 = criteria_p(out_4, lb)
            loss2_4 = criteria_16(mid_4, lb)
            ir_seg_loss = lossp_4 + 0.1 * loss2_4

            cons_loss = torch.abs(vi_seg_loss - de_vi_seg_loss) + torch.abs(ir_seg_loss - de_ir_seg_loss)

            out, mid = segmodel(fusion_image)
            lossp = criteria_p(out, lb)
            loss2 = criteria_16(mid, lb)
            seg_loss = lossp + 0.1 * loss2
          
            loss_fusion, loss_in, loss_grad, loss_per1, loss_per2 = criteria_fusion(
                image_vis_ycrcb, image_ir, de_vi, de_ir, logits
            )

            loss_total = loss_fusion + seg_loss
          
            loss_seg = seg_loss.item() + loss_seg
            loss_decom = loss_per1.item() + loss_per2.item() + cons_loss.item() + loss_decom
            loss_fus = loss_in.item() + loss_grad.item() + loss_fus

            loss_total.backward()
            optimizer.step()
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it) * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % 10 == 0:
                loss_seg = seg_loss.item()
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'loss_per1: {loss_per1:.4f}',
                        'loss_per2: {loss_per2:.4f}',
                        'loss_seg: {loss_seg:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    loss_per1=loss_per1,
                    loss_per2=loss_per2,
                    loss_seg=loss_seg,
                    time=t_intv,
                    eta=eta,
                )
                logger.info(msg)
                st = ed

        train_de_losses.append(loss_decom / train_loader.n_iter)
        train_seg_losses.append(loss_seg / train_loader.n_iter)
        train_fusion_losses.append(loss_fus / train_loader.n_iter)

        if (epo + 1) % 1 == 0:
            fusion_model_file = os.path.join(modelpth, 'fusion_model{}.pth'.format(epo + 1))
            torch.save(fusionmodel.state_dict(), fusion_model_file)
            print(epo)
        gc.collect()

    fig, ax = plt.subplots(1, 3)
    ax[0].plot(np.arange(len(train_de_losses)), train_de_losses)
    ax[0].set_title('train_de_losses')
    ax[1].plot(np.arange(len(train_seg_losses)), train_seg_losses)
    ax[1].set_title('train_seg_losses')
    ax[2].plot(np.arange(len(train_fusion_losses)), train_fusion_losses)
    ax[2].set_title('train_fusion_losses')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1)
    plt.show()

    plt.title('Model loss')
    fig.savefig("./model/1.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='SeAFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=16)  # 16 a
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--Train', type=bool, default=True)
    parser.add_argument('--num_workers', '-j', type=int, default=8)  # 8 a
    args = parser.parse_args()
   
    logpath = './logs'
    logger = logging.getLogger()
  
    train_fusion(logger)
    print("training Done!")
