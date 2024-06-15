# coding:utf-8
import torchvision.transforms.functional as TF
import os
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import glob
import torchvision.transforms as transforms
from natsort import natsorted
import cv2


class Fusion_dataset(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        if split == 'train':
            self.vis_dir = './MSRS/Visible/train/MSRS/'
            self.ir_dir = './MSRS/Infrared/train/MSRS/'
            self.label_dir = './MSRS/Label/train/MSRS/'
            self.filelist = natsorted(os.listdir(self.vis_dir))
            self.split = split
            self.length = min(len(self.filelist), len(self.filelist))

        elif split == 'val' or split == 'test':
            self.vis_dir = vi_path
            self.ir_dir = ir_path
            self.filelist = natsorted(os.listdir(self.vis_dir))
            self.split = split
            self.length = min(len(self.filelist), len(self.filelist))

    def __getitem__(self, index):
        img_name = self.filelist[index]
        vis_path = os.path.join(self.vis_dir, img_name)
        ir_path = os.path.join(self.ir_dir, img_name)          
        img_vis = self.imread(path=vis_path)
        img_ir = self.imread(path=ir_path, vis_flage=False)            
        if self.split=='train':            
            label_path = os.path.join(self.label_dir, img_name)  
            label = self.imread(path=label_path, label=True)
            label = label.type(torch.LongTensor)   
                  
        if self.split=='train': 
            return img_vis, img_ir, label, img_name
        else:
            return img_vis, img_ir, img_name

    def __len__(self):
        return self.length
    
    @staticmethod
    def imread(path, label=False, vis_flage=True):
        if label:
            img = Image.open(path)
            im_ts = TF.to_tensor(img) * 255
        else:
            if vis_flage: ## visible images; RGB channel
                img = Image.open(path).convert('RGB')
                im_ts = TF.to_tensor(img)
            else: ## infrared images single channel 
                img = Image.open(path).convert('L') 
                im_ts = TF.to_tensor(img)
        return im_ts

class Fusion_med_dataset(Dataset):
    def __init__(self):
        super(Fusion_med_dataset, self).__init__()

        self.data_dir_vis = './Med_test/PET-MRI/sigma30_PET/'
        self.data_dir_ir = './Med_test/PET-MRI/sigma30_MRI/'
        self.img_names = sorted(os.listdir(self.data_dir_ir))
        self.length = len(self.img_names)
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        img_name = self.img_names[index]
        vis = cv2.imread(os.path.join(self.data_dir_vis, img_name), 1)
        ir = cv2.imread(os.path.join(self.data_dir_ir, img_name), 0)

        vis = Image.fromarray(vis)
        ir = Image.fromarray(ir, mode='L')

        vis = self.transform_train(vis)
        ir = self.transform_train(ir)

        return vis,  ir,  img_name

    def __len__(self):
        return self.length