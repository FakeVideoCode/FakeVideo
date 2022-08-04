### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import torch
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import cv2
import torchvision.transforms as T
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.img_size = None
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.opt = opt
        self.root = opt.dataroot    

        ### input mask (foreground mask)
        dir_mask = '_mask'
        self.dir_mask = os.path.join(opt.dataroot, opt.phase + dir_mask)
        self.mask_paths = sorted(make_dataset(self.dir_mask))

        ### input A (label maps)
        dir_A = '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        if opt.target == 'final':
            dir_synbg = '_bg'
            if not opt.isTrain:
                phase = 'syn'
            else:
                phase = opt.phase
            self.dir_synbg = os.path.join(opt.dataroot, phase + dir_synbg)
            self.synbg_paths = sorted(make_dataset(self.dir_synbg))
            dir_synfg = '_fg'
            self.dir_synfg = os.path.join(opt.dataroot, phase + dir_synfg)
            self.synfg_paths = sorted(make_dataset(self.dir_synfg))

        ### input B (real images)
        if opt.isTrain:
            dir_B = '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):
        ### input A (label maps)
        A_path = self.A_paths[index]
        A = Image.open(A_path)
        params = get_params(self.opt, A.size)
        # transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        # A_tensor = transform_A(A) * 255.0
        transform_A, transform_mask = get_transform(self.opt, params, for_mask=True)
        A_tensor = transform_mask(A) * 255.0
        if self.img_size is None:
            self.img_size = A_tensor.size()
        if A_tensor.size() != self.img_size:
            A_tensor = self.pre_label
        else:
            self.pre_label = A_tensor

        synbg_tensor = synfg_tensor = 0
        if self.opt.target == 'final':
            synbg_path = self.synbg_paths[index]
            synbg = Image.open(synbg_path)
            synbg_tensor = transform_A(synbg.convert('RGB'))
            synfg_path = self.synfg_paths[index]
            synfg = Image.open(synfg_path)
            synfg_tensor = transform_A(synfg.convert('RGB'))

        ### input mask (foreground mask)
        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path)
        mask_tensor = transform_mask(mask) * 255.0

        # add previous frames
        pre_index = index - 1
        pre_frame_0 = A_tensor
        if pre_index >= 0:
            pre_path = self.A_paths[pre_index]
            frame = Image.open(pre_path)
            pre_frame_0 = transform_mask(frame) * 255.0
            if pre_frame_0.size() != self.img_size:
                pre_frame_0 = self.pre_label

        pre_index = pre_index - 1
        pre_frame_1 = A_tensor
        if pre_index >= 0:
            pre_path = self.A_paths[pre_index]
            frame = Image.open(pre_path)
            pre_frame_1 = transform_mask(frame) * 255.0
            if pre_frame_1.size() != self.img_size:
                pre_frame_1 = self.pre_label

        size = (A_tensor.size()[0], 3, A_tensor.size()[1], A_tensor.size()[2])
        labels = torch.zeros(size)
        labels[:, 0, :, :] = A_tensor
        labels[:, 1, :, :] = pre_frame_0
        labels[:, 2, :, :] = pre_frame_1

        B_tensor = inst_tensor = feat_tensor = images = 0
        ### input B (real images)
        if self.opt.isTrain:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            if index - 1 >= 0:
                pre_B_0 = Image.open(self.B_paths[index - 1]).convert('RGB')
            else:
                pre_B_0 = B
            if index - 2 >= 0:
                pre_B_1 = Image.open(self.B_paths[index - 2]).convert('RGB')
            else:
                pre_B_1 = B
            # transform_B = get_transform(self.opt, params)
            B_tensor = transform_A(B)
            pre_B_tensor_0 = transform_A(pre_B_0)
            pre_B_tensor_1 = transform_A(pre_B_1)

            gt_img = cv2.imread(B_path)
            gt_img_tensor = self.transform(gt_img)
            # gt_img_tensor = gt_img_tensor.unsqueeze(0)

            size = (B_tensor.size()[0], 3, B_tensor.size()[1], B_tensor.size()[2])
            images = torch.zeros(size)
            images[:, 0, :, :] = B_tensor
            images[:, 1, :, :] = pre_B_tensor_0
            images[:, 2, :, :] = pre_B_tensor_1

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': labels, 'inst': inst_tensor,
                      'image': images, 'mask': mask_tensor,
                      'feat': feat_tensor, 'path': A_path,
                      'synfg': synfg_tensor, 'synbg': synbg_tensor,
                      ###
                      # 'gt_img':gt_img_tensor
                      ###
                      }

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'