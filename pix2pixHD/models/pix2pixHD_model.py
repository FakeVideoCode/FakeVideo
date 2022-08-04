### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from .discriminator_t import Discriminator_t
from . import networks
import src.utils.cv_utils as cv_utils

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, True, True)

        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake, dt_real, dt_fake):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake, dt_real, dt_fake), flags) if f]

        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.target = opt.target
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features

        ######extra add
        self.gpu_ids = opt.gpu_ids
        ####################
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks
        # Generator network
        netG_input_nc = input_nc
        if self.target == 'fg':
            netG_input_nc = input_nc * 2
        elif self.target == 'final':
            netG_input_nc = input_nc + 6
        else:
            netG_input_nc = input_nc * 3
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids, target=self.target)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            if self.target != 'final':
                self.netDt = Discriminator_t(n_class=2, target=self.target)

        ### Encoder network
        if self.gen_features:
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder',
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)
        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G'+self.target, opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D'+self.target, opt.which_epoch, pretrained_path)
                if self.target != 'final':
                    self.load_network(self.netDt, 'Dt'+self.target, opt.which_epoch, pretrained_path)
            if self.gen_features:
                self.load_network(self.netE, 'E'+self.target, opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)


            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake','Dt_real', 'Dt_fake')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))
            else:
                params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            if self.target != 'final':
                params = list(self.netDt.parameters())
                self.optimizer_Dt = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, labels, mask, inst_map=None, real_images=None, feat_map=None, infer=False):
        size = labels.size()
        oneHot_size = (size[0], self.opt.label_nc, size[2], size[3], size[4])
        input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(1, labels.data.long().cuda(), 1.0)
        labels = input_label

        label_map = labels[:, :, 0, :, :]
        input_label = label_map.data.cuda()
        input_label = Variable(input_label, volatile=infer)

        cur_label = input_label

        labels = Variable(labels.data.cuda())
        mask = Variable(mask.data.cuda())

        # real images for training
        cur_image = 0
        if real_images is not None:
            real_image = Variable(real_images.data.cuda())
            cur_image = real_images[:, :, 0, :, :]

        return cur_label, labels, mask, inst_map, real_images, cur_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, inst, mask, image, feat, infer=True, synbg=None, synfg=None):
        if self.target == 'fg':
            return self.forward_fg(label, inst, mask, image, feat, infer=True)
        elif self.target == 'bg':
            return self.forward_bg(label, inst, mask, image, feat, infer=True)
        elif self.target == 'final':
            return self.forward_final(label, inst, synbg, synfg, image, feat, infer=True)

    def forward_fg(self, label, inst, mask, image, feat, infer=False):
        # Encode Inputs
        cur_label, labels, mask, inst_map, real_images, cur_image, feat_map = self.encode_input(label, mask, inst, image, feat)
        cur_label = cur_label.cuda()
        labels = labels.cuda()
        mask = mask.cuda()
        real_images = real_images.cuda()
        real_fgs = real_images.clone().cuda()
        real_fgs[:, :, 0, :, :] = real_fgs[:, :, 0, :, :] * mask
        real_fgs[:, :, 1, :, :] = real_fgs[:, :, 1, :, :] * mask
        real_fgs[:, :, 2, :, :] = real_fgs[:, :, 2, :, :] * mask
        cur_fg = real_fgs[:, :, 0, :, :]

        # Fake Generation
        bs, h, w = labels.size(0), labels.size(3), labels.size(4)
        fake_fg = self.netG.forward(labels[:, :, :2, :, :].contiguous().view(bs, -1, h, w).float())

        # Foreground
        pred_fake_fg_pool = self.discriminate(cur_label, fake_fg, use_pool=True)
        loss_D_fg_fake = self.criterionGAN(pred_fake_fg_pool, False)
        pred_real_fg = self.discriminate(cur_label, cur_fg)
        loss_D_fg_real = self.criterionGAN(pred_real_fg, True)

        bs, h, w = real_fgs.size(0), real_fgs.size(3), real_fgs.size(4)
        loss_Dt_fg_real = - torch.mean(self.netDt.forward(real_fgs[:, :, :2, :, :].contiguous().view(bs, -1, h, w), torch.tensor(1).cuda()))
        fake_fgs = real_fgs.clone().cuda()
        fake_fgs[:, :, 0, :, :] = fake_fg
        # TODO temporal dis
        loss_Dt_fg_fake = self.netDt.forward(fake_fgs[:, :, :2, :, :].contiguous().view(bs, -1, h, w), torch.tensor(0).cuda()).mean()

        # loss_Dt_fg_real = - torch.mean(self.netDt.forward(real_fgs[:, :, :2, :, :], torch.tensor(1).cuda()))
        # fake_fgs = torch.tensor(real_fgs).cuda()
        # fake_fgs[:, :, 0, :, :] = fake_fg
        # loss_Dt_fg_fake = self.netDt.forward(fake_fgs[:, :, :2, :, :], torch.tensor(0).cuda()).mean()

        # GAN loss (Fake Passability Loss)
        pred_fake_fg = self.netD.forward(torch.cat((cur_label, fake_fg), dim=1))
        loss_G_fg_GAN = self.criterionGAN(pred_fake_fg, True)
        # GAN feature matching loss
        loss_G_fg_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_fg[i])-1):
                    loss_G_fg_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake_fg[i][j], pred_real_fg[i][j].detach()) * self.opt.lambda_feat
        # VGG feature matching loss
        loss_G_fg_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_fg_VGG = self.criterionVGG(fake_fg, cur_fg) * self.opt.lambda_feat

        # Only return the fake_B image if necessary to save BW
        return [self.loss_filter(loss_G_fg_GAN, loss_G_fg_GAN_Feat, loss_G_fg_VGG,
                                 loss_D_fg_real, loss_D_fg_fake,
                                 # 0, 0), None if not infer else fake_fg]
                                 loss_Dt_fg_real, loss_Dt_fg_fake), None if not infer else fake_fg]

    def forward_bg(self, label, inst, mask, image, feat, infer=False):
        # Encode Inputs
        cur_label, labels, mask, inst_map, real_images, cur_image, feat_map = self.encode_input(label, mask, inst, image, feat)
        cur_label = cur_label.cuda()
        labels = labels.cuda()
        mask = mask.cuda()
        real_images = real_images.cuda()
        real_bgs = real_images.clone().cuda()
        real_bgs[:, :, 0, :, :] = real_bgs[:, :, 0, :, :] * (1 - mask)
        real_bgs[:, :, 1, :, :] = real_bgs[:, :, 1, :, :] * (1 - mask)
        real_bgs[:, :, 2, :, :] = real_bgs[:, :, 2, :, :] * (1 - mask)
        cur_bg = real_bgs[:, :, 0, :, :]

        # cv_utils.save_cv2_img(cur_bg[0].permute(1, 2, 0).cpu().numpy(), 'cur_bg.png', normalize=True)
        # cv_utils.save_cv2_img(real_images[:, :, 0, :, :][0].permute(1, 2, 0).cpu().numpy(), 'cur_image.png', normalize=True)
        # cv_utils.save_cv2_img(cur_label[0].permute(1, 2, 0).cpu().numpy(), 'cur_label.png', normalize=True)
        # cv_utils.save_cv2_img(cropmask[0].permute(1, 2, 0).cpu().numpy(), 'cur_cropmask.png', normalize=True)
        # import a

        # Fake Generation
        bs, h, w = labels.size(0), labels.size(3), labels.size(4)
        fake_bg = self.netG.forward(labels.contiguous().view(bs, -1, h, w).float())

        # Background
        pred_fake_bg_pool = self.discriminate(cur_label, fake_bg, use_pool=True)
        loss_D_bg_fake = self.criterionGAN(pred_fake_bg_pool, False)
        pred_real_bg = self.discriminate(cur_label, cur_bg)
        loss_D_bg_real = self.criterionGAN(pred_real_bg, True)

        bs, h, w = real_bgs.size(0), real_bgs.size(3), real_bgs.size(4)
        loss_Dt_bg_real = - torch.mean(self.netDt.forward(real_bgs.contiguous().view(bs, -1, h, w), torch.tensor(1).cuda()))
        fake_bgs = real_bgs.clone().cuda()
        fake_bgs[:, :, 0, :, :] = fake_bg
        loss_Dt_bg_fake = self.netDt.forward(fake_bgs.contiguous().view(bs, -1, h, w), torch.tensor(0).cuda()).mean()

        # loss_Dt_bg_real = - torch.mean(self.netDt.forward(real_bgs, torch.tensor(1).cuda()))
        # fake_bgs = real_bgs.clone().cuda()
        # fake_bgs[:, :, 0, :, :] = fake_bg
        # loss_Dt_bg_fake = self.netDt.forward(fake_bgs, torch.tensor(0).cuda()).mean()

        # GAN loss (Fake Passability Loss)
        pred_fake_bg = self.netD.forward(torch.cat((cur_label, fake_bg), dim=1))
        loss_G_bg_GAN = self.criterionGAN(pred_fake_bg, True)
        # GAN feature matching loss
        loss_G_bg_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_bg[i])-1):
                    loss_G_bg_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake_bg[i][j], pred_real_bg[i][j].detach()) * self.opt.lambda_feat
        # VGG feature matching loss
        loss_G_bg_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_bg_VGG = self.criterionVGG(fake_bg, cur_bg) * self.opt.lambda_feat

        # Only return the fake_B image if necessary to save BW
        return [self.loss_filter(loss_G_bg_GAN, loss_G_bg_GAN_Feat, loss_G_bg_VGG,
                                 loss_D_bg_real, loss_D_bg_fake,
                                 # 0, 0), None if not infer else fake_final]
                                 loss_Dt_bg_real, loss_Dt_bg_fake), None if not infer else fake_bg]

    def forward_final(self, label, inst, synbg, synfg, image, feat, infer=False):
        # Encode Inputs
        cur_label, _, _, _, _, cur_image, _ = self.encode_input(label, synbg, inst, image, feat)
        cur_label = cur_label.cuda()
        cur_image = cur_image.cuda()
        synbg = Variable(synbg.data.cuda())
        synfg = Variable(synfg.data.cuda())

        # cv_utils.save_cv2_img(synfg[0].permute(1, 2, 0).cpu().numpy(), 'cur_fg.png', normalize=True)
        # cv_utils.save_cv2_img(synbg[0].permute(1, 2, 0).cpu().numpy(), 'cur_bg.png', normalize=True)
        # cv_utils.save_cv2_img(self.fake_bg[0].permute(1, 2, 0).cpu().detach().numpy(), 'fake_bg.png', normalize=True)
        # cv_utils.save_cv2_img(self.fake_fg[0].permute(1, 2, 0).cpu().detach().numpy(), 'fake_fg.png', normalize=True)
        # import a

        # Fake Generation
        netG_final_input = torch.cat((cur_label, synbg, synfg), dim=1).float()
        fake_final = self.netG.forward(netG_final_input)

        # Final Fake Image
        pred_fake_final_pool = self.discriminate(cur_label, fake_final, use_pool=True)
        loss_D_final_fake = self.criterionGAN(pred_fake_final_pool, False)
        pred_real_final = self.discriminate(cur_label, cur_image)
        loss_D_final_real = self.criterionGAN(pred_real_final, True)

        # GAN loss (Fake Passability Loss)
        pred_fake_final = self.netD.forward(torch.cat((cur_label, fake_final), dim=1))
        loss_G_final_GAN = self.criterionGAN(pred_fake_final, True)
        # GAN feature matching loss
        loss_G_final_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_final[i])-1):
                    loss_G_final_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake_final[i][j], pred_real_final[i][j].detach()) * self.opt.lambda_feat
        # VGG feature matching loss
        loss_G_final_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_final_VGG = self.criterionVGG(fake_final, cur_image) * self.opt.lambda_feat

        # Only return the fake_B image if necessary to save BW
        return [self.loss_filter(loss_G_final_GAN, loss_G_final_GAN_Feat, loss_G_final_VGG,
                                 loss_D_final_real, loss_D_final_fake, 0, 0), None if not infer else fake_final]

    def inference(self, label, inst, mask, image, feat, synbg=None, synfg=None):
        if self.target == 'fg':
            return self.inference_fg(label, inst, mask, image, feat)
        elif self.target == 'bg':
            return self.inference_bg(label, inst, mask, image, feat)
        elif self.target == 'final':
            return self.inference_final(label, inst, synbg, synfg, image, feat)

    def inference_fg(self, label, inst, mask, image, feat):
        # Encode Inputs
        _, labels, _, _, _, _, _ = self.encode_input(label, mask)
        labels = labels.cuda()

        # Fake Generation
        bs, h, w = labels.size(0), labels.size(3), labels.size(4)
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_fg = self.netG.forward(labels[:, :, :2, :, :].contiguous().view(bs, -1, h, w).float())
        else:
            fake_fg = self.netG.forward(labels[:, :, :2, :, :].contiguous().view(bs, -1, h, w).float())
        return fake_fg

    def inference_bg(self, label, inst, mask, image, feat):
        # Encode Inputs
        _, labels, _, _, _, _, _ = self.encode_input(label, mask)
        labels = labels.cuda()

        # Fake Generation
        bs, h, w = labels.size(0), labels.size(3), labels.size(4)
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_bg = self.netG.forward(labels.contiguous().view(bs, -1, h, w).float())
        else:
            fake_bg = self.netG.forward(labels.contiguous().view(bs, -1, h, w).float())
        return fake_bg

    def inference_final(self, label, inst, synbg, synfg, image, feat):
        # Encode Inputs
        cur_label, _, _, _, _, _, _ = self.encode_input(label, synbg)
        synbg = Variable(synbg.data.cuda())
        synfg = Variable(synfg.data.cuda())
        cur_label = cur_label.cuda()

        # Fake Generation
        netG_final_input = torch.cat((cur_label, synbg, synfg), dim=1).float()
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_final = self.netG.forward(netG_final_input)
        else:
            fake_final = self.netG.forward(netG_final_input)
        return fake_final

    def sample_features(self, inst):
        # read precomputed feature clusters
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path).item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])

                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G'+self.target, which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D'+self.target, which_epoch, self.gpu_ids)
        if self.target != 'final':
            self.save_network(self.netDt, 'Dt'+self.target, which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E'+self.target, which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        if self.target != 'final':
            for param_group in self.optimizer_Dt.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)

        
