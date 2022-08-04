import os
import torch
import torch.nn.functional as F
from collections import OrderedDict
import src.pose.utils.util as util
from src.pose.models.hmr import HumanModelRecovery
from src.pose.models.nmr import SMPLRenderer
import src.utils.cv_utils as cv_utils


class BodyRecoveryFlow(torch.nn.Module):

    def __init__(self, opt):
        super(BodyRecoveryFlow, self).__init__()
        self._name = 'BodyRecoveryFlow'
        self.opt = opt

        # create networks
        self._init_create_networks()

    def _create_hmr(self):
        hmr = HumanModelRecovery(smpl_pkl_path=self.opt.smpl_model)
        saved_data = torch.load(self.opt.hmr_model)
        hmr.load_state_dict(saved_data)
        hmr.eval()
        return hmr

    def _create_render(self):
        render = SMPLRenderer(map_name=self.opt.map_name,
                              uv_map_path=self.opt.uv_mapping,
                              tex_size=self.opt.tex_size,
                              image_size=self.opt.image_size, fill_back=False,
                              anti_aliasing=True, background_color=(0, 0, 0), has_front=False)

        return render

    def _init_create_networks(self):
        # hmr and render
        self.hmr = self._create_hmr()
        self.render = self._create_render()

    def forward(self, src_img, src_smpl):
        # get smpl information
        src_info = self.hmr.get_details(src_smpl)

        # process source inputs
        src_f2verts, src_fim, _ = self.render.render_fim_wim(src_info['cam'], src_info['verts'])
        src_f2verts = src_f2verts[:, :, :, 0:2]
        src_f2verts[:, :, :, 1] *= -1
        src_cond, _ = self.render.encode_fim(src_info['cam'], src_info['verts'], fim=src_fim, transpose=True)
        src_crop_mask = util.morph(src_cond[:, -1:, :, :], ks=3, mode='erode')

        # src input
        input_G_src = torch.cat([src_img * (1 - src_crop_mask), src_cond], dim=1)

        print('src_cond.shape ' + str(src_cond.shape))
        print('src_crop_mask.shape ' + str(src_crop_mask.shape))
        cv_utils.save_cv2_img((src_img * (1 - src_crop_mask))[0].permute(1, 2, 0).cpu().numpy(), './debug/train/src_img-1-src_crop_mask.png', normalize=True)
        cv_utils.save_cv2_img(src_img[0].permute(1, 2, 0).cpu().numpy(), './debug/train/src_img.png', normalize=True)
        cv_utils.save_cv2_img(src_cond[0].permute(1, 2, 0).cpu().numpy(), './debug/train/src_cond.png', normalize=True)
        cv_utils.save_cv2_img(src_crop_mask[0].permute(1, 2, 0).cpu().numpy(), './debug/train/src_crop_mask.png', normalize=True)

        # bg input
        src_bg_mask = util.morph(src_cond[:, -1:, :, :], ks=15, mode='erode')
        input_G_src_bg = torch.cat([src_img * src_bg_mask, src_bg_mask], dim=1)

        print('src_bg_mask.shape ' + str(src_bg_mask.shape))
        cv_utils.save_cv2_img(src_bg_mask[0].permute(1, 2, 0).cpu().numpy(), './debug/train/src_bg_mask.png', normalize=True)
        cv_utils.save_cv2_img((src_img * src_bg_mask)[0].permute(1, 2, 0).cpu().numpy(), './debug/train/src_img-src_bg_mask.png', normalize=True)

        return input_G_src_bg, input_G_tsf_bg, input_G_src, input_G_tsf, \
               T, src_crop_mask, tsf_crop_mask, head_bbox, body_bbox

    def cal_head_bbox(self, kps):
        """
        Args:
            kps: (N, 19, 2)

        Returns:
            bbox: (N, 4)
        """
        NECK_IDS = 12

        image_size = self.opt.image_size

        kps = (kps + 1) / 2.0

        necks = kps[:, NECK_IDS, 0]
        zeros = torch.zeros_like(necks)
        ones = torch.ones_like(necks)

        # min_x = int(max(0.0, np.min(kps[HEAD_IDS:, 0]) - 0.1) * image_size)
        min_x, _ = torch.min(kps[:, NECK_IDS:, 0] - 0.05, dim=1)
        min_x = torch.max(min_x, zeros)

        max_x, _ = torch.max(kps[:, NECK_IDS:, 0] + 0.05, dim=1)
        max_x = torch.min(max_x, ones)

        # min_x = int(max(0.0, np.min(kps[HEAD_IDS:, 0]) - 0.1) * image_size)
        min_y, _ = torch.min(kps[:, NECK_IDS:, 1] - 0.05, dim=1)
        min_y = torch.max(min_y, zeros)

        max_y, _ = torch.max(kps[:, NECK_IDS:, 1], dim=1)
        max_y = torch.min(max_y, ones)

        min_x = (min_x * image_size).long()  # (T, 1)
        max_x = (max_x * image_size).long()  # (T, 1)
        min_y = (min_y * image_size).long()  # (T, 1)
        max_y = (max_y * image_size).long()  # (T, 1)

        # print(min_x.shape, max_x.shape, min_y.shape, max_y.shape)
        rects = torch.stack((min_x, max_x, min_y, max_y), dim=1)
        # import ipdb
        # ipdb.set_trace()
        return rects

    def cal_body_bbox(self, kps, factor=1.2):
        """
        Args:
            kps (torch.cuda.FloatTensor): (N, 19, 2)
            factor (float):

        Returns:
            bbox: (N, 4)
        """
        image_size = self.opt.image_size
        bs = kps.shape[0]
        kps = (kps + 1) / 2.0
        zeros = torch.zeros((bs,), device=kps.device)
        ones = torch.ones((bs,), device=kps.device)

        min_x, _ = torch.min(kps[:, :, 0], dim=1)
        max_x, _ = torch.max(kps[:, :, 0], dim=1)
        middle_x = (min_x + max_x) / 2
        width = (max_x - min_x) * factor
        min_x = torch.max(zeros, middle_x - width / 2)
        max_x = torch.min(ones, middle_x + width / 2)

        min_y, _ = torch.min(kps[:, :, 1], dim=1)
        max_y, _ = torch.max(kps[:, :, 1], dim=1)
        middle_y = (min_y + max_y) / 2
        height = (max_y - min_y) * factor
        min_y = torch.max(zeros, middle_y - height / 2)
        max_y = torch.min(ones, middle_y + height / 2)

        min_x = (min_x * image_size).long()  # (T,)
        max_x = (max_x * image_size).long()  # (T,)
        min_y = (min_y * image_size).long()  # (T,)
        max_y = (max_y * image_size).long()  # (T,)

        # print(min_x.shape, max_x.shape, min_y.shape, max_y.shape)
        bboxs = torch.stack((min_x, max_x, min_y, max_y), dim=1)

        return bboxs

