from mmdet.apis import init_detector, inference_detector
# from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import cv2

config_file = 'src/msrcnn/ms_rcnn_x101_64x4d_fpn_1x.py'
checkpoint_file = 'src/msrcnn/ms_rcnn_x101_64x4d_fpn_1x_20190628-dec32bda.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

score_thr = 0.3


def get_mask(img_path):
    result = inference_detector(model, img_path)
    bbox_result, segm_result = result
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        mask = maskUtils.decode(segms[0]).astype(np.int)
        # cv2.imwrite('bgmask.png', mask)
    return mask
