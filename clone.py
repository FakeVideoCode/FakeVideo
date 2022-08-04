import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm


def get_mask_center(mask):
    mask = mask.copy()
    map = np.argwhere(mask != mask[0, 0])[:, 0]
    h = (map[-1] + map[0]) // 2
    mask = mask.transpose(1, 0, 2)
    map = np.argwhere(mask != mask[0, 0])[:, 0]
    w = (map[-1] + map[0]) // 2
    return w, h


save_dir = Path('./data/source/')
fg_dir = save_dir.joinpath('syn_fg')
bg_dir = save_dir.joinpath('fill_bg')
mask_dir = save_dir.joinpath('test_bg')

result_dir = save_dir.joinpath('bg_transfer')
result_dir.mkdir(exist_ok=True)

for idx in tqdm(range(len(os.listdir(str(fg_dir))))):
    fg_path = fg_dir.joinpath('{:05}.png'.format(idx))
    bg_path = bg_dir.joinpath('{:05}.png'.format(idx))
    mask_path = mask_dir.joinpath('{:05}.png'.format(idx))
    fg = cv2.imread(str(fg_path))
    bg = cv2.imread(str(bg_path))
    mask = (1 - cv2.imread(str(mask_path))) * 255
    center = get_mask_center(mask)
    try:
        normal_clone = cv2.seamlessClone(fg, bg, mask, center, cv2.NORMAL_CLONE)
    except:
        center = (256, 256)
        normal_clone = cv2.seamlessClone(fg, bg, mask, center, cv2.NORMAL_CLONE)
    cv2.imwrite(str(result_dir.joinpath('{:05}.png'.format(idx))), normal_clone)
