from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import src.utils.cv_utils as cv_utils
from pathlib import Path


target_img = cv_utils.read_cv2_img('./data/target/train/train_label/00001.png')
target_img_rgb = cv_utils.read_cv2_img('./data/target/train/train_img/00001.png')
source_img = cv_utils.read_cv2_img('./data/source/test_label_ori/00001.png')
source_img_rgb = cv_utils.read_cv2_img('./data/source/test_img/00001.png')

label_path = './data/source/test_label_ori/'
bg_path = './data/source/test_bg_ori/'
mask_path = './data/source/test_mask_ori/'
save_dir = Path('./data/source/')
label_output = save_dir.joinpath('test_label')
label_output.mkdir(exist_ok=True)
bg_output = save_dir.joinpath('test_bg')
bg_output.mkdir(exist_ok=True)
mask_output = save_dir.joinpath('test_mask')
mask_output.mkdir(exist_ok=True)
head_dir = save_dir.joinpath('test_head')
head_dir.mkdir(exist_ok=True)
lhand_dir = save_dir.joinpath('test_lhand')
lhand_dir.mkdir(exist_ok=True)
rhand_dir = save_dir.joinpath('test_rhand')
rhand_dir.mkdir(exist_ok=True)
lfoot_dir = save_dir.joinpath('test_lfoot')
lfoot_dir.mkdir(exist_ok=True)
rfoot_dir = save_dir.joinpath('test_rfoot')
rfoot_dir.mkdir(exist_ok=True)
head_pose_dir = Path('./data/source/head_source.npy')
head_cords = np.load(str(head_pose_dir))
lhand_pose_dir = Path('./data/source/lhand_source.npy')
lhand_cords = np.load(str(lhand_pose_dir))
rhand_pose_dir = Path('./data/source/rhand_source.npy')
rhand_cords = np.load(str(rhand_pose_dir))
lfoot_pose_dir = Path('./data/source/lfoot_source.npy')
lfoot_cords = np.load(str(lfoot_pose_dir))
rfoot_pose_dir = Path('./data/source/rfoot_source.npy')
rfoot_cords = np.load(str(rfoot_pose_dir))

plt.subplot(222)
plt.imshow(target_img)
plt.subplot(221)
plt.imshow(target_img_rgb)
plt.subplot(224)
plt.imshow(source_img)
plt.subplot(223)
plt.imshow(source_img_rgb)
plt.savefig('norm.png')
plt.show()
plt.close()


def get_scale(label_img):
    label_img = label_img.copy()
    if len(label_img.shape) == 3:
        label_img = label_img[:, :, 0]
    label_img[label_img == 7] = 0
    label_img[label_img == 8] = 0
    label_img[label_img == 9] = 0
    label_img[label_img == 10] = 0
    label_img[label_img == 11] = 0
    label_img[label_img == 12] = 0
    label_img[label_img >= 25] = 0
    any1 = label_img.any(axis=1)
    linspace1 = np.arange(len(any1))
    head_y, height = linspace1[list(any1)][0], len(linspace1[list(any1)])
    top = head_y
    bottom = label_img.shape[0] - top - height
    return top, height, bottom


def pltcurrent(ori, target, new, part, save_path):
    plt.subplot(222)
    plt.imshow(ori * 10)
    plt.subplot(221)
    plt.imshow(target * 10)
    plt.subplot(224)
    plt.imshow(new * 10)
    plt.subplot(223)
    plt.imshow(part * 10)
    plt.savefig(save_path)
    plt.clf()


target_top, target_height, target_bottom = get_scale(target_img)
target_bottom = int(target_bottom / 3)
source_top, source_height, source_bottom = get_scale(source_img)
scale = target_height / source_height
print('scale: ' + str(scale))

new_head_pose = []
new_lhand_pose = []
new_rhand_pose = []
new_lfoot_pose = []
new_rfoot_pose = []

for img_idx in tqdm(range(len(os.listdir(label_path)))):
    label = cv2.imread(label_path + '{:05}.png'.format(img_idx))
    source_top, source_height, source_bottom = get_scale(label)
    scale = target_height / source_height
    bg = cv2.imread(bg_path + '{:05}.png'.format(img_idx))
    mask = cv2.imread(mask_path + '{:05}.png'.format(img_idx))

    source_head_cord_x, source_head_cord_y = head_cords[img_idx]
    source_lhand_cord_x, source_lhand_cord_y = lhand_cords[img_idx]
    source_rhand_cord_x, source_rhand_cord_y = rhand_cords[img_idx]
    source_lfoot_cord_x, source_lfoot_cord_y = lfoot_cords[img_idx]
    source_rfoot_cord_x, source_rfoot_cord_y = rfoot_cords[img_idx]
    try:
        ori_top, ori_height, ori_bottom = get_scale(label[:, :, 0])

        label_resize = cv2.resize(label, (int(label.shape[0] * scale), int(label.shape[1] * scale)))
        bg_resize = cv2.resize(bg, (int(bg.shape[0] * scale), int(bg.shape[1] * scale)))
        mask_resize = cv2.resize(mask, (int(mask.shape[0] * scale), int(mask.shape[1] * scale)))
        # mask_resize = np.expand_dims(mask_resize, 2)

        # print(label_resize.shape)
        # print(bg_resize.shape)
        # print(mask_resize.shape)

        label_pad = np.pad(label_resize, ((1000, 1000), (1000, 1000), (0, 0)), mode='edge')
        bg_pad = np.pad(bg_resize, ((1000, 1000), (1000, 1000), (0, 0)), mode='edge')
        mask_pad = np.pad(mask_resize, ((1000, 1000), (1000, 1000), (0, 0)), mode='edge')

        source_top_rs, source_height_rs, source_bottom_rs = get_scale(label_pad[:, :, 0])

        new_label = label_pad[
                    label_pad.shape[0] - source_bottom_rs + target_bottom - target_img.shape[0]:
                    label_pad.shape[0] - source_bottom_rs + target_bottom,
                    int((label_pad.shape[1] - target_img.shape[1]) / 2):
                    int((label_pad.shape[1] - (label_pad.shape[1] - target_img.shape[1]) / 2))
                    ]
        new_bg = bg_pad[
                 bg_pad.shape[0] - source_bottom_rs + target_bottom - target_img.shape[0]:
                 bg_pad.shape[0] - source_bottom_rs + target_bottom,
                 int((bg_pad.shape[1] - target_img.shape[1]) / 2):
                 int((bg_pad.shape[1] - (bg_pad.shape[1] - target_img.shape[1]) / 2))
                 ]
        new_mask = mask_pad[
                   mask_pad.shape[0] - source_bottom_rs + target_bottom - target_img.shape[0]:
                   mask_pad.shape[0] - source_bottom_rs + target_bottom,
                   int((mask_pad.shape[1] - target_img.shape[1]) / 2):
                   int((mask_pad.shape[1] - (mask_pad.shape[1] - target_img.shape[1]) / 2))
                   ]

        new_source_top, new_source_height, new_source_bottom = get_scale(new_label[:, :, 0])

        new_head_x = int(scale * source_head_cord_x + (1 - scale) * target_img.shape[1] / 2)
        new_head_y = int(
            target_img.shape[0] - target_bottom - scale * (target_img.shape[0] - ori_bottom - source_head_cord_y))
        new_lhand_x = int(scale * source_lhand_cord_x + (1 - scale) * target_img.shape[1] / 2)
        new_lhand_y = int(
            target_img.shape[0] - target_bottom - scale * (target_img.shape[0] - ori_bottom - source_lhand_cord_y))
        new_rhand_x = int(scale * source_rhand_cord_x + (1 - scale) * target_img.shape[1] / 2)
        new_rhand_y = int(
            target_img.shape[0] - target_bottom - scale * (target_img.shape[0] - ori_bottom - source_rhand_cord_y))
        new_lfoot_x = int(scale * source_lfoot_cord_x + (1 - scale) * target_img.shape[1] / 2)
        new_lfoot_y = int(
            target_img.shape[0] - target_bottom - scale * (target_img.shape[0] - ori_bottom - source_lfoot_cord_y))
        new_rfoot_x = int(scale * source_rfoot_cord_x + (1 - scale) * target_img.shape[1] / 2)
        new_rfoot_y = int(
            target_img.shape[0] - target_bottom - scale * (target_img.shape[0] - ori_bottom - source_rfoot_cord_y))

        crop_size = 25
        new_head_pose.append([new_head_x, new_head_y])
        new_lhand_pose.append([new_lhand_x, new_lhand_y])
        new_rhand_pose.append([new_rhand_x, new_rhand_y])
        new_lfoot_pose.append([new_lfoot_x, new_lfoot_y])
        new_rfoot_pose.append([new_rfoot_x, new_rfoot_y])
        head = new_label[int(new_head_y - crop_size): int(new_head_y + crop_size),
               int(new_head_x - crop_size): int(new_head_x + crop_size), :]
        lhand = new_label[int(new_lhand_y - crop_size): int(new_lhand_y + crop_size),
                int(new_lhand_x - crop_size): int(new_lhand_x + crop_size), :]
        rhand = new_label[int(new_rhand_y - crop_size): int(new_rhand_y + crop_size),
                int(new_rhand_x - crop_size): int(new_rhand_x + crop_size), :]
        lfoot = new_label[int(new_lfoot_y - crop_size): int(new_lfoot_y + crop_size),
                int(new_lfoot_x - crop_size): int(new_lfoot_x + crop_size), :]
        rfoot = new_label[int(new_rfoot_y - crop_size): int(new_rfoot_y + crop_size),
                int(new_rfoot_x - crop_size): int(new_rfoot_x + crop_size), :]
        try:
            pltcurrent(label, target_img, new_label, head, str(head_dir.joinpath('head_{}.jpg'.format(img_idx))))
            pltcurrent(label, target_img, new_label, lhand, str(lhand_dir.joinpath('lhand_{}.jpg'.format(img_idx))))
            pltcurrent(label, target_img, new_label, rhand, str(rhand_dir.joinpath('rhand_{}.jpg'.format(img_idx))))
            pltcurrent(label, target_img, new_label, lfoot, str(lfoot_dir.joinpath('lfoot_{}.jpg'.format(img_idx))))
            pltcurrent(label, target_img, new_label, rfoot, str(rfoot_dir.joinpath('rfoot_{}.jpg'.format(img_idx))))
        except:
            pass

        cv2.imwrite(str(label_output) + '/{:05}.png'.format(img_idx), new_label)
        cv2.imwrite(str(bg_output) + '/{:05}.png'.format(img_idx), new_bg)
        cv2.imwrite(str(mask_output) + '/{:05}.png'.format(img_idx), new_mask)
    except:
        label = cv2.imread(label_path + '{:05}.png'.format(img_idx))
        bg = cv2.imread(bg_path + '{:05}.png'.format(img_idx))
        mask = cv2.imread(mask_path + '{:05}.png'.format(img_idx))

        cv2.imwrite(str(label_output) + '/{:05}.png'.format(img_idx), label)
        cv2.imwrite(str(bg_output) + '/{:05}.png'.format(img_idx), bg)
        cv2.imwrite(str(mask_output) + '/{:05}.png'.format(img_idx), mask)
        new_head_pose.append([source_head_cord_x, source_head_cord_y])
        new_lhand_pose.append([source_lhand_cord_x, source_lhand_cord_y])
        new_rhand_pose.append([source_rhand_cord_x, source_rhand_cord_y])
        new_lfoot_pose.append([source_lfoot_cord_x, source_lfoot_cord_y])
        new_rfoot_pose.append([source_rfoot_cord_x, source_rfoot_cord_y])
        print(img_idx)

head_cords_arr = np.array(new_head_pose, dtype=np.int)
np.save(str((save_dir.joinpath('head_source_norm.npy'))), head_cords_arr)
lhand_cords_arr = np.array(new_lhand_pose, dtype=np.int)
np.save(str((save_dir.joinpath('lhand_source_norm.npy'))), lhand_cords_arr)
rhand_cords_arr = np.array(new_rhand_pose, dtype=np.int)
np.save(str((save_dir.joinpath('rhand_source_norm.npy'))), rhand_cords_arr)
lfoot_cords_arr = np.array(new_lfoot_pose, dtype=np.int)
np.save(str((save_dir.joinpath('lfoot_source_norm.npy'))), lfoot_cords_arr)
rfoot_cords_arr = np.array(new_rfoot_pose, dtype=np.int)
np.save(str((save_dir.joinpath('rfoot_source_norm.npy'))), rfoot_cords_arr)
