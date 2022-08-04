import cv2
import json
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import torch
import joblib
import src.utils.get_label as get_label
import src.utils.get_mask as get_mask
import src.utils.cv_utils as cv_utils
import src.pose.utils.util as pose_util
import src.config.src_opt as opt


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s


save_dir = Path('./data/target/')
save_dir.mkdir(exist_ok=True)

img_dir = save_dir.joinpath('images')
img_dir.mkdir(exist_ok=True)

json_dir = save_dir.joinpath('poses')
json_dir.mkdir(exist_ok=True)

if len(os.listdir('./data/target/images')) < 100:
    cap = cv2.VideoCapture(str(save_dir.joinpath('mv.mp4')))
    i = 0
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        cv2.imwrite(str(img_dir.joinpath('{:05}.png'.format(i))), frame)
        if i % 100 == 0:
            print('Has generated %d picetures' % i)
        i += 1

train_dir = save_dir.joinpath('train')
train_dir.mkdir(exist_ok=True)

train_img_dir = train_dir.joinpath('train_img')
train_img_dir.mkdir(exist_ok=True)
train_label_dir = train_dir.joinpath('train_label')
train_label_dir.mkdir(exist_ok=True)
train_mask_dir = train_dir.joinpath('train_mask')
train_mask_dir.mkdir(exist_ok=True)
train_fg_dir = train_dir.joinpath('train_fg')
train_fg_dir.mkdir(exist_ok=True)
train_head_dir = train_dir.joinpath('head_img')
train_head_dir.mkdir(exist_ok=True)
train_lhand_dir = train_dir.joinpath('lhand_img')
train_lhand_dir.mkdir(exist_ok=True)
train_rhand_dir = train_dir.joinpath('rhand_img')
train_rhand_dir.mkdir(exist_ok=True)
train_lfoot_dir = train_dir.joinpath('lfoot_img')
train_lfoot_dir.mkdir(exist_ok=True)
train_rfoot_dir = train_dir.joinpath('rfoot_img')
train_rfoot_dir.mkdir(exist_ok=True)

head_cords = []
lhand_cords = []
rhand_cords = []
lfoot_cords = []
rfoot_cords = []
head_lens = []
eye_lens = []
nose_leyes = []
nose_reyes = []
nose_lears = []
nose_rears = []

for idx in tqdm(range(len(os.listdir(str(img_dir))))):
    img_path = img_dir.joinpath('{:05}.png'.format(idx))
    json_path = json_dir.joinpath('{:08}_keypoints.json'.format(idx + 1))
    img = cv2.imread(str(img_path))

    with open(str(json_path), 'r') as f:
        pose = json.load(f)

    try:
        mask = np.expand_dims((get_mask.get_mask(str(img_path)))[:, :, 0], 2)
        label, head_cord, lhand_cord, rhand_cord, lfoot_cord, rfoot_cord, \
        head_len, eye_len, nose_leye, nose_reye, nose_lear, nose_rear = get_label.get_label(img, pose)
    except:
        continue
    cv2.imwrite(str(train_mask_dir.joinpath('{:05}.png'.format(idx))), mask)
    cv2.imwrite(str(train_fg_dir.joinpath('{:05}.png'.format(idx))), img * mask)


    head_lens.append(head_len)
    eye_lens.append(eye_len)
    nose_leyes.append(nose_leye)
    nose_reyes.append(nose_reye)
    nose_lears.append(nose_lear)
    nose_rears.append(nose_rear)

    head_cords.append(head_cord)
    lhand_cords.append(lhand_cord)
    rhand_cords.append(rhand_cord)
    lfoot_cords.append(lfoot_cord)
    rfoot_cords.append(rfoot_cord)
    crop_size = get_label.crop_size
    head = img[int(head_cord[1] - crop_size): int(head_cord[1] + crop_size),
               int(head_cord[0] - crop_size): int(head_cord[0] + crop_size), :]
    lhand = img[int(lhand_cord[1] - crop_size): int(lhand_cord[1] + crop_size),
                int(lhand_cord[0] - crop_size): int(lhand_cord[0] + crop_size), :]
    rhand = img[int(rhand_cord[1] - crop_size): int(rhand_cord[1] + crop_size),
                int(rhand_cord[0] - crop_size): int(rhand_cord[0] + crop_size), :]
    lfoot = img[int(lfoot_cord[1] - crop_size): int(lfoot_cord[1] + crop_size),
                int(lfoot_cord[0] - crop_size): int(lfoot_cord[0] + crop_size), :]
    rfoot = img[int(rfoot_cord[1] - crop_size): int(rfoot_cord[1] + crop_size),
                int(rfoot_cord[0] - crop_size): int(rfoot_cord[0] + crop_size), :]
    plt.imshow(head)
    plt.savefig(str(train_head_dir.joinpath('head_{}.png'.format(idx))))
    plt.clf()
    plt.imshow(lhand)
    plt.savefig(str(train_lhand_dir.joinpath('lhand_{}.png'.format(idx))))
    plt.clf()
    plt.imshow(rhand)
    plt.savefig(str(train_rhand_dir.joinpath('rhand_{}.png'.format(idx))))
    plt.clf()
    plt.imshow(lfoot)
    plt.savefig(str(train_lfoot_dir.joinpath('lfoot_{}.png'.format(idx))))
    plt.clf()
    plt.imshow(rfoot)
    plt.savefig(str(train_rfoot_dir.joinpath('rfoot_{}.png'.format(idx))))
    plt.clf()

    cv2.imwrite(str(train_img_dir.joinpath('{:05}.png'.format(idx))), img)
    cv2.imwrite(str(train_label_dir.joinpath('{:05}.png'.format(idx))), label)

sub_lens = [[0] * len(head_lens) for i in head_lens]
for i in range(len(head_lens)):
    for j in range(len(head_lens)):
        sub_lens[i][j] = - abs(head_lens[i] - head_lens[j]) - abs(eye_lens[i] - eye_lens[j]) \
                         - abs(nose_leyes[i] - nose_leyes[j]) - abs(nose_reyes[i] - nose_reyes[j]) \
                         - abs(nose_lears[i] - nose_lears[j]) - abs(nose_rears[i] - nose_rears[j])
sub_lens = torch.Tensor(sub_lens)
weights, indices = sub_lens.topk(3, dim=1)
np.save(str((save_dir.joinpath('indices.npy'))), indices.numpy())
np.save(str((save_dir.joinpath('weights.npy'))), softmax(weights.numpy()))
lens_dict = {
    'head_lens': head_lens,
    'eye_lens': eye_lens,
    'nose_leyes': nose_leyes,
    'nose_reyes': nose_reyes,
    'nose_lears': nose_lears,
    'nose_rears': nose_rears,
}
joblib.dump(lens_dict, str((save_dir.joinpath('lens_dict.pkl'))))

head_cords_arr = np.array(head_cords, dtype=np.int)
np.save(str((save_dir.joinpath('head.npy'))), head_cords_arr)
lhand_cords_arr = np.array(lhand_cords, dtype=np.int)
np.save(str((save_dir.joinpath('lhand.npy'))), lhand_cords_arr)
rhand_cords_arr = np.array(rhand_cords, dtype=np.int)
np.save(str((save_dir.joinpath('rhand.npy'))), rhand_cords_arr)
lfoot_cords_arr = np.array(lfoot_cords, dtype=np.int)
np.save(str((save_dir.joinpath('lfoot.npy'))), lfoot_cords_arr)
rfoot_cords_arr = np.array(rfoot_cords, dtype=np.int)
np.save(str((save_dir.joinpath('rfoot.npy'))), rfoot_cords_arr)
