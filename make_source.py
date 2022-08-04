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


save_dir = Path('./data/source/')
save_dir.mkdir(exist_ok=True)

img_dir = save_dir.joinpath('images')
img_dir.mkdir(exist_ok=True)

json_dir = save_dir.joinpath('poses')
json_dir.mkdir(exist_ok=True)

start_frame = 0
if len(os.listdir('./data/source/images')) < 100:
    cap = cv2.VideoCapture(str(save_dir.joinpath('mv.mp4')))
    i = 0
    while cap.isOpened():
        flag, frame = cap.read()
        i += 1
        if not flag or i >= start_frame:
            break
    i = 0
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:# or i >= 10000:
            break
        if frame.shape[0] > frame.shape[1]:
            top = bottom = 0
            left = (frame.shape[0] - frame.shape[1]) // 2
            right = frame.shape[0] - frame.shape[1] - left
        else:
            left = right = 0
            top = (frame.shape[1] - frame.shape[0]) // 2
            bottom = frame.shape[1] - frame.shape[0] - top
        frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        cv2.imwrite(str(img_dir.joinpath('{:05}.png'.format(i))), frame)
        if i % 100 == 0:
            print('Has generated %d picetures' % i)
        i += 1

test_img_dir = save_dir.joinpath('test_img')
test_img_dir.mkdir(exist_ok=True)
test_label_dir = save_dir.joinpath('test_label')
test_label_dir.mkdir(exist_ok=True)
test_mask_dir = save_dir.joinpath('test_mask')
test_mask_dir.mkdir(exist_ok=True)
test_bg_dir = save_dir.joinpath('test_bg')
test_bg_dir.mkdir(exist_ok=True)
test_head_dir = save_dir.joinpath('test_head')
test_head_dir.mkdir(exist_ok=True)
test_lhand_dir = save_dir.joinpath('test_lhand')
test_lhand_dir.mkdir(exist_ok=True)
test_rhand_dir = save_dir.joinpath('test_rhand')
test_rhand_dir.mkdir(exist_ok=True)
test_lfoot_dir = save_dir.joinpath('test_lfoot')
test_lfoot_dir.mkdir(exist_ok=True)
test_rfoot_dir = save_dir.joinpath('test_rfoot')
test_rfoot_dir.mkdir(exist_ok=True)

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
    json_path = json_dir.joinpath('{:08}_keypoints.json'.format(idx + start_frame + 1))
    img = cv2.imread(str(img_path))

    with open(str(json_path), 'r') as f:
        pose = json.load(f)

    try:
        mask = np.expand_dims((get_mask.get_mask(str(img_path)))[:, :, 0], 2)
        label, head_cord, lhand_cord, rhand_cord, lfoot_cord, rfoot_cord, \
        head_len, eye_len, nose_leye, nose_reye, nose_lear, nose_rear = get_label.get_label(img, pose)
    except:
        #mask = np.zeros((512, 512, 1))
        continue
    cv2.imwrite(str(test_mask_dir.joinpath('{:05}.png'.format(idx))), mask)
    cv2.imwrite(str(test_bg_dir.joinpath('{:05}.png'.format(idx))), img * (1 - mask))

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
    plt.savefig(str(test_head_dir.joinpath('head_{}.png'.format(idx))))
    plt.clf()
    plt.imshow(lhand)
    plt.savefig(str(test_lhand_dir.joinpath('lhand_{}.png'.format(idx))))
    plt.clf()
    plt.imshow(rhand)
    plt.savefig(str(test_rhand_dir.joinpath('rhand_{}.png'.format(idx))))
    plt.clf()
    plt.imshow(lfoot)
    plt.savefig(str(test_lfoot_dir.joinpath('lfoot_{}.png'.format(idx))))
    plt.clf()
    plt.imshow(rfoot)
    plt.savefig(str(test_rfoot_dir.joinpath('rfoot_{}.png'.format(idx))))
    plt.clf()

    cv2.imwrite(str(test_img_dir.joinpath('{:05}.png'.format(idx))), img)
    cv2.imwrite(str(test_label_dir.joinpath('{:05}.png'.format(idx))), label)
    if idx % 100 == 0 and idx != 0:
        head_cords_arr = np.array(head_cords, dtype=np.int)
        np.save(str((save_dir.joinpath('head_source.npy'))), head_cords_arr)
        lhand_cords_arr = np.array(lhand_cords, dtype=np.int)
        np.save(str((save_dir.joinpath('lhand_source.npy'))), lhand_cords_arr)
        rhand_cords_arr = np.array(rhand_cords, dtype=np.int)
        np.save(str((save_dir.joinpath('rhand_source.npy'))), rhand_cords_arr)
        lfoot_cords_arr = np.array(lfoot_cords, dtype=np.int)
        np.save(str((save_dir.joinpath('lfoot_source.npy'))), lfoot_cords_arr)
        rfoot_cords_arr = np.array(rfoot_cords, dtype=np.int)
        np.save(str((save_dir.joinpath('rfoot_source.npy'))), rfoot_cords_arr)

target_info = joblib.load('./data/target/lens_dict.pkl')
sub_lens = [[0] * len(target_info['head_lens']) for i in head_lens]
for i in range(len(head_lens)):
    for j in range(len(target_info['head_lens'])):
        sub_lens[i][j] = - abs(head_lens[i] - target_info['head_lens'][j]) - abs(eye_lens[i] - target_info['eye_lens'][j]) \
                         - abs(nose_leyes[i] - target_info['nose_leyes'][j]) - abs(nose_reyes[i] - target_info['nose_reyes'][j]) \
                         - abs(nose_lears[i] - target_info['nose_lears'][j]) - abs(nose_rears[i] - target_info['nose_rears'][j])
sub_lens = torch.Tensor(sub_lens)
weights, indices = sub_lens.topk(3, dim=1)
np.save(str((save_dir.joinpath('indices.npy'))), indices.numpy())
np.save(str((save_dir.joinpath('weights.npy'))), softmax(weights.numpy()))

head_cords_arr = np.array(head_cords, dtype=np.int)
np.save(str((save_dir.joinpath('head_source_norm.npy'))), head_cords_arr)
lhand_cords_arr = np.array(lhand_cords, dtype=np.int)
np.save(str((save_dir.joinpath('lhand_source_norm.npy'))), lhand_cords_arr)
rhand_cords_arr = np.array(rhand_cords, dtype=np.int)
np.save(str((save_dir.joinpath('rhand_source_norm.npy'))), rhand_cords_arr)
lfoot_cords_arr = np.array(lfoot_cords, dtype=np.int)
np.save(str((save_dir.joinpath('lfoot_source_norm.npy'))), lfoot_cords_arr)
rfoot_cords_arr = np.array(rfoot_cords, dtype=np.int)
np.save(str((save_dir.joinpath('rfoot_source_norm.npy'))), rfoot_cords_arr)
