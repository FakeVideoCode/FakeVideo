import cv2
import json
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sys import argv
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
    s = x_exp / (x_sum + 1e-6)
    return s


os.system('rm -rf ./data/source')
os.system('rm -rf ./data/part_enhance')
os.system('rm -rf ./results')

source_dir = Path('../dance_{:02}/data/target/'.format(int(argv[1])))

save_dir = Path('./data/source/')
save_dir.mkdir(exist_ok=True)

img_dir = save_dir.joinpath('images')
img_dir.mkdir(exist_ok=True)

json_dir = save_dir.joinpath('poses')
# json_dir.mkdir(exist_ok=True)

print('copying...')
os.system('cp -a ' + str(source_dir.joinpath('train/train_img')) + ' ' + str(save_dir.joinpath('test_img')))
os.system('cp -a ' + str(source_dir.joinpath('train/train_label')) + ' ' + str(save_dir.joinpath('test_label')))
os.system('cp -a ' + str(source_dir.joinpath('train/train_mask')) + ' ' + str(save_dir.joinpath('test_mask')))
os.system('cp -a ' + str(source_dir.joinpath('train/fill_bg')) + ' ' + str(save_dir.joinpath('syn_bg')))
os.system('cp -a ' + str(source_dir.joinpath('train/head_img')) + ' ' + str(save_dir.joinpath('test_head')))
os.system('cp -a ' + str(source_dir.joinpath('train/lhand_img')) + ' ' + str(save_dir.joinpath('test_lhand')))
os.system('cp -a ' + str(source_dir.joinpath('train/rhand_img')) + ' ' + str(save_dir.joinpath('test_rhand')))
os.system('cp -a ' + str(source_dir.joinpath('train/lfoot_img')) + ' ' + str(save_dir.joinpath('test_lfoot')))
os.system('cp -a ' + str(source_dir.joinpath('train/rfoot_img')) + ' ' + str(save_dir.joinpath('test_rfoot')))
os.system('cp -a ' + str(source_dir.joinpath('poses')) + ' ' + str(save_dir.joinpath('poses')))
# print('cp ' + str(source_dir.joinpath('head.npy')) + ' ' + str(save_dir.joinpath('head_source_norm.npy')))
# print('cp ' + str(source_dir.joinpath('lhand.npy')) + ' ' + str(save_dir.joinpath('lhand_source_norm.npy')))
# print('cp ' + str(source_dir.joinpath('rhand.npy')) + ' ' + str(save_dir.joinpath('rhand_source_norm.npy')))
# print('cp ' + str(source_dir.joinpath('lfoot.npy')) + ' ' + str(save_dir.joinpath('lfoot_source_norm.npy')))
# print('cp ' + str(source_dir.joinpath('rfoot.npy')) + ' ' + str(save_dir.joinpath('rfoot_source_norm.npy')))

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

for idx in tqdm(range(len(os.listdir(str(json_dir))))):
    if len(argv) >= 3 and argv[2] == 'imitate':
        bg = cv2.imread('./data/target/train/fill_bg/00000.png')
        cv2.imwrite(str(save_dir.joinpath('syn_bg/{:05}.png'.format(idx))), bg)

    img = np.ones((3, 3))
    json_path = json_dir.joinpath('{:08}_keypoints.json'.format(idx + 1))

    with open(str(json_path), 'r') as f:
        pose = json.load(f)

    try:
        label, head_cord, lhand_cord, rhand_cord, lfoot_cord, rfoot_cord, \
        head_len, eye_len, nose_leye, nose_reye, nose_lear, nose_rear = get_label.get_label(img, pose)
    except:
        continue

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
