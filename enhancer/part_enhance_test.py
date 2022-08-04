import model
import dataset
import cv2
from trainer import Trainer
import os
from tqdm import tqdm
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from skimage.io import imsave
from imageio import get_writer
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
image_transforms = transforms.Compose([
        Image.fromarray,
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
    ])
    
device = torch.device('cuda')
np.set_printoptions(threshold=np.nan)
crop_size = 48
radius = crop_size / 2 - 2
mask = torch.ones((1, 3, crop_size, crop_size)).to(device)
mask_value = 0.5
for i in range(crop_size):
    for j in range(crop_size):
        if (i-crop_size/2)**2 + (j-crop_size/2)**2 >= radius**2:
            print(i, j)
            mask[0, :, i, j] = torch.Tensor((mask_value, mask_value, mask_value))
mask1 = torch.ones((1, 3, crop_size, crop_size)).to(device)
print(mask1)
print(mask.mul(mask1))
print(mask[0, :, crop_size//2, crop_size//2])
import a


def load_models(directory):
    generator = model.GlobalGenerator(n_downsampling=2, n_blocks=6)
    gen_name = os.path.join(directory, '40000_generator.pth')

    if os.path.isfile(gen_name):
        gen_dict = torch.load(gen_name)
        generator.load_state_dict(gen_dict)
        
    return generator.to(device)


def torch2numpy(tensor):
        generated = tensor.detach().cpu().permute(1, 2, 0).numpy()
        generated[generated < -1] = -1
        generated[generated > 1] = 1
        generated = (generated + 1) / 2 * 255
        return generated.astype(np.uint8)
    
    
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    dataset_dir = '../data/part_enhance'
    head_pose_name = '../data/source/head_source_norm.npy'
    lhand_pose_name = '../data/source/lhand_source_norm.npy'
    rhand_pose_name = '../data/source/rhand_source_norm.npy'
    lfoot_pose_name = '../data/source/lfoot_source_norm.npy'
    rfoot_pose_name = '../data/source/rfoot_source_norm.npy'
    head_ckpt_dir = '../checkpoints/head'
    lhand_ckpt_dir = '../checkpoints/lhand'
    rhand_ckpt_dir = '../checkpoints/rhand'
    lfoot_ckpt_dir = '../checkpoints/lfoot'
    rfoot_ckpt_dir = '../checkpoints/rfoot'
    save_dir = dataset_dir + '/enhanced/'

    target_dir = Path('../results/target/test_latest/images')
    target_synth_paths = sorted(target_dir.glob('*synthesized*'))
    target_synth_paths = [str(path) for path in target_synth_paths]
    # print('Copying the fake images...')
    # for img_idx in tqdm(range(len(target_synth_paths))):
    #     img = cv2.imread(target_synth_paths[img_idx])
    #     cv2.imwrite(dataset_dir + '/original/{:05}.png'.format(img_idx), img)

    if not os.path.exists(save_dir):
        print('generate %s' % save_dir)
        os.mkdir(save_dir)
    else:
        print(save_dir, 'is existing...')

    image_folder = dataset.ImageFolderDataset(dataset_dir, cache=os.path.join(dataset_dir, 'local.db'), is_test=True)
    head_dataset = dataset.FaceCropDataset(image_folder, head_pose_name, image_transforms, crop_size=48)
    lhand_dataset = dataset.FaceCropDataset(image_folder, lhand_pose_name, image_transforms, crop_size=48)
    rhand_dataset = dataset.FaceCropDataset(image_folder, rhand_pose_name, image_transforms, crop_size=48)
    lfoot_dataset = dataset.FaceCropDataset(image_folder, lfoot_pose_name, image_transforms, crop_size=48)
    rfoot_dataset = dataset.FaceCropDataset(image_folder, rfoot_pose_name, image_transforms, crop_size=48)
    length = len(head_dataset)
    print('Picture number', length)

    head_generator = load_models(os.path.join(head_ckpt_dir))
    lhand_generator = load_models(os.path.join(lhand_ckpt_dir))
    rhand_generator = load_models(os.path.join(rhand_ckpt_dir))
    lfoot_generator = load_models(os.path.join(lfoot_ckpt_dir))
    rfoot_generator = load_models(os.path.join(rfoot_ckpt_dir))

    for i in tqdm(range(length)):
        _, fake_head, head_top, head_bottom, head_left, head_right, _, fake_full = head_dataset.get_full_sample(i)
        _, fake_lhand, lhand_top, lhand_bottom, lhand_left, lhand_right, _, _ = lhand_dataset.get_full_sample(i)
        _, fake_rhand, rhand_top, rhand_bottom, rhand_left, rhand_right, _, _ = rhand_dataset.get_full_sample(i)
        _, fake_lfoot, lfoot_top, lfoot_bottom, lfoot_left, lfoot_right, _, _ = lfoot_dataset.get_full_sample(i)
        _, fake_rfoot, rfoot_top, rfoot_bottom, rfoot_left, rfoot_right, _, _ = rfoot_dataset.get_full_sample(i)

        with torch.no_grad():
            fake_head.unsqueeze_(0)
            fake_head = fake_head.to(device)
            head_residual = head_generator(fake_head)
            head_enhanced = fake_head + head_residual
            print(fake_head.shape)
            import a

            fake_lhand.unsqueeze_(0)
            fake_lhand = fake_lhand.to(device)
            lhand_residual = lhand_generator(fake_lhand)
            lhand_enhanced = fake_lhand + lhand_residual

            fake_rhand.unsqueeze_(0)
            fake_rhand = fake_rhand.to(device)
            rhand_residual = rhand_generator(fake_rhand)
            rhand_enhanced = fake_rhand + rhand_residual

            fake_lfoot.unsqueeze_(0)
            fake_lfoot = fake_lfoot.to(device)
            lfoot_residual = lfoot_generator(fake_lfoot)
            lfoot_enhanced = fake_lfoot + lfoot_residual

            fake_rfoot.unsqueeze_(0)
            fake_rfoot = fake_rfoot.to(device)
            rfoot_residual = rfoot_generator(fake_rfoot)
            rfoot_enhanced = fake_rfoot + rfoot_residual

        head_enhanced.squeeze_()
        head_enhanced = torch2numpy(head_enhanced)
        fake_full[head_top:head_bottom, head_left:head_right, :] = head_enhanced

        lhand_enhanced.squeeze_()
        lhand_enhanced = torch2numpy(lhand_enhanced)
        fake_full[lhand_top:lhand_bottom, lhand_left:lhand_right, :] = lhand_enhanced

        rhand_enhanced.squeeze_()
        rhand_enhanced = torch2numpy(rhand_enhanced)
        fake_full[rhand_top:rhand_bottom, rhand_left:rhand_right, :] = rhand_enhanced

        lfoot_enhanced.squeeze_()
        lfoot_enhanced = torch2numpy(lfoot_enhanced)
        fake_full[lfoot_top:lfoot_bottom, lfoot_left:lfoot_right, :] = lfoot_enhanced

        rfoot_enhanced.squeeze_()
        rfoot_enhanced = torch2numpy(rfoot_enhanced)
        fake_full[rfoot_top:rfoot_bottom, rfoot_left:rfoot_right, :] = rfoot_enhanced

        b, g, r = cv2.split(fake_full)
        fake_full = cv2.merge([r, g, b])
        cv2.imwrite(save_dir + '{:05}.png'.format(i), fake_full)

