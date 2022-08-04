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

crop_size = 48
radius = crop_size / 2 - 4
mask = torch.ones((1, 3, crop_size, crop_size)).to(device)
mask_value = 0.3
for i in range(crop_size):
    for j in range(crop_size):
        if (i-crop_size/2)**2 + (j-crop_size/2)**2 >= radius**2:
            mask[0, :, i, j] = torch.Tensor((mask_value, mask_value, mask_value))


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

def center_enhance(r):
    return r.mul(mask)
    
    
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    dataset_dir = '../data/part_enhance'
    head_pose_name = '../data/source/head_source_norm.npy'
    head_ckpt_dir = '../checkpoints/supervise'
    save_dir = dataset_dir + '/enhanced/'
    weight_name = '../data/source/weights.npy'
    index_name = '../data/source/indices.npy'

    if not os.path.exists(dataset_dir):
        print('generate %s' % dataset_dir)
        os.mkdir(dataset_dir)
    if not os.path.exists(dataset_dir):
        print('generate %s' % dataset_dir)
        os.mkdir(dataset_dir)
    if not os.path.exists(save_dir):
        print('generate %s' % save_dir)
        os.mkdir(save_dir)
    else:
        print(save_dir, 'is existing...')

    os.system('rm -rf ../data/part_enhance/original')
    os.system('cp -a ../data/source/syn_fg ../data/part_enhance/original/')

    image_folder = dataset.ImageFolderDataset(dataset_dir, is_test=True)
    head_dataset = dataset.FaceCropDataset(image_folder, head_pose_name, image_transforms, crop_size=crop_size, weight_name=weight_name, index_name=index_name)
    length = len(head_dataset)
    print('Picture number', length)

    head_generator = load_models(os.path.join(head_ckpt_dir))

    for i in tqdm(range(length)):
        try:
            _, fake_head, head_top, head_bottom, head_left, head_right, _, fake_full = head_dataset.get_full_sample(i)

            with torch.no_grad():
                fake_head.unsqueeze_(0)
                fake_head = fake_head.to(device)
                head_residual = head_generator(fake_head)
                head_residual = center_enhance(head_residual)
                head_enhanced = fake_head + head_residual

            head_enhanced.squeeze_()
            head_enhanced = torch2numpy(head_enhanced)
            fake_full[head_top:head_bottom, head_left:head_right, :] = head_enhanced

            b, g, r = cv2.split(fake_full)
            fake_full = cv2.merge([r, g, b])
            cv2.imwrite(save_dir + '{:05}.png'.format(i), fake_full)
            last_fake = fake_full
        except:
            cv2.imwrite(save_dir + '{:05}.png'.format(i), last_fake)

    os.system('rm -rf ../data/source/syn_fg')
    os.system('cp -a ../data/part_enhance/enhanced/ ../data/source/syn_fg')
