import cv2
import os
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

        dir_bg = '_mask'
        self.dir_bg = os.path.join(opt.dataroot, opt.phase + dir_bg)
        self.bg_paths = sorted(make_dataset(self.dir_bg))

        dir_img = '_img' if opt.isTrain else '_bg'
        self.dir_img = os.path.join(opt.dataroot, opt.phase + dir_img)
        self.img_paths = sorted(make_dataset(self.dir_img))

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        bg_path = self.bg_paths[index]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.dilate(mask, self.kernel)
        img = cv2.resize(img, (512, 512))
        img = cv2.copyMakeBorder(img, 0, 0, 84, 84, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, 0, 0, 84, 84, cv2.BORDER_CONSTANT, value=1)
        # print(mask.shape)
        # print((1 - np.stack([mask, mask, mask], axis=2)).shape)
        # newmask = np.ones((512, 680, 3)) * 255 * (1 - np.stack([mask, mask, mask], axis=2))
        newmask = np.ones((512, 680, 3)) * 255 * (np.stack([mask], axis=2))
        # newmask = np.ones((512, 680, 3)) * 255 * mask
        input = img * (1 - np.stack([mask], axis=2)) + newmask
        # input = img * (1 - mask) + newmask
        # newmask = np.ones((512, 680, 3)) * 255 * (np.stack([mask, mask, mask], axis=2))
        # input = img * (1 - np.stack([mask, mask, mask], axis=2)) + newmask
        # input = img * np.stack([mask, mask, mask], axis=2) + newmask
        input_image = np.concatenate([input, newmask], axis=1)
        input_image = np.expand_dims(input_image, 0)

        return input_image

    def __len__(self):
        return len(self.img_paths)

