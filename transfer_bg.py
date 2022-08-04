import os
import torch
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
import sys
pix2pixhd_dir = Path('./src/pix2pixHD/')
sys.path.append(str(pix2pixhd_dir))

from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import src.config.test_opt as opt
import src.utils.cv_utils as cv_utils
import numpy as np

opt.target = 'bg'
opt.checkpoints_dir += 'bg/'
opt.results_dir += 'bg/'

results_dir = Path('./data/source/syn_bg')
results_dir.mkdir(exist_ok=True)

torch.cuda.set_device(0)
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

model = create_model(opt)

for idx, data in enumerate(tqdm(dataset)):
    minibatch = 1
    generated = model.inference(data['label'], data['inst'], data['mask'], data['image'], data['feat'])

    cv_utils.save_cv2_img(generated[0].permute(1, 2, 0).cpu().numpy(), str(results_dir.joinpath('{:05}.png'.format(idx))), normalize=True)

    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][:, :, 0, :, :][0], opt.label_nc)),
                           ('synthesized_bg', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    visualizer.save_images(webpage, visuals, img_path)
webpage.save()
torch.cuda.empty_cache()
