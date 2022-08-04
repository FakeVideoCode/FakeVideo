import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


source_dir = Path('./data/source/test_img')
# target_dir = Path('./data/source/bg_transfer')
# target_dir = Path('./results/final/target/test_latest/images')
target_dir = Path('./data/part_enhance/enhanced')
# label_dir = Path('./data/source/syn_fg')
label_dir = Path('./data/source/test_label')

source_img_paths = sorted(source_dir.iterdir())
# target_synth_paths = sorted(target_dir.glob('*synthesized*'))
target_synth_paths = sorted(target_dir.iterdir())
target_label_paths = sorted(label_dir.iterdir())

num = min(len(source_img_paths), len(target_synth_paths), len(target_label_paths))

fps = 25
size = (1536, 512)
videoWriter = cv2.VideoWriter('./data/source/output.mp4', cv2.VideoWriter_fourcc('X','V','I','D'), fps, size)

for i in tqdm(range(num)):
    source_img = cv2.imread(str(source_img_paths[i]))
    target_synth = cv2.imread(str(target_synth_paths[i]))
    target_label = cv2.imread(str(target_label_paths[i]))
    target_label = cv2.resize(target_label, (512, 512))
    frame = np.concatenate((source_img, target_label*4, target_synth), axis=1)
    videoWriter.write(frame)