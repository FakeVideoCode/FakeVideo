import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
from pathlib import Path
from tqdm import tqdm

import src.config.train_opt as opt
from src.inpaint.inpaint_model import InpaintCAModel
from src.inpaint.dataset import Dataset

save_dir = Path('./data/target/train/')
save_dir.mkdir(exist_ok=True)

fill_dir = save_dir.joinpath('fill_bg')
fill_dir.mkdir(exist_ok=True)


if __name__ == "__main__":
    FLAGS = ng.Config('./src/inpaint/inpaint.yml')

    model = InpaintCAModel()
    dataset = Dataset(opt)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        X = tf.placeholder(dtype=tf.float32, shape=(1, 512, 1360, 3))
        output = model.build_server_graph(FLAGS, X)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable('./src/inpaint/model_logs/release_places2_256', from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        for idx, input_image in enumerate(tqdm(dataset)):
            result = sess.run(output, feed_dict={X: input_image})
            cv2.imwrite(str(fill_dir.joinpath('{:05}.png'.format(idx))), result[0][:, :, ::-1][:, 84:84+512, :])
