import os
import numpy as np
import torch
import time
import sys
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn as nn
from pathlib import Path
import warnings
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
warnings.filterwarnings('ignore')
mainpath = os.getcwd()
pix2pixhd_dir = Path(mainpath + '/src/pix2pixHD/')
sys.path.append(str(pix2pixhd_dir))

from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import src.config.train_opt as opt
import src.utils.cv_utils as cv_utils

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True

opt.checkpoints_dir += 'final/'
opt.target = 'final'
def freeze_plane2(model):
    for param in model.parameters():
        param.requires_grad = False
def main():
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    start_epoch, epoch_iter = 1, 0
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq

    model_final = create_model(opt)
    model_final = model_final.cuda()
    visualizer = Visualizer(opt)

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        for i, data in enumerate(dataset, start=epoch_iter):
            torch.cuda.empty_cache()
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == display_delta

            losses, generated = model_final(Variable(data['label']), Variable(data['inst']), None,
                                            Variable(data['image']), Variable(data['feat']), infer=save_fake,
                                            synbg=Variable(data['synbg']), synfg=Variable(data['synfg']))

            ###
            # import torchvision
            # res_net = torchvision.models.resnet18(pretrained=True)
            # res_net.cuda()
            # freeze_plane2(res_net)
            # # new_gen = torch.nn.functional.interpolate(generated, size = (328, 328), mode='bilinear')
            # # new_gt = torch.nn.functional.interpolate(data['gt_img'], size = (328, 328), mode='bilinear')
            # new_gen = generated.cuda()
            # new_gt = data['gt_img'].cuda()
            # gen_fearture = res_net(new_gen)
            # gt_fearture = res_net(new_gt)
            # ####
            #
            # # cri_mse = torch.nn.MSELoss()
            # # loss_feature_alignment = cri_mse(gen_fearture, gt_fearture)
            #
            # print("epoch:%d,i:%d,loss_feature_alignment:%f" % (epoch, i, loss_feature_alignment * 100))
            ###
            ####
            import torchvision
            import optimal_transport_loss as ot
            opt_loss = ot.FeatureOptimalLoss()
            res_net = torchvision.models.resnet18(pretrained=True)
            res_net.cuda()
            freeze_plane2(res_net)
            # new_gen = torch.nn.functional.interpolate(generated, size = (328, 328), mode='bilinear')
            # new_gen = torch.nn.functional.interpolate(generated, scale_factor = 2, mode='bilinear')
            # new_gt = torch.nn.functional.interpolate(data['image'], scale_factor = 2, mode='nearest')
            # new_gt = torch.nn.functional.interpolate(data['gt_img'], size = (328, 328), mode='bilinear')
            new_gen = generated.cuda()
            new_gt = data['gt_img'].cuda()
            gen_fearture = res_net(new_gen)
            gt_fearture = res_net(new_gt)

            ####

            # cri_mse = torch.nn.MSELoss()

            # loss_feature_alignment = cri_mse(gen_fearture, gt_fearture)
            loss_feature_alignment = opt_loss(gen_fearture, gt_fearture)
            print("epoch:%d,i:%d,loss_feature_alignment:%f" % (epoch, i, loss_feature_alignment / 100))
            ####
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model_final.loss_names, losses))

            loss_D_final = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            # print("loss_dict['G_GAN']:%f,loss_dict.get('G_GAN_Feat', 0):%f,loss_dict.get('G_VGG', 0):%f,loss_feature_alignment:%f"% (loss_dict['G_GAN'],loss_dict.get('G_GAN_Feat', 0),loss_dict.get('G_VGG', 0),loss_feature_alignment/100))
            loss_G_final = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)+loss_feature_alignment / 100

            model_final.optimizer_G.zero_grad()
            loss_G_final.backward()
            model_final.optimizer_G.step()
            model_final.optimizer_D.zero_grad()
            loss_D_final.backward()
            model_final.optimizer_D.step()

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            ### display output images
            if save_fake:
                visuals = OrderedDict(
                    [('input_label', util.tensor2label(data['label'][:, :, 0, :, :][0], opt.label_nc)),
                     ('input_bg', util.tensor2im(data['synbg'][0])),
                     ('input_fg', util.tensor2im(data['synfg'][0])),
                     ('synthesized_image', util.tensor2im(generated.data[0])),
                     ('real_image', util.tensor2im(data['image'][:, :, 0, :, :][0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model_final.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break

        # end of epoch
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model_final.save('latest')
            model_final.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model_final.update_fixed_params()

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model_final.update_learning_rate()

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
