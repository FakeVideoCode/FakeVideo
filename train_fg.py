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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#
warnings.filterwarnings('ignore')
mainpath = os.getcwd()
pix2pixhd_dir = Path(mainpath + '/src/pix2pixHD/')
sys.path.append(str(pix2pixhd_dir))

from enhancer.src_enhance.pix2pixHD.data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import src.config.train_opt as opt
import src.utils.cv_utils as cv_utils

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True

opt.checkpoints_dir += 'fg/'
opt.target = 'fg'


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

    model_fg = create_model(opt)
    # model_fg = nn.DataParallel(model_fg)
    model_fg = model_fg.cuda()
    visualizer = Visualizer(opt)

    ###
    checkpoint_folder = os.path.join(opt.checkpoints_dir, opt.name)

    files = [file_name for file_name in next(os.walk(checkpoint_folder))[2]]

    files = [file_name for file_name in files if file_name.endswith("pth")]

    max_epoch = 0
    for file_name in files:
        temp_epoch = file_name.split("_")[0]
        if temp_epoch != "latest":
            epoch_num = int(temp_epoch)
            max_epoch = epoch_num if epoch_num > max_epoch else max_epoch

    if max_epoch != 0:
        start_epoch = max_epoch
        Dfg_checkpoint_path = os.path.join(checkpoint_folder, f"{max_epoch}_net_Dfg.pth")
        Dtfg_checkpoint_path = os.path.join(checkpoint_folder, f"{max_epoch}_net_Dtfg.pth")
        Gfg_checkpoint_path = os.path.join(checkpoint_folder, f"{max_epoch}_net_Gfg.pth")

        D_optimizer_path = os.path.join(checkpoint_folder, f"{max_epoch}_optimizer_D.pth")
        Dt_optimizer_path = os.path.join(checkpoint_folder, f"{max_epoch}_optimizer_Dt.pth")
        G_optimizer_path = os.path.join(checkpoint_folder, f"{max_epoch}_optimizer_G.pth")
        model_fg.netD.load_state_dict(torch.load(Dfg_checkpoint_path))
        model_fg.netDt.load_state_dict(torch.load(Dtfg_checkpoint_path))
        model_fg.netG.load_state_dict(torch.load(Gfg_checkpoint_path))
        model_fg.optimizer_D.load_state_dict(torch.load(D_optimizer_path))
        model_fg.optimizer_Dt.load_state_dict(torch.load(Dt_optimizer_path))
        model_fg.optimizer_G.load_state_dict(torch.load(G_optimizer_path))

    ###

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        random = 15
        threshold = 20
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        for i, data in enumerate(dataset, start=epoch_iter):
            # torch.cuda.empty_cache()
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == display_delta

            losses, generated = model_fg(Variable(data['label']), Variable(data['inst']), Variable(data['mask']),
                                         Variable(data['image']), Variable(data['feat']), infer=True)
            ###
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
            print("epoch:%d,i:%d,loss_feature_alignment:%f"%(epoch,i,loss_feature_alignment / 100) )
            ###
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model_fg.loss_names, losses))

            loss_D_fg = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_Dt_fg = (loss_dict['Dt_fake'] + loss_dict['Dt_real']) * 0.5
            loss_G_fg = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0) + loss_dict[
                'Dt_fake'] * 0.25 + loss_feature_alignment / 100

            model_fg.optimizer_G.zero_grad()
            loss_G_fg.backward(retain_graph=True)
            model_fg.optimizer_G.step()
            model_fg.optimizer_D.zero_grad()
            loss_D_fg.backward()
            model_fg.optimizer_D.step()
            model_fg.optimizer_Dt.zero_grad()
            loss_Dt_fg.backward()
            model_fg.optimizer_Dt.step()

            if i%random ==0 or loss_G_fg< threshold:
                losses, generated = model_fg(Variable(data['label']), Variable(data['inst']), Variable(data['mask']),
                                             Variable(data['image']), Variable(data['feat']), infer=True)
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
                ###
                losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
                loss_dict = dict(zip(model_fg.loss_names, losses))

                loss_D_fg = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
                loss_Dt_fg = (loss_dict['Dt_fake'] + loss_dict['Dt_real']) * 0.5
                loss_G_fg = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0) + loss_dict[
                    'Dt_fake'] * 0.25 + loss_feature_alignment / 100

                model_fg.optimizer_G.zero_grad()
                loss_G_fg.backward(retain_graph=True)
                model_fg.optimizer_G.step()
                model_fg.optimizer_D.zero_grad()
                loss_D_fg.backward()
                model_fg.optimizer_D.step()
                model_fg.optimizer_Dt.zero_grad()
                loss_Dt_fg.backward()
                model_fg.optimizer_Dt.step()




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
                     ('synthesized_fg', util.tensor2im(generated.data[0])),
                     ('real_fg', util.tensor2im((data['image'][:, :, 0, :, :] * data['mask'])[0])),
                     ('real_image', util.tensor2im(data['image'][:, :, 0, :, :][0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))

                model_fg.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break

        # end of epoch
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model_fg.save('latest')
            model_fg.save(epoch)

            optimizer_D = model_fg.optimizer_D
            optimizer_Dt = model_fg.optimizer_Dt
            optimizer_G = model_fg.optimizer_G

            torch.save(optimizer_D.state_dict(), f"{epoch}_optimizer_D.pth")
            torch.save(optimizer_Dt.state_dict(), f"{epoch}_optimizer_Dt.pth")
            torch.save(optimizer_G.state_dict(), f"{epoch}_optimizer_G.pth")


            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model_fg.update_fixed_params()

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model_fg.update_learning_rate()

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
