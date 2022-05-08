from .net import RendererNet, BoundingBoxNet, NLayerDiscriminator, GANLoss
from .net import cal_gradient_penalty
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn as nn
import torch as t
import utils
import os


class FloorPlanRNN(nn.Module):
    def __init__(self, opt, log_file):
        super(FloorPlanRNN, self).__init__()
        # specify the training losses you want to print out.
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'D', 'GP']
        self.model_names = ['G', 'D']

        self.isTrain = True
        self.lambda_L1 = opt.lambda_L1
        self.save_dir = opt.checkpoints_dir
        self.device = t.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else t.device('cpu')

        # configure model
        self.netG = BoundingBoxNet().to(self.device)
        self.netR = RendererNet().to(self.device)
        self.netD = NLayerDiscriminator().to(self.device)
        self.print_networks(log_file)

        # criterion and optimizer
        utils.log(log_file, 'Building criterion and optimizer...')
        lr = opt.lr_base
        self.optimizers = []
        self.optimizer_G = t.optim.Adam(self.netG.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
        self.optimizer_D = t.optim.Adam(self.netD.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.criterionGAN = GANLoss('wgangp').to(self.device)
        self.criterionL1 = t.nn.SmoothL1Loss(reduction='sum').to(self.device)

    def set_input(self, input):
        b_imgs, packed_r_centers, packed_r_types, insides, packed_r_boxes, packed_fp_states, _, data_ids = input
        self.data_ids = data_ids
        self.b_imgs = b_imgs.to(self.device)
        self.packed_r_centers = packed_r_centers.to(self.device)
        self.packed_r_types = packed_r_types.to(self.device)
        self.insides = insides.to(self.device)
        self.gt_packed_r_boxes = packed_r_boxes.to(self.device)
        self.gt_packed_fp_states = packed_fp_states.to(self.device)

        self.step_sizes = self.packed_r_centers[1]
        self.sorted_indices = self.packed_r_centers[2]
        self.unsorted_indices = self.packed_r_centers[3]
        self.fp_num = self.b_imgs.size(0)
        self.room_num = self.step_sizes.sum()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netG, False)
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights

        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def forward(self):
        packed_r_centers = self.packed_r_centers.data
        packed_r_types = self.packed_r_types.data
        states = self.b_imgs
        insides = self.insides
        step_sizes = self.step_sizes
        num = len(step_sizes)

        pred_r_boxes_list = []
        pred_fp_states_list = []

        ind = 0
        for n in range(num):
            sz = step_sizes[n]
            states = states[:sz, :, :, :]
            r_centers = packed_r_centers[ind: ind + sz, :, :]
            r_types = packed_r_types[ind: ind + sz]
            insides = insides[:sz, :, :, :]
            output, states = self.sub_step(states, r_centers, r_types, insides)
            pred_r_boxes_list.append(output)
            pred_fp_states_list.append(states)
            ind = t.sum(step_sizes[:n + 1])
        self.pred_packed_r_boxes = t.cat(pred_r_boxes_list, 0)  # tensor
        self.pred_packed_fp_states = t.cat(pred_fp_states_list, 0)  #tensor


    def sub_step(self, states, centers, types, insides):
        types = types.view(-1, 1, 1, 1).expand_as(states)
        centers = centers.view(states.size())
        norm_states = self.normalize(states)
        input = t.cat([norm_states, centers, types / (utils.num_category - 1)], dim=1)
        input = input.contiguous()
        if self.isTrain:
            input = Variable(input, requires_grad=True)
        r_boxes = self.netG(input)
        r_masks = self.netR(r_boxes)
        r_masks = r_masks * insides
        new_states = states * (1 - r_masks) + r_masks * (types)
        return r_boxes, new_states

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fp_state_fake = self.normalize(self.pred_packed_fp_states)
        pred_state_fake = self.netD(fp_state_fake)
        self.loss_D_fake = self.criterionGAN(pred_state_fake, False)

        # Real
        fp_state_real = self.normalize(self.gt_packed_fp_states.data)
        fp_state_real = fp_state_real.unsqueeze(1)
        gt_state_real = self.netD(fp_state_real)
        self.loss_D_real = self.criterionGAN(gt_state_real, True)

        # combine loss and calculate gradients
        self.loss_GP = cal_gradient_penalty(self.netD, fp_state_real, fp_state_fake, self.device)
        self.loss_D = self.loss_D_fake + self.loss_D_real + self.loss_GP
        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fp_state_fake = self.normalize(self.pred_packed_fp_states)
        pred_state_fake = self.netD(fp_state_fake)
        self.loss_G_GAN = self.criterionGAN(pred_state_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.pred_packed_r_boxes, self.gt_packed_r_boxes.data) / self.room_num

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.lambda_L1
        self.loss_G.backward()

    def evaluate(self):
        with t.no_grad():
            self.forward()  # compute fake images: G(A)

        # discriminator loss
        fp_state_fake = self.normalize(self.pred_packed_fp_states)
        pred_state_fake = self.netD(fp_state_fake)
        self.loss_D_fake = self.criterionGAN(pred_state_fake, False)

        # Real
        fp_state_real = self.normalize(self.gt_packed_fp_states.data)
        fp_state_real = fp_state_real.unsqueeze(1)
        gt_state_real = self.netD(fp_state_real)
        self.loss_D_real = self.criterionGAN(gt_state_real, True)

        # combine loss and calculate gradients
        self.loss_GP = cal_gradient_penalty(self.netD, fp_state_real, fp_state_fake, self.device, phrase='test')
        self.loss_D = self.loss_D_fake + self.loss_D_real + self.loss_GP

        # generator loss
        self.loss_G_GAN = self.criterionGAN(pred_state_fake, True)
        self.loss_G_L1 = self.criterionL1(self.pred_packed_r_boxes, self.gt_packed_r_boxes.data) / self.room_num
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.lambda_L1

    def train(self):
        self.netG.train()
        self.netD.train()
        self.netR.eval()
        self.isTrain = True

    def eval(self):
        self.netG.eval()
        self.netD.eval()
        self.netR.eval()
        self.isTrain = False

    def set_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def normalize(self, data):
        data = t.div(data, utils.total_num_category - 1)
        data = t.sub(data, 0.5)
        data = t.div(data, 0.5)
        return data

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
                errors_ret[name] = errors_ret[name]
        return errors_ret

    def print_networks(self, log_file):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if True:
                    print(net)
                utils.log(log_file, '[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        utils.log(log_file, '-----------------------------------------------')

    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (name, epoch)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                t.save(net.state_dict(), save_path)

    def load_networks(self, opt, log_file):
        if opt.load_netG_path:
            utils.log(log_file, 'Loading the model: {}'.format(opt.load_netG_path))
            self.netG.load_state_dict(t.load(opt.load_netG_path))
        if opt.load_netD_path:
            utils.log(log_file, 'Loading the model: {}'.format(opt.load_netD_path))
            self.netD.load_state_dict(t.load(opt.load_netD_path))