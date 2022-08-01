from .net import RendererNet, BoundingBoxNet, NLayerDiscriminator, GANLoss
from .net import cal_gradient_penalty
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn as nn
import torch as t
from room_partition import utils
import os


class FloorPlanRNN(nn.Module):
    def __init__(self, opt):
        super(FloorPlanRNN, self).__init__()
        self.device = t.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else t.device('cpu')

        # configure model
        self.netG = BoundingBoxNet().to(self.device)
        self.netR = RendererNet().to(self.device)

    def forward(self, img, rCenters, rTypes, inside):
        pred_rBoxes_list = []
        pred_states_list = []

        num = len(rTypes)
        state = img.unsqueeze(0).unsqueeze(0)
        for n in range(num):
            r_center = rCenters[n:n + 1, :, :].unsqueeze(0)
            r_type = rTypes[n:n + 1]
            output, state = self.sub_step(state, r_center, r_type, inside)
            pred_rBoxes_list.append(output)
            pred_states_list.append(state)
        self.pred_rBoxes = t.cat(pred_rBoxes_list, 0)  # tensor
        self.pred_states = t.cat(pred_states_list, 0)  # tensor

    def sub_step(self, states, centers, types, insides):
        types = types.view(-1, 1, 1, 1).expand_as(states)
        centers = centers.view(states.size())
        norm_states = self.normalize(states)
        input = t.cat([norm_states, centers, types / (utils.num_category - 1)], dim=1)
        input = input.contiguous()
        r_boxes = self.netG(input)
        r_masks = self.netR(r_boxes)
        r_masks = r_masks * insides
        new_states = states * (1 - r_masks) + r_masks * (types)
        return r_boxes, new_states

    def evaluate(self, img, rCenters, rTypes, inside):
        img = img.to(self.device).to(t.float32)
        rCenters = rCenters.to(self.device).to(t.float32)
        rTypes = rTypes.to(self.device).to(t.float32)
        inside = inside.to(self.device).to(t.float32)

        with t.no_grad():
            self.forward(img, rCenters, rTypes, inside)  # compute fake images: G(A)

    def normalize(self, data):
        data = t.div(data, utils.total_num_category - 1)
        data = t.sub(data, 0.5)
        data = t.div(data, 0.5)
        return data

    def load_networks(self, opt):
        if opt.load_netG_path:
            self.netG.load_state_dict(t.load(opt.load_netG_path))
