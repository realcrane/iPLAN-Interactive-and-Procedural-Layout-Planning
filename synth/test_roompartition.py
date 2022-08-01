from synth.floorplan.roompartition_fp import FloorPlan
from room_partition.models.loss_layer import LossFun
from room_partition.utils import colorize_mask
from room_partition import models
import scipy.io as sio
import numpy as np
import torch
import os

def getList(path):
    mat_list = os.listdir(path)
    mat_list.sort()
    temp_list = []
    for name in mat_list[100:200]:
        mat_path = os.path.join(path, name)
        data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)['data']
        if len(data.rBoxes) == 0 and len(data.rCenters) !=0:
            temp_list.append(name)
    return temp_list


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_netG_path', type=str, default='../room_partition/weights/G_net_210.pth',
                        help='path of net G')
    parser.add_argument('--max_iter', type=int, default=200, help='maximum of iteration')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0], help='id of GPU')
    parser.add_argument('--lr', type=float, default=10)
    parser.add_argument('--coverage', type=float, default=1)
    parser.add_argument('--inside', type=float, default=0.5)
    parser.add_argument('--mutex', type=float, default=0)
    parser.add_argument('--root', type=str, default='../data', help='id of GPU')
    opt = parser.parse_args()

    # step1: configure model
    print('Building model...')
    fp_rnn = models.FloorPlanRNNTest(opt)
    fp_rnn.load_networks(opt)
    criterion = LossFun(device=fp_rnn.device)

    # step2: data
    print('Building dataset...')
    data_root = os.path.join(opt.root, 'test')
    floorplans = getList(data_root)
    print('The length of testing data is {}'.format(len(floorplans)))

    # step3: testing
    print('Starting to Testing...')
    for i in range(len(floorplans)):
        fp_name = floorplans[i]
        fp_path = os.path.join(data_root, fp_name)
        fp = FloorPlan(fp_path)
        img, rCenters, rTypes, inside, boundary = fp.get_input()
        fp_rnn.evaluate(img, rCenters, rTypes, inside)

        boundary = boundary[:, [1, 0, 2, 3]]
        pred_rBoxes = fp_rnn.pred_rBoxes
        pred_rBoxes = pred_rBoxes[:, [1, 0, 3, 2]]

        layout = get_image(img, pred_rBoxes, rTypes, inside)
        living_box, layout, living_mask = obtain_living(layout, boundary)

        criterion.get_initial_mutex(pred_rBoxes, boundary)

        iter = 0
        while get_ratio(layout) > 0.0005 and iter < opt.max_iter:
            iter = iter + 1
            pred_rBoxes = fine_turning(pred_rBoxes, boundary, living_mask, criterion, opt)
            layout = get_image(img, pred_rBoxes, rTypes, inside, living_mask)
        image = colorize_mask(layout.numpy())
        image.save(f'image/{fp.name}.png')
        print(fp.name)

        fp.update_rBoxes(pred_rBoxes.detach().cpu().numpy())
        data = fp.to_dict()
        # sio.savemat(fp_path, {'data': data})

def fine_turning(pred_rBoxes, boundary, living_mask, criterion, opt):
    lr = opt.lr
    pred_rBoxes = torch.autograd.Variable(pred_rBoxes, requires_grad=True)
    loss = criterion(pred_rBoxes, boundary, living_mask, opt)
    loss.backward()
    grad = pred_rBoxes.grad
    pred_rBoxes = pred_rBoxes - lr * grad
    return pred_rBoxes

def get_ratio(layout):
    spare_area = (layout == 16).sum()
    inside_area = (layout < 13).sum() + spare_area
    ratio = spare_area / inside_area
    return ratio

def obtain_living(layout, boundary):
    living_mask = layout.clone()
    living_mask[living_mask < 16] = 0
    min_y, min_x = np.min(boundary[:, :2], 0)
    max_y, max_x = np.max(boundary[:, :2], 0)
    for h in range(int(min_x) - 5, int(max_x) + 5):
        for w in range(int(min_y) - 5, int(max_y) + 5):
            if living_mask[h, w] == 0 and living_mask[h + 1, w] == 16 and living_mask[h + 2, w] == 0:
                living_mask[h + 1, w] = 0
            elif living_mask[h, w] == 0 and living_mask[h + 1, w] == 16 and living_mask[h + 2, w] == 16 and living_mask[
                h + 3, w] == 0:
                living_mask[h + 1, w] = 0
                living_mask[h + 2, w] = 0
            elif living_mask[h, w] == 0 and living_mask[h + 1, w] == 16 and living_mask[h + 2, w] == 16 and living_mask[
                h + 3, w] == 16 and living_mask[h + 4, w] == 0:
                living_mask[h + 1, w] = 0
                living_mask[h + 2, w] = 0
                living_mask[h + 3, w] = 0

            if living_mask[h, w] == 0 and living_mask[h, w + 1] == 16 and living_mask[h, w + 2] == 0:
                living_mask[h, w + 1] = 0
            elif living_mask[h, w] == 0 and living_mask[h, w + 1] == 16 and living_mask[h, w + 2] == 16 and living_mask[
                h, w + 3] == 0:
                living_mask[h, w + 1] = 0
                living_mask[h, w + 2] = 0
            elif living_mask[h, w] == 0 and living_mask[h, w + 1] == 16 and living_mask[h, w + 2] == 16 and living_mask[
                h, w + 3] == 16 and living_mask[h, w + 4] == 0:
                living_mask[h, w + 1] = 0
                living_mask[h, w + 2] = 0
                living_mask[h, w + 3] = 0
    layout[living_mask==16]=0
    index = torch.where(living_mask)
    min_x, max_x = torch.min(index[0]),torch.max(index[0])
    min_y, max_y = torch.min(index[1]), torch.max(index[1])
    box = torch.stack([min_y, min_x, max_y, max_x]) / 127
    return box, layout, living_mask/16

def get_image(img, rBoxes, rTypes, inside, living_mask=None):
    if not living_mask==None:
        img[living_mask==1]=0
    for r_i in range(len(rTypes)):
        r_box = rBoxes[r_i, :] * 127
        r_box = r_box.long()
        r_type = rTypes[r_i]
        mask = torch.zeros(inside.size())
        mask[r_box[1]:r_box[3] + 1, r_box[0]:r_box[2] + 1] = 1
        mask = mask * inside
        img = img * (1 - mask) + mask * r_type
    return img

if __name__ == '__main__':
    main()
