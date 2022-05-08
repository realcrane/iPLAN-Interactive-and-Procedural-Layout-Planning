from torch.nn.utils.rnn import pad_packed_sequence
from collections import OrderedDict
from PIL import Image
import numpy as np
import torch as t

room_label = [(0, 'LivingRoom'),
              (1, 'MasterRoom'),
              (2, 'Kitchen'),
              (3, 'Bathroom'),
              (4, 'DiningRoom'),
              (5, 'ChildRoom'),
              (6, 'StudyRoom'),
              (7, 'SecondRoom'),
              (8, 'GuestRoom'),
              (9, 'Balcony'),
              (10, 'Entrance'),
              (11, 'Storage'),
              (12, 'Wall-in'),
              (13, 'External'),
              (14, 'ExteriorWall'),
              (15, 'FrontDoor'),
              (16, 'Interior')]
category = [category for category in room_label if category[1] not in set(['External', \
                                                                           'ExteriorWall', 'FrontDoor', 'Interior'])]

num_category = len(category)
total_num_category = len(room_label)


def log(file, msg='', is_print=True):
    if is_print:
        print(msg)
    file.write(msg + '\n')
    file.flush()


def get_color_map():
    color = np.array([
        [244, 242, 229],  # living room
        [230, 214, 130],  # [253,244,171], # bedroom
        [224, 134, 131],  # [234,216,214], # kitchen
        [144, 188, 219],  # [205,233,252], # bathroom
        [147, 190, 171],  # [208,216,135], # balcony
        [225, 175, 131],  # [249,222,189], # Storage
        [79, 79, 79],  # exterior wall
        [255, 225, 25],  # FrontDoor
        [128, 128, 128],  # interior
        [255, 255, 255]
    ], dtype=np.int64)
    cIdx = np.array([1, 2, 3, 4, 1, 2, 2, 2, 2, 5, 1, 6, 1, 10, 7, 8, 9, 10]) - 1
    return color[cIdx]


def colorize_mask(img):
    palette = get_color_map().reshape(-1).tolist()
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    new_img = Image.fromarray(img.astype(np.uint8)).convert('P')
    new_img.putpalette(palette)
    new_img = new_img.convert('RGB')
    return new_img


def get_fp_samples(packed_r_boxes, data):
    fp_samples = []
    b_imgs = data[0]
    insides = data[3]
    unsorted_index = packed_r_boxes[3]
    packed_r_types = data[2]
    packed_fp_states = data[5]

    packed_r_boxes, sizes = pad_packed_sequence(packed_r_boxes, batch_first=True)
    packed_r_types, sizes = pad_packed_sequence(packed_r_types, batch_first=True)
    packed_fp_states, sizes = pad_packed_sequence(packed_fp_states, batch_first=True)

    r_ind = np.random.permutation(packed_r_boxes.size(0))   # randomly choose 32 floorplans to show
    r_ind = r_ind[:32]

    for r_i in r_ind:
        i = unsorted_index[r_i]
        pred_fp = b_imgs[i, :, :, :].squeeze()
        inside = insides[i, :, :, :].squeeze()

        gt_fp = packed_fp_states[r_i, sizes[r_i]-1,:,:].squeeze()

        r_boxes = packed_r_boxes[r_i, :sizes[r_i], :]
        r_types = packed_r_types[r_i, :sizes[r_i]]
        for i in range(r_boxes.size(0)):
            r_box = r_boxes[i, :] * 127
            r_box = r_box.long()
            r_type = r_types[i]
            r_mask = t.zeros(inside.size())
            r_mask[r_box[0]:r_box[2] + 1, r_box[1]:r_box[3] + 1] = 1
            r_mask = r_mask * inside
            pred_fp = pred_fp * (1 - r_mask) + r_mask * r_type
        fp_samples.append(gt_fp)
        fp_samples.append(pred_fp)

    return get_images(fp_samples)

def get_images(samples):
    images = []
    for img in samples:
        img = img.squeeze().cpu().numpy()
        new_img = colorize_mask(img)
        new_img = 255 - np.array(new_img)
        images.append(new_img)
    return t.FloatTensor(np.stack(images, 0))

class Loss_AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, loss_name):
        self.loss_names = loss_name
        self.reset()

    def reset(self):
        """Reset the loss value"""
        self.avg = OrderedDict()
        self.sum = OrderedDict()
        self.val = OrderedDict()
        self.count = 0

        for name in self.loss_names:
            self.avg[name] = 0.0
            self.sum[name] = 0.0
            self.val[name] = 0.0

    def update(self, losses, n):
        self.count += n
        for name, value in losses.items():
            self.val[name] = value / n
            self.sum[name] += value
            self.avg[name] = self.sum[name] / self.count
