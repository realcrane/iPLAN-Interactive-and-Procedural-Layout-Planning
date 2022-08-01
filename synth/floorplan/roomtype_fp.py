import torch
import scipy.io as sio
import numpy as np
import cv2
from room_type import utils

class FloorPlan():
    def __init__(self, fp_path):
        data = sio.loadmat(fp_path, squeeze_me=True, struct_as_record=False)['data']
        self.gt_rTypes = data.gt_rTypes
        self.gt_rBoxes = data.gt_rBoxes
        self.rTypes = data.rTypes
        self.rBoxes = data.rBoxes
        self.rCenters = data.rCenters
        self.name = data.name
        self.exterior_boundary = data.Boundary

    def normalize(self, data):
        data = torch.div(data, utils.total_num_category-1)
        data = torch.sub(data, 0.5)
        data = torch.div(data, 0.5)
        return data

    def init_input_img(self, boundary):
        boundary = boundary.astype(np.int)
        boundary = boundary[:, [1, 0, 2, 3]]

        image = np.ones((128, 128)) * 13
        image = cv2.polylines(image, boundary[:, :2].reshape(1, -1, 2), True, 14, 5)

        for w, h in boundary[:, :2]:
            image[h - 3:h + 4, w - 3:w + 4] = 14

        if boundary[0, 2] == 0:
            image[boundary[0, 1] - 3:boundary[1, 1], boundary[0, 0]: boundary[1, 0]] = 15
        elif boundary[0, 2] == 1:
            image[boundary[0, 1]:boundary[1, 1], boundary[0, 0] + 1: boundary[1, 0] + 4] = 15
        elif boundary[0, 2] == 2:
            image[boundary[0, 1] + 1:boundary[1, 1] + 4, boundary[1, 0]: boundary[0, 0]] = 15
        elif boundary[0, 2] == 3:
            image[boundary[1, 1]:boundary[0, 1], boundary[0, 0] - 3: boundary[1, 0]] = 15

        image = cv2.fillPoly(image, boundary[:, :2].reshape(1, -1, 2), 16)
        return image

    def update_rTypes(self, sample, max_room):
        sample[sample > 0.7] = 1
        sample[sample <= 0.7] = 0
        rTypes = []

        ind = 0
        for n in range(13):
            num = np.sum(sample[ind: ind + max_room[n]])
            for i in range(int(num)):
                rTypes.append(n)
            ind = ind + max_room[n]

        self.rTypes = np.array(rTypes)

    def to_dict(self, dtype=int):
        '''
        Compress data, notice:
        '''
        return {
            'name': self.name,
            'gt_rTypes': self.gt_rTypes.astype(dtype),
            'gt_rBoxes': self.gt_rBoxes.astype(dtype),
            'Boundary': self.exterior_boundary.astype(dtype),
            'rTypes': self.rTypes.astype(dtype),
            'rBoxes': self.rBoxes.astype(dtype),
            'rCenters': self.rCenters.astype(dtype)
        }
