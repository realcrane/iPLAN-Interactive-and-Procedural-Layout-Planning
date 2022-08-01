import torch
import scipy.io as sio
import numpy as np
import cv2

class FloorPlan():
    def __init__(self, fp_path, mask_size=4):
        data = sio.loadmat(fp_path, squeeze_me=True, struct_as_record=False)['data']
        self.gt_rTypes = data.gt_rTypes
        self.gt_rBoxes = data.gt_rBoxes
        self.rTypes = data.rTypes
        self.rCenters = data.rCenters
        self.rBoxes = data.rBoxes
        self.name = data.name
        self.exterior_boundary = data.Boundary

        img = self.init_input_img(self.exterior_boundary)
        h, w = img.shape
        self.data_size = h
        self.mask_size = mask_size

        inside = np.zeros((h, h))
        inside[img == 16] = 1
        inside[img < 13] = 1

        "inside_mask"
        self.inside_mask = torch.from_numpy(inside)
        self.img = torch.from_numpy(img)

    def get_input(self):
        ind = self.rTypes == 0
        self.continue_rTypes = self.rTypes[~ind]
        self.continue_rCenters = self.rCenters[~ind]

        rCenters=[]
        for i in range(len(self.continue_rTypes)):
            r_c = np.zeros((128, 128))
            c_x, c_y = self.continue_rCenters[i,:]
            r_c[c_x - 1:c_x + 2, c_y - 1:c_y + 2] = 1
            rCenters.append(r_c)

        rCenters = np.stack(rCenters, 0)
        rCenters = torch.FloatTensor(rCenters)

        return self.img, rCenters, torch.from_numpy(self.continue_rTypes), self.inside_mask, self.exterior_boundary

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


    def update_rBoxes(self, rBoxes):
        self.rBoxes = rBoxes

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
