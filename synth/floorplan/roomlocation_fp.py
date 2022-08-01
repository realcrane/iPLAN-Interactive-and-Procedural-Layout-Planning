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
        ind = self.rTypes == 0
        self.continue_rTypes = self.rTypes[~ind]
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

        boundary = np.zeros((h, h))
        boundary[img == 14] = 1
        boundary[img == 15] = 0.5

        front_door = np.zeros((h, h))
        front_door[img == 15] = 1

        "inside_mask"
        self.inside_mask = torch.from_numpy(inside)

        "boundary_mask"
        self.boundary_mask = torch.from_numpy(boundary)

        "front_door_mask"
        self.front_door_mask = torch.from_numpy(front_door)

        "category_mask"
        self.category_mask = torch.zeros((13, self.data_size, self.data_size))

    def get_composite_living(self, num_extra_channels=0):
        composite = torch.zeros((num_extra_channels + 3, self.data_size, self.data_size))
        composite[0] = self.inside_mask
        composite[1] = self.boundary_mask
        composite[2] = self.front_door_mask
        return composite

    def get_composite_location(self, num_extra_channels=0, num_category=13):
        composite = torch.zeros((num_category + num_extra_channels + 4, self.data_size, self.data_size))
        composite[0] = self.inside_mask
        composite[1] = self.boundary_mask
        composite[2] = self.front_door_mask
        composite[3] = self.category_mask.sum(0)
        for i in range(num_category):
            composite[i + 4] = self.category_mask[i]
        return composite

    def add_room(self, room):
        index = self.label2index(room['category'])
        h, w = room['centroid']
        min_h = max(h - self.mask_size, 0)
        max_h = min(h + self.mask_size, self.data_size - 1)
        min_w = max(w - self.mask_size, 0)
        max_w = min(w + self.mask_size, self.data_size - 1)
        self.category_mask[index, min_h:max_h + 1, min_w:max_w + 1] = 1.0

    def label2index(self, label=0):
        if label < 0 or label > 17:
            raise Exception("Invalid label!", label)
        else:
            return label

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

    def update_rCenters(self, centers, types):
        centers = np.array(centers)
        types = np.array(types)
        index = np.argsort(types)
        centers = centers[index, :]
        types = types[index]

        self.rCenters = centers
        self.rTypes = types

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
