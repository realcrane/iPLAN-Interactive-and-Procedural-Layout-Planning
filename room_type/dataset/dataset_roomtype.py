import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import utils
import copy
import os
import cv2


class RoomTypeDataset(Dataset):
    def __init__(self, data_root, phase, max_room_per_type):
        self.data_split_root = os.path.join(data_root, phase)
        self.floorplans = os.listdir(self.data_split_root)
        self.max_room_per_type = max_room_per_type
        self.max_room_num = np.array(max_room_per_type).sum()
        self.len = self.__len__()

    def __len__(self):
        return len(self.floorplans)

    def __getitem__(self, index):
        assert index <= self.__len__(), 'index range error'
        floorplan_name = self.floorplans[index]
        floorplan_path = os.path.join(self.data_split_root, floorplan_name)
        floorplan = copy.deepcopy(sio.loadmat(floorplan_path, squeeze_me=True, struct_as_record=False))['data']

        boundary = floorplan.Boundary
        rTypes = floorplan.gt_rTypes

        input_img = self.init_input_img(boundary)
        input_vector = self.init_input_vector(rTypes)

        input_img = torch.FloatTensor(input_img).unsqueeze(0)
        input_img = self.normalize(input_img)
        input_vector = torch.FloatTensor(input_vector)

        return input_img, input_vector

    def normalize(self, data):
        data = torch.div(data, utils.total_num_category-1)
        data = torch.sub(data, 0.5)
        data = torch.div(data, 0.5)
        return data

    def init_input_vector(self, rTypes):
        input_vector = []
        for i in range(13):  # 13 is the total room types
            temp = np.zeros(self.max_room_per_type[i])
            num = (rTypes == i).sum()
            temp[:num] = 1
            input_vector.append(temp)

        input_vector = np.concatenate(input_vector, 0)
        return input_vector

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
