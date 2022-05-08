from PIL import Image
import numpy as np
import torch as t
import random
import copy
import scipy.io as sio
import cv2

class LoadFloorplanTrain():
    """
    Loading a floorplan for train
    """ 
    def __init__(self, path, random_shuffle=True):
        "load floorplan from mat file"
        data = copy.deepcopy(sio.loadmat(path, squeeze_me=True, struct_as_record=False))['data']
        exterior_boundary = data.Boundary

        img = self.init_input_img(exterior_boundary)
        h, w = img.shape

        inside = np.zeros((h, h))
        inside[img == 16] = 1
        inside[img < 13] = 1

        boundary = np.zeros((h, h))
        boundary[img == 14] = 1
        boundary[img == 15] = 0.5

        front_door = np.zeros((h, h))
        front_door[img == 15] = 1

        "inside_mask"
        self.inside_mask = t.from_numpy(inside)

        "boundary_mask"
        self.boundary_mask = t.from_numpy(boundary)

        "front_door_mask"
        self.front_door_mask = t.from_numpy(front_door)


        rBoxes = data.gt_rBoxes
        rTypes = data.gt_rTypes
        room_node=[]
        for i in range(len(rTypes)):
            node = {}
            node['category'] = int(rTypes[i])
            x = (rBoxes[i, 0] + rBoxes[i, 2]) / 2.0
            y = (rBoxes[i, 1] + rBoxes[i, 3]) / 2.0
            node['centroid'] = (x, y)
            room_node.append(node)

        "randomly order rooms"
        if random_shuffle:
            random.shuffle(room_node)

        self.data_size = h
        self.continue_node = []
        for node in room_node:
            if node['category'] == 0:
                self.living_node = node
            else:
                self.continue_node.append(node)

    def get_composite_living(self, num_extra_channels=0):
        composite = t.zeros((num_extra_channels+3, self.data_size, self.data_size))  
        composite[0] = self.inside_mask
        composite[1] = self.boundary_mask
        composite[2] = self.front_door_mask
        return composite

    def init_input_img(self, boundary):
        """
           generate the initial boundary image from exterior boundary
        """
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