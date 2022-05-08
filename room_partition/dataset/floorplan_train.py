from PIL import Image
import numpy as np
import torch as t
import random
import pickle
import utils
import copy
import scipy.io as sio
import cv2


class LoadFloorplanTrain():
    """
    Loading a floorplan for train
    """

    def __init__(self, path, random_shuffle=False):
        "load floorplan from mat file"
        self.data = copy.deepcopy(sio.loadmat(path, squeeze_me=True, struct_as_record=False))['data']

    def get_input(self):
        data = self.data
        exterior_boundary = data.Boundary
        data_id = int(data.name)

        b_img = self.init_input_img(exterior_boundary)
        h, w = b_img.shape

        inside = np.zeros((h, h))
        inside[b_img == 16] = 1
        inside[b_img < 13] = 1

        rTypes = data.gt_rTypes
        rBoxes = data.gt_rBoxes

        # don't consider living room
        ind = rTypes > 0
        r_type_seq = rTypes[ind]  # the sequence of room type in a layout
        r_box_seq = rBoxes[ind, :]  # the sequence of room bounding box in a layout
        r_partition_seq = []  # the sequence of room partition in a layout
        r_center_seq = []  # the sequence of room location in a layout
        fp_state_seq = []

        fp_state = b_img
        for i in range(len(r_type_seq)):
            r_partition = np.zeros((128, 128))
            r_center = np.zeros((128, 128))
            tl_x, tl_y, br_x, br_y = r_box_seq[i, :]
            r_type = r_type_seq[i:i + 1]
            r_partition[tl_x:br_x + 1, tl_y:br_y + 1] = 1
            r_partition = r_partition * inside
            fp_state = (1 - r_partition) * fp_state + r_partition * r_type

            c_x = (br_x + tl_x) // 2
            c_y = (br_y + tl_y) // 2
            r_center[c_x - 1:c_x + 2, c_y - 1:c_y + 2] = 1

            r_partition_seq.append(r_partition)
            r_center_seq.append(r_center)
            fp_state_seq.append(fp_state)

        fp_state_seq = np.stack(fp_state_seq, 0)
        fp_state_seq = t.FloatTensor(fp_state_seq).view(-1, 128, 128)
        r_center_seq = np.stack(r_center_seq, 0)
        r_center_seq = t.FloatTensor(r_center_seq).view(-1, 128, 128)

        r_type_seq = t.FloatTensor(r_type_seq).view(-1)
        r_box_seq = t.FloatTensor(r_box_seq) / 127

        b_img = t.FloatTensor(b_img).view(-1, 128, 128)
        inside = t.FloatTensor(inside).view(-1, 128, 128)

        boundary = t.FloatTensor(exterior_boundary)
        data_id = t.LongTensor([data_id])
        return b_img, r_center_seq, r_type_seq, inside, r_box_seq, fp_state_seq, boundary, data_id

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
