import torch
from skimage import measure
import numpy as np
import scipy.io as sio
from PIL import Image
import os
import numpy as np
from shapely import geometry


class FloorPlan():
    def __init__(self, data_folder, data_name, log):
        data_path = os.path.join(data_folder, data_name)

        with Image.open(data_path) as temp:
            image_array = np.asarray(temp, dtype=np.uint8)
        boundary_mask = image_array[:, :, 0]
        category_mask = image_array[:, :, 1]
        index_mask = image_array[:, :, 2]
        inside_mask = image_array[:, :, 3]

        b_img = np.ones((128, 128)) * 13

        # exterior wall
        temp = self.im_resize(boundary_mask == 127, 128)
        b_img = b_img * (1 - temp) + temp * 14
        # front door
        temp = self.im_resize(boundary_mask == 255, 128)
        b_img = b_img * (1 - temp) + temp * 15
        # indside mask
        temp = self.im_resize(inside_mask / 255, 128)
        b_img = b_img * (1 - temp) + temp * 16
        self.b_img = b_img

        rBoxes = []
        rTypes = []
        indexs = np.unique(index_mask[index_mask > 0])

        for index in indexs:
            r_mask = index_mask == index
            category = np.unique(category_mask[r_mask])[0]
            r_mask = self.im_resize(r_mask, 128)
            if not r_mask.sum() == 0:
                min_h, max_h = np.where(np.any(r_mask, axis=1))[0][[0, -1]]
                min_w, max_w = np.where(np.any(r_mask, axis=0))[0][[0, -1]]
                box = [min_h, min_w, max_h, max_w]
            else:
                self.log.write('room area equal to zero: {}'.format(data_name))
                continue
            rBoxes.append(box)
            rTypes.append(category)

        rTypes = np.array(rTypes, dtype=int)
        rBoxes = np.array(rBoxes, dtype=int)
        inds = np.argsort(rTypes, 0)
        self.gt_rTypes = rTypes[inds]
        self.gt_rBoxes = rBoxes[inds, :]
        self.name = data_name[:-4]
        self.log = log

        self.check_boundary()
        self.exterior_boundary = self.get_exterior_boundary()

        self.rCenters = None
        self.rTypes = None
        self.rBoxes = None

    def check_inside(self, inside):
        min_h, max_h = np.where(np.any(inside, axis=1))[0][[0, -1]]
        min_w, max_w = np.where(np.any(inside, axis=0))[0][[0, -1]]
        w, h = inside.shape
        min_h = max(min_h - 5, 0)
        min_w = max(min_w - 5, 0)
        max_h = min(max_h + 5, h)
        max_w = min(max_w + 5, w)

        for h in range(min_h, max_h):
            for w in range(min_w, max_w):
                if inside[h, w - 1] == 0 and inside[h, w + 1] == 0 and inside[h, w] == 255:
                    self.log.write('delete one pixel\n')
                    inside[h, w] = 0

                if inside[h - 1, w] == 0 and inside[h + 1, w] == 0 and inside[h, w] == 255:
                    self.log.write('delete one pixel\n')
                    inside[h, w] = 0
        return inside

    def check_boundary(self):
        if np.sum(self.b_img == 15) < 12:
            front_door = self.get_front_door()
            door_y1, door_x1, door_y2, door_x2 = front_door
            if door_y2 - door_y1 <= 3 and 14 in self.b_img[[door_y1 - 1, door_y2], door_x1:door_x2]:
                door_y2 = door_y1 + 4

            elif door_x2 - door_x1 <= 3 and 14 in self.b_img[door_y1:door_y2, [door_x1 - 1, door_x2]]:
                door_x2 = door_x1 + 4

            self.b_img[door_y1:door_y2, door_x1:door_x2] = 15
            self.log.write('modify the front door\n')

    def get_front_door(self):
        front_door_mask = self.b_img == 15
        region = measure.regionprops(front_door_mask.astype(int))[0]
        front_door = np.array(region.bbox, dtype=int)
        return front_door

    def get_exterior_boundary(self):
        front_door = self.get_front_door()
        exterior_boundary = []
        inside = self.b_img == 16
        inside = inside * 255

        inside = self.check_inside(inside)

        s_h, s_w = inside.shape

        min_h, max_h = np.where(np.any(inside, axis=1))[0][[0, -1]]
        min_w, max_w = np.where(np.any(inside, axis=0))[0][[0, -1]]
        min_h = max(min_h - 8, 0)
        min_w = max(min_w - 8, 0)
        max_h = min(max_h + 8, s_h)
        max_w = min(max_w + 8, s_w)

        # src: http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html
        # search direction:0(right)/1(down)/2(left)/3(up)
        # find the left-top point
        flag = False
        for h in range(min_h, max_h):
            for w in range(min_w, max_w):
                if inside[h, w] == 255:
                    exterior_boundary.append((h, w, 0))
                    flag = True
                    break
            if flag:
                break

        # left/top edge: inside
        # right/bottom edge: outside
        while (flag):
            if exterior_boundary[-1][2] == 0:
                for w in range(exterior_boundary[-1][1] + 1, max_w):
                    corner_sum = 0
                    if inside[exterior_boundary[-1][0], w] == 255:
                        corner_sum += 1
                    if inside[exterior_boundary[-1][0] - 1, w] == 255:
                        corner_sum += 1
                    if inside[exterior_boundary[-1][0], w + 1] == 255:
                        corner_sum += 1
                    if inside[exterior_boundary[-1][0] - 1, w + 1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (exterior_boundary[-1][0], w, 1)
                        break
                    if corner_sum == 3:
                        new_point = (exterior_boundary[-1][0], w + 1, 3)
                        break
                    if corner_sum == 4:
                        new_point = (exterior_boundary[-1][0], w, 3)
                        break

            if exterior_boundary[-1][2] == 1:
                for h in range(exterior_boundary[-1][0] + 1, max_h):
                    corner_sum = 0
                    if inside[h, exterior_boundary[-1][1] + 1] == 255:
                        corner_sum += 1
                    if inside[h + 1, exterior_boundary[-1][1] + 1] == 255:
                        corner_sum += 1
                    if inside[h, exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if inside[h + 1, exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (h, exterior_boundary[-1][1], 2)
                        break
                    if corner_sum == 3:
                        new_point = (h + 1, exterior_boundary[-1][1], 0)
                        break
                    if corner_sum == 4:
                        new_point = (h, exterior_boundary[-1][1], 0)
                        break

            if exterior_boundary[-1][2] == 2:
                for w in range(exterior_boundary[-1][1] - 1, min_w, -1):
                    corner_sum = 0
                    if inside[exterior_boundary[-1][0], w] == 255:
                        corner_sum += 1
                    if inside[exterior_boundary[-1][0] + 1, w] == 255:
                        corner_sum += 1
                    if inside[exterior_boundary[-1][0], w - 1] == 255:
                        corner_sum += 1
                    if inside[exterior_boundary[-1][0] + 1, w - 1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (exterior_boundary[-1][0], w, 3)
                        break
                    if corner_sum == 3:
                        new_point = (exterior_boundary[-1][0], w - 1, 1)
                        break
                    if corner_sum == 4:
                        new_point = (exterior_boundary[-1][0], w, 1)
                        break

            if exterior_boundary[-1][2] == 3:
                for h in range(exterior_boundary[-1][0] - 1, min_h, -1):
                    corner_sum = 0
                    if inside[h, exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if inside[h - 1, exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if inside[h, exterior_boundary[-1][1] - 1] == 255:
                        corner_sum += 1
                    if inside[h - 1, exterior_boundary[-1][1] - 1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (h, exterior_boundary[-1][1], 0)
                        break
                    if corner_sum == 3:
                        new_point = (h - 1, exterior_boundary[-1][1], 2)
                        break
                    if corner_sum == 4:
                        new_point = (h, exterior_boundary[-1][1], 2)
                        break

            if new_point != exterior_boundary[0]:
                exterior_boundary.append(new_point)
            else:
                flag = False
        exterior_boundary = [[r, c, d, 0] for r, c, d in exterior_boundary]

        door_y1, door_x1, door_y2, door_x2 = front_door

        door_h, door_w = door_y2 - door_y1, door_x2 - door_x1
        is_vertical = door_h > door_w

        insert_index = None
        door_index = None
        new_p = []
        th = 3
        for i in range(len(exterior_boundary)):
            y1, x1, d, _ = exterior_boundary[i]
            y2, x2, _, _ = exterior_boundary[(i + 1) % len(exterior_boundary)]
            if is_vertical != d % 2:
                continue
            if is_vertical and (x1 - th < door_x1 < x1 + th or x1 - th < door_x2 < x1 + th):  # 1:down 3:up
                l1 = geometry.LineString([[y1, x1], [y2, x2]])
                l2 = geometry.LineString([[door_y1, x1], [door_y2, x1]])
                l12 = l1.intersection(l2)
                if l12.length > 1:
                    dy1, dy2 = l12.xy[0]  # (y1>y2)==(dy1>dy2)
                    insert_index = i
                    door_index = i + (y1 != dy1)
                    if y1 != dy1: new_p.append([dy1, x1, d, 1])
                    if y2 != dy2: new_p.append([dy2, x1, d, 1])
            elif not is_vertical and (y1 - th < door_y1 < y1 + th or y1 - th < door_y2 < y1 + th):
                l1 = geometry.LineString([[y1, x1], [y2, x2]])
                l2 = geometry.LineString([[y1, door_x1], [y1, door_x2]])
                l12 = l1.intersection(l2)
                if l12.length > 1:
                    dx1, dx2 = l12.xy[1]  # (x1>x2)==(dx1>dx2)
                    insert_index = i
                    door_index = i + (x1 != dx1)
                    if x1 != dx1: new_p.append([y1, dx1, d, 1])
                    if x2 != dx2: new_p.append([y1, dx2, d, 1])

        if len(new_p) > 0:
            exterior_boundary = exterior_boundary[:insert_index + 1] + new_p + exterior_boundary[insert_index + 1:]
        else:
            self.log.write('the door is totally equal to the side\n')
        exterior_boundary = exterior_boundary[door_index:] + exterior_boundary[:door_index]

        exterior_boundary = np.array(exterior_boundary, dtype=int)
        return exterior_boundary

    def im_resize(self, im, sz):
        im = im.astype(int)
        temp = np.zeros((sz, sz), dtype=np.uint8)
        for h in range(sz):
            for w in range(sz):
                value = np.max(im[h * 2:h * 2 + 2, w * 2:w * 2 + 2])
                temp[h, w] = value
        return temp

    def to_dict(self, dtype=int):
        '''
        Compress data, notice:
        '''
        return {
            'name': self.name,
            'gt_rTypes': self.gt_rTypes.astype(dtype),
            'gt_rBoxes': self.gt_rBoxes.astype(dtype),
            'Boundary': self.exterior_boundary.astype(dtype),
            'rTypes': np.array([]) if self.rTypes == None else self.rTypes.astype(dtype),
            'rBoxes': np.array([]) if self.rBoxes == None else self.rBoxes.astype(dtype),
            'rCenters': np.array([]) if self.rCenters == None else self.rCenters.astype(dtype)
        }
