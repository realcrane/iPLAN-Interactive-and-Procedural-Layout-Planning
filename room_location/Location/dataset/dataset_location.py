from .floorplan_train import LoadFloorplanTrain
from torch.utils import data
import torch as t
import random
import utils
import scipy.io as sio
import os


class LocationDataset(data.Dataset):
    def __init__(self, data_root, mask_size, phase):
        self.data_split_root = os.path.join(data_root, phase)
        self.floorplans = os.listdir(self.data_split_root)
        self.mask_size = mask_size
        self.len = self.__len__()

    def __len__(self):
        return len(self.floorplans)

    def __getitem__(self, index):
        floorplan_name = self.floorplans[index]
        floorplan_path = os.path.join(self.data_split_root, floorplan_name)

        floorplan = LoadFloorplanTrain(floorplan_path, self.mask_size)

        num_category = utils.num_category
        OUTSIDE = num_category + 2
        NOTHING = num_category + 1
        EXISTING = num_category

        living_node = floorplan.living_node
        floorplan.add_room(living_node)
        continue_node = floorplan.continue_node
        random_num = random.randint(0, len(continue_node) - 1)

        for i in range(random_num):
            node = continue_node[i]
            floorplan.add_room(node)

        input = floorplan.get_composite_location(num_extra_channels=0)

        target = t.zeros((floorplan.data_size, floorplan.data_size), dtype=t.long)
        target[target == 0] = OUTSIDE
        target[floorplan.inside_mask != 0] = NOTHING
        location_mask_size = self.mask_size

        min_h = max(living_node['centroid'][0] - location_mask_size, 0)
        max_h = min(living_node['centroid'][0] + location_mask_size, floorplan.data_size - 1)
        min_w = max(living_node['centroid'][1] - location_mask_size, 0)
        max_w = min(living_node['centroid'][1] + location_mask_size, floorplan.data_size - 1)
        target[min_h:max_h + 1, min_w:max_w + 1] = EXISTING

        for i in range(random_num):
            node = continue_node[i]
            min_h = max(node['centroid'][0] - location_mask_size, 0)
            max_h = min(node['centroid'][0] + location_mask_size, floorplan.data_size - 1)
            min_w = max(node['centroid'][1] - location_mask_size, 0)
            max_w = min(node['centroid'][1] + location_mask_size, floorplan.data_size - 1)
            target[min_h:max_h + 1, min_w:max_w + 1] = EXISTING

        i = random_num
        node = continue_node[i]
        room_type = node['category']
        room_type = t.nn.functional.one_hot(t.LongTensor([room_type]), num_classes=13)
        min_h = max(node['centroid'][0] - location_mask_size, 0)
        max_h = min(node['centroid'][0] + location_mask_size, floorplan.data_size - 1)
        min_w = max(node['centroid'][1] - location_mask_size, 0)
        max_w = min(node['centroid'][1] + location_mask_size, floorplan.data_size - 1)
        target[min_h:max_h + 1, min_w:max_w + 1] = utils.label2index(node["category"])

        return input, target, room_type.squeeze().float()
