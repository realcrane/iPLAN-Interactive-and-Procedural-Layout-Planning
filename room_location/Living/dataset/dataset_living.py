from .floorplan_train import LoadFloorplanTrain
from torch.utils import data
import torch as t
import os 

class LivingDataset(data.Dataset):
    def __init__(self, data_root, phase):
        self.data_split_root = os.path.join(data_root, phase)
        self.floorplans = os.listdir(self.data_split_root)
        self.len = self.__len__()

    def __len__(self):
        return len(self.floorplans)

    def __getitem__(self, index):
        floorplan_name = self.floorplans[index]
        floorplan_path = os.path.join(self.data_split_root, floorplan_name)
        floorplan = LoadFloorplanTrain(floorplan_path)
        living_h, living_w = floorplan.living_node['centroid']

        input = floorplan.get_composite_living(num_extra_channels=0)
        target = t.zeros(2)
        target[0] = living_h
        target[1] = living_w
        
        return input, target