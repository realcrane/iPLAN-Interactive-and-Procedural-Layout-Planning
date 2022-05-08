from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from .floorplan_train import LoadFloorplanTrain
from torch.utils import data
import torch as t
import os


class PartitionDataset(data.Dataset):
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
        input_data = floorplan.get_input()
        return input_data


def get_length(data):
    return data[2].size(0)


def collate_fn(batch):
    batch.sort(key=get_length, reverse=True)
    b_img_list = []
    r_center_seq_list = []
    r_type_seq_list = []
    inside_list = []
    r_box_seq_list = []
    fp_state_seq_list = []
    boundary_id_list = []
    data_id_list = []
    len_list = []

    for i, (b_img, r_center_seq, r_type_seq, inside, r_box_seq, fp_state_seq, boundary, data_id) in enumerate(batch):
        b_img_list.append(b_img)
        r_center_seq_list.append(r_center_seq)
        r_type_seq_list.append(r_type_seq)
        inside_list.append(inside)
        r_box_seq_list.append(r_box_seq)
        fp_state_seq_list.append(fp_state_seq)
        data_id_list.append(data_id)
        id_vector = t.ones((boundary.size(0), 1)) * i
        bounadry_id = t.cat([boundary, id_vector], 1)
        boundary_id_list.append(bounadry_id)
        len_list.append(r_type_seq.size(0))

    seq_len = t.FloatTensor(len_list)

    pad_b_imgs = pad_sequence(b_img_list, batch_first=True, padding_value=0)
    pad_r_centers = pad_sequence(r_center_seq_list, batch_first=True, padding_value=0)
    pad_r_types = pad_sequence(r_type_seq_list, batch_first=True, padding_value=0)
    pad_insides = pad_sequence(inside_list, batch_first=True, padding_value=0)
    pad_r_boxes = pad_sequence(r_box_seq_list, batch_first=True, padding_value=0)
    pad_fp_states = pad_sequence(fp_state_seq_list, batch_first=True, padding_value=0)
    pad_data_ids = pad_sequence(data_id_list, batch_first=True, padding_value=0)

    packed_r_boxes = pack_padded_sequence(pad_r_boxes, seq_len, batch_first=True, enforce_sorted=False)
    packed_r_types = pack_padded_sequence(pad_r_types, seq_len, batch_first=True, enforce_sorted=False)
    packed_r_centers = pack_padded_sequence(pad_r_centers, seq_len, batch_first=True, enforce_sorted=False)
    packed_fp_states = pack_padded_sequence(pad_fp_states, seq_len, batch_first=True, enforce_sorted=False)
    boundary_ids = t.cat(boundary_id_list, 0)

    assert check_order(packed_r_boxes, packed_r_centers, packed_r_types,
                       packed_fp_states), 'packed index error '
    index = packed_r_types[2]
    b_imgs = pad_b_imgs.index_select(0, index)
    insides = pad_insides.index_select(0, index)
    data_ids = pad_data_ids.index_select(0, index)
    return b_imgs, packed_r_centers, packed_r_types, insides, packed_r_boxes, packed_fp_states, boundary_ids, data_ids


def check_order(packed_data_1, packed_data_2, packed_data_3, packed_data_4):
    order_1 = packed_data_1[2]
    order_2 = packed_data_2[2]
    order_3 = packed_data_3[2]
    order_4 = packed_data_4[2]

    if (order_1 - order_2).abs().sum() == 0 and (order_1 - order_3).abs().sum() == 0 and (
            order_1 - order_4).abs().sum() == 0:
        return True
    else:
        return False
