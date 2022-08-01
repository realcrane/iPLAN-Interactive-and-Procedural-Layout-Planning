from synth.floorplan.roomtype_fp import FloorPlan
from room_type import models
import scipy.io as sio
import numpy as np
import torch
import os

def getList(path):
    mat_list = os.listdir(path)
    mat_list.sort()
    temp_list = []
    for name in mat_list[100:200]:
        mat_path = os.path.join(path, name)
        data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)['data']
        if type(data.rTypes) is not np.ndarray or len(data.rTypes) == 0:
            temp_list.append(name)

    return temp_list

def main():
    max_room_per_type = [1, 2, 1, 2, 1, 1, 1, 3, 1, 3, 1, 1, 1]
    max_room_num = np.sum(np.array(max_room_per_type))
    noise_dim = 32
    root = '../data'
    load_cvae_path = '../room_type/weights/roomtype_cvae_150.pth'

    # step1: configure model
    print('Building model...')
    cvae = models.cvae(
        modul_name='roomtype',
        model_name='cvae',
        input_dim=max_room_num,
        hidden_dim1=128,
        hidden_dim2=64,
        z_dim=noise_dim
    )
    print(cvae)
    print('###################')

    cvae.load_model(load_cvae_path)
    cvae.cuda()

    # step2: data
    print('Building dataset...')

    data_root = os.path.join(root, 'test')
    floorplans = getList(data_root)
    print('The length of testing data is {}'.format(len(floorplans)))

    # step3: testing
    print('Starting to Testing...')
    for i in range(len(floorplans)):
        fp_name = floorplans[i]
        fp_path = os.path.join(data_root, fp_name)
        fp = FloorPlan(fp_path)

        # if given room types
        # fp.rTypes = fp.gt_rTypes
        # fp.rBoxes = np.array([])
        # fp.rCenters= np.array([])
        # data = fp.to_dict()
        # sio.savemat(fp_path, {'data': data})

        input_img = fp.init_input_img(fp.exterior_boundary)
        input_img = torch.FloatTensor(input_img).unsqueeze(0).unsqueeze(0).cuda()
        input_img = fp.normalize(input_img)

        with torch.no_grad():
            emb = cvae.embed(input_img)
            z = torch.randn(input_img.size(0), 32).cuda()
            o_z = torch.cat([z, emb], 1)
            sample = cvae.decoder(o_z).cuda()
            sample = sample.view(-1, 19)
            fp.update_rTypes(sample.squeeze().cpu().numpy(), max_room_per_type)
            fp.rBoxes = np.array([])
            fp.rCenters = np.array([])
            data = fp.to_dict()
            sio.savemat(fp_path, {'data': data})

if __name__ == '__main__':
    main()