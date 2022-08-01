from synth.floorplan.roomlocation_fp import FloorPlan
from room_location.Living import models as living
from room_location.Location import models as location
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
        if len(data.rCenters) == 0:
            temp_list.append(name)
    return temp_list


def main():
    root = '../data'

    # step1: configure model
    print('Building model...')
    living_model = living.model(
        module_name="living",
        model_name="resnet18_fc1",
        input_channel=3,
        output_channel=2,
        pretrained=False,
    )
    living_connect = living.connect(
        module_name="living",
        model_name="resnet18_fc1",
        input_channel=512,
        output_channel=2,
        reshape=True
    )
    epoch = 300
    load_living_model_path = f'../room_location/weights/living_resnet18_{epoch}.pth'
    load_living_connect_path = f'../room_location/weights/living_fc1_{epoch}.pth'
    living_model.load_model(load_living_model_path)
    living_connect.load_model(load_living_connect_path)
    living_model.cuda()
    living_connect.cuda()

    location_model = location.model(
        module_name='location',
        model_name='resnet18_up1',
        input_channel=13 + 4,
        output_channel=13 + 3,
        pretrained=True,
        pretrained_path='../room_location/pretrained_model/resnet18-5c106cde.pth'
    )

    location_connect = location.connect(
        module_name='location',
        model_name='resnet18_up1',
        input_channel=512,
        output_channel=13 + 3,
        reshape=False
    )
    location_embedding = location.embedding(
        module_name='location',
        model_name='resnet18_up1',
        input_channel=13,
        output_channel=256,
        reshape=False
    )
    epoch = 100
    load_location_model_path = f'../room_location/weights/location_resnet18_{epoch}.pth'
    load_location_connect_path = f'../room_location/weights/location_up1_{epoch}.pth'
    load_location_embedding_path = f'../room_location/weights/location_embed_{epoch}.pth'

    location_model.load_model(load_location_model_path)
    location_connect.load_model(load_location_connect_path)
    location_embedding.load_model(load_location_embedding_path)
    location_model.cuda()
    location_connect.cuda()
    location_embedding.cuda()

    living_model.eval()
    living_connect.eval()
    location_model.eval()
    location_connect.eval()
    location_embedding.eval()

    # step2: data
    print('Building dataset...')
    data_root = os.path.join(root, 'test')
    floorplans = getList(data_root)
    print('The length of testing data is {}'.format(len(floorplans)))

    # step3: testing
    print('Starting to Testing...')
    total = 0
    for i in range(len(floorplans)):
        fp_name = floorplans[i]
        fp_path = os.path.join(data_root, fp_name)
        fp = FloorPlan(fp_path)

        input_living = fp.get_composite_living()
        input_living = input_living.unsqueeze(0).cuda()

        with torch.no_grad():
            score_model = living_model(input_living)
            score_connect_living = living_connect(score_model)
            living_center = torch.round(score_connect_living)
            living_center = living_center.cpu().squeeze().numpy()

        node = {}
        node['category'] = int(0)
        x = living_center[0]
        y = living_center[1]
        node['centroid'] = (int(x), int(y))
        fp.add_room(node)
        fp.living_node = node

        iteration = 0
        # the maximum sampling for room location
        while iteration < 50:
            iteration = iteration + 1
            pred_rCenters, pred_rTypes, flag = get_rcenters(fp, location_model, location_connect, location_embedding)
            if flag:
                fp.update_rCenters(pred_rCenters, pred_rTypes)
                data = fp.to_dict()
                sio.savemat(fp_path, {'data': data})
                total = total + 1
                print('{}/{}, has processd {} '.format(i + 1, total, data['name']))
                break

    print(f'{total} layouts')


def get_rcenters(fp, location_model, location_connect, location_embedding, mask_size=4):
    pred_rCenters = []
    pred_rTypes = []

    living_node = fp.living_node
    h, w = living_node['centroid']
    pred_rCenters.append(np.array([int(h), int(w)]))
    pred_rTypes.append(0)

    input_location = fp.get_composite_location(num_extra_channels=0)
    continue_rTypes = fp.continue_rTypes.squeeze()
    input_location = input_location.unsqueeze(0).cuda()

    room_num = continue_rTypes.shape[0]

    iter = 0
    flag = True
    while room_num > 0:
        update_id = np.zeros(room_num, dtype=bool)
        iter = iter + 1

        np.random.shuffle(continue_rTypes)
        with torch.no_grad():
            for n in range(room_num):
                score_model = location_model(input_location)
                r_t = torch.LongTensor(continue_rTypes[n:n + 1]).cuda()
                one_hot_label = torch.nn.functional.one_hot(r_t, num_classes=13)
                score_embedding = location_embedding(one_hot_label.float())
                score_temp = torch.cat([score_model, score_embedding], 1)
                score_connect = location_connect(score_temp)

                score_softmax = torch.softmax(score_connect, dim=1)
                output = score_softmax.cpu().numpy()
                predict = np.argmax(output, axis=1)[0]

                center = find_center(predict, r_t)
                h, w = center

                if h == 0 or w == 0 or input_location[0, 0, h, w] == 0:
                    continue

                flag = check_rcenters(center, pred_rCenters)

                if flag:
                    pred_rCenters.append(np.array([h, w]))
                    pred_rTypes.append(r_t.cpu().numpy().reshape(1)[0])
                    update_id[n] = True
                    min_h = max(h - mask_size, 0)
                    max_h = min(h + mask_size, 128 - 1)
                    min_w = max(w - mask_size, 0)
                    max_w = min(w + mask_size, 128 - 1)
                    input_location[0, r_t + 4, min_h:max_h + 1, min_w:max_w + 1] = 1.0
                    input_location[0, 3, :, :] = input_location[0, 4:, :, :].sum(0)

                    iter = 0

        continue_rTypes = continue_rTypes[~update_id]
        room_num = continue_rTypes.shape[0]

        if iter > 5:
            if room_num == 0:  # 0: all room centers are located, 1: one room missing, 2: two room missing
                flag = True
            else:
                flag = False
            break

    return pred_rCenters, pred_rTypes, flag


def find_center(predict, r_t, mask_size=4):
    index_point = []
    index = np.where(predict == r_t.cpu().numpy())
    for ind in range(index[0].shape[0]):
        index_point.append((index[0][ind], index[1][ind]))

    num_point = 0
    if len(index_point) > 0:
        min_h = np.min(index[0])
        min_w = np.min(index[1])
        max_h = np.max(index[0])
        max_w = np.max(index[1])

        if max_h - min_h + 1 <= 2 * mask_size and max_w - min_w + 1 <= 2 * mask_size:
            predict_h = (min_h + max_h) // 2
            predict_w = (min_w + max_w) // 2
            num_point = len(index_point)
        else:
            predict_h, predict_w = index_point[0]
            for point in index_point:
                new_num_point = 0
                for other_point in index_point:
                    if abs(other_point[0] - point[0]) <= mask_size and abs(
                            other_point[1] - point[1]) <= mask_size:
                        new_num_point += 1
                if new_num_point > num_point:
                    predict_h, predict_w = point
                    num_point = new_num_point
    if num_point > 0:
        return np.array([predict_h, predict_w])
    else:
        return np.array([0, 0])


def check_rcenters(center, rCenters):
    flag = True
    center = np.array(center)
    for i in range(len(rCenters)):
        temp = np.array(rCenters[i])
        dist = np.linalg.norm(center - temp)
        if dist < 9:
            flag = False
            return flag
    return flag


if __name__ == '__main__':
    main()
