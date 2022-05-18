from floorplan import FloorPlan
from datetime import datetime
import scipy.io as sio
import os


def main():
    root = '../../../../../dataset/floorplan_dataset'
    split = ['test']  # ['train', 'test']

    num = 0
    for s in split:
        now = datetime.now()
        current_time = now.strftime("%Y%m%d")
        log = open(f'log/log_{s}_{current_time}.txt', 'a')

        data_folder = os.path.join(root, s)
        data_list = os.listdir(data_folder)

        save_root = f'{s}'

        for data_name in data_list:
            num += 1
            if num % 100 == 0:
                print(f'processed {num} data')

            mat_name = data_name.replace('png', 'mat')
            save_path = os.path.join(save_root, mat_name)
            if os.path.exists(save_path):
                continue

            log.write(f'start to process data {data_name}\n')

            fp = FloorPlan(data_folder, data_name, log)
            data = fp.to_dict()

            sio.savemat(save_path, {'data': data})

if __name__ == '__main__':
    main()
