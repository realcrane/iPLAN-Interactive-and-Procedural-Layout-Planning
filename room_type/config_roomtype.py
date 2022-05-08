import warnings
import utils
import numpy as np

class RoomTypeConfig(object):
    data_root = '../data'
    save_log_root = 'log'
    load_cvae_path = None
    module_name='roomtype'
    model_name='cvae'

    batch_size = 256
    noise_dim = 32
    max_room_per_type = [1, 2, 1, 2, 1, 1, 1, 3, 1, 3, 1, 1, 1]
    max_room_num = np.sum(np.array(max_room_per_type))

    num_workers = 16
    max_epoch = 150
    current_epoch = 0

    print_freq = 30
    save_freq = 10
    val_freq = 1

    threshold = 0.7

    update_lr = True
    lr_decay_freq = 10
    lr_base = 1e-3
    weight_decay = 1e-4

    def parse(self, kwargs, file):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                utils.log(file, f'{k}: {getattr(self, k)}')