import warnings
import utils

class LocationConfig(object):
    data_root = '../../data'
    save_log_root = 'log'
    result_file = 'result_location.csv'
    module_name = 'location'
    model_name = 'resnet18_up1'
    load_model_path = None
    load_connect_path = None
    load_embedding_path = None
    mask_size = 4

    batch_size = 64
    num_workers = 4
    print_freq = 50

    max_epoch = 100
    current_epoch = 0
    save_freq = 10
    val_freq = 1

    update_lr = True
    lr_decay_freq = 5
    lr_base = 1e-4
    weight_decay = 1e-4

    def parse(self, kwargs, file):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                utils.log(file, f'{k}: {getattr(self, k)}')