import warnings
import utils


class PartitionConfig(object):
    data_root = '../data'
    save_log_root = 'log'
    checkpoints_dir = 'checkpoints'

    load_netG_path = None
    load_netD_path = None

    gpu_ids = '1'

    update_lr = True
    n_decay_epoch = 200
    lr_base = 1e-4
    weight_decay = 1e-4

    batch_size = 128
    num_workers = 16
    n_epoch = 10
    current_epoch = 0
    max_epoch = n_epoch + n_decay_epoch
    lambda_L1 = 1000

    print_freq = 20
    save_freq = 5
    val_freq = 1

    def parse(self, kwargs, file):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                utils.log(file, f'{k}: {getattr(self, k)}')
