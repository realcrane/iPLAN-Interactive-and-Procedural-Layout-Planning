from config_roomtype import RoomTypeConfig
from torch.utils.data import DataLoader
from dataset import RoomTypeDataset
import torch.nn.functional as F
import torch as t
import numpy as np
import random
import models
import utils
import time
import os

opt = RoomTypeConfig()
log = utils.log


def setup_seed(seed):
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    t.backends.cudnn.deterministic = True


def loss_function(recon_x, x, mu, log_var):
    """return reconstruction error + KL divergence losses"""
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * t.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD * 0.5, BCE, KLD


def train(**kwargs):
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    name = time.strftime('roomtype_train_%Y%m%d_%H%M%S')
    log_file = open(f"{opt.save_log_root}/{name}.txt", 'w')

    opt.parse(kwargs, log_file)
    start_time = time.strftime("%b %d %Y %H:%M:%S")
    log(log_file, f'Training start time: {start_time}')

    # step1: configure model
    log(log_file, 'Building model...')
    cvae = models.cvae(
        modul_name=opt.module_name,
        model_name=opt.model_name,
        input_dim=opt.max_room_num,
        hidden_dim1=128,
        hidden_dim2=64,
        z_dim=opt.noise_dim
    )
    print(cvae)
    print('###################')

    num_params = 0
    for param in cvae.parameters():
        num_params += param.numel()
    print('[Network %s] Total number of parameters : %.3f M' % ('cvae', num_params / 1e6))
    print('-----------------------------------------------')

    if opt.load_cvae_path:
        cvae.load_model(opt.load_cvae_path)

    cvae.cuda()

    # step2: data
    log(log_file, 'Building dataset...')
    train_data = RoomTypeDataset(data_root=opt.data_root, phase='train', max_room_per_type=opt.max_room_per_type)
    val_data = RoomTypeDataset(data_root=opt.data_root, phase='test', max_room_per_type=opt.max_room_per_type)
    log(log_file, 'The length of training data is {}'.format(train_data.len))
    log(log_file, 'The length of testing data is {}'.format(val_data.len))

    log(log_file, 'Building data loader...')
    train_dataloader = DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers
    )
    # step3: criterion and optimizer
    log(log_file, 'Building criterion and optimizer...')
    lr = opt.lr_base
    optimizer = t.optim.Adam(cvae.parameters(), lr=lr, weight_decay=opt.weight_decay)
    current_epoch = opt.current_epoch

    # step4: training
    log(log_file, 'Starting to train...')
    losses = utils.Loss_AverageMeter(['total', 'bce', 'kld'])
    while current_epoch < opt.max_epoch:
        current_epoch += 1
        losses.reset()
        log(log_file)
        log(log_file, f'Training epoch: {current_epoch}')

        # update learning rate
        if opt.update_lr:
            if current_epoch % opt.lr_decay_freq == 0:
                lr = lr * (1 - float(current_epoch) / opt.max_epoch) ** 1.5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    log(log_file, f'Updating learning rate: {lr}')

        for i, (input_img, input_vector) in enumerate(train_dataloader):
            input_img = input_img.cuda()
            input_vector = input_vector.cuda()
            optimizer.zero_grad()
            recon, mu, log_var = cvae(input_vector, input_img)
            loss, bce_loss, kld_loss = loss_function(recon, input_vector, mu, log_var)
            loss.backward()
            optimizer.step()
            losses.update([loss, bce_loss, kld_loss], input_img.size(0))

            if i % opt.print_freq == opt.print_freq - 1:
                current_num = i * opt.batch_size + len(input_vector)
                message = 'Train Epoch: {} [{}/{}]\t Loss: {:.4f}({:.4f}), BCE_Loss:{:.4f}({:.4f}), KLD_Loss: {:.4f}({:.4f}))'.format(
                    current_epoch, str(current_num).zfill(5), train_data.len, losses.val['total'], losses.avg['total'],
                    losses.val['bce'], losses.avg['bce'], losses.val['kld'], losses.avg['kld'])
                log(log_file, message)
        log(log_file, 'Train Epoch: {}, Average Loss ===> Loss: {:.4f}, BCE_Loss: {:.4f}, KLD_Loss: {:.4f}'.format(current_epoch, losses.avg['total'], losses.avg['bce'],losses.avg['kld']))

        if current_epoch % opt.save_freq == 0:
            cvae.save_model(current_epoch)

        # validate
        if current_epoch % opt.val_freq == 0:
            val(cvae, val_dataloader, log_file, current_epoch)

    end_time = time.strftime("%b %d %Y %H:%M:%S")
    log(log_file, f'Training end time: {end_time}')
    log_file.close()


def val(cvae, dataloader, log_file, current_epoch):
    cvae.eval()
    losses = utils.Loss_AverageMeter(['total', 'bce', 'kld'])
    with t.no_grad():
        for batch_idx, (input_img, input_vector) in enumerate(dataloader):
            input_img = input_img.cuda()
            input_vector = input_vector.cuda()

            recon, mu, log_var = cvae(input_vector, input_img)
            loss, bce_loss, kld_loss = loss_function(recon, input_vector, mu, log_var)

            losses.update([loss, bce_loss, kld_loss], input_img.size(0))

        log(log_file, 'Test Epoch: {}, Average Loss ===> Loss: {:.4f}, BCE_Loss: {:.4f}, KLD_Loss: {:.4f}'.format(current_epoch, losses.avg['total'], losses.avg['bce'], losses.avg['kld']))
    cvae.train()

if __name__ == '__main__':
    train()
