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

max_room_per_type = [1, 2, 1, 2, 1, 1, 1, 3, 1, 3, 1, 1, 1]
max_room_num = np.sum(np.array(max_room_per_type))
noise_dim = 32
data_root = '../data'
load_cvae_path = 'weights/roomtype_cvae_150.pth'

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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


cvae.load_model(load_model_path)

cvae.cuda()

# step2: data
print('Building dataset...')
train_data = RoomTypeDataset(data_root=data_root, phase='train', max_room_per_type=opt.max_room_per_type)
val_data = RoomTypeDataset(data_root=opt.data_root, phase='test', max_room_per_type=opt.max_room_per_type)
log(log_file, 'The length of training data is {}'.format(train_data.len))
log(log_file, 'The length of testing data is {}'.format(val_data.len))

log(log_file, 'Building data loader...')
val_dataloader = DataLoader(
    val_data,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.num_workers
)

# step3: testing
log(log_file, 'Starting to Testing...')
losses = utils.Loss_AverageMeter(['total', 'bce', 'kld'])
losses.reset()
log(log_file)
log(log_file, f'Testing epoch: {current_epoch}')

for i, (input_img, input_vector) in enumerate(val_dataloader):
    input_img = input_img.cuda()
    input_vector = input_vector.cuda()
    optimizer.zero_grad()
    recon, mu, log_var = model(input_vector, input_img)
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
log(log_file,
    'Train Epoch: {}, Average Loss ===> Loss: {:.4f}, BCE_Loss: {:.4f}, KLD_Loss: {:.4f}'.format(current_epoch,
                                                                                                 losses.avg['total'],
                                                                                                 losses.avg['bce'],
                                                                                                 losses.avg['kld']))

end_time = time.strftime("%b %d %Y %H:%M:%S")
log(log_file, f'Training end time: {end_time}')
log_file.close()

if __name__ == '__main__':
    train()
