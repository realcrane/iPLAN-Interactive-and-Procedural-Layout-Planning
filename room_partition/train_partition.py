from torch.utils.tensorboard import SummaryWriter
from dataset import PartitionDataset, collate_fn
from config_partition import PartitionConfig
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data import DataLoader
import torch as t
import models
import utils
import time
import os


def set_seeds(seed=0):
    t.manual_seed(seed)  # sets the seed for generating random numbers.
    t.cuda.manual_seed(seed)  # Sets the seed for generating random numbers for the current GPU.
    t.cuda.manual_seed_all(seed)  # Sets the seed for generating random numbers on all GPUs
    t.backends.cudnn.deterministic = True


opt = PartitionConfig()
log = utils.log


def train(**kwargs):
    set_seeds(seed=19920409)
    name = time.strftime('partition_train_%Y%m%d_%H%M%S')
    log_file = open(f"{opt.save_log_root}/{name}.txt", 'w')

    opt.parse(kwargs, log_file)
    start_time = time.strftime("%b %d %Y %H:%M:%S")
    log(log_file, f'Training start time: {start_time}')

    writer = SummaryWriter(os.path.join(opt.checkpoints_dir, 'tensorboard'))

    # step1: configure model; define criterion and optimizer
    log(log_file, 'Building model...')
    fp_rnn = models.FloorPlanRNN(opt, log_file)

    # step2: data
    train_data = PartitionDataset(data_root=opt.data_root, phase='train')
    val_data = PartitionDataset(data_root=opt.data_root, phase='test')
    log(log_file, 'The length of training data is {}'.format(train_data.len))
    log(log_file, 'The length of testing data is {}'.format(val_data.len))

    train_dataloader = DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        collate_fn=collate_fn
    )

    # step3 loading model
    fp_rnn.load_networks(opt, log_file)

    # obtain initial performance
    best_loss, _ = val(fp_rnn, val_dataloader, log_file, 0)
    log(log_file, 'Initialize the the average smoothl1 loss is: {}'.format(best_loss))

    # step4: training
    log(log_file, 'Starting to train...')
    losses_meter = utils.Loss_AverageMeter(fp_rnn.loss_names)
    current_epoch = opt.current_epoch
    while current_epoch < opt.max_epoch:
        current_epoch += 1
        losses_meter.reset()
        log(log_file)
        log(log_file, f'Training epoch: {current_epoch}')

        # update learning rate
        if opt.update_lr and current_epoch > opt.n_epoch:
            old_lr = fp_rnn.optimizers[0].param_groups[0]['lr']
            alpha = 1.0 - max(0, current_epoch - opt.n_epoch) / float(opt.n_decay_epoch)
            lr = opt.lr_base * alpha
            for optimizer in fp_rnn.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            log(log_file, f'Updating learning rate: {old_lr}---->{lr}')

        for i, data in enumerate(train_dataloader):
            fp_rnn.set_input(data)  # prepare training data
            fp_rnn.optimize_parameters()
            losses = fp_rnn.get_current_losses()

            room_num = fp_rnn.room_num
            losses_meter.update(losses, room_num)

            if i % opt.print_freq == opt.print_freq - 1:
                message = '(epoch: %d, iters: %d) ' % (current_epoch, i + 1)
                for k, v in losses_meter.val.items():
                    message += '%s: %.6f (%.6f) ' % (k, v, losses_meter.avg[k])
                log(log_file, message)

            # get the generated layouts for visualise
            if i == 0:
                pred_r_boxes = fp_rnn.pred_packed_r_boxes  # tensor: N x 4
                pred_r_boxes = PackedSequence(pred_r_boxes, fp_rnn.step_sizes, fp_rnn.sorted_indices,
                                              fp_rnn.unsorted_indices)  # PackedSewuence N x 4
                train_fp_samples = utils.get_fp_samples(pred_r_boxes, data)

        log(log_file,'epoch: {}, Train, the average smoothl1 loss is: {}'.format(current_epoch, losses_meter.avg['G_L1']))

        # output results to tensorboard
        writer.add_image('Images/train', train_fp_samples, current_epoch, dataformats='NWHC')
        for name, value in losses.items():
            writer.add_scalar('Loss/train_' + name, value, current_epoch)

        # save model
        if current_epoch % opt.save_freq == 0:
            fp_rnn.save_networks(current_epoch)

        # validate
        if current_epoch % opt.val_freq == 0:
            test_loss, test_fp_samples = val(fp_rnn, val_dataloader, log_file, current_epoch)

            # output results to tensorboard
            writer.add_image('Images/test', train_fp_samples, current_epoch, dataformats='NWHC')
            for name, value in losses.items():
                writer.add_scalar('Loss/test_' + name, value, current_epoch)

            is_best = test_loss < best_loss
            best_loss = min(test_loss, best_loss)
            if is_best:
                fp_rnn.save_networks('best')

    end_time = time.strftime("%b %d %Y %H:%M:%S")
    log(log_file, f'Training end time: {end_time}')
    log_file.close()
    writer.close()


def val(fp_rnn, dataloader, log_file, current_epoch):
    fp_rnn.eval()
    losses_meter = utils.Loss_AverageMeter(fp_rnn.loss_names)

    for i, data in enumerate(dataloader):  # inner loop within one epoch
        fp_rnn.set_input(data)
        fp_rnn.evaluate()  # calculate loss functions
        losses = fp_rnn.get_current_losses()

        room_num = fp_rnn.room_num
        losses_meter.update(losses, room_num)

        if i % opt.print_freq == opt.print_freq - 1:
            message = '(epoch: %d, iters: %d) ' % (current_epoch, i + 1)
            for k, v in losses_meter.val.items():
                message += '%s: %.6f (%.6f) ' % (k, v, losses_meter.avg[k])
            log(log_file, message)

        # get the generated layouts for visualise
        if i == 0:
            pred_r_boxes = fp_rnn.pred_packed_r_boxes  # tensor: N x 4
            pred_r_boxes = PackedSequence(pred_r_boxes, fp_rnn.step_sizes, fp_rnn.sorted_indices,
                                          fp_rnn.unsorted_indices)  # PackedSewuence N x 4
            fp_samples = utils.get_fp_samples(pred_r_boxes, data)

    fp_rnn.train()
    log(log_file, 'epoch: {}, Test, the average smoothl1 loss is: {}'.format(current_epoch, losses_meter.avg['G_L1']))
    return losses_meter.avg['G_L1'], fp_samples


if __name__ == '__main__':
    train()
