from config_living import LivingConfig
from torch.utils.data import DataLoader
from dataset import LivingDataset
from inspect import getsource
from torchnet import meter
from tqdm import tqdm
import torch as t
import models
import utils
import time
import csv
import os

opt = LivingConfig()
log = utils.log


def train(**kwargs):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    name = time.strftime('living_train_%Y%m%d_%H%M%S')
    log_file = open(f"{opt.save_log_root}/{name}.txt", 'w')

    opt.parse(kwargs, log_file)
    start_time = time.strftime("%b %d %Y %H:%M:%S")
    log(log_file, f'Training start time: {start_time}')

    # step1: configure model
    log(log_file, 'Building model...')
    model = models.model(
        module_name=opt.module_name,
        model_name=opt.model_name,
        input_channel=3,
        output_channel=2,
        pretrained=False
    )
    input_channel = 512
    connect = models.connect(
        module_name=opt.module_name,
        model_name=opt.model_name,
        input_channel=input_channel,
        output_channel=2,
        reshape=True
    )

    print(model)
    print(connect)

    if opt.load_model_path:
        log(log_file,'Loading the model: {}'.format(opt.load_model_path))
        model.load_model(opt.load_model_path)
    if opt.load_connect_path:
        log(log_file, 'Loading the model: {}'.format(opt.load_connect_path))
        connect.load_model(opt.load_connect_path)

    model.cuda()
    connect.cuda()

    # step2: data
    log(log_file, 'Building dataset...')
    train_data = LivingDataset(data_root=opt.data_root, phase='train')
    val_data = LivingDataset(data_root=opt.data_root, phase='test')
    log(log_file, 'The length of training data is {}'.format(train_data.len))
    log(log_file, 'The length of testing data is {}'.format(val_data.len))

    log(log_file, 'Building data loader...')
    train_dataloader = DataLoader(
        train_data,
        opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers
    )
    val_dataloader = DataLoader(
        val_data,
        opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers
    )

    # step3: criterion and optimizer
    log(log_file, 'Building criterion and optimizer...')
    lr = opt.lr_base
    optimizer = t.optim.Adam(
        list(model.parameters()) + list(connect.parameters()),
        lr=lr,
        weight_decay=opt.weight_decay
    )
    current_epoch = opt.current_epoch
    criterion = t.nn.SmoothL1Loss()
    loss_meter = meter.AverageValueMeter()

    # step4: training
    log(log_file, 'Starting to train...')
    if current_epoch == 0 and os.path.exists(opt.result_file):
        os.remove(opt.result_file)
    result_file = open(opt.result_file, 'a', newline='')
    writer = csv.writer(result_file)
    if current_epoch == 0:
        data_name = ['Epoch', 'Average Loss', 'Val Loss']
        writer.writerow(data_name)
        result_file.flush()

    while current_epoch < opt.max_epoch:
        current_epoch += 1
        running_loss = 0.0
        loss_meter.reset()
        log(log_file)
        log(log_file, f'Training epoch: {current_epoch}')

        for i, (input, target) in tqdm(enumerate(train_dataloader)):
            input = input.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            score_model = model(input)
            score_connect = connect(score_model)
            loss = criterion(score_connect, target)
            loss.backward()
            optimizer.step()

            # log info
            running_loss += loss.item()
            if i % opt.print_freq == opt.print_freq - 1:
                log(log_file, f'loss {running_loss / opt.print_freq:.5f}')
                running_loss = 0.0
            loss_meter.add(loss.item())

        if current_epoch % opt.save_freq == 0:
            model.save_model(current_epoch)
            connect.save_model(current_epoch)
        average_loss = round(loss_meter.value()[0], 5)
        log(log_file, f'Average Loss: {average_loss}')

        # validate
        if current_epoch % opt.val_freq == 0:
            val_error = val(model, connect, val_dataloader, log_file)
            results = [current_epoch, average_loss, val_error]
            writer.writerow(results)
            result_file.flush()

        # update learning rate
        if opt.update_lr:
            if current_epoch % opt.lr_decay_freq == 0:
                lr = lr * 0.5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    log(log_file, f'Updating learning rate: {lr}')

    end_time = time.strftime("%b %d %Y %H:%M:%S")
    log(log_file, f'Training end time: {end_time}')
    log_file.close()
    result_file.close()


def val(model, connect, dataloader, file):
    model.eval()
    connect.eval()
    error_meter = meter.AverageValueMeter()

    for _, (input, target) in enumerate(dataloader):
        batch_size = input.shape[0]
        with t.no_grad():
            input = input.cuda()
            score_model = model(input)
            score_connect = connect(score_model)

        output = score_connect.cpu().numpy().astype(int)
        target = target.numpy()
        distance_error = (output[:, 0] - target[:, 0]) ** 2 + (output[:, 1] - target[:, 1]) ** 2
        for i in range(batch_size):
            error_meter.add(distance_error[i] ** 0.5 * utils.pixel2length)

    model.train()
    connect.train()
    val_error = round(error_meter.value()[0], 5)
    log(file, f'Val Error: {val_error}')
    return val_error


def help():
    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | help
    example: 
            python {0} train --lr=0.01
            python {0} help
    avaiable args:""".format(__file__))

    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    train()
