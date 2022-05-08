from config_location import LocationConfig
from torch.utils.data import DataLoader
from dataset import LocationDataset
from inspect import getsource
from torchnet import meter
import numpy as np
import torch as t
import models
import utils
import time
import csv
import os

opt = LocationConfig()
log = utils.log
    
def train(**kwargs):
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    name = time.strftime('location_train_%Y%m%d_%H%M%S')
    log_file = open(f"{opt.save_log_root}/{name}.txt", 'w')

    opt.parse(kwargs, log_file)
    start_time = time.strftime("%b %d %Y %H:%M:%S")
    log(log_file, f'Training start time: {start_time}')

    # step1: configure model
    log(log_file, 'Building model...')
    model = models.model(
        module_name=opt.module_name,
        model_name=opt.model_name,
        input_channel=utils.num_category+4,
        output_channel=utils.num_category+3, 
        pretrained=True
    )
    input_channel = 512
    connect = models.connect(
        module_name=opt.module_name,
        model_name=opt.model_name,
        input_channel=input_channel, 
        output_channel=utils.num_category+3,
        reshape=False
    )
    embedding = models.embedding(
        module_name=opt.module_name,
        model_name=opt.model_name,
        input_channel=13,
        output_channel=256,
        reshape=False
    )
    print(model)
    print('###################')
    print(connect)
    print('###################')
    print(embedding)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('[Network %s] Total number of parameters : %.3f M' % ('model', num_params / 1e6))
    print('-----------------------------------------------')
    num_params = 0
    for param in embedding.parameters():
        num_params += param.numel()
    print('[Network %s] Total number of parameters : %.3f M' % ('embedding', num_params / 1e6))
    print('-----------------------------------------------')
    num_params = 0
    for param in connect.parameters():
        num_params += param.numel()
    print('[Network %s] Total number of parameters : %.3f M' % ('connect', num_params / 1e6))
    print('-----------------------------------------------')

    if opt.load_model_path:
        log(log_file, 'Loading the model: {}'.format(opt.load_model_path))
        model.load_model(opt.load_model_path)
    if opt.load_connect_path:
        log(log_file, 'Loading the model: {}'.format(opt.load_connect_path))
        connect.load_model(opt.load_connect_path)
    if opt.load_embedding_path:
        log(log_file, 'Loading the model: {}'.format(opt.load_embedding_path))
        embedding.load_model(opt.load_embedding_path)

    model.cuda()
    connect.cuda()
    embedding.cuda()

    # step2: data
    log(log_file, 'Building dataset...')
    train_data = LocationDataset(data_root=opt.data_root, mask_size=opt.mask_size, phase='train')
    val_data = LocationDataset(data_root=opt.data_root, mask_size=opt.mask_size, phase='test')
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
        shuffle=False,
        num_workers=opt.num_workers
    )
    
    # step3: criterion and optimizer
    log(log_file, 'Building criterion and optimizer...')
    lr = opt.lr_base
    optimizer = t.optim.Adam(
        list(model.parameters())+list(embedding.parameters())+list(connect.parameters()),
        lr=lr,
        weight_decay=opt.weight_decay
    )
    current_epoch = opt.current_epoch
    weight = t.ones(utils.num_category+3)
    for i in range(utils.num_category+1):
        if i < 13:
            weight[i] = 2
        else:
            weight[i] = 1.25
    weight = weight.cuda()
    criterion = t.nn.CrossEntropyLoss(weight=weight)
    loss_meter = meter.AverageValueMeter() 
            
    # step4: training
    log(log_file, 'Starting to train...')
    if current_epoch == 0 and os.path.exists(opt.result_file):
        os.remove(opt.result_file)
    result_file = open(opt.result_file, 'a', newline='')
    writer = csv.writer(result_file)
    if current_epoch == 0:
        data_name = ['Epoch', 'Average Train Loss', 'Average Val Loss', 'Predict Accuracy', 'Number of Predict Category Right', \
            'Number of Target Category', 'Number of Predict Category', 'Category Accuracy', 'Category Proportion', \
            'Number of Specific Target Category', 'Number of Specific Predict Category', 'Number of Specific Predict Category', 'Accuracy of Specific Category']
        writer.writerow(data_name)
        result_file.flush()

    while current_epoch < opt.max_epoch:
        current_epoch += 1
        running_loss = 0.0
        loss_meter.reset()
        log(log_file)
        log(log_file, f'Training epoch: {current_epoch}')

        for i, (input, target, room_type) in enumerate(train_dataloader):
            input = input.cuda()
            target = target.cuda()
            room_type = room_type.cuda()
            optimizer.zero_grad()
            score_model = model(input)
            score_embedding = embedding(room_type)
            score_temp = t.cat([score_model, score_embedding], 1)
            score_connect = connect(score_temp)
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
            embedding.save_model(current_epoch)
        average_train_loss = round(loss_meter.value()[0], 5)
        log(log_file, f'Average Train Loss: {average_train_loss}')

        # validate
        if current_epoch % opt.val_freq == 0: 
            average_val_loss, predict_accuracy, num_predict_Category_right, num_target_Category, num_predict_Category, \
                Category_accuracy, Category_proportion,num_target_specific_category, \
            num_predict_specific_category,num_predict_specific_category_right, accuracy_specific_category = val(model, connect,embedding, criterion, val_dataloader, log_file)
            results = [current_epoch, average_train_loss, average_val_loss, predict_accuracy, num_predict_Category_right, \
                num_target_Category, num_predict_Category, Category_accuracy, Category_proportion,
                       num_target_specific_category, num_predict_specific_category, num_predict_specific_category_right, accuracy_specific_category]
            writer.writerow(results)
            result_file.flush()

        # update learning rate
        if opt.update_lr:
            if current_epoch % opt.lr_decay_freq == 0:       
                lr = lr * (1 - float(current_epoch) / opt.max_epoch) ** 1.5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    log(log_file, f'Updating learning rate: {lr}')

    end_time = time.strftime("%b %d %Y %H:%M:%S")
    log(log_file, f'Training end time: {end_time}')
    log_file.close()
    result_file.close()

def val(model, connect, embedding, criterion, dataloader, file):
    model.eval()
    connect.eval()
    embedding.eval()
    loss_meter = meter.AverageValueMeter()
    loss_meter.reset()
    predict_accuracy = 0
    num_predict_Category_right = 0
    num_target_Category = 0
    num_predict_Category = 0
    num_target_specific_category = 0
    num_predict_specific_category = 0
    num_predict_specific_category_right = 0
    softmax = t.nn.Softmax(dim=1)

    for _, (input, target, room_type) in enumerate(dataloader):
        batch_size = input.shape[0]
        with t.no_grad():
            input = input.cuda()
            room_type = room_type.cuda()
            target = target.cuda()
            score_model = model(input)
            score_embedding = embedding(room_type)
            score_temp = t.cat([score_model, score_embedding], 1)
            score_connect = connect(score_temp)
            loss = criterion(score_connect, target)
            loss_meter.add(loss.item())
    
        score_softmax = softmax(score_connect)
        output = score_softmax.cpu().numpy()
        predict = np.argmax(output, axis=1)
        target = target.cpu().numpy()
        for i in range(batch_size):
            type = t.where(room_type[i,:]==1)[0][0]
            num_predict = np.sum(predict[i] == target[i]) 
            predict_accuracy += (num_predict / (input.shape[2]*input.shape[3]))
            for k in range(utils.num_category+1):
                num_predict_Category_right += np.sum((predict[i] == k) & (target[i] == k))
                num_target_Category += np.sum(target[i] == k)
                num_predict_Category += np.sum(predict[i] == k)

                if k == type:
                    num_target_specific_category += np.sum(target[i] == k)
                    num_predict_specific_category += np.sum((predict[i] == k))
                    num_predict_specific_category_right += np.sum((predict[i] == k) & (target[i] == k))

    average_val_loss = round(loss_meter.value()[0], 5)
    log(file, f'Average Val Loss: {average_val_loss}')

    model.train()
    connect.train()
    embedding.train()
    predict_accuracy = round(predict_accuracy/len(dataloader.dataset), 5)
    num_predict_Category_right = int(num_predict_Category_right/len(dataloader.dataset))
    num_target_Category = int(num_target_Category/len(dataloader.dataset))
    num_predict_Category = int(num_predict_Category/len(dataloader.dataset))
    num_target_specific_category = int(num_target_specific_category/len(dataloader.dataset))
    num_predict_specific_category = int(num_predict_specific_category/len(dataloader.dataset))
    num_predict_specific_category_right = int(num_predict_specific_category_right / len(dataloader.dataset))
    accuracy_specific_category = round(num_predict_specific_category_right/num_target_specific_category, 5)
    if num_target_Category != 0:
        category_accuracy = round(num_predict_Category_right / num_target_Category, 5)
    else:
        category_accuracy = "nan"
    if num_predict_Category != 0:
        category_proportion = round(num_predict_Category_right / num_predict_Category, 5)
    else:
        category_proportion = "nan"
    log(file, f'Predict Accuracy: {predict_accuracy}')
    log(file, f'Number of Predict Category Right: {num_predict_Category_right}')
    log(file, f'Number of Target Category: {num_target_Category}')
    log(file, f'Number of Predict Category: {num_predict_Category}')
    log(file, f'Category Accuracy: {category_accuracy}')
    log(file, f'Category Proportion: {category_proportion}')
    log(file, f'Number of Specific Target Category: {num_target_specific_category}')
    log(file, f'Number of Specific Predict Category: {num_predict_specific_category}')
    log(file, f'Number of Specific Predict Category Right: {num_predict_specific_category_right}')
    log(file, f'Accuracy of Specific Category: {accuracy_specific_category}')
    return average_val_loss, predict_accuracy, num_predict_Category_right, num_target_Category, num_predict_Category, \
        category_accuracy, category_proportion, num_target_specific_category, num_predict_specific_category, num_predict_specific_category_right, accuracy_specific_category

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

if __name__=='__main__':
    train()