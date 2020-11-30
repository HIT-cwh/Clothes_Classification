from options.train_options import TrainOptions
from models.models import initialize_model
from data.data_loader import CreateDataLoader

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import datetime
import copy
import os
from tqdm import tqdm


def train_model(model, dataloaders, criterion, optimizer, device, output_path, num_epochs, is_inception,
                save_interval):
    print(output_path)
    since = time.time()
    val_acc_history = []
    # writer = SummaryWriter()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(1, num_epochs+1)):
        start_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for n_iter, data in enumerate(dataloaders[phase]):
                inputs = data['img'].to(device)
                labels = data['catagory'].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                iter_loss = loss.item() * inputs.size(0)
                iter_accuracy = torch.sum(preds == labels.data)

                running_loss += iter_loss
                running_corrects += iter_accuracy

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f} Time: {}'.format(phase, epoch_loss, epoch_acc,
                                                              str(datetime.timedelta(
                                                                  seconds=time.time() - start_time))))
        if epoch % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(output_path, str(epoch) + '.pth'))

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(output_path, 'best.pth'))
        if phase == 'val':
            val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, time_elapsed // 60,
                                                                time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def train_multiGPUS():
    opt = TrainOptions().parse()
    assert opt.isTrain == True

    gpu_ids = opt.gpu_ids
    exp_name = opt.name
    model = opt.model
    batchSize = opt.batchSize
    num_classes = 5
    feature_extract = opt.feature_extract
    nThreads = opt.nThreads
    checkpoints_dir = opt.checkpoints_dir
    serial_batches = opt.serial_batches
    epoch = opt.epoch
    nesterov = opt.nesterov
    save_interval = opt.save_interval
    weight_decay = opt.weight_decay

    model_ft, input_size = initialize_model(model, num_classes, feature_extract)
    data_loader = CreateDataLoader(input_size, batchSize, serial_batches, nThreads)
    dataloaders_dict = data_loader.load_data()
    print('#train images = %d' % len(dataloaders_dict['train'].dataset))
    print('#val images = %d' % len(dataloaders_dict['val'].dataset))

    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        model_ft = torch.nn.DataParallel(model_ft, device_ids=opt.gpu_ids)
    device = gpu_ids[0]
    model_ft.to(device)

    params_to_update = model_ft.parameters()
    print("Params to learn:")

    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad is True:
                params_to_update.append(param)
                print(name, "\t", end='')
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad is True:
                print("\t", name)

    #---------------Train------------------------
    optimizer_ft = optim.SGD(params_to_update, lr=0.0001, momentum=0.9, weight_decay=weight_decay,
                                     nesterov=nesterov)
    criterion = nn.CrossEntropyLoss()
    output_path = os.path.join(checkpoints_dir, exp_name)
    # model_ft.load_state_dict(torch.load(os.path.join(output_path, 'best.pth')))
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, output_path,
                                 num_epochs=epoch,is_inception=(model == "inception"), 
                                 save_interval=save_interval
                                 )


if __name__ == '__main__':
    # train()
    train_multiGPUS()
    # python train.py --gpu_ids 4,5 --name experiment2 --batchSize 8 --nThreads 8 --epoch 10 --save_interval 2