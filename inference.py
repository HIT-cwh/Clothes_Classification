from options.inference_options import InferenceOptions
from models.models import initialize_model

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import datetime
import copy
import os
from tqdm import tqdm

import torchvision.transforms as transforms
from PIL import Image



def Transform(path, input_size):
    input_size = (input_size, input_size)
    transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    img = Image.open(path).convert('RGB')
    img = transform(img).unsqueeze(0)
    return img

def load_ckpt(model_ft, load_path):
    state_dict = torch.load(load_path)
    for k, v in state_dict.items():
        if k[:7] != 'module.':
            print('load checkpoints trained by single GPU...')
            model_ft.load_state_dict(state_dict)
            return model_ft
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    print('load checkpoints trained by multi GPUs...')
    for k, v in state_dict.items():
        namekey = k.replace('module.', '', 1)
        # k[7:] # remove `module.`
        new_state_dict[namekey] = v
    # load params
    model_ft.load_state_dict(new_state_dict)
    return model_ft

def inference():
    opt = InferenceOptions().parse()
    assert opt.isTrain == False
    category = ['jeans', 'shorts', 'sweater', 'trousers', 't-shirts']

    gpu_ids = opt.gpu_ids
    exp_name = opt.name
    model = opt.model
    num_classes = 5
    feature_extract = opt.feature_extract
    checkpoints_dir = opt.checkpoints_dir
    which_epoch = opt.which_epoch
    dataroot = opt.dataroot

    model_ft, input_size = initialize_model(model, num_classes, feature_extract)
    device = gpu_ids[0]
    model_ft.to(device)

    inputs = Transform(dataroot, input_size)
    inputs = inputs.to(device)
    load_path = os.path.join(checkpoints_dir, exp_name, which_epoch + '.pth')
    # model_ft.load_state_dict(torch.load(os.path.join(output_path, which_epoch + '.pth')))
    model_ft = load_ckpt(model_ft, load_path)
    model_ft.eval()

    outputs = model_ft(inputs)
    _, preds = torch.max(outputs, 1)
    print(category[preds])

if __name__ == '__main__':
    inference()