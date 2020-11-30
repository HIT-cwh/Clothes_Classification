import torch.utils.data as data
import torchvision.transforms as transforms
import torch
from PIL import Image
import os
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, phase):
        super(Dataset, self).__init__()
        self.phase = phase
    
    def initialize(self, input_size):
        self.root = 'data'
        path = os.path.join(self.root, self.phase+'.npy')
        self.dir = np.load(path)

        input_size = (input_size, input_size)
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        }
        self.transform = data_transforms[self.phase]

    def __getitem__(self, index):
        path, catagory = self.dir[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        catagory = torch.tensor(int(catagory))
        return {'img': img, 'catagory': catagory}

    def __len__(self):
        return len(self.dir)

    def name(self):
        return self.phase + 'Dataset'

def CreateDataset(input_size):
    dataset_train = Dataset('train')
    print("dataset [%s] was created" % (dataset_train.name()))
    dataset_train.initialize(input_size)
    dataset_val = Dataset('val')
    print("dataset [%s] was created" % (dataset_val.name()))
    dataset_val.initialize(input_size)
    image_dataset = {'train': dataset_train, 'val': dataset_val}
    return image_dataset