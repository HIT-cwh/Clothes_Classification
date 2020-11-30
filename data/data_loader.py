from data.dataset2 import CreateDataset
import torch.utils.data

class DatasetDataLoader():
    def __init__(self):
        pass

    def name(self):
        return 'DatasetDataLoader'

    def initialize(self, input_size, batchSize, serial_batches, nThreads):
        # self.dataset = Dataset()
        # print("dataset [%s] was created" % (self.dataset.name()))
        # self.dataset.initialize(opt)
        dataset = CreateDataset(input_size)
        self.dataloaders_dict = {
            x: torch.utils.data.DataLoader(
                dataset[x],
                batch_size=batchSize,   #'--batchSize', type=int, default=1, help='input batch size'
                shuffle=not serial_batches,   #'--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly'
                num_workers=int(nThreads))
            for x in ['train', 'val']
        }
        # self.dataloader = torch.utils.data.DataLoader(
        #     self.dataset,
        #     batch_size=batchSize,   #'--batchSize', type=int, default=1, help='input batch size'
        #     shuffle=not serial_batches,   #'--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly'
        #     num_workers=int(nThreads))   #'--nThreads', default=2, type=int, help='# threads for loading data'

    def load_data(self):
        return self.dataloaders_dict

    # def __len__(self):
    #     return len(self.dataset)

def CreateDataLoader(input_size, batchSize, serial_batches, nThreads):
    data_loader = DatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(input_size, batchSize, serial_batches, nThreads)
    return data_loader