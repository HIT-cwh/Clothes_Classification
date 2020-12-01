from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--num_classes', type=int, required=True, default=5, help='N classification')
        self.parser.add_argument('-e', '--epoch', help='epoch number', type=int, default=200)

        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        
        self.parser.add_argument('--nesterov', help='open nesterov', action="store_true")
        self.parser.add_argument('--save_interval', type=int, help='iterators number to save', default=50)
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='w-decay (default: 5e-4)')

        self.isTrain = True
