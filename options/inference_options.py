from .base_options import BaseOptions


class InferenceOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='best', help='which epoch to load? set to best to use best model')
        self.parser.add_argument('--dataroot', required=True, type=str, help='path to the image')

        self.isTrain = False
