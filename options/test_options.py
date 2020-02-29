from .base_options import BaseOptions


class TestOptions(BaseOptions):
    # def initialize(self):
    def __init__(self):
        super().__init__()
        # BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.is_train = False
