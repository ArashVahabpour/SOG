from .base_options import BaseOptions


class TestOptions(BaseOptions):
    # def initialize(self):
    def __init__(self):
        super().__init__()
        # BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test batches to run')
        self.parser.add_argument('--grid_width', type=int, default=6, help='how many splits per dimension for full grid of latent space')
        self.is_train = False
