import argparse
import os
from util import util
import torch


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # experiment specifics
        self.parser.add_argument('--name', type=str, default='mnist',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset', type=str, default='mnist',
                                 help='name of the predefined dataset: e.g. mnist, emnist, fashion-mnist')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--net_type', type=str, default='deconv', help='which network to use: e.g. deconv, mlp')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')

        # input/output sizes
        self.parser.add_argument('--img_size', type=int, default=28, help='scale images to this size')
        self.parser.add_argument('--n_latent', default=2, type=int, help='# of latent dimensions')
        self.parser.add_argument('--nc', type=int, default=1, help='# of image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./datasets/')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tensorboard', action='store_false', help='if specified, use tensorboard logging. Requires tensorboard installed')  # TODO store_true

        # for deconv generator
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--deconv_depth', type=int, default=4,
                                 help='number of deconv blocks in the generator network')

        # for mlp generator
        self.parser.add_argument('--n_hidden', type=int, default=128, help='number of hidden neurons in intermediate layers')
        self.parser.add_argument('--mlp_depth', type=int, default=28, help='depth (number of hidden layers)')

        # coordinate search optimizer parameters
        self.parser.add_argument('--block_size', type=int, default=2, help='size of coordinate search blocks')
        self.parser.add_argument('--n_rounds', type=int, default=1, help='number of coordinate search rounds')
        self.parser.add_argument('--samples_per_dim', type=int, default=32, help='number of samples per dimension')
        self.parser.add_argument('--match_criterion', type=str, default='l1', help='loss function used for finding the matching code')
        self.parser.add_argument('--criterion', type=str, default='l1', help='optimization loss function')

    def parse(self, save=True):
        self.opt = self.parser.parse_args()
        self.opt.is_train = self.is_train  # train or test

        assert self.opt.n_latent % self.opt.block_size == 0, 'n_latent must be divisible by block_size'
        self.opt.latent_batch_size = self.opt.samples_per_dim ** self.opt.block_size

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            int_id = int(str_id)
            if int_id >= 0:
                self.opt.gpu_ids.append(int_id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            main_gpu = self.opt.gpu_ids[0]
            torch.cuda.set_device(main_gpu)
            self.opt.device = torch.device('cuda:{}'.format(main_gpu))
        else:
            self.opt.device = torch.device('cpu')

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

        return self.opt
