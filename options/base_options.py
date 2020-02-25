import argparse
import os
from util import util
import torch


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='mnist',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset', type=str, default='mnist',
                                 help='name of the predefined dataset: e.g. mnist, emnist, fashion-mnist')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='deconv', help='which model to use: e.g. deconv, mlp')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--fp16', action='store_true', default=False, help='train with AMP')  # TODO: develop a fast fp16 version

        # input/output sizes
        self.parser.add_argument('--img_size', type=int, default=28, help='scale images to this size')
        self.parser.add_argument('--n_latent', default=2, type=int, help='# of latent dimensions')
        self.parser.add_argument('--nc', type=int, default=1, help='# of image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./datasets/')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')

        # for deconv generator
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--deconv_depth', type=int, default=2,
                                 help='number of deconv blocks in the generator network')

        # for mlp generator
        self.parser.add_argument('--n_hidden', type=int, default=128, help='number of hidden neurons in intermediate layers')
        self.parser.add_argument('--mlp_depth', type=int, default=28, help='depth (number of hidden layers)')

        # # D_in is input dimension;
        # # H is hidden dimension;
        # # D_out is output dimension.

        # coordinate search optimizer parameters
        self.parser.add_argument('--block_size', type=int, default=2, help='size of coordinate search blocks')
        self.parser.add_argument('--n_rounds', type=int, default=1, help='number of coordinate search rounds')
        self.parser.add_argument('--samples_per_dim', type=int, default=None, help='number of samples per dimension')

        self.initialized = True

    def parse(self, save=True):
        # if not self.initialized:
        self.initialize()
        self.opt = self.parser.parse_args()
        # self.opt.is_train = self.is_train  # train or test

        assert self.opt.n_latent % self.opt.block_size == 0, 'n_latent must be divisible by block_size'
        self.opt.latent_batch_size = self.opt.samples_per_dim ** self.opt.block_size

        str_ids = self.opt.gpu_ids.split(',')
        for str_id in str_ids:
            int_id = int(str_id)
            if int_id >= 0:
                self.opt.gpu_ids.append(int_id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

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
