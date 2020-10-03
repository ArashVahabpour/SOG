import argparse
import os
from util import util
import torch
import warnings


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # experiment specifics
        self.parser.add_argument('--name', type=str, default='mnist',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset', type=str, default='mnist',
                                 help='name of the predefined dataset: e.g. mnist, emnist, fashion-mnist, celeba, power, gas, hepmass, miniboone, bsds300, gym')
        self.parser.add_argument('--normalize_data', action='store_true', help='if dataset is normalized to mean 0 and std 1')

        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--net_type', type=str, default='deconv', help='which network to use: e.g. deconv, mlp, flat_mlp, infogail_mlp.')
        self.parser.add_argument('--norm_type', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--last_activation', type=str, default='sigmoid', help='last layer activation: e.g. sigmoid, tanh, none')  # todo link to normalization
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')

        # input/output sizes
        self.parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        self.parser.add_argument('--img_size', type=int, default=28, help='scale images to this size')
        self.parser.add_argument('--n_latent', default=2, type=int, help='# of latent dimensions')
        self.parser.add_argument('--nc', type=int, default=1, help='# of image channels')

        # TODO nc, img_size to be determined by dataset by default

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./datasets/')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tensorboard', action='store_true', help='if specified, use tensorboard logging. Requires tensorboard installed')  #TODO debug this feature

        # for deconv generator
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_deconv', type=int, default=4,
                                 help='number of deconv blocks in the generator network')
        self.parser.add_argument('--n_conv', type=int, default=0,
                                 help='number of conv blocks in the generator network')

        # for mlp generator
        self.parser.add_argument('--n_hidden', type=int, default=128, help='number of hidden neurons in intermediate layers')
        self.parser.add_argument('--mlp_depth', type=int, default=3, help='depth (number of hidden layers)')

        # latent optimizers
        self.parser.add_argument('--latent_optimizer', type=str, default='bcs', help='method to find best latent code: e.g. "bcs" for block coorindate search, or "ohs" for one-hot-search.')
        self.parser.add_argument('--criterion', type=str, default='l1', help='optimization loss function: e.g. l1, mse')  # TODO move to train options

        # block coordinate search
        self.parser.add_argument('--block_size', type=int, default=2, help='size of coordinate search blocks')
        self.parser.add_argument('--n_rounds', type=int, default=1, help='number of coordinate search rounds')
        self.parser.add_argument('--samples_per_dim', type=int, default=32, help='number of samples per dimension')
        self.parser.add_argument('--match_criterion', type=str, default='l1', help='loss function used for finding the matching code: e.g. l1, mse')

        # gym
        self.parser.add_argument('--radii', type=str, default='-10,10,20', help='a list of radii to be sampled uniformly at random for "Circles-v0" environment. a negative sign implies that the circle is to be drawn downwards.')
        self.parser.add_argument('--env_name', type=str, default='Circles-v0', help='environment to train')
        self.parser.add_argument('--gen_expert', action='store_true', help='if specified, generate (new) expert dataset and store on disk')
        self.parser.add_argument('--render_gym', action='store_true', help='if specified, gym environment will get rendered, useful for debugging.')
        # TODO conditionally omit some options / at least from printing in the beginning of the run

    def parse(self, save=True):
        self.opt = self.parser.parse_args()
        self.opt.is_train = self.is_train  # train or test

        if self.opt.latent_optimizer == 'bcs':
            assert self.opt.n_latent % self.opt.block_size == 0, 'n_latent must be divisible by block_size'
            self.opt.latent_batch_size = self.opt.samples_per_dim ** self.opt.block_size
        else:
            self.opt.latent_batch_size = self.opt.n_latent

        # set gpu ids
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            int_id = int(str_id)
            if int_id >= 0:
                self.opt.gpu_ids.append(int_id)
        if len(self.opt.gpu_ids) > 0:
            main_gpu = self.opt.gpu_ids[0]
            torch.cuda.set_device(main_gpu)
            self.opt.device = torch.device('cuda:{}'.format(main_gpu))
        else:
            self.opt.device = torch.device('cpu')

        self.opt.dataset_type = {
            **dict.fromkeys(['mnist', 'emnist', 'fashion-mnist', 'celeba'], 'image'),
            **dict.fromkeys(['power', 'gas', 'hepmass', 'miniboone', 'bsds300'], 'tabular'),
            **dict.fromkeys(['gym'], 'gym'),
        }.get(self.opt.dataset, default=None)

        if self.opt.dataset == 'gym':  # imitation learning of a gym environment
            self.opt.radii = [int(r) for r in self.opt.radii.split(',')]

            if self.opt.env_name == 'Circles-v0':
                # maximum action magnitude in Circles-v0 environment
                self.opt.max_ac_mag = max(map(abs, self.opt.radii)) * 0.075

        # conditional generative model, where some input is provided
        self.opt.is_conditional = self.opt.dataset == 'gym'

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

        if self.opt.batch_size * self.opt.latent_batch_size >= 2 ** 16:
            warnings.warn(
                '''Try reducing samples_per_dim or batch_size, this can cause cuDNN bugs.
                https://github.com/pytorch/pytorch/issues/4107#issuecomment-544739942
                ''', RuntimeWarning)

        return self.opt
