import torch
import os
from . import networks


class SOGModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        if not opt.is_train:
            torch.backends.cudnn.benchmark = True
        self.is_train = opt.is_train

        self.netG = networks.define_G(opt)
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        # load networks
        if not self.is_train or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)

        if self.is_train:
            self.old_lr = opt.lr
            self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        if opt.criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        else:
            raise NotImplementedError('criterion {} is not implemented!'.format(opt.criterion))

        self.latent_optimizer = None

    # helper saving function that can be used by subclasses
    def save(self, epoch):
        save_filename = 'G_{}.pth'.format(epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.netG.cpu().state_dict(), save_path)
        if len(self.opt.gpu_ids) and torch.cuda.is_available():
            self.netG.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '{}_{}.pth'.format(network_label, epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise Exception('Generator must exist!')
        else:
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                raise Exception('The network architecture does not match the saved weights')

    def set_latent_optimizer(self, latent_optimizer):
        self.latent_optimizer = latent_optimizer

    def forward(self, real, infer=False):
        # batch_size x n_gaussian
        best_z = self.latent_optimizer.optimize(real)

        fake = self.netG.forward(best_z)

        # in case we are predicting images with a fully connected net, we have to give it appropriate width and height
        if self.opt.net_type == 'mlp':
            fake = fake.reshape(real.shape)

        loss = self.criterion(real, fake)

        return loss, fake if infer else None

    def inference(self, z):
        torch.backends.cudnn.benchmark = False
        self.netG.eval()

        with torch.no_grad():
            y = self.netG(z)

        self.netG.train()
        torch.backends.cudnn.benchmark = True

        return y

    def update_fixed_params(self):
        params = list(self.netG.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        self.old_lr = lr
