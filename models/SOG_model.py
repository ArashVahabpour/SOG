import torch
import os
from . import networks
import latent_optimizers
from models.losses import VGGLoss


class SOGModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        # Speedup
        if not opt.is_train:
            torch.backends.cudnn.benchmark = True
        self.is_train = opt.is_train

        self.netG = networks.define_G(opt)
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        # load networks
        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

        if self.is_train:
            self.old_lr = opt.lr
            self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        # Criterion which we use to backprop
        if opt.criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif opt.criterion == 'l1_asym':  # TODO: REMOVE, DEBUG.
            F = torch.nn.functional
            self.criterion = lambda x, y: (5 * F.relu(x - y) + F.relu(y - x)).mean()
        elif opt.criterion == 'vgg':
            self.criterion = VGGLoss()
        elif opt.criterion == 'vgg_l1':
            self.criterion = lambda x,y: torch.nn.L1Loss()(x,y) + VGGLoss()(x,y)
        else:
            raise NotImplementedError('Criterion {} is not implemented!'.format(opt.criterion))

        if opt.latent_optimizer == 'bcs':
            self.latent_optimizer = latent_optimizers.BlockCoordinateSearch(opt.match_criterion)
        elif opt.latent_optimizer == 'lbfgs':
            self.latent_optimizer = latent_optimizers.LBFGS(opt.match_criterion, opt.num_lbfgs_steps)
        else:
            raise NotImplementedError('latent optimizer {} not implemented!'.format(opt.latent_optimizer == 'bcs'))
        # TODO: analyze and remove dependency cycle between latent_optimizer and SOG_model, refer to https://stackoverflow.com/questions/40532274/two-python-class-instances-have-a-reference-to-each-other  / https://www.google.com/search?q=is+it+right+practice+if+two+classes+have+reference+to+one+another+python&oq=is+it+right+practice+if+two+classes+have+reference+to+one+another+python&aqs=chrome..69i57.17847j0j7&sourceid=chrome&ie=UTF-8
        self.latent_optimizer.initialize(opt, self)

    # helper saving function that can be used by subclasses
    def save(self, epoch):
        save_filename = 'G_{}.pth'.format(epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        netG_module = self.netG.module if hasattr(self.netG, 'module') else self.netG  # to avoid DataParallel to affect
        torch.save(netG_module.cpu().state_dict(), save_path)
        if len(self.opt.gpu_ids) and torch.cuda.is_available():
            self.netG.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, epoch_label):
        save_filename = 'G_{}.pth'.format(epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        netG_module = self.netG.module if hasattr(self.netG, 'module') else self.netG  # to avoid DataParallel to affect
        if not os.path.isfile(save_path):
            print('%s does not exist yet!' % save_path)
            raise Exception('Generator must exist!')
        else:
            try:
                netG_module.load_state_dict(torch.load(save_path))
            except Exception:
                raise Exception('The network architecture does not match the saved weights')

    def forward(self, real, infer=False):
        # batch_size x n_gaussian
        best_z = self.latent_optimizer.optimize(real)

        fake = self.netG.forward(best_z)

        # in case we are predicting images with a fully connected net, we have to give it appropriate width and height
        if self.opt.net_type == 'mlp':
            fake = fake.reshape(real.shape)

        loss = self.criterion(fake, real)

        return loss, fake if infer else None

    def decode(self, z, requires_grad=False):
        # TODO: when to toggle benchmark?
        # https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
        # directly give model a Z to generate image 

        if requires_grad:
            y = self.netG(z)
        else:
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
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print('>>>>>>> DEBUG, old lr:{}, new lr:{}'.format(self.old_lr, lr)) # TODO directly assign line's value, call in the beginning if continue_train / lrd = self.opt.lr / self.opt.niter_decay * max(0, opt.start_epoch - opt.niter)
        self.old_lr = lr
