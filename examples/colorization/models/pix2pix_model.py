import torch
from .base_model import BaseModel
from . import networks
import latent_optimizers
import numpy as np
from scipy import stats


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        if not opt.is_train:
            torch.backends.cudnn.benchmark = True

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.is_train:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.n_latent, opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.opt = opt

        if self.is_train:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if opt.latent_optimizer == 'bcs':
            self.latent_optimizer = latent_optimizers.BlockCoordinateSearch(opt.match_criterion)
        else:
            raise NotImplementedError('latent optimizer {} not implemented!'.format(opt.latent_optimizer == 'bcs'))
        # TODO: analyze and remove dependency cycle between latent_optimizer and SOG_model, refer to https://stackoverflow.com/questions/40532274/two-python-class-instances-have-a-reference-to-each-other  / https://www.google.com/search?q=is+it+right+practice+if+two+classes+have+reference+to+one+another+python&oq=is+it+right+practice+if+two+classes+have+reference+to+one+another+python&aqs=chrome..69i57.17847j0j7&sourceid=chrome&ie=UTF-8
        self.latent_optimizer.initialize(opt, self)

        if self.is_train:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        # batch_size x n_gaussian
        best_z = self.latent_optimizer.optimize(self.real_A, self.real_B)

        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # generate the best sample
        self.fake_B = self.netG(best_z, self.real_A)  # G(A)

    def decode(self, z, real_A, requires_grad=False):
        # TODO: when to toggle benchmark?
        #  https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not

        if requires_grad:
            y = self.netG(z, real_A)
        else:
            torch.backends.cudnn.benchmark = False
            self.netG.eval()

            with torch.no_grad():
                y = self.netG(z, real_A)

            self.netG.train()
            torch.backends.cudnn.benchmark = True

        return y

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate gradients for G
        self.optimizer_G.step()             # update G's weights

    def diversify(self):
        cdf_begin = 0.01
        cdf_end = 1 - cdf_begin

        z1 = np.linspace(cdf_begin, cdf_end, self.opt.grid_width)
        z1 = stats.norm.ppf(z1)
        z1 = torch.tensor(z1, device=self.opt.device, dtype=torch.float32)

        # x_test[i1, i2, ..., ik, :] = x1[i], x1[j], ..., x1[k]
        z_test = torch.cat([xv.unsqueeze(-1) for xv in torch.meshgrid([z1] * self.opt.n_latent)], dim=-1)
        z_test = z_test.reshape(-1, self.opt.n_latent)

        self.fake_B_samples = self.decode()

    def test(self):
        self.diversify()
        super().test()

