# import numpy as np
import torch
import os
# from torch.autograd import Variable
# from util.image_pool import ImagePool
# from .base_model import BaseModel
from . import networks
from .latent_optimizers import BlockCoordinateSearch


class SOGmodel(torch.nn.Module):
    # def name(self):
    #     return 'Pix2PixHDModel'
    def __init__(self, opt):
        super().__init__()
        # if opt.resize_or_crop != 'none' or not opt.is_train:  # when training at full res this causes OOM
        if not opt.is_train:
            torch.backends.cudnn.benchmark = True
        self.is_train = opt.is_train

        # Generator network
        self.netG = networks.define_G(opt, gpu_ids=self.gpu_ids)

        # # Discriminator network
        # if self.is_train:
        #     use_sigmoid = opt.no_lsgan
        #     netD_input_nc = input_nc + opt.output_nc
        #     if not opt.no_instance:
        #         netD_input_nc += 1
        #     self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
        #                                   opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        #
        # ### Encoder network
        # if self.gen_features:
        #     self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder',
        #                                   opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)
        # if self.opt.verbose:
        #     print('---------- Networks initialized -------------')

        # load networks
        if not self.is_train or opt.continue_train:
            pretrained_path = '' if not self.is_train else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            # if self.is_train:
            #     self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            # if self.gen_features:
            #     self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)

                # set loss functions and optimizers
        if self.is_train:
            # if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
            #     raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            # self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            # self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)

            # self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            # self.criterionFeat = torch.nn.L1Loss()
            # if not opt.no_vgg_loss:
            #     self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            # self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake')
            #
            # # initialize optimizers
            # # optimizer G
            # if opt.niter_fix_global > 0:
            #     import sys
            #     if sys.version_info >= (3, 0):
            #         finetune_list = set()
            #     else:
            #         from sets import Set
            #         finetune_list = Set()
            #
            #     params_dict = dict(self.netG.named_parameters())
            #     params = []
            #     for key, value in params_dict.items():
            #         if key.startswith('model' + str(opt.n_local_enhancers)):
            #             params += [value]
            #             finetune_list.add(key.split('.')[0])
            #     print(
            #         '------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
            #     print('The layers that are finetuned are ', sorted(finetune_list))
            # else:
            # params = list(self.netG.parameters())
            # if self.gen_features:
            #     params += list(self.netE.parameters())
            self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.latent_optimizer == 'bcs':
                self.latent_optimizer = BlockCoordinateSearch(opt)
            # # optimizer D
            # params = list(self.netD.parameters())
            # self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    # helper saving function that can be used by subclasses
    def save(self, epoch):
        save_filename = 'G_{}.pth'.format(epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.netG.cpu().state_dict(), save_path)
        if len(self.gpu_ids) and torch.cuda.is_available():
            self.netG.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise Exception('Generator must exist!')
        else:
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                raise Exception('The network architecture does not match the saved weights')
                                    # model_dict = network.state_dict()
                # try:
                #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                #     network.load_state_dict(pretrained_dict)
                #     if self.opt.verbose:
                #         print(
                #             'Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                # except:
                #     print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                #     for k, v in pretrained_dict.items():
                #         if v.size() == model_dict[k].size():
                #             model_dict[k] = v
                #
                #     if sys.version_info >= (3, 0):
                #         not_initialized = set()
                #     else:
                #         from sets import Set
                #         not_initialized = Set()
                #
                #     for k, v in model_dict.items():
                #         if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                #             not_initialized.add(k.split('.')[0])
                #
                #     print(sorted(not_initialized))
                #     network.load_state_dict(model_dict)

    def forward(self, z, eval=False):
        fake_image = self.netG.forward(z)

        # # VGG feature matching loss
        # loss_G_VGG = 0
        # if not self.opt.no_vgg_loss:
        #     loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        #
        # # Only return the fake_B image if necessary to save BW
        # return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake),
        #         None if not infer else fake_image]

    # def inference(self, label, inst, image=None):
    #     # # Encode Inputs
    #     # image = Variable(image) if image is not None else None
    #     # input_label, inst_map, real_image, _ = self.encode_input(Variable(label), Variable(inst), image, infer=True)
    #     #
    #     # # Fake Generation
    #     # if self.use_features:
    #     #     if self.opt.use_encoded_image:
    #     #         # encode the real image to get feature map
    #     #         feat_map = self.netE.forward(real_image, inst_map)
    #     #     else:
    #     #         # sample clusters from precomputed features
    #     #         feat_map = self.sample_features(inst_map)
    #     #     input_concat = torch.cat((input_label, feat_map), dim=1)
    #     # else:
    #     #     input_concat = input_label
    #     #
    #     # if torch.__version__.startswith('0.4'):
    #     #     with torch.no_grad():
    #     #         fake_image = self.netG.forward(input_concat)
    #     # else:
    #     #     fake_image = self.netG.forward(input_concat)
    #     # return fake_image
    #     return

    # def save(self, which_epoch):
    #     self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        # self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        # if self.gen_features:
        #     self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        # if self.gen_features:
        #     params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        # if self.opt.verbose:
        #     print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        # for param_group in self.optimizer_D.param_groups:
        #     param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        # if self.opt.verbose:
        #     print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
