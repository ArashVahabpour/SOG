from typing import Any

import torch
import torch.nn as nn
import functools
from math import log2, ceil  # TODO implement default choice of depth


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer {} is not found'.format(norm_type))
    return norm_layer


def get_activation_layer(activation_type):
    if activation_type == 'sigmoid':
        activation_layer = nn.Sigmoid()
    elif activation_type == 'tanh':
        activation_layer = nn.Tanh()
    elif activation_type == 'none':
        activation_layer = None
    else:
        raise NotImplementedError('normalization layer {} is not found'.format(activation_type))
    return activation_layer


def define_G(opt):
    netG = {
        'deconv': Deconv,
        'mlp': MLP,
        'flat_mlp': FlatMLP,
        'infogail_mlp': InfoGAILMLP,
    }.get(opt.net_type, None)(opt)

    if netG is None:
        raise NotImplementedError('generator of type {} not implemented!'.format(opt.net_type))

    print(netG)

    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netG.cuda(opt.gpu_ids[0])
    netG.apply(weights_init)

    if len(opt.gpu_ids):
        netG = torch.nn.DataParallel(netG, device_ids=opt.gpu_ids)

    return netG


# class Deconv(nn.Module):
#     def __init__(self, opt):
#         super().__init__()
#
#         self.output_size = 2 ** (opt.n_deconv + 1)
#         self.img_size = opt.img_size
#         assert self.output_size >= self.img_size, 'please consider a higher depth.'
#
#         activation = nn.ReLU(True)  # TODO check why true?
#         norm_layer = get_norm_layer(opt.norm)
#         last_activation = nn.Sigmoid()  # TODO change this into an option
#
#         model = []
#
#         for i in range(opt.n_deconv):
#             last_layer = i == opt.n_deconv - 1
#
#             mult = 2 ** (opt.n_deconv - i - 1)
#             model.append(nn.ConvTranspose2d(opt.ngf * mult if i else opt.n_latent,
#                                             int(opt.ngf * mult / 2) if not last_layer else opt.nc,
#                                             kernel_size=4,
#                                             stride=2 if i else 1,
#                                             padding=1 if i else 0,
#                                             bias=False))
#
#             if not last_layer:
#                 model += [norm_layer(int(opt.ngf * mult / 2)), activation]
#             else:
#                 model.append(last_activation)
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, z):
#         y = self.model(z.view(*z.shape, 1, 1))
#         for dim in (2, 3):
#             y = y.narrow(dim, start=(self.output_size - self.img_size)//2, length=self.img_size)
#
#         return y

class Deconv(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.output_size = 2 ** (opt.n_deconv + 1)
        self.img_size = opt.img_size
        assert self.output_size >= self.img_size, 'please consider a higher depth.'  # TODO: verify 2** is exact

        activation = nn.ReLU(True)
        norm_layer = get_norm_layer(opt.norm_type)
        last_activation = get_activation_layer(opt.last_activation)

        model = []

        for i in range(opt.n_deconv):
            last_deconv = i == opt.n_deconv - 1
            last_layer = last_deconv and opt.n_conv == 0

            mult = 2 ** (opt.n_deconv - 1 - i)
            model += [nn.ConvTranspose2d(opt.ngf * mult if i else opt.n_latent,
                                            opt.ngf * mult // 2 if not last_deconv else (opt.nc if last_layer else opt.ngf),
                                            kernel_size=4,
                                            stride=2 if i else 1,
                                            padding=1 if i else 0,
                                            bias=False)]
            if not last_layer:
                model += [norm_layer(opt.ngf * mult // 2 if not last_deconv else opt.ngf), activation]

        for i in range(opt.n_conv):
            last_layer = i == opt.n_conv - 1

            model += [nn.ReflectionPad2d(1),
                      nn.Conv2d(opt.ngf, opt.ngf if not last_layer else opt.nc, kernel_size=3, padding=0)]
            if not last_layer:
                model += [norm_layer(opt.ngf), activation]

        # last layer
        if last_activation is not None:
            model.append(last_activation)

        self.model = nn.Sequential(*model)
        #todo check model layers for combination of deconv and conv

    def forward(self, z):
        y = self.model(z.view(*z.shape, 1, 1))
        for dim in (2, 3):
            y = y.narrow(dim, start=(self.output_size - self.img_size)//2, length=self.img_size)

        return y


class MLP(nn.Module):
    # Multi-Layer Perceptron (Fully Connected) Model; Used for Image Data
    # architecture: n_latent --> n_hidden --> n_hidden --> ... --> n_hidden --> img_size^2
    def __init__(self, opt):
        super().__init__()

        self.output_shape = (opt.nc, opt.img_size, opt.img_size)

        assert (opt.mlp_depth >= 0)
        activation = nn.ReLU()
        last_activation = get_activation_layer(opt.last_activation)

        model = []

        for i in range(opt.mlp_depth):
            last_layer = i + 1 == opt.mlp_depth

            model.append(torch.nn.Linear(opt.n_hidden if i else opt.n_latent,
                                         opt.nc * opt.img_size ** 2 if last_layer else opt.n_hidden))

            if not last_layer:
                model.append(activation)
            elif last_activation is not None:
                model.append(last_activation)

        self.model = nn.Sequential(*model)

    def forward(self, z):
        return self.model(z).view(self.output_shape)


class FlatMLP(nn.Module):
    # Multi-Layer Perceptron (Fully Connected) Model; Used for Tabular Data
    # architecture: n_latent --> n_hidden --> n_hidden --> ... --> n_hidden --> n_features
    def __init__(self, opt):
        super().__init__()

        assert (opt.mlp_depth >= 0)
        activation = nn.ReLU()
        last_activation = get_activation_layer(opt.last_activation)

        model = []

        for i in range(opt.mlp_depth):
            last_layer = i + 1 == opt.mlp_depth

            model.append(torch.nn.Linear(opt.n_hidden if i else opt.n_latent,
                                         opt.n_features if last_layer else opt.n_hidden))

            if not last_layer:
                model.append(activation)
            elif last_activation is not None:
                model.append(last_activation)

        self.model = nn.Sequential(*model)

    def forward(self, z):
        return self.model(z)


class InfoGAILMLP(nn.Module):
    # Multi-Layer Perceptron (Fully Connected) Model Used in InfoGAIL paper
    # https://arxiv.org/pdf/1703.08840.pdf ---> Appendix A

    def __init__(self, opt):
        super().__init__()

        activation = nn.LeakyReLU()
        last_activation = nn.Tanh()

        self.state_encoder = nn.Sequential(
            torch.nn.Linear(opt.state_dim, 128),
            activation,
            torch.nn.Linear(128, 128),
        )

        self.latent_encoder = torch.nn.Linear(opt.n_latent, 128)

        self.decoder = nn.Sequential(
            activation,
            torch.nn.Linear(128, opt.action_dim),
            last_activation,
        )

        self.scale = opt.max_ac_mag

    def forward(self, z, obs):
        return self.decoder(self.state_encoder(obs) + self.latent_encoder(z)) * self.scale
