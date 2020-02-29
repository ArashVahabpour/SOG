import torch
import torch.nn as nn
import functools
from math import log2, ceil # TODO implement default choice of depth


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer {} is not found'.format(norm_type))
    return norm_layer


def define_G(opt):
    if opt.net_type == 'deconv':
        netG = Deconv(opt)
    elif opt.net_type == 'mlp':
        netG = MLP(opt)
    else:
        raise NotImplementedError('generator of type {} not implemented!'.format(opt.net_type))

    print(netG)

    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netG.cuda(opt.gpu_ids[0])
    netG.apply(weights_init)

    if opt.is_train and len(opt.gpu_ids):
        netG = torch.nn.DataParallel(netG, device_ids=opt.gpu_ids)

    return netG


class Deconv(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.output_size = 2 ** (opt.deconv_depth + 1)
        self.img_size = opt.img_size
        assert self.output_size > self.img_size, 'please consider a higher depth.'

        activation = nn.ReLU(True)  # TODO check why true?
        norm_layer = get_norm_layer(opt.norm)
        last_activation = nn.Sigmoid()  # TODO change this into an option

        model = []

        for i in range(opt.deconv_depth):
            last_layer = i == opt.deconv_depth - 1

            mult = 2 ** (opt.deconv_depth - i - 1)
            model.append(nn.ConvTranspose2d(opt.ngf * mult if i else opt.n_latent,
                                            int(opt.ngf * mult / 2) if not last_layer else opt.nc,
                                            kernel_size=4,
                                            stride=2 if i else 1,
                                            padding=1 if i else 0,
                                            bias=False))

            if not last_layer:
                model += [norm_layer(int(opt.ngf * mult / 2)), activation]
            else:
                model.append(last_activation)

        self.model = nn.Sequential(*model)

    def forward(self, z):
        y = self.model(z.view(*z.shape, 1, 1))
        for dim in (2, 3):
            y = y.narrow(dim, start=(self.output_size - self.img_size)//2, length=self.img_size)
        return y


class MLP(nn.Module):
    # Multi-Layer Perceptron (Fully Connected) Model
    # architecture: n_latent --> n_hidden --> n_hidden --> ... --> n_hidden --> img_size^2
    def __init__(self, opt):
        super().__init__()

        self.output_shape = (opt.nc, opt.img_size, opt.img_size)

        assert (opt.mlp_depth >= 0)
        activation = nn.ReLU()
        last_activation = nn.Sigmoid()  # TODO change this into an option

        model = []

        for i in range(opt.mlp_depth):
            last_layer = i + 1 == opt.deconv_depth

            model += [torch.nn.Linear(opt.n_hidden if i else opt.n_latent,
                                      opt.nc * opt.img_size ** 2 if last_layer else opt.n_hidden),
                      torch.nn.ReLU(), last_activation if last_layer else activation]

        self.model = nn.Sequential(*model)

    def forward(self, z):
        return self.model(z).view(self.output_shape)
