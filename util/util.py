from __future__ import print_function
import torch
import torchvision
import os

import numpy as np
from scipy.stats import norm
from math import ceil, log2
from PIL import Image


# Converts the tensor of a batch of images to a grid visualisation
def make_grid(image_tensor):  # TODO add normalization option
    batch_size = image_tensor.shape[0]
    grid_width = 2 ** ceil(log2(batch_size) // 2)
    img = torchvision.utils.make_grid(image_tensor, nrow=grid_width, padding=2,
                                      normalize=False, range=None, scale_each=False,
                                      pad_value=0)
    img = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)  # changing float mode to uint8

    return img


def generate_full_grid(sog_model, opt):  # TODO add this to training
    grid_width = opt.grid_width

    cdf_begin = 0.01
    cdf_end = 1 - cdf_begin

    z1 = np.linspace(cdf_begin, cdf_end, grid_width)
    z1 = norm.ppf(z1)
    z1 = torch.tensor(z1, device=device, dtype=torch.float32)

    # x_test[i1, i2, ..., ik, :] = x1[i], x1[j], ..., x1[k]
    z_test = torch.cat([xv.unsqueeze(-1) for xv in torch.meshgrid([z1] * opt.n_latent)], dim=-1)
    z_test = z_test.reshape(-1, opt.n_latent)

    y_pred = sog_model.inference(z_test).reshape(-1, 1, opt.img_size, opt.img_size).cpu()

    nrow = grid_width ** ceil(opt.n_latent // 2)
    img = torchvision.utils.make_grid(y_pred, nrow=nrow, padding=2, normalize=False, pad_value=0)
    img = np.ndarray.astype(img[0].numpy() * 255, np.uint8)

    return img


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
