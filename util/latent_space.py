import numpy as np
from scipy import stats, linalg
from math import ceil, log2

import torch
import torchvision

import cv2 as cv
import os


# Converts the tensor of a batch of images to a grid visualisation
def make_grid(image_tensor):  # TODO add normalization option
    batch_size = image_tensor.shape[0]
    grid_width = 2 ** ceil(log2(batch_size) // 2)
    # print('WARNING only>>>>tanh'); image_tensor = image_tensor/2+.5  # todo
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
    z1 = stats.norm.ppf(z1)
    z1 = torch.tensor(z1, device=opt.device, dtype=torch.float32)

    # x_test[i1, i2, ..., ik, :] = x1[i], x1[j], ..., x1[k]
    z_test = torch.cat([xv.unsqueeze(-1) for xv in torch.meshgrid([z1] * opt.n_latent)], dim=-1)
    z_test = z_test.reshape(-1, opt.n_latent)

    y_pred = torch.cat([sog_model.decode(z_test[i:i + 1], requires_grad=False).reshape(-1, opt.nc, opt.img_size, opt.img_size).cpu()
                        for i in range(len(z_test))])  # obtain grid's test results with a batch size of 1

    nrow = grid_width ** ceil(opt.n_latent // 2)
    img = torchvision.utils.make_grid(y_pred, nrow=nrow, padding=2, normalize=False, pad_value=0)
    img = np.ndarray.astype(img[0].numpy() * 255, np.uint8)

    return img


class RandomMotion:
    def __init__(self, dim, tau=10, v_mag=0.02):
        self.dim = dim
        self.tau = tau
        self.v_mag = v_mag
        self.loc = np.random.rand(dim)
        self._renew_v()

    def _renew_v(self):
        rand_dir = np.random.randn(self.dim)
        self.v = rand_dir / linalg.norm(rand_dir) * self.v_mag

    def _update_loc(self):
        self.loc += self.v
        bounds = np.clip(self.loc, 0, 1)
        bounce_mask = bounds != self.loc
        self.loc[bounce_mask] = 2 * bounds[bounce_mask] - self.loc[bounce_mask]
        self.v[bounce_mask] *= -1

    def tick(self):
        if np.random.rand() < 1. / self.tau:
            self._renew_v()
        self._update_loc()

    def generate_seq(self, count=1000):
        locs = []
        for _ in range(count):
            locs.append(self.loc.copy())
            self.tick()
        return np.vstack(locs)


def generate_video(sog_model, opt, web_dir):
    seq = RandomMotion(dim=opt.n_latent, tau=100, v_mag=.02).generate_seq(10000)
    seq = stats.norm.ppf(seq)  # map to normal distribution
    seq = np.clip(seq, -3, 3)  # avoid off-3-sigma values

    z_test = torch.tensor(seq, device=opt.device, dtype=torch.float32)
    y_pred = sog_model.decode(z_test, requires_grad=False).reshape(-1, opt.nc, opt.img_size, opt.img_size).cpu().numpy()
    y_pred = y_pred.transpose(0, 2, 3, 1)  # channels last format for opencv
    if opt.nc == 1:
        y_pred = y_pred.repeat(3, axis=3)  # fake RGB channels
    y_pred = (y_pred * 255).astype(np.uint8)

    video_dir = os.path.join(web_dir, 'morph.avi')
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    video_writer = cv.VideoWriter(video_dir, fourcc, 30., (opt.img_size, opt.img_size))

    for frame in y_pred:
        video_writer.write(frame)

    video_writer.release()

    # TODO refractor web_dir as results_dir
    # TODO organize all this as a class in a reasonable way, ask Yipeng about it
