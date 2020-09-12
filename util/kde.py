from scipy.stats import gaussian_kde
import torch
import numpy as np


class KDE:
    def __init__(self, opt, sog_model, n_parzen=10000):
        fake_samples = []
        for _ in range(n_parzen // opt.batch_size):
            torch.cuda.synchronize()
            fake_samples.append(sog_model.decode(torch.empty(opt.batch_size, opt.n_latent).normal_(), requires_grad=False))
        values = torch.cat(fake_samples).cpu().numpy()

        self._kde = gaussian_kde(values.T)

    def __call__(self, data_loader):
        values = []

        print('please wait while log-likelihoods are calculated...')
        for i, (data, _) in enumerate(data_loader):
            print('processing batch #{}/{} ...'.format(i, len(data_loader)))
            values.append(np.log(self._kde(data.cpu().numpy().T)))  # append log-likelihoods

        return np.concatenate(values).mean()
