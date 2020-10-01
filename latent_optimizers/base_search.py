import torch
from models.losses import VGGLoss


class BaseSearch:
    def __init__(self, match_criterion):
        self.opt = None
        self.sog_model = None

        # match criterion is the loss that searches for the best match
        if match_criterion == 'l1':
            self.MatchCriterion = torch.nn.L1Loss
        elif match_criterion == 'mse':
            self.MatchCriterion = torch.nn.MSELoss
        elif match_criterion == 'vgg':
            self.MatchCriterion = VGGLoss
        else:
            raise NotImplementedError('match criterion {} is not implemented!'.format(match_criterion))

    def initialize(self, opt, sog_model):
        #TODO: Fix loop between sog_model and latent_optimizers
        self.opt = opt
        self.sog_model = sog_model

    def search_iter(self, all_z, real_y, real_x=None):
        batch_size = real_y.shape[0]

        # (batch_size * latent_batch_size) x n_latent
        all_z_r = all_z.reshape(batch_size * self.opt.latent_batch_size, self.opt.n_latent)

        # batch_size x dim_1 x ... x dim_k ---> batch_size x latent_batch_size x dim_1 x ... x dim_k
        all_shape = lambda tensor: [tensor.shape[0], self.opt.latent_batch_size, *tensor.shape[1:]]

        if self.opt.is_conditional:
            # batch_size x latent_batch_size x dim_1 x ... x dim_kx
            all_shape_x = all_shape(real_x)
            real_x_all = real_x.unsqueeze(1).expand(all_shape_x)
            # (batch_size * latent_batch_size) x dim_1 x ... x dim_kx
            real_x_all_r = real_x_all.reshape(-1, *real_x_all.shape[2:])
        else:
            real_x_all_r = None

        if len(self.opt.gpu_ids) > 0:
            torch.cuda.synchronize()  # TODO: fix device input (+ anywhere else that `synchronize` is called)
        with torch.no_grad():  # no need to store the gradients while searching
            # (batch_size * latent_batch_size) x dim_1 x ... x dim_ky
            fake_all = self.sog_model.decode(all_z_r, real_x=real_x_all_r, requires_grad=False)

        # batch_size x latent_batch_size x dim_1 x ... x dim_ky
        all_shape_y = all_shape(real_y)
        fake_all = fake_all.reshape(all_shape_y)
        real_all = real_y.unsqueeze(1).expand(all_shape_y)

        # batch_size x latent_batch_size x dim_1 x ... x dim_ky
        loss = self.match_criterion_no_reduction(real_all, fake_all)

        # batch_size x latent_batch_size x -1
        loss = loss.reshape([*all_shape_y[:2], -1])

        # batch_size x latent_batch_size
        loss = loss.mean(dim=2)

        # batch_size
        _, argmin = loss.min(dim=1)

        # new_z: batch_size x latent_batch_size x n_latent
        # best_idx: batch_size x 1 x n_latent
        best_idx = argmin[:, None, None].repeat(1, 1, self.opt.n_latent)

        # batch_size x 1 x n_latent
        best_z = torch.gather(all_z, 1, best_idx)

        # batch_size x n_latent
        best_z = best_z.squeeze(1)  # TODO unit test with batch size of 1

        return best_z

    def optimize(self, real_y, real_x=None):
        pass
