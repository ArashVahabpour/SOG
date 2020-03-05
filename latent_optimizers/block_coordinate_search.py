import torch
import itertools
from .base_search import BaseSearch


class BlockCoordinateSearch(BaseSearch):
    """
    Block coordinate grid search optimizer over the distribution of points
    in the latent space.
    """

    def __init__(self, match_criterion):
        super().__init__(match_criterion)
        self.match_criterion_no_reduction = self.MatchCriterion(reduction='none')

    def _sample(self, old_z, block_idx):
        """
        Takes the best codes and perturbs

        Args:
            old_z: batch_size x n_latent
        Returns:
            new_z: batch_size x latent_batch_size x n_latent
        """

        new_z = old_z.unsqueeze(1).repeat(1, self.opt.latent_batch_size, 1)
        new_z[:, :, block_idx * self.opt.block_size:(block_idx + 1) * self.opt.block_size].normal_()

        return new_z

    def optimize(self, real):
        """
        Find the loss between the optimal fake data_loaders and the real data_loaders.

        Args:
            real: batch_size x dim_1 x ... x dim_k

        Returns:
            best_z: batch_size x n_latent
        """

        batch_size = real.shape[0]  # to accommodate for the end of the dataset when batchsize might change
        best_z = torch.zeros(batch_size, self.opt.n_latent, device=self.opt.device)

        for round_idx, block_idx in itertools.product(range(self.opt.n_rounds),
                                                      range(self.opt.n_latent // self.opt.block_size)):
            # batch_size x latent_batch_size x n_latent
            new_z = self._sample(best_z, block_idx)

            # (batch_size * latent_batch_size) x n_latent
            new_z_r = new_z.reshape(batch_size * self.opt.latent_batch_size, self.opt.n_latent)

            # (batch_size * latent_batch_size) x dim_1 x ... x dim_k
            torch.cuda.synchronize()  # TODO: fix device input
            with torch.no_grad():  # no need to store the gradients while searching
                fake_all = self.sog_model.decode(new_z_r, requires_grad=False)

            # batch_size x latent_batch_size x dim_1 x ... x dim_k

            all_shape = [batch_size, self.opt.latent_batch_size, *real.shape[1:]]

            # batch_size x latent_batch_size x dim_1 x ... x dim_k
            fake_all = fake_all.reshape(all_shape)
            real_all = real.unsqueeze(1).expand(all_shape)

            # batch_size x latent_batch_size x dim_1 x ... x dim_k
            loss = self.match_criterion_no_reduction(real_all, fake_all)

            # batch_size x latent_batch_size x -1
            loss = loss.reshape([*all_shape[:2], -1])

            # batch_size x latent_batch_size
            loss = loss.mean(dim=2)

            # batch_size
            _, argmin = loss.min(dim=1)

            # new_z: batch_size x latent_batch_size x n_latent
            # best_idx: batch_size x 1 x n_latent
            best_idx = argmin[:, None, None].repeat(1, 1, self.opt.n_latent)

            # batch_size x 1 x n_latent
            best_z = torch.gather(new_z, 1, best_idx)

            # batch_size x n_latent
            best_z = best_z.squeeze(1)  # TODO unit test with batch size of 1

        return best_z
