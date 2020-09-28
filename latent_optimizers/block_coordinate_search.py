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
        Take old optimum code and repeat code 'latent_batch_size' times
        Then sample 'block_size' blocks from a normal distribution

        Args:
            old_z: batch_size x n_latent

        Returns:
            new_z: batch_size x latent_batch_size x n_latent
        """
        new_z = old_z.unsqueeze(1).repeat(1, self.opt.latent_batch_size, 1)
        new_z[:, :, block_idx * self.opt.block_size:(block_idx + 1) * self.opt.block_size].normal_()

        return new_z

    def optimize(self, real_y, real_x=None):
        """
        Find the loss between the optimal fake data and the real data.

        Args:
            real_y: batch_size x dim_1 x ... x dim_ky
            real_x: batch_size x dim_1 x ... x dim_kx

        Returns:
            best_z: batch_size x n_latent
        """

        batch_size = real_y.shape[0]  # to accommodate for the end of the dataset when batchsize might change
        # TODO: Perhaps we can initialize from best z of last epoch or normal
        best_z = torch.zeros(batch_size, self.opt.n_latent, device=self.opt.device)
        # Go back over the latent vector and re-search 
        for round_idx, block_idx in itertools.product(range(self.opt.n_rounds),
                                                      range(self.opt.n_latent // self.opt.block_size)):
            # batch_size x latent_batch_size x n_latent
            new_z = self._sample(best_z, block_idx)
            best_z = self.search_iter(new_z, real_y=real_y, real_x=real_x)

        return best_z
