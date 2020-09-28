import torch
import itertools
from .base_search import BaseSearch


class OneHotSearch(BaseSearch):
    """
    Search over possible one-hot latent codes.
    """

    def __init__(self, match_criterion):
        super().__init__(match_criterion)
        self.match_criterion_no_reduction = self.MatchCriterion(reduction='none')

    def _one_hot_codes(self, batch_size):
        """
        Create possible one-hot codes for each batch sample

        Returns:
            z: batch_size x n_latent x n_latent
        """

        z = torch.eye(self.opt.n_latent, device=self.opt.device).unsqueeze(0).expand(batch_size, -1, -1)
        return z

    def optimize(self, real_y, real_x=None):
        """
        Find the loss between the optimal fake data and the real data.

        Args:
            real_y: batch_size x dim_1 x ... x dim_ky
            real_x: batch_size x dim_1 x ... x dim_kx

        Returns:
            best_z: batch_size x n_latent
        """

        if not self.opt.is_conditional:
            raise NotImplementedError('one-hot search has been only implemented for the case of conditional generative model.')

        batch_size = real_y.shape[0]  # to accommodate for the end of the dataset when batch size might change
        # batch_size x latent_batch_size x n_latent
        all_z = self._one_hot_codes(batch_size)

        best_z = self.search_iter(all_z, real_y=real_y, real_x=real_x)

        return best_z
