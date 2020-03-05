import torch
from torch import distributions, optim
from .base_search import BaseSearch


class LBFGS(BaseSearch):
    """
    LBFGS optimizer over the distribution of points in the latent space.
    """

    def __init__(self, match_criterion, num_steps):
        """
        Args:
            num_steps: number of lbfgs steps for each actual data point
        """
        super().__init__(match_criterion)
        self.icdf = distributions.normal.Normal(0, 1).icdf
        self.num_steps = num_steps
        self.match_criterion = self.MatchCriterion()

    def _optimize_instance(self, real_instance):
        z = torch.rand(1, self.opt.n_latent, device=self.opt.device)  # batch_size = 1
        optimizer_z = optim.LBFGS([z.requires_grad_()])  # show that input is a parameter that requires a gradient

        for _ in range(self.num_steps):
            def closure():
                """
                The closure for LBFGS should clear the gradients, compute the loss, and return it.
                """

                # correct the values of z to make sure it is between 0, 1 (and thus a valid cdf)
                z.data.clamp_(0.001, 0.999)

                optimizer_z.zero_grad()
                fake = self.sog_model.decode(self.icdf(z), requires_grad=True)
                loss = self.match_criterion(real_instance, fake)
                loss.backward()

                return loss

            optimizer_z.step(closure)

        # a last correction
        z.data.clamp_(0, 1)

        z.detach_()  # undo requires_grad_
        return z

    def optimize(self, real):
        # TODO: implement parallel processing across instances
        best_z = torch.cat([self._optimize_instance(real_instance) for real_instance in real.split(split_size=1)])
        return best_z
