import torch
from models.losses import VGGLoss


class BaseSearch:
    # match criterion is the loss that initially searches 
    # to find the best matching image (ususally lighter than what we backprop with)
    
    def __init__(self, match_criterion):
        self.opt = None
        self.sog_model = None

        if match_criterion == 'l1':
            self.MatchCriterion = torch.nn.L1Loss
        elif match_criterion == 'l1_asym':
            # TODO: REMOVE, DEBUG.
            F = torch.nn.functional
            raise('correct below')
            self.MatchCriterion = lambda x, y: (5 * F.relu(x - y) + F.relu(y - x))
        elif match_criterion == 'vgg':
            self.MatchCriterion = VGGLoss
        else:
            raise NotImplementedError('match criterion {} is not implemented!'.format(match_criterion))

    def initialize(self, opt, sog_model):
        #TODO: Fix loop between sog_model and latent_optimizers
        self.opt = opt
        self.sog_model = sog_model

    def optimize(self, real):
        pass
