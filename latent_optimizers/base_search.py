import torch


class BaseSearch:
    def __init__(self):
        self.opt = None
        self.sog_model = None
        self.match_criterion_no_reduction = None

    def initialize(self, opt, sog_model):
        self.opt = opt
        self.sog_model = sog_model

        if opt.match_criterion == 'l1':
            self.match_criterion_no_reduction = torch.nn.L1Loss(reduction='none')
        else:
            raise NotImplementedError('match criterion {} is not implemented!'.format(opt.match_criterion))

    def optimize(self, real):
        pass
