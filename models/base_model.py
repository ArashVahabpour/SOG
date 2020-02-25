import os
import torch
import sys


class BaseModel(torch.nn.Module):
    # def name(self):
    #     return 'BaseModel'

    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        # self.gpu_ids = opt.gpu_ids
        # self.is_train = opt.is_train
        # self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass



    def update_learning_rate():
        pass
