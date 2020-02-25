import torch
from models.networks import conv_model, mlp_model
import re

def create_model(opt):
    if 'deconv' in opt.model:
        pass
    else:
    	model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.is_train and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
