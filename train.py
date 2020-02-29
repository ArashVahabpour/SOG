import os
import numpy as np
import time
from math import gcd
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import create_data_loader
from models.SOG_model import SOGModel
import util.util as util
import latent_optimizers
from util.visualizer import Visualizer


def lcm(a, b): return abs(a * b) / gcd(a, b) if a and b else 0


opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

# opt.print_freq = lcm(opt.print_freq, opt.batchSize)
# if opt.debug:
#     opt.display_freq = 1
#     opt.print_freq = 1
#     opt.niter = 1
#     opt.niter_decay = 0
#     opt.max_dataset_size = 10

data_loader = create_data_loader(opt)
dataset = data_loader.dataset
dataset_size = len(dataset)
print('#training images = %d' % dataset_size)

sog_model = SOGModel(opt)
if opt.latent_optimizer == 'bcs':
    latent_optimizer = latent_optimizers.BlockCoordinateSearch()
else:
    raise NotImplementedError('latent optimizer {} not implemented!'.format(opt.latent_optimizer == 'bcs'))

# TODO: analyze and remove dependency cycle between latent_optimizer and SOG_model, refer to https://stackoverflow.com/questions/40532274/two-python-class-instances-have-a-reference-to-each-other  / https://www.google.com/search?q=is+it+right+practice+if+two+classes+have+reference+to+one+another+python&oq=is+it+right+practice+if+two+classes+have+reference+to+one+another+python&aqs=chrome..69i57.17847j0j7&sourceid=chrome&ie=UTF-8

latent_optimizer.initialize(opt, sog_model)
sog_model.set_latent_optimizer(latent_optimizer)

visualizer = Visualizer(opt)

optimizer = sog_model.optimizer

total_steps = (start_epoch - 1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, (data, _) in enumerate(data_loader, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size
        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        data = data.to(opt.device)
        loss, generated = sog_model(data, infer=save_fake)

        # # sum per device losses
        # losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        # loss_dict = dict(zip(model.module.loss_names, losses))

        # # calculate final loss scalar
        # loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        # loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)

        # update generator weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print out errors
        if total_steps % opt.print_freq == print_delta:
            # errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_loss(epoch, epoch_iter, loss, t)  # loss.data.item?! TODO
            visualizer.plot_current_loss(loss, total_steps)

        # display output images
        if save_fake:
            visuals = OrderedDict([('real_image', util.make_grid(data)),
                                   ('synthesized_image', util.make_grid(generated.data))])
            visualizer.display_current_results(visuals, epoch, total_steps)

        # save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            sog_model.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        sog_model.save('latest')
        sog_model.save(epoch)
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

    # linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        sog_model.update_learning_rate()
