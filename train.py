import os
import numpy as np
import time
from math import gcd
from collections import OrderedDict
from options.train_options import TrainOptions
from data_loaders import create_data_loader
from models.SOG_model import SOGModel
import util.latent_space as latent_space
from util.visualizer import Visualizer


def lcm(a, b): return abs(a * b) / gcd(a, b) if a and b else 0


opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

if opt.continue_train:
    if opt.which_epoch == 'latest':
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
    else:
        start_epoch, epoch_iter = int(opt.which_epoch), 0
    print('Resuming from epoch {} at iteration {}'.format(start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batch_size)

data_loader = create_data_loader(opt)
dataset = data_loader.dataset
dataset_size = len(dataset)
print('#training images = %d' % dataset_size)

sog_model = SOGModel(opt)
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
    for _, (data, _) in enumerate(data_loader, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size
        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        data = data.to(opt.device)
        loss, generated = sog_model(data, infer=save_fake)

        # update generator weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print out errors
        if total_steps % opt.print_freq == print_delta:
            # errors = {k: v.data_loaders.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_loss(epoch, epoch_iter, loss, t)  # loss.data_loaders.item?! TODO
            visualizer.plot_current_loss(loss, total_steps)

        # display output images
        if save_fake:
            visuals = OrderedDict([('Real Data', latent_space.make_grid(data/2+.5)),
                                   ('Synthesized Data (Reconstructed)', latent_space.make_grid(generated.data/2+.5))])
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

    # TODO generate full grid here

    # save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        sog_model.save('latest')
        sog_model.save(epoch)
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

    # linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        sog_model.update_learning_rate()
