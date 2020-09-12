import os
from collections import OrderedDict
from options.test_options import TestOptions
from data_loaders import create_data_loader
from models.SOG_model import SOGModel
from util import util, latent_space
from util.visualizer import Visualizer
from util import kde, html

opt = TestOptions().parse(save=False)
# opt.batchSize = 1  # test code only supports batchSize = 1  #TODO keep to beat batchnorm

data_loader = create_data_loader(opt)

# test
sog_model = SOGModel(opt)

if opt.dataset in ['power', 'gas', 'hepmass', 'miniboone', 'bsds300']:  # if tabular dataset
    kde = kde.KDE(opt, sog_model, n_parzen=10000)
    print('{:.2f}'.format(kde(data_loader)))

else:  # if image dataset
    visualizer = Visualizer(opt)

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, 'test_{}'.format(opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = {}, Epoch = {}'.format(opt.name, opt.which_epoch))

    # for i, (data, _) in enumerate(data_loader):
    #     idx = i + 1
    #     if idx > opt.how_many:
    #         break
    #
    #     _, generated = sog_model(data.to(opt.device), True)
    #
    #     visuals = OrderedDict([('Real Data', util.make_grid(data)),
    #                            ('Synthesized Data (Reconstructed)', util.make_grid(generated.data))])
    #
    #     print('processing batch {}/{}...'.format(idx, opt.how_many))
    #     visualizer.save_images(webpage, visuals, idx)
    #
    # webpage.save()
    #
    # print('generating a full grid...')
    # full_grid = latent_space.generate_full_grid(sog_model, opt)
    # util.save_image(full_grid, os.path.join(web_dir, 'full_grid_{}.png'.format(opt.which_epoch)))

    print('creating morphing video...')
    latent_space.generate_video(sog_model, opt, web_dir)
