from torch.utils.tensorboard import SummaryWriter

import os
import time

from . import util
from . import html

from io import BytesIO  # Python 3.x


class Visualizer:
    def __init__(self, opt):
        # self.opt = opt
        self.tensorboard = opt.tensorboard
        self.use_html = opt.is_train
        self.win_size = opt.display_winsize
        self.name = opt.name

        if self.tensorboard:
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = SummaryWriter(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):
    #     if self.tensorboard: # show images in tensorboard output
    #         img_summaries = []
    #         for label, image_numpy in visuals.items():
    #             # Write the image to a string
    #             try:
    #                 s = StringIO()
    #             except:
    #                 s = BytesIO()
    #             scipy.misc.toimage(image_numpy).save(s, format="jpeg")
    #             # Create an Image object
    #             img_sum = self.writer.add_image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
    #             # Create a Summary value
    #             img_summaries.append(self.tf.Summary.Value(tag=label, image=img_sum))
    #
    #         # Create and write Summary
    #         summary = self.tf.Summary(value=img_summaries)
    #         self.writer.add_summary(summary, step)

        if self.use_html:   # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.jpg' % (epoch, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.jpg' % (epoch, label))
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_%s_%d.jpg' % (n, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_%s.jpg' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    def plot_current_loss(self, loss, step):
        return
        if self.tensorboard:
            summary = self.tf.Summary(value=[self.tf.Summary.Value(tag='loss', simple_value=loss)])
            self.writer.add_summary(summary, step)

    def print_current_loss(self, epoch, i, loss, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        message += 'loss: %.3f ' % loss

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, img_id):
        image_dir = webpage.get_image_dir()
        # short_path = ntpath.basename(image_path[0])
        name = str(img_id)#os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.jpg' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
