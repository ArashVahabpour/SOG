from .pix2pix_model import Pix2PixModel
import torch
from skimage import color  # used for lab2rgb
import numpy as np
from scipy import stats

class ColorizationModel(Pix2PixModel):
    """This is a subclass of Pix2PixModel for image colorization (black & white image -> colorful images).

    The model training requires '-dataset_model colorization' dataset.
    It trains a pix2pix model, mapping from L channel to ab channels in Lab color space.
    By default, the colorization dataset will automatically set '--input_nc 1' and '--output_nc 2'.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add طtraining-specific or test-specific options.

        Returns:
            the modified parser.

        By default, we use 'colorization' dataset for this model.
        See the original pix2pix paper (https://arxiv.org/pdf/1611.07004.pdf) and colorization results (Figure 9 in the paper)
        """
        Pix2PixModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(dataset_mode='colorization')
        return parser

    def __init__(self, opt):
        """Initialize the class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        For visualization, we set 'visual_names' as 'real_A' (input real image),
        'real_B_rgb' (ground truth RGB image), and 'fake_B_rgb' (predicted RGB image)
        We convert the Lab image 'real_B' (inherited from Pix2pixModel) to a RGB image 'real_B_rgb'.
        we convert the Lab image 'fake_B' (inherited from Pix2pixModel) to a RGB image 'fake_B_rgb'.
        """
        # reuse the pix2pix model
        Pix2PixModel.__init__(self, opt)
        # specify the images to be visualized.
        self.visual_names = ['real_A', 'real_B_rgb', 'fake_B_rgb']

        if not self.is_train:  # if test
            self.visual_names += ['fake_B_rgb_sample_{}'.format(i) for i in range(self.opt.grid_width ** self.opt.n_latent)]

    def lab2rgb(self, L, AB):
        """Convert an Lab tensor image to a RGB numpy output
        Parameters:
            L  (1-channel tensor array): L channel images (range: [-1, 1], torch tensor array)
            AB (2-channel tensor array):  ab channel images (range: [-1, 1], torch tensor array)

        Returns:
            rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
        """
        AB2 = AB * 110.0
        L2 = (L + 1.0) * 50.0
        Lab = torch.cat([L2, AB2], dim=1)
        Lab = Lab[0].data.cpu().float().numpy()
        Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
        rgb = color.lab2rgb(Lab) * 255
        return rgb

    def diversify(self):
        cdf_begin = 0.01
        cdf_end = 1 - cdf_begin

        z1 = np.linspace(cdf_begin, cdf_end, self.opt.grid_width)
        z1 = stats.norm.ppf(z1)
        z1 = torch.tensor(z1, device=self.opt.device, dtype=torch.float32)

        # x_test[i1, i2, ..., ik, :] = x1[i], x1[j], ..., x1[k]
        z_test = torch.cat([xv.unsqueeze(-1) for xv in torch.meshgrid([z1] * self.opt.n_latent)], dim=-1)
        z_test = z_test.reshape(-1, self.opt.n_latent)

        real_A_test = self.real_A.expand(z_test.shape[0], *[-1]*(len(self.real_A.shape) - 1))

        self.fake_B_samples = self.decode(z_test, real_A_test)

    def test(self):
        self.diversify()
        super().test()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        self.real_B_rgb = self.lab2rgb(self.real_A, self.real_B)
        self.fake_B_rgb = self.lab2rgb(self.real_A, self.fake_B)

        if not self.is_train:
            for i, fake_B_sample in enumerate(self.fake_B_samples):
                setattr(self, 'fake_B_rgb_sample_{}'.format(i), self.lab2rgb(self.real_A, fake_B_sample[None]))

