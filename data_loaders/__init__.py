import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_olivetti_faces
from .tabular import POWER, GAS, HEPMASS, MINIBOONE, BSDS300
import os


def create_data_loader(opt):
    switcher = {
        'mnist': _mnist,
        'emnist': _emnist,
        'fashion-mnist': _fashion_mnist,
        'olivetti-faces': _olivetti_faces,
        'celeba': _celeba,
        'power': _tabular,
        'gas': _tabular,
        'hepmass': _tabular,
        'miniboone': _tabular,
        'bsds300': _tabular,
        'gym': _gym,
    }
    func = switcher.get(opt.dataset, None)

    if func is None:
        raise NotImplementedError('Built-in dataset {} is not implemented.'.format(opt.dataset))

    return func(opt)


def _mnist(opt):
    return DataLoader(
        datasets.MNIST(opt.dataroot, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=opt.batch_size, shuffle=True)


def _emnist(opt):
    # """ TODO DEBUG REMOVE
    # Dilate thin characters since they are hard to learn with l1 loss, etc.
    # """
    # def dilate_if_thin(image):
    #     n_white = np.sum(image) / 255.  # number of bright pixels after opening
    #     image_opened = image.filter(MinFilter(3)).filter(MaxFilter(3))
    #     n_darkened = np.abs(np.array(image) - np.array(image_opened)).sum() / 255.  # number of bright pixels darkened after opening
    #
    #     return image if n_darkened / n_white < 0.25 else image.filter(MaxFilter())

    return DataLoader(
        datasets.EMNIST(opt.dataroot, split='balanced', train=True, download=True,
                        transform=transforms.Compose([
                            # transforms.Lambda(lambda x: dilate_if_thin(x)), TODO DEBUG REMOVE
                            transforms.ToTensor(),
                            transforms.Lambda(lambda data: data.transpose(1, 2))
                            # transforms.Normalize((0.1751,), (0.3332,))
                        ])),
        batch_size=opt.batch_size, shuffle=True)

    # TODO: require up-to-date pytorch
    # TODO: add normalization and last layer activation as an option >> --normalize_data already added to base_options


def _fashion_mnist(opt):
    return DataLoader(
        datasets.FashionMNIST(opt.dataroot, train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Lambda(lambda data: 1 - data)  # invert
                                  # transforms.Normalize((0.2860,), (0.3530,))
                                  ])),
        batch_size=opt.batch_size, shuffle=True)


def _olivetti_faces(opt):
    # Load the faces datasets
    data = fetch_olivetti_faces()
    tensor_y = torch.tensor(data.images).unsqueeze(1)  # transform to torch tensor

    # dataset_mean, dataset_std = 0.5470, 0.1725
    # tensor_y = (tensor_y - dataset_mean) / dataset_std  # (tensor_y - tensor_y.mean()) / tensor_y.std()

    dataset = TensorDataset(tensor_y)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    return data_loader


def _celeba(opt):
    """
    Data loader fo Celeb-A Faces dataset which can be downloaded at its website:
        - http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    or in Google Drive:
        - https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg
    The dataset will download as a file named img_align_celeba.zip.
    Once downloaded, create a directory named celeba and extract the zip file into dataroot directory (by default ./datasets).
    The resulting directory structure should be:

    dataroot/celeba
        -> img_align_celeba
            -> 188242.jpg
            -> 173822.jpg
            -> 284702.jpg
            -> 537394.jpg
               ...

    This is an important step because we will be using the ImageFolder dataset class, which requires there to be subdirectories in the dataset’s root folder.
    """

    dataset = datasets.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(opt.img_size),
                                       transforms.CenterCrop(opt.img_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)  # todo, num_workers=workers

    return data_loader


def _tabular(opt):
    switcher = {
        'power': POWER,
        'gas': GAS,
        'hepmass': HEPMASS,
        'miniboone': MINIBOONE,
        'bsds300': BSDS300,
    }
    dataset_class = switcher.get(opt.dataset)(opt)

    data_numpy = dataset_class.trn if opt.is_train else dataset_class.tst  # elif val: dataset_numpy.val
    tensor_y = torch.tensor(data_numpy.x)
    opt.n_features = int(tensor_y.shape[1])  # number of features is used in defining the architecture of `FlatMLP` networks. see `networks.py`
    dataset = TensorDataset(tensor_y, torch.empty([tensor_y.shape[0], 0]))
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    return data_loader


def _gym(opt):
    filename = 'trajs_{}.pt'.format(opt.env_name.split('-')[0].lower())
    data_dir = os.path.join(opt.dataroot, filename)

    if opt.gen_expert or not os.path.exists(data_dir):
        from util.gym import Expert
        Expert(opt).generate_data()

    data_dict = torch.load(data_dir)

    # num_traj x traj_len x dim
    tensor_x = data_dict['states']
    tensor_y = data_dict['actions']

    # (num_traj * traj_len) x dim
    tensor_x = tensor_x.reshape(-1, tensor_x.shape[-1])
    tensor_y = tensor_y.reshape(-1, tensor_y.shape[-1])

    dataset = TensorDataset(tensor_y, tensor_x)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    opt.state_dim = int(tensor_x.shape[-1])
    opt.action_dim = int(tensor_y.shape[-1])

    return data_loader
