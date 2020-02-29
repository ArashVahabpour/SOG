import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_olivetti_faces


def create_data_loader(opt):
    switcher = {
        'mnist': _mnist,
        'emnist': _emnist,
        'fashion-mnist': _fashion_mnist,
        'olivetti-faces': _olivetti_faces,
    }
    func = switcher.get(opt.dataset, None)

    if func is None:
        raise NotImplementedError('dataset {} not implemented!'.format(opt.dataset))

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
    return DataLoader(
        datasets.EMNIST(opt.dataroot, split='balanced', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda data: data.transpose(1, 2))
                            # transforms.Normalize((0.1751,), (0.3332,))
                        ])),
        batch_size=opt.batch_size, shuffle=True)

    # TODO: require up-to-date pytorch
    # TODO: add normalization and last layer activation as an option


def _fashion_mnist(opt):
    return DataLoader(
        datasets.FashionMNIST(opt.dataroot, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Normalize((0.2860,), (0.3530,))
                        ])),
        batch_size=opt.batch_size, shuffle=True)


def _olivetti_faces(opt):
    # Load the faces datasets
    data = fetch_olivetti_faces()
    tensor_x = torch.tensor(data.images).unsqueeze(1)  # transform to torch tensor

    # dataset_mean, dataset_std = 0.5470, 0.1725
    # tensor_x = (tensor_x - dataset_mean) / dataset_std  # (tensor_x - tensor_x.mean()) / tensor_x.std()

    dataset = TensorDataset(tensor_x)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    return data_loader
