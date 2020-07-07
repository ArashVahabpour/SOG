# %%

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
import argparse
import os
from util.visulize_flow import visualize_transform
import math
from skimage import color
from imageio import imread

n_samples = 10000

parser = argparse.ArgumentParser('Continuous Normalizing Flow')

parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings','bezos'],
    type=str, default='pinwheel'
)
parser.add_argument('--save', type=str, default='pinwheel')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1024)
parser.add_argument('--latent_batch_size', type=int, default=1024*4)
parser.add_argument('--niters', type=int, default=100001)
parser.add_argument('--viz_freq', type=int, default=1000)
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--gpu', type=int, default=0)


args = parser.parse_args()

def create_blobs(n_modes):
    # n_modes: number of modes for ground truth noise

    r = 10  # constellation radius
    noise_std = 0  # 0 for a perfect model, 1 for SNR=1, etc.

    modes = np.random.randint(low=0, high=n_modes, size=n_samples)
    angles = (2 * np.pi / n_modes) * modes
    bias = r * np.vstack([np.cos(angles), np.sin(angles)]).T

    noise = np.random.randn(*x.shape) * noise_std

    y = bias + np.random.randn(n_samples, 2) + noise

    return y, modes


# %%

def create_spirals(n_modes):
    noise_std = 0.5

    n = np.sqrt(np.random.rand(n_samples)) * (4 * np.pi)
    d1x = -np.cos(n) * n + np.random.rand(n_samples) * noise_std
    d1y = np.sin(n) * n + np.random.rand(n_samples) * noise_std

    modes = np.random.randint(low=0, high=n_modes, size=n_samples)
    angles = (2 * np.pi / n_modes) * modes

    y = np.hstack(((d1x * np.cos(angles) + d1y * np.sin(angles))[:, None],
                   (-d1x * np.sin(angles) + d1y * np.cos(angles))[:, None]))

    return y, modes



# Dataset iterator
def inf_train_gen(data, rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X, random_state=rng)

        # Add noise
        X = X + rng.normal(scale=0.08, size=X.shape)

        return X.astype("float32")

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data

    elif data == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    elif data == 'bezos':
        path = '/home/arche/PycharmProjects/Feature_Discovery/'
        image= color.rgb2gray(imread(os.path.join(path,'bezos.jpg')))
        image = image[:,75:553]
        image_norm =(1 - image)/(1-image).sum()
        n, m =image_norm.shape
        rand = np.random.choice(n*m, size=batch_size, replace=True, p=np.ndarray.flatten(image_norm))
        row = rand//n
        col = rand%m
        row_pos = row.astype(np.float32)/max(row)
        col_pos = col.astype(np.float32)/max(col)
        return np.stack((col_pos-col_pos.mean(),row_pos-row_pos.mean())).T
    else:
        return inf_train_gen("8gaussians", rng, batch_size)

def get_transforms(model):

    def sample_fn(z, logpz=None):
        if logpz is not None:
            return model(z)
        else:
            return model(z)

    def density_fn(x, logpx=None):
        if logpx is not None:
            return model(x)
        else:
            return model(x)

    return sample_fn, density_fn

def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2
# %%
def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-{}.pth'.format(epoch))
    torch.save(state, filename)

def compute_loss(args, model, batch_size=None):
    if batch_size is None: batch_size = args.batch_size

    # load data
    x = inf_train_gen(args.data, batch_size=batch_size)
    x = torch.from_numpy(x).type(torch.float32).to(device)
    zero = torch.zeros(x.shape[0], 1).to(x)

    # transform to z
    z = model(x)
    del x
    # compute log q(z)
    logpz = standard_normal_logprob(z).sum(1, keepdim=True)

    logpx = logpz
    loss = -torch.mean(logpx)
    del z
    return loss

# y, modes = create_blobs(4); n_epochs = 100  # 4 blobs
# y, modes = create_blobs(360); n_epochs = 100  # ring
## This is not actually batch size its actually sample size
y = inf_train_gen(args.data,batch_size=100000)
# y, modes = create_spirals(2);
n_epochs = 300
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# %%

# plt.scatter(y[:, 0], y[:, 1], c=modes, s=1)
# plt.axis('equal')
# plt.show()

# %%

y = torch.from_numpy(y).float().cuda(device)

# %%

n_latent_gaussian = 2  # size of the latent gaussian variable

batch_size = args.batch_size  # samples of data distribution at each iteration
latent_batch_size = args.latent_batch_size  # number of possible latent codes to try for each sample


# %%

def sample_latent_codes(n_latent_samples):
    gaussian_samples = np.random.randn(n_latent_samples, n_latent_gaussian)

    return torch.from_numpy(gaussian_samples).float()


# %%

# D_in is input dimension;
# H is hidden dimension;
# D_out is output dimension.

D_in, H, D_out = n_latent_gaussian , 100, 2

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

model.cuda(device)
# %%

params = list(model.parameters())

optimizer = torch.optim.Adam(params, betas=(0.5, 0.999))
#optimizer = torch.optim.Adamax(params, lr=args.lr, weight_decay=args.weight_decay)
loss_fn = torch.nn.L1Loss(reduction='none') #torch.nn.MSELoss(reduction='none')

# %%
# Perhaps try without replacement 
# Also add the old visulization to the plots

loss_log = []
best_loss = float('inf')
current_epoch_losses = []
# pwd = os.getcwd()
for itr in range(args.niters):
    rand_idx = np.random.randint(low=0, high=n_samples, size=args.batch_size)

    x_train = sample_latent_codes(batch_size * latent_batch_size)
    # torch.cuda.synchronize()
    x_train = x_train.cuda(device)
    # y = inf_train_gen(args.data,batch_size=batch_size)
    # y = torch.from_numpy(y).float()
    y_train = y[rand_idx].repeat(latent_batch_size, 1)
    y_pred = model(x_train)

    loss_all_modes = loss_fn(y_pred, y_train)

    # Zero the gradients before running the backward pass.
    optimizer.zero_grad()
    selective_loss = loss_all_modes.mean(dim=1).reshape(latent_batch_size, batch_size).min(dim=0)[0].sum()
    selective_loss.backward()

    current_epoch_losses.append(selective_loss)

    optimizer.step()

    # loss_log.append(torch.tensor(current_epoch_losses).mean().detach().cpu().numpy())

    # if itr % args.viz_freq == 0 or itr == args.niters:
    #     x_test = sample_latent_codes(n_samples)
    #     x_test = x_test.cuda(device)
    #     y_pred = model(x_test).detach().cpu().numpy()
    #     # y_pred = y_pred.to(device)
    #     del x_test
    #     fig = plt.figure()
    #     plt.scatter(y_pred[:, 0], y_pred[:, 1], marker='.', s=0.1)
    #     #plt.aixis('equal')
    #     # plt.show()
    #     dir =  os.path.join('results/'+args.save)
    #     if not os.path.exists(dir):
    #         os.mkdir(dir)
    #     fig_filename = os.path.join(dir, '{:04d}.png'.format(itr))
    #     fig.savefig(fig_filename)
    #     plt.close(fig)
    #     del y_pred
#    if itr % args.val_freq == 0 or itr == args.niters:
#        torch.cuda.synchronize()
#        with torch.no_grad():
#            model.eval()
#            test_loss = compute_loss(args, model, batch_size=args.test_batch_size)
#
#            if test_loss.item() < best_loss:
#                best_loss = test_loss.item()
#                dir =  os.path.join('results/'+args.save)
#                if not os.path.exists(dir):
#                    os.mkdir(dir)
#                # torch.save({
#                #     'args': args,
#                #     'state_dict': model.state_dict(),
#                # }, os.path.join(args.save, 'checkpt.pth'))
#                save_checkpoint(model.state_dict(),dir+'/checkpts/',itr)
#            model.train()

    if itr % args.viz_freq == 0 or itr == args.niters:
        torch.cuda.synchronize()
        with torch.no_grad():
            model.eval()
            p_samples = sample_latent_codes(n_samples)
            p_samples = p_samples.cuda(device)
            sample_fn, density_fn = get_transforms(model)

            plt.figure(figsize=(9, 3))
            visualize_transform(
                p_samples, torch.randn, standard_normal_logprob, transform=sample_fn, inverse_transform=None,
                samples=True, npts=200, device=device
            )
            dir =  os.path.join('results/'+args.save)
            if not os.path.exists(dir):
                os.mkdir(dir)
            fig_filename = os.path.join(dir, '{:04d}.jpg'.format(itr))
            plt.savefig(fig_filename)
            plt.close()
            model.train()
            del p_samples
