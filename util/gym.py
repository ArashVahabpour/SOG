import gym
from util import gym_sog
import os
import numpy as np
import scipy.linalg as LA
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from torch.distributions.normal import Normal


class Expert:
    def __init__(self, opt):
        self.opt = opt
        self.env_name = opt.env_name

    def generate_data(self):
        print('generating expert trajectories...')
        if self.env_name == 'Circles-v0':
            data_dict = self._generate_circle_data()
        else:
            raise NotImplementedError('expert for environment {} not implemented.'.format(self.env_name))

        self._save_data(data_dict)

    def _generate_circle_data(self):
        env = gym.make(self.opt.env_name, opt=self.opt, state_len=5)

        num_traj = 500  # number of trajectories
        traj_len = 1000  # length of each trajectory --- WARNING: DO NOT CHANGE THIS TO VALUES LOWER THAN 1000 OR IT CAN CAUSE ISSUES IN GAIL RUN
        expert_data = {'states': [],
                       'actions': [],
                       'radii': [],
                       'lengths': torch.tensor([traj_len] * num_traj, dtype=torch.int32)}

        max_ac_mag = self.opt.max_ac_mag  # max action magnitude

        for traj_id in range(num_traj):
            print('traj #{}'.format(traj_id + 1))

            observation = env.reset()
            step = 0
            states = []
            actions = []
            while step < traj_len:
                if self.opt.render_gym:
                    env.render()

                radius = env.radius

                ########## compute speed vector ##########
                delta_theta = 2 * np.pi / 100
                start = env.state[-2:]
                center = np.array([0, radius])
                rot_mat_T = np.array([
                    [np.cos(delta_theta), -np.sin(delta_theta)],
                    [np.sin(delta_theta), np.cos(delta_theta)]
                ]).T
                radial_dist = (start - center).dot(rot_mat_T)
                circ_dest = radial_dist + center
                circ_speed = circ_dest - start
                length = LA.norm(radial_dist)
                speed = circ_speed - (radial_dist / length) * (length - abs(radius))

                # clip action to fit inside its box
                ac_mag = LA.norm(speed, np.inf)
                if ac_mag > max_ac_mag:
                    speed = speed / ac_mag * max_ac_mag

                action = speed
                ##########################################

                states.append(observation)
                observation, reward, done, info = env.step(action)
                actions.append(action)

                step += 1

                if done:
                    # start over a new trajectory hoping that this time it completes
                    observation = env.reset()
                    step = 0
                    states = []
                    actions = []
                    print('warning: an incomplete trajectory occured.')

            expert_data['states'].append(torch.FloatTensor(np.array(states)))
            expert_data['actions'].append(torch.FloatTensor(np.array(actions)))
            expert_data['radii'].append(radius)

        env.close()

        expert_data['states'] = torch.stack(expert_data['states'])
        expert_data['actions'] = torch.stack(expert_data['actions'])
        expert_data['radii'] = torch.tensor(expert_data['radii'])

        return expert_data

    def _save_data(self, data_dict):
        data_dir = self.opt.dataroot
        os.makedirs(data_dir, exist_ok=True)
        filename = 'trajs_{}.pt'.format(self.opt.env_name.split('-')[0].lower())
        torch.save(data_dict, os.path.join(data_dir, filename))
        print('expert data saved successfully.')


def test_env(sog_model):
    opt = sog_model.opt
    env = gym.make(opt.env_name, opt=opt, state_len=5)
    traj_len = 1000  # length of each trajectory

    # TODO if  pass; elif...
    if opt.latent_optimizer == 'ohs':
        num_traj = 10  # number of trajectories
        all_modes = torch.eye(opt.n_latent, device=opt.device)
        continuous = False
    elif opt.latent_optimizer == 'bcs':
        num_traj = 20  # number of trajectories
        lower_cdf, upper_cdf = 0.21, 0.69  # choose 0 < lower_cdf < upper_cdf < 1 for selection of latent codes
        m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        all_modes = m.icdf(torch.linspace(lower_cdf, upper_cdf, num_traj, device=opt.device))[:, None]
        continuous = True
    else:
        raise NotImplementedError

    imitated_data = {'states': [], 'actions': []}
    for traj_id in range(num_traj):
        # selecting a random one-hot code; unsequeeze the batch dimension.
        mode_idx = np.array([traj_id]) if continuous else np.random.randint(0, opt.n_latent, 1)
        mode = all_modes[mode_idx].unsqueeze(0)

        print('traj #{}, latent code: {}'.format(traj_id + 1, mode if continuous else mode_idx[0]))

        obs = env.reset()
        step = 0
        states = []
        actions = []
        while step < traj_len:
            if opt.render_gym:
                env.render()

            obs_tensor = torch.tensor(obs, device=opt.device, dtype=torch.float32).unsqueeze(0)
            action = sog_model.decode(mode, obs_tensor, requires_grad=False).squeeze().cpu().numpy()

            states.append(obs)
            obs, reward, done, info = env.step(action)
            actions.append(action)

            step += 1

            if done:
                print('warning: an incomplete trajectory occurred.')
                break

            # TODO record rewards
        imitated_data['states'].append(np.array(states))
        imitated_data['actions'].append(np.array(actions))

    env.close()

    save_dir = os.path.join(opt.results_dir, opt.name, 'test_{}'.format(opt.which_epoch))
    os.makedirs(save_dir, exist_ok=True)
    torch.save(imitated_data, os.path.join(save_dir, 'trajs_circles.pt'))

    def visualize_trajectories(states, max_r):
        num_traj = len(states)
        cm = plt.get_cmap('gist_rainbow')
        color_list = [cm(1. * i / num_traj) for i in range(num_traj)]
        from cycler import cycler
        custom_cycler = cycler(color=color_list)

        # state visualization
        fig, axs = plt.subplots(ncols=1, figsize=(10, 20))
        axs.set_aspect('equal', 'box')
        axs.set_prop_cycle(custom_cycler)

        for i, traj in enumerate(states):
            if continuous:
                axs.plot(traj[:, -2], traj[:, -1])
            else:
                axs.plot(traj[:, -2], traj[:, -1], "*", label=str(i))

        if continuous:  # plot the max radius circles
            for shift in [-max_r, max_r]:
                axs.plot(max_r * np.cos(np.linspace(0, 4 * np.pi, 500)),
                         max_r * np.sin(np.linspace(0, 4 * np.pi, 500)) + shift,
                         color='000000')

        if not continuous:
            plt.legend()
            plt.tight_layout()
        else:
            plt.axis('equal')
            plt.xlim([-max_r * 1.5, max_r * 1.5])
            plt.ylim([-max_r * 3, max_r * 3])
            plt.axis('off')
        plt.title('SOG')

    visualize_trajectories(imitated_data['states'], max(map(abs, opt.radii)))
    plt.show()
    plt.savefig(os.path.join(save_dir, 'trajs.png'))
    print('imitated results saved successfully.')


def test_env_interactive(sog_model):
    opt = sog_model.opt
    env = gym.make(opt.env_name, opt=opt, state_len=5)
    traj_len = 1000  # length of each trajectory

    if not opt.latent_optimizer == 'bcs':
        raise NotImplementedError('interactive demo is only implemented for the case with continuous latent code.')

    def get_traj(cdf):
        m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        z = m.icdf(torch.tensor([[cdf]], dtype=torch.float32, device=opt.device))

        obs = env.reset()
        states = []

        for _ in range(traj_len):
            obs_tensor = torch.tensor(obs, device=opt.device, dtype=torch.float32).unsqueeze(0)
            action = sog_model.decode(z, obs_tensor, requires_grad=False).squeeze().cpu().numpy()

            states.append(obs)
            obs, reward, done, info = env.step(action)

            if done:
                print('warning: an incomplete trajectory occurred.')
                break
        states = np.array(states)
        x, y = states[:, 0], states[:, 1]
        return x, y

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    cdf0 = 0.5
    x, y = get_traj(cdf0)
    l, = plt.plot(x, y, color='blue')

    max_r = max(map(abs, opt.radii))
    for shift in [-max_r, max_r]:
        plt.plot(max_r * np.cos(np.linspace(0, 4 * np.pi, 500)),
                 max_r * np.sin(np.linspace(0, 4 * np.pi, 500)) + shift,
                 color='000000')

    plt.axis('equal')
    plt.axis([-max_r * 1.5, max_r * 1.5, -max_r * 3, max_r * 3])
    plt.axis('off')

    axcolor = 'lightgoldenrodyellow'
    axcdf = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

    scdf = Slider(axcdf, 'Latent Code', 0.01, 0.99, valinit=cdf0)

    def update(val):
        cdf = scdf.val
        x, y = get_traj(cdf)
        l.set_xdata(x)
        l.set_ydata(y)
        fig.canvas.draw_idle()

    scdf.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        scdf.reset()

    button.on_clicked(reset)

    plt.show()

