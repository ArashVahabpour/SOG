import gym
from util import gym_sog
import os
import numpy as np
import scipy.linalg as LA
import torch
import matplotlib.pyplot as plt


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

        radii = self.opt.radii

        env = gym.make(self.opt.env_name, radii=radii, state_len=5)

        num_traj = 500  # number of trajectories
        traj_len = 1000  # length of each trajectory --- WARNING: DO NOT CHANGE THIS TO VALUES LOWER THAN 1000 OR IT CAN CAUSE ISSUES IN GAIL RUN
        expert_data = {'states': [],
                       'actions': [],
                       'radii': [],
                       'lengths': torch.tensor([traj_len] * num_traj, dtype=torch.int32)}

        max_ac_mag = env.max_ac_mag  # max action magnitude

        for traj_id in range(num_traj):
            print('traj #{}'.format(traj_id + 1))

            observation = env.reset()
            step = 0
            states = []
            actions = []
            while step < traj_len:
                # env.render()  # uncomment for visualisation / debugging

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
                if LA.norm(speed) > max_ac_mag:
                    speed = speed / LA.norm(speed) * max_ac_mag

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

    env = gym.make(opt.env_name, radii=opt.radii, state_len=5)

    num_traj = 10  # number of trajectories
    traj_len = 1000  # length of each trajectory

    # TODO if opt.latent_optimizer == 'bcs': pass; elif...
    if opt.latent_optimizer == 'ohs':
        all_modes = torch.eye(opt.n_latent, device=opt.device)
    else:
        raise NotImplementedError('creating trajectories not implemented for continuous latent code.')

    max_ac_mag = env.max_ac_mag  # max action magnitude

    imitated_data = {'states': [], 'actions': []}
    for traj_id in range(num_traj):
        # selecting a random one-hot code; unsequeeze the batch dimension.
        mode_idx = np.random.randint(0, opt.n_latent, 1)
        mode = all_modes[mode_idx].unsqueeze(0)

        print('traj #{}, latent code: {}'.format(traj_id + 1, mode_idx[0]))

        obs = env.reset()
        step = 0
        states = []
        actions = []
        while step < traj_len:
            env.render()  # uncomment for visualization / debugging

            obs_tensor = torch.tensor(obs, device=opt.device, dtype=torch.float32).unsqueeze(0)

            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            action = sog_model.decode(mode, obs_tensor, requires_grad=False).squeeze().cpu().numpy()

            if LA.norm(action) > max_ac_mag:
                action = action / LA.norm(action) * max_ac_mag

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
    visualize_trajectories(imitated_data['states'], )
    plt.savefig(os.path.join(save_dir, 'trajs.png'))
    print('imitated results saved successfully.')


def visualize_trajectories(states):
    from cycler import cycler
    NUM_COLORS = len(states)  # number of trajectories
    cm = plt.get_cmap('gist_rainbow')
    color_list = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    custom_cycler = cycler(color=color_list)

    # state visualization
    fig, axs = plt.subplots(ncols=1, figsize=(10, 20))
    axs.set_aspect('equal', 'box')
    axs.set_prop_cycle(custom_cycler)
    for i, traj in enumerate(states):
        axs.plot(traj[:, -2], traj[:, -1], "*", label=str(i))

    plt.legend()
    plt.tight_layout()
    plt.title('SOG')
