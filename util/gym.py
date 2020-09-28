import gym
import gym_sog  # TODO give instruction to install this
import numpy as np
import torch


def control_env(opt, sog_model):
    state_len = 5
    env = gym.make(opt.env_name, radii=opt.radii, state_len=5)

    num_traj = 10  # number of trajectories
    traj_len = 1000  # length of each trajectory

    max_ac_mag = env.max_ac_mag  # max action magnitude

    for traj_id in range(num_traj):
        print('traj #{}'.format(traj_id + 1))
        done = False

        observation = env.reset()
        step = 0
        states = []
        actions = []
        while step < traj_len:
            env.render()  # uncomment for visualisation purposes

            radius = env.radius

            action = sog_model

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

            # TODO record rewards
    env.close()

    torch.save(expert_data, 'trajs_circles.pt')
    print('expert data saved successfully.')
