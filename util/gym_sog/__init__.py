from gym.envs.registration import register


register(id='Circles-v0',
         entry_point='util.gym_sog.envs:CirclesEnv',
         )
