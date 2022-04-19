"""
Simple environment with known optimal policy and value function.

This environment has just two actions.
Action 0 yields randomly 0 or 5 reward and then terminates the session.
Action 1 yields randomly 1 or 3 reward and then terminates the session.

Optimal policy: action 0.

Optimal value function: v(0)=2.5 (there is only one state, state 0)
"""

from rand_param_envs import gym
from rand_param_envs.gym import spaces
from rand_param_envs.gym.utils import seeding

class OneRoundNondeterministicRewardEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(1)
        self._seed()
        self._reset()

    def _step(self, action):
        assert self.action_space.contains(action)
        if action:
            #your agent should figure out that this option has expected value 2.5
            reward = self.np_random.choice([0, 5])
        else:
            #your agent should figure out that this option has expected value 2.0
            reward = self.np_random.choice([1, 3])

        done = True
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return 0

    def _reset(self):
        return self._get_obs()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
