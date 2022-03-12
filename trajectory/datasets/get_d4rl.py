import os
import d4rl
import gym
import numpy as np
from tqdm.auto import trange


def d4rl_dataset(env):
    dataset = env.get_dataset()
    N = dataset['rewards'].shape[0]

    obs_, action_, reward_ = [], [], []
    done_, real_done_ = [], []

    for i in trange(N, desc=f"Loading {env.spec.id} dataset"):
        obs = dataset['observations'][i]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        # dones
        real_done = bool(dataset["terminals"][i])
        done = bool(dataset["timeouts"][i]) or real_done

        obs_.append(obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done)
        real_done_.append(real_done)

    return {
        'states': np.array(obs_),
        'actions': np.array(action_),
        'rewards': np.array(reward_)[:, None],
        'dones': np.array(done_)[:, None],
        'real_dones': np.array(real_done_)[:, None]
    }


if __name__ == "__main__":
    DATASETS = [
        # halfcheetah
        "halfcheetah-medium-v2",
        "halfcheetah-expert-v2",
        "halfcheetah-medium-replay-v2",
        # hopper
        "hopper-medium-v2",
        "hopper-expert-v2",
        "hopper-medium-replay-v2",
        # walker
        "walker2d-medium-v2",
        "walker2d-expert-v2",
        "walker2d-medium-replay-v2",
    ]
    from trajectory.utils.env import create_env
    env = create_env(DATASETS[3])
    data = d4rl_dataset(env)

    print(data["real_terminals"].sum())