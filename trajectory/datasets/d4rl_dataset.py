import os
import torch
import numpy as np

import pickle
from tqdm.auto import trange, tqdm
from torch.utils.data import Dataset

from trajectory.datasets.get_d4rl import d4rl_dataset
from trajectory.utils.common import pad_along_axis
from trajectory.utils.discretization import KBinsDiscretizer
from trajectory.utils.env import create_env


def join_trajectory(states, actions, rewards, discount=0.99):
    traj_length = states.shape[0]
    # I can vectorize this for all dataset as once,
    # but better to be safe and do it once and slow and right (and cache it)
    discounts = (discount ** np.arange(traj_length))

    values = np.zeros_like(rewards)
    for t in range(traj_length):
        # discounted return-to-go from state s_t:
        # r_{t+1} + y * r_{t+2} + y^2 * r_{t+3} + ...
        # .T as rewards of shape [len, 1], see https://github.com/Howuhh/faster-trajectory-transformer/issues/9
        values[t] = (rewards[t + 1:].T * discounts[:-t - 1]).sum()

    joined_transition = np.concatenate([states, actions, rewards, values], axis=-1)

    return joined_transition


def segment(states, actions, rewards, terminals):
    assert len(states) == len(terminals)
    trajectories = {}

    episode_num = 0
    for t in trange(len(terminals), desc="Segmenting"):
        if episode_num not in trajectories:
            trajectories[episode_num] = {
                "states": [],
                "actions": [],
                "rewards": []
            }
        
        trajectories[episode_num]["states"].append(states[t])
        trajectories[episode_num]["actions"].append(actions[t])
        trajectories[episode_num]["rewards"].append(rewards[t])

        if terminals[t].item():
            # next episode
            episode_num = episode_num + 1

    trajectories_lens = [len(v["states"]) for k, v in trajectories.items()]

    for t in trajectories:
        trajectories[t]["states"] = np.stack(trajectories[t]["states"], axis=0)
        trajectories[t]["actions"] = np.stack(trajectories[t]["actions"], axis=0)
        trajectories[t]["rewards"] = np.stack(trajectories[t]["rewards"], axis=0)

    return trajectories, trajectories_lens


# adapted from https://github.com/jannerm/trajectory-transformer/blob/master/trajectory/datasets/sequence.py
class DiscretizedDataset(Dataset):
    def __init__(self, env_name, num_bins=100, seq_len=10, discount=0.99, strategy="uniform", cache_path=None):
        self.seq_len = seq_len
        self.discount = discount
        self.num_bins = num_bins
        self.env = create_env(env_name)

        dataset = d4rl_dataset(self.env)
        trajectories, traj_lengths = segment(
            dataset["states"],
            dataset["actions"],
            dataset["rewards"],
            dataset["dones"]
        )
        self.cache_path = cache_path
        self.cache_name = f"{env_name}_{num_bins}_{seq_len}_{strategy}_{discount}"

        if cache_path is None or not os.path.exists(os.path.join(cache_path, self.cache_name)):
            self.joined_transitions = []
            for t in tqdm(trajectories, desc="Joining transitions"):
                self.joined_transitions.append(
                    join_trajectory(trajectories[t]["states"], trajectories[t]["actions"], trajectories[t]["rewards"])
                )

            os.makedirs(os.path.join(cache_path), exist_ok=True)
            # save cached version
            with open(os.path.join(cache_path, self.cache_name), "wb") as f:
                pickle.dump(self.joined_transitions, f)
        else:
            with open(os.path.join(cache_path, self.cache_name), "rb") as f:
                self.joined_transitions = pickle.load(f)

        self.discretizer = KBinsDiscretizer(
            np.concatenate(self.joined_transitions, axis=0),
            num_bins=num_bins,
            strategy=strategy
        )

        # get valid indices for seq_len sampling
        indices = []
        for path_ind, length in enumerate(traj_lengths):
            end = length - 1
            for i in range(end):
                indices.append((path_ind, i, i + self.seq_len))
        self.indices = np.array(indices)

    def get_env_name(self):
        return self.env.name

    def get_discretizer(self):
        return self.discretizer

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start_idx, end_idx = self.indices[idx]
        joined = self.joined_transitions[traj_idx][start_idx:end_idx]

        loss_pad_mask = np.ones((self.seq_len, joined.shape[-1]))
        if joined.shape[0] < self.seq_len:
            # pad to seq_len if at the end of trajectory, mask for padding
            loss_pad_mask[joined.shape[0]:] = 0
            joined = pad_along_axis(joined, pad_to=self.seq_len, axis=0)

        joined_discrete = self.discretizer.encode(joined).reshape(-1).astype(np.long)
        loss_pad_mask = loss_pad_mask.reshape(-1)

        return joined_discrete[:-1], joined_discrete[1:], loss_pad_mask[:-1]
