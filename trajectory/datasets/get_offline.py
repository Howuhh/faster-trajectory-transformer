import os
import glob
import numpy as np
from tqdm import tqdm


def load_paths(data_dir, n_trj, tasks):
    trj_paths = []
    for n in range(n_trj):
        trj_paths += glob.glob(os.path.join(data_dir, "goal_idx*", "trj_evalsample%d_step*.npy" % (n)))
    paths = [
        trj_path
        for trj_path in trj_paths
        if int(trj_path.split("/")[-2].split("goal_idx")[-1]) in tasks
    ]
    task_idxs = [
        int(trj_path.split("/")[-2].split("goal_idx")[-1])
        for trj_path in trj_paths
        if int(trj_path.split("/")[-2].split("goal_idx")[-1]) in tasks
    ]
    return paths, task_idxs


def offline_dataset(data_dir, n_trj, tasks, ratio=1.0):
    paths, _ = load_paths(data_dir, n_trj, tasks)
    if ratio < 1.0:
        indexes = np.random.choice(range(len(paths)), int(len(paths) * ratio), replace=False).tolist()
        paths = [paths[index] for index in indexes]
    obs_, action_, reward_ = [], [], []
    done_, real_done_ = [], []

    for path in tqdm(paths):
        dataset = np.load(path, allow_pickle=True)
        for idx, (obs, action, reward, next_obs) in enumerate(dataset):
            real_done = 1 if idx == len(dataset) - 1 else 0
            done = real_done

            obs_.append(obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done)
            real_done_.append(real_done)

    return {
        "states": np.array(obs_),
        "actions": np.array(action_),
        "rewards": np.array(reward_)[:, None],
        "dones": np.array(done_)[:, None],
        "real_dones": np.array(real_done_)[:, None],
    }


if __name__ == "__main__":
    path = "/home/changyeon/CaDM_MerPO/data/walker-rand-param"
    data = offline_dataset(path, 50, range(5), range(5, 10))

    print(data["real_dones"].sum())
