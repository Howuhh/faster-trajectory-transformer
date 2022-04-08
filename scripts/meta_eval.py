import os
import torch
import argparse
import numpy as np

from tqdm.auto import tqdm, trange
from omegaconf import OmegaConf

from stable_baselines3.common.vec_env import DummyVecEnv

from trajectory.models.gpt import GPT
from trajectory.utils.common import set_seed
from trajectory.utils.env import create_env, create_meta_env, rollout, vec_rollout


def create_argparser():
    parser = argparse.ArgumentParser(description="Trajectory Transformer evaluation hyperparameters. All can be set from command line.")
    parser.add_argument("--config", default="configs/medium/halfcheetah_medium.yaml")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cpu", type=str)

    return parser


def run_experiment(config, seed, device):
    set_seed(seed=seed)

    run_config = OmegaConf.load(os.path.join(config.checkpoints_path, "config.yaml"))
    discretizer = torch.load(os.path.join(config.checkpoints_path, "discretizer.pt"), map_location=device)

    model = GPT(**run_config.model)
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(config.checkpoints_path, config.model_name), map_location=device))
    eval_tasks = range(run_config.dataset.n_trj - run_config.dataset.eval_tasks, run_config.dataset.n_trj)

    if config.vectorized:
        env = DummyVecEnv([lambda: create_meta_env(data_dir=run_config.dataset.data_dir, n_tasks=run_config.dataset.n_trj) for _ in range(run_config.dataset.eval_tasks * config.num_episodes)])
        env_indices = np.repeat(eval_tasks, config.num_episodes)
        for env_idx, task_idx in enumerate(env_indices):
            env.envs[env_idx].reset_task(task_idx)

        rewards = vec_rollout(
            vec_env=env,
            model=model,
            discretizer=discretizer,
            beam_context_size=config.beam_context,
            beam_width=config.beam_width,
            beam_steps=config.beam_steps,
            plan_every=config.plan_every,
            sample_expand=config.sample_expand,
            k_act=config.k_act,
            k_obs=config.k_obs,
            k_reward=config.k_reward,
            temperature=config.temperature,
            discount=config.discount,
            max_steps=env.envs[0].max_episode_steps,
            device=device
        )
    else:
        rewards = []

        env = create_meta_env(data_dir=run_config.dataset.data_dir, n_tasks=run_config.dataset.n_trj)
        for i in tqdm(eval_tasks, desc="Evaluation (not vectorized)"):
            env.reset_task(i)
            for j in trange(config.num_episodes, desc=f"Evaluation on task {i}"):
                reward = rollout(
                    env=env,
                    model=model,
                    discretizer=discretizer,
                    beam_context_size=config.beam_context,
                    beam_width=config.beam_width,
                    beam_steps=config.beam_steps,
                    plan_every=config.plan_every,
                    sample_expand=config.sample_expand,
                    k_act=config.k_act,
                    k_obs=config.k_obs,
                    k_reward=config.k_reward,
                    temperature=config.temperature,
                    discount=config.discount,
                    max_steps=config.get("max_steps", None) or env.max_episode_steps,
                    render_path=os.path.join(config.render_path, str(i)),
                    device=device
                )
                rewards.append(reward)

    # make sync with MerPO
    # In MerPO, there's more complicated logics (guarantee num_steps_per_eval (600)), but I removed it.
    # It's more robust setup for getting rewards.
    final_returns = [elem for idx, elem in enumerate(rewards) if idx % config.num_episodes == (config.num_episodes - 1)]
    final_returns_mean, final_returns_std = np.mean(final_returns), np.std(final_returns)
    reward_mean, reward_std = np.mean(rewards), np.std(rewards)

    print(f"Evalution on {run_config.dataset.env_name}")
    print(f"AvgReturn_all_test_tasks: {final_returns_mean} ± {final_returns_std}")
    print(f"AvgReturn_online_test_tasks: {reward_mean} ± {reward_std}")


def main():
    args, override = create_argparser().parse_known_args()
    config = OmegaConf.merge(
        OmegaConf.load(args.config),
        OmegaConf.from_cli(override)
    )
    run_experiment(
        config=config,
        seed=args.seed,
        device=args.device
    )


if __name__ == "__main__":
    main()