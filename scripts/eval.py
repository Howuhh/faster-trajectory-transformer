import os
import torch
import argparse
import numpy as np

from tqdm.auto import trange
from omegaconf import OmegaConf

from stable_baselines3.common.vec_env import DummyVecEnv

from trajectory.models.gpt import GPT
from trajectory.utils.common import set_seed
from trajectory.utils.env import create_env, rollout, vec_rollout


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

    if config.vectorized:
        env = DummyVecEnv([lambda: create_env(run_config.dataset.env_name) for _ in range(config.num_episodes)])

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
        scores = [env.envs[0].get_normalized_score(r) for r in rewards]
    else:
        rewards, scores = [], []

        env = create_env(run_config.dataset.env_name)
        for i in trange(config.num_episodes, desc="Evaluation (not vectorized)"):
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
            scores.append(env.get_normalized_score(reward))

    reward_mean, reward_std = np.mean(rewards), np.std(rewards)
    score_mean, score_std = np.mean(scores), np.std(scores)

    print(f"Evalution on {run_config.dataset.env_name}")
    print(f"Mean reward: {reward_mean} ± {reward_std}")
    print(f"Mean score: {score_mean} ± {score_std}")


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