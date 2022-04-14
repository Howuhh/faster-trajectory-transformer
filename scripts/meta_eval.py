import os
import torch
import argparse
import numpy as np

from tqdm.auto import tqdm, trange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from trajectory.datasets.offline_dataset import DiscretizedOfflineDataset

from trajectory.models.gpt import GPT
from trajectory.models.gpt.gpt_trainer import GPTTrainer
from trajectory.utils.common import set_seed
from trajectory.utils.env import create_env, create_meta_env, rollout, vec_rollout


def create_argparser():
    parser = argparse.ArgumentParser(description="Trajectory Transformer evaluation hyperparameters. All can be set from command line.")
    parser.add_argument("--config", default="configs/medium/halfcheetah_medium.yaml")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cpu", type=str)

    return parser


def finetune_model(model, task_idx, config, device, seed=42, finetune_ratio=0.1, num_epochs=10):
    finetune_path = os.path.join(config.trainer.checkpoints_path, "finetune", str(task_idx))
    if os.path.exists(finetune_path):
        model.load_state_dict(torch.load(os.path.join(finetune_path, "model_last.pt"), map_location=device))
    else:
        # finetune the trained model for target dataset.
        set_seed(seed=seed)

        trainer_conf = config.trainer
        data_conf = config.dataset

        dataset = DiscretizedOfflineDataset(
            env_name=data_conf.env_name,
            data_dir=data_conf.data_dir,
            n_trj=data_conf.n_trj,
            tasks=[task_idx],
            ratio=finetune_ratio,
            seq_len=data_conf.seq_len,
            cache_path=data_conf.cache_path,
            num_bins=data_conf.num_bins,
            discount=data_conf.discount,
            strategy=data_conf.strategy,
        )
        # data_length = int(len(dataset) * finetune_ratio)
        # train_idx, val_idx = torch.utils.data.train_test_split(data_length, test_size=finetune_ratio)
        # _, finetune_dataset = torch.utils.data.random_split(dataset, [len(dataset) - data_length, data_length])
        dataloader = DataLoader(dataset, batch_size=data_conf.batch_size, shuffle=True, num_workers=8, pin_memory=True)

        config.wandb.mode = "offline"
        wandb.init(
            **config.wandb,
            config=dict(OmegaConf.to_container(config, resolve=True))
        )
        model = GPT(**config.model)
        model.to(device)

        warmup_tokens = len(dataset) * data_conf.seq_len * config.model.transition_dim
        final_tokens = warmup_tokens * num_epochs

        trainer = GPTTrainer(
            final_tokens=final_tokens,
            warmup_tokens=warmup_tokens,
            action_weight=trainer_conf.action_weight,
            value_weight=trainer_conf.value_weight,
            reward_weight=trainer_conf.reward_weight,
            learning_rate=trainer_conf.lr,
            betas=trainer_conf.betas,
            weight_decay=trainer_conf.weight_decay,
            clip_grad=trainer_conf.clip_grad,
            eval_seed=trainer_conf.eval_seed,
            eval_every=1e10,
            eval_episodes=trainer_conf.eval_episodes,
            eval_temperature=trainer_conf.eval_temperature,
            eval_discount=trainer_conf.eval_discount,
            eval_plan_every=trainer_conf.eval_plan_every,
            eval_beam_width=trainer_conf.eval_beam_width,
            eval_beam_steps=trainer_conf.eval_beam_steps,
            eval_beam_context=trainer_conf.eval_beam_context,
            eval_sample_expand=trainer_conf.eval_sample_expand,
            eval_k_obs=trainer_conf.eval_k_obs,  # as in original implementation
            eval_k_reward=trainer_conf.eval_k_reward,
            eval_k_act=trainer_conf.eval_k_act,
            eval_data_dir=data_conf.data_dir,
            eval_tasks=data_conf.eval_tasks,
            eval_n_trj=data_conf.n_trj,
            eval_max_steps=trainer_conf.eval_max_steps,
            checkpoints_path=finetune_path,
            save_every=num_epochs,
            device=device
        )
        trainer.train(
            model=model,
            dataloader=dataloader,
            num_epochs=num_epochs,
            log_every = 1e10
        )
        # finetune with data


def run_experiment(config, seed, device):
    set_seed(seed=seed)

    run_config = OmegaConf.load(os.path.join(config.checkpoints_path, "config.yaml"))
    discretizer = torch.load(os.path.join(config.checkpoints_path, "discretizer.pt"), map_location=device)

    model = GPT(**run_config.model)
    model.eval()
    model.to(device)
    # model.load_state_dict(torch.load(os.path.join(config.checkpoints_path, config.model_name), map_location=device))
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
            model.load_state_dict(torch.load(os.path.join(config.checkpoints_path, config.model_name), map_location=device))
            print(f"task idx: {i}")
            if config.finetune:
                print(f"finetune on task idx {i}")
                finetune_model(
                    model,
                    task_idx=i,
                    config=run_config,
                    device=device,
                    seed=seed,
                    finetune_ratio=config.finetune_ratio,
                    num_epochs=config.finetune_epochs
                )
            model.eval()
            task_rewards = []
            env.reset_task(i, verbose=True)
            for _ in trange(config.num_episodes, desc=f"Evaluation on task {i}"):
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
                task_rewards.append(reward)
            print(f"AvgReturn on task {i}: {np.mean(task_rewards)} ± {np.std(task_rewards)}")
            

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