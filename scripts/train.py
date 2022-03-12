import os
import wandb
import argparse

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from trajectory.models.gpt import GPT, GPTTrainer
from trajectory.datasets.d4rl_dataset import DiscretizedDataset
from trajectory.utils.common import set_seed


def create_argparser():
    parser = argparse.ArgumentParser(description="Trajectory models training hyperparameters. All can be set from command line.")
    parser.add_argument("--config", default="configs/halfcheetah_medium.yaml")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cpu", type=str)

    return parser


def run_experiment(config, seed, device):
    config.run_seed = seed
    os.makedirs(config.trainer.checkpoints_path, exist_ok=True)
    OmegaConf.save(OmegaConf.to_container(config, resolve=True), os.path.join(config.trainer.checkpoints_path, "config.yaml"))

    set_seed(seed=seed)

    trainer_conf = config.trainer
    data_conf = config.dataset

    dataset = DiscretizedDataset(
        env_name=data_conf.env_name,
        seq_len=data_conf.seq_len,
        cache_path=data_conf.cache_path,
        num_bins=data_conf.num_bins,
        discount=data_conf.discount,
        strategy=data_conf.strategy
    )
    dataloader = DataLoader(dataset, batch_size=data_conf.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    model = GPT(**config.model)
    model.to(device)

    num_epochs = int(1e6 / len(dataset) * trainer_conf.num_epochs_ref)

    warmup_tokens = len(dataset) * data_conf.seq_len * config.model.transition_dim
    final_tokens = warmup_tokens * num_epochs

    wandb.init(
        **config.wandb,
        config=dict(OmegaConf.to_container(config, resolve=True))
    )
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
        eval_every=trainer_conf.eval_every,
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
        checkpoints_path=trainer_conf.checkpoints_path,
        save_every=1,
        device=device
    )
    trainer.train(
        model=model,
        dataloader=dataloader,
        num_epochs=num_epochs
    )


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