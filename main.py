import torch
from torch.utils.data import DataLoader

from stable_baselines3.common.vec_env import DummyVecEnv

from trajectory.models.gpt import GPT, GPTTrainer
from trajectory.utils.env import create_env, vec_rollout
from trajectory.datasets.d4rl_dataset import DiscretizedDataset

DEVICE = "cpu"
DATASETS = [
    # halfcheetah
    "halfcheetah-medium-expert-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    # hopper
    "hopper-medium-expert-v2",
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    # walker
    "walker2d-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
]


def main():
    # This is example of training and evaluation if you want to train on your own without configs and scripts/train.py
    torch.manual_seed(42)
    dataset = DiscretizedDataset(
        env_name=DATASETS[1],
        seq_len=10,
        cache_path="data",
        num_bins=100,
        discount=0.99,
        strategy="uniform"
    )
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)

    model = GPT(
        vocab_size=100,
        transition_dim=25,
        observation_dim=17,
        action_dim=6,
        seq_len=25 * 10,
        embedding_dim=128,
        num_layers=4,
        num_heads=4,
        use_sep_heads=True
    )
    model.to(DEVICE)
    print("Number of model parameters:", sum(p.numel() for p in model.parameters()))

    num_epochs = int(1e6 / len(dataset) * 50)
    print(f"Training for {num_epochs} epochs")

    warmup_tokens = len(dataset) * 10 * 25
    final_tokens = warmup_tokens * num_epochs

    trainer = GPTTrainer(
        final_tokens=final_tokens,
        warmup_tokens=warmup_tokens,
        action_weight=5,
        learning_rate=6e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        clip_grad=1.0,
        eval_seed=42,
        eval_every=50,
        eval_episodes=5,
        eval_plan_every=1,
        eval_beam_width=32,
        eval_beam_steps=5,
        eval_beam_context=5,
        eval_sample_expand=2,
        eval_k_obs=1,  # as in original implementation
        eval_k_reward=1,
        eval_k_act=None,
        checkpoints_path=f"checkpoints/gpt/{DATASETS[1]}",
        save_every=1,
        device=DEVICE
    )
    trainer.train(
        model=model,
        dataloader=dataloader,
        num_epochs=num_epochs
    )

    # evaluation after training is done
    discretizer = dataset.get_discretizer()
    discretizer.to(DEVICE)

    vec_env = DummyVecEnv([lambda: create_env(DATASETS[1]) for _ in range(25)])
    rewards = vec_rollout(
        env=vec_env,
        model=model,
        discretizer=discretizer,
        beam_width=32,
        beam_context_size=5,
        beam_steps=5,
        plan_every=1,
        sample_expand=2,
        k_reward=1,
        k_obs=1,
        k_act=None,
        device=DEVICE
    )
    scores = [vec_env.envs[0].get_normalized_score(r) for r in rewards]

    print("Rewards:", rewards)
    print("Scores:", scores)


if __name__ == "__main__":
    main()