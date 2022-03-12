import os
import torch
import wandb
import numpy as np
import torch.nn.functional as F

from tqdm.auto import tqdm, trange
from stable_baselines3.common.vec_env import DummyVecEnv

from trajectory.utils.env import vec_rollout, create_env
from trajectory.models.gpt.ein_linear import EinLinear
from trajectory.utils.scheduler import GPTScheduler
from trajectory.utils.common import weight_decay_groups, set_seed


class GPTTrainer:
    def __init__(
            self,
            final_tokens,
            warmup_tokens=1_000_000,
            learning_rate=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            clip_grad=None,
            eval_seed=42,
            eval_every=10,
            eval_episodes=10,
            eval_plan_every=1,
            eval_beam_width=256,
            eval_beam_steps=64,
            eval_beam_context=16,
            eval_sample_expand=1,
            eval_temperature=1,
            eval_discount=0.99,
            eval_k_act=None,
            eval_k_obs=1,
            eval_k_reward=1,
            action_weight=1,
            value_weight=1,
            reward_weight=1,
            save_every=5,
            checkpoints_path=None,
            device="cpu"
    ):
        # optimizer params
        self.betas = betas
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_grad = clip_grad
        # loss params
        self.action_weight = action_weight
        self.reward_weight = reward_weight
        self.value_weight = value_weight
        # scheduler params
        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens
        # eval params
        self.eval_seed = eval_seed
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.eval_plan_every = eval_plan_every
        self.eval_beam_width = eval_beam_width
        self.eval_beam_steps = eval_beam_steps
        self.eval_beam_context = eval_beam_context
        self.eval_sample_expand = eval_sample_expand
        self.eval_temperature = eval_temperature
        self.eval_discount = eval_discount
        self.eval_k_act = eval_k_act
        self.eval_k_obs = eval_k_obs
        self.eval_k_reward = eval_k_reward
        # checkpoints
        self.save_every = save_every
        self.checkpoints_path = checkpoints_path

        self.device = device

    def get_optimizer(self, model):
        param_groups = weight_decay_groups(
            model=model,
            whitelist_modules=(torch.nn.Linear,  torch.nn.MultiheadAttention, EinLinear),
            blacklist_modules=(torch.nn.LayerNorm, torch.nn.Embedding),
            blacklist_named=("pos_emb",)
        )
        optim_groups = [
            {"params": param_groups["decay"], "weight_decay": self.weight_decay},
            {"params": param_groups["nodecay"], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=self.betas)

        return optimizer

    def get_scheduler(self, optimizer):
        scheduler = GPTScheduler(
            optimizer,
            warmup_tokens=self.warmup_tokens,
            final_tokens=self.final_tokens,
            decay=True,
        )
        return scheduler

    def __get_loss(self, model, batch):
        tokens, targets, loss_pad_mask = batch
        logits, state = model(tokens)

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        if self.action_weight != 1 or self.value_weight != 1 or self.reward_weight != 1:
            n_states = int(np.ceil(tokens.shape[1] / model.transition_dim))
            weights = torch.cat([
                torch.ones(model.observation_dim, device=tokens.device),
                torch.ones(model.action_dim, device=tokens.device) * self.action_weight,
                torch.ones(1, device=tokens.device) * self.reward_weight,
                torch.ones(1, device=tokens.device) * self.value_weight,
            ])
            weights = weights.repeat(n_states)[1:].repeat(tokens.shape[0], 1)
            loss = loss * weights.view(-1)

        loss = (loss * loss_pad_mask.view(-1)).mean()

        return loss

    def eval(self, env_name, model, discretizer, seed=None):
        model.eval()
        set_seed(seed=seed)

        vec_env = DummyVecEnv([lambda: create_env(env_name) for _ in range(self.eval_episodes)])
        rewards = vec_rollout(
            vec_env=vec_env,
            model=model,
            discretizer=discretizer,
            beam_context_size=self.eval_beam_context,
            beam_width=self.eval_beam_width,
            beam_steps=self.eval_beam_steps,
            plan_every=self.eval_plan_every,
            sample_expand=self.eval_sample_expand,
            k_act=self.eval_k_act,
            k_obs=self.eval_k_obs,
            k_reward=self.eval_k_reward,
            temperature=self.eval_temperature,
            discount=self.eval_discount,
            max_steps=vec_env.envs[0].max_episode_steps,
            device=self.device
        )
        scores = [vec_env.envs[0].get_normalized_score(r) for r in rewards]

        model.train()
        return np.mean(rewards), np.std(rewards), np.mean(scores), np.std(scores)

    def train(self, model, dataloader, num_epochs=1, log_every=100):
        model.train()

        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer)

        os.makedirs(self.checkpoints_path, exist_ok=True)
        if self.checkpoints_path is not None:
            torch.save(dataloader.dataset.get_discretizer(), os.path.join(self.checkpoints_path, "discretizer.pt"))

        for epoch in trange(1, num_epochs + 1, desc="Training"):
            epoch_losses = []
            for i, batch in enumerate(tqdm(dataloader, desc="Epoch", leave=False)):
                batch = [b.to(self.device) for b in batch]
                loss = self.__get_loss(model, batch)

                # step first, to prevent starting from high lr
                scheduler.step(batch_size=batch[0].reshape(-1).shape[0])
                optimizer.zero_grad()
                loss.backward()
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
                optimizer.step()

                # log here!!
                epoch_losses.append(loss.item())
                if i % log_every == 0:
                    wandb.log({
                        "train/loss_batch": loss.item(),
                        "train/lr": scheduler.get_current_lr()
                    })

            if epoch % self.eval_every == 0:
                env_name = dataloader.dataset.get_env_name()
                discretizer = dataloader.dataset.get_discretizer()
                discretizer.to(self.device)

                reward_mean, reward_std, score_mean, score_std = self.eval(env_name, model, discretizer, seed=self.eval_seed)

                wandb.log({
                    "eval/reward_mean": reward_mean,
                    "eval/reward_std": reward_std,
                    "eval/score_mean": score_mean,
                    "eval/score_std": score_std,
                })
                print(f"   EVAL {epoch}:", reward_mean, reward_std)

            if self.checkpoints_path is not None and epoch % self.save_every == 0:
                path = os.path.join(self.checkpoints_path, f"model_{epoch}.pt")
                torch.save(model.state_dict(), path)

            loss_mean = np.mean(epoch_losses)
            wandb.log({
                "train/loss_mean": loss_mean,
                "train/epoch": epoch
            })

            print(f'   EPOCH {epoch}:', loss_mean)

        if self.checkpoints_path is not None:
            torch.save(model.state_dict(), os.path.join(self.checkpoints_path, "model_last.pt"))

        return model