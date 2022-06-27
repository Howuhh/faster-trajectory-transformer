import os
import gym
import d4rl
import torch
import imageio
import numpy as np

from tqdm.auto import trange
from trajectory.planning.beam import beam_plan, batch_beam_plan


def create_env(name):
    wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env


@torch.no_grad()
def rollout(
        env,
        model,
        discretizer,
        beam_context_size,
        beam_steps,
        beam_width,
        sample_expand=1,
        k_reward=1,
        k_act=None,
        k_obs=None,
        temperature=1,
        discount=0.99,
        plan_every=1,
        max_steps=1000,
        value_placeholder=1e6,
        render_path=None,
        device="cpu"
    ):
    assert plan_every <= beam_steps, "too short planning horizon"
    os.makedirs(render_path, exist_ok=True)

    transition_dim, obs_dim, act_dim = model.transition_dim, model.observation_dim, model.action_dim
    # trajectory of tokens for model action planning
    # +1 just to avoid index error while updating context on the last step
    context = torch.zeros(1, model.transition_dim * (max_steps + 1), dtype=torch.long).to(device)

    obs = env.reset()
    obs_tokens = discretizer.encode(obs, subslice=(0, obs_dim)).squeeze()

    context[:, :model.observation_dim] = torch.as_tensor(obs_tokens, device=device)  # initial tokens for planning

    done, total_reward, render_frames = False, 0.0, []
    for step in (pbar := trange(max_steps, desc="Rollout steps", leave=False)):
        if step % plan_every == 0:
            # removing zeros from future, keep only context updated so far
            context_offset = model.transition_dim * (step + 1) - model.action_dim - 2
            # prediction: [a, s, a, s, ...]
            prediction_tokens = beam_plan(
                model, discretizer, context[:, :context_offset],
                steps=beam_steps,
                beam_width=beam_width,
                context_size=beam_context_size,
                k_act=k_act, k_obs=k_obs, k_reward=k_reward,
                temperature=temperature,
                discount=discount,
                sample_expand=sample_expand
            )
        else:
            # shift one transition forward in plan
            prediction_tokens = prediction_tokens[transition_dim:]

        action_tokens = prediction_tokens[:act_dim]
        action = discretizer.decode(action_tokens.cpu().numpy(), subslice=(obs_dim, obs_dim + act_dim)).squeeze()

        obs, reward, done, _ = env.step(action)

        total_reward += reward
        pbar.set_postfix({'Reward': total_reward})
        if done:
            break

        obs_tokens = discretizer.encode(obs, subslice=(0, obs_dim)).squeeze()
        reward_tokens = discretizer.encode(
            np.array([reward, value_placeholder]),
            subslice=(transition_dim - 2, transition_dim)
        )

        # updating context with new action and obs
        context_offset = model.transition_dim * step
        # [s, ...] -> [s, a, ...]
        context[:, context_offset + obs_dim:context_offset + obs_dim + act_dim] = torch.as_tensor(action_tokens, device=device)
        # [s, a, ...] -> [s, a, r, v, ...]
        context[:, context_offset + transition_dim - 2:context_offset + transition_dim] = torch.as_tensor(reward_tokens, device=device)
        # [s, a, r, v, ...] -> [s, a, r, v, s, ...]
        context[:, context_offset + model.transition_dim:context_offset + model.transition_dim + model.observation_dim] = torch.as_tensor(obs_tokens, device=device)

        if render_path is not None:
            # add frame to rollout video
            render_frames.append(env.render(mode="rgb_array", height=128, width=128))

    if render_path is not None:
        imageio.mimsave(os.path.join(render_path, "rollout.gif"), render_frames, fps=64, format="gif")

    return total_reward

@torch.no_grad()
def vec_rollout(
        vec_env,
        model,
        discretizer,
        beam_context_size,
        beam_steps,
        beam_width,
        sample_expand=1,
        k_reward=1,
        k_act=None,
        k_obs=None,
        temperature=1,
        discount=0.99,
        plan_every=1,
        max_steps=1000,
        value_placeholder=1e6,
        device="cpu"
    ):
    transition_dim, obs_dim, act_dim = model.transition_dim, model.observation_dim, model.action_dim
    # some value to place, it will be masked out in the attention
    value_placeholder = np.ones((vec_env.num_envs, 1)) * value_placeholder

    # trajectory of tokens for model action planning
    # +1 just to avoid index error while updating context on the last step
    context = torch.zeros(vec_env.num_envs, transition_dim * (max_steps + 1), dtype=torch.long, device=device)

    obs = vec_env.reset()
    obs_tokens = discretizer.encode(obs, subslice=(0, obs_dim))

    context[:, :obs_dim] = torch.as_tensor(obs_tokens, device=device)  # initial tokens for planning

    total_rewards = np.zeros(vec_env.num_envs)
    dones = np.zeros(vec_env.num_envs, dtype=np.bool)
    for step in trange(max_steps, desc="Rollout steps", leave=True):
        # removing zeros from future, keep only context updated so far
        if step % plan_every == 0:
            context_offset = transition_dim * (step + 1) - act_dim - 2

            # plan only for envs which are not done yet
            context_not_dones = context[~dones, :context_offset]

            prediction_tokens = batch_beam_plan(
                model, discretizer, context_not_dones, steps=beam_steps, beam_width=beam_width, context_size=beam_context_size,
                k_act=k_act, k_obs=k_obs, k_reward=k_reward, temperature=temperature, discount=discount, sample_expand=sample_expand
            )

            plan_buffer = torch.zeros(vec_env.num_envs, prediction_tokens.shape[-1], dtype=torch.long, device=device)
            plan_buffer[~dones] = prediction_tokens
        else:
            plan_buffer = plan_buffer[:, transition_dim:]

        action_tokens = plan_buffer[:, :act_dim]
        action = discretizer.decode(action_tokens.cpu().numpy(), subslice=(obs_dim, obs_dim + act_dim))

        obs, reward, done, _ = vec_env.step(action)

        obs_tokens = discretizer.encode(obs[~dones], subslice=(0, obs_dim))
        reward_tokens = discretizer.encode(
            np.hstack([reward.reshape(-1, 1), value_placeholder]),
            subslice=(transition_dim - 2, transition_dim)
        )
        # updating context with new obs, action and reward
        context_offset = model.transition_dim * step
        # [s, ...] -> [s, a, ...]
        context[~dones, context_offset + obs_dim:context_offset + obs_dim + act_dim] = torch.as_tensor(action_tokens[~dones], device=device)
        # [s, a, ...] -> [s, a, r, v, ...]
        context[~dones, context_offset + transition_dim - 2:context_offset + transition_dim] = torch.as_tensor(reward_tokens[~dones], device=device)
        # [s, a, r, v, ...] -> [s, a, r, v, s, ...]
        context[~dones, context_offset + model.transition_dim:context_offset + model.transition_dim + model.observation_dim] = torch.as_tensor(obs_tokens, device=device)

        # updating total reward for logging
        total_rewards[~dones] += reward[~dones]

        dones[done] = True
        if np.all(dones):
            break

    return total_rewards










