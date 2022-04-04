import torch
import numpy as np

from tqdm.auto import trange

from trajectory.planning.sample import sample
from trajectory.utils.common import round_to_multiple


def convert_to_seq(discretizer, cp_obs, cp_act, cp_reward, curr_obs, curr_action, obs_dim, act_dim, value_placeholder=1e6, device=torch.device("cpu")):
    # cp_obs: [batch_size, context_size * observation_dim]
    # cp_act: [batch_size, context_size * action_dim]
    batch_size, context_size = cp_obs.shape
    context_size = context_size // obs_dim
    transition_dim = obs_dim + act_dim + 2
    value_placeholder = np.ones((batch_size, 1)) * value_placeholder

    context = torch.zeros(batch_size, transition_dim * context_size + obs_dim + act_dim, dtype=torch.long).to(device)
    for t in trange(context_size, desc="Converting steps", leave=True):
        obs = cp_obs[:, t * obs_dim:(t + 1) * obs_dim]
        act = cp_act[:, t * act_dim:(t + 1) * act_dim]
        reward = cp_reward[:, t:(t + 1)]
        obs_tokens = discretizer.encode(
            obs, subslice=(0, obs_dim)
        )
        action_tokens = discretizer.encode(
            act, subslice=(obs_dim, obs_dim + act_dim)
        )
        reward_tokens = discretizer.encode(
            np.hstack([reward, value_placeholder]),
            subslice=(transition_dim - 2, transition_dim)
        )
        context_offset = transition_dim * t
        context[:, context_offset:context_offset + obs_dim] = torch.as_tensor(obs_tokens, device=device)
        context[:, context_offset + obs_dim:context_offset + obs_dim + act_dim] = torch.as_tensor(action_tokens, device=device)
        context[:, context_offset + transition_dim - 2:context_offset + transition_dim] = torch.as_tensor(reward_tokens, device=device)

    # add current obs and action
    curr_obs_tokens = discretizer.encode(
        curr_obs, subslice=(0, obs_dim)
    )
    curr_action_tokens = discretizer.encode(
        curr_action, subslice=(obs_dim, obs_dim + act_dim)
    )
    context_offset = transition_dim * context_size
    context[:, context_offset:context_offset + obs_dim] = torch.as_tensor(curr_obs_tokens, device=device)
    context[:, context_offset + obs_dim:context_offset + obs_dim + act_dim] = torch.as_tensor(curr_action_tokens, device=device)

    return context


@torch.no_grad()
def offline_rollout(model, discretizer, context, context_size=None, 
                    k_obs=1, k_reward=1, k_act=None, temperature=1.0):
    transition_dim, observation_dim = model.transition_dim, model.observation_dim

    if context_size is not None:
        context_size = context_size * transition_dim

        n_crop = round_to_multiple(max(0, context.shape[1] - context_size), transition_dim)
        context = context[:, n_crop:]

    model_state = None
    context, model_state, _ = sample(
        model, context, model_state=model_state, steps=2, top_k=k_reward, temperature=temperature
    )

    context, model_state, _ = sample(
        model, context, model_state=model_state, steps=observation_dim, top_k=k_obs, temperature=temperature
    )

    next_obs_tokens = context[:, -observation_dim:]
    reward_token = context[:, -observation_dim - 2:-observation_dim]
    next_obs = discretizer.decode(
        next_obs_tokens.cpu().numpy(),
        subslice=(0, observation_dim)
    )
    reward = discretizer.decode(
        reward_token.cpu().numpy(),
        subslice=(transition_dim - 2, transition_dim)
    )

    return next_obs, reward[:, 0]
