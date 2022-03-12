import torch
import torch.nn.functional as F

from tqdm.auto import trange

from trajectory.planning.sample import sample
from trajectory.utils.common import round_to_multiple


@torch.no_grad()
def beam_plan(model, discretizer, context, steps, beam_width, sample_expand, discount=0.99, context_size=None,
                    k_obs=1, k_reward=1, k_act=None, temperature=1.0):
    # as exact as possible to original implementation (not batched)
    rewards = torch.zeros(beam_width, steps + 1, device=context.device)
    discounts = discount ** torch.arange(steps + 1, device=context.device)

    if context_size is not None:
        context_size = context_size * model.transition_dim

        n_crop = round_to_multiple(max(0, context.shape[1] - context_size), model.transition_dim)
        context = context[:, n_crop:]

    plan = context.repeat(beam_width, 1)
    model_state = None
    for t in trange(steps, leave=False):
        # [beam_width * sample_expand, ...]
        plan = plan.repeat(sample_expand, 1)
        rewards = rewards.repeat(sample_expand, 1)

        if model_state is not None:
            # [beam_width * sample_expand, cache_len, emb_dim]
            model_state = [s.repeat(sample_expand, 1, 1) for s in model_state]

        # sample action tokens
        plan, model_state, _ = sample(
            model, plan, model_state=model_state, steps=model.action_dim, top_k=k_act, temperature=temperature
        )
        # sample reward and value estimates
        plan, model_state, logits = sample(
            model, plan, model_state=model_state, steps=2, top_k=k_reward, temperature=temperature
        )
        probs = F.softmax(logits, dim=-1)
        reward_and_value = discretizer.expectation(probs, subslice=[model.transition_dim - 2, model.transition_dim])

        rewards[:, t:t + 2] = reward_and_value

        values = (rewards * discounts).sum(-1)
        values, idxs = torch.topk(values, k=beam_width)

        plan, rewards = plan[idxs], rewards[idxs]
        model_state = [s[idxs] for s in model_state]

        if t < steps - 1:
            # sample obs unless last step
            plan, model_state, _ = sample(
                model, plan, model_state=model_state, steps=model.observation_dim, top_k=k_obs, temperature=temperature
            )

    best_idx = torch.argmax(values)
    best_plan = plan[best_idx, context.shape[1]:]

    return best_plan


@torch.no_grad()
def batch_beam_plan(model, discretizer, context, steps, beam_width, sample_expand, discount=0.99, context_size=None,
                    k_obs=1, k_reward=1, k_act=None, temperature=1.0):
    batch_size, seq_len = context.shape[0], model.get_seq_len()

    # construct reward and discount tensors for estimating values
    rewards = torch.zeros(batch_size, beam_width, steps + 1, device=context.device)
    discounts = discount ** torch.arange(steps + 1, device=context.device)

    if context_size is not None:
        context_size = context_size * model.transition_dim

        n_crop = round_to_multiple(max(0, context.shape[1] - context_size), model.transition_dim)
        context = context[:, n_crop:]

    # [batch_size, context_len] -> [batch_size, beam_width, context_len]
    plan = context.unsqueeze(1).repeat(1, beam_width, 1)

    model_state = None
    for t in trange(steps, leave=False):
        # [batch_size * (beam_width * sample_expand), context_len]
        plan = plan.repeat(1, sample_expand, 1).flatten(0, 1)
        # [batch_size * beam_width * sample_expand, steps + 1]
        rewards = rewards.repeat(1, sample_expand, 1).flatten(0, 1)

        if model_state is not None:
            new_model_state = []
            for s in model_state:
                _, seq_len, emb_dim = s.shape
                new_model_state.append(
                    s.view(batch_size, beam_width, seq_len, emb_dim).repeat(1, sample_expand, 1, 1).flatten(0, 1)
                )
            model_state = new_model_state

        # sample action tokens
        plan, model_state, _ = sample(
            model, plan, model_state=model_state, steps=model.action_dim, top_k=k_act, temperature=temperature
        )
        # sample reward and value tokens
        plan, model_state, logits = sample(
            model, plan, model_state=model_state, steps=2, top_k=k_reward, temperature=temperature
        )
        probs = F.softmax(logits, dim=-1)

        # expected reward and value estimate
        reward_and_value = discretizer.expectation(probs, subslice=[model.transition_dim - 2, model.transition_dim])
        rewards[..., t:t + 2] = reward_and_value
        # [batch_size * beam_width * sample_expand]
        values = (rewards * discounts).sum(-1)

        # [batch_size, beam_width]
        values, indices = torch.topk(values.view(batch_size, -1), k=beam_width, dim=-1)
        indices = indices.unsqueeze(-1)

        # selecting best candidates by cumulative reward
        rewards = rewards.view(batch_size, beam_width * sample_expand, -1)
        rewards = rewards.gather(1, indices.repeat(1, 1, steps + 1))

        plan = plan.view(batch_size, beam_width * sample_expand, -1)
        plan = plan.gather(1, indices.repeat(1, 1, plan.shape[-1]))

        # [batch_size * beam_width * sample_expand, seq_len, emb_dim]
        best_model_state = []
        for s in model_state:
            _, seq_len, emb_dim = s.shape

            s = s.view(batch_size, beam_width * sample_expand, seq_len, emb_dim)
            best_model_state.append(
                s.gather(1, indices.unsqueeze(-1).repeat(1, 1, seq_len, emb_dim)).flatten(0, 1)
            )

        if t < steps - 1:
            plan = plan.flatten(0, 1)
            # sample next obs unless end of planning horizon
            plan, model_state, _ = sample(
                model, plan, model_state=best_model_state, steps=model.observation_dim, top_k=k_obs, temperature=temperature
            )
            plan = plan.view(batch_size, beam_width, -1)

    argmax = torch.argmax(values, dim=-1)

    best_plan = plan[torch.arange(batch_size), argmax]
    best_plan = best_plan[:, context.shape[1]:]

    return best_plan


