import warnings

import torch
import torch.nn.functional as F

from trajectory.utils.common import top_k_logits, round_to_multiple


def _sample_inner(logits, top_k, temperature, greedy=False):
    logits = logits / temperature

    if top_k is not None:
        logits = top_k_logits(logits, k=top_k)

    probs = F.softmax(logits, dim=-1)

    if greedy:
        idx = torch.topk(probs, k=1, dim=-1)[-1]
    else:
        idx = torch.multinomial(probs, num_samples=1)

    return idx


def sample(model, context, steps, top_k=None, model_state=None, temperature=1.0, greedy=False):
    batch_size = context.shape[0]

    raw_logits = torch.zeros(batch_size, steps, model.vocab_size, device=context.device)

    if model_state is None:
        logits, model_state = model(context, state=model_state)
        sampled_tokens = _sample_inner(logits[:, -1, :], top_k, temperature, greedy)
        context = torch.hstack([context, sampled_tokens])

        raw_logits[:, 0] = logits[:, -1, :]
        range_steps = range(1, steps)
    else:
        range_steps = range(steps)

    for t in range_steps:
        # +1 to account to one new token, as it will be added to cache during forward
        n_crop = round_to_multiple(max(0, (model_state[0].shape[1] + 1) - model.get_seq_len()), model.transition_dim)
        if n_crop != 0:
            warnings.warn("Seems like cache exceeded max attention length. Please, remember that cached version"
                          " is identical to quadratic attention only when (context + planning) length <= model.seq_len."
                          " Cache will be cropped in same fashion as context, but the model may begin to behave"
                          " in the wrong way, as it has not seen such thing during training.")
            model_state = [s[:, n_crop:] for s in model_state]

        logits, model_state = model(context[:, -1:], state=model_state)

        sampled_tokens = _sample_inner(logits[:, -1, :], top_k, temperature, greedy)
        context = torch.hstack([context, sampled_tokens])

        raw_logits[:, t] = logits[:, -1, :]

    return context, model_state, raw_logits
