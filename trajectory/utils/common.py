import os
import uuid
import torch
import random
import numpy as np
import warnings


def weight_decay_groups(model, whitelist_modules, blacklist_modules, blacklist_named=None):
    # from https://github.com/karpathy/minGPT
    decay, no_decay = set(), set()

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

            # starts with for rnn's, endswith other
            if pn.startswith('bias') or pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif (pn.startswith('weight') or pn.endswith('weight')) and isinstance(m, blacklist_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif (pn.startswith('weight') or pn.endswith('weight')) and isinstance(m, whitelist_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)

    if blacklist_named is not None:
        for name in blacklist_named:
            no_decay.add(name)  # also no decay

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    if len(inter_params) != 0:
        warnings.warn(f"parameters {str(inter_params)} made it into both decay/no_decay sets! They will be added to only no_decay by default.")
        decay = decay - no_decay

    inter_params = decay & no_decay
    union_params = decay | no_decay
    if len(param_dict.keys() - union_params) != 0:
        warnings.warn(f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set! They will be added to decay by default.")
        decay = decay | (param_dict.keys() - union_params)

    optim_groups = {
        "decay": [param_dict[pn] for pn in sorted(list(decay))],
        "nodecay": [param_dict[pn] for pn in sorted(list(no_decay))]
    }
    return optim_groups


def set_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)


def round_to_multiple(number, multiple):
    pad = (multiple - number % multiple) % multiple
    return number + pad


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    logits[logits < v[:, [-1]]] = -float('Inf')
    return logits


def topk_2d(x, k):
    M, N = x.shape
    k = min(M * N, k) # or assert, beam width bigger than number of candidates ???

    x_top, indices = torch.topk(x.view(-1), k)
    # indices // N, indices % N
    rows, cols = torch.div(indices, N, rounding_mode='floor'), indices % N

    return x_top.view(k, 1), rows, cols


def pad_along_axis(arr, pad_to, axis=0, fill_value=0):
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)

    return np.pad(arr, pad_width=npad, mode='constant', constant_values=fill_value)