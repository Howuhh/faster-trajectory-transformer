import math


class GPTScheduler:
    """
    Linear warmup to optimizer inital_lr for #warmup_tokens,
    then cosine decay to inital_lr * final_lr_ratio for the rest #final_tokens
    source: https://github.com/karpathy/minGPT
    """
    def __init__(self, optimizer, warmup_tokens, final_tokens, final_lr_ratio=0.1, decay=True):
        self.optimizer = optimizer
        # assuming that lr same for all group
        self.init_lr = optimizer.param_groups[0]["lr"]

        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens
        self.final_lr_ratio = final_lr_ratio
        self.decay = decay

        self.tokens_count = 0.0

    def step(self, batch_size):
        lr_mult = self.__get_lr_multiplier(batch_size)

        for group in self.optimizer.param_groups:
            group["lr"] = self.init_lr * lr_mult

    def get_current_lr(self):
        lr_mult = self.__get_lr_multiplier(0.0)
        return self.init_lr * lr_mult

    def __get_lr_multiplier(self, batch_size):
        self.tokens_count += batch_size

        assert self.tokens_count <= self.final_tokens, f"number of tokens {self.tokens_count} already bigger than number of tokens for one cycle"

        if self.tokens_count < self.warmup_tokens:
            lr_mult = float(self.tokens_count) / float(max(1, self.warmup_tokens))
        elif self.tokens_count >= self.warmup_tokens and self.decay:
            tokens_passed = self.tokens_count - self.warmup_tokens
            tokens_left = self.final_tokens - self.warmup_tokens

            progress = float(tokens_passed) / float(max(1, tokens_left))
            lr_mult = max(self.final_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        else:
            lr_mult = 1.0

        return lr_mult

    def state_dict(self):
        # just for checkpoint callback
        pass
