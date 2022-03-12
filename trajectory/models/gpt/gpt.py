import torch
import numpy as np
import torch.nn as nn

from trajectory.models.gpt.ein_linear import EinLinear
from trajectory.utils.common import round_to_multiple


class TransformerBlock(nn.Module):
    """"  Pre-norm transformer block   """
    def __init__(self, transition_dim, seq_len, embedding_dim, num_heads, attention_dropout, residual_dropout):
        super().__init__()
        self.seq_len = seq_len
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, attention_dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer("attn_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool))
        # mask out previous value estimates (as they have information about future)
        self.attn_mask[:, transition_dim - 1::transition_dim] = True

    def forward(self, x, state=None, attn_pad_mask=None):
        # state is a previous input to this layer
        x = self.norm1(x)

        if state is None:
            # if context_len < seq_len
            attn_mask = self.attn_mask[:x.shape[1], :x.shape[1]]
            q, k, v = x, x, x
        else:
            assert x.size(1) == 1, f'when using memory input should be 1-time-step tensor, got {x.size(1)} timesteps.'
            assert state.shape[1] + 1 <= self.seq_len, f"{state.shape[1] + 1}"

            attn_mask = None
            q, k, v = x, torch.cat([state, x], dim=1), torch.cat([state, x], dim=1)

        new_state = k
        x = x + self.drop(self.attention(q, k, v, attn_mask=attn_mask, key_padding_mask=attn_pad_mask, need_weights=False)[0])
        x = x + self.mlp(self.norm2(x))

        return x, new_state


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        transition_dim,
        observation_dim,
        action_dim,
        seq_len,
        embedding_dim=32,
        num_layers=4,
        num_heads=4,
        embedding_dropout=0.1,
        residual_dropout=0.1,
        attention_dropout=0.1,
        use_sep_heads=False
    ):
        super().__init__()
        # input embedding, vocab_size tokens for each state, action dimension
        self.tok_emb = nn.Embedding(vocab_size * transition_dim, embedding_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, embedding_dim))

        self.drop_emb = nn.Dropout(embedding_dropout)

        # transformer
        self.blocks = nn.ModuleList(
            [TransformerBlock(transition_dim, seq_len, embedding_dim, num_heads, attention_dropout, residual_dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embedding_dim)

        # token's classifier
        if use_sep_heads:
            # see https://github.com/jannerm/trajectory-transformer/issues/3
            self.head = EinLinear(transition_dim, embedding_dim, vocab_size, bias=False)
        else:
            self.head = nn.Linear(embedding_dim, vocab_size)

        # constants
        self.vocab_size = vocab_size
        self.stop_token = vocab_size * transition_dim
        self.seq_len = seq_len
        self.num_layers = num_layers

        self.action_dim = action_dim
        self.observation_dim = observation_dim
        # action_dim + state_dim + reward + value
        self.transition_dim = transition_dim
        self.embedding_dim = embedding_dim
        self.use_sep_heads = use_sep_heads

        self.apply(self._init_weights)

    def get_seq_len(self):
        return self.seq_len

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        # new, from https://github.com/karpathy/minGPT/pull/62
        elif isinstance(module, GPT):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def _offset_tokens(self, tokens, state=None):
        t = tokens.shape[1] if state is None else state[0].shape[1] + 1
        # offset token indices to appropriate range in nn.Embedding
        # example for state_dim=3, action_dim=2, vocab_size=10:
        # tokens: [4, 0, 1, 5, 10]
        # offset by vocab_size for each index: [0, 10, 20, 30, 40]
        # -> [4, 10, 21, 35, 50]
        n_states = int(np.ceil(t / self.transition_dim))
        offsets = torch.arange(self.transition_dim, device=tokens.device) * self.vocab_size
        offsets = offsets.repeat(n_states)

        if state is None:
            offset_idx = tokens + offsets[:t]
        else:
            offset_idx = tokens + offsets[:t][-1]

        return offset_idx

    def _pad_to_full_transition(self, x):
        b, t, _ = x.shape  # [batch , seq_len, embedding_dim]
        # tokens to be next multiple of transition_dim
        n_pad = round_to_multiple(t, multiple=self.transition_dim) - t
        padding = torch.zeros(b, n_pad, self.embedding_dim, device=x.device)
        # [batch, round_to_multiple(seq_len, transition_dim), embedding_dim]
        x_pad = torch.cat([x, padding], dim=1)

        return x_pad, n_pad

    def forward(self, tokens, state=None, attn_pad_mask=None):
        b, t = tokens.size()
        assert t <= self.seq_len, "Cannot forward, model sequence length is exhausted."
        if state is not None:
            assert t == 1, f'when using memory input should be 1-time-step tensor, got {t} timesteps.'

        # [batch, seq_len]
        offset_idx = self._offset_tokens(tokens, state=state)
        # [batch, seq_len, embedding_dim]
        token_embeddings = self.tok_emb(offset_idx)

        # [1, seq_len, embedding_dim]
        if state is None:
            state = [None for _ in range(self.num_layers)]
            position_embeddings = self.pos_emb[:, :t, :]
        else:
            position_embeddings = self.pos_emb[:, state[0].shape[1], :]

        # [batch, seq_len, embedding_dim]
        x = self.drop_emb(token_embeddings + position_embeddings)

        new_state = []
        for block, block_state in zip(self.blocks, state):
            x, new_block_state = block(x, state=block_state, attn_pad_mask=attn_pad_mask)
            new_state.append(new_block_state)

        # [batch, seq_len, embedding_dim]
        x = self.norm(x)

        if self.use_sep_heads:
            # uses separate heads for each token in transition_dim
            if state[0] is None:
                # pad with zeros to be multiple of transition_dim
                x_pad, n_pad = self._pad_to_full_transition(x)
                # [batch * (round_to_multiple(seq_len, transition_dim) / transition_dim), transition_dim, embedding_dim]
                x_pad = x_pad.view(-1, self.transition_dim, self.embedding_dim)
                # [batch * (round_to_multiple(seq_len, transition_dim) / transition_dim), transition_dim, vocab_size]
                logits = self.head(x_pad, model_idx=None)
                # [batch, round_to_multiple(seq_len, transition_dim), vocab_size]
                logits = logits.reshape(b, t + n_pad, self.vocab_size)
                # [batch, seq_len, embedding_dim]
                logits = logits[:, :t, :]
            else:
                # select only needed head from heads
                transition_idx = state[0].shape[1] % self.transition_dim

                logits = self.head(x.squeeze(1), model_idx=transition_idx).unsqueeze(1)
        else:
            # one head for all tokens in transition
            # [batch, seq_len, vocab_size]
            logits = self.head(x)

        return logits, new_state