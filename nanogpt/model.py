"""A compact nanoGPT-style causal Transformer.

The implementation follows the public nanoGPT model structure while keeping
the hidden-state return hook needed by the RHM experiments.  The training
objective is standard causal next-token prediction; when no targets are
provided, only the final position is projected for efficient autoregressive
inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """Layer normalization with the optional bias used by nanoGPT."""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.size()
        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)
        key = key.view(batch_size, sequence_length, self.n_head, channels // self.n_head).transpose(1, 2)
        query = query.view(batch_size, sequence_length, self.n_head, channels // self.n_head).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.n_head, channels // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            attention = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
            attention = attention.masked_fill(self.bias[:, :, :sequence_length, :sequence_length] == 0, float("-inf"))
            attention = F.softmax(attention, dim=-1)
            attention = self.attn_dropout(attention)
            y = attention @ value

        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        if min(self.block_size, self.vocab_size, self.n_layer, self.n_head, self.n_embd) <= 0:
            raise ValueError("GPT dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        number_of_parameters = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {number_of_parameters / 1e6:.2f}M")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[
        torch.Tensor, torch.Tensor | None, list[torch.Tensor], torch.Tensor
    ]:
        if idx.ndim != 2:
            raise ValueError("idx must have shape [batch, sequence_length]")
        batch_size, sequence_length = idx.size()
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"cannot forward sequence of length {sequence_length}, block size is only {self.config.block_size}"
            )
        if targets is not None:
            valid_target_lengths = {sequence_length}
            if sequence_length > 0:
                valid_target_lengths.add(sequence_length - 1)
            if targets.ndim != 2 or targets.size(0) != batch_size or targets.size(1) not in valid_target_lengths:
                raise ValueError("targets must have shape [batch, sequence_length] or [batch, sequence_length-1]")

        position = torch.arange(0, sequence_length, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(position))
        hidden_states = [x] if return_hidden else None
        for block in self.transformer.h:
            x = block(x)
            if hidden_states is not None:
                hidden_states.append(x)
        x = self.transformer.ln_f(x)

        if targets is not None or return_hidden:
            logits = self.lm_head(x)
        else:
            logits = self.lm_head(x[:, [-1], :])

        loss = None
        if targets is not None:
            loss_logits = logits if targets.size(1) == sequence_length else logits[:, :-1]
            loss = F.cross_entropy(
                loss_logits.reshape(-1, loss_logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )

        if hidden_states is not None:
            return logits, loss, hidden_states, x
        return logits, loss
