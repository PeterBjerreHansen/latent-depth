"""Fixed-depth next-latent prediction used by Stage 02 experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from config import ExperimentConfig


class NextLatentPredictor(nn.Module):
    """Small predictor shared across all fixed target-depth arms."""

    def __init__(self, n_embd: int, hidden_mult: int = 2) -> None:
        super().__init__()
        if n_embd <= 0 or hidden_mult <= 0:
            raise ValueError("predictor dimensions must be positive")
        hidden = n_embd * hidden_mult
        self.net = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_embd),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_auxiliary_predictor(
    cfg: ExperimentConfig,
    device: torch.device,
) -> NextLatentPredictor | None:
    """Construct the predictor without advancing the training RNG stream."""
    if cfg.auxiliary.mode == "none":
        return None
    if cfg.auxiliary.mode != "next_latent":
        raise ValueError(f"unsupported auxiliary mode: {cfg.auxiliary.mode}")

    # Construct on CPU from a private CPU generator state, then restore the
    # caller's CPU RNG exactly.  We deliberately avoid ``torch.manual_seed``
    # here because that API also seeds accelerator generators.  Moving an
    # already initialized module to CUDA/MPS does not draw random numbers, so
    # paired target-depth arms retain the same backbone/training RNG stream.
    caller_state = torch.get_rng_state().clone()
    private = torch.Generator(device="cpu").manual_seed(int(cfg.auxiliary.seed))
    try:
        torch.set_rng_state(private.get_state())
        predictor = NextLatentPredictor(
            cfg.model.n_embd,
            hidden_mult=cfg.auxiliary.predictor_hidden_mult,
        )
    finally:
        torch.set_rng_state(caller_state)
    return predictor.to(device)


def next_latent_views(
    hidden_states: list[torch.Tensor],
    target_layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source h_t^D and detached target h_{t+1}^j.

    Hidden-state convention is the same as the diagnostics: element 0 is the
    token+position embedding stream and j>0 is post Transformer block j.
    """
    if len(hidden_states) < 2:
        raise ValueError("hidden_states must include embedding and at least one block stream")
    if not 0 <= int(target_layer) < len(hidden_states):
        raise ValueError(
            f"target_layer must lie in [0,{len(hidden_states) - 1}]"
        )
    source_full = hidden_states[-1]
    target_full = hidden_states[int(target_layer)]
    if source_full.ndim != 3 or target_full.shape != source_full.shape:
        raise ValueError("hidden states must share shape [batch, sequence, channels]")
    if source_full.size(1) < 2:
        raise ValueError("next-latent prediction requires sequence length >= 2")
    source = source_full[:, :-1, :]
    target = target_full[:, 1:, :].detach()
    return source, target


def next_latent_loss(
    hidden_states: list[torch.Tensor],
    predictor: NextLatentPredictor,
    target_layer: int,
) -> torch.Tensor:
    """Mean cosine-distance loss from final-depth source to detached target."""
    source, target = next_latent_views(hidden_states, target_layer)
    prediction = predictor(source)
    if prediction.shape != target.shape:
        raise RuntimeError("predictor output does not match target shape")
    return (1.0 - F.cosine_similarity(prediction, target, dim=-1)).mean()

