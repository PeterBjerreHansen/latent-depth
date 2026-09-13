"""Latent-structure diagnostics for a frozen causal RHM language model."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from config import ExperimentConfig, DiagnosticsConfig
from nanogpt import GPT
from rhm.dataset import RHMSplit, slice_rhm_split
from rhm.interventions import (
    latent_labels,
    latent_location,
    non_synonym_pairing,
    synonym_counterfactual,
    variable_counterfactual,
)
from rhm.random_hierarchy_model import TensorDict


def _capture_rng_state() -> dict[str, Any]:
    mps = getattr(torch, "mps", None)
    mps_state = None
    if mps is not None and torch.backends.mps.is_available() and hasattr(mps, "get_rng_state"):
        if hasattr(mps, "synchronize"):
            mps.synchronize()
        mps_state = mps.get_rng_state().cpu().clone()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps": mps_state,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    mps = getattr(torch, "mps", None)
    if (
        mps is not None
        and torch.backends.mps.is_available()
        and state.get("mps") is not None
        and hasattr(mps, "set_rng_state")
    ):
        mps.set_rng_state(state["mps"])


@torch.no_grad()
def _features_at_positions(
    model: GPT,
    leaves: torch.Tensor,
    positions: list[int],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return residual-stream features with shape ``[R, J, N, C]`` on CPU."""
    if leaves.ndim != 2:
        raise ValueError("leaves must have shape [num_examples, sequence_length]")
    if not positions:
        raise ValueError("positions must be non-empty")
    if min(positions) < 0 or max(positions) >= leaves.shape[1]:
        raise ValueError("diagnostic position lies outside the leaf sequence")

    by_position: list[list[torch.Tensor]] = [[] for _ in positions]
    for start in range(0, leaves.shape[0], batch_size):
        tokens = leaves[start : start + batch_size].to(device)
        _, _, hidden, _ = model(tokens, return_hidden=True)
        stacked = torch.stack(hidden, dim=0)  # [J, B, T, C]
        for index, position in enumerate(positions):
            by_position[index].append(stacked[:, :, position, :].detach().cpu())
    return torch.stack([torch.cat(parts, dim=1) for parts in by_position], dim=0)


def extract_features_at_positions(
    model: GPT,
    leaves: torch.Tensor,
    positions: list[int],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return frozen residual-stream features at selected causal positions.

    The returned tensor has shape ``[positions, layers, examples, channels]``
    and is kept on CPU.  This is the public seam used by offline controls that
    need to fit more than one probe split on the same frozen representation.
    """
    return _features_at_positions(
        model, leaves, positions, batch_size=batch_size, device=device
    )


def fit_linear_probes(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    vocab_size: int,
    steps: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    eps: float = 1e-8,
    fit_indices: torch.Tensor | Sequence[int] | None = None,
    eval_indices: torch.Tensor | Sequence[int] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
]:
    """Fit independent probes on a supplied or deterministic random split.

    When ``fit_indices`` and ``eval_indices`` are supplied, they define the
    complete split and are not resampled.  This lets controls vary probe
    sample size while keeping one fixed held-out evaluation set.
    """
    if features.ndim != 4 or labels.ndim != 2:
        raise ValueError("unexpected probe feature/label rank")
    R, J, N, C = features.shape
    if labels.shape != (R, N):
        raise ValueError("probe labels do not align with features")
    if N < 4:
        raise ValueError("at least four examples are required for probe fitting")

    if (fit_indices is None) != (eval_indices is None):
        raise ValueError("fit_indices and eval_indices must be supplied together")
    if fit_indices is None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        permutation = torch.randperm(N, generator=generator)
        fit_idx, eval_idx = permutation[: N // 2], permutation[N // 2 :]
    else:
        fit_idx = torch.as_tensor(fit_indices, dtype=torch.long).flatten().cpu()
        eval_idx = torch.as_tensor(eval_indices, dtype=torch.long).flatten().cpu()
        if fit_idx.numel() == 0 or eval_idx.numel() == 0:
            raise ValueError("probe fit and evaluation splits must be nonempty")
        if torch.any(fit_idx < 0) or torch.any(fit_idx >= N):
            raise ValueError("probe fit indices lie outside the feature set")
        if torch.any(eval_idx < 0) or torch.any(eval_idx >= N):
            raise ValueError("probe evaluation indices lie outside the feature set")
        if torch.unique(fit_idx).numel() != fit_idx.numel():
            raise ValueError("probe fit indices must be unique")
        if torch.unique(eval_idx).numel() != eval_idx.numel():
            raise ValueError("probe evaluation indices must be unique")
        if torch.isin(fit_idx, eval_idx).any():
            raise ValueError("probe fit and evaluation indices must be disjoint")
    fit_size = int(fit_idx.numel())
    eval_size = int(eval_idx.numel())
    x_fit = features[:, :, fit_idx, :].to(device=device, dtype=torch.float32)
    x_eval = features[:, :, eval_idx, :].to(device=device, dtype=torch.float32)
    fit_mean = x_fit.mean(dim=2, keepdim=True)
    fit_std = x_fit.std(dim=2, keepdim=True, unbiased=False).clamp_min(eps)
    x_fit = (x_fit - fit_mean) / fit_std
    x_eval = (x_eval - fit_mean) / fit_std
    y_fit_base = labels[:, fit_idx].to(device=device, dtype=torch.long)
    y_eval_base = labels[:, eval_idx].to(device=device, dtype=torch.long)

    weight = torch.nn.Parameter(torch.zeros(R, J, vocab_size, C, device=device))
    bias = torch.nn.Parameter(torch.zeros(R, J, vocab_size, device=device))
    optimizer = torch.optim.Adam([weight, bias], lr=learning_rate)
    y_fit = y_fit_base[:, None, :].expand(R, J, fit_size)
    for _ in range(steps):
        logits = torch.einsum("rjnc,rjvc->rjnv", x_fit, weight) + bias[:, :, None, :]
        per_example = F.cross_entropy(
            logits.reshape(-1, vocab_size), y_fit.reshape(-1), reduction="none"
        ).view(R, J, fit_size)
        loss = per_example.mean(dim=-1).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = torch.einsum("rjnc,rjvc->rjnv", x_eval, weight) + bias[:, :, None, :]
        y_eval = y_eval_base[:, None, :].expand(R, J, eval_size)
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == y_eval).float().mean(dim=-1).cpu()
        ce = F.cross_entropy(
            logits.reshape(-1, vocab_size), y_eval.reshape(-1), reduction="none"
        ).view(R, J, eval_size).mean(dim=-1).cpu()
        predictions_cpu = predictions.cpu()
        labels_cpu = y_eval_base.cpu()
        ordinary_majority_accuracy = torch.empty(R, dtype=torch.float64)
        balanced_majority_accuracy = torch.empty(R, dtype=torch.float64)
        represented_classes = torch.empty(R, dtype=torch.int64)
        balanced_accuracy = torch.empty(R, J, dtype=torch.float64)
        for r in range(R):
            class_counts = torch.bincount(labels_cpu[r], minlength=vocab_size).to(torch.float64)
            present = class_counts > 0
            ordinary_majority_accuracy[r] = class_counts.max() / eval_size
            represented_classes[r] = present.sum()
            balanced_majority_accuracy[r] = 1.0 / represented_classes[r]
            for j in range(J):
                correct = (predictions_cpu[r, j] == labels_cpu[r]).to(torch.float64)
                correct_by_class = torch.bincount(
                    labels_cpu[r], weights=correct, minlength=vocab_size
                )
                balanced_accuracy[r, j] = (
                    (correct_by_class[present] / class_counts[present]).mean()
                )
    return (
        accuracy,
        ce,
        ordinary_majority_accuracy,
        balanced_majority_accuracy,
        represented_classes,
        balanced_accuracy,
        fit_size,
        eval_size,
    )


def _fit_linear_probes(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    vocab_size: int,
    steps: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    eps: float = 1e-8,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
]:
    """Backward-compatible internal wrapper using the ordinary random split."""
    return fit_linear_probes(
        features,
        labels,
        vocab_size=vocab_size,
        steps=steps,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
        eps=eps,
    )


def run_probe_control(
    model: GPT,
    split: RHMSplit,
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    shuffle_labels: bool = False,
    settings: DiagnosticsConfig | None = None,
) -> dict[str, Any]:
    """Fit the same frozen linear probes with real or shuffled latent labels.

    The shuffled-label mode is a negative control for probe capacity and the
    held-out split. It preserves the per-level class frequencies but breaks
    the relationship to the representation.
    """
    settings = settings or DiagnosticsConfig()
    settings.validate()
    n = min(int(settings.num_sequences), int(split.leaves.shape[0]))
    if n < 4:
        raise ValueError("diagnostic split must provide at least four examples")
    diagnostic_split = slice_rhm_split(split, n)
    levels = list(range(1, cfg.rhm.L))
    positions = [
        latent_location(L=cfg.rhm.L, s=cfg.rhm.s, abstraction_level=level).completion_position
        for level in levels
    ]
    rng_state = _capture_rng_state()
    was_training = model.training
    try:
        model.eval()
        features = _features_at_positions(
            model,
            diagnostic_split.leaves,
            positions,
            batch_size=cfg.train.batch_size,
            device=device,
        )
        labels = torch.stack(
            [latent_labels(diagnostic_split, L=cfg.rhm.L, abstraction_level=level) for level in levels],
            dim=0,
        )
        if shuffle_labels:
            generator = torch.Generator(device="cpu").manual_seed(settings.seed + 91_337)
            labels = labels[:, torch.randperm(n, generator=generator)]
        (
            accuracy,
            ce,
            ordinary_majority_accuracy,
            balanced_majority_accuracy,
            represented_classes,
            balanced_accuracy,
            fit_size,
            eval_size,
        ) = _fit_linear_probes(
            features,
            labels,
            vocab_size=cfg.rhm.v,
            steps=settings.probe_steps,
            learning_rate=settings.probe_lr,
            seed=settings.seed,
            device=device,
            eps=settings.eps,
        )
        return {
            "shuffle_labels": shuffle_labels,
            "uniform_random_accuracy": 1.0 / cfg.rhm.v,
            "ordinary_majority_accuracy_by_level": {
                str(level): float(ordinary_majority_accuracy[r])
                for r, level in enumerate(levels)
            },
            "represented_classes_by_level": {
                str(level): int(represented_classes[r])
                for r, level in enumerate(levels)
            },
            "balanced_majority_accuracy_by_level": {
                str(level): float(balanced_majority_accuracy[r])
                for r, level in enumerate(levels)
            },
            "fit_examples": fit_size,
            "eval_examples": eval_size,
            "levels": levels,
            "positions": positions,
            "accuracy_by_level": {
                str(level): [float(x) for x in accuracy[r].tolist()]
                for r, level in enumerate(levels)
            },
            "ce_by_level": {
                str(level): [float(x) for x in ce[r].tolist()]
                for r, level in enumerate(levels)
            },
            "balanced_accuracy_by_level": {
                str(level): [float(x) for x in balanced_accuracy[r].tolist()]
                for r, level in enumerate(levels)
            },
        }
    finally:
        model.train(was_training)
        _restore_rng_state(rng_state)


def clustering_score_from_features(
    original: torch.Tensor,
    synonym: torch.Tensor,
    non_synonym: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``1 - d_synonym / d_non_synonym`` after feature standardization."""
    if original.shape != synonym.shape or original.shape != non_synonym.shape:
        raise ValueError("clustering feature tensors must have identical shapes")
    if original.ndim != 3:
        raise ValueError("clustering features must have shape [layers, examples, channels]")
    x = original.to(torch.float64)
    x_syn = synonym.to(torch.float64)
    x_non = non_synonym.to(torch.float64)
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    x = (x - mean) / std
    x_syn = (x_syn - mean) / std
    x_non = (x_non - mean) / std
    d_syn = (x - x_syn).square().mean(dim=-1).mean(dim=-1)
    d_non = (x - x_non).square().mean(dim=-1).mean(dim=-1)
    return 1.0 - d_syn / d_non.clamp_min(eps), d_syn, d_non


def _synonym_clustering(
    model: GPT,
    split: RHMSplit,
    rules: TensorDict,
    *,
    cfg: ExperimentConfig,
    settings: DiagnosticsConfig,
    levels: list[int],
    positions: list[int],
    original_features: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    R, J, _, _ = original_features.shape
    scores = torch.empty(R, J, dtype=torch.float64)
    synonym_distances = torch.empty(R, J, dtype=torch.float64)
    generic_distances = torch.empty(R, J, dtype=torch.float64)
    for r_index, (level, position) in enumerate(zip(levels, positions)):
        synonym = synonym_counterfactual(
            split,
            rules,
            L=cfg.rhm.L,
            s=cfg.rhm.s,
            abstraction_level=level,
            seed=settings.seed + 10_000 * level,
        )
        synonym_features = _features_at_positions(
            model,
            synonym.leaves,
            [position],
            batch_size=cfg.train.batch_size,
            device=device,
        )[0]
        labels = latent_labels(split, L=cfg.rhm.L, abstraction_level=level)
        pairing = non_synonym_pairing(labels, seed=settings.seed + 20_000 * level)
        score, d_syn, d_generic = clustering_score_from_features(
            original_features[r_index],
            synonym_features,
            original_features[r_index][:, pairing, :],
            eps=settings.eps,
        )
        scores[r_index], synonym_distances[r_index], generic_distances[r_index] = (
            score,
            d_syn,
            d_generic,
        )
    return scores, synonym_distances, generic_distances


def _variable_sensitivity(
    model: GPT,
    split: RHMSplit,
    rules: TensorDict,
    *,
    cfg: ExperimentConfig,
    settings: DiagnosticsConfig,
    levels: list[int],
    positions: list[int],
    original_features: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure latent-replacement distance against the same negative pairing."""
    R, J, _, _ = original_features.shape
    distances = torch.empty(R, J, dtype=torch.float64)
    generic_distances = torch.empty(R, J, dtype=torch.float64)
    for r_index, (level, position) in enumerate(zip(levels, positions)):
        variable = variable_counterfactual(
            split,
            rules,
            L=cfg.rhm.L,
            s=cfg.rhm.s,
            abstraction_level=level,
            seed=settings.seed + 30_000 * level,
        )
        variable_features = _features_at_positions(
            model,
            variable.leaves,
            [position],
            batch_size=cfg.train.batch_size,
            device=device,
        )[0]
        labels = latent_labels(split, L=cfg.rhm.L, abstraction_level=level)
        pairing = non_synonym_pairing(labels, seed=settings.seed + 20_000 * level)
        _, d_variable, d_generic = clustering_score_from_features(
            original_features[r_index],
            variable_features,
            original_features[r_index][:, pairing, :],
            eps=settings.eps,
        )
        distances[r_index], generic_distances[r_index] = d_variable, d_generic
    return distances, generic_distances


def run_latent_diagnostics(
    model: GPT,
    split: RHMSplit,
    rules: TensorDict,
    cfg: ExperimentConfig,
    device: torch.device,
    *, settings: DiagnosticsConfig | None = None,
) -> dict[str, Any]:
    """Run configured diagnostics without changing model parameters or RNG."""
    settings = settings or DiagnosticsConfig()
    settings.validate()
    if cfg.rhm.L < 2:
        raise ValueError("latent diagnostics require RHM depth L >= 2")
    if settings.synonym_clustering and cfg.rhm.m < 2:
        raise ValueError("synonym clustering requires at least two productions per latent")

    n = min(int(settings.num_sequences), int(split.leaves.shape[0]))
    if n < 4:
        raise ValueError("diagnostic split must provide at least four examples")
    diagnostic_split = slice_rhm_split(split, n)
    levels = list(range(1, cfg.rhm.L))
    positions = [
        latent_location(L=cfg.rhm.L, s=cfg.rhm.s, abstraction_level=level).completion_position
        for level in levels
    ]

    rng_state = _capture_rng_state()
    was_training = model.training
    try:
        model.eval()
        original_features = _features_at_positions(
            model,
            diagnostic_split.leaves,
            positions,
            batch_size=cfg.train.batch_size,
            device=device,
        )
        labels = torch.stack(
            [latent_labels(diagnostic_split, L=cfg.rhm.L, abstraction_level=level) for level in levels],
            dim=0,
        )
        output: dict[str, Any] = {
            "num_sequences": n,
            "levels": levels,
            "positions": positions,
            "layer_convention": "0=embedding stream; j>0=post Transformer block j; final LN excluded",
        }
        if settings.linear_probe:
            (
                accuracy,
                ce,
                ordinary_majority_accuracy,
                balanced_majority_accuracy,
                represented_classes,
                balanced_accuracy,
                fit_size,
                eval_size,
            ) = _fit_linear_probes(
                original_features,
                labels,
                vocab_size=cfg.rhm.v,
                steps=settings.probe_steps,
                learning_rate=settings.probe_lr,
                seed=settings.seed,
                device=device,
                eps=settings.eps,
            )
            probe: dict[str, Any] = {
                "uniform_random_accuracy": 1.0 / cfg.rhm.v,
                "ordinary_majority_accuracy_by_level": {
                    str(level): float(ordinary_majority_accuracy[r])
                    for r, level in enumerate(levels)
                },
                "represented_classes_by_level": {
                    str(level): int(represented_classes[r])
                    for r, level in enumerate(levels)
                },
                "balanced_majority_accuracy_by_level": {
                    str(level): float(balanced_majority_accuracy[r])
                    for r, level in enumerate(levels)
                },
                "probe_steps": settings.probe_steps,
                "probe_lr": settings.probe_lr,
                "fit_examples": fit_size,
                "eval_examples": eval_size,
                "by_level": {},
            }
            for r_index, level in enumerate(levels):
                probe["by_level"][str(level)] = {
                    "completion_position": positions[r_index],
                    "accuracy_by_layer": [float(x) for x in accuracy[r_index].tolist()],
                    "balanced_accuracy_by_layer": [
                        float(x) for x in balanced_accuracy[r_index].tolist()
                    ],
                    "ordinary_majority_accuracy": float(ordinary_majority_accuracy[r_index]),
                    "represented_classes": int(represented_classes[r_index]),
                    "balanced_majority_accuracy": float(
                        balanced_majority_accuracy[r_index]
                    ),
                    "ce_by_layer": [float(x) for x in ce[r_index].tolist()],
                }
            output["linear_probe"] = probe
        if settings.synonym_clustering:
            score, d_syn, d_generic = _synonym_clustering(
                model,
                diagnostic_split,
                rules,
                cfg=cfg,
                settings=settings,
                levels=levels,
                positions=positions,
                original_features=original_features,
                device=device,
            )
            clustering: dict[str, Any] = {
                "definition": "1 - standardized synonym MSE / standardized non-synonym MSE",
                "reference": "unrelated in-distribution example with a different level-r latent",
                "by_level": {},
            }
            for r_index, level in enumerate(levels):
                clustering["by_level"][str(level)] = {
                    "completion_position": positions[r_index],
                    "score_by_layer": [float(x) for x in score[r_index].tolist()],
                    "synonym_distance_by_layer": [float(x) for x in d_syn[r_index].tolist()],
                    "non_synonym_distance_by_layer": [float(x) for x in d_generic[r_index].tolist()],
                }
            output["synonym_clustering"] = clustering
            variable_distance, variable_generic = _variable_sensitivity(
                model,
                diagnostic_split,
                rules,
                cfg=cfg,
                settings=settings,
                levels=levels,
                positions=positions,
                original_features=original_features,
                device=device,
            )
            output["variable_sensitivity"] = {
                "definition": "standardized latent-replacement MSE / standardized non-synonym MSE",
                "reference": "the same unrelated in-distribution pairing used by synonym clustering",
                "by_level": {
                    str(level): {
                        "completion_position": positions[r_index],
                        "sensitivity_by_layer": [
                            float(x)
                            for x in (
                                variable_distance[r_index]
                                / variable_generic[r_index].clamp_min(settings.eps)
                            ).tolist()
                        ],
                        "variable_distance_by_layer": [
                            float(x) for x in variable_distance[r_index].tolist()
                        ],
                        "non_synonym_distance_by_layer": [
                            float(x) for x in variable_generic[r_index].tolist()
                        ],
                    }
                    for r_index, level in enumerate(levels)
                },
            }
        return output
    finally:
        model.train(was_training)
        _restore_rng_state(rng_state)
