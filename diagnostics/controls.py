"""Focused controls for separating surface recoverability from abstraction.

The controls in this module operate on a frozen model and a fixed RHM split.
They deliberately do not train a Transformer.  The public functions return
ordinary Python data so the Stage-02 control runner can persist compact,
reviewable records without saving features or model states.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from config import ExperimentConfig
from rhm.dataset import RHMSplit, slice_rhm_split
from rhm.interventions import (
    latent_labels,
    latent_location,
    non_synonym_pairing,
    synonym_counterfactual,
    variable_counterfactual,
)
from rhm.random_hierarchy_model import TensorDict

from .latent import (
    clustering_score_from_features,
    extract_features_at_positions,
    fit_linear_probes,
)


def make_fixed_partition(
    num_examples: int, *, eval_examples: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one fixed fit pool and held-out evaluation index set."""
    if num_examples < 4:
        raise ValueError("control data must contain at least four examples")
    if not 0 < eval_examples < num_examples:
        raise ValueError("eval_examples must lie strictly between zero and num_examples")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(num_examples, generator=generator)
    return permutation[eval_examples:], permutation[:eval_examples]


def make_balanced_nested_fit_indices(
    labels: torch.Tensor,
    fit_pool: torch.Tensor,
    fit_sizes: Sequence[int],
    *,
    seed: int,
) -> dict[int, torch.Tensor]:
    """Make deterministic, nested, approximately balanced fitting subsets.

    The same held-out set can therefore be used for every sample size.  The
    subsets are balanced with respect to the supplied level labels, while
    preserving the original example indices.
    """
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    fit_pool = torch.as_tensor(fit_pool, dtype=torch.long).flatten().cpu()
    if fit_pool.numel() == 0:
        raise ValueError("fit_pool must be nonempty")
    sizes = sorted({int(size) for size in fit_sizes})
    if not sizes or sizes[0] <= 0 or sizes[-1] > fit_pool.numel():
        raise ValueError("fit_sizes must be positive and fit within fit_pool")
    if torch.unique(fit_pool).numel() != fit_pool.numel():
        raise ValueError("fit_pool indices must be unique")
    if torch.any(fit_pool < 0) or torch.any(fit_pool >= labels.numel()):
        raise ValueError("fit_pool indices lie outside labels")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    classes = torch.unique(labels[fit_pool], sorted=True)
    if classes.numel() < 2:
        raise ValueError("balanced fitting requires at least two represented classes")
    per_class: list[torch.Tensor] = []
    for value in classes:
        indices = fit_pool[labels[fit_pool] == value]
        order = torch.randperm(indices.numel(), generator=generator)
        per_class.append(indices[order])

    # Round-robin selection gives every represented class a chance to enter
    # each nested prefix before a class receives a second example.
    order: list[int] = []
    offsets = [0] * len(per_class)
    while len(order) < fit_pool.numel():
        advanced = False
        for class_index, indices in enumerate(per_class):
            offset = offsets[class_index]
            if offset < indices.numel():
                order.append(int(indices[offset]))
                offsets[class_index] += 1
                advanced = True
        if not advanced:
            break
    nested = torch.tensor(order, dtype=torch.long)
    return {size: nested[:size].clone() for size in sizes}


def _classification_metrics(
    predictions: torch.Tensor, labels: torch.Tensor, *, vocab_size: int
) -> dict[str, float | int]:
    predictions = predictions.detach().cpu().long().flatten()
    labels = labels.detach().cpu().long().flatten()
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have the same shape")
    counts = torch.bincount(labels, minlength=vocab_size).to(torch.float64)
    present = counts > 0
    correct = predictions == labels
    correct_by_class = torch.bincount(
        labels, weights=correct.to(torch.float64), minlength=vocab_size
    )
    return {
        "accuracy": float(correct.to(torch.float64).mean()),
        "balanced_accuracy": float(
            (correct_by_class[present] / counts[present]).mean()
        ),
        "represented_classes": int(present.sum()),
        "balanced_majority_accuracy": float(1.0 / present.sum()),
    }


def linear_probe_curve(
    features: torch.Tensor,
    labels: torch.Tensor,
    fit_indices: Mapping[int, torch.Tensor],
    eval_indices: torch.Tensor,
    *,
    vocab_size: int,
    steps: int,
    learning_rate: float,
    probe_seed: int,
    device: torch.device,
    eps: float = 1e-8,
) -> list[dict[str, Any]]:
    """Evaluate a linear probe over fixed held-out data and fit sizes.

    ``features`` has shape ``[layers, examples, channels]``.  One row is
    returned for every fit-size/layer pair.
    """
    if features.ndim != 3 or labels.ndim != 1:
        raise ValueError("features must be [layers, examples, channels] and labels [examples]")
    if features.shape[1] != labels.numel():
        raise ValueError("features and labels have different example counts")
    rows: list[dict[str, Any]] = []
    for fit_size in sorted(fit_indices):
        (
            accuracy,
            ce,
            _ordinary_majority,
            balanced_majority,
            represented,
            balanced_accuracy,
            _fit_count,
            _eval_count,
        ) = fit_linear_probes(
            features.unsqueeze(0),
            labels.unsqueeze(0),
            vocab_size=vocab_size,
            steps=steps,
            learning_rate=learning_rate,
            seed=probe_seed,
            device=device,
            eps=eps,
            fit_indices=fit_indices[fit_size],
            eval_indices=eval_indices,
        )
        for layer in range(features.shape[0]):
            rows.append(
                {
                    "fit_examples": int(fit_size),
                    "observer_layer": int(layer),
                    "accuracy": float(accuracy[0, layer]),
                    "balanced_accuracy": float(balanced_accuracy[0, layer]),
                    "ce": float(ce[0, layer]),
                    "represented_classes": int(represented[0]),
                    "balanced_majority_accuracy": float(balanced_majority[0]),
                    "probe_seed": int(probe_seed),
                }
            )
    return rows


def leaf_prefix_one_hot(
    leaves: torch.Tensor, *, prefix_length: int, vocab_size: int
) -> torch.Tensor:
    """Encode a visible leaf prefix without using hidden tree labels."""
    if leaves.ndim != 2:
        raise ValueError("leaves must have shape [examples, sequence_length]")
    if not 1 <= prefix_length <= leaves.shape[1]:
        raise ValueError("prefix_length must lie within the leaf sequence")
    return F.one_hot(
        leaves[:, :prefix_length].long(), num_classes=vocab_size
    ).reshape(leaves.shape[0], -1).to(torch.float32)


def _prefix_key(prefix: torch.Tensor, vocab_size: int) -> torch.Tensor:
    powers = torch.tensor(
        [vocab_size**index for index in range(prefix.shape[1])], dtype=torch.long
    )
    return (prefix.long() * powers).sum(dim=1)


def surface_lookup_curve(
    leaves: torch.Tensor,
    labels: torch.Tensor,
    fit_indices: Mapping[int, torch.Tensor],
    eval_indices: torch.Tensor,
    *,
    prefix_length: int,
    vocab_size: int,
) -> list[dict[str, Any]]:
    """Predict labels from exact visible-prefix lookup tables."""
    prefixes = leaves[:, :prefix_length]
    eval_keys = _prefix_key(prefixes[eval_indices], vocab_size)
    rows: list[dict[str, Any]] = []
    for fit_size in sorted(fit_indices):
        fit_idx = fit_indices[fit_size]
        keys = _prefix_key(prefixes[fit_idx], vocab_size)
        counts: dict[int, torch.Tensor] = {}
        for key, label in zip(keys.tolist(), labels[fit_idx].tolist()):
            bucket = counts.setdefault(int(key), torch.zeros(vocab_size, dtype=torch.int64))
            bucket[int(label)] += 1
        fit_majority = int(torch.bincount(labels[fit_idx], minlength=vocab_size).argmax())
        predictions_list: list[int] = []
        seen_list: list[bool] = []
        for key in eval_keys.tolist():
            bucket = counts.get(int(key))
            if bucket is None:
                predictions_list.append(fit_majority)
                seen_list.append(False)
            else:
                predictions_list.append(int(bucket.argmax()))
                seen_list.append(True)
        predictions = torch.tensor(predictions_list, dtype=torch.long)
        seen = torch.tensor(seen_list, dtype=torch.bool)
        metrics = _classification_metrics(predictions, labels[eval_indices], vocab_size=vocab_size)
        rows.append(
            {
                "baseline": "surface_lookup",
                "fit_examples": int(fit_size),
                "observer_layer": 0,
                "prefix_length": int(prefix_length),
                "lookup_coverage": float(seen.to(torch.float64).mean()),
                **metrics,
            }
        )
    return rows


def _seeded_mlp(
    input_size: int, hidden_size: int, vocab_size: int, *, seed: int, device: torch.device
) -> nn.Module:
    state = torch.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, vocab_size),
        )
    finally:
        torch.set_rng_state(state)
    return model.to(device)


def surface_mlp_curve(
    features: torch.Tensor,
    labels: torch.Tensor,
    fit_indices: Mapping[int, torch.Tensor],
    eval_indices: torch.Tensor,
    *,
    vocab_size: int,
    hidden_size: int,
    steps: int,
    learning_rate: float,
    probe_seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Fit a fixed-size MLP using only one-hot visible-prefix features."""
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.numel():
        raise ValueError("surface MLP features and labels are mis-shaped")
    rows: list[dict[str, Any]] = []
    for fit_size in sorted(fit_indices):
        model = _seeded_mlp(
            features.shape[1], hidden_size, vocab_size, seed=probe_seed, device=device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        x_fit = features[fit_indices[fit_size]].to(device)
        y_fit = labels[fit_indices[fit_size]].to(device)
        x_eval = features[eval_indices].to(device)
        y_eval = labels[eval_indices].to(device)
        model.train()
        for _ in range(int(steps)):
            loss = F.cross_entropy(model(x_fit), y_fit)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x_eval)
            predictions = logits.argmax(dim=-1)
            metrics = _classification_metrics(predictions, y_eval, vocab_size=vocab_size)
            metrics["ce"] = float(F.cross_entropy(logits, y_eval).cpu())
        rows.append(
            {
                "baseline": "surface_mlp",
                "fit_examples": int(fit_size),
                "observer_layer": 0,
                "probe_seed": int(probe_seed),
                **metrics,
            }
        )
        del model, optimizer
    return rows


def child_pair_oracle(
    split: RHMSplit,
    rules: TensorDict,
    *,
    L: int,
    s: int,
    abstraction_level: int,
) -> dict[str, Any]:
    """Evaluate the exact grammar lookup from the two child latent symbols."""
    location = latent_location(L=L, s=s, abstraction_level=abstraction_level)
    rule_table = rules[location.tree_level]
    if rule_table.ndim != 3 or rule_table.shape[2] != 2:
        raise ValueError("child-pair oracle requires binary hidden productions")
    lookup: dict[tuple[int, int], int] = {}
    for parent in range(rule_table.shape[0]):
        for rule in range(rule_table.shape[1]):
            pair = tuple(int(value) for value in rule_table[parent, rule].tolist())
            if pair in lookup and lookup[pair] != parent:
                raise ValueError("grammar is ambiguous for child-pair oracle")
            lookup[pair] = parent

    children = split.trees[location.tree_level + 1][:, :2]
    labels = latent_labels(split, L=L, abstraction_level=abstraction_level)
    predictions = torch.tensor(
        [lookup[tuple(int(value) for value in pair)] for pair in children.tolist()],
        dtype=torch.long,
    )
    metrics = _classification_metrics(predictions, labels, vocab_size=rule_table.shape[0])
    return {
        "level": int(abstraction_level),
        "input": "true_child_latents",
        "num_examples": int(labels.numel()),
        **metrics,
    }


def repeated_invariance_diagnostics(
    model: nn.Module,
    split: RHMSplit,
    rules: TensorDict,
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    levels: Sequence[int],
    num_sequences: int,
    replicates: int,
    seed: int,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Repeat synonym/variable interventions and return C/Q observations."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    diagnostic_split = slice_rhm_split(split, int(num_sequences))
    level_list = [int(level) for level in levels]
    positions = [
        latent_location(
            L=cfg.rhm.L, s=cfg.rhm.s, abstraction_level=level
        ).completion_position
        for level in level_list
    ]
    was_training = model.training
    model.eval()
    try:
        original = extract_features_at_positions(
            model,
            diagnostic_split.leaves,
            positions,
            batch_size=cfg.train.batch_size,
            device=device,
        )
        labels_by_level = {
            level: latent_labels(
                diagnostic_split, L=cfg.rhm.L, abstraction_level=level
            )
            for level in level_list
        }
        records: list[dict[str, Any]] = []
        for replicate in range(int(replicates)):
            for level_index, level in enumerate(level_list):
                level_seed = int(seed) + 100_000 * replicate + 1_000 * level
                synonym = synonym_counterfactual(
                    diagnostic_split,
                    rules,
                    L=cfg.rhm.L,
                    s=cfg.rhm.s,
                    abstraction_level=level,
                    seed=level_seed + 1,
                )
                variable = variable_counterfactual(
                    diagnostic_split,
                    rules,
                    L=cfg.rhm.L,
                    s=cfg.rhm.s,
                    abstraction_level=level,
                    seed=level_seed + 2,
                )
                synonym_features = extract_features_at_positions(
                    model,
                    synonym.leaves,
                    [positions[level_index]],
                    batch_size=cfg.train.batch_size,
                    device=device,
                )[0]
                variable_features = extract_features_at_positions(
                    model,
                    variable.leaves,
                    [positions[level_index]],
                    batch_size=cfg.train.batch_size,
                    device=device,
                )[0]
                labels = labels_by_level[level]
                pairing = non_synonym_pairing(labels, seed=level_seed + 3)
                score, d_synonym, d_non_synonym = clustering_score_from_features(
                    original[level_index],
                    synonym_features,
                    original[level_index][:, pairing, :],
                    eps=eps,
                )
                _, d_variable, _ = clustering_score_from_features(
                    original[level_index],
                    variable_features,
                    original[level_index][:, pairing, :],
                    eps=eps,
                )
                q = (d_variable - d_synonym) / d_non_synonym.clamp_min(eps)
                for layer in range(original.shape[1]):
                    records.append(
                        {
                            "replicate": int(replicate),
                            "level": int(level),
                            "observer_layer": int(layer),
                            "clustering": float(score[layer]),
                            "q": float(q[layer]),
                            "synonym_distance": float(d_synonym[layer]),
                            "variable_distance": float(d_variable[layer]),
                            "non_synonym_distance": float(d_non_synonym[layer]),
                        }
                    )
        return {
            "num_sequences": int(num_sequences),
            "replicates": int(replicates),
            "levels": level_list,
            "positions": positions,
            "records": records,
        }
    finally:
        model.train(was_training)
