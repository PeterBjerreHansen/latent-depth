"""Controlled RHM interventions used by representation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .dataset import RHMSplit
from .random_hierarchy_model import TensorDict


@dataclass(frozen=True)
class LatentLocation:
    abstraction_level: int
    tree_level: int
    completion_position: int


def latent_location(*, L: int, s: int, abstraction_level: int) -> LatentLocation:
    """Map a leaf-counted abstraction level to its causal completion point.

    ``r=1`` is the parent of visible leaves, ``r=2`` is its parent, and so on.
    The first level-r constituent completes at zero-based position ``s**r - 1``.
    """
    r = int(abstraction_level)
    if not 1 <= r < L:
        raise ValueError(f"abstraction_level must satisfy 1 <= r < L; got r={r}, L={L}")
    return LatentLocation(r, L - r, s**r - 1)


def latent_labels(split: RHMSplit, *, L: int, abstraction_level: int) -> torch.Tensor:
    """Return the identity of the first completed level-r constituent."""
    values = split.trees[L - int(abstraction_level)]
    if values.ndim == 1:
        return values.clone()
    return values[:, 0].clone()


def _local_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(seed))


def _different_uniform(
    values: torch.Tensor,
    cardinality: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if cardinality < 2:
        raise ValueError("a different categorical value requires cardinality >= 2")
    offsets = torch.randint(1, cardinality, values.shape, generator=generator, dtype=torch.long)
    return (values.long() + offsets) % cardinality


def synonym_counterfactual(
    split: RHMSplit,
    rules: TensorDict,
    *,
    L: int,
    s: int,
    abstraction_level: int,
    seed: int,
) -> RHMSplit:
    """Regenerate the first level-r constituent through a different synonym."""
    loc = latent_location(L=L, s=s, abstraction_level=abstraction_level)
    level0 = loc.tree_level
    if level0 not in split.choices:
        raise ValueError("selected latent has no recorded production choice")

    original_values = split.trees[level0]
    original_choices = split.choices[level0]
    if original_values.ndim != 2 or original_choices.ndim != 2:
        raise ValueError("synonym_counterfactual requires non-root per-position tensors")

    generator = _local_generator(seed)
    new_trees = {level: tensor.clone() for level, tensor in split.trees.items()}
    new_choices = {level: tensor.clone() for level, tensor in split.choices.items()}

    parent = original_values[:, 0:1].clone()
    old_rule = original_choices[:, 0:1]
    new_rule = _different_uniform(old_rule, int(rules[level0].shape[1]), generator)

    current = parent
    for level in range(level0, L):
        span_nodes = current.shape[1]
        if level == level0:
            chosen = new_rule
        else:
            chosen = torch.randint(
                0,
                int(rules[level].shape[1]),
                current.shape,
                generator=generator,
                dtype=torch.long,
            )
        new_choices[level][:, :span_nodes] = chosen
        current = rules[level][current, chosen].flatten(start_dim=1)
        new_trees[level + 1][:, : current.shape[1]] = current

    if not torch.equal(new_trees[level0][:, 0], split.trees[level0][:, 0]):
        raise RuntimeError("synonym intervention changed the selected latent")
    if torch.any(new_choices[level0][:, 0] == split.choices[level0][:, 0]):
        raise RuntimeError("synonym intervention failed to choose a different production")
    return RHMSplit(trees=new_trees, choices=new_choices)


def variable_counterfactual(
    split: RHMSplit,
    rules: TensorDict,
    *,
    L: int,
    s: int,
    abstraction_level: int,
    seed: int,
) -> RHMSplit:
    """Replace the first completed latent with a different symbol.

    The intervention preserves all preceding tree levels and all unrelated
    constituents, then regenerates only the selected latent's descendants.
    This is the matched sensitivity control for the synonym intervention:
    rule changes preserve the latent, while symbol changes do not.
    """
    loc = latent_location(L=L, s=s, abstraction_level=abstraction_level)
    level0 = loc.tree_level
    if level0 not in split.choices:
        raise ValueError("selected latent has no recorded production choice")

    original_values = split.trees[level0]
    original_choices = split.choices[level0]
    if original_values.ndim != 2 or original_choices.ndim != 2:
        raise ValueError("variable_counterfactual requires non-root per-position tensors")

    generator = _local_generator(seed)
    new_trees = {level: tensor.clone() for level, tensor in split.trees.items()}
    new_choices = {level: tensor.clone() for level, tensor in split.choices.items()}

    old_symbol = original_values[:, 0:1]
    new_symbol = _different_uniform(old_symbol, int(rules[level0].shape[0]), generator)
    if torch.any(new_symbol == old_symbol):
        raise RuntimeError("variable intervention failed to choose a different latent")
    new_trees[level0][:, 0] = new_symbol[:, 0]

    current = new_symbol
    for level in range(level0, L):
        span_nodes = current.shape[1]
        chosen = torch.randint(
            0,
            int(rules[level].shape[1]),
            current.shape,
            generator=generator,
            dtype=torch.long,
        )
        new_choices[level][:, :span_nodes] = chosen
        current = rules[level][current, chosen].flatten(start_dim=1)
        new_trees[level + 1][:, : current.shape[1]] = current

    if torch.equal(new_trees[level0][:, 0], split.trees[level0][:, 0]):
        raise RuntimeError("variable intervention failed to change the selected latent")
    return RHMSplit(trees=new_trees, choices=new_choices)


def non_synonym_pairing(labels: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Pair each example with an in-distribution example of a different latent."""
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    n = int(labels.numel())
    if n < 2:
        raise ValueError("at least two examples are required")
    if torch.unique(labels).numel() < 2:
        raise ValueError("non-synonym pairing requires at least two represented latent classes")

    generator = _local_generator(seed)
    result = torch.empty(n, dtype=torch.long)
    all_indices = torch.arange(n, dtype=torch.long)
    for i in range(n):
        candidates = all_indices[labels != labels[i]]
        pick = torch.randint(0, candidates.numel(), (1,), generator=generator).item()
        result[i] = candidates[pick]
    if torch.any(labels[result] == labels):
        raise RuntimeError("non-synonym pairing produced an equal-latent pair")
    return result
