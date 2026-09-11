"""Random Hierarchy Model (RHM) data generation.

Adapted from fracagnetta/random-hierarchy-model, datasets/random_hierarchy_model.py
(origin and licensing documented in ../documents/PROVENANCE.md), under the MIT License.

The core generative process is intentionally kept faithful to upstream. Two small
changes are deliberate:
  1. RNGs are local rather than mutating global Python/PyTorch RNG state.
  2. sample_trees can optionally retain the chosen production-rule indices.

Those changes make paired/reproducible experiments safer without changing the
uniform RHM distribution.
"""

from __future__ import annotations

from itertools import product
import random
from typing import Dict, Optional, Tuple

import torch

TensorDict = Dict[int, torch.Tensor]


def _torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def sample_rules(v: int, n: int, m: int, s: int, L: int, seed: int = 42) -> TensorDict:
    """Sample an unambiguous RHM grammar.

    Args:
        v: Vocabulary size for all non-root levels (and terminals).
        n: Number of possible root symbols. Figure 2 uses n=v.
        m: Number of synonymous production rules per nonterminal.
        s: Branching factor / production tuple size.
        L: Number of production levels. Leaf sequence length is s**L.
        seed: Grammar seed.

    Returns:
        rules[l] with shape (n,m,s) for l=0 and (v,m,s) thereafter.
        Each child s-tuple is assigned to at most one parent within a level.
    """
    if min(v, n, m, s, L) <= 0:
        raise ValueError("v, n, m, s, and L must all be positive")
    num_tuples = v**s
    if n * m > num_tuples:
        raise ValueError(f"root requires n*m={n*m} tuples but only v**s={num_tuples} exist")
    if v * m > num_tuples:
        raise ValueError(
            f"unambiguous hidden-level rules require v*m <= v**s; got {v*m} > {num_tuples}"
        )

    tuples = list(product(range(v), repeat=s))
    rng = random.Random(int(seed))

    rules: TensorDict = {}
    rules[0] = torch.tensor(rng.sample(tuples, n * m), dtype=torch.long).reshape(n, m, s)
    for level in range(1, L):
        rules[level] = torch.tensor(rng.sample(tuples, v * m), dtype=torch.long).reshape(v, m, s)
    return rules


def sample_trees(
    num_data: int,
    rules: TensorDict,
    prior: Optional[torch.Tensor] = None,
    probs: Optional[TensorDict] = None,
    seed: int = 42,
    *,
    return_choices: bool = False,
) -> TensorDict | Tuple[TensorDict, TensorDict]:
    """Sample RHM derivation trees from a fixed grammar.

    This mirrors the upstream expansion logic. Under the default ``probs=None``,
    the root and each production rule are sampled uniformly.

    ``trees[0]`` contains roots; ``trees[len(rules)]`` contains visible leaves.
    When ``return_choices=True``, ``choices[l]`` records the production-rule
    index selected for every level-l symbol before it is expanded.
    """
    if num_data <= 0:
        raise ValueError("num_data must be positive")
    if not rules:
        raise ValueError("rules must be non-empty")

    L = len(rules)
    if sorted(rules) != list(range(L)):
        raise ValueError("rules must use consecutive integer levels 0..L-1")

    g = _torch_generator(seed)
    trees: TensorDict = {}
    choices: TensorDict = {}

    if prior is None:
        labels = torch.randint(0, rules[0].shape[0], (num_data,), generator=g, dtype=torch.long)
    else:
        if prior.ndim != 1 or prior.numel() != rules[0].shape[0]:
            raise ValueError("prior must be a 1D tensor matching the number of root symbols")
        labels = torch.multinomial(prior.float(), num_data, replacement=True, generator=g).long()
    trees[0] = labels.clone()

    for level in range(L):
        m = rules[level].shape[1]
        if probs is None:
            chosen_rule = torch.randint(0, m, labels.shape, generator=g, dtype=torch.long)
        else:
            p = probs[level]
            # Upstream's intended use is a common distribution over the m rules.
            if p.ndim != 1 or p.numel() != m:
                raise ValueError(f"probs[{level}] must be a length-{m} vector")
            chosen_rule = torch.multinomial(
                p.float(), labels.numel(), replacement=True, generator=g
            ).reshape(labels.shape).long()

        choices[level] = chosen_rule.clone()
        labels = rules[level][labels, chosen_rule].flatten(start_dim=1)
        trees[level + 1] = labels.clone()

    if return_choices:
        return trees, choices
    return trees


def sample_trees_unif(num_data: int, rules: TensorDict, seed: int = 42) -> TensorDict:
    """Compatibility wrapper for uniform sampling."""
    return sample_trees(num_data, rules, seed=seed)  # type: ignore[return-value]


def resample_rules(
    trees: TensorDict,
    rules: TensorDict,
    probs: Optional[TensorDict],
    level: int,
    position: int,
    p_resample: Optional[torch.Tensor] = None,
    seed: int = 42,
) -> TensorDict:
    """Resample one production choice and regenerate all descendants.

    Retained for future mechanistic diagnostics. The selected parent symbol is
    preserved; only its realization is changed.
    """
    L = len(rules)
    if not (0 <= level < L):
        raise ValueError("level must index a production level")
    if not (0 <= position < trees[level].shape[1] if trees[level].ndim > 1 else position == 0):
        raise ValueError("position out of range")

    g = _torch_generator(seed)
    new_trees = {l: trees[l].clone() for l in range(level + 1)}
    span = 1
    source = trees[level]
    if source.ndim == 1:
        source = source.unsqueeze(1)
    new_features = source[:, position : position + 1].clone()
    pos = position

    for l in range(level, L):
        m = rules[l].shape[1]
        if l == level and p_resample is not None:
            new_rule = torch.multinomial(
                p_resample.float(), new_features.numel(), replacement=True, generator=g
            ).reshape(new_features.shape)
        elif probs is None:
            new_rule = torch.randint(0, m, new_features.shape, generator=g)
        else:
            new_rule = torch.multinomial(
                probs[l].float(), new_features.numel(), replacement=True, generator=g
            ).reshape(new_features.shape)

        new_features = rules[l][new_features, new_rule].flatten(start_dim=1)
        pos *= rules[l].shape[2]
        span *= rules[l].shape[2]
        new_trees[l + 1] = trees[l + 1].clone()
        new_trees[l + 1][:, pos : pos + span] = new_features

    return new_trees


def resample_symbols(
    trees: TensorDict,
    rules: TensorDict,
    probs: Optional[TensorDict],
    level: int,
    position: int,
    p_resample: Optional[torch.Tensor] = None,
    seed: int = 42,
) -> TensorDict:
    """Replace one hidden symbol and regenerate all of its descendants."""
    L = len(rules)
    if not (0 <= level < L):
        raise ValueError("level must index a nonterminal level")
    g = _torch_generator(seed)

    new_trees = {l: trees[l].clone() for l in range(level + 1)}
    source = trees[level]
    width = 1 if source.ndim == 1 else source.shape[1]
    if not 0 <= position < width:
        raise ValueError("position out of range")

    batch = trees[0].shape[0]
    if p_resample is None:
        new_features = torch.randint(
            0, rules[level].shape[0], (batch, 1), generator=g, dtype=torch.long
        )
    else:
        new_features = torch.multinomial(
            p_resample.float(), batch, replacement=True, generator=g
        ).reshape(batch, 1).long()

    if new_trees[level].ndim == 1:
        new_trees[level] = new_features[:, 0].clone()
    else:
        new_trees[level][:, position] = new_features[:, 0]

    pos = position
    span = 1
    for l in range(level, L):
        m = rules[l].shape[1]
        if probs is None:
            new_rule = torch.randint(0, m, new_features.shape, generator=g)
        else:
            new_rule = torch.multinomial(
                probs[l].float(), new_features.numel(), replacement=True, generator=g
            ).reshape(new_features.shape)
        new_features = rules[l][new_features, new_rule].flatten(start_dim=1)
        pos *= rules[l].shape[2]
        span *= rules[l].shape[2]
        new_trees[l + 1] = trees[l + 1].clone()
        new_trees[l + 1][:, pos : pos + span] = new_features

    return new_trees


class RHM:
    """Convenience object mirroring the upstream RHM class."""

    def __init__(
        self,
        v: int,
        n: int,
        m: int,
        s: int,
        L: int,
        seed_rules: int,
        seed_samples: int,
        num_data: int,
        prior: Optional[torch.Tensor] = None,
        probs: Optional[TensorDict] = None,
        transform=None,
    ) -> None:
        self.vocab_size = v
        self.rules = sample_rules(v, n, m, s, L, seed=seed_rules)
        self.trees, self.choices = sample_trees(
            num_data,
            self.rules,
            prior=prior,
            probs=probs,
            seed=seed_samples,
            return_choices=True,
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.trees[0])
