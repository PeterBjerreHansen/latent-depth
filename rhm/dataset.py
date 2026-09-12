"""Dataset wrappers and split construction for RHM language-model experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import Dataset

from .random_hierarchy_model import TensorDict, sample_rules, sample_trees


class LeafSequenceDataset(Dataset):
    """A fixed finite dataset of raw RHM leaf-token sequences."""

    def __init__(self, leaves: torch.Tensor):
        if leaves.ndim != 2:
            raise ValueError("leaves must have shape [num_examples, sequence_length]")
        if leaves.dtype != torch.long:
            leaves = leaves.long()
        self.leaves = leaves.contiguous()

    def __len__(self) -> int:
        return self.leaves.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.leaves[index]


@dataclass
class RHMSplit:
    trees: TensorDict
    choices: TensorDict

    @property
    def leaves(self) -> torch.Tensor:
        return self.trees[max(self.trees)]


@dataclass
class RHMBundle:
    rules: TensorDict
    train: RHMSplit
    val: RHMSplit
    test: RHMSplit


def _sample_split(num_data: int, rules: TensorDict, seed: int) -> RHMSplit:
    trees, choices = sample_trees(num_data, rules, seed=seed, return_choices=True)
    return RHMSplit(trees=trees, choices=choices)


def sample_leaf_sequences(num_data: int, rules: TensorDict, seed: int) -> torch.Tensor:
    """Sample a fresh pool of leaf sequences from fixed rules."""
    trees = sample_trees(num_data, rules, seed=seed)
    return trees[max(trees)]


def build_rhm_bundle(
    *,
    v: int,
    n: int,
    m: int,
    s: int,
    L: int,
    rule_seed: int,
    train_seed: int,
    val_seed: int,
    test_seed: int,
    train_size: int,
    val_size: int,
    test_size: int,
) -> RHMBundle:
    """Generate train/validation/test samples from one fixed grammar.

    The three splits use independent local sample RNG streams while sharing the
    exact same production rules. This is the intended experimental unit.
    """
    rules = sample_rules(v=v, n=n, m=m, s=s, L=L, seed=rule_seed)
    return RHMBundle(
        rules=rules,
        train=_sample_split(train_size, rules, train_seed),
        val=_sample_split(val_size, rules, val_seed),
        test=_sample_split(test_size, rules, test_seed),
    )


def slice_rhm_split(split: RHMSplit, count: int) -> RHMSplit:
    """Return the first ``count`` aligned derivations from an RHM split."""
    if count <= 0:
        raise ValueError("count must be positive")
    total = split.leaves.shape[0]
    if count > total:
        raise ValueError(f"requested {count} examples from split of size {total}")
    return RHMSplit(
        trees={level: tensor[:count].clone() for level, tensor in split.trees.items()},
        choices={level: tensor[:count].clone() for level, tensor in split.choices.items()},
    )
