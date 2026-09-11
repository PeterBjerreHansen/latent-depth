"""Small analytical helpers from Cagnetta & Wyart (NeurIPS 2024)."""

from __future__ import annotations

import math


def sample_complexity(level: int, *, v: int, m: int, s: int) -> float:
    """Eq. (12): characteristic P_level for resolving level-wise correlations."""
    if level < 1:
        raise ValueError("level must be >= 1")
    denom = 1.0 - m / (v ** (s - 1))
    if denom <= 0:
        raise ValueError("requires m < v**(s-1) for nonzero hierarchical correlations")
    return v * (m ** (2 * level - 1)) / denom


def sample_complexities(L: int, *, v: int, m: int, s: int) -> list[float]:
    return [sample_complexity(level, v=v, m=m, s=s) for level in range(1, L + 1)]


def compatible_symbol_counts(L: int, *, v: int, m: int, s: int) -> list[float]:
    """Recurrence underlying the cross-entropy upper bounds in Eq. (11).

    Returns [N_0, N_1, ..., N_L], where N_0=v (no context) and N_l is the
    average number of compatible last-token symbols after exploiting level l.
    """
    if L < 0:
        raise ValueError("L must be nonnegative")
    if v <= 1 or m <= 0 or s <= 0:
        raise ValueError("invalid RHM parameters")
    denom = v**s - 1
    counts = [float(v)]
    if L == 0:
        return counts
    n_prev = 1.0 + (v - 1.0) * (m * v - 1.0) / denom
    counts.append(n_prev)
    for _ in range(2, L + 1):
        n_prev = 1.0 + (v - 1.0) * (m * n_prev - 1.0) / denom
        counts.append(n_prev)
    return counts


def loss_upper_bounds(L: int, *, v: int, m: int, s: int) -> list[float]:
    """log of compatible-symbol counts, including the no-context log(v) bound."""
    return [math.log(n) for n in compatible_symbol_counts(L, v=v, m=m, s=s)]
