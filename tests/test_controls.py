import torch

from config import ExperimentConfig
from diagnostics.controls import (
    child_pair_oracle,
    linear_probe_curve,
    make_balanced_nested_fit_indices,
    make_fixed_partition,
    repeated_invariance_diagnostics,
    surface_lookup_curve,
)
from rhm.dataset import RHMSplit
from rhm.random_hierarchy_model import sample_rules, sample_trees
from training import build_model


def test_control_partition_is_fixed_nested_and_balanced():
    labels = torch.arange(48) % 4
    fit_pool, eval_indices = make_fixed_partition(48, eval_examples=16, seed=11)
    subsets = make_balanced_nested_fit_indices(
        labels, fit_pool, [8, 16, 32], seed=12
    )

    assert not set(fit_pool.tolist()) & set(eval_indices.tolist())
    assert torch.equal(subsets[8], subsets[16][:8])
    assert torch.equal(subsets[16], subsets[32][:16])
    for indices in (subsets[8], subsets[16]):
        counts = torch.bincount(labels[indices], minlength=4)
        assert int(counts.max() - counts.min()) <= 1
    assert set(subsets[32].tolist()) == set(fit_pool.tolist())


def test_linear_probe_curve_accepts_a_fixed_evaluation_set():
    labels = torch.arange(32) % 4
    features = torch.nn.functional.one_hot(labels, num_classes=4).float()
    features = torch.stack([features, torch.zeros_like(features)], dim=0)
    fit_pool, eval_indices = make_fixed_partition(32, eval_examples=8, seed=13)
    fit_indices = make_balanced_nested_fit_indices(
        labels, fit_pool, [8, 16, 24], seed=14
    )
    rows = linear_probe_curve(
        features,
        labels,
        fit_indices,
        eval_indices,
        vocab_size=4,
        steps=30,
        learning_rate=0.1,
        probe_seed=15,
        device=torch.device("cpu"),
    )

    assert len(rows) == 2 * 3
    assert {row["fit_examples"] for row in rows} == {8, 16, 24}
    assert all(row["represented_classes"] >= 2 for row in rows)
    assert max(row["balanced_accuracy"] for row in rows if row["observer_layer"] == 0) > 0.8


def test_surface_lookup_uses_only_visible_prefixes():
    pairs = torch.tensor([[left, right] for left in range(4) for right in range(4)])
    leaves = pairs.repeat((2, 1))
    labels = ((pairs[:, 0] + 2 * pairs[:, 1]) % 4).repeat(2)
    fit_pool, eval_indices = make_fixed_partition(32, eval_examples=8, seed=16)
    fit_indices = make_balanced_nested_fit_indices(
        labels, fit_pool, [8, 16, 24], seed=17
    )
    rows = surface_lookup_curve(
        leaves,
        labels,
        fit_indices,
        eval_indices,
        prefix_length=2,
        vocab_size=4,
    )

    assert len(rows) == 3
    assert all(row["baseline"] == "surface_lookup" for row in rows)
    assert all(0.0 <= row["lookup_coverage"] <= 1.0 for row in rows)


def test_child_pair_oracle_is_exact_for_an_unambiguous_grammar():
    rules = sample_rules(v=4, n=4, m=2, s=2, L=3, seed=7)
    trees, choices = sample_trees(64, rules, seed=8, return_choices=True)
    split = RHMSplit(trees=trees, choices=choices)

    for level in (1, 2):
        result = child_pair_oracle(
            split, rules, L=3, s=2, abstraction_level=level
        )
        assert result["balanced_accuracy"] == 1.0
        assert result["input"] == "true_child_latents"


def test_repeated_invariance_diagnostics_return_one_row_per_measurement():
    cfg = ExperimentConfig.from_dict(
        {
            "rhm": {"v": 4, "n": 4, "m": 2, "s": 2, "L": 3},
            "data": {"train_size": 32, "val_size": 32, "test_size": 32},
            "model": {"n_layer": 2, "n_head": 2, "n_embd": 16},
            "train": {"batch_size": 8},
        }
    )
    rules = sample_rules(v=4, n=4, m=2, s=2, L=3, seed=9)
    trees, choices = sample_trees(32, rules, seed=10, return_choices=True)
    split = RHMSplit(trees=trees, choices=choices)
    model = build_model(cfg)
    result = repeated_invariance_diagnostics(
        model,
        split,
        rules,
        cfg,
        torch.device("cpu"),
        levels=[1, 2],
        num_sequences=16,
        replicates=2,
        seed=11,
    )

    assert len(result["records"]) == 2 * 2 * (cfg.model.n_layer + 1)
    assert {record["level"] for record in result["records"]} == {1, 2}
