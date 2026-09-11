from itertools import product

import torch

from rhm.random_hierarchy_model import sample_rules, sample_trees


def _tuple_set(rule_tensor: torch.Tensor) -> set[tuple[int, ...]]:
    flat = rule_tensor.reshape(-1, rule_tensor.shape[-1])
    return {tuple(row.tolist()) for row in flat}


def test_rules_are_reproducible_and_unambiguous():
    kwargs = dict(v=16, n=16, m=4, s=2, L=3)
    a = sample_rules(**kwargs, seed=7)
    b = sample_rules(**kwargs, seed=7)
    c = sample_rules(**kwargs, seed=8)

    assert all(torch.equal(a[l], b[l]) for l in a)
    assert any(not torch.equal(a[l], c[l]) for l in a)

    for level, rules in a.items():
        # Every (child_1,...,child_s) tuple is used at most once at a level,
        # which is the RHM unambiguity condition.
        assert len(_tuple_set(rules)) == rules.shape[0] * rules.shape[1]


def test_tree_sampling_uses_seed_and_has_expected_shapes():
    rules = sample_rules(v=16, n=16, m=4, s=2, L=3, seed=0)
    trees_a, choices_a = sample_trees(37, rules, seed=123, return_choices=True)
    trees_b, choices_b = sample_trees(37, rules, seed=123, return_choices=True)
    trees_c, _ = sample_trees(37, rules, seed=124, return_choices=True)

    assert all(torch.equal(trees_a[l], trees_b[l]) for l in trees_a)
    assert all(torch.equal(choices_a[l], choices_b[l]) for l in choices_a)
    assert any(not torch.equal(trees_a[l], trees_c[l]) for l in trees_a)

    assert trees_a[0].shape == (37,)
    assert trees_a[1].shape == (37, 2)
    assert trees_a[2].shape == (37, 4)
    assert trees_a[3].shape == (37, 8)
    assert choices_a[0].shape == (37,)
    assert choices_a[1].shape == (37, 2)
    assert choices_a[2].shape == (37, 4)
    assert trees_a[3].dtype == torch.long
    assert int(trees_a[3].min()) >= 0
    assert int(trees_a[3].max()) < 16


def test_tiny_tree_expansion_matches_independent_path_enumerator():
    rules = sample_rules(v=4, n=4, m=2, s=2, L=2, seed=3)
    trees, choices = sample_trees(32, rules, seed=9, return_choices=True)

    # Re-expand each recorded derivation without using the generator's tensor
    # flattening operations. This is an independent oracle for level order,
    # child order, and the choice-index convention.
    for row in range(trees[0].shape[0]):
        symbols = [int(trees[0][row])]
        for level in range(2):
            expanded = []
            for symbol_index, symbol in enumerate(symbols):
                rule_index = int(choices[level][row].reshape(-1)[symbol_index])
                expanded.extend(rules[level][symbol, rule_index].tolist())
            symbols = expanded
            assert symbols == trees[level + 1][row].reshape(-1).tolist()

    # The tiny grammar has exactly 4 * 2^2 * 2^4 possible derivations.
    assert rules[0].shape == (4, 2, 2)
    assert rules[1].shape == (4, 2, 2)
    assert len(_tuple_set(rules[0])) == 8
    assert len(_tuple_set(rules[1])) == 8

    all_leaf_paths: set[tuple[int, ...]] = set()
    for root in range(4):
        for root_rule in range(2):
            children = rules[0][root, root_rule].tolist()
            for child_rules in product(range(2), repeat=2):
                leaves: list[int] = []
                for child, child_rule in zip(children, child_rules):
                    leaves.extend(rules[1][child, child_rule].tolist())
                all_leaf_paths.add(tuple(leaves))
    assert len(all_leaf_paths) == 32
    assert all(tuple(row.tolist()) in all_leaf_paths for row in trees[2])


def test_sampling_does_not_perturb_global_torch_rng():
    rules = sample_rules(v=8, n=8, m=2, s=2, L=2, seed=0)
    torch.manual_seed(987)
    expected = torch.rand(5)

    torch.manual_seed(987)
    sample_trees(20, rules, seed=111)
    observed = torch.rand(5)

    assert torch.equal(expected, observed)
