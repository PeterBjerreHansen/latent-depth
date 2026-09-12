import random

import numpy as np
import torch
import torch.nn.functional as F

from config import ExperimentConfig
from diagnostics import clustering_score_from_features, run_latent_diagnostics, run_probe_control
from diagnostics.latent import _fit_linear_probes
from rhm.dataset import build_rhm_bundle, slice_rhm_split
from rhm.interventions import (
    latent_labels,
    latent_location,
    non_synonym_pairing,
    synonym_counterfactual,
    variable_counterfactual,
)
from training import build_model


def _cfg() -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "rhm": {
                "v": 8, "n": 8, "m": 2, "s": 2, "L": 3,
                "rule_seed": 7, "train_seed": 10, "val_seed": 20, "test_seed": 30,
            },
            "data": {"train_size": 32, "val_size": 32, "test_size": 32},
            "model": {"n_layer": 2, "n_head": 2, "n_embd": 16, "dropout": 0.0},
            "diagnostics": {
                "enabled": True, "linear_probe": True, "synonym_clustering": True,
                "every_evals": 1, "num_sequences": 16, "probe_steps": 8,
                "probe_lr": 0.01, "seed": 123,
            },
            "train": {"batch_size": 8, "max_epochs": 2, "max_updates": 2,
                      "device": "cpu", "deterministic": True},
            "model_seed": 5,
        }
    )


def _bundle(cfg: ExperimentConfig):
    return build_rhm_bundle(
        v=cfg.rhm.v, n=cfg.rhm.n, m=cfg.rhm.m, s=cfg.rhm.s, L=cfg.rhm.L,
        rule_seed=cfg.rhm.rule_seed, train_seed=cfg.rhm.train_seed,
        val_seed=cfg.rhm.val_seed, test_seed=cfg.rhm.test_seed,
        train_size=cfg.data.train_size, val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )


def test_latent_level_indexing_matches_causal_constituent_completion():
    cfg = _cfg()
    bundle = _bundle(cfg)
    loc1 = latent_location(L=3, s=2, abstraction_level=1)
    loc2 = latent_location(L=3, s=2, abstraction_level=2)
    assert (loc1.tree_level, loc1.completion_position) == (2, 1)
    assert (loc2.tree_level, loc2.completion_position) == (1, 3)
    assert torch.equal(latent_labels(bundle.val, L=3, abstraction_level=1), bundle.val.trees[2][:, 0])
    assert torch.equal(latent_labels(bundle.val, L=3, abstraction_level=2), bundle.val.trees[1][:, 0])


def test_synonym_counterfactual_preserves_latent_and_forces_new_production():
    cfg = _cfg()
    bundle = _bundle(cfg)
    split = slice_rhm_split(bundle.val, 16)
    r = 2
    tree_level = cfg.rhm.L - r
    transformed = synonym_counterfactual(
        split, bundle.rules, L=cfg.rhm.L, s=cfg.rhm.s, abstraction_level=r, seed=999
    )
    assert torch.equal(transformed.trees[tree_level][:, 0], split.trees[tree_level][:, 0])
    assert torch.all(transformed.choices[tree_level][:, 0] != split.choices[tree_level][:, 0])
    assert torch.all(
        (transformed.trees[tree_level + 1][:, : cfg.rhm.s]
         != split.trees[tree_level + 1][:, : cfg.rhm.s]).any(dim=1)
    )
    for level in range(tree_level + 1):
        assert torch.equal(transformed.trees[level], split.trees[level])


def test_variable_counterfactual_changes_only_selected_subtree():
    cfg = _cfg()
    bundle = _bundle(cfg)
    split = slice_rhm_split(bundle.val, 16)
    r = 2
    tree_level = cfg.rhm.L - r
    transformed = variable_counterfactual(
        split, bundle.rules, L=cfg.rhm.L, s=cfg.rhm.s, abstraction_level=r, seed=999
    )
    assert torch.all(transformed.trees[tree_level][:, 0] != split.trees[tree_level][:, 0])
    for level in range(tree_level):
        assert torch.equal(transformed.trees[level], split.trees[level])
    assert torch.equal(transformed.trees[tree_level][:, 1:], split.trees[tree_level][:, 1:])
    assert torch.all(
        (transformed.trees[cfg.rhm.L][:, : cfg.rhm.s ** r]
         != split.trees[cfg.rhm.L][:, : cfg.rhm.s ** r]).any(dim=1)
    )
    assert torch.equal(
        transformed.trees[cfg.rhm.L][:, cfg.rhm.s ** r :],
        split.trees[cfg.rhm.L][:, cfg.rhm.s ** r :],
    )


def test_non_synonym_pairing_always_changes_target_latent():
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    pairing = non_synonym_pairing(labels, seed=42)
    assert pairing.shape == labels.shape
    assert torch.all(labels[pairing] != labels)
    assert torch.equal(pairing, non_synonym_pairing(labels, seed=42))


def test_linear_probe_fitter_recovers_linearly_encoded_labels():
    vocab = 4
    n = 128
    labels = torch.arange(n) % vocab
    base = F.one_hot(labels, num_classes=vocab).float()
    features = torch.stack(
        [torch.stack([base, 2.0 * base, 0.5 * base], dim=0),
         torch.stack([1.5 * base, 0.75 * base, 3.0 * base], dim=0)], dim=0
    )
    probe_labels = torch.stack([labels, labels], dim=0)
    (
        accuracy,
        ce,
        majority_accuracy,
        balanced_majority_accuracy,
        represented_classes,
        balanced_accuracy,
        fit_size,
        eval_size,
    ) = _fit_linear_probes(
        features, probe_labels, vocab_size=vocab, steps=80, learning_rate=0.05,
        seed=123, device=torch.device("cpu")
    )
    assert fit_size == 64 and eval_size == 64
    assert torch.all(accuracy > 0.99)
    assert torch.all(ce < 0.2)
    assert torch.all(majority_accuracy >= 0.25)
    assert torch.all(majority_accuracy <= 1.0)
    assert torch.all(represented_classes == vocab)
    assert torch.allclose(
        balanced_majority_accuracy,
        torch.full((2,), 1.0 / vocab, dtype=balanced_majority_accuracy.dtype),
    )
    assert torch.all(balanced_accuracy > 0.99)


def test_linear_probe_standardization_removes_feature_scale_effect():
    torch.manual_seed(17)
    features = torch.randn(1, 2, 32, 4)
    labels = (torch.arange(32) % 4).unsqueeze(0)
    scales = torch.tensor([0.1, 2.0, 7.0, 30.0])
    scaled_features = features * scales

    unscaled = _fit_linear_probes(
        features,
        labels,
        vocab_size=4,
        steps=12,
        learning_rate=0.03,
        seed=91,
        device=torch.device("cpu"),
    )
    scaled = _fit_linear_probes(
        scaled_features,
        labels,
        vocab_size=4,
        steps=12,
        learning_rate=0.03,
        seed=91,
        device=torch.device("cpu"),
    )

    for left, right in zip(unscaled[:4], scaled[:4]):
        torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)


def test_clustering_score_has_expected_endpoints():
    original = torch.randn(3, 40, 6)
    non_synonym = original.roll(shifts=1, dims=1)
    perfect, d_syn, d_non = clustering_score_from_features(original, original, non_synonym)
    assert torch.allclose(perfect, torch.ones_like(perfect), atol=1e-12, rtol=0)
    assert torch.allclose(d_syn, torch.zeros_like(d_syn), atol=1e-12, rtol=0)
    assert torch.all(d_non > 0)
    none, _, _ = clustering_score_from_features(original, non_synonym, non_synonym)
    assert torch.allclose(none, torch.zeros_like(none), atol=1e-12, rtol=0)


def test_full_diagnostics_preserve_backbone_rng_and_mode():
    cfg = _cfg()
    bundle = _bundle(cfg)
    torch.manual_seed(cfg.model_seed)
    model = build_model(cfg)
    model.train()
    before = {name: p.detach().clone() for name, p in model.state_dict().items()}
    random.seed(111)
    np.random.seed(222)
    torch.manual_seed(333)
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    result = run_latent_diagnostics(model, bundle.val, bundle.rules, cfg, torch.device("cpu"))
    assert model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0.0, rtol=0.0)
    assert random.getstate() == py_state
    current_np = np.random.get_state()
    assert current_np[0] == np_state[0]
    assert np.array_equal(current_np[1], np_state[1])
    assert current_np[2:] == np_state[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_state, atol=0.0, rtol=0.0)
    assert result["levels"] == [1, 2]
    assert result["positions"] == [1, 3]
    for level in ("1", "2"):
        assert len(result["linear_probe"]["by_level"][level]["accuracy_by_layer"]) == 3
        assert "ordinary_majority_accuracy" in result["linear_probe"]["by_level"][level]
        assert "majority_accuracy" not in result["linear_probe"]["by_level"][level]
        assert len(result["linear_probe"]["by_level"][level]["balanced_accuracy_by_layer"]) == 3
        assert "balanced_majority_accuracy" in result["linear_probe"]["by_level"][level]
        assert len(result["synonym_clustering"]["by_level"][level]["score_by_layer"]) == 3
        assert len(result["variable_sensitivity"]["by_level"][level]["sensitivity_by_layer"]) == 3


def test_probe_controls_preserve_rng_and_report_uniform_reference():
    cfg = _cfg()
    bundle = _bundle(cfg)
    torch.manual_seed(cfg.model_seed)
    model = build_model(cfg).eval()
    before = torch.get_rng_state().clone()
    result = run_probe_control(model, bundle.val, cfg, torch.device("cpu"), shuffle_labels=True)
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0.0, rtol=0.0)
    assert result["shuffle_labels"] is True
    assert result["uniform_random_accuracy"] == 1.0 / cfg.rhm.v
    assert set(result["ordinary_majority_accuracy_by_level"]) == {"1", "2"}
    assert "majority_accuracy_by_level" not in result
    assert set(result["represented_classes_by_level"]) == {"1", "2"}
    assert set(result["balanced_majority_accuracy_by_level"]) == {"1", "2"}
    assert set(result["balanced_accuracy_by_level"]) == {"1", "2"}
    assert set(result["accuracy_by_level"]) == {"1", "2"}
