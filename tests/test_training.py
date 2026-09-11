import copy
from pathlib import Path

import pytest
import torch

from config import ExperimentConfig
from diagnose import diagnose_checkpoint
from rhm.dataset import LeafSequenceDataset, build_rhm_bundle
from training import (
    build_model,
    evaluate_with_positions,
    load_model_from_checkpoint,
    load_training_state,
    make_loader,
    save_checkpoint,
    train_model,
)


def _tiny_cfg() -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "rhm": {
                "v": 8,
                "n": 8,
                "m": 2,
                "s": 2,
                "L": 2,
                "rule_seed": 0,
                "train_seed": 10,
                "val_seed": 20,
                "test_seed": 30,
            },
            "data": {"train_size": 32, "val_size": 32, "test_size": 32},
            "model": {"n_layer": 1, "n_head": 2, "n_embd": 32, "dropout": 0.0},
            "objective": {"mode": "next_token"},
            "optim": {
                "name": "adamw",
                "learning_rate": 0.001,
                "betas": [0.9, 0.95],
                "weight_decay": 0.0,
                "warmup_epochs": 0.0,
            },
            "train": {
                "batch_size": 16,
                "max_epochs": 10,
                "max_updates": 3,
                "grad_clip": 1.0,
                "eval_every_epochs": 1,
                "num_workers": 0,
                "device": "cpu",
                "deterministic": True,
                "save_checkpoints": True,
            },
            "model_seed": 5,
        }
    )


def test_checkpoint_roundtrip_preserves_logits(tmp_path: Path):
    cfg = _tiny_cfg()
    torch.manual_seed(3)
    model = build_model(cfg).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 8, (4, 3))
    before, _ = model(x)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        epoch=1,
        global_step=7,
        metrics={"val_ce": 1.0},
    )
    loaded, loaded_cfg, ckpt = load_model_from_checkpoint(path)
    loaded.eval()
    after, _ = loaded(x)

    torch.testing.assert_close(before, after, atol=0.0, rtol=0.0)
    assert loaded_cfg.to_dict() == cfg.to_dict()
    assert ckpt["global_step"] == 7


def test_end_to_end_training_smoke(tmp_path: Path):
    cfg = _tiny_cfg()
    bundle = build_rhm_bundle(
        v=cfg.rhm.v,
        n=cfg.rhm.n,
        m=cfg.rhm.m,
        s=cfg.rhm.s,
        L=cfg.rhm.L,
        rule_seed=cfg.rhm.rule_seed,
        train_seed=cfg.rhm.train_seed,
        val_seed=cfg.rhm.val_seed,
        test_seed=cfg.rhm.test_seed,
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    metrics = train_model(
        cfg,
        LeafSequenceDataset(bundle.train.leaves),
        LeafSequenceDataset(bundle.val.leaves),
        LeafSequenceDataset(bundle.test.leaves),
        output_dir=tmp_path,
        verbose=False,
    )

    assert metrics["global_step"] == 3
    assert len(metrics["history"]) == 2
    assert all(
        len(row["val_nll_by_position"]) == cfg.rhm.s**cfg.rhm.L - 1
        for row in metrics["history"]
    )
    assert len(metrics["val_nll_by_position"]) == cfg.rhm.s**cfg.rhm.L - 1
    assert len(metrics["test_nll_by_position"]) == cfg.rhm.s**cfg.rhm.L - 1
    assert "converged" not in metrics
    assert torch.isfinite(torch.tensor(metrics["test_ce"]))
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "metrics.json").exists()


def test_training_metrics_report_last_position_and_exposure_baselines(tmp_path: Path):
    cfg = _tiny_cfg()
    cfg.train.max_updates = 2
    bundle = build_rhm_bundle(
        v=cfg.rhm.v,
        n=cfg.rhm.n,
        m=cfg.rhm.m,
        s=cfg.rhm.s,
        L=cfg.rhm.L,
        rule_seed=cfg.rhm.rule_seed,
        train_seed=cfg.rhm.train_seed,
        val_seed=cfg.rhm.val_seed,
        test_seed=cfg.rhm.test_seed,
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    metrics = train_model(
        cfg,
        LeafSequenceDataset(bundle.train.leaves),
        LeafSequenceDataset(bundle.val.leaves),
        LeafSequenceDataset(bundle.test.leaves),
        output_dir=tmp_path,
        rules=bundle.rules,
        verbose=False,
    )

    assert metrics["val_last_position_nll"] == metrics["val_nll_by_position"][-1]
    assert metrics["test_last_position_nll"] == metrics["test_nll_by_position"][-1]
    assert metrics["uniform_baseline_nll"] == pytest.approx(torch.log(torch.tensor(float(cfg.rhm.v))).item())
    assert metrics["total_samples_seen"] == cfg.train.max_updates * cfg.train.batch_size
    assert metrics["total_tokens_seen"] == (
        metrics["total_samples_seen"] * (cfg.rhm.s**cfg.rhm.L - 1)
    )
    assert metrics["history"][-1]["samples_seen"] == metrics["total_samples_seen"]
    assert metrics["history"][-1]["tokens_seen"] == metrics["total_tokens_seen"]
    assert metrics["history"][-1]["val_last_position_nll"] == metrics["last_val_last_position_nll"]

    best_model, _, _ = load_model_from_checkpoint(tmp_path / "best.pt")
    best_model.train()
    val_loader = make_loader(
        LeafSequenceDataset(bundle.val.leaves),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
        seed=0,
        device=torch.device("cpu"),
    )
    selected_val_ce, selected_val_positions = evaluate_with_positions(
        best_model, val_loader, cfg, torch.device("cpu")
    )
    assert best_model.training
    assert metrics["val_ce"] == pytest.approx(selected_val_ce)
    assert metrics["selected_val_ce"] == pytest.approx(selected_val_ce)
    assert metrics["val_nll_by_position"] == pytest.approx(selected_val_positions)
    assert metrics["val_last_position_nll"] == pytest.approx(selected_val_positions[-1])


def test_update_based_evaluation_records_step_zero_and_fixed_steps(tmp_path: Path):
    cfg = _tiny_cfg()
    cfg.train.max_epochs = 10
    cfg.train.max_updates = 5
    cfg.train.eval_every_updates = 2
    cfg.train.eval_at_start = True
    cfg.train.save_checkpoints = False
    bundle = build_rhm_bundle(
        v=cfg.rhm.v,
        n=cfg.rhm.n,
        m=cfg.rhm.m,
        s=cfg.rhm.s,
        L=cfg.rhm.L,
        rule_seed=cfg.rhm.rule_seed,
        train_seed=cfg.rhm.train_seed,
        val_seed=cfg.rhm.val_seed,
        test_seed=cfg.rhm.test_seed,
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    metrics = train_model(
        cfg,
        LeafSequenceDataset(bundle.train.leaves),
        LeafSequenceDataset(bundle.val.leaves),
        LeafSequenceDataset(bundle.test.leaves),
        output_dir=tmp_path,
        verbose=False,
    )
    assert [row["global_step"] for row in metrics["history"]] == [0, 2, 4, 5]
    assert metrics["history"][0]["running_train_ce"] is None
    assert metrics["global_step"] == 5


def test_checkpoint_records_mps_rng_slot(tmp_path: Path):
    cfg = _tiny_cfg()
    bundle = build_rhm_bundle(
        v=cfg.rhm.v,
        n=cfg.rhm.n,
        m=cfg.rhm.m,
        s=cfg.rhm.s,
        L=cfg.rhm.L,
        rule_seed=cfg.rhm.rule_seed,
        train_seed=cfg.rhm.train_seed,
        val_seed=cfg.rhm.val_seed,
        test_seed=cfg.rhm.test_seed,
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    train_model(
        cfg,
        LeafSequenceDataset(bundle.train.leaves),
        LeafSequenceDataset(bundle.val.leaves),
        LeafSequenceDataset(bundle.test.leaves),
        output_dir=tmp_path,
        rules=bundle.rules,
        verbose=False,
    )
    checkpoint = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert "mps" in checkpoint["rng"]
    if torch.backends.mps.is_available():
        assert checkpoint["rng"]["mps"] is not None


def test_saturated_rhm_reports_no_hierarchical_theory_bound(tmp_path: Path):
    cfg = _tiny_cfg()
    cfg.rhm.v = 8
    cfg.rhm.n = 8
    cfg.rhm.m = 8
    cfg.train.max_updates = 1
    bundle = build_rhm_bundle(
        v=cfg.rhm.v,
        n=cfg.rhm.n,
        m=cfg.rhm.m,
        s=cfg.rhm.s,
        L=cfg.rhm.L,
        rule_seed=cfg.rhm.rule_seed,
        train_seed=cfg.rhm.train_seed,
        val_seed=cfg.rhm.val_seed,
        test_seed=cfg.rhm.test_seed,
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    metrics = train_model(
        cfg,
        LeafSequenceDataset(bundle.train.leaves),
        LeafSequenceDataset(bundle.val.leaves),
        LeafSequenceDataset(bundle.test.leaves),
        output_dir=tmp_path,
        rules=bundle.rules,
        verbose=False,
    )
    assert metrics["theory_last_token_nll_bounds"] is None


def test_training_state_restores_optimizer_rng_and_loader_state(tmp_path: Path):
    cfg = _tiny_cfg()
    bundle = build_rhm_bundle(
        v=cfg.rhm.v,
        n=cfg.rhm.n,
        m=cfg.rhm.m,
        s=cfg.rhm.s,
        L=cfg.rhm.L,
        rule_seed=cfg.rhm.rule_seed,
        train_seed=cfg.rhm.train_seed,
        val_seed=cfg.rhm.val_seed,
        test_seed=cfg.rhm.test_seed,
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    train_model(
        cfg,
        LeafSequenceDataset(bundle.train.leaves),
        LeafSequenceDataset(bundle.val.leaves),
        LeafSequenceDataset(bundle.test.leaves),
        output_dir=tmp_path,
        verbose=False,
    )

    path = tmp_path / "last.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    torch.set_rng_state(checkpoint["rng"]["torch"])
    expected_random = torch.rand(4)

    model, loaded_cfg, optimizer, loaded_checkpoint = load_training_state(path)
    actual_random = torch.rand(4)

    torch.testing.assert_close(expected_random, actual_random, atol=0.0, rtol=0.0)
    assert loaded_cfg.to_dict() == cfg.to_dict()
    assert optimizer.state
    assert loaded_checkpoint["loader_states"]["train_sampler"]["indices"].numel() > 0
    assert model.config.block_size == cfg.rhm.s**cfg.rhm.L

    continued_cfg = copy.deepcopy(cfg)
    continued_cfg.train.max_updates = 5
    continued_metrics = train_model(
        continued_cfg,
        LeafSequenceDataset(bundle.train.leaves),
        LeafSequenceDataset(bundle.val.leaves),
        LeafSequenceDataset(bundle.test.leaves),
        output_dir=tmp_path / "continued",
        resume_from=path,
        verbose=False,
    )
    assert continued_metrics["global_step"] == 5


def test_resume_rejects_mismatched_rhm_rules(tmp_path: Path):
    cfg = _tiny_cfg()
    bundle = build_rhm_bundle(
        v=cfg.rhm.v,
        n=cfg.rhm.n,
        m=cfg.rhm.m,
        s=cfg.rhm.s,
        L=cfg.rhm.L,
        rule_seed=cfg.rhm.rule_seed,
        train_seed=cfg.rhm.train_seed,
        val_seed=cfg.rhm.val_seed,
        test_seed=cfg.rhm.test_seed,
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    train_ds = LeafSequenceDataset(bundle.train.leaves)
    val_ds = LeafSequenceDataset(bundle.val.leaves)
    test_ds = LeafSequenceDataset(bundle.test.leaves)
    train_model(
        cfg,
        train_ds,
        val_ds,
        test_ds,
        output_dir=tmp_path,
        rules=bundle.rules,
        verbose=False,
    )

    mismatched_rules = {level: values.clone() for level, values in bundle.rules.items()}
    mismatched_rules[1].reshape(-1)[0] = (mismatched_rules[1].reshape(-1)[0] + 1) % cfg.rhm.v
    with pytest.raises(ValueError, match="RHM rules"):
        train_model(
            cfg,
            train_ds,
            val_ds,
            test_ds,
            output_dir=tmp_path / "resume",
            resume_from=tmp_path / "last.pt",
            rules=mismatched_rules,
            verbose=False,
        )


def test_online_and_offline_diagnostics_are_identical(tmp_path: Path):
    cfg = _tiny_cfg()
    cfg.rhm.L = 3
    cfg.model.n_layer = 2
    cfg.train.max_updates = 2
    cfg.diagnostics.enabled = True
    cfg.diagnostics.num_sequences = 16
    cfg.diagnostics.probe_steps = 3
    cfg.diagnostics.probe_lr = 0.01
    bundle = build_rhm_bundle(
        v=cfg.rhm.v,
        n=cfg.rhm.n,
        m=cfg.rhm.m,
        s=cfg.rhm.s,
        L=cfg.rhm.L,
        rule_seed=cfg.rhm.rule_seed,
        train_seed=cfg.rhm.train_seed,
        val_seed=cfg.rhm.val_seed,
        test_seed=cfg.rhm.test_seed,
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    metrics = train_model(
        cfg,
        LeafSequenceDataset(bundle.train.leaves),
        LeafSequenceDataset(bundle.val.leaves),
        LeafSequenceDataset(bundle.test.leaves),
        output_dir=tmp_path,
        diagnostic_split=bundle.val,
        rules=bundle.rules,
        verbose=False,
    )
    offline = diagnose_checkpoint(tmp_path / "last.pt", split="val", device="cpu")
    assert offline["diagnostics"] == metrics["history"][-1]["diagnostics"]


def _assert_nested_equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_nested_equal(x, y)
    else:
        assert a == b


def test_exact_resume_matches_uninterrupted_training(tmp_path: Path):
    cfg = _tiny_cfg()
    cfg.data.train_size = 48
    cfg.train.batch_size = 16
    cfg.train.max_updates = 4
    cfg.train.max_epochs = 4
    cfg.train.eval_every_updates = 2
    cfg.train.eval_at_start = True
    cfg.train.deterministic_strict = True
    bundle = build_rhm_bundle(
        v=cfg.rhm.v, n=cfg.rhm.n, m=cfg.rhm.m, s=cfg.rhm.s, L=cfg.rhm.L,
        rule_seed=cfg.rhm.rule_seed, train_seed=cfg.rhm.train_seed,
        val_seed=cfg.rhm.val_seed, test_seed=cfg.rhm.test_seed,
        train_size=cfg.data.train_size, val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    train_ds = LeafSequenceDataset(bundle.train.leaves)
    val_ds = LeafSequenceDataset(bundle.val.leaves)
    test_ds = LeafSequenceDataset(bundle.test.leaves)
    full_dir = tmp_path / "full"
    train_model(cfg, train_ds, val_ds, test_ds, output_dir=full_dir, rules=bundle.rules, verbose=False)
    first_cfg = copy.deepcopy(cfg)
    first_cfg.train.max_updates = 2
    split_dir = tmp_path / "split"
    train_model(first_cfg, train_ds, val_ds, test_ds, output_dir=split_dir, rules=bundle.rules, verbose=False)
    resumed_dir = tmp_path / "resumed"
    train_model(
        cfg, train_ds, val_ds, test_ds, output_dir=resumed_dir,
        resume_from=split_dir / "last.pt", rules=bundle.rules, verbose=False,
    )
    full = torch.load(full_dir / "last.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_dir / "last.pt", map_location="cpu", weights_only=False)
    _assert_nested_equal(full["model"], resumed["model"])
    _assert_nested_equal(full["optimizer"], resumed["optimizer"])
    _assert_nested_equal(full["loader_states"]["train_sampler"], resumed["loader_states"]["train_sampler"])
    assert full["global_step"] == resumed["global_step"] == 4


def test_diagnostics_do_not_change_training_trajectory(tmp_path: Path):
    cfg = _tiny_cfg()
    cfg.train.max_updates = 4
    cfg.train.max_epochs = 3
    cfg.train.deterministic_strict = True
    bundle = build_rhm_bundle(
        v=cfg.rhm.v, n=cfg.rhm.n, m=cfg.rhm.m, s=cfg.rhm.s, L=cfg.rhm.L,
        rule_seed=cfg.rhm.rule_seed, train_seed=cfg.rhm.train_seed,
        val_seed=cfg.rhm.val_seed, test_seed=cfg.rhm.test_seed,
        train_size=cfg.data.train_size, val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    train_ds = LeafSequenceDataset(bundle.train.leaves)
    val_ds = LeafSequenceDataset(bundle.val.leaves)
    test_ds = LeafSequenceDataset(bundle.test.leaves)
    plain_dir = tmp_path / "plain"
    train_model(cfg, train_ds, val_ds, test_ds, output_dir=plain_dir, rules=bundle.rules, verbose=False)
    diagnostic_cfg = copy.deepcopy(cfg)
    diagnostic_cfg.diagnostics.enabled = True
    diagnostic_cfg.diagnostics.num_sequences = 16
    diagnostic_cfg.diagnostics.probe_steps = 3
    diagnostic_cfg.diagnostics.probe_lr = 0.01
    metrics = train_model(
        diagnostic_cfg, train_ds, val_ds, test_ds,
        output_dir=tmp_path / "diagnostic", diagnostic_split=bundle.val,
        rules=bundle.rules, verbose=False,
    )
    plain = torch.load(plain_dir / "last.pt", map_location="cpu", weights_only=False)
    diagnostic = torch.load(tmp_path / "diagnostic" / "last.pt", map_location="cpu", weights_only=False)
    _assert_nested_equal(plain["model"], diagnostic["model"])
    assert any("diagnostics" in row for row in metrics["history"])


def test_periodic_checkpoint_is_self_contained_with_rules(tmp_path: Path):
    cfg = _tiny_cfg()
    cfg.train.checkpoint_every_evals = None
    cfg.train.checkpoint_every_updates = 2
    bundle = build_rhm_bundle(
        v=cfg.rhm.v, n=cfg.rhm.n, m=cfg.rhm.m, s=cfg.rhm.s, L=cfg.rhm.L,
        rule_seed=cfg.rhm.rule_seed, train_seed=cfg.rhm.train_seed,
        val_seed=cfg.rhm.val_seed, test_seed=cfg.rhm.test_seed,
        train_size=cfg.data.train_size, val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    train_model(
        cfg, LeafSequenceDataset(bundle.train.leaves), LeafSequenceDataset(bundle.val.leaves),
        LeafSequenceDataset(bundle.test.leaves), output_dir=tmp_path,
        rules=bundle.rules, verbose=False,
    )
    snapshots = sorted((tmp_path / "checkpoints").glob("step_*.pt"))
    assert snapshots
    assert [path.name for path in snapshots] == ["step_00000002.pt"]
    checkpoint = torch.load(snapshots[-1], map_location="cpu", weights_only=False)
    assert checkpoint["rules"] is not None
    for level in bundle.rules:
        assert torch.equal(checkpoint["rules"][level], bundle.rules[level])


def test_update_checkpoint_metrics_match_unmeasured_checkpoint_step(tmp_path: Path):
    cfg = _tiny_cfg()
    cfg.train.max_updates = 4
    cfg.train.max_epochs = 10
    cfg.train.eval_every_updates = 2
    cfg.train.checkpoint_every_updates = 3
    bundle = build_rhm_bundle(
        v=cfg.rhm.v, n=cfg.rhm.n, m=cfg.rhm.m, s=cfg.rhm.s, L=cfg.rhm.L,
        rule_seed=cfg.rhm.rule_seed, train_seed=cfg.rhm.train_seed,
        val_seed=cfg.rhm.val_seed, test_seed=cfg.rhm.test_seed,
        train_size=cfg.data.train_size, val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )

    train_model(
        cfg,
        LeafSequenceDataset(bundle.train.leaves),
        LeafSequenceDataset(bundle.val.leaves),
        LeafSequenceDataset(bundle.test.leaves),
        output_dir=tmp_path,
        rules=bundle.rules,
        verbose=False,
    )

    checkpoint = torch.load(
        tmp_path / "checkpoints" / "step_00000003.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["global_step"] == 3
    assert checkpoint["metrics"]["global_step"] == 3
    assert checkpoint["metrics"]["tokens_seen"] == 3 * cfg.train.batch_size * (cfg.rhm.s**cfg.rhm.L - 1)
