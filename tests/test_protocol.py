import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from config import ExperimentConfig, TargetDepthSweepConfig
from rhm.dataset import LeafSequenceDataset, build_rhm_bundle
from sweep_data_size import fixed_exposure_budget
from sweep_target_depth import arm_name
from training import evaluate_with_positions, load_model_from_checkpoint, make_loader, train_model


def _cfg() -> ExperimentConfig:
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
                "max_updates": 5,
                "grad_clip": 1.0,
                "eval_every_epochs": 1,
                "eval_every_updates": 2,
                "eval_at_start": True,
                "num_workers": 0,
                "device": "cpu",
                "deterministic": True,
                "deterministic_strict": True,
                "save_checkpoints": True,
                "checkpoint_every_updates": 3,
            },
            "model_seed": 5,
        }
    )


def _bundle(cfg: ExperimentConfig):
    return build_rhm_bundle(
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


def _datasets(bundle):
    return (
        LeafSequenceDataset(bundle.train.leaves),
        LeafSequenceDataset(bundle.val.leaves),
        LeafSequenceDataset(bundle.test.leaves),
    )


def test_update_based_evaluation_records_step_zero_fixed_steps_and_positions(tmp_path: Path):
    cfg = _cfg()
    bundle = _bundle(cfg)
    metrics = train_model(
        cfg,
        *_datasets(bundle),
        output_dir=tmp_path,
        rules=bundle.rules,
        verbose=False,
    )
    assert [row["global_step"] for row in metrics["history"]] == [0, 2, 4, 5]
    assert metrics["history"][0]["running_train_ce"] is None
    assert metrics["history"][0]["running_total_loss"] is None
    width = cfg.rhm.s**cfg.rhm.L - 1
    assert all(len(row["val_nll_by_position"]) == width for row in metrics["history"])


def test_selected_validation_metrics_are_recomputed_from_best_checkpoint(tmp_path: Path):
    cfg = _cfg()
    cfg.train.max_updates = 4
    bundle = _bundle(cfg)
    val_ds = LeafSequenceDataset(bundle.val.leaves)
    metrics = train_model(
        cfg,
        *_datasets(bundle),
        output_dir=tmp_path,
        rules=bundle.rules,
        verbose=False,
    )
    best_model, _, _ = load_model_from_checkpoint(tmp_path / "best.pt")
    loader = make_loader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
        seed=0,
        device=torch.device("cpu"),
    )
    val_ce, positions = evaluate_with_positions(best_model, loader, cfg, torch.device("cpu"))
    assert metrics["val_ce"] == pytest.approx(val_ce)
    assert metrics["selected_val_ce"] == pytest.approx(val_ce)
    assert metrics["val_nll_by_position"] == pytest.approx(positions)


def test_exact_update_checkpoint_written_even_between_evaluations(tmp_path: Path):
    cfg = _cfg()
    cfg.train.max_updates = 4
    bundle = _bundle(cfg)
    train_model(
        cfg,
        *_datasets(bundle),
        output_dir=tmp_path,
        rules=bundle.rules,
        verbose=False,
    )
    checkpoint_path = tmp_path / "checkpoints" / "step_00000003.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 3
    assert checkpoint["metrics"]["global_step"] == 3
    assert checkpoint["predictor"] is None


def test_fixed_exposure_budget_uses_effective_batch_size():
    updates, per_epoch = fixed_exposure_budget(
        train_size=32, batch_size=64, samples_per_example=8
    )
    assert per_epoch == 1
    assert updates == 8
    updates, per_epoch = fixed_exposure_budget(
        train_size=100, batch_size=32, samples_per_example=3
    )
    assert per_epoch == 4
    assert updates == 12


def test_target_depth_config_rejects_duplicates_and_out_of_range(tmp_path: Path):
    base = _cfg().to_dict()
    base["auxiliary"] = {
        "mode": "none",
        "target_layer": None,
        "weight": 0.1,
        "predictor_hidden_mult": 2,
        "seed": 54321,
    }
    path = tmp_path / "sweep.json"
    path.write_text(
        json.dumps(
            {
                "experiment": base,
                "target_layers": [None, 0, 0],
                "grammar_seeds": [0],
                "model_seeds": [0],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        TargetDepthSweepConfig.from_json(path)

    payload = json.loads(path.read_text())
    payload["target_layers"] = [None, 2]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        TargetDepthSweepConfig.from_json(path)

    payload["target_layers"] = [None, 1.5]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integers"):
        TargetDepthSweepConfig.from_json(path)


def test_disabled_auxiliary_mode_allows_zero_weight():
    cfg = _cfg()
    cfg.auxiliary.weight = 0.0
    cfg.validate()


def test_enabled_auxiliary_mode_requires_positive_weight():
    cfg = _cfg()
    cfg.auxiliary.mode = "next_latent"
    cfg.auxiliary.target_layer = 1
    cfg.auxiliary.weight = 0.0
    with pytest.raises(ValueError, match="positive"):
        cfg.validate()


def test_target_depth_arm_names_are_stable():
    assert arm_name(None) == "ntp"
    assert arm_name(0) == "target_0"
    assert arm_name(8) == "target_8"


def test_target_depth_sweep_cli_smoke_and_resume(tmp_path: Path):
    config_path = tmp_path / "target_depth.json"
    payload = {
        "experiment": _cfg().to_dict(),
        "target_layers": [None, 0, 1],
        "grammar_seeds": [0],
        "model_seeds": [0],
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    output_dir = tmp_path / "screen"
    command = [
        sys.executable,
        "sweep_target_depth.py",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    result = subprocess.run(
        command,
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    metrics_path = output_dir / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert {row["arm"] for row in rows} == {"ntp", "target_0", "target_1"}
    for arm in ("ntp", "target_0", "target_1"):
        run_dir = output_dir / "grammar_0" / "model_0" / arm
        assert (run_dir / "config.json").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "last.pt").exists()

    resumed = subprocess.run(
        command + ["--resume"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stdout + "\n" + resumed.stderr
    resumed_rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert resumed_rows == rows
