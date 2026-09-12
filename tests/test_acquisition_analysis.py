import copy
import json
from pathlib import Path

import pytest
import torch

from diagnostics.latent import _fit_linear_probes
from plot_trajectory import _heatmap, plot_target_depth_comparison
from summarize_target_depth import summarize_target_depth, summarize_target_depth_data
from summarize_trajectory import summarize_trajectory_data


RULE = {
    "persistence_checkpoints": 2,
    "accessibility": {
        "metric": "balanced_accuracy",
        "margin_over_step0": 0.05,
        "margin_over_shuffled": 0.05,
        "margin_over_balanced_baseline": 0.05,
    },
    "clustering": {"min_score": 0.05, "margin_over_step0": 0.05},
    "probe_milestones": [0.50, 0.75, 0.90],
    "epsilon": 1e-8,
}


def _vector_at(values: list[list[float]], index: int, layers: int = 3) -> list[float]:
    value = values[index]
    assert len(value) == layers
    return value


def _trajectory(
    steps=(0, 100, 200),
    *,
    balanced: dict[int, list[list[float]]] | None = None,
    clustering: dict[int, list[list[float]]] | None = None,
    distances: tuple[float, float, float] = (1.0, 3.0, 2.0),
) -> dict:
    balanced = balanced or {}
    clustering = clustering or {}
    records = []
    for index, step in enumerate(steps):
        by_level = {}
        shuffled_by_level = {}
        for level in (2, 3):
            a_values = balanced.get(level, [[0.1, 0.1, 0.1] for _ in steps])
            c_values = clustering.get(level, [[0.0, 0.0, 0.0] for _ in steps])
            by_level[str(level)] = {
                "completion_position": level,
                "accuracy_by_layer": _vector_at(a_values, index),
                "balanced_accuracy_by_layer": _vector_at(a_values, index),
                "ce_by_layer": [1.0, 1.0, 1.0],
            }
            shuffled_by_level[str(level)] = [0.1, 0.1, 0.1]
        diagnostics = {
            "num_sequences": 32,
            "levels": [2, 3],
            "positions": [2, 3],
            "linear_probe": {
                "uniform_random_accuracy": 1.0 / 16,
                "ordinary_majority_accuracy_by_level": {"2": 0.5, "3": 0.5},
                "majority_accuracy_by_level": {"2": 0.5, "3": 0.5},
                "represented_classes_by_level": {"2": 4, "3": 4},
                "balanced_majority_accuracy_by_level": {"2": 0.25, "3": 0.25},
                "by_level": by_level,
            },
            "synonym_clustering": {
                "by_level": {
                    str(level): {
                        "score_by_layer": _vector_at(
                            clustering.get(level, [[0.0, 0.0, 0.0] for _ in steps]), index
                        ),
                        "synonym_distance_by_layer": [distances[0]] * 3,
                        "non_synonym_distance_by_layer": [distances[2]] * 3,
                    }
                    for level in (2, 3)
                }
            },
            "variable_sensitivity": {
                "by_level": {
                    str(level): {
                        "variable_distance_by_layer": [distances[1]] * 3,
                        "non_synonym_distance_by_layer": [distances[2]] * 3,
                    }
                    for level in (2, 3)
                }
            },
        }
        records.append(
            {
                "checkpoint_global_step": step,
                "diagnostics": diagnostics,
                "probe_controls": {
                    "trained_backbone_shuffled_labels": {
                        "balanced_accuracy_by_level": shuffled_by_level
                    }
                },
            }
        )
    return {
        "run_dir": "synthetic",
        "split": "val",
        "num_sequences": 32,
        "probe_steps": 8,
        "checkpoints": records,
    }


def _with_onsets(steps=(0, 50, 100, 150, 200, 250), h2=100, h3=200):
    balanced = {}
    clustering = {}
    for level, onset in ((2, h2), (3, h3)):
        balanced[level] = [
            [0.9, 0.9, 0.9] if step >= onset else [0.1, 0.1, 0.1]
            for step in steps
        ]
        clustering[level] = [
            [0.2, 0.2, 0.2] if step >= onset else [0.0, 0.0, 0.0]
            for step in steps
        ]
    return _trajectory(steps, balanced=balanced, clustering=clustering)


def test_cross_layer_accessibility_and_clustering_do_not_form_acquisition():
    trajectory = _trajectory(
        balanced={2: [[0.1, 0.1, 0.1], [0.9, 0.1, 0.1], [0.9, 0.1, 0.1]]},
        clustering={2: [[0.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.2, 0.0]]},
    )
    summary = summarize_trajectory_data(trajectory, RULE)
    assert summary["levels"]["2"]["emergence"]["status"] == "censored"
    assert all(
        event["status"] == "censored"
        for event in summary["levels"]["2"]["layerwise_onsets"].values()
    )


def test_one_checkpoint_is_not_persistent():
    trajectory = _trajectory(
        balanced={2: [[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [0.1, 0.1, 0.1]]},
        clustering={2: [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]},
    )
    summary = summarize_trajectory_data(trajectory, RULE)
    assert summary["levels"]["2"]["emergence"]["status"] == "censored"


def test_same_layer_persistence_reports_onset_and_confirmation():
    trajectory = _trajectory(
        balanced={2: [[0.1, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.9, 0.1]]},
        clustering={2: [[0.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.2, 0.0]]},
    )
    summary = summarize_trajectory_data(trajectory, RULE)
    event = summary["levels"]["2"]["layerwise_onsets"]["1"]
    assert event == {"status": "observed", "onset_step": 100, "confirmed_step": 200}
    assert summary["levels"]["2"]["emergence"]["observer_layer"] == 1
    assert summary["levels"]["2"]["tau_AC"] == 100


def test_unreached_level_is_explicitly_censored():
    summary = summarize_trajectory_data(_trajectory(), RULE)
    level = summary["levels"]["3"]
    assert level["emergence"] == {"status": "censored", "through_step": 200}
    assert all(
        event == {"status": "censored", "through_step": 200}
        for event in level["layerwise_onsets"].values()
    )


def test_probe_milestones_are_independent_of_acquisition():
    trajectory = _trajectory(
        balanced={2: [[0.1, 0.1, 0.1], [0.6, 0.1, 0.1], [0.8, 0.1, 0.1]]}
    )
    summary = summarize_trajectory_data(trajectory, RULE)
    level = summary["levels"]["2"]
    assert level["probe_milestones"] == {"0.50": 100, "0.75": 200, "0.90": None}
    assert level["probe_milestone_status"]["0.90"] == {
        "status": "censored",
        "through_step": 200,
    }
    assert level["emergence"]["status"] == "censored"


def test_q_is_derived_from_stored_distances():
    summary = summarize_trajectory_data(
        _trajectory(distances=(1.0, 3.0, 2.0)), RULE
    )
    assert summary["levels"]["2"]["layerwise"]["0"]["q"] == pytest.approx(
        [1.0, 1.0, 1.0]
    )


def test_balanced_majority_uses_represented_classes_on_eval_split():
    labels = torch.tensor([[0] * 12 + [1] * 4 + [2] * 4])
    features = torch.zeros(1, 1, 20, 3)
    result = _fit_linear_probes(
        features,
        labels,
        vocab_size=4,
        steps=1,
        learning_rate=0.01,
        seed=123,
        device=torch.device("cpu"),
    )
    _, _, ordinary, balanced_majority, represented, _, _, eval_size = result
    generator = torch.Generator(device="cpu").manual_seed(123)
    eval_labels = labels[0, torch.randperm(20, generator=generator)[10:]]
    counts = torch.bincount(eval_labels, minlength=4)
    present = counts > 0
    assert int(represented[0]) == int(present.sum())
    assert float(ordinary[0]) == pytest.approx(float(counts.max() / eval_size))
    assert float(balanced_majority[0]) == pytest.approx(1.0 / int(present.sum()))


def _config(target_layer):
    return {
        "rhm": {
            "v": 16,
            "n": 16,
            "m": 4,
            "s": 2,
            "L": 5,
            "rule_seed": 0,
            "train_seed": 1000,
            "val_seed": 2000,
            "test_seed": 3000,
        },
        "data": {
            "train_size": 64,
            "val_size": 64,
            "test_size": 64,
            "resample_train_each_epoch": True,
        },
        "model": {"n_layer": 3, "n_head": 2, "n_embd": 16, "dropout": 0.0, "bias": True},
        "objective": {"mode": "next_token"},
        "auxiliary": {
            "mode": "none" if target_layer is None else "next_latent",
            "target_layer": target_layer,
            "weight": 0.1,
            "predictor_hidden_mult": 2,
            "seed": 54321,
        },
        "optim": {
            "name": "adamw",
            "learning_rate": 0.001,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "warmup_epochs": 0.0,
        },
        "diagnostics": {
            "enabled": True,
            "linear_probe": True,
            "synonym_clustering": True,
            "every_evals": 1,
            "num_sequences": 32,
            "probe_steps": 8,
            "probe_lr": 0.01,
            "seed": 123,
            "eps": 1e-8,
        },
        "train": {
            "batch_size": 16,
            "max_epochs": 5,
            "max_updates": 6,
            "grad_clip": 1.0,
            "eval_every_epochs": 1,
            "eval_every_updates": 1,
            "eval_at_start": True,
            "num_workers": 0,
            "device": "cpu",
            "deterministic": True,
            "deterministic_strict": True,
            "save_checkpoints": True,
            "checkpoint_every_updates": 1,
        },
        "model_seed": 0,
    }


def _arm(target_layer, summary, steps=(0, 50, 100, 150, 200, 250), *, best=1.0):
    history = [{"global_step": step, "val_ce": best + (250 - step) / 1000} for step in steps]
    metrics = {
        "arm": "ntp" if target_layer is None else f"target_{target_layer}",
        "auxiliary_target_layer": target_layer,
        "rule_seed": 0,
        "model_seed": 0,
        "best_val_ce": best,
        "history": history,
    }
    return {
        "run_dir": metrics["arm"],
        "arm": metrics["arm"],
        "target_layer": target_layer,
        "grammar_seed": 0,
        "model_seed": 0,
        "config": _config(target_layer),
        "metrics": metrics,
        "history": history,
        "summary": summary,
    }


def _paired_arms(ntp_summary, target_summary, steps=(0, 50, 100, 150, 200, 250), *, target_best=1.0):
    return [
        _arm(None, ntp_summary, steps=steps),
        *[
            _arm(layer, target_summary, steps=steps, best=target_best)
            for layer in range(4)
        ],
    ]


def test_target_depth_deltas_and_interval_keep_absolute_times():
    steps = (0, 50, 100, 150, 200, 250)
    ntp_summary = summarize_trajectory_data(_with_onsets(steps, h2=100, h3=200), RULE)
    target_summary = summarize_trajectory_data(_with_onsets(steps, h2=50, h3=150), RULE)
    result = summarize_target_depth_data(
        _paired_arms(ntp_summary, target_summary, target_best=0.99),
        primary_levels=[2, 3],
    )
    target = next(row for row in result["groups"][0]["primary"] if row["target"] == "target_2")
    assert target["tau_2"] == 50
    assert target["delta_tau_2"] == -50
    assert target["tau_3"] == 150
    assert target["delta_tau_3"] == -50
    assert target["tau_3_minus_tau_2"] == 100
    assert target["best_val_ce_delta_vs_ntp"] == pytest.approx(-0.01)
    assert target["levels"]["2"]["matched_validation_ce"]["delta_vs_ntp"] == pytest.approx(-0.01)
    assert len(result["validation_ce_by_step"]) == len(steps)


def test_censored_paired_comparisons_produce_bounds():
    steps = (0, 50, 100, 150, 200, 250)
    ntp_summary = summarize_trajectory_data(_with_onsets(steps, h2=50, h3=150), RULE)
    target_summary = summarize_trajectory_data(_with_onsets(steps, h2=50, h3=999), RULE)
    result = summarize_target_depth_data(
        _paired_arms(ntp_summary, target_summary), primary_levels=[2, 3]
    )
    target = next(row for row in result["groups"][0]["primary"] if row["target"] == "target_2")
    assert target["tau_3"] is None
    assert target["tau_3_status"] == "censored"
    assert target["delta_tau_3"] is None
    assert target["delta_tau_3_status"] == "lower_bound"
    assert target["delta_tau_3_bound"] == 100


def test_sweep_refuses_mismatched_evaluation_schedules():
    steps = (0, 50, 100, 150, 200, 250)
    ntp_summary = summarize_trajectory_data(_with_onsets(steps, h2=100, h3=200), RULE)
    target_summary = summarize_trajectory_data(_with_onsets((0, 100, 200), h2=100, h3=200), RULE)
    with pytest.raises(ValueError, match="checkpoint schedule"):
        summarize_target_depth_data(
            [
                _arm(None, ntp_summary),
                *[
                    _arm(
                        layer,
                        target_summary,
                        steps=(0, 100, 200) if layer == 2 else steps,
                    )
                    for layer in range(4)
                ],
            ],
            primary_levels=[2, 3],
        )


def test_target_depth_refuses_incomplete_grid():
    steps = (0, 50, 100, 150, 200, 250)
    summary = summarize_trajectory_data(_with_onsets(steps), RULE)
    with pytest.raises(ValueError, match="incomplete target-depth screen"):
        summarize_target_depth_data(
            [_arm(None, summary), _arm(2, summary)], primary_levels=[2, 3]
        )


def test_q_heatmap_helper_uses_stored_distances():
    trajectory = _trajectory()
    matrix, steps, layers = _heatmap(
        trajectory["checkpoints"], "q", ["2", "3"]
    )
    assert matrix.shape == (len(steps) * layers, 2)
    assert matrix[0, 0] == pytest.approx(1.0)


def test_comparison_plotter_writes_sweep_artifacts(tmp_path: Path):
    steps = (0, 50, 100, 150, 200, 250)
    ntp_summary = summarize_trajectory_data(_with_onsets(steps, h2=100, h3=200), RULE)
    target_summary = summarize_trajectory_data(_with_onsets(steps, h2=50, h3=150), RULE)
    result = summarize_target_depth_data(
        _paired_arms(ntp_summary, target_summary), primary_levels=[2, 3]
    )
    plot_target_depth_comparison(result, tmp_path)
    for name in (
        "acquisition_times.png",
        "delta_tau.png",
        "validation_ce.png",
        "h2_accessibility.png",
        "h2_clustering.png",
        "h2_q.png",
        "h3_accessibility.png",
        "h3_clustering.png",
        "h3_q.png",
    ):
        assert (tmp_path / name).exists()
    assert (tmp_path / "layerwise_onsets" / "h2.png").exists()


def test_target_depth_cli_writes_comparison_and_matched_ce_csv(tmp_path: Path):
    steps = (0, 50, 100, 150, 200, 250)
    ntp_summary = summarize_trajectory_data(_with_onsets(steps, h2=100, h3=200), RULE)
    target_summary = summarize_trajectory_data(_with_onsets(steps, h2=50, h3=150), RULE)
    screen = tmp_path / "screen"
    for target_layer, summary in (
        [(None, ntp_summary)] + [(layer, target_summary) for layer in range(4)]
    ):
        arm = _arm(target_layer, summary)
        run_dir = screen / "grammar_0" / "model_0" / arm["arm"]
        (run_dir / "trajectory").mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps(arm["config"]), encoding="utf-8")
        (run_dir / "metrics.json").write_text(json.dumps(arm["metrics"]), encoding="utf-8")
        (run_dir / "trajectory" / "acquisition.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
    rule_path = tmp_path / "acquisition_rule.json"
    rule_path.write_text(json.dumps(RULE), encoding="utf-8")
    result = summarize_target_depth(
        screen,
        tmp_path / "analysis",
        rule_path=rule_path,
        primary_levels=[2, 3],
    )
    assert result["groups"][0]["primary"][0]["target"] == "ntp"
    csv_text = (tmp_path / "analysis" / "validation_ce_by_step.csv").read_text()
    assert "grammar_seed,model_seed,step,ntp,target_0,target_1,target_2,target_3" in (
        csv_text.splitlines()[0]
    )
