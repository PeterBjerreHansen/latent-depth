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
    "accessibility": {"metric": "balanced_accuracy", "primary_threshold": 0.75},
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
    include_controls: bool = True,
) -> dict:
    balanced = balanced or {}
    clustering = clustering or {}
    records = []
    for index, step in enumerate(steps):
        by_level = {}
        for level in (2, 3):
            a_values = balanced.get(level, [[0.1, 0.1, 0.1] for _ in steps])
            c_values = clustering.get(level, [[0.0, 0.0, 0.0] for _ in steps])
            by_level[str(level)] = {
                "completion_position": level,
                "accuracy_by_layer": _vector_at(a_values, index),
                "balanced_accuracy_by_layer": _vector_at(a_values, index),
                "ce_by_layer": [1.0, 1.0, 1.0],
            }
        diagnostics = {
            "num_sequences": 32,
            "levels": [2, 3],
            "positions": [2, 3],
            "linear_probe": {
                "uniform_random_accuracy": 1.0 / 16,
                "ordinary_majority_accuracy_by_level": {"2": 0.5, "3": 0.5},
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
        if include_controls:
            diagnostics["probe_controls"] = {
                "trained_backbone_shuffled_labels": {
                    "balanced_accuracy_by_level": {
                        str(level): [0.1, 0.1, 0.1] for level in (2, 3)
                    }
                }
            }
        records.append({"checkpoint_global_step": step, "diagnostics": diagnostics})
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


def test_accessibility_uses_one_layer_even_when_clustering_moves_elsewhere():
    trajectory = _trajectory(
        balanced={2: [[0.1, 0.1, 0.1], [0.9, 0.1, 0.1], [0.9, 0.1, 0.1]]},
        clustering={2: [[0.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.2, 0.0]]},
    )
    level = summarize_trajectory_data(trajectory, RULE)["levels"]["2"]
    assert level["accessibility"] == {
        "status": "observed", "onset_step": 100, "confirmed_step": 200,
        "observer_layer": 0,
    }
    assert level["layerwise_onsets"]["1"]["status"] == "not_confirmed"


def test_one_checkpoint_is_not_persistent():
    trajectory = _trajectory(
        balanced={2: [[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [0.1, 0.1, 0.1]]}
    )
    level = summarize_trajectory_data(trajectory, RULE)["levels"]["2"]
    assert level["accessibility"] == {"status": "not_confirmed", "through_step": 200}


def test_same_layer_persistence_reports_onset_and_confirmation():
    trajectory = _trajectory(
        balanced={2: [[0.1, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.9, 0.1]]}
    )
    level = summarize_trajectory_data(trajectory, RULE)["levels"]["2"]
    assert level["layerwise_onsets"]["1"] == {
        "status": "observed", "onset_step": 100, "confirmed_step": 200,
    }
    assert level["accessibility"]["observer_layer"] == 1
    assert level["tau_accessibility"] == 100
    assert level["confirmation_step"] == 200
    assert level["probe_milestone_status"]["0.75"] == {
        "status": "observed",
        "onset_step": 100,
        "confirmed_step": 200,
        "observer_layer": 1,
    }


def test_unreached_level_is_explicitly_not_confirmed():
    level = summarize_trajectory_data(_trajectory(), RULE)["levels"]["3"]
    assert level["accessibility"] == {"status": "not_confirmed", "through_step": 200}
    assert all(
        event == {"status": "not_confirmed", "through_step": 200}
        for event in level["layerwise_onsets"].values()
    )


def test_probe_milestones_use_the_same_persistent_event():
    trajectory = _trajectory(
        balanced={2: [[0.1, 0.1, 0.1], [0.6, 0.1, 0.1], [0.8, 0.1, 0.1]]}
    )
    level = summarize_trajectory_data(trajectory, RULE)["levels"]["2"]
    assert level["probe_milestones"] == {"0.50": 100, "0.75": None, "0.90": None}
    assert level["probe_milestone_status"]["0.75"] == {
        "status": "not_confirmed", "through_step": 200,
    }
    assert level["probe_milestone_status"]["0.90"] == {
        "status": "not_confirmed", "through_step": 200,
    }
    assert level["accessibility"]["status"] == "not_confirmed"


def test_optional_controls_are_not_required_for_primary_analysis():
    summary = summarize_trajectory_data(_trajectory(include_controls=False), RULE)
    layer = summary["levels"]["2"]["layerwise"]["0"]
    assert layer["shuffled_balanced_accuracy"] is None
    assert layer["balanced_majority_accuracy"] == [0.25, 0.25, 0.25]


def test_q_is_derived_from_stored_distances():
    summary = summarize_trajectory_data(_trajectory(distances=(1.0, 3.0, 2.0)), RULE)
    assert summary["levels"]["2"]["layerwise"]["0"]["q"] == pytest.approx([1.0, 1.0, 1.0])


def test_balanced_majority_uses_represented_classes_on_eval_split():
    labels = torch.tensor([[0] * 12 + [1] * 4 + [2] * 4])
    features = torch.zeros(1, 1, 20, 3)
    result = _fit_linear_probes(
        features, labels, vocab_size=4, steps=1, learning_rate=0.01,
        seed=123, device=torch.device("cpu"),
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
            "v": 16, "n": 16, "m": 4, "s": 2, "L": 5,
            "rule_seed": 0, "train_seed": 1000, "val_seed": 2000, "test_seed": 3000,
        },
        "data": {
            "train_size": 64, "val_size": 64, "test_size": 64,
            "resample_train_each_epoch": True,
        },
        "model": {"n_layer": 3, "n_head": 2, "n_embd": 16, "dropout": 0.0, "bias": True},

        "auxiliary": {
            "mode": "none" if target_layer is None else "next_latent",
            "target_layer": target_layer, "weight": 0.1,
            "predictor_hidden_mult": 2, "seed": 54321,
        },
        "optim": {
            "name": "adamw", "learning_rate": 0.001, "betas": [0.9, 0.95],
            "weight_decay": 0.0, "warmup_epochs": 0.0,
        },
        "train": {
            "batch_size": 16, "max_epochs": 5, "max_updates": 6,
            "grad_clip": 1.0, "eval_every_epochs": 1, "eval_every_updates": 1,
            "eval_at_start": True, "num_workers": 0, "device": "cpu",
            "deterministic": True, "deterministic_strict": True,
            "save_checkpoints": True, "checkpoint_every_updates": 1, "diagnostic_snapshot_every_updates": 1,
        },
        "model_seed": 0,
    }


def _arm(target_layer, summary, steps=(0, 50, 100, 150, 200, 250), *, best=1.0):
    history = [{"global_step": step, "val_ce": best + (250 - step) / 1000} for step in steps]
    metrics = {
        "arm": "ntp" if target_layer is None else f"target_{target_layer}",
        "auxiliary_target_layer": target_layer, "rule_seed": 0, "model_seed": 0,
        "best_val_ce": best, "history": history,
        "per_epoch_train_pool": 64, "total_optimizer_updates": steps[-1],
        "total_sequence_draws": len(steps), "total_predicted_tokens": len(steps) * 10,
    }
    return {
        "run_dir": metrics["arm"], "arm": metrics["arm"], "target_layer": target_layer,
        "grammar_seed": 0, "model_seed": 0, "config": _config(target_layer),
        "metrics": metrics, "history": history, "summary": summary,
    }


def _paired_arms(ntp_summary, target_summary, steps=(0, 50, 100, 150, 200, 250), *, target_best=1.0):
    return [
        _arm(None, ntp_summary, steps=steps),
        *[_arm(layer, target_summary, steps=steps, best=target_best) for layer in range(4)],
    ]


def test_target_depth_deltas_and_interval_keep_absolute_times():
    steps = (0, 50, 100, 150, 200, 250)
    ntp_summary = summarize_trajectory_data(_with_onsets(steps, h2=100, h3=200), RULE)
    target_summary = summarize_trajectory_data(_with_onsets(steps, h2=50, h3=200), RULE)
    result = summarize_target_depth_data(
        _paired_arms(ntp_summary, target_summary, target_best=0.99), primary_levels=[2, 3]
    )
    target = next(row for row in result["groups"][0]["primary"] if row["target"] == "target_2")
    assert target["tau_2"] == 50
    assert target["delta_tau_2"] == -50
    assert target["tau_3"] == 200
    assert target["delta_tau_3"] == 0
    assert target["transitions"]["2->3"] == {
        "from_level": 2,
        "to_level": 3,
        "tau_interval": {"status": "observed", "value": 150},
        "ntp_tau_interval": {"status": "observed", "value": 100},
        "delta_tau_interval_vs_ntp": {"status": "observed", "value": 50},
    }
    assert target["best_val_ce_delta_vs_ntp"] == pytest.approx(-0.01)
    assert target["levels"]["2"]["matched_validation_ce"]["delta_vs_ntp"] == pytest.approx(-0.01)
    assert len(result["validation_ce_by_step"]) == len(steps)


def test_not_confirmed_paired_comparisons_are_unavailable_without_bounds():
    steps = (0, 50, 100, 150, 200, 250)
    ntp_summary = summarize_trajectory_data(_with_onsets(steps, h2=50, h3=150), RULE)
    target_summary = summarize_trajectory_data(_with_onsets(steps, h2=50, h3=999), RULE)
    result = summarize_target_depth_data(
        _paired_arms(ntp_summary, target_summary), primary_levels=[2, 3]
    )
    target = next(row for row in result["groups"][0]["primary"] if row["target"] == "target_2")
    assert target["tau_3"] is None
    assert target["tau_3_status"] == "not_confirmed"
    assert target["tau_3_through_step"] == 250
    assert target["delta_tau_3"] is None
    assert target["delta_tau_3_status"] == "unavailable"
    assert "delta_tau_3_bound" not in target
    assert target["transitions"]["2->3"] == {
        "from_level": 2,
        "to_level": 3,
        "tau_interval": {"status": "unavailable"},
        "ntp_tau_interval": {"status": "observed", "value": 100},
        "delta_tau_interval_vs_ntp": {"status": "unavailable"},
    }


def test_sweep_refuses_mismatched_evaluation_schedules():
    steps = (0, 50, 100, 150, 200, 250)
    ntp_summary = summarize_trajectory_data(_with_onsets(steps, h2=100, h3=200), RULE)
    target_summary = summarize_trajectory_data(_with_onsets((0, 100, 200), h2=100, h3=200), RULE)
    with pytest.raises(ValueError, match="checkpoint schedule"):
        summarize_target_depth_data(
            [_arm(None, ntp_summary), _arm(2, target_summary, steps=(0, 100, 200))],
            primary_levels=[2, 3],
        )


def test_partial_target_comparison_is_allowed_and_reported():
    summary = summarize_trajectory_data(_with_onsets(), RULE)
    result = summarize_target_depth_data(
        [_arm(None, summary), _arm(2, summary)], primary_levels=[2, 3]
    )
    group = result["groups"][0]
    assert group["targets_present"] == ["ntp", "target_2"]
    assert [row["target"] for row in group["primary"]] == ["ntp", "target_2"]


def test_q_heatmap_helper_uses_stored_distances():
    trajectory = _trajectory()
    matrix, steps, layers = _heatmap(trajectory["checkpoints"], "q", ["2", "3"])
    assert matrix.shape == (len(steps) * layers, 2)
    assert matrix[0, 0] == pytest.approx(1.0)


def test_comparison_plotter_writes_one_combined_timing_artifact(tmp_path: Path):
    steps = (0, 50, 100, 150, 200, 250)
    ntp_summary = summarize_trajectory_data(_with_onsets(steps, h2=100, h3=200), RULE)
    target_summary = summarize_trajectory_data(_with_onsets(steps, h2=50, h3=150), RULE)
    result = summarize_target_depth_data(
        _paired_arms(ntp_summary, target_summary), primary_levels=[2, 3]
    )
    plot_target_depth_comparison(result, tmp_path)
    for name in (
        "acquisition_times.png", "validation_ce.png", "h2_accessibility.png",
        "h2_clustering.png", "h2_q.png", "h3_accessibility.png",
        "h3_clustering.png", "h3_q.png", "transition_intervals.png",
    ):
        assert (tmp_path / name).exists()
    assert not (tmp_path / "delta_tau.png").exists()
    assert (tmp_path / "layerwise_onsets" / "h2.png").exists()


def test_ntp_censoring_does_not_hide_auxiliary_layer_curves(tmp_path: Path):
    steps = (0, 50, 100, 150, 200, 250)
    ntp_summary = summarize_trajectory_data(_with_onsets(steps, h2=999, h3=200), RULE)
    target_summary = summarize_trajectory_data(_with_onsets(steps, h2=50, h3=150), RULE)
    result = summarize_target_depth_data(
        [_arm(None, ntp_summary), _arm(2, target_summary)], primary_levels=[2, 3]
    )
    plot_target_depth_comparison(result, tmp_path)
    assert (tmp_path / "h2_accessibility.png").exists()


def test_target_depth_cli_reads_raw_trajectories_and_writes_compact_comparison(tmp_path: Path):
    steps = (0, 50, 100, 150, 200, 250)
    screen = tmp_path / "screen"
    for target_layer in (None, 2):
        trajectory = _with_onsets(steps, h2=100 if target_layer is None else 50, h3=200)
        summary = summarize_trajectory_data(trajectory, RULE)
        arm = _arm(target_layer, summary)
        run_dir = screen / "grammar_0" / "model_0" / arm["arm"]
        (run_dir / "trajectory").mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps(arm["config"]), encoding="utf-8")
        (run_dir / "metrics.json").write_text(json.dumps(arm["metrics"]), encoding="utf-8")
        (run_dir / "trajectory" / "trajectory.json").write_text(
            json.dumps(trajectory), encoding="utf-8"
        )
    rule_path = tmp_path / "acquisition_rule.json"
    rule_path.write_text(json.dumps(RULE), encoding="utf-8")
    result = summarize_target_depth(
        screen, tmp_path / "analysis", rule_path=rule_path, primary_levels=[2, 3]
    )
    assert result["groups"][0]["targets_present"] == ["ntp", "target_2"]
    disk = json.loads((tmp_path / "analysis" / "comparison.json").read_text())
    assert all("summary" not in arm for arm in disk["groups"][0]["arms"])
    csv_text = (tmp_path / "analysis" / "validation_ce_by_step.csv").read_text()
    assert "grammar_seed,model_seed,step,ntp,target_2" in csv_text.splitlines()[0]
    transition_csv = (tmp_path / "analysis" / "transition_intervals.csv").read_text()
    assert transition_csv.splitlines()[0] == (
        "grammar_seed,model_seed,target,transition,from_level,to_level,"
        "tau_interval,tau_interval_status,ntp_tau_interval,ntp_tau_interval_status,"
        "delta_tau_interval_vs_ntp,delta_tau_interval_vs_ntp_status"
    )
    plot_target_depth_comparison(
        tmp_path / "analysis" / "comparison.json", tmp_path / "plots"
    )
    assert (tmp_path / "plots" / "h2_accessibility.png").exists()


def test_paired_comparison_rejects_different_probe_settings():
    summary = summarize_trajectory_data(_with_onsets((0, 50, 100, 150, 200, 250), h2=50, h3=150), RULE)
    arms = _paired_arms(summary, summary)
    import copy
    arms = copy.deepcopy(arms)
    arms[1]['summary'] = copy.deepcopy(arms[1]['summary'])
    arms[1]['summary']['diagnostic_config'] = {'seed': 999, 'probe_lr': .01}
    with pytest.raises(ValueError, match='diagnostic'):
        summarize_target_depth_data(arms, primary_levels=[2, 3])
