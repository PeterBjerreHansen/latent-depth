#!/usr/bin/env python3
"""Turn raw checkpoint diagnostics into a fixed acquisition analysis.

The trajectory collector deliberately makes no developmental judgments.  This
module is the small, deterministic analysis seam between those raw
measurements and the Stage-01/Stage-02 comparison reports.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _vector(value: Any, name: str, *, length: int | None = None) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    if length is not None and len(value) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(value)}")
    return [_finite_number(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _threshold_key(value: float) -> str:
    rounded = round(value, 2)
    if value == rounded:
        return f"{value:.2f}"
    return format(value, ".12g")


def load_acquisition_rule(path: str | Path) -> dict[str, Any]:
    """Load and validate the committed acquisition rule.

    Margins are intentionally required here rather than supplied by code.  A
    missing rule or a missing threshold is therefore a protocol error, not an
    invitation to use an undocumented default.  The Q epsilon is only a
    numerical stabilizer and may use ``1e-8`` when omitted by older rule files.
    """
    rule_path = Path(path)
    try:
        payload = json.loads(rule_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"acquisition rule does not exist: {rule_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"acquisition rule is not valid JSON: {rule_path}") from exc

    rule = dict(_mapping(payload, "acquisition rule"))
    return _validate_rule_mapping(rule)


def _validate_rule_mapping(rule: dict[str, Any]) -> dict[str, Any]:
    """Validate a decoded rule mapping and return its normalized copy."""
    persistence = rule.get("persistence_checkpoints")
    if isinstance(persistence, bool) or not isinstance(persistence, int) or persistence < 2:
        raise ValueError("acquisition rule persistence_checkpoints must be an integer >= 2")

    accessibility = dict(_mapping(rule.get("accessibility"), "accessibility rule"))
    if accessibility.get("metric") != "balanced_accuracy":
        raise ValueError("accessibility.metric must be 'balanced_accuracy'")
    for key in (
        "margin_over_step0",
        "margin_over_shuffled",
        "margin_over_balanced_baseline",
    ):
        value = _finite_number(accessibility.get(key), f"accessibility.{key}")
        if value < 0:
            raise ValueError(f"accessibility.{key} must be nonnegative")
    rule["accessibility"] = accessibility

    clustering = dict(_mapping(rule.get("clustering"), "clustering rule"))
    for key in ("min_score", "margin_over_step0"):
        _finite_number(clustering.get(key), f"clustering.{key}")
    rule["clustering"] = clustering

    milestones = rule.get("probe_milestones")
    if not isinstance(milestones, list) or not milestones:
        raise ValueError("probe_milestones must be a nonempty array")
    milestone_values = [_finite_number(value, "probe_milestones entry") for value in milestones]
    if any(value < 0 or value > 1 for value in milestone_values):
        raise ValueError("probe_milestones entries must lie in [0, 1]")
    if len({_threshold_key(value) for value in milestone_values}) != len(milestone_values):
        raise ValueError("probe_milestones entries must be unique")
    rule["probe_milestones"] = milestone_values

    epsilon = _finite_number(rule.get("epsilon", 1e-8), "epsilon")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    rule["epsilon"] = epsilon
    return rule


def _event_for_layer(
    joint_pass: Sequence[bool],
    steps: Sequence[int],
    persistence: int,
) -> dict[str, Any]:
    for end in range(persistence - 1, len(joint_pass)):
        start = end - persistence + 1
        if all(joint_pass[start : end + 1]):
            return {
                "status": "observed",
                "onset_step": int(steps[start]),
                "confirmed_step": int(steps[end]),
            }
    return {"status": "censored", "through_step": int(steps[-1])}


def _milestone_event(
    curve: Sequence[float], steps: Sequence[int], threshold: float
) -> dict[str, Any]:
    for value, step in zip(curve, steps):
        if value >= threshold:
            return {"status": "observed", "step": int(step)}
    return {"status": "censored", "through_step": int(steps[-1])}


def _first_milestone(
    layer_events: Mapping[str, Mapping[str, Any]], threshold_key: str
) -> dict[str, Any]:
    observed = [
        (int(event["step"]), int(layer), event)
        for layer, milestones in layer_events.items()
        for event in [milestones[threshold_key]]
        if event["status"] == "observed"
    ]
    if not observed:
        # Every layer has the same trajectory horizon by construction.
        through_steps = {
            int(event["through_step"])
            for milestones in layer_events.values()
            for event in [milestones[threshold_key]]
            if event["status"] == "censored"
        }
        if len(through_steps) != 1:
            raise ValueError("inconsistent probe milestone censoring horizons")
        return {"status": "censored", "through_step": through_steps.pop()}
    step, layer, _ = min(observed, key=lambda item: (item[0], item[1]))
    return {"status": "observed", "step": step, "observer_layer": layer}


def _steps_from_records(records: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], list[int]]:
    if not records:
        raise ValueError("trajectory contains no checkpoint records")
    ordered = sorted(records, key=lambda record: int(record["checkpoint_global_step"]))
    steps = [int(record["checkpoint_global_step"]) for record in ordered]
    if len(set(steps)) != len(steps):
        raise ValueError("trajectory contains duplicate checkpoint steps")
    if steps[0] != 0:
        raise ValueError(
            "trajectory must include checkpoint step 0; rerun diagnostics with eval_at_start"
        )
    return ordered, steps


def _probe_controls(
    record: Mapping[str, Any], diagnostics: Mapping[str, Any], step: int
) -> Mapping[str, Any]:
    """Read controls from the current diagnose.py nesting, with a test-friendly fallback."""
    controls = diagnostics.get("probe_controls", record.get("probe_controls"))
    return _mapping(controls, f"checkpoint {step} probe_controls")


def summarize_trajectory_data(
    trajectory: Mapping[str, Any], rule: Mapping[str, Any], *, source: str | Path | None = None
) -> dict[str, Any]:
    """Return a deterministic acquisition summary for one raw trajectory."""
    records, steps = _steps_from_records(trajectory.get("checkpoints", []))
    validated_rule = load_rule_mapping(rule)

    first_diagnostics = _mapping(records[0].get("diagnostics"), "checkpoint diagnostics")
    first_probe = _mapping(first_diagnostics.get("linear_probe"), "linear_probe diagnostics")
    first_by_level = _mapping(first_probe.get("by_level"), "linear_probe.by_level")
    if not first_by_level:
        raise ValueError("trajectory has no linear-probe levels")
    levels = sorted((str(level) for level in first_by_level), key=int)

    # All trajectories must use one observer schedule.  This also catches
    # accidentally concatenated or partially regenerated raw JSON files.
    layer_count: int | None = None
    positions: dict[str, Any] = {}
    for index, record in enumerate(records):
        diagnostics = _mapping(record.get("diagnostics"), f"checkpoint {steps[index]} diagnostics")
        probe = _mapping(diagnostics.get("linear_probe"), "linear_probe diagnostics")
        by_level = _mapping(probe.get("by_level"), "linear_probe.by_level")
        if sorted((str(level) for level in by_level), key=int) != levels:
            raise ValueError("trajectory checkpoints do not share the same hierarchy levels")
        shuffled_control = _mapping(
            _probe_controls(record, diagnostics, steps[index]).get(
                "trained_backbone_shuffled_labels"
            ),
            "trained_backbone_shuffled_labels",
        )
        shuffled_by_level = _mapping(
            shuffled_control.get("balanced_accuracy_by_level"),
            "shuffled balanced_accuracy_by_level",
        )
        baseline_by_level = _mapping(
            probe.get("balanced_majority_accuracy_by_level"),
            "linear_probe balanced_majority_accuracy_by_level",
        )
        for level in levels:
            probe_entry = _mapping(by_level.get(level), f"linear_probe.by_level[{level}]")
            balanced = _vector(
                probe_entry.get("balanced_accuracy_by_layer"),
                f"balanced accuracy level {level} step {steps[index]}",
            )
            if layer_count is None:
                layer_count = len(balanced)
            if len(balanced) != layer_count:
                raise ValueError("trajectory checkpoints do not share the same observer layers")
            shuffled_value = shuffled_by_level.get(level)
            if isinstance(shuffled_value, Mapping):
                shuffled_value = shuffled_value.get("balanced_accuracy_by_layer")
            _vector(
                shuffled_value,
                f"shuffled balanced accuracy level {level} step {steps[index]}",
                length=layer_count,
            )
            _finite_number(baseline_by_level.get(level), f"balanced baseline level {level}")
            positions.setdefault(level, probe_entry.get("completion_position"))
            if positions[level] != probe_entry.get("completion_position"):
                raise ValueError(f"level {level} completion position changes across checkpoints")

    if layer_count is None:
        raise ValueError("trajectory has no observer layers")

    persistence = int(validated_rule["persistence_checkpoints"])

    levels_output: dict[str, Any] = {}
    for level in levels:
        layer_output: dict[str, Any] = {}
        layer_milestones: dict[str, dict[str, Any]] = {}
        for layer in range(layer_count):
            balanced_curve: list[float] = []
            ordinary_curve: list[float] = []
            shuffled_curve: list[float] = []
            baseline_curve: list[float] = []
            clustering_curve: list[float] = []
            synonym_distance_curve: list[float] = []
            variable_distance_curve: list[float] = []
            non_synonym_distance_curve: list[float] = []

            for index, record in enumerate(records):
                diagnostics = _mapping(record["diagnostics"], "checkpoint diagnostics")
                probe = _mapping(diagnostics["linear_probe"], "linear_probe diagnostics")
                probe_entry = _mapping(probe["by_level"][level], f"probe level {level}")
                balanced = _vector(
                    probe_entry["balanced_accuracy_by_layer"],
                    f"balanced accuracy level {level}",
                    length=layer_count,
                )
                ordinary = _vector(
                    probe_entry.get("accuracy_by_layer"),
                    f"ordinary accuracy level {level}",
                    length=layer_count,
                )
                control = _mapping(
                    _probe_controls(record, diagnostics, steps[index])[
                        "trained_backbone_shuffled_labels"
                    ],
                    "trained_backbone_shuffled_labels",
                )
                shuffled_values = control.get("balanced_accuracy_by_level", {}).get(level)
                if shuffled_values is None:
                    raise ValueError(f"shuffled control has no level {level}")
                shuffled = _vector(
                    shuffled_values,
                    f"shuffled balanced accuracy level {level}",
                    length=layer_count,
                )
                baseline = _finite_number(
                    probe["balanced_majority_accuracy_by_level"][level],
                    f"balanced baseline level {level}",
                )
                clustering_entry = _mapping(
                    _mapping(diagnostics.get("synonym_clustering"), "synonym_clustering")[
                        "by_level"
                    ][level],
                    "synonym clustering level",
                )
                clustering = _vector(
                    clustering_entry["score_by_layer"],
                    f"clustering level {level}",
                    length=layer_count,
                )
                synonym_distance = _vector(
                    clustering_entry["synonym_distance_by_layer"],
                    f"synonym distance level {level}",
                    length=layer_count,
                )
                variable_entry = _mapping(
                    _mapping(diagnostics.get("variable_sensitivity"), "variable_sensitivity")[
                        "by_level"
                    ][level],
                    "variable sensitivity level",
                )
                variable_distance = _vector(
                    variable_entry["variable_distance_by_layer"],
                    f"variable distance level {level}",
                    length=layer_count,
                )
                non_synonym_distance = _vector(
                    variable_entry["non_synonym_distance_by_layer"],
                    f"non-synonym distance level {level}",
                    length=layer_count,
                )
                balanced_curve.append(balanced[layer])
                ordinary_curve.append(ordinary[layer])
                shuffled_curve.append(shuffled[layer])
                baseline_curve.append(baseline)
                clustering_curve.append(clustering[layer])
                synonym_distance_curve.append(synonym_distance[layer])
                variable_distance_curve.append(variable_distance[layer])
                non_synonym_distance_curve.append(non_synonym_distance[layer])

            accessibility_rule = validated_rule["accessibility"]
            clustering_rule = validated_rule["clustering"]
            a_threshold = [
                max(
                    balanced_curve[0] + float(accessibility_rule["margin_over_step0"]),
                    shuffled + float(accessibility_rule["margin_over_shuffled"]),
                    baseline + float(accessibility_rule["margin_over_balanced_baseline"]),
                )
                for shuffled, baseline in zip(shuffled_curve, baseline_curve)
            ]
            c_threshold = max(
                clustering_curve[0] + float(clustering_rule["margin_over_step0"]),
                float(clustering_rule["min_score"]),
            )
            a_pass = [value >= threshold for value, threshold in zip(balanced_curve, a_threshold)]
            c_pass = [value >= c_threshold for value in clustering_curve]
            joint_pass = [a and c for a, c in zip(a_pass, c_pass)]
            q_curve = [
                (variable - synonym) / (non_synonym + float(validated_rule["epsilon"]))
                for variable, synonym, non_synonym in zip(
                    variable_distance_curve,
                    synonym_distance_curve,
                    non_synonym_distance_curve,
                )
            ]
            milestones = {
                _threshold_key(threshold): _milestone_event(
                    balanced_curve, steps, threshold
                )
                for threshold in validated_rule["probe_milestones"]
            }
            layer_milestones[str(layer)] = milestones
            acquisition = _event_for_layer(joint_pass, steps, persistence)
            layer_output[str(layer)] = {
                "observer_layer": layer,
                "balanced_accuracy": balanced_curve,
                "ordinary_accuracy": ordinary_curve,
                "shuffled_balanced_accuracy": shuffled_curve,
                "balanced_majority_accuracy": baseline_curve,
                "clustering": clustering_curve,
                "synonym_distance": synonym_distance_curve,
                "variable_distance": variable_distance_curve,
                "non_synonym_distance": non_synonym_distance_curve,
                "q": q_curve,
                "accessibility_threshold": a_threshold,
                "clustering_threshold": [c_threshold] * len(steps),
                "a_pass": a_pass,
                "c_pass": c_pass,
                "joint_a_c_pass": joint_pass,
                "probe_milestones": milestones,
                "acquisition": acquisition,
            }

        layerwise_onsets = {
            layer: layer_output[layer]["acquisition"] for layer in layer_output
        }
        observed = [
            (event["onset_step"], int(layer), event)
            for layer, event in layerwise_onsets.items()
            if event["status"] == "observed"
        ]
        if observed:
            onset_step, observer_layer, event = min(observed, key=lambda item: (item[0], item[1]))
            emergence = {
                "status": "observed",
                "onset_step": int(onset_step),
                "confirmed_step": int(event["confirmed_step"]),
                "observer_layer": observer_layer,
            }
        else:
            emergence = {"status": "censored", "through_step": int(steps[-1])}

        compact_milestones: dict[str, int | None] = {}
        compact_milestone_status: dict[str, dict[str, Any]] = {}
        for threshold in validated_rule["probe_milestones"]:
            key = _threshold_key(threshold)
            event = _first_milestone(layer_milestones, key)
            compact_milestone_status[key] = event
            compact_milestones[key] = int(event["step"]) if event["status"] == "observed" else None

        levels_output[level] = {
            "completion_position": positions[level],
            "emergence": emergence,
            "tau_AC": (
                int(emergence["onset_step"]) if emergence["status"] == "observed" else None
            ),
            "confirmation_step": (
                int(emergence["confirmed_step"])
                if emergence["status"] == "observed"
                else None
            ),
            "layerwise_onsets": layerwise_onsets,
            "layerwise": layer_output,
            "probe_milestones": compact_milestones,
            "probe_milestone_status": compact_milestone_status,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "source_trajectory": str(source) if source is not None else None,
        "run_dir": trajectory.get("run_dir"),
        "split": trajectory.get("split"),
        "num_sequences": trajectory.get("num_sequences"),
        "probe_steps": trajectory.get("probe_steps"),
        "checkpoint_steps": steps,
        "through_step": int(steps[-1]),
        "levels": levels_output,
        "rule": validated_rule,
    }


def load_rule_mapping(rule: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an already-loaded rule without requiring a filesystem path."""
    return _validate_rule_mapping(dict(rule))


def summarize_trajectory(
    trajectory_path: str | Path, rule_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    trajectory_file = Path(trajectory_path)
    rule_file = Path(rule_path)
    try:
        trajectory = json.loads(trajectory_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"trajectory does not exist: {trajectory_file}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"trajectory is not valid JSON: {trajectory_file}") from exc
    summary = summarize_trajectory_data(
        _mapping(trajectory, "trajectory"), load_acquisition_rule(rule_file), source=trajectory_file
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, help="trajectory.json from diagnose_trajectory.py")
    parser.add_argument("--rule", required=True, help="committed acquisition_rule.json")
    parser.add_argument("--output", required=True, help="acquisition summary JSON path")
    args = parser.parse_args()
    summarize_trajectory(args.trajectory, args.rule, args.output)


if __name__ == "__main__":
    main()
