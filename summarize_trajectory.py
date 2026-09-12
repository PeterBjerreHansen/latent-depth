"""Internal helpers for accessibility analysis of raw diagnostic trajectories."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 2


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
    """Load the committed primary accessibility rule."""
    rule_path = Path(path)
    try:
        payload = json.loads(rule_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"acquisition rule does not exist: {rule_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"acquisition rule is not valid JSON: {rule_path}") from exc
    return _validate_rule_mapping(dict(_mapping(payload, "acquisition rule")))


def _validate_rule_mapping(rule: dict[str, Any]) -> dict[str, Any]:
    persistence = rule.get("persistence_checkpoints")
    if isinstance(persistence, bool) or not isinstance(persistence, int) or persistence < 2:
        raise ValueError("acquisition rule persistence_checkpoints must be an integer >= 2")

    accessibility = dict(_mapping(rule.get("accessibility"), "accessibility rule"))
    if accessibility.get("metric") != "balanced_accuracy":
        raise ValueError("accessibility.metric must be 'balanced_accuracy'")
    primary_threshold = _finite_number(
        accessibility.get("primary_threshold"), "accessibility.primary_threshold"
    )
    if not 0 <= primary_threshold <= 1:
        raise ValueError("accessibility.primary_threshold must lie in [0, 1]")
    accessibility["primary_threshold"] = primary_threshold
    rule["accessibility"] = accessibility

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


def load_rule_mapping(rule: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an already-loaded rule mapping."""
    return _validate_rule_mapping(dict(rule))


def _event_for_layer(
    passed: Sequence[bool], steps: Sequence[int], persistence: int
) -> dict[str, Any]:
    for end in range(persistence - 1, len(passed)):
        start = end - persistence + 1
        if all(passed[start : end + 1]):
            return {
                "status": "observed",
                "onset_step": int(steps[start]),
                "confirmed_step": int(steps[end]),
            }
    return {"status": "not_confirmed", "through_step": int(steps[-1])}


def _first_milestone(
    layer_events: Mapping[str, Mapping[str, Any]], milestone: str
) -> dict[str, Any]:
    observed = [
        (int(event["onset_step"]), int(layer), int(event["confirmed_step"]))
        for layer, events in layer_events.items()
        for event in [events[milestone]]
        if event["status"] == "observed"
    ]
    if observed:
        onset_step, layer, confirmed_step = min(
            observed, key=lambda item: (item[0], item[1])
        )
        return {
            "status": "observed",
            "onset_step": onset_step,
            "confirmed_step": confirmed_step,
            "observer_layer": layer,
        }
    through_steps = {
        int(event["through_step"])
        for events in layer_events.values()
        for event in [events[milestone]]
    }
    if len(through_steps) != 1:
        raise ValueError("inconsistent probe milestone horizons")
    return {"status": "not_confirmed", "through_step": through_steps.pop()}


def _steps_from_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[int]]:
    if not records:
        raise ValueError("trajectory contains no checkpoint records")
    ordered = sorted(records, key=lambda record: int(record["checkpoint_global_step"]))
    steps = [int(record["checkpoint_global_step"]) for record in ordered]
    if len(set(steps)) != len(steps):
        raise ValueError("trajectory contains duplicate checkpoint steps")
    if steps[0] != 0:
        raise ValueError("trajectory must include checkpoint step 0")
    return ordered, steps


def _optional_control_curve(
    record: Mapping[str, Any], level: str, layer_count: int
) -> list[float] | None:
    diagnostics = _mapping(record["diagnostics"], "checkpoint diagnostics")
    controls = diagnostics.get("probe_controls")
    if "probe_controls" in record and "probe_controls" not in diagnostics:
        raise ValueError(
            "probe_controls must be nested under checkpoint diagnostics"
        )
    if not isinstance(controls, Mapping):
        return None
    shuffled = controls.get("trained_backbone_shuffled_labels")
    if not isinstance(shuffled, Mapping):
        return None
    values_by_level = shuffled.get("balanced_accuracy_by_level")
    if not isinstance(values_by_level, Mapping):
        return None
    values = values_by_level.get(level)
    if isinstance(values, Mapping):
        values = values.get("balanced_accuracy_by_layer")
    if values is None:
        return None
    return _vector(values, f"shuffled balanced accuracy level {level}", length=layer_count)


def summarize_trajectory_data(
    trajectory: Mapping[str, Any], rule: Mapping[str, Any], *, source: str | Path | None = None
) -> dict[str, Any]:
    """Analyze one raw trajectory in memory.

    Accessibility at the fixed primary threshold is the event used for paired
    timing. Clustering and Q are retained only when the raw trajectory contains
    those diagnostics; neither is an acquisition hurdle.
    """
    records, steps = _steps_from_records(trajectory.get("checkpoints", []))
    validated_rule = load_rule_mapping(rule)
    primary_threshold = float(validated_rule["accessibility"]["primary_threshold"])

    first_diagnostics = _mapping(records[0]["diagnostics"], "checkpoint diagnostics")
    first_probe = _mapping(first_diagnostics.get("linear_probe"), "linear_probe diagnostics")
    first_by_level = _mapping(first_probe.get("by_level"), "linear_probe.by_level")
    if not first_by_level:
        raise ValueError("trajectory has no linear-probe levels")
    levels = sorted((str(level) for level in first_by_level), key=int)

    layer_count: int | None = None
    positions: dict[str, Any] = {}
    for step, record in zip(steps, records):
        diagnostics = _mapping(record["diagnostics"], f"checkpoint {step} diagnostics")
        probe = _mapping(diagnostics.get("linear_probe"), "linear_probe diagnostics")
        by_level = _mapping(probe.get("by_level"), "linear_probe.by_level")
        if sorted((str(level) for level in by_level), key=int) != levels:
            raise ValueError("trajectory checkpoints do not share the same hierarchy levels")
        for level in levels:
            entry = _mapping(by_level[level], f"linear_probe.by_level[{level}]")
            balanced = _vector(
                entry.get("balanced_accuracy_by_layer"),
                f"balanced accuracy level {level} step {step}",
            )
            if layer_count is None:
                layer_count = len(balanced)
            if len(balanced) != layer_count:
                raise ValueError("trajectory checkpoints do not share the same observer layers")
            positions.setdefault(level, entry.get("completion_position"))
            if positions[level] != entry.get("completion_position"):
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
            ordinary_curve: list[float] | None = []
            shuffled_curve: list[float] | None = []
            baseline_curve: list[float] | None = []
            clustering_curve: list[float] | None = []
            synonym_curve: list[float] | None = []
            variable_curve: list[float] | None = []
            non_synonym_curve: list[float] | None = []

            for record in records:
                diagnostics = _mapping(record["diagnostics"], "checkpoint diagnostics")
                probe = _mapping(diagnostics["linear_probe"], "linear_probe diagnostics")
                entry = _mapping(probe["by_level"][level], f"probe level {level}")
                balanced = _vector(
                    entry["balanced_accuracy_by_layer"],
                    f"balanced accuracy level {level}",
                    length=layer_count,
                )
                balanced_curve.append(balanced[layer])

                ordinary = entry.get("accuracy_by_layer")
                if ordinary_curve is not None and ordinary is not None:
                    ordinary_curve.append(
                        _vector(
                            ordinary,
                            f"ordinary accuracy level {level}",
                            length=layer_count,
                        )[layer]
                    )
                else:
                    ordinary_curve = None

                shuffled = _optional_control_curve(record, level, layer_count)
                if shuffled_curve is not None and shuffled is not None:
                    shuffled_curve.append(shuffled[layer])
                else:
                    shuffled_curve = None

                baseline_by_level = probe.get("balanced_majority_accuracy_by_level")
                if baseline_curve is not None and isinstance(baseline_by_level, Mapping):
                    baseline = baseline_by_level.get(level)
                    if baseline is not None:
                        baseline_curve.append(
                            _finite_number(baseline, f"balanced baseline level {level}")
                        )
                    else:
                        baseline_curve = None
                else:
                    baseline_curve = None

                clustering_data = diagnostics.get("synonym_clustering")
                if clustering_curve is not None and isinstance(clustering_data, Mapping):
                    clustering_entry = _mapping(
                        clustering_data.get("by_level", {}).get(level),
                        f"synonym clustering level {level}",
                    )
                    values = _vector(
                        clustering_entry.get("score_by_layer"),
                        f"clustering level {level}",
                        length=layer_count,
                    )
                    clustering_curve.append(values[layer])
                    synonym_values = _vector(
                        clustering_entry.get("synonym_distance_by_layer"),
                        f"synonym distance level {level}",
                        length=layer_count,
                    )
                    if synonym_curve is not None:
                        synonym_curve.append(synonym_values[layer])
                else:
                    clustering_curve = None
                    synonym_curve = None

                variable_data = diagnostics.get("variable_sensitivity")
                if variable_curve is not None and isinstance(variable_data, Mapping):
                    variable_entry = _mapping(
                        variable_data.get("by_level", {}).get(level),
                        f"variable sensitivity level {level}",
                    )
                    variable_values = _vector(
                        variable_entry.get("variable_distance_by_layer"),
                        f"variable distance level {level}",
                        length=layer_count,
                    )
                    non_synonym_values = _vector(
                        variable_entry.get("non_synonym_distance_by_layer"),
                        f"non-synonym distance level {level}",
                        length=layer_count,
                    )
                    variable_curve.append(variable_values[layer])
                    if non_synonym_curve is not None:
                        non_synonym_curve.append(non_synonym_values[layer])
                else:
                    variable_curve = None
                    non_synonym_curve = None

            accessibility_pass = [value >= primary_threshold for value in balanced_curve]
            accessibility_event = _event_for_layer(accessibility_pass, steps, persistence)
            milestones = {
                _threshold_key(threshold): _event_for_layer(
                    [value >= threshold for value in balanced_curve],
                    steps,
                    persistence,
                )
                for threshold in validated_rule["probe_milestones"]
            }
            layer_milestones[str(layer)] = milestones
            q_curve = None
            if synonym_curve is not None and variable_curve is not None and non_synonym_curve is not None:
                q_curve = [
                    (variable - synonym) / (non_synonym + float(validated_rule["epsilon"]))
                    for variable, synonym, non_synonym in zip(
                        variable_curve, synonym_curve, non_synonym_curve
                    )
                ]
            layer_output[str(layer)] = {
                "observer_layer": layer,
                "balanced_accuracy": balanced_curve,
                "ordinary_accuracy": ordinary_curve,
                "shuffled_balanced_accuracy": shuffled_curve,
                "balanced_majority_accuracy": baseline_curve,
                "clustering": clustering_curve,
                "synonym_distance": synonym_curve,
                "variable_distance": variable_curve,
                "non_synonym_distance": non_synonym_curve,
                "q": q_curve,
                "accessibility_threshold": [primary_threshold] * len(steps),
                "accessibility_pass": accessibility_pass,
                "probe_milestones": milestones,
                "accessibility_event": accessibility_event,
            }

        layerwise_onsets = {
            layer: layer_output[layer]["accessibility_event"] for layer in layer_output
        }
        observed = [
            (event["onset_step"], int(layer), event)
            for layer, event in layerwise_onsets.items()
            if event["status"] == "observed"
        ]
        if observed:
            onset_step, observer_layer, event = min(observed, key=lambda item: (item[0], item[1]))
            accessibility = {
                "status": "observed",
                "onset_step": int(onset_step),
                "confirmed_step": int(event["confirmed_step"]),
                "observer_layer": observer_layer,
            }
        else:
            accessibility = {"status": "not_confirmed", "through_step": int(steps[-1])}

        compact_milestones: dict[str, int | None] = {}
        milestone_status: dict[str, dict[str, Any]] = {}
        for threshold in validated_rule["probe_milestones"]:
            key = _threshold_key(threshold)
            event = _first_milestone(layer_milestones, key)
            milestone_status[key] = event
            compact_milestones[key] = (
                int(event["onset_step"]) if event["status"] == "observed" else None
            )

        levels_output[level] = {
            "completion_position": positions[level],
            "primary_accessibility_threshold": primary_threshold,
            "accessibility": accessibility,
            "tau_accessibility": (
                int(accessibility["onset_step"])
                if accessibility["status"] == "observed"
                else None
            ),
            "confirmation_step": (
                int(accessibility["confirmed_step"])
                if accessibility["status"] == "observed"
                else None
            ),
            "layerwise_onsets": layerwise_onsets,
            "layerwise": layer_output,
            "probe_milestones": compact_milestones,
            "probe_milestone_status": milestone_status,
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
