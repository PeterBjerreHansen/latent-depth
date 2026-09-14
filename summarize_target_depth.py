#!/usr/bin/env python3
"""Analyze raw target-depth trajectories as a paired comparison."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from summarize_trajectory import load_acquisition_rule, summarize_trajectory_data

SCHEMA_VERSION = 3


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{description} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _arm_name(target_layer: int | None) -> str:
    return "ntp" if target_layer is None else f"target_{target_layer}"


def _group_label(key: tuple[int, int]) -> str:
    return f"grammar_{key[0]}/model_{key[1]}"


def _discover_arm_dirs(screen_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for metrics_path in sorted(screen_dir.rglob("metrics.json")):
        run_dir = metrics_path.parent
        if not run_dir.parent.name.startswith("model_"):
            continue
        if not run_dir.parent.parent.name.startswith("grammar_"):
            continue
        paths.append(run_dir)
    return paths


def _load_arm(run_dir: Path, rule: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _load_json(run_dir / "metrics.json", "arm metrics")
    config = _load_json(run_dir / "config.json", "arm config")
    trajectory_path = run_dir / "metrics.json"
    trajectory = metrics
    summary = summarize_trajectory_data(
        trajectory, rule, source=trajectory_path
    )
    target_value = metrics.get("auxiliary_target_layer", config.get("auxiliary", {}).get("target_layer"))
    if target_value is not None and (
        isinstance(target_value, bool) or not isinstance(target_value, int)
    ):
        raise ValueError(f"{run_dir}: auxiliary target layer must be an integer or null")
    target_layer = None if target_value is None else int(target_value)
    arm = metrics.get("arm", run_dir.name)
    if arm != _arm_name(target_layer):
        raise ValueError(
            f"{run_dir}: arm name {arm!r} does not match target layer {target_layer!r}"
        )
    history = metrics.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"{run_dir}: metrics.json has no validation history")
    for row in history:
        if not isinstance(row, dict) or "global_step" not in row or "val_ce" not in row:
            raise ValueError(f"{run_dir}: every history row needs global_step and val_ce")
    return {
        "run_dir": str(run_dir),
        "arm": arm,
        "target_layer": target_layer,
        "grammar_seed": int(metrics.get("rule_seed", config.get("rhm", {}).get("rule_seed"))),
        "model_seed": int(metrics.get("model_seed", config.get("model_seed", 0))),
        "config": config,
        "metrics": metrics,
        "history": history,
        "trajectory": trajectory,
        "summary": summary,
    }


def _comparison_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return config fields that must be identical within a paired arm group."""
    result = deepcopy(dict(config))
    auxiliary = dict(result.get("auxiliary", {}))
    auxiliary.pop("mode", None)
    auxiliary.pop("target_layer", None)
    result["auxiliary"] = auxiliary
    # The seed itself is represented by the paired group key.  Keeping it in
    # the config comparison catches accidental cross-grammar mixing as well.
    return result


def _trajectory_signature(arm: Mapping[str, Any]) -> dict[str, Any]:
    summary = arm["summary"]
    levels = summary.get("levels")
    if not isinstance(levels, Mapping) or not levels:
        raise ValueError(f"{arm['run_dir']}: trajectory analysis has no levels")
    layer_shapes = {
        str(level): sorted(str(layer) for layer in level_data.get("layerwise", {}))
        for level, level_data in levels.items()
    }
    return {
        "split": summary.get("split"),
        "num_sequences": summary.get("num_sequences"),
        "probe_steps": summary.get("probe_steps"),
        "diagnostic_config": summary.get("diagnostic_config"),
        "checkpoint_steps": summary.get("checkpoint_steps"),
        "levels": sorted(str(level) for level in levels),
        "layer_shapes": layer_shapes,
        "rule": summary.get("rule"),
    }


def _history_steps(arm: Mapping[str, Any]) -> list[int]:
    steps = [int(row["global_step"]) for row in arm["history"]]
    if len(set(steps)) != len(steps) or steps != sorted(steps):
        raise ValueError(f"{arm['run_dir']}: validation history has duplicate or unsorted steps")
    return steps


def _validate_paired_group(arms: list[dict[str, Any]], key: tuple[int, int]) -> None:
    if not any(arm["arm"] == "ntp" for arm in arms):
        raise ValueError(f"paired group {_group_label(key)} has no ntp baseline")
    if not any(arm["arm"] != "ntp" for arm in arms):
        raise ValueError(f"paired group {_group_label(key)} has no target arm")
    names = [arm["arm"] for arm in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"paired group {_group_label(key)} contains duplicate arm names")
    config = _comparison_config(arms[0]["config"])
    trajectory = _trajectory_signature(arms[0])
    history_steps = _history_steps(arms[0])
    for arm in arms[1:]:
        if _comparison_config(arm["config"]) != config:
            raise ValueError(
                f"refusing to compare {_group_label(key)}: arms do not share the same "
                "hierarchy/training configuration"
            )
        if _trajectory_signature(arm) != trajectory:
            raise ValueError(
                f"refusing to compare {_group_label(key)}: arms do not share the same "
                "diagnostic hierarchy/checkpoint schedule or accessibility rule"
            )
        if _history_steps(arm) != history_steps:
            raise ValueError(
                f"refusing to compare {_group_label(key)}: arms do not share the same "
                "validation evaluation schedule"
            )


def _event(summary: Mapping[str, Any], level: int) -> dict[str, Any]:
    levels = summary.get("levels", {})
    value = levels.get(str(level))
    if not isinstance(value, Mapping) or not isinstance(value.get("accessibility"), Mapping):
        raise ValueError(f"trajectory analysis has no hierarchy level {level}")
    return dict(value["accessibility"])


def _time_fields(event: Mapping[str, Any]) -> tuple[int | None, str, int | None]:
    status = event.get("status")
    if status == "observed":
        return int(event["onset_step"]), "observed", None
    if status == "not_confirmed":
        return None, "not_confirmed", int(event["through_step"])
    raise ValueError(f"invalid acquisition event status: {status!r}")


def _difference(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe left - right only when both event times are observed."""
    if left["status"] == "observed" and right["status"] == "observed":
        return {
            "status": "observed",
            "value": int(left["onset_step"]) - int(right["onset_step"]),
        }
    return {"status": "unavailable"}


def _difference_values(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe the difference between two already-computed differences."""
    if left["status"] == "observed" and right["status"] == "observed":
        return {
            "status": "observed",
            "value": int(left["value"]) - int(right["value"]),
        }
    return {"status": "unavailable"}


def _direct_difference_value(comparison: Mapping[str, Any]) -> int | None:
    return int(comparison["value"]) if comparison["status"] == "observed" else None


def _history_by_step(arm: Mapping[str, Any]) -> dict[int, float]:
    return {int(row["global_step"]): float(row["val_ce"]) for row in arm["history"]}


def _matched_ce(
    arm: Mapping[str, Any], ntp: Mapping[str, Any], event: Mapping[str, Any]
) -> dict[str, Any] | None:
    if event["status"] != "observed":
        return None
    step = int(event["onset_step"])
    arm_history = _history_by_step(arm)
    ntp_history = _history_by_step(ntp)
    if step not in arm_history or step not in ntp_history:
        return None
    arm_ce = arm_history[step]
    ntp_ce = ntp_history[step]
    return {
        "step": step,
        "arm_val_ce": arm_ce,
        "ntp_val_ce": ntp_ce,
        "delta_vs_ntp": arm_ce - ntp_ce,
    }


def _transition_key(lower: int, upper: int) -> str:
    return f"{lower}->{upper}"


def _adjacent_level_pairs(levels: list[int]) -> list[tuple[int, int]]:
    """Return only numeric l-1 -> l transitions present in the trajectory."""
    return [
        (lower, upper)
        for lower, upper in zip(levels, levels[1:])
        if upper == lower + 1
    ]


def _transition_comparison(
    events: Mapping[int, Mapping[str, Any]],
    ntp_events: Mapping[int, Mapping[str, Any]],
    lower: int,
    upper: int,
) -> dict[str, Any]:
    """Compare one arm's l-1 -> l interval with the NTP interval."""
    interval = _difference(events[upper], events[lower])
    ntp_interval = _difference(ntp_events[upper], ntp_events[lower])
    delta_vs_ntp = _difference_values(interval, ntp_interval)
    return {
        "from_level": lower,
        "to_level": upper,
        "tau_interval": interval,
        "ntp_tau_interval": ntp_interval,
        "delta_tau_interval_vs_ntp": delta_vs_ntp,
    }


def _primary_row(
    arm: Mapping[str, Any],
    ntp: Mapping[str, Any],
    primary_levels: list[int],
    transition_pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    event_levels = {
        level
        for pair in transition_pairs
        for level in pair
    } | set(primary_levels)
    events = {level: _event(arm["summary"], level) for level in event_levels}
    ntp_events = {level: _event(ntp["summary"], level) for level in event_levels}
    row: dict[str, Any] = {
        "target": arm["arm"],
        "target_layer": arm["target_layer"],
        "best_val_ce": float(arm["metrics"]["best_val_ce"]),
        "best_val_ce_delta_vs_ntp": float(arm["metrics"]["best_val_ce"])
        - float(ntp["metrics"]["best_val_ce"]),
        "levels": {},
    }
    for level in primary_levels:
        tau, status, through = _time_fields(events[level])
        baseline_tau, baseline_status, baseline_through = _time_fields(ntp_events[level])
        delta = _difference(events[level], ntp_events[level])
        row[f"tau_{level}"] = tau
        row[f"tau_{level}_status"] = status
        row[f"tau_{level}_through_step"] = through
        row[f"delta_tau_{level}"] = _direct_difference_value(delta)
        row[f"delta_tau_{level}_status"] = delta["status"]
        row["levels"][str(level)] = {
            "event": events[level],
            "ntp_event": ntp_events[level],
            "ntp_tau": baseline_tau,
            "ntp_tau_status": baseline_status,
            "ntp_tau_through_step": baseline_through,
            "delta_tau": delta,
            "matched_validation_ce": _matched_ce(arm, ntp, events[level]),
        }

    row["transitions"] = {
        _transition_key(lower, upper): _transition_comparison(
            events, ntp_events, lower, upper
        )
        for lower, upper in transition_pairs
    }
    return row


def _validation_rows(arms: list[Mapping[str, Any]], key: tuple[int, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    names = [str(arm["arm"]) for arm in arms]
    histories = {name: _history_by_step(arm) for name, arm in zip(names, arms)}
    steps = sorted(next(iter(histories.values())))
    rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    ntp_history = histories["ntp"]
    for step in steps:
        row: dict[str, Any] = {
            "grammar_seed": key[0],
            "model_seed": key[1],
            "step": step,
        }
        delta_row = dict(row)
        for name in names:
            row[name] = histories[name][step]
            delta_row[name] = histories[name][step] - ntp_history[step]
        rows.append(row)
        delta_rows.append(delta_row)
    return rows, delta_rows


def _transition_rows(
    primary_rows: list[Mapping[str, Any]], key: tuple[int, int]
) -> list[dict[str, Any]]:
    """Flatten per-arm transition comparisons for the analysis CSV."""
    rows: list[dict[str, Any]] = []
    for primary in primary_rows:
        for transition_key, transition in primary["transitions"].items():
            tau_interval = transition["tau_interval"]
            ntp_tau_interval = transition["ntp_tau_interval"]
            delta_vs_ntp = transition["delta_tau_interval_vs_ntp"]
            rows.append(
                {
                    "grammar_seed": key[0],
                    "model_seed": key[1],
                    "target": primary["target"],
                    "transition": transition_key,
                    "from_level": transition["from_level"],
                    "to_level": transition["to_level"],
                    "tau_interval": _direct_difference_value(tau_interval),
                    "tau_interval_status": tau_interval["status"],
                    "ntp_tau_interval": _direct_difference_value(ntp_tau_interval),
                    "ntp_tau_interval_status": ntp_tau_interval["status"],
                    "delta_tau_interval_vs_ntp": _direct_difference_value(delta_vs_ntp),
                    "delta_tau_interval_vs_ntp_status": delta_vs_ntp["status"],
                }
            )
    return rows


def summarize_target_depth_data(
    arms: list[dict[str, Any]],
    *,
    primary_levels: list[int] | None = None,
    screen_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate already-loaded arm artifacts, grouped into paired runs."""
    if not arms:
        raise ValueError("target-depth screen contains no arm directories")
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for arm in arms:
        groups.setdefault((arm["grammar_seed"], arm["model_seed"]), []).append(arm)

    output_groups: list[dict[str, Any]] = []
    all_validation_rows: list[dict[str, Any]] = []
    all_delta_rows: list[dict[str, Any]] = []
    all_transition_rows: list[dict[str, Any]] = []
    requested_levels = list(primary_levels) if primary_levels is not None else None
    selected_levels: list[int] | None = requested_levels
    chosen_level_sets: list[tuple[int, ...]] = []
    for key in sorted(groups):
        group_arms = sorted(
            groups[key], key=lambda arm: (-1 if arm["target_layer"] is None else arm["target_layer"])
        )
        _validate_paired_group(group_arms, key)
        available_levels = sorted(int(level) for level in group_arms[0]["summary"]["levels"])
        group_levels = selected_levels
        if group_levels is None:
            group_levels = [level for level in (2, 3) if level in available_levels]
            if not group_levels:
                group_levels = available_levels
        missing = [level for level in group_levels if level not in available_levels]
        if missing:
            raise ValueError(
                f"paired group {_group_label(key)} has no requested hierarchy level(s): {missing}"
            )
        chosen_level_sets.append(tuple(group_levels))
        transition_pairs = _adjacent_level_pairs(available_levels)
        ntp = next(arm for arm in group_arms if arm["arm"] == "ntp")
        primary = [
            _primary_row(arm, ntp, group_levels, transition_pairs)
            for arm in group_arms
        ]
        transition_rows = _transition_rows(primary, key)
        validation_rows, delta_rows = _validation_rows(group_arms, key)
        all_validation_rows.extend(validation_rows)
        all_delta_rows.extend(delta_rows)
        all_transition_rows.extend(transition_rows)
        output_groups.append(
            {
                "grammar_seed": key[0],
                "model_seed": key[1],
                "primary_levels": group_levels,
                "diagnostic_levels": available_levels,
                "transition_pairs": [
                    _transition_key(lower, upper)
                    for lower, upper in transition_pairs
                ],
                "checkpoint_steps": group_arms[0]["summary"]["checkpoint_steps"],
                "targets_present": [arm["arm"] for arm in group_arms],
                "protocol": {
                    "per_epoch_train_pool": group_arms[0]["metrics"].get("per_epoch_train_pool"),
                    "total_optimizer_updates": group_arms[0]["metrics"].get("total_optimizer_updates"),
                    "total_sequence_draws": group_arms[0]["metrics"].get("total_sequence_draws"),
                    "total_predicted_tokens": group_arms[0]["metrics"].get("total_predicted_tokens"),
                    "resample_train_each_epoch": group_arms[0]["config"].get("data", {}).get(
                        "resample_train_each_epoch"
                    ),
                },
                "arms": [
                    {
                        "target": arm["arm"],
                        "target_layer": arm["target_layer"],
                        "run_dir": arm["run_dir"],
                        "config": arm["config"],
                        "metrics": {
                            "best_val_ce": arm["metrics"]["best_val_ce"],
                            "history": arm["history"],
                        },
                        "summary": arm["summary"],
                    }
                    for arm in group_arms
                ],
                "primary": primary,
                "transition_intervals": transition_rows,
                "validation_ce_delta_vs_ntp": delta_rows,
            }
        )

    if requested_levels is not None:
        reported_levels: list[int] | None = requested_levels
    elif chosen_level_sets and all(
        levels == chosen_level_sets[0] for levels in chosen_level_sets
    ):
        reported_levels = list(chosen_level_sets[0])
    else:
        reported_levels = None

    return {
        "schema_version": SCHEMA_VERSION,
        "screen_dir": str(screen_dir) if screen_dir is not None else None,
        "primary_levels": reported_levels,
        "groups": output_groups,
        "validation_ce_by_step": all_validation_rows,
        "validation_ce_delta_vs_ntp": all_delta_rows,
        "transition_intervals": all_transition_rows,
    }


def summarize_target_depth(
    screen_dir: str | Path,
    output_dir: str | Path,
    *,
    rule_path: str | Path | None = None,
    primary_levels: list[int] | None = None,
) -> dict[str, Any]:
    screen_path = Path(screen_dir)
    run_dirs = _discover_arm_dirs(screen_path)
    if not run_dirs:
        raise ValueError(
            f"no grammar_*/model_*/<arm>/metrics.json files found under {screen_path}"
        )
    if rule_path is None:
        raise ValueError("--rule is required so the primary accessibility rule is explicit")
    rule = load_acquisition_rule(rule_path)
    arms = [_load_arm(run_dir, rule) for run_dir in run_dirs]
    result = summarize_target_depth_data(
        arms, primary_levels=primary_levels, screen_dir=screen_path
    )
    if rule is not None:
        result["rule"] = rule
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    disk_result = deepcopy(result)
    for group in disk_result["groups"]:
        for arm in group["arms"]:
            arm.pop("summary", None)
            arm.pop("config", None)
    (output / "comparison.json").write_text(
        json.dumps(disk_result, indent=2) + "\n", encoding="utf-8"
    )

    rows = result["validation_ce_by_step"]
    names = sorted(
        {
            name
            for row in rows
            for name in row
            if name not in {"grammar_seed", "model_seed", "step"}
        },
        key=lambda name: (-1 if name == "ntp" else int(name.split("_")[-1])),
    )
    with (output / "validation_ce_by_step.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["grammar_seed", "model_seed", "step", *names],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    with (output / "validation_ce_delta_vs_ntp.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["grammar_seed", "model_seed", "step", *names],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(result["validation_ce_delta_vs_ntp"])
    with (output / "transition_intervals.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "grammar_seed",
                "model_seed",
                "target",
                "transition",
                "from_level",
                "to_level",
                "tau_interval",
                "tau_interval_status",
                "ntp_tau_interval",
                "ntp_tau_interval_status",
                "delta_tau_interval_vs_ntp",
                "delta_tau_interval_vs_ntp_status",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(result["transition_intervals"])
    print(output / "comparison.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", required=True, help="completed Stage-02 screen directory")
    parser.add_argument("--rule", required=True, help="committed acquisition_rule.json")
    parser.add_argument("--output-dir", required=True, help="directory for comparison JSON/CSV")
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=None,
        help="hierarchy levels for the primary table (default: H2 and H3 when present)",
    )
    args = parser.parse_args()
    summarize_target_depth(
        args.screen_dir, args.output_dir, rule_path=args.rule, primary_levels=args.levels
    )


if __name__ == "__main__":
    main()
