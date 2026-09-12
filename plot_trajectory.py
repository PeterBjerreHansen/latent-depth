#!/usr/bin/env python3
"""Plot NTP and latent diagnostics over checkpoints from one training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from summarize_trajectory import load_acquisition_rule


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _levels(records: list[dict[str, Any]]) -> list[str]:
    for record in records:
        probe = record["diagnostics"].get("linear_probe")
        if probe is not None:
            return list(probe["by_level"])
    raise RuntimeError("trajectory contains no linear-probe diagnostics")


def _heatmap(
    records: list[dict[str, Any]],
    metric: str,
    levels: list[str],
    *,
    balanced: bool = False,
    q_epsilon: float = 1e-8,
) -> tuple[np.ndarray, list[int], int]:
    blocks: list[np.ndarray] = []
    steps: list[int] = []
    for record in records:
        payload_metric = "synonym_clustering" if metric == "q" else metric
        payload = record["diagnostics"].get(payload_metric)
        if payload is None:
            raise RuntimeError(f"trajectory checkpoint has no {metric} diagnostics")
        values_by_level = []
        for level in levels:
            entry = payload["by_level"][level]
            if metric == "linear_probe":
                key = "balanced_accuracy_by_layer" if balanced else "accuracy_by_layer"
            elif metric == "synonym_clustering":
                key = "score_by_layer"
            elif metric == "variable_sensitivity":
                key = "sensitivity_by_layer"
            elif metric == "q":
                synonym = np.asarray(entry["synonym_distance_by_layer"], dtype=float)
                variable_entry = record["diagnostics"]["variable_sensitivity"]["by_level"][level]
                variable = np.asarray(variable_entry["variable_distance_by_layer"], dtype=float)
                non_synonym = np.asarray(
                    variable_entry["non_synonym_distance_by_layer"], dtype=float
                )
                values_by_level.append((variable - synonym) / (non_synonym + q_epsilon))
            else:
                raise ValueError(f"unsupported trajectory metric: {metric}")
            if metric != "q":
                values_by_level.append(np.asarray(entry[key], dtype=float))
        layer_values = np.stack(values_by_level, axis=1)
        if blocks and layer_values.shape != blocks[0].shape:
            raise RuntimeError(f"inconsistent {metric} layer/level shape across checkpoints")
        blocks.append(layer_values)
        steps.append(int(record["checkpoint_global_step"]))
    if not blocks:
        raise RuntimeError(f"trajectory contains no {metric} diagnostics")
    return np.concatenate(blocks, axis=0), steps, blocks[0].shape[0]


def _draw_heatmap(
    axis: Any,
    records: list[dict[str, Any]],
    metric: str,
    levels: list[str],
    *,
    title: str,
    balanced: bool = False,
    q_epsilon: float = 1e-8,
) -> None:
    matrix, steps, layers = _heatmap(
        records, metric, levels, balanced=balanced, q_epsilon=q_epsilon
    )
    if metric == "linear_probe":
        vmin, vmax = 0.0, 1.0
    elif metric == "q":
        magnitude = float(np.nanmax(np.abs(matrix)))
        magnitude = max(magnitude, 1e-8)
        vmin, vmax = -magnitude, magnitude
    else:
        vmin = float(np.nanmin(matrix))
        vmax = float(np.nanmax(matrix))
        if vmin == vmax:
            padding = max(abs(vmin) * 0.05, 1e-6)
            vmin -= padding
            vmax += padding
    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    centers = np.arange(len(steps), dtype=float) * layers + (layers - 1) / 2
    axis.set_yticks(centers)
    axis.set_yticklabels([str(step) for step in steps])
    axis.set_xticks(np.arange(len(levels)))
    axis.set_xticklabels([f"r={level}" for level in levels])
    axis.set_ylabel("checkpoint step\n(rows = embedding, blocks)")
    axis.set_title(title)
    axis.grid(axis="y", which="minor", alpha=0.2)
    axis.figure.colorbar(image, ax=axis, pad=0.01)


def _position_history(history: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Return validation NLL by prediction position over training age."""
    rows = [row for row in history if "val_nll_by_position" in row]
    if not rows:
        raise RuntimeError("metrics history contains no per-position validation NLL")
    widths = {len(row["val_nll_by_position"]) for row in rows}
    if len(widths) != 1:
        raise RuntimeError("metrics history has inconsistent per-position NLL lengths")
    steps = np.asarray([int(row["global_step"]) for row in rows])
    matrix = np.asarray(
        [[float(value) for value in row["val_nll_by_position"]] for row in rows],
        dtype=float,
    )
    return matrix, steps


def plot_trajectory(
    trajectory: str | Path,
    metrics: str | Path,
    output: str | Path,
    *,
    q_epsilon: float = 1e-8,
) -> None:
    """Write NTP curves and layer-by-level diagnostic heatmaps."""
    trajectory_data = _load(Path(trajectory))
    records = trajectory_data["checkpoints"]
    if not records:
        raise RuntimeError("trajectory contains no checkpoint records")
    metric_data = _load(Path(metrics))
    history = metric_data.get("history", [])
    levels = _levels(records)

    figure, axes = plt.subplots(5, 1, figsize=(10.0, 20.0))
    history_steps = np.asarray([int(row["global_step"]) for row in history])
    val_ce = np.asarray([float(row["val_ce"]) for row in history])
    val_last = np.asarray([float(row["val_last_position_nll"]) for row in history])
    axes[0].plot(history_steps, val_ce, marker="o", label="validation CE")
    axes[0].plot(history_steps, val_last, marker="o", label="validation final-position NLL")
    axes[0].set_ylabel("NLL")
    axes[0].set_title("Vanilla NTP training-age trajectory")
    axes[0].legend(fontsize=8)

    try:
        position_nll, position_steps = _position_history(history)
    except RuntimeError as exc:
        axes[1].text(0.5, 0.5, str(exc), ha="center", va="center")
        axes[1].set_axis_off()
    else:
        image = axes[1].imshow(position_nll, aspect="auto", interpolation="nearest")
        axes[1].set_yticks(np.arange(len(position_steps)))
        axes[1].set_yticklabels([str(step) for step in position_steps])
        axes[1].set_xticks(np.arange(position_nll.shape[1]))
        axes[1].set_xticklabels(
            [str(position) for position in range(1, position_nll.shape[1] + 1)]
        )
        axes[1].set_ylabel("optimizer step")
        axes[1].set_xlabel("prediction position")
        axes[1].set_title("Validation NLL by prediction position")
        axes[1].figure.colorbar(image, ax=axes[1], pad=0.01)

    _draw_heatmap(
        axes[2],
        records,
        "linear_probe",
        levels,
        title="Balanced latent accessibility by layer and level",
        balanced=True,
    )
    _draw_heatmap(
        axes[3],
        records,
        "synonym_clustering",
        levels,
        title="Synonym invariance by layer and level",
    )
    _draw_heatmap(
        axes[4],
        records,
        "q",
        levels,
        title="Normalized intervention contrast Q by layer and level",
        q_epsilon=q_epsilon,
    )
    axes[4].set_xlabel("hierarchy level")

    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    print(output_path)


def _save_figure(figure: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _group_directory(output_dir: Path, group: dict[str, Any], multiple: bool) -> Path:
    if not multiple:
        return output_dir
    return output_dir / f"grammar_{group['grammar_seed']}_model_{group['model_seed']}"


def _plot_acquisition_bars(group: dict[str, Any], output_dir: Path) -> None:
    rows = group["primary"]
    labels = [row["target"] for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(11.0, 8.0), sharex=True)
    for axis, value_name, title in (
        (axes[0], "tau_2", "Absolute acquisition time"),
        (axes[1], "delta_tau_2", "Acquisition-time difference versus NTP"),
    ):
        values = [row.get(value_name) for row in rows]
        plotted = [np.nan if value is None else float(value) for value in values]
        axis.plot(labels, plotted, marker="o", label="H2")
        if value_name == "tau_2":
            other = [row.get("tau_3") for row in rows]
            axis.plot(
                labels,
                [np.nan if value is None else float(value) for value in other],
                marker="o",
                label="H3",
            )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.set_ylabel("updates")
        axis.set_title(title)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    axes[1].set_xlabel("target arm")
    _save_figure(figure, output_dir / "acquisition_times.png")

    # Keep the delta plot as a separate artifact because absolute time and a
    # paired difference answer different scientific questions.
    figure, axis = plt.subplots(figsize=(11.0, 4.5))
    for level, color in ((2, "tab:blue"), (3, "tab:orange")):
        values = [row.get(f"delta_tau_{level}") for row in rows]
        axis.plot(
            labels,
            [np.nan if value is None else float(value) for value in values],
            marker="o",
            color=color,
            label=f"H{level}",
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("target arm")
    axis.set_ylabel("updates versus NTP")
    axis.set_title("Paired acquisition-time differences")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    _save_figure(figure, output_dir / "delta_tau.png")


def _plot_validation_ce(group: dict[str, Any], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.0, 5.0))
    for arm in group["arms"]:
        history = arm["metrics"]["history"]
        axis.plot(
            [int(row["global_step"]) for row in history],
            [float(row["val_ce"]) for row in history],
            marker="o",
            markersize=3,
            label=arm["target"],
        )
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("validation CE")
    axis.set_title("Matched-update validation cross-entropy")
    axis.legend(fontsize=8, ncol=2)
    axis.grid(alpha=0.25)
    _save_figure(figure, output_dir / "validation_ce.png")


def _plot_level_curves(
    group: dict[str, Any],
    output_dir: Path,
    level: int,
    metric: str,
    title: str,
    *,
    limits: tuple[float, float] | None = None,
) -> None:
    arms = group["arms"]
    ntp = next(arm for arm in arms if arm["target"] == "ntp")
    level_summary = ntp["summary"]["levels"].get(str(level))
    file_metric = {
        "balanced_accuracy": "accessibility",
        "clustering": "clustering",
        "q": "q",
    }.get(metric, metric)
    figure, axis = plt.subplots(figsize=(11.0, 5.0))
    if level_summary is None or level_summary["emergence"]["status"] != "observed":
        axis.text(
            0.5,
            0.5,
            f"NTP has no observed H{level} onset layer",
            ha="center",
            va="center",
        )
        axis.set_axis_off()
        _save_figure(figure, output_dir / f"h{level}_{file_metric}.png")
        return
    layer = int(level_summary["emergence"]["observer_layer"])
    curves: list[np.ndarray] = []
    labels: list[str] = []
    steps = np.asarray(group["checkpoint_steps"], dtype=int)
    for arm in arms:
        layer_data = arm["summary"]["levels"][str(level)]["layerwise"][str(layer)]
        curve = np.asarray(layer_data[metric], dtype=float)
        if curve.shape != steps.shape:
            raise ValueError(
                f"comparison arm {arm['target']} has an invalid H{level} {metric} curve"
            )
        curves.append(curve)
        labels.append(arm["target"])
    matrix = np.stack(curves)
    if limits is None:
        if metric == "balanced_accuracy":
            limits = (0.0, 1.0)
        elif metric == "q":
            magnitude = max(float(np.nanmax(np.abs(matrix))), 1e-8)
            limits = (-magnitude, magnitude)
        else:
            low, high = float(np.nanmin(matrix)), float(np.nanmax(matrix))
            if low == high:
                padding = max(abs(low) * 0.05, 1e-6)
                low -= padding
                high += padding
            limits = (low, high)
    axis.set_ylim(*limits)
    for curve, label in zip(curves, labels):
        axis.plot(steps, curve, marker="o", markersize=3, label=label)
    axis.set_xlabel("optimizer step")
    axis.set_ylabel(metric)
    axis.set_title(f"H{level}: {title} at NTP onset layer {layer}")
    axis.legend(fontsize=8, ncol=2)
    axis.grid(alpha=0.25)
    _save_figure(figure, output_dir / f"h{level}_{file_metric}.png")


def _plot_layerwise_onsets(group: dict[str, Any], output_dir: Path) -> None:
    onset_dir = output_dir / "layerwise_onsets"
    onset_dir.mkdir(parents=True, exist_ok=True)
    for level in group["primary_levels"]:
        rows = []
        labels = []
        for arm in group["arms"]:
            summary = arm["summary"]["levels"].get(str(level))
            if summary is None:
                continue
            labels.append(arm["target"])
            values = []
            for layer in sorted(summary["layerwise_onsets"], key=int):
                event = summary["layerwise_onsets"][layer]
                values.append(
                    float(event.get("onset_step", event.get("through_step")))
                )
            rows.append(values)
        if not rows:
            continue
        figure, axis = plt.subplots(figsize=(9.0, max(3.0, 0.45 * len(rows))))
        image = axis.imshow(np.asarray(rows), aspect="auto", interpolation="nearest")
        axis.set_yticks(np.arange(len(labels)))
        axis.set_yticklabels(labels)
        axis.set_xlabel("observer layer")
        axis.set_ylabel("target arm")
        axis.set_title(f"H{level} layerwise onset (censoring shown at horizon)")
        axis.figure.colorbar(image, ax=axis, pad=0.01, label="onset / censoring step")
        _save_figure(figure, onset_dir / f"h{level}.png")


def _shared_curve_limits(
    groups: list[dict[str, Any]], level: int, metric: str
) -> tuple[float, float] | None:
    curves: list[np.ndarray] = []
    for group in groups:
        for arm in group["arms"]:
            level_summary = arm["summary"]["levels"].get(str(level))
            if level_summary is None:
                continue
            ntp_level = next(
                candidate["summary"]["levels"].get(str(level))
                for candidate in group["arms"]
                if candidate["target"] == "ntp"
            )
            if ntp_level is None or ntp_level["emergence"]["status"] != "observed":
                continue
            layer = str(ntp_level["emergence"]["observer_layer"])
            curves.append(np.asarray(level_summary["layerwise"][layer][metric], dtype=float))
    if not curves:
        return None
    matrix = np.stack(curves)
    if metric == "balanced_accuracy":
        return 0.0, 1.0
    if metric == "q":
        magnitude = max(float(np.nanmax(np.abs(matrix))), 1e-8)
        return -magnitude, magnitude
    low, high = float(np.nanmin(matrix)), float(np.nanmax(matrix))
    if low == high:
        padding = max(abs(low) * 0.05, 1e-6)
        low -= padding
        high += padding
    return low, high


def plot_target_depth_comparison(
    comparison: str | Path | Mapping[str, Any], output_dir: str | Path
) -> None:
    """Write shared-scale sweep figures from ``comparison.json``."""
    if isinstance(comparison, (str, Path)):
        comparison_data = _load(Path(comparison))
    else:
        comparison_data = dict(comparison)
    groups = comparison_data.get("groups", [])
    if not groups:
        raise RuntimeError("comparison contains no paired groups")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    multiple = len(groups) > 1
    shared_limits = {
        (level, metric): _shared_curve_limits(groups, level, metric)
        for level in (2, 3)
        for metric in ("balanced_accuracy", "clustering", "q")
    }
    for group in groups:
        group_output = _group_directory(output, group, multiple)
        _plot_acquisition_bars(group, group_output)
        _plot_validation_ce(group, group_output)
        for level in (2, 3):
            _plot_level_curves(
                group,
                group_output,
                level,
                "balanced_accuracy",
                "balanced accessibility",
                limits=shared_limits[(level, "balanced_accuracy")],
            )
            _plot_level_curves(
                group,
                group_output,
                level,
                "clustering",
                "synonym invariance C",
                limits=shared_limits[(level, "clustering")],
            )
            _plot_level_curves(
                group,
                group_output,
                level,
                "q",
                "intervention contrast Q",
                limits=shared_limits[(level, "q")],
            )
        _plot_layerwise_onsets(group, group_output)


def _main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trajectory", help="trajectory.json from diagnose_trajectory.py")
    source.add_argument("--comparison", help="comparison.json from summarize_target_depth.py")
    parser.add_argument("--metrics", help="metrics.json from the same training run")
    parser.add_argument("--output", help="one-run PNG output path")
    parser.add_argument("--output-dir", help="directory for sweep comparison PNGs")
    parser.add_argument(
        "--rule",
        help="acquisition_rule.json (used for the Q numerical stabilizer in one-run plots)",
    )
    return parser


def main() -> None:
    parser = _main_parser()
    args = parser.parse_args()
    if args.trajectory is not None:
        if args.metrics is None or args.output is None:
            parser.error("--trajectory requires --metrics and --output")
        q_epsilon = 1e-8 if args.rule is None else load_acquisition_rule(args.rule)["epsilon"]
        plot_trajectory(args.trajectory, args.metrics, args.output, q_epsilon=q_epsilon)
    else:
        if args.output_dir is None:
            parser.error("--comparison requires --output-dir")
        plot_target_depth_comparison(args.comparison, args.output_dir)


if __name__ == "__main__":
    main()
