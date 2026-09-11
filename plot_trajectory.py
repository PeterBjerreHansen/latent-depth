#!/usr/bin/env python3
"""Plot NTP and latent diagnostics over checkpoints from one training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


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
) -> tuple[np.ndarray, list[int], int]:
    blocks: list[np.ndarray] = []
    steps: list[int] = []
    for record in records:
        payload = record["diagnostics"].get(metric)
        if payload is None:
            raise RuntimeError(f"trajectory checkpoint has no {metric} diagnostics")
        values_by_level = []
        for level in levels:
            entry = payload["by_level"][level]
            if metric == "linear_probe":
                key = "balanced_accuracy_by_layer" if balanced else "accuracy_by_layer"
            elif metric == "synonym_clustering":
                key = "score_by_layer"
            else:
                key = "sensitivity_by_layer"
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
) -> None:
    matrix, steps, layers = _heatmap(records, metric, levels, balanced=balanced)
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest")
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
        "variable_sensitivity",
        levels,
        title="Latent-replacement sensitivity by layer and level",
    )
    axes[4].set_xlabel("hierarchy level")

    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True, help="trajectory.json from diagnose_trajectory.py")
    parser.add_argument("--metrics", required=True, help="metrics.json from the same training run")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot_trajectory(args.trajectory, args.metrics, args.output)


if __name__ == "__main__":
    main()
