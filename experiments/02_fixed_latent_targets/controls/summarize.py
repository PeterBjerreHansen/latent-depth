#!/usr/bin/env python3
"""Summarize and plot the compact Stage-02 control result bundle."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    preferred = [
        "baseline",
        "level",
        "model_seed",
        "probe_seed",
        "fit_examples",
        "observer_layer",
        "prefix_length",
        "lookup_coverage",
        "accuracy",
        "balanced_accuracy",
        "ce",
        "represented_classes",
        "balanced_majority_accuracy",
    ]
    keys = set().union(*(row.keys() for row in rows))
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _max_fit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    maximum = max(int(row["fit_examples"]) for row in rows)
    return [row for row in rows if int(row["fit_examples"]) == maximum]


def _summarize_by_fit_and_level(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_run: dict[tuple[str, int, int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        per_run[
            (
                str(row["baseline"]),
                int(row["level"]),
                int(row["fit_examples"]),
                str(row.get("model_seed", "")),
                str(row.get("probe_seed", "")),
            )
        ].append(float(row["balanced_accuracy"]))
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for (baseline, level, fit_examples, _model_seed, _probe_seed), values in per_run.items():
        grouped[(baseline, level, fit_examples)].append(max(values))
    return [
        {
            "baseline": baseline,
            "level": level,
            "fit_examples": fit_examples,
            "mean_balanced_accuracy": statistics.fmean(values),
            "min_balanced_accuracy": min(values),
            "max_balanced_accuracy": max(values),
            "n": len(values),
        }
        for (baseline, level, fit_examples), values in sorted(grouped.items())
    ]


def _plot_sample_complexity(
    random_rows: list[dict[str, Any]], surface_rows: list[dict[str, Any]], *, output: Path, vocab_size: int
) -> None:
    levels = sorted({int(row["level"]) for row in random_rows + surface_rows})
    if not levels:
        return
    figure, axes = plt.subplots(1, len(levels), figsize=(6 * len(levels), 4), squeeze=False)
    for axis, level in zip(axes[0], levels):
        for baseline, rows in (
            ("random_features", random_rows),
            ("surface_linear", surface_rows),
            ("surface_mlp", surface_rows),
            ("surface_lookup", surface_rows),
        ):
            selected = [
                row
                for row in rows
                if int(row["level"]) == level and str(row["baseline"]) == baseline
            ]
            summary = [
                row
                for row in _summarize_by_fit_and_level(selected)
                if row["baseline"] == baseline
            ]
            if not summary:
                continue
            axis.plot(
                [row["fit_examples"] for row in summary],
                [row["mean_balanced_accuracy"] for row in summary],
                marker="o",
                label=baseline,
            )
        axis.axhline(1.0 / vocab_size, color="black", linestyle="--", linewidth=0.8, label="chance")
        axis.set_title(f"H{level}")
        axis.set_xlabel("probe fitting examples")
        axis.set_ylabel("balanced accuracy")
        axis.set_xscale("symlog", linthresh=1)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(output / "sample_complexity.png", dpi=160)
    plt.close(figure)


def _plot_random_backbones(random_rows: list[dict[str, Any]], *, output: Path) -> None:
    rows = _max_fit_rows(random_rows)
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["level"]), int(row["model_seed"]))].append(
            float(row["balanced_accuracy"])
        )
    levels = sorted({level for level, _ in grouped})
    if not levels:
        return
    figure, axis = plt.subplots(figsize=(7, 4))
    for level_index, level in enumerate(levels):
        values = [
            max(grouped[(level, seed)])
            for seed in sorted(seed for current_level, seed in grouped if current_level == level)
        ]
        axis.scatter([level_index] * len(values), values, label=f"H{level}")
    axis.set_xticks(range(len(levels)), [f"H{level}" for level in levels])
    axis.set_ylabel("maximum balanced accuracy at largest fit size")
    axis.set_ylim(0, 1.02)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "random_backbone_replicates.png", dpi=160)
    plt.close(figure)


def _summarize_invariance(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in runs:
        for row in run.get("records", []):
            key = (
                str(run["arm"]),
                int(run["step"]),
                int(row["level"]),
                int(row["observer_layer"]),
            )
            grouped[key]["clustering"].append(float(row["clustering"]))
            grouped[key]["q"].append(float(row["q"]))
    rows: list[dict[str, Any]] = []
    for (arm, step, level, layer), values in sorted(grouped.items()):
        row: dict[str, Any] = {
            "arm": arm,
            "step": step,
            "level": level,
            "observer_layer": layer,
            "replicates": len(values["q"]),
        }
        for name, observations in values.items():
            row[f"{name}_mean"] = statistics.fmean(observations)
            row[f"{name}_stdev"] = statistics.stdev(observations) if len(observations) > 1 else 0.0
            row[f"{name}_min"] = min(observations)
            row[f"{name}_max"] = max(observations)
        rows.append(row)
    return rows


def summarize(input_path: str | Path, output_dir: str | Path) -> Path:
    input_path = Path(input_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    random_rows = list(payload.get("random_feature_rows", []))
    surface_rows = list(payload.get("surface_rows", []))
    shuffled_rows = list(payload.get("shuffled_rows", []))
    _write_csv(output / "random_backbone_replicates.csv", random_rows)
    _write_csv(output / "surface_baselines.csv", surface_rows)
    _write_csv(output / "shuffled_labels.csv", shuffled_rows)
    (output / "h1_oracle.json").write_text(
        json.dumps(payload.get("oracles", []), indent=2) + "\n", encoding="utf-8"
    )
    (output / "invariance_replicates.json").write_text(
        json.dumps(payload.get("invariance", []), indent=2) + "\n", encoding="utf-8"
    )
    invariance_summary = _summarize_invariance(payload.get("invariance", []))
    _write_csv(output / "invariance_summary.csv", invariance_summary)

    protocol = payload.get("protocol", {})
    cfg = protocol.get("experiment", {})
    vocab_size = int(cfg.get("rhm", {}).get("v", 1))
    summary = {
        "schema_version": 1,
        "source": str(input_path),
        "random_features_by_fit_and_level": _summarize_by_fit_and_level(random_rows),
        "surface_baselines_by_fit_and_level": _summarize_by_fit_and_level(surface_rows),
        "shuffled_labels_by_fit_and_level": _summarize_by_fit_and_level(shuffled_rows),
        "oracles": payload.get("oracles", []),
        "invariance_summary": invariance_summary,
        "invariance_runs": [
            {
                "arm": item.get("arm"),
                "step": item.get("step"),
                "num_sequences": item.get("num_sequences"),
                "replicates": item.get("replicates"),
                "levels": item.get("levels"),
            }
            for item in payload.get("invariance", [])
        ],
        "seed_zero_reference": payload.get("seed_zero_reference"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    _plot_sample_complexity(random_rows, surface_rows, output=plots, vocab_size=vocab_size)
    _plot_random_backbones(random_rows, output=plots)
    print(output / "summary.json")
    return output / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw_controls.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summarize(args.input, args.output_dir)


if __name__ == "__main__":
    main()
