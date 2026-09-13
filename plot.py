#!/usr/bin/env python3
"""Plot held-out cross-entropy against training-set size P."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import SweepConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="metrics.jsonl from sweep_data_size.py")
    parser.add_argument("--config", required=True, help="same sweep config used for training")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--metric",
        default="val_ce",
        choices=("val_ce", "val_last_position_nll"),
        help="held-out metric to plot",
    )
    args = parser.parse_args()

    sweep = SweepConfig.from_json(args.config)
    grouped: dict[int, list[float]] = defaultdict(list)
    with open(args.metrics, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                grouped[int(row["train_size"])].append(float(row[args.metric]))
    if not grouped:
        raise RuntimeError("no metrics found")

    P = np.array(sorted(grouped), dtype=float)
    means = np.array([np.mean(grouped[int(p)]) for p in P])
    sems = np.array(
        [
            np.std(grouped[int(p)], ddof=1) / math.sqrt(len(grouped[int(p)]))
            if len(grouped[int(p)]) > 1
            else 0.0
            for p in P
        ]
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.errorbar(P, means, yerr=sems, marker="o", capsize=3, label="nanoGPT-style causal Transformer")
    ax.set_xscale("log")
    ax.set_xlabel("training set size P")
    ax.set_ylabel(
        "held-out final-token NLL"
        if args.metric == "val_last_position_nll"
        else "held-out next-token cross-entropy"
    )

    ax.legend(fontsize=8)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
