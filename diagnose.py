#!/usr/bin/env python3
"""Run latent diagnostics on a saved RHM causal-LM checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from config import DiagnosticsConfig
from diagnostics import run_latent_diagnostics, run_probe_control
from rhm.dataset import RHMSplit
from rhm.random_hierarchy_model import sample_trees
from training import load_model_from_checkpoint, resolve_device, artifact_rules


def diagnose_checkpoint(
    checkpoint: str | Path,
    *,
    split: str = "val",
    device: str = "auto",
    metric: str = "all",
    num_sequences: int | None = None,
    probe_steps: int | None = None,
    controls: bool = False,
    probe_seed: int = 12345,
    probe_lr: float = 1e-3,
) -> dict[str, Any]:
    """Measure a frozen backbone using independent offline probe settings."""
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    if metric not in {"all", "probe", "clustering"}:
        raise ValueError("metric must be 'all', 'probe', or 'clustering'")

    resolved_device = resolve_device(device)
    model, cfg, checkpoint_data = load_model_from_checkpoint(checkpoint, device=str(resolved_device))
    settings = DiagnosticsConfig(
        linear_probe=metric in {"all", "probe"},
        synonym_clustering=metric in {"all", "clustering"},
        num_sequences=num_sequences if num_sequences is not None else 1024,
        probe_steps=probe_steps if probe_steps is not None else 300,
        seed=probe_seed, probe_lr=probe_lr,
    )
    settings.validate()
    rules = artifact_rules(checkpoint, checkpoint_data)

    configured_size = cfg.data.val_size if split == "val" else cfg.data.test_size
    sample_seed = cfg.rhm.val_seed if split == "val" else cfg.rhm.test_seed
    # Recreate the configured split size before diagnostics select their first
    # N examples. RHM sampling uses shape-dependent tensor RNG kernels, so
    # sampling N directly is not guaranteed to match slicing the full split.
    trees, choices = sample_trees(configured_size, rules, seed=sample_seed, return_choices=True)
    diagnostic_split = RHMSplit(trees=trees, choices=choices)
    result = run_latent_diagnostics(model, diagnostic_split, rules, cfg, resolved_device, settings=settings)
    if controls and settings.linear_probe:
        result["probe_controls"] = {
            "trained_backbone_shuffled_labels": run_probe_control(
                model, diagnostic_split, cfg, resolved_device, shuffle_labels=True, settings=settings
            ),
        }
    return {
        "checkpoint": str(Path(checkpoint)),
        "checkpoint_global_step": int(checkpoint_data.get("global_step", -1)),
        "checkpoint_epoch": int(checkpoint_data.get("epoch", -1)),
        "split": split,
        "rule_seed": cfg.rhm.rule_seed,
        "model_seed": cfg.model_seed,
        "diagnostics": result,
        "diagnostic_config": asdict(settings),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="checkpoint produced by training.py")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metric", choices=("all", "probe", "clustering"), default="all")
    parser.add_argument("--num-sequences", type=int, default=None)
    parser.add_argument("--probe-steps", type=int, default=None)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--controls", action="store_true", help="include the shuffled-label probe")
    args = parser.parse_args()

    payload = diagnose_checkpoint(
        args.checkpoint,
        split=args.split,
        device=args.device,
        metric=args.metric,
        num_sequences=args.num_sequences,
        probe_steps=args.probe_steps,
        controls=args.controls, probe_seed=args.probe_seed, probe_lr=args.probe_lr,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(output)


if __name__ == "__main__":
    main()
