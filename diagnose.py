#!/usr/bin/env python3
"""Run latent diagnostics on a saved RHM causal-LM checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from diagnostics import run_latent_diagnostics, run_probe_control
from rhm.dataset import RHMSplit
from rhm.random_hierarchy_model import sample_rules, sample_trees
from training import build_model, load_model_from_checkpoint, resolve_device


def diagnose_checkpoint(
    checkpoint: str | Path,
    *,
    split: str = "val",
    device: str = "auto",
    metric: str = "all",
    num_sequences: int | None = None,
    probe_steps: int | None = None,
    controls: bool = False,
) -> dict[str, Any]:
    """Re-run the same observer diagnostics used during training."""
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    if metric not in {"all", "probe", "clustering"}:
        raise ValueError("metric must be 'all', 'probe', or 'clustering'")

    resolved_device = resolve_device(device)
    model, cfg, checkpoint_data = load_model_from_checkpoint(checkpoint, device=str(resolved_device))
    cfg.diagnostics.enabled = True
    cfg.diagnostics.linear_probe = metric in {"all", "probe"}
    cfg.diagnostics.synonym_clustering = metric in {"all", "clustering"}
    if num_sequences is not None:
        cfg.diagnostics.num_sequences = num_sequences
    if probe_steps is not None:
        cfg.diagnostics.probe_steps = probe_steps
    cfg.validate()

    rules = checkpoint_data.get("rules")
    if rules is None:
        rules = sample_rules(
            v=cfg.rhm.v,
            n=cfg.rhm.n,
            m=cfg.rhm.m,
            s=cfg.rhm.s,
            L=cfg.rhm.L,
            seed=cfg.rhm.rule_seed,
        )

    configured_size = cfg.data.val_size if split == "val" else cfg.data.test_size
    sample_seed = cfg.rhm.val_seed if split == "val" else cfg.rhm.test_seed
    # Recreate the configured split size before diagnostics select their first
    # N examples. RHM sampling uses shape-dependent tensor RNG kernels, so
    # sampling N directly is not guaranteed to match slicing the full split.
    trees, choices = sample_trees(configured_size, rules, seed=sample_seed, return_choices=True)
    diagnostic_split = RHMSplit(trees=trees, choices=choices)
    result = run_latent_diagnostics(model, diagnostic_split, rules, cfg, resolved_device)
    if controls and cfg.diagnostics.linear_probe:
        result["probe_controls"] = {
            "trained_backbone_shuffled_labels": run_probe_control(
                model, diagnostic_split, cfg, resolved_device, shuffle_labels=True
            ),
        }
        rng_state = torch.get_rng_state()
        mps = getattr(torch, "mps", None)
        mps_state = None
        if mps is not None and resolved_device.type == "mps" and hasattr(mps, "get_rng_state"):
            mps.synchronize()
            mps_state = mps.get_rng_state().cpu().clone()
        try:
            torch.manual_seed(int(cfg.model_seed) + 91_337)
            untrained = build_model(cfg).to(resolved_device)
        finally:
            torch.set_rng_state(rng_state)
            if mps_state is not None and hasattr(mps, "set_rng_state"):
                mps.set_rng_state(mps_state)
        result["probe_controls"]["untrained_backbone_real_labels"] = run_probe_control(
            untrained, diagnostic_split, cfg, resolved_device, shuffle_labels=False
        )
    return {
        "checkpoint": str(Path(checkpoint)),
        "checkpoint_global_step": int(checkpoint_data.get("global_step", -1)),
        "checkpoint_epoch": int(checkpoint_data.get("epoch", -1)),
        "split": split,
        "rule_seed": cfg.rhm.rule_seed,
        "model_seed": cfg.model_seed,
        "diagnostics": result,
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
    parser.add_argument("--controls", action="store_true", help="include shuffled-label and untrained probes")
    args = parser.parse_args()

    payload = diagnose_checkpoint(
        args.checkpoint,
        split=args.split,
        device=args.device,
        metric=args.metric,
        num_sequences=args.num_sequences,
        probe_steps=args.probe_steps,
        controls=args.controls,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(output)


if __name__ == "__main__":
    main()
