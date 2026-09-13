#!/usr/bin/env python3
"""Sweep fixed next-latent target depth for Stage 02."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from config import TargetDepthSweepConfig
from rhm.dataset import LeafSequenceDataset, build_rhm_bundle
from training import train_model


def _derived_seed(base: int, grammar_seed: int) -> int:
    return int((base + 1_000_003 * grammar_seed) % (2**63 - 1))


def arm_name(target_layer: int | None) -> str:
    return "ntp" if target_layer is None else f"target_{target_layer}"


def _existing_keys(metrics_path: Path) -> set[tuple[int, int, int | None]]:
    keys: set[tuple[int, int, int | None]] = set()
    if not metrics_path.exists():
        return keys
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        target = row.get("auxiliary_target_layer")
        keys.add(
            (
                int(row["rule_seed"]),
                int(row["model_seed"]),
                None if target is None else int(target),
            )
        )
    return keys


def _validate_resume(
    *, metrics_path: Path, snapshot_path: Path, snapshot: dict[str, object], resume: bool
) -> None:
    has_metrics = metrics_path.exists() and metrics_path.stat().st_size > 0
    if has_metrics and not resume:
        raise RuntimeError(
            f"{metrics_path} already contains results; use --resume or a new output directory"
        )
    if not resume:
        return
    if has_metrics and not snapshot_path.exists():
        raise RuntimeError("existing target-depth results have no sweep_config.json")
    if snapshot_path.exists():
        previous = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if previous != snapshot:
            raise RuntimeError(
                "refusing to mix results: existing sweep_config.json differs from requested config"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON target-depth sweep config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true", help="skip completed arms")
    parser.add_argument("--arms", nargs="+", help="run only these arm names from the configured screen")
    args = parser.parse_args()

    sweep = TargetDepthSweepConfig.from_json(args.config)
    if sweep.experiment.auxiliary.mode != "none":
        raise ValueError(
            "target-depth sweep base experiment must use auxiliary.mode='none'; "
            "the sweep enables it per target arm"
        )

    if args.arms and set(args.arms) - {arm_name(layer) for layer in sweep.target_layers}:
        raise ValueError("--arms contains an arm outside the configured screen")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    snapshot_path = out / "sweep_config.json"
    snapshot = {
        # Normalize tuples to JSON-native lists before comparing with the
        # previously written snapshot on resume.
        "experiment": json.loads(json.dumps(sweep.experiment.to_dict())),
        "target_layers": sweep.target_layers,
        "grammar_seeds": sweep.grammar_seeds,
        "model_seeds": sweep.model_seeds,
        "replicates": (
            [{"grammar_seed": g, "model_seed": m} for g, m in sweep.replicates]
            if sweep.replicates is not None
            else None
        ),
    }
    _validate_resume(
        metrics_path=metrics_path,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
        resume=args.resume,
    )
    completed = _existing_keys(metrics_path) if args.resume else set()
    snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")

    pairs = sweep.replicate_pairs()
    models_by_grammar: dict[int, list[int]] = {}
    for grammar_seed, model_seed in pairs:
        models_by_grammar.setdefault(grammar_seed, []).append(model_seed)

    for grammar_seed, grammar_model_seeds in models_by_grammar.items():
        cfg0 = copy.deepcopy(sweep.experiment)
        cfg0.rhm.rule_seed = grammar_seed
        cfg0.rhm.train_seed = _derived_seed(cfg0.rhm.train_seed, grammar_seed)
        cfg0.rhm.val_seed = _derived_seed(cfg0.rhm.val_seed, grammar_seed)
        cfg0.rhm.test_seed = _derived_seed(cfg0.rhm.test_seed, grammar_seed)
        cfg0.validate()

        print(f"generating grammar {grammar_seed} at P={cfg0.data.train_size}")
        bundle = build_rhm_bundle(
            v=cfg0.rhm.v,
            n=cfg0.rhm.n,
            m=cfg0.rhm.m,
            s=cfg0.rhm.s,
            L=cfg0.rhm.L,
            rule_seed=cfg0.rhm.rule_seed,
            train_seed=cfg0.rhm.train_seed,
            val_seed=cfg0.rhm.val_seed,
            test_seed=cfg0.rhm.test_seed,
            train_size=cfg0.data.train_size,
            val_size=cfg0.data.val_size,
            test_size=cfg0.data.test_size,
        )
        grammar_dir = out / f"grammar_{grammar_seed}"
        grammar_dir.mkdir(parents=True, exist_ok=True)
        train_ds = LeafSequenceDataset(bundle.train.leaves)
        val_ds = LeafSequenceDataset(bundle.val.leaves)

        for model_seed in grammar_model_seeds:
            for target_layer in sweep.target_layers:
                if args.arms and arm_name(target_layer) not in args.arms:
                    continue
                key = (grammar_seed, model_seed, target_layer)
                if key in completed:
                    print("skip completed", key)
                    continue

                cfg = copy.deepcopy(cfg0)
                cfg.model_seed = model_seed
                if target_layer is None:
                    cfg.auxiliary.mode = "none"
                    cfg.auxiliary.target_layer = None
                else:
                    cfg.auxiliary.mode = "next_latent"
                    cfg.auxiliary.target_layer = int(target_layer)
                cfg.validate()

                run_dir = grammar_dir / f"model_{model_seed}" / arm_name(target_layer)
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "config.json").write_text(
                    json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8"
                )
                print(
                    f"\n=== grammar={grammar_seed} model_seed={model_seed} "
                    f"arm={arm_name(target_layer)} ==="
                )
                metrics = train_model(
                    cfg,
                    train_ds,
                    val_ds,
                    output_dir=run_dir,
                    rules=bundle.rules,
                )
                row = {k: v for k, v in metrics.items() if k != "history"}
                row.update(
                    {
                        "train_seed": cfg.rhm.train_seed,
                        "val_seed": cfg.rhm.val_seed,
                        "test_seed": cfg.rhm.test_seed,
                        "arm": arm_name(target_layer),
                    }
                )
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                completed.add(key)

    print(f"\ncompleted target-depth sweep; metrics: {metrics_path}")


if __name__ == "__main__":
    main()
