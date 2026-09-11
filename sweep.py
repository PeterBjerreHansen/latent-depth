#!/usr/bin/env python3
"""Sweep training-set size P for RHM next-token experiments."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch

from config import SweepConfig
from provenance import file_sha256, git_provenance
from rhm.dataset import LeafSequenceDataset, build_rhm_bundle
from training import train_model


def _derived_seed(base: int, grammar_seed: int) -> int:
    # Stable, simple integer mixing; grammar 0 uses the configured base seed,
    # while different grammars do not reuse identical root/rule-choice draws.
    return int((base + 1_000_003 * grammar_seed) % (2**63 - 1))


def _existing_keys(metrics_path: Path) -> set[tuple[int, int, int]]:
    keys: set[tuple[int, int, int]] = set()
    if not metrics_path.exists():
        return keys
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add((int(row["rule_seed"]), int(row["model_seed"]), int(row["train_size"])))
    return keys


def fixed_exposure_budget(
    *, train_size: int, batch_size: int, samples_per_example: int
) -> tuple[int, int]:
    """Return updates for complete dataset passes at a fixed exposure target."""
    samples_float = float(samples_per_example)
    if (
        train_size <= 0
        or batch_size <= 0
        or not math.isfinite(samples_float)
        or samples_float <= 0
        or not samples_float.is_integer()
    ):
        raise ValueError(
            "train_size and batch_size must be positive; samples_per_example must be a positive whole number"
        )
    effective_batch_size = min(int(batch_size), int(train_size))
    updates_per_epoch = math.ceil(train_size / effective_batch_size)
    return int(samples_float) * updates_per_epoch, updates_per_epoch


def _validate_resume_state(
    *, metrics_path: Path, snapshot_path: Path, snapshot: dict[str, object], resume: bool
) -> None:
    """Reject result reuse when the sweep provenance cannot be verified."""
    has_metrics = metrics_path.exists() and metrics_path.stat().st_size > 0
    if has_metrics and not resume:
        raise RuntimeError(
            f"{metrics_path} already contains results; use --resume or a new output directory"
        )
    if not resume:
        return
    if has_metrics and not snapshot_path.exists():
        raise RuntimeError(
            f"cannot resume {metrics_path}: existing results have no sweep_config.json"
        )
    if snapshot_path.exists():
        previous = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if previous != snapshot:
            raise RuntimeError(
                "refusing to mix results: existing sweep_config.json differs from requested config"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON sweep config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true", help="skip completed (grammar, model, P) runs")
    args = parser.parse_args()

    sweep = SweepConfig.from_json(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    snapshot_path = out / "sweep_config.json"
    config_path = Path(args.config).resolve()
    provenance = git_provenance(Path(__file__).resolve().parent)
    snapshot = {
        "experiment": sweep.experiment.to_dict(),
        "train_sizes": sweep.train_sizes,
        "grammar_seeds": sweep.grammar_seeds,
        "model_seeds": sweep.model_seeds,
        "replicates": (
            [{"grammar_seed": g, "model_seed": m} for g, m in sweep.replicates]
            if sweep.replicates is not None
            else None
        ),
        "samples_per_example": sweep.samples_per_example,
        "input_config": str(config_path),
        "input_config_sha256": file_sha256(config_path),
        **provenance,
        "effective_train_settings": {
            str(P): {
                "batch_size": min(sweep.experiment.train.batch_size, P),
                "max_updates": (
                    fixed_exposure_budget(
                        train_size=P,
                        batch_size=sweep.experiment.train.batch_size,
                        samples_per_example=sweep.samples_per_example,
                    )[0]
                    if sweep.samples_per_example is not None
                    else sweep.experiment.train.max_updates
                ),
                "max_epochs": (
                    max(sweep.experiment.train.max_epochs, sweep.samples_per_example)
                    if sweep.samples_per_example is not None
                    else sweep.experiment.train.max_epochs
                ),
            }
            for P in sweep.train_sizes
        },
    }

    _validate_resume_state(
        metrics_path=metrics_path,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
        resume=args.resume,
    )

    completed = _existing_keys(metrics_path) if args.resume else set()
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    max_p = max(sweep.train_sizes)
    pairs = sweep.replicate_pairs()
    models_by_grammar: dict[int, list[int]] = {}
    for grammar_seed, model_seed in pairs:
        models_by_grammar.setdefault(grammar_seed, []).append(model_seed)

    for grammar_index, (grammar_seed, grammar_model_seeds) in enumerate(models_by_grammar.items()):
        cfg0 = copy.deepcopy(sweep.experiment)
        cfg0.rhm.rule_seed = grammar_seed
        cfg0.rhm.train_seed = _derived_seed(cfg0.rhm.train_seed, grammar_seed)
        cfg0.rhm.val_seed = _derived_seed(cfg0.rhm.val_seed, grammar_seed)
        cfg0.rhm.test_seed = _derived_seed(cfg0.rhm.test_seed, grammar_seed)
        cfg0.data.train_size = max_p
        cfg0.validate()

        print(f"generating grammar {grammar_seed} with nested train pool P_max={max_p}")
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
            train_size=max_p,
            val_size=cfg0.data.val_size,
            test_size=cfg0.data.test_size,
        )
        grammar_dir = out / f"grammar_{grammar_seed}"
        grammar_dir.mkdir(exist_ok=True)
        torch.save(bundle.rules, grammar_dir / "rules.pt")

        val_ds = LeafSequenceDataset(bundle.val.leaves)
        test_ds = LeafSequenceDataset(bundle.test.leaves)

        for model_seed in grammar_model_seeds:
            for P in sweep.train_sizes:
                key = (grammar_seed, model_seed, P)
                if key in completed:
                    print("skip completed", key)
                    continue

                cfg = copy.deepcopy(cfg0)
                cfg.model_seed = model_seed
                cfg.data.train_size = P
                if sweep.samples_per_example is not None:
                    cfg.train.max_updates, _ = fixed_exposure_budget(
                        train_size=P,
                        batch_size=cfg.train.batch_size,
                        samples_per_example=sweep.samples_per_example,
                    )
                    cfg.train.max_epochs = max(
                        cfg.train.max_epochs,
                        sweep.samples_per_example,
                    )
                cfg.validate()
                train_ds = LeafSequenceDataset(bundle.train.leaves[:P])
                run_dir = grammar_dir / f"model_{model_seed}" / f"P_{P}"
                run_dir.mkdir(parents=True, exist_ok=True)
                with open(run_dir / "config.json", "w", encoding="utf-8") as f:
                    json.dump(cfg.to_dict(), f, indent=2)
                print(f"\n=== grammar={grammar_seed} model_seed={model_seed} P={P} ===")
                metrics = train_model(
                    cfg,
                    train_ds,
                    val_ds,
                    test_ds,
                    output_dir=run_dir,
                    diagnostic_split=bundle.val if cfg.diagnostics.enabled else None,
                    rules=bundle.rules,
                )
                row = {k: v for k, v in metrics.items() if k != "history"}
                row.update(
                    {
                        "grammar_index": grammar_index,
                        "train_seed": cfg.rhm.train_seed,
                        "val_seed": cfg.rhm.val_seed,
                        "test_seed": cfg.rhm.test_seed,
                        "effective_max_updates": cfg.train.max_updates,
                        "samples_per_example": sweep.samples_per_example,
                        "code_commit": provenance["code_commit"],
                    }
                )
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                    f.flush()
                completed.add(key)

    print(f"\ncompleted sweep; metrics: {metrics_path}")


if __name__ == "__main__":
    main()
