#!/usr/bin/env python3
"""Train one nanoGPT-style causal model on a fixed finite RHM dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import ExperimentConfig
from rhm.dataset import LeafSequenceDataset, build_rhm_bundle
from training import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON experiment config")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_json(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bundle = build_rhm_bundle(
        v=cfg.rhm.v,
        n=cfg.rhm.n,
        m=cfg.rhm.m,
        s=cfg.rhm.s,
        L=cfg.rhm.L,
        rule_seed=cfg.rhm.rule_seed,
        train_seed=cfg.rhm.train_seed,
        val_seed=cfg.rhm.val_seed,
        test_seed=cfg.rhm.test_seed,
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
    )
    torch.save(bundle.rules, out / "rules.pt")
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print("RHM sequence length:", cfg.rhm.s**cfg.rhm.L)

    metrics = train_model(
        cfg,
        LeafSequenceDataset(bundle.train.leaves),
        LeafSequenceDataset(bundle.val.leaves),
        LeafSequenceDataset(bundle.test.leaves),
        output_dir=out,
        diagnostic_split=bundle.val if cfg.diagnostics.enabled else None,
        rules=bundle.rules,
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
