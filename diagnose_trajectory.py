#!/usr/bin/env python3
"""Run latent diagnostics over every exact-step checkpoint in one run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from diagnose import diagnose_checkpoint


def checkpoint_step(path: Path) -> int:
    """Return the integer step encoded by ``step_XXXXXXXX.pt``."""
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"checkpoint name does not encode a step: {path}") from exc


def diagnose_trajectory(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str = "val",
    device: str = "auto",
    num_sequences: int | None = None,
    probe_steps: int | None = None,
    controls: bool = False,
) -> dict[str, Any]:
    """Diagnose all exact-step snapshots and write one JSON per checkpoint."""
    run_path = Path(run_dir)
    output_path = Path(output_dir)
    checkpoints = sorted(
        (run_path / "checkpoints").glob("step_*.pt"), key=checkpoint_step
    )
    if not checkpoints:
        raise RuntimeError(f"no step checkpoints found under {run_path / 'checkpoints'}")

    output_path.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        result = diagnose_checkpoint(
            checkpoint,
            split=split,
            device=device,
            metric="all",
            num_sequences=num_sequences,
            probe_steps=probe_steps,
            controls=controls,
        )
        result_path = output_path / f"step_{step:08d}.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        records.append(result)
        print(result_path)

    summary = {
        "run_dir": str(run_path),
        "split": split,
        "device": device,
        "num_sequences": num_sequences,
        "probe_steps": probe_steps,
        "controls": controls,
        "checkpoints": records,
    }
    summary_path = output_path / "trajectory.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="one P_* training run directory")
    parser.add_argument("--output-dir", required=True, help="directory for diagnostic JSON files")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-sequences", type=int, default=None)
    parser.add_argument("--probe-steps", type=int, default=None)
    parser.add_argument(
        "--controls",
        action="store_true",
        help="include the shuffled-label probe at every checkpoint",
    )
    args = parser.parse_args()
    diagnose_trajectory(
        args.run_dir,
        args.output_dir,
        split=args.split,
        device=args.device,
        num_sequences=args.num_sequences,
        probe_steps=args.probe_steps,
        controls=args.controls,
    )


if __name__ == "__main__":
    main()
