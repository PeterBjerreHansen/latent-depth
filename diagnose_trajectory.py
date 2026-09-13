#!/usr/bin/env python3
"""Measure selected model snapshots offline; derive a trajectory from raw records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from diagnose import diagnose_checkpoint


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.rsplit('_', 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f'checkpoint name does not encode a step: {path}') from exc


def diagnose_trajectory(
    run_dir: str | Path, output_dir: str | Path, *,
    snapshot_dir: str | Path | None = None, split: str = 'val', device: str = 'auto',
    metric: str = 'probe', num_sequences: int = 1024, probe_steps: int = 300,
    probe_seed: int = 12345, probe_lr: float = 1e-3, controls: bool = False,
    steps: list[int] | None = None, every_updates: int | None = None,
) -> dict[str, Any]:
    run_path, output_path = Path(run_dir), Path(output_dir)
    source = Path(snapshot_dir) if snapshot_dir is not None else run_path / 'diagnostic_snapshots'
    if steps is not None and every_updates is not None:
        raise ValueError('choose explicit steps or an update interval')
    if every_updates is not None and every_updates <= 0:
        raise ValueError('every_updates must be positive')
    checkpoints = sorted(source.glob('step_*.pt'), key=checkpoint_step)
    if steps is not None:
        missing = set(steps) - {checkpoint_step(p) for p in checkpoints}
        if missing:
            raise ValueError(f'missing snapshot steps: {sorted(missing)}')
        checkpoints = [p for p in checkpoints if checkpoint_step(p) in steps]
    if every_updates is not None:
        checkpoints = [p for p in checkpoints if checkpoint_step(p) % every_updates == 0]
    if not checkpoints:
        raise RuntimeError(f'no selected step snapshots found under {source}')
    output_path.mkdir(parents=True, exist_ok=True)
    # Invalidate the derived view until all selected measurements are complete.
    summary_path = output_path / 'trajectory.json'
    summary_path.unlink(missing_ok=True)
    raw_path = output_path / 'records.jsonl'
    with raw_path.open('w') as handle:
        for checkpoint in checkpoints:
            result = diagnose_checkpoint(
                checkpoint, split=split, device=device, metric=metric,
                num_sequences=num_sequences, probe_steps=probe_steps,
                probe_seed=probe_seed, probe_lr=probe_lr, controls=controls,
            )
            handle.write(json.dumps(result) + '\n')
            handle.flush()
            print(f'{checkpoint.name}: {metric}', flush=True)
    records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    settings = records[0]['diagnostic_config']
    summary = {
        'run_dir': str(run_path), 'snapshot_dir': str(source), 'split': split,
        'device': device, 'metric': metric, 'controls': controls,
        'num_sequences': records[0]['diagnostics']['num_sequences'],
        'probe_steps': settings['probe_steps'], 'diagnostic_config': settings,
        'checkpoints': records,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--snapshot-dir', help='defaults to RUN/diagnostic_snapshots')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--metric', choices=('probe', 'clustering', 'all'), default='probe')
    parser.add_argument('--num-sequences', type=int, default=1024)
    parser.add_argument('--probe-steps', type=int, default=300)
    parser.add_argument('--probe-seed', type=int, default=12345)
    parser.add_argument('--probe-lr', type=float, default=1e-3)
    parser.add_argument('--controls', action='store_true')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--steps', type=int, nargs='+')
    selection.add_argument('--every-updates', type=int)
    args = parser.parse_args()
    diagnose_trajectory(**vars(args))


if __name__ == '__main__':
    main()
