#!/usr/bin/env python3
"""Explicit held-out NTP evaluation of a chosen snapshot; never called by training."""
import argparse
import json
from pathlib import Path

from rhm.dataset import LeafSequenceDataset, sample_leaf_sequences
from training import artifact_rules, evaluate_with_positions, load_model_from_checkpoint, make_loader, resolve_device


def evaluate_snapshot(checkpoint: str | Path, *, split: str = 'val', device: str = 'auto') -> dict:
    if split not in {'val', 'test'}:
        raise ValueError('split must be val or test')
    resolved = resolve_device(device)
    model, cfg, artifact = load_model_from_checkpoint(checkpoint, device=str(resolved))
    rules = artifact_rules(checkpoint, artifact)
    count = cfg.data.val_size if split == 'val' else cfg.data.test_size
    seed = cfg.rhm.val_seed if split == 'val' else cfg.rhm.test_seed
    dataset = LeafSequenceDataset(sample_leaf_sequences(count, rules, seed=seed))
    loader = make_loader(dataset, batch_size=cfg.train.batch_size, shuffle=False,
                         num_workers=0, seed=0, device=resolved)
    ce, positions = evaluate_with_positions(model, loader, cfg, resolved)
    return {'checkpoint': str(checkpoint), 'global_step': artifact['global_step'],
            'split': split, 'num_sequences': count, 'ce': ce, 'nll_by_position': positions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    result = evaluate_snapshot(args.checkpoint, split=args.split, device=args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
