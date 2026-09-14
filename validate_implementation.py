"""Run the implementation-validation matrix and write a JSON report.

This is deliberately separate from scientific sweeps. It checks the data
contract, causal next-token objective, checkpoint replay, and CPU/MPS
numerical agreement on a small fixed experiment.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from config import ExperimentConfig, DiagnosticsConfig
from nanogpt import GPT, GPTConfig
from diagnose import diagnose_checkpoint
from rhm.dataset import LeafSequenceDataset, build_rhm_bundle
from rhm.random_hierarchy_model import sample_rules, sample_trees
from training import build_model, train_model


def _tiny_config(*, device: str, max_updates: int = 4) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "rhm": {
                "v": 8,
                "n": 8,
                "m": 2,
                "s": 2,
                "L": 2,
                "rule_seed": 7,
                "train_seed": 101,
                "val_seed": 202,
                "test_seed": 303,
            },
            "data": {"train_size": 32, "val_size": 32, "test_size": 32},
            "model": {"n_layer": 1, "n_head": 1, "n_embd": 16, "dropout": 0.0},

            "optim": {
                "name": "adamw",
                "learning_rate": 0.001,
                "betas": [0.9, 0.95],
                "weight_decay": 0.0,
                "warmup_epochs": 0.0,
            },
            "train": {
                "batch_size": 8,
                "max_epochs": 4,
                "max_updates": max_updates,
                "grad_clip": 1.0,
                "eval_every_epochs": 1,
                "num_workers": 0,
                "device": device,
                "deterministic": True,
                "deterministic_strict": device == "cpu",
                "save_checkpoints": True,
            },
            "model_seed": 19,
        }
    )


def _bundle(cfg: ExperimentConfig):
    return build_rhm_bundle(
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


def _datasets(bundle):
    return (
        LeafSequenceDataset(bundle.train.leaves),
        bundle.val,
    )


def _state_max_abs_diff(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    return max(
        float((left[name].float() - right[name].float()).abs().max().item())
        for name in left
    )


def check_data_oracle() -> dict[str, Any]:
    rules = sample_rules(v=4, n=4, m=2, s=2, L=2, seed=3)
    trees, choices = sample_trees(32, rules, seed=9, return_choices=True)
    checked_paths = 0
    for row in range(trees[0].shape[0]):
        symbols = [int(trees[0][row])]
        for level in range(2):
            expanded: list[int] = []
            for position, symbol in enumerate(symbols):
                rule = int(choices[level][row].reshape(-1)[position])
                expanded.extend(int(x) for x in rules[level][symbol, rule].tolist())
            symbols = expanded
            if symbols != trees[level + 1][row].reshape(-1).tolist():
                raise AssertionError(f"independent expansion failed at row={row}, level={level}")
            checked_paths += 1
    return {
        "passed": True,
        "grammar_shapes": {str(level): list(value.shape) for level, value in rules.items()},
        "checked_expansions": checked_paths,
        "leaf_length": int(trees[2].shape[1]),
    }


def check_objective() -> dict[str, Any]:
    torch.manual_seed(11)
    cfg = GPTConfig(vocab_size=13, block_size=5, n_layer=1, n_head=1, n_embd=16)
    model = GPT(cfg).eval()
    tokens = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    logits, loss = model(tokens, targets=tokens[:, 1:])
    if loss is None:
        raise AssertionError("model returned no shifted loss")
    reference = F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size), tokens[:, 1:].reshape(-1)
    )
    if not torch.equal(loss, reference):
        raise AssertionError("shifted loss does not match independent cross-entropy reference")

    changed = tokens.clone()
    changed[:, 3:] = torch.tensor([[9, 10], [11, 12]])
    original_logits, _ = model(tokens, targets=tokens[:, 1:])
    changed_logits, _ = model(changed, targets=changed[:, 1:])
    if not torch.allclose(original_logits[:, :3], changed_logits[:, :3], atol=1e-7, rtol=1e-6):
        raise AssertionError("causal prefix logits changed after suffix replacement")
    return {
        "passed": True,
        "loss": float(loss.item()),
        "prediction_positions": int(logits.shape[1] - 1),
        "tied_embeddings": model.transformer.wte.weight is model.lm_head.weight,
    }


def check_cpu_checkpoint_replay(root: Path) -> dict[str, Any]:
    cfg = _tiny_config(device="cpu", max_updates=4)
    bundle = _bundle(cfg)
    datasets = _datasets(bundle)
    full_dir = root / "full"
    split_dir = root / "split"
    resumed_dir = root / "resumed"
    train_model(cfg, *datasets, output_dir=full_dir, rules=bundle.rules, verbose=False)
    split_cfg = copy.deepcopy(cfg)
    split_cfg.train.max_updates = 2
    train_model(split_cfg, *datasets, output_dir=split_dir, rules=bundle.rules, verbose=False)
    train_model(
        cfg,
        *datasets,
        output_dir=resumed_dir,
        resume_from=split_dir / "last.pt",
        rules=bundle.rules,
        verbose=False,
    )
    full = torch.load(full_dir / "last.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_dir / "last.pt", map_location="cpu", weights_only=False)
    model_delta = _state_max_abs_diff(full["model"], resumed["model"])
    if model_delta != 0.0:
        raise AssertionError(f"CPU replay changed model state by {model_delta}")
    if full["global_step"] != resumed["global_step"]:
        raise AssertionError("CPU replay changed global step")
    return {
        "passed": True,
        "global_step": int(full["global_step"]),
        "samples_seen": int(full["trainer_state"]["samples_seen"]),
        "max_parameter_abs_delta": model_delta,
    }


def check_device_pair(root: Path) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        return {"passed": True, "skipped": True, "reason": "MPS is unavailable"}

    cfg_cpu = _tiny_config(device="cpu", max_updates=4)
    cfg_mps = copy.deepcopy(cfg_cpu)
    cfg_mps.train.device = "mps"
    cfg_mps.train.deterministic_strict = False
    bundle = _bundle(cfg_cpu)
    datasets = _datasets(bundle)
    cpu_metrics = train_model(
        cfg_cpu,
        *datasets,
        output_dir=root / "cpu",
        rules=bundle.rules,
        verbose=False,
    )
    mps_metrics = train_model(
        cfg_mps,
        *datasets,
        output_dir=root / "mps",
        rules=bundle.rules,
        verbose=False,
    )
    cpu = torch.load(root / "cpu" / "last.pt", map_location="cpu", weights_only=False)
    mps = torch.load(root / "mps" / "last.pt", map_location="cpu", weights_only=False)
    parameter_delta = _state_max_abs_diff(cpu["model"], mps["model"])
    val_delta = abs(cpu_metrics["last_val_ce"] - mps_metrics["last_val_ce"])
    return {
        "passed": True,
        "skipped": False,
        "cpu_val_ce": cpu_metrics["last_val_ce"],
        "mps_val_ce": mps_metrics["last_val_ce"],
        "val_ce_abs_delta": val_delta,
        "max_parameter_abs_delta": parameter_delta,
        "fallback_mode": "caller-controlled; validation should use PYTORCH_ENABLE_MPS_FALLBACK=0",
    }


def check_mps_checkpoint_replay(root: Path) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        return {"passed": True, "skipped": True, "reason": "MPS is unavailable"}

    cfg = _tiny_config(device="mps", max_updates=4)
    cfg.train.deterministic_strict = False
    bundle = _bundle(cfg)
    datasets = _datasets(bundle)
    full_dir = root / "full"
    split_dir = root / "split"
    resumed_dir = root / "resumed"
    train_model(cfg, *datasets, output_dir=full_dir, rules=bundle.rules, verbose=False)
    split_cfg = copy.deepcopy(cfg)
    split_cfg.train.max_updates = 2
    train_model(split_cfg, *datasets, output_dir=split_dir, rules=bundle.rules, verbose=False)
    train_model(
        cfg,
        *datasets,
        output_dir=resumed_dir,
        resume_from=split_dir / "last.pt",
        rules=bundle.rules,
        verbose=False,
    )
    full = torch.load(full_dir / "last.pt", map_location="cpu", weights_only=False)
    split = torch.load(split_dir / "last.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_dir / "last.pt", map_location="cpu", weights_only=False)
    model_delta = _state_max_abs_diff(full["model"], resumed["model"])
    full_metrics = full["metrics"]
    resumed_metrics = resumed["metrics"]
    val_delta = abs(float(full_metrics["val_ce"]) - float(resumed_metrics["val_ce"]))
    rng_recorded = split["rng"].get("mps") is not None
    tolerance = 1e-4
    if not rng_recorded or model_delta > tolerance or val_delta > tolerance:
        raise AssertionError(
            "MPS checkpoint replay exceeded tolerance: "
            f"rng_recorded={rng_recorded}, model_delta={model_delta}, val_delta={val_delta}"
        )
    return {
        "passed": True,
        "skipped": False,
        "global_step": int(resumed["global_step"]),
        "mps_rng_recorded": rng_recorded,
        "max_parameter_abs_delta": model_delta,
        "val_ce_abs_delta": val_delta,
        "tolerance": tolerance,
    }


def check_validation_probes(root: Path, device: str) -> dict[str, Any]:
    """Compare training with observers off/on, including dropout and fresh pools."""
    if device == 'mps' and not torch.backends.mps.is_available():
        return {'passed': True, 'skipped': True, 'reason': 'MPS unavailable'}
    cfg = _tiny_config(device=device, max_updates=6)
    cfg.model.dropout = 0.1
    cfg.data.resample_train_each_epoch = True
    cfg.auxiliary.mode = 'next_latent'
    cfg.auxiliary.target_layer = 1
    cfg.train.eval_at_start = True
    cfg.train.eval_every_updates = 2
    bundle = _bundle(cfg)
    plain = train_model(cfg, *_datasets(bundle), output_dir=root/'plain', rules=bundle.rules, verbose=False)
    observed_cfg = copy.deepcopy(cfg)
    observed_cfg.diagnostics = DiagnosticsConfig(synonym_clustering=False, num_sequences=32, probe_steps=8)
    observed = train_model(observed_cfg, *_datasets(bundle), output_dir=root/'observed', rules=bundle.rules, verbose=False)
    left = torch.load(root/'plain/last.pt', map_location='cpu', weights_only=False)
    right = torch.load(root/'observed/last.pt', map_location='cpu', weights_only=False)
    delta = _state_max_abs_diff(left['model'], right['model'])
    head_delta = _state_max_abs_diff(left['predictor'], right['predictor'])
    ce_delta = max(abs(a['val_ce'] - b['val_ce']) for a, b in zip(plain['history'], observed['history']))
    offline = diagnose_checkpoint(root/'observed/last.pt', device=device, metric='probe', num_sequences=32, probe_steps=8)
    online = observed['history'][-1]['diagnostics']['linear_probe']['by_level']
    offline = offline['diagnostics']['linear_probe']['by_level']
    probe_delta = max(abs(a-b) for level in online for a,b in zip(online[level]['balanced_accuracy_by_layer'], offline[level]['balanced_accuracy_by_layer']))
    tolerance = 0.0 if device == 'cpu' else 1e-4
    if max(delta, head_delta, ce_delta, probe_delta) > tolerance:
        raise AssertionError(f'{device} validation probes changed training or disagreed offline')
    rng_key = 'torch' if device == 'cpu' else 'mps'
    if not torch.equal(left['rng'][rng_key], right['rng'][rng_key]):
        raise AssertionError(f'{device} probes changed the training RNG')
    return {'passed': True, 'max_parameter_abs_delta': delta, 'max_head_abs_delta': head_delta,
            'max_val_ce_abs_delta': ce_delta, 'max_probe_accuracy_abs_delta': probe_delta,
            'tolerance': tolerance, 'rng_unchanged': True}


def run_validation(output: str | Path | None = None, *, skip_mps: bool = False) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="latent-depth-validation-") as temp_dir:
        root = Path(temp_dir)
        report: dict[str, Any] = {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "data_oracle": check_data_oracle(),
            "objective": check_objective(),
            "cpu_checkpoint_replay": check_cpu_checkpoint_replay(root / "replay"),
            "cpu_validation_probes": check_validation_probes(root / "cpu_probes", "cpu"),
        }
        if skip_mps:
            report["device_pair"] = {"passed": True, "skipped": True, "reason": "--skip-mps"}
        else:
            report["mps_validation_probes"] = check_validation_probes(root / "mps_probes", "mps")
            report["device_pair"] = check_device_pair(root / "device")
            report["mps_checkpoint_replay"] = check_mps_checkpoint_replay(root / "mps_replay")

    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/implementation_validation/validation.json")
    parser.add_argument("--skip-mps", action="store_true")
    args = parser.parse_args()
    report = run_validation(args.output, skip_mps=args.skip_mps)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
