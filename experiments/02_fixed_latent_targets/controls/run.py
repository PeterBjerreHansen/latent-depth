#!/usr/bin/env python3
"""Run Stage-02 initialization and invariance controls without Transformer training.

The script is intentionally an experiment-specific adapter.  Numerical
control routines live in ``diagnostics.controls`` and all persisted outputs are
compact JSON; no model or feature tensors are written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import ExperimentConfig, TargetDepthSweepConfig
from diagnostics.controls import (
    child_pair_oracle,
    leaf_prefix_one_hot,
    linear_probe_curve,
    make_balanced_nested_fit_indices,
    make_fixed_partition,
    repeated_invariance_diagnostics,
    surface_lookup_curve,
    surface_mlp_curve,
)
from diagnostics.latent import extract_features_at_positions
from rhm.dataset import RHMSplit, slice_rhm_split
from rhm.interventions import latent_labels, latent_location
from rhm.random_hierarchy_model import sample_rules, sample_trees
from training import (
    artifact_rules,
    build_model,
    load_model_from_checkpoint,
    resolve_device,
    seed_everything,
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"control config does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"control config is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("control config must be a JSON object")
    return payload


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _portable_path(path: Path) -> str:
    """Prefer repository-relative paths in committed diagnostic records."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_experiment(spec: dict[str, Any], base: Path) -> ExperimentConfig:
    if "experiment_config" in spec and "experiment" in spec:
        raise ValueError("control config must use experiment_config or experiment, not both")
    if "experiment_config" in spec:
        sweep = TargetDepthSweepConfig.from_json(_resolve(base, spec["experiment_config"]))
        return sweep.experiment
    if "experiment" in spec:
        if not isinstance(spec["experiment"], dict):
            raise ValueError("experiment must be an object")
        return ExperimentConfig.from_dict(spec["experiment"])
    raise ValueError("control config requires experiment_config or experiment")


def _load_rules(spec: dict[str, Any], base: Path, cfg: ExperimentConfig):
    path_value = spec.get("rules_path")
    if path_value is None:
        return sample_rules(
            v=cfg.rhm.v,
            n=cfg.rhm.n,
            m=cfg.rhm.m,
            s=cfg.rhm.s,
            L=cfg.rhm.L,
            seed=cfg.rhm.rule_seed,
        )
    path = _resolve(base, path_value)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except FileNotFoundError as exc:
        raise ValueError(f"rules file does not exist: {path}") from exc


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _rules_digest(rules: dict[int, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for level in sorted(rules):
        digest.update(str(level).encode("utf-8"))
        digest.update(_tensor_digest(rules[level]).encode("utf-8"))
    return digest.hexdigest()


def _model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_digest(value).encode("utf-8"))
    return digest.hexdigest()


def _int_list(spec: dict[str, Any], key: str, *, default: list[int] | None = None) -> list[int]:
    value = spec.get(key, default)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a nonempty array")
    result = [int(item) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{key} must contain unique values")
    return result


def _validate_spec(spec: dict[str, Any], cfg: ExperimentConfig) -> dict[str, Any]:
    levels = _int_list(spec, "levels", default=[2, 3])
    if any(level < 1 or level >= cfg.rhm.L for level in levels):
        raise ValueError(f"levels must lie in [1, {cfg.rhm.L - 1}]")
    num_sequences = int(spec.get("num_sequences", cfg.data.val_size))
    eval_examples = int(spec.get("eval_examples", num_sequences // 2))
    if not 4 <= num_sequences <= cfg.data.val_size:
        raise ValueError("num_sequences must lie between 4 and the configured validation size")
    if not 0 < eval_examples < num_sequences:
        raise ValueError("eval_examples must lie strictly between zero and num_sequences")
    fit_sizes = _int_list(spec, "fit_sizes", default=[num_sequences - eval_examples])
    if max(fit_sizes) > num_sequences - eval_examples:
        raise ValueError("fit_sizes cannot exceed the fixed fitting pool")
    model_seeds = _int_list(spec, "model_seeds", default=[cfg.model_seed])
    probe_seeds = _int_list(spec, "probe_seeds", default=[12345])
    sample_curve_model_seeds = _int_list(
        spec, "sample_curve_model_seeds", default=[model_seeds[0]]
    )
    if not set(sample_curve_model_seeds) <= set(model_seeds):
        raise ValueError("sample_curve_model_seeds must be drawn from model_seeds")
    replicate_fit_sizes = _int_list(
        spec, "replicate_fit_sizes", default=[fit_sizes[-1]]
    )
    if max(replicate_fit_sizes) > num_sequences - eval_examples:
        raise ValueError("replicate_fit_sizes cannot exceed the fixed fitting pool")
    replicate_probe_seeds = _int_list(
        spec, "replicate_probe_seeds", default=[probe_seeds[0]]
    )
    partition_seed = int(spec.get("partition_seed", 271828))
    probe_steps = int(spec.get("probe_steps", 3000))
    probe_lr = float(spec.get("probe_lr", 0.01))
    if probe_steps <= 0 or probe_lr <= 0:
        raise ValueError("probe_steps and probe_lr must be positive")
    mlp_hidden = int(spec.get("surface_mlp_hidden", 128))
    mlp_steps = int(spec.get("surface_mlp_steps", 1000))
    mlp_lr = float(spec.get("surface_mlp_lr", 0.01))
    if min(mlp_hidden, mlp_steps) <= 0 or mlp_lr <= 0:
        raise ValueError("surface MLP settings must be positive")
    return {
        "levels": levels,
        "num_sequences": num_sequences,
        "eval_examples": eval_examples,
        "fit_sizes": fit_sizes,
        "partition_fit_sizes": sorted(set(fit_sizes) | set(replicate_fit_sizes)),
        "model_seeds": model_seeds,
        "probe_seeds": probe_seeds,
        "sample_curve_model_seeds": sample_curve_model_seeds,
        "replicate_fit_sizes": replicate_fit_sizes,
        "replicate_probe_seeds": replicate_probe_seeds,
        "partition_seed": partition_seed,
        "probe_steps": probe_steps,
        "probe_lr": probe_lr,
        "surface_mlp_hidden": mlp_hidden,
        "surface_mlp_steps": mlp_steps,
        "surface_mlp_lr": mlp_lr,
        "surface_mlp_seed": int(spec.get("surface_mlp_seed", 12345)),
    }


def _annotate(rows: list[dict[str, Any]], **values: Any) -> list[dict[str, Any]]:
    return [{**values, **row} for row in rows]


def _run_initialization_controls(
    cfg: ExperimentConfig,
    rules: dict[int, torch.Tensor],
    split: RHMSplit,
    settings: dict[str, Any],
    *,
    device: torch.device,
    reference_checkpoint: Path | None,
) -> dict[str, Any]:
    levels = settings["levels"]
    positions = [
        latent_location(
            L=cfg.rhm.L, s=cfg.rhm.s, abstraction_level=level
        ).completion_position
        for level in levels
    ]
    labels_by_level = {
        level: latent_labels(split, L=cfg.rhm.L, abstraction_level=level)
        for level in levels
    }
    fit_pool, eval_indices = make_fixed_partition(
        settings["num_sequences"],
        eval_examples=settings["eval_examples"],
        seed=settings["partition_seed"],
    )
    fit_indices_by_level = {
        level: make_balanced_nested_fit_indices(
            labels_by_level[level],
            fit_pool,
            settings["partition_fit_sizes"],
            seed=settings["partition_seed"] + 10_000 * level,
        )
        for level in levels
    }

    random_feature_rows: list[dict[str, Any]] = []
    shuffled_rows: list[dict[str, Any]] = []
    model_digests: dict[str, str] = {}
    seed_zero_reference: dict[str, Any] = {"status": "not_requested"}
    for model_seed in settings["model_seeds"]:
        fit_sizes = (
            settings["fit_sizes"]
            if model_seed in settings["sample_curve_model_seeds"]
            else settings["replicate_fit_sizes"]
        )
        probe_seeds = (
            settings["probe_seeds"]
            if model_seed in settings["sample_curve_model_seeds"]
            else settings["replicate_probe_seeds"]
        )
        model_cfg = copy.deepcopy(cfg)
        model_cfg.model_seed = model_seed
        seed_everything(
            model_seed,
            model_cfg.train.deterministic,
            model_cfg.train.deterministic_strict,
        )
        model = build_model(model_cfg).to(device)
        model.eval()
        model_digests[str(model_seed)] = _model_digest(model)
        features = extract_features_at_positions(
            model,
            split.leaves,
            positions,
            batch_size=cfg.train.batch_size,
            device=device,
        )
        for level_index, level in enumerate(levels):
            fit_indices = {
                size: fit_indices_by_level[level][size] for size in fit_sizes
            }
            for probe_seed in probe_seeds:
                rows = linear_probe_curve(
                    features[level_index],
                    labels_by_level[level],
                    fit_indices,
                    eval_indices,
                    vocab_size=cfg.rhm.v,
                    steps=settings["probe_steps"],
                    learning_rate=settings["probe_lr"],
                    probe_seed=probe_seed,
                    device=device,
                )
                random_feature_rows.extend(
                    _annotate(
                        rows,
                        baseline="random_features",
                        level=level,
                        model_seed=model_seed,
                    )
                )
            if model_seed == settings["model_seeds"][0]:
                generator = torch.Generator(device="cpu").manual_seed(
                    settings["partition_seed"] + 91_337 + level
                )
                shuffled_labels = labels_by_level[level][
                    torch.randperm(settings["num_sequences"], generator=generator)
                ]
                for probe_seed in settings["probe_seeds"]:
                    rows = linear_probe_curve(
                        features[level_index],
                        shuffled_labels,
                        {
                            size: fit_indices_by_level[level][size]
                            for size in settings["fit_sizes"]
                        },
                        eval_indices,
                        vocab_size=cfg.rhm.v,
                        steps=settings["probe_steps"],
                        learning_rate=settings["probe_lr"],
                        probe_seed=probe_seed,
                        device=device,
                    )
                    shuffled_rows.extend(
                        _annotate(
                            rows,
                            baseline="shuffled_labels",
                            level=level,
                            model_seed=model_seed,
                        )
                    )
        del features, model

    if reference_checkpoint is not None:
        if reference_checkpoint.exists():
            reference_model, _, _ = load_model_from_checkpoint(
                reference_checkpoint, device="cpu"
            )
            reference_digest = _model_digest(reference_model)
            generated_digest = model_digests.get("0")
            seed_zero_reference = {
                "status": "matched" if generated_digest == reference_digest else "mismatch",
                "checkpoint": _portable_path(reference_checkpoint),
                "generated_digest": generated_digest,
                "reference_digest": reference_digest,
            }
        else:
            seed_zero_reference = {
                "status": "missing",
                "checkpoint": _portable_path(reference_checkpoint),
            }

    surface_rows: list[dict[str, Any]] = []
    for level in levels:
        prefix_length = latent_location(
            L=cfg.rhm.L, s=cfg.rhm.s, abstraction_level=level
        ).completion_position + 1
        surface_features = leaf_prefix_one_hot(
            split.leaves, prefix_length=prefix_length, vocab_size=cfg.rhm.v
        )
        surface_rows.extend(
            _annotate(
                surface_lookup_curve(
                    split.leaves,
                    labels_by_level[level],
                    fit_indices_by_level[level],
                    eval_indices,
                    prefix_length=prefix_length,
                    vocab_size=cfg.rhm.v,
                ),
                level=level,
            )
        )
        for probe_seed in settings["probe_seeds"]:
            surface_rows.extend(
                _annotate(
                    linear_probe_curve(
                        surface_features.unsqueeze(0),
                        labels_by_level[level],
                        fit_indices_by_level[level],
                        eval_indices,
                        vocab_size=cfg.rhm.v,
                        steps=settings["probe_steps"],
                        learning_rate=settings["probe_lr"],
                        probe_seed=probe_seed,
                        device=device,
                    ),
                    baseline="surface_linear",
                    level=level,
                    prefix_length=prefix_length,
                )
            )
        surface_rows.extend(
            _annotate(
                surface_mlp_curve(
                    surface_features,
                    labels_by_level[level],
                    fit_indices_by_level[level],
                    eval_indices,
                    vocab_size=cfg.rhm.v,
                    hidden_size=settings["surface_mlp_hidden"],
                    steps=settings["surface_mlp_steps"],
                    learning_rate=settings["surface_mlp_lr"],
                    probe_seed=settings["surface_mlp_seed"],
                    device=device,
                ),
                level=level,
                prefix_length=prefix_length,
            )
        )

    oracles = [
        child_pair_oracle(
            split,
            rules,
            L=cfg.rhm.L,
            s=cfg.rhm.s,
            abstraction_level=level,
        )
        for level in levels
    ]
    return {
        "random_feature_rows": random_feature_rows,
        "surface_rows": surface_rows,
        "shuffled_rows": shuffled_rows,
        "oracles": oracles,
        "model_digests": model_digests,
        "seed_zero_reference": seed_zero_reference,
        "partition": {
            "num_examples": settings["num_sequences"],
            "eval_examples": settings["eval_examples"],
            "fit_pool_examples": int(fit_pool.numel()),
            "fit_sizes": settings["fit_sizes"],
            "partition_seed": settings["partition_seed"],
            "eval_indices": [int(index) for index in eval_indices.tolist()],
            "eval_indices_digest": hashlib.sha256(
                eval_indices.numpy().tobytes()
            ).hexdigest(),
            "fit_pool_digest": hashlib.sha256(
                fit_pool.numpy().tobytes()
            ).hexdigest(),
            "fit_indices_digests": {
                str(level): {
                    str(size): hashlib.sha256(
                        indices.numpy().tobytes()
                    ).hexdigest()
                    for size, indices in fit_indices_by_level[level].items()
                }
                for level in levels
            },
        },
    }


def _run_invariance_controls(
    spec: dict[str, Any],
    base: Path,
    cfg: ExperimentConfig,
    canonical_rules: dict[int, torch.Tensor],
    split: RHMSplit,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    invariance = spec.get("invariance")
    if not isinstance(invariance, dict):
        raise ValueError("invariance must be an object when requested")
    arms = invariance.get("arms")
    steps = invariance.get("steps")
    levels = invariance.get("levels", [2, 3, 4])
    if not isinstance(arms, list) or not arms or not isinstance(steps, list) or not steps:
        raise ValueError("invariance arms and steps must be nonempty arrays")
    if not isinstance(levels, list) or not levels:
        raise ValueError("invariance levels must be a nonempty array")
    num_sequences = int(invariance.get("num_sequences", 1024))
    replicates = int(invariance.get("replicates", 8))
    seed = int(invariance.get("seed", 12345))
    outputs: list[dict[str, Any]] = []
    for arm in arms:
        run_value = spec.get("reference_runs", {}).get(arm)
        if run_value is None:
            raise ValueError(f"no reference_runs entry for invariance arm {arm}")
        run_dir = _resolve(base, run_value)
        for step in steps:
            checkpoint = run_dir / "diagnostic_snapshots" / f"step_{int(step):08d}.pt"
            if not checkpoint.exists():
                raise ValueError(f"invariance checkpoint does not exist: {checkpoint}")
            model, loaded_cfg, artifact = load_model_from_checkpoint(
                checkpoint, device=str(device)
            )
            loaded_rules = artifact_rules(checkpoint, artifact)
            if _rules_digest(loaded_rules) != _rules_digest(canonical_rules):
                raise ValueError(f"grammar mismatch for invariance checkpoint: {checkpoint}")
            if loaded_cfg.rhm.L != cfg.rhm.L or loaded_cfg.rhm.s != cfg.rhm.s:
                raise ValueError(f"RHM shape mismatch for invariance checkpoint: {checkpoint}")
            result = repeated_invariance_diagnostics(
                model,
                split,
                canonical_rules,
                cfg,
                device,
                levels=[int(level) for level in levels],
                num_sequences=num_sequences,
                replicates=replicates,
                seed=seed,
            )
            result.update(
                {
                    "arm": str(arm),
                    "step": int(step),
                    "checkpoint": _portable_path(checkpoint),
                    "model_seed": int(loaded_cfg.model_seed),
                    "model_digest": _model_digest(model),
                }
            )
            outputs.append(result)
            del model
    return outputs


def run_controls(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "auto",
    include_invariance: bool = False,
) -> Path:
    """Run controls and write the compact raw result bundle."""
    config_path = Path(config_path).resolve()
    base = config_path.parent
    spec = _read_json(config_path)
    cfg = _load_experiment(spec, base)
    rules = _load_rules(spec, base, cfg)
    settings = _validate_spec(spec, cfg)
    resolved_device = resolve_device(device)

    trees, choices = sample_trees(
        cfg.data.val_size, rules, seed=cfg.rhm.val_seed, return_choices=True
    )
    split = slice_rhm_split(
        RHMSplit(trees=trees, choices=choices), settings["num_sequences"]
    )
    reference_value = spec.get("reference_step_zero_checkpoint")
    reference_checkpoint = _resolve(base, reference_value) if reference_value else None
    result = _run_initialization_controls(
        cfg,
        rules,
        split,
        settings,
        device=resolved_device,
        reference_checkpoint=reference_checkpoint,
    )
    invariance_enabled = include_invariance or bool(
        isinstance(spec.get("invariance"), dict)
        and spec["invariance"].get("enabled", False)
    )
    if invariance_enabled:
        result["invariance"] = _run_invariance_controls(
            spec,
            base,
            cfg,
            rules,
            split,
            device=resolved_device,
        )
    else:
        result["invariance"] = []

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "control_config": _portable_path(config_path),
        "device": str(resolved_device),
        "protocol": {
            "experiment": cfg.to_dict(),
            "settings": settings,
            "levels": settings["levels"],
            "positions": [
                latent_location(
                    L=cfg.rhm.L, s=cfg.rhm.s, abstraction_level=level
                ).completion_position
                for level in settings["levels"]
            ],
            "rules_digest": _rules_digest(rules),
            "validation_seed": cfg.rhm.val_seed,
        },
        **result,
    }
    raw_path = output / "raw_controls.json"
    raw_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(raw_path)
    return raw_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="controls JSON config")
    parser.add_argument("--output-dir", required=True, help="compact result directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--include-invariance",
        action="store_true",
        help="run repeated C/Q controls against local saved snapshots",
    )
    args = parser.parse_args()
    run_controls(
        args.config,
        args.output_dir,
        device=args.device,
        include_invariance=args.include_invariance,
    )


if __name__ == "__main__":
    main()
