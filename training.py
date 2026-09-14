"""Training/evaluation utilities for fixed finite RHM datasets."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from auxiliary import NextLatentPredictor, build_auxiliary_predictor, next_latent_loss
from config import ExperimentConfig
from nanogpt import GPT, GPTConfig
from rhm.dataset import LeafSequenceDataset, RHMSplit, sample_leaf_sequences
from diagnostics import run_latent_diagnostics
from rhm.random_hierarchy_model import TensorDict
from rhm.theory import loss_upper_bounds


def seed_everything(seed: int, deterministic: bool = True, strict: bool = False) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(deterministic, warn_only=not strict)
    if deterministic and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mps_rng_state() -> Optional[torch.Tensor]:
    """Capture the MPS generator when this PyTorch build exposes it."""
    mps = getattr(torch, "mps", None)
    if mps is None or not torch.backends.mps.is_available():
        return None
    if hasattr(mps, "synchronize"):
        mps.synchronize()
    get_state = getattr(mps, "get_rng_state", None)
    if get_state is None:
        return None
    return get_state().cpu().clone()


def input_block_size(cfg: ExperimentConfig) -> int:
    return cfg.rhm.s**cfg.rhm.L


def _resampled_train_seed(base_seed: int, epoch: int) -> int:
    """Return the deterministic local RNG seed for a generated train pool."""
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    return int((int(base_seed) + 1_000_003 * (epoch - 1)) % (2**63 - 1))


def build_model(cfg: ExperimentConfig) -> GPT:
    gpt_cfg = GPTConfig(
        vocab_size=cfg.rhm.v,
        block_size=input_block_size(cfg),
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        dropout=cfg.model.dropout,
        bias=cfg.model.bias,
    )
    return GPT(gpt_cfg)


def objective_inputs(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the full causal input and its shifted next-token labels."""
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [batch, sequence_length]")
    if tokens.size(1) < 2:
        raise ValueError("sequence must contain at least two tokens")
    return tokens, tokens[:, 1:]


def loss_from_tokens(model: GPT, tokens: torch.Tensor) -> torch.Tensor:
    """Ordinary next-token loss; intentionally independent of auxiliary mode."""
    x, y = objective_inputs(tokens)
    _, loss = model(x, targets=y)
    if loss is None:
        raise RuntimeError("model did not return a loss for next-token training")
    return loss


def training_losses(
    model: GPT,
    predictor: NextLatentPredictor | None,
    tokens: torch.Tensor,
    cfg: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Return NTP, optional next-latent, and combined training losses."""
    x, y = objective_inputs(tokens)
    if cfg.auxiliary.mode == "none":
        if predictor is not None:
            raise ValueError("predictor must be None when auxiliary mode is disabled")
        _, ntp_loss = model(x, targets=y)
        if ntp_loss is None:
            raise RuntimeError("model did not return a next-token loss")
        return ntp_loss, None, ntp_loss

    if cfg.auxiliary.mode != "next_latent":
        raise ValueError(f"unsupported auxiliary mode: {cfg.auxiliary.mode}")
    if predictor is None or cfg.auxiliary.target_layer is None:
        raise ValueError("next_latent mode requires predictor and target_layer")
    _, ntp_loss, hidden_states, _ = model(x, targets=y, return_hidden=True)
    if ntp_loss is None:
        raise RuntimeError("model did not return a next-token loss")
    aux_loss = next_latent_loss(
        hidden_states,
        predictor,
        target_layer=int(cfg.auxiliary.target_layer),
    )
    total_loss = ntp_loss + float(cfg.auxiliary.weight) * aux_loss
    return ntp_loss, aux_loss, total_loss


def _per_position_nll(model: GPT, tokens: torch.Tensor) -> torch.Tensor:
    """Return one NLL value per next-token position and batch example."""
    x, y = objective_inputs(tokens)
    logits, _ = model(x, targets=y)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        y.reshape(-1),
        reduction="none",
    ).view(tokens.size(0), -1)


class StatefulRandomSampler(Sampler[int]):
    """Random sampler that can continue from a partially consumed epoch."""

    def __init__(self, dataset_size: int, generator: torch.Generator) -> None:
        self.dataset_size = int(dataset_size)
        self.generator = generator
        self.indices: Optional[torch.Tensor] = None
        self.position = 0

    def __iter__(self):
        if self.indices is None or self.position >= self.dataset_size:
            self.indices = torch.randperm(self.dataset_size, generator=self.generator)
            self.position = 0
        while self.position < self.dataset_size:
            index = int(self.indices[self.position])
            self.position += 1
            yield index

    def __len__(self) -> int:
        return self.dataset_size

    def state_dict(self) -> dict[str, Any]:
        return {
            "generator": self.generator.get_state(),
            "indices": None if self.indices is None else self.indices.clone(),
            "position": self.position,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.generator.set_state(state["generator"])
        self.indices = state["indices"]
        self.position = int(state["position"])


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    sampler: Optional[Sampler[int]] = None,
) -> DataLoader:
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if sampler is not None and shuffle:
        raise ValueError("shuffle must be false when a sampler is provided")
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        generator=generator,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, cfg: ExperimentConfig, device: torch.device) -> float:
    return evaluate_with_positions(model, loader, cfg, device)[0]


@torch.no_grad()
def evaluate_with_positions(
    model: GPT,
    loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
) -> tuple[float, list[float]]:
    """Return mean NTP NLL and its mean for each prediction position."""
    del cfg  # retained in the public signature for evaluation symmetry
    was_training = model.training
    model.eval()
    try:
        total_examples = 0
        position_nll: Optional[torch.Tensor] = None
        for tokens in loader:
            tokens = tokens.to(device, non_blocking=True)
            batch_position_nll = _per_position_nll(model, tokens)
            if position_nll is None:
                position_nll = torch.zeros(batch_position_nll.size(1), dtype=torch.float64)
            position_nll += batch_position_nll.detach().sum(dim=0).cpu().to(torch.float64)
            total_examples += tokens.size(0)
        if total_examples == 0 or position_nll is None:
            raise RuntimeError("empty evaluation loader")
        per_position = (position_nll / total_examples).tolist()
        return float(sum(per_position) / len(per_position)), per_position
    finally:
        model.train(was_training)


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _cpu_tensor_dict(values: Optional[TensorDict]) -> Optional[TensorDict]:
    if values is None:
        return None
    return {int(k): value.detach().cpu().clone() for k, value in values.items()}


def _rules_equal(left: TensorDict, right: TensorDict) -> bool:
    return (
        set(left) == set(right)
        and all(torch.equal(left[level].cpu(), right[level].cpu()) for level in left)
    )


def _optimizer_parameters(
    model: GPT,
    predictor: NextLatentPredictor | None,
) -> list[torch.nn.Parameter]:
    params = list(model.parameters())
    if predictor is not None:
        params.extend(predictor.parameters())
    return params


def _make_optimizer(
    model: GPT,
    cfg: ExperimentConfig,
    predictor: NextLatentPredictor | None = None,
) -> torch.optim.Optimizer:
    parameters = _optimizer_parameters(model, predictor)
    if cfg.optim.name == "adam":
        return torch.optim.Adam(parameters, lr=cfg.optim.learning_rate,
                                betas=tuple(cfg.optim.betas), weight_decay=cfg.optim.weight_decay)
    groups = [
        {"params": [p for p in parameters if p.dim() >= 2], "weight_decay": cfg.optim.weight_decay},
        {"params": [p for p in parameters if p.dim() < 2], "weight_decay": 0.0},
    ]
    extra = {}
    if next(model.parameters()).device.type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        extra["fused"] = True
    return torch.optim.AdamW(groups, lr=cfg.optim.learning_rate,
                             betas=tuple(cfg.optim.betas), **extra)


def _set_warmup_lr(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    update_index: int,
    warmup_updates: int,
) -> float:
    if warmup_updates <= 0:
        lr = base_lr
    else:
        lr = base_lr * min((update_index + 1) / warmup_updates, 1.0)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def save_checkpoint(
    path: str | Path,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
    epoch: int,
    global_step: int,
    metrics: dict[str, Any],
    predictor: NextLatentPredictor | None = None,
    loader_states: Optional[dict[str, Any]] = None,
    trainer_state: Optional[dict[str, Any]] = None,
    rules: Optional[TensorDict] = None,
) -> None:
    """Save a self-contained training checkpoint, including auxiliary state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mps_rng = _mps_rng_state()
    _save_torch(
        {
            "artifact_type": "continuation",
            "model": _cpu_state_dict(model),
            "predictor": None if predictor is None else _cpu_state_dict(predictor),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "config": cfg.to_dict(),
            "rules": _cpu_tensor_dict(rules),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "metrics": metrics,
            "loader_states": loader_states or {},
            "trainer_state": trainer_state or {},
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "mps": mps_rng,
            },
        },
        path,
    )


def restore_rng_state(rng: dict[str, Any]) -> None:
    """Restore Python, NumPy, and PyTorch RNG states from a checkpoint."""
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    mps = getattr(torch, "mps", None)
    if (
        mps is not None
        and torch.backends.mps.is_available()
        and rng.get("mps") is not None
        and hasattr(mps, "set_rng_state")
    ):
        mps.set_rng_state(rng["mps"])


def load_model_from_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[GPT, ExperimentConfig, dict[str, Any]]:
    """Load the GPT backbone only; auxiliary state is intentionally ignored."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Saving schedules have no role in offline model inspection. Existing
    # research snapshots remain readable after retiring their writer.
    saved_config = copy.deepcopy(ckpt["config"])
    saved_config["train"].pop("diagnostic_snapshot_every_updates", None)
    cfg = ExperimentConfig.from_dict(saved_config)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(torch.device(device))
    return model, cfg, ckpt


def saved_checkpoints(run_dir: str | Path) -> dict[int, Path]:
    """Find periodic states plus the final state, which is stored only once."""
    run = Path(run_dir)
    paths = {int(p.stem.removeprefix('step_')): p for p in (run / 'checkpoints').glob('step_*.pt')}
    final = run / 'last.pt'
    if final.exists():
        state = torch.load(final, map_location='cpu', weights_only=False)
        paths[int(state['global_step'])] = final
    return dict(sorted(paths.items()))


def _clip_parameters(
    model: GPT,
    predictor: NextLatentPredictor | None,
) -> Iterable[torch.nn.Parameter]:
    if predictor is None:
        return model.parameters()
    return list(model.parameters()) + list(predictor.parameters())


def _save_torch(payload: dict[str, Any], path: str | Path) -> None:
    """Replace a file only after serialization succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_json(payload: dict[str, Any], path: Path) -> None:
    """Keep the previous measurement history intact if a write is interrupted."""
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def artifact_rules(path: str | Path, artifact: dict[str, Any]) -> TensorDict:
    """Read the exact stored grammar; detect missing or substituted run files."""
    if artifact.get("rules") is not None:
        return artifact["rules"]
    grammar_path = Path(path).parent / artifact["grammar_file"]
    if hashlib.sha256(grammar_path.read_bytes()).hexdigest() != artifact["grammar_sha256"]:
        raise ValueError(f"snapshot grammar checksum mismatch: {grammar_path}")
    return torch.load(grammar_path, map_location="cpu", weights_only=True)


def train_model(
    cfg: ExperimentConfig, train_dataset: Dataset, val_split: RHMSplit, *,
    output_dir: Optional[str | Path] = None,
    resume_from: Optional[str | Path] = None,
    rules: Optional[TensorDict] = None, verbose: bool = True,
) -> dict[str, Any]:
    """Train to the requested budget; report validation without selecting a model.

    Probes observe the validation split in memory; test evaluation is explicit.
    Continuation checkpoints preserve the exact optimizer and data-stream state.
    """
    cfg.validate()
    val_dataset = LeafSequenceDataset(val_split.leaves)
    if len(train_dataset) != cfg.data.train_size or len(val_dataset) != cfg.data.val_size:
        raise ValueError("training/validation dataset sizes do not match configuration")
    if cfg.train.num_workers != 0:
        raise ValueError("exact training continuation requires num_workers=0")
    device = resolve_device(cfg.train.device)
    out = Path(output_dir) if output_dir is not None else None
    checkpoint = None
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        if checkpoint.get("artifact_type") != "continuation":
            raise ValueError("resume requires a continuation checkpoint, not a diagnostic snapshot")
        saved = copy.deepcopy(checkpoint["config"])
        for field in ("max_epochs", "max_updates", "save_checkpoints", "checkpoint_every_updates"):
            saved["train"][field] = cfg.to_dict()["train"][field]
        if saved != cfg.to_dict():
            raise ValueError("resume checkpoint configuration does not match requested configuration")
        saved_rules = checkpoint.get("rules")
        if rules is not None and saved_rules is not None and not _rules_equal(rules, saved_rules):
            raise ValueError("resume checkpoint RHM rules do not match supplied rules")
        if rules is None:
            rules = saved_rules
    if cfg.diagnostics is not None and rules is None:
        raise ValueError("validation diagnostics require RHM rules")
    if cfg.data.resample_train_each_epoch and rules is None:
        raise ValueError("resampling training requires RHM rules")
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        if rules is not None:
            grammar_path = out / 'rules.pt'
            if grammar_path.exists():
                existing = torch.load(grammar_path, map_location='cpu', weights_only=True)
                if not _rules_equal(existing, rules):
                    raise ValueError("run directory contains different RHM rules")
            else:
                _save_torch(_cpu_tensor_dict(rules), grammar_path)
        (out / 'metrics.json').unlink(missing_ok=True)
        (out / 'config.json').write_text(json.dumps(cfg.to_dict(), indent=2) + '\n')

    seed_everything(cfg.model_seed, cfg.train.deterministic, cfg.train.deterministic_strict)
    model = build_model(cfg).to(device)
    predictor = build_auxiliary_predictor(cfg, device)
    optimizer = _make_optimizer(model, cfg, predictor)
    sampler_generator = torch.Generator(device='cpu').manual_seed(cfg.model_seed + 17)
    loader_generator = torch.Generator(device='cpu').manual_seed(cfg.model_seed + 19)
    sampler = StatefulRandomSampler(len(train_dataset), sampler_generator)
    history: list[dict[str, Any]] = []
    global_step = samples_seen = 0
    start_epoch = 1
    running = dict(ntp=0.0, aux=0.0, total=0.0, seen=0)
    resume_partial = False
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        if predictor is not None:
            predictor.load_state_dict(checkpoint['predictor'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        sampler.load_state_dict(checkpoint['loader_states']['train_sampler'])
        loader_generator.set_state(checkpoint['loader_states']['train_loader'])
        state = checkpoint['trainer_state']
        history = state['history']
        samples_seen = state['samples_seen']
        global_step = checkpoint['global_step']
        resume_partial = 0 < sampler.position < len(train_dataset)
        start_epoch = max(1, checkpoint['epoch']) if sampler.position < len(train_dataset) else checkpoint['epoch'] + 1
        if resume_partial:
            running = state['running']
        restore_rng_state(checkpoint['rng'])
    if out is not None:
        _save_json({'history': history}, out / 'history.json')
    val_loader = make_loader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False,
                             num_workers=0, seed=0, device=device)
    updates_per_epoch = math.ceil(len(train_dataset) / min(cfg.train.batch_size, len(train_dataset)))
    warmup_updates = math.ceil(cfg.optim.warmup_epochs * updates_per_epoch)

    def record_evaluation(epoch: int) -> None:
        val_ce, positions = evaluate_with_positions(model, val_loader, cfg, device)
        n = running['seen']
        record = {
            'epoch': epoch, 'global_step': global_step,
            'running_train_ce': running['ntp'] / n if n else None,
            'running_aux_loss': running['aux'] / n if n and predictor is not None else None,
            'running_total_loss': running['total'] / n if n else None,
            'samples_seen': samples_seen, 'tokens_seen': samples_seen * (input_block_size(cfg) - 1),
            'val_ce': val_ce, 'val_nll_by_position': positions, 'val_last_position_nll': positions[-1],
            'lr': float(optimizer.param_groups[0]['lr']),
            'auxiliary_weight': cfg.auxiliary.weight if predictor is not None else None,
            'target_layer': cfg.auxiliary.target_layer,
        }
        if cfg.diagnostics is not None:
            record['diagnostics'] = run_latent_diagnostics(
                model, val_split, rules, cfg, device, settings=cfg.diagnostics)
            record['diagnostic_config'] = asdict(cfg.diagnostics)
        history.append(record)
        if out is not None:
            # Persist measurements during training; metrics.json marks a completed run.
            _save_json({'history': history}, out / 'history.json')
        if verbose:
            print(f"epoch={epoch:4d} step={global_step:7d} val_ce={val_ce:.6f}", flush=True)

    def save_full(path: Path, epoch: int) -> None:
        metrics = history[-1] if history and history[-1]['global_step'] == global_step else {
            'epoch': epoch, 'global_step': global_step, 'samples_seen': samples_seen,
            'tokens_seen': samples_seen * (input_block_size(cfg) - 1),
        }
        save_checkpoint(path, model=model, predictor=predictor, optimizer=optimizer,
                        cfg=cfg, epoch=epoch, global_step=global_step, metrics=metrics,
                        rules=rules,
                        loader_states={'train_sampler': sampler.state_dict(), 'train_loader': loader_generator.get_state()},
                        trainer_state={'history': history, 'samples_seen': samples_seen, 'running': running})

    def save_states(epoch: int, *, final: bool = False) -> None:
        if out is None or not cfg.train.save_checkpoints:
            return
        if final:
            save_full(out / 'last.pt', epoch)
        elif cfg.train.checkpoint_every_updates is not None and global_step % cfg.train.checkpoint_every_updates == 0:
            save_full(out / 'checkpoints' / f'step_{global_step:08d}.pt', epoch)

    if checkpoint is None:
        if cfg.train.eval_at_start or cfg.diagnostics is not None:
            record_evaluation(0)
        save_states(0)
    final_epoch = checkpoint['epoch'] if checkpoint is not None else 0
    for epoch in range(start_epoch, cfg.train.max_epochs + 1):
        if cfg.train.max_updates is not None and global_step >= cfg.train.max_updates:
            break
        if cfg.data.resample_train_each_epoch and (epoch > 1 or resume_partial):
            train_dataset = LeafSequenceDataset(sample_leaf_sequences(
                cfg.data.train_size, rules, seed=_resampled_train_seed(cfg.rhm.train_seed, epoch)))
        if not resume_partial:
            running = dict(ntp=0.0, aux=0.0, total=0.0, seen=0)
        loader = make_loader(train_dataset, batch_size=cfg.train.batch_size, shuffle=False,
                             num_workers=0, seed=0, device=device, generator=loader_generator, sampler=sampler)
        loader_state = loader_generator.get_state()
        iterator = iter(loader)
        if resume_partial:
            # Recreating an interrupted iterator must not consume an extra loader seed.
            loader_generator.set_state(loader_state)
        resume_partial = False
        model.train()
        if predictor is not None:
            predictor.train()
        for tokens in iterator:
            _set_warmup_lr(optimizer, base_lr=cfg.optim.learning_rate,
                           update_index=global_step, warmup_updates=warmup_updates)
            optimizer.zero_grad(set_to_none=True)
            ntp, aux, total = training_losses(model, predictor, tokens.to(device), cfg)
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite training loss at update {global_step + 1}")
            total.backward()
            if cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(_clip_parameters(model, predictor), cfg.train.grad_clip)
            optimizer.step()
            n = tokens.size(0)
            running['ntp'] += float(ntp.item()) * n
            running['aux'] += float(aux.item()) * n if aux is not None else 0
            running['total'] += float(total.item()) * n
            running['seen'] += n
            samples_seen += n
            global_step += 1
            final_epoch = epoch
            budget_done = cfg.train.max_updates is not None and global_step >= cfg.train.max_updates
            epoch_done = sampler.position >= len(train_dataset)
            final = budget_done or (epoch_done and epoch == cfg.train.max_epochs)
            interval = cfg.train.eval_every_updates
            evaluate_due = global_step % interval == 0 if interval is not None else (epoch_done and epoch % cfg.train.eval_every_epochs == 0)
            if evaluate_due or final:
                record_evaluation(epoch)
            save_states(epoch, final=final)
            if budget_done:
                break
    if not history:
        raise RuntimeError('no validation evaluation was performed')
    # A resume request at an already reached budget still produces a final artifact.
    if checkpoint is not None and global_step == checkpoint['global_step']:
        save_states(final_epoch, final=True)
    best = min(history, key=lambda row: row['val_ce'])
    last = history[-1]
    backbone_params = sum(p.numel() for p in model.parameters())
    aux_params = sum(p.numel() for p in predictor.parameters()) if predictor is not None else 0
    metrics = {
        'train_size': cfg.data.train_size, 'val_size': cfg.data.val_size,
        'model_seed': cfg.model_seed, 'rule_seed': cfg.rhm.rule_seed,
        'best_epoch': best['epoch'], 'best_step': best['global_step'], 'best_val_ce': best['val_ce'],
        'val_ce': last['val_ce'], 'last_val_ce': last['val_ce'],
        'val_nll_by_position': last['val_nll_by_position'],
        'val_last_position_nll': last['val_last_position_nll'],
        'last_val_nll_by_position': last['val_nll_by_position'],
        'last_val_last_position_nll': last['val_last_position_nll'],
        'uniform_baseline_nll': math.log(cfg.rhm.v),
        'theory_last_token_nll_bounds': loss_upper_bounds(cfg.rhm.L, v=cfg.rhm.v, m=cfg.rhm.m, s=cfg.rhm.s) if cfg.rhm.m < cfg.rhm.v ** (cfg.rhm.s-1) else None,
        'global_step': global_step, 'total_optimizer_updates': global_step,
        'per_epoch_train_pool': cfg.data.train_size, 'total_sequence_draws': samples_seen,
        'total_predicted_tokens': samples_seen * (input_block_size(cfg)-1),
        'resample_train_each_epoch': cfg.data.resample_train_each_epoch, 'max_updates': cfg.train.max_updates,
        'backbone_num_parameters': backbone_params, 'auxiliary_num_parameters': aux_params,
        'total_num_parameters': backbone_params + aux_params, 'device': str(device),
        'auxiliary_mode': cfg.auxiliary.mode, 'auxiliary_target_layer': cfg.auxiliary.target_layer,
        'auxiliary_weight': cfg.auxiliary.weight if predictor is not None else None,
        'history': history,
    }
    if out is not None:
        _save_json(metrics, out / 'metrics.json')
        (out / 'history.json').unlink(missing_ok=True)
    return metrics
