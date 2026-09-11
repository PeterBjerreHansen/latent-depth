"""Training/evaluation utilities for fixed finite RHM datasets."""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from auxiliary import NextLatentPredictor, build_auxiliary_predictor, next_latent_loss
from config import ExperimentConfig
from diagnostics import run_latent_diagnostics
from nanogpt import GPT, GPTConfig
from rhm.dataset import RHMSplit
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
    # Preserve the original NTP optimizer construction exactly when no
    # auxiliary predictor exists.
    if predictor is None:
        if cfg.optim.name == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=cfg.optim.learning_rate,
                betas=tuple(cfg.optim.betas),
                weight_decay=cfg.optim.weight_decay,
            )
        return model.configure_optimizers(
            weight_decay=cfg.optim.weight_decay,
            learning_rate=cfg.optim.learning_rate,
            betas=tuple(cfg.optim.betas),
            device_type=next(model.parameters()).device.type,
        )

    parameters = _optimizer_parameters(model, predictor)
    if cfg.optim.name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=cfg.optim.learning_rate,
            betas=tuple(cfg.optim.betas),
            weight_decay=cfg.optim.weight_decay,
        )

    decay = [parameter for parameter in parameters if parameter.dim() >= 2]
    no_decay = [parameter for parameter in parameters if parameter.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.optim.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=cfg.optim.learning_rate,
        betas=tuple(cfg.optim.betas),
    )


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
    best_state: Optional[dict[str, torch.Tensor]] = None,
    rules: Optional[TensorDict] = None,
) -> None:
    """Save a self-contained training checkpoint, including auxiliary state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mps_rng = _mps_rng_state()
    torch.save(
        {
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
            "best_model": best_state,
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
    cfg = ExperimentConfig.from_dict(ckpt["config"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(torch.device(device))
    return model, cfg, ckpt


def load_training_state(
    path: str | Path,
    device: str = "cpu",
) -> tuple[GPT, ExperimentConfig, torch.optim.Optimizer, dict[str, Any]]:
    """Load NTP model, optimizer, and RNG state for compatibility tests.

    Auxiliary runs should be resumed through :func:`train_model`, which also
    restores the predictor module.  This helper deliberately retains its
    historical four-value interface.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.from_dict(ckpt["config"])
    resolved_device = torch.device(device)
    model = build_model(cfg).to(resolved_device)
    model.load_state_dict(ckpt["model"], strict=True)
    predictor = build_auxiliary_predictor(cfg, resolved_device)
    if predictor is not None:
        predictor_state = ckpt.get("predictor")
        if predictor_state is None:
            raise ValueError("auxiliary checkpoint is missing predictor state")
        predictor.load_state_dict(predictor_state, strict=True)
    optimizer = _make_optimizer(model, cfg, predictor)
    optimizer.load_state_dict(ckpt["optimizer"])
    restore_rng_state(ckpt["rng"])
    return model, cfg, optimizer, ckpt


def _clip_parameters(
    model: GPT,
    predictor: NextLatentPredictor | None,
) -> Iterable[torch.nn.Parameter]:
    if predictor is None:
        return model.parameters()
    return list(model.parameters()) + list(predictor.parameters())


def train_model(
    cfg: ExperimentConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    *,
    output_dir: Optional[str | Path] = None,
    resume_from: Optional[str | Path] = None,
    diagnostic_split: Optional[RHMSplit] = None,
    rules: Optional[TensorDict] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train one model and select the checkpoint with the best NTP validation NLL.

    The optional Stage-02 auxiliary objective affects training only.  Model
    selection, validation metrics, test metrics, and representation diagnostics
    remain defined by the causal NTP backbone exactly as in Stage 01.
    """
    cfg.validate()
    if len(train_dataset) != cfg.data.train_size:
        raise ValueError(f"train dataset length {len(train_dataset)} != configured {cfg.data.train_size}")
    if len(val_dataset) != cfg.data.val_size or len(test_dataset) != cfg.data.test_size:
        raise ValueError("validation/test dataset sizes do not match configuration")
    if cfg.diagnostics.enabled and (diagnostic_split is None or rules is None):
        raise ValueError("enabled diagnostics require diagnostic_split and RHM rules")

    device = resolve_device(cfg.train.device)
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    resume_rng: Optional[dict[str, Any]] = None
    if resume_from is None:
        seed_everything(
            cfg.model_seed,
            cfg.train.deterministic,
            cfg.train.deterministic_strict,
        )
        # Build GPT first.  Auxiliary predictor construction forks CPU RNG so
        # adding an auxiliary head cannot alter the backbone initialization or
        # subsequent training RNG stream.
        model = build_model(cfg).to(device)
        predictor = build_auxiliary_predictor(cfg, device)
        optimizer = _make_optimizer(model, cfg, predictor)
        train_sampler_generator = torch.Generator(device="cpu").manual_seed(cfg.model_seed + 17)
        train_loader_generator = torch.Generator(device="cpu").manual_seed(cfg.model_seed + 19)
        train_sampler = StatefulRandomSampler(len(train_dataset), train_sampler_generator)
        start_epoch = 1
        global_step = 0
        best_val = float("inf")
        best_epoch = 0
        best_step = 0
        best_state: Optional[dict[str, torch.Tensor]] = None
        history: list[dict[str, Any]] = []
        eval_count = 0
        samples_seen = 0
        last_train_ce = float("inf")
        last_val_ce = float("inf")
        last_val_nll_by_position: list[float] = []
    else:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        checkpoint_rules = checkpoint.get("rules")
        if checkpoint_rules is not None and rules is not None and not _rules_equal(checkpoint_rules, rules):
            raise ValueError("resume checkpoint RHM rules do not match supplied rules")
        saved_cfg = ExperimentConfig.from_dict(checkpoint["config"])
        saved_cfg_dict = saved_cfg.to_dict()
        requested_cfg_dict = cfg.to_dict()
        for field_name in ("max_epochs", "max_updates"):
            saved_cfg_dict["train"][field_name] = requested_cfg_dict["train"][field_name]
        if saved_cfg_dict != requested_cfg_dict:
            raise ValueError("resume checkpoint configuration does not match requested configuration")
        model = build_model(cfg).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        predictor = build_auxiliary_predictor(cfg, device)
        predictor_state = checkpoint.get("predictor")
        if predictor is None:
            if predictor_state is not None:
                raise ValueError("NTP checkpoint unexpectedly contains auxiliary predictor state")
        else:
            if predictor_state is None:
                raise ValueError("auxiliary checkpoint is missing predictor state")
            predictor.load_state_dict(predictor_state, strict=True)
        optimizer = _make_optimizer(model, cfg, predictor)
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_sampler_generator = torch.Generator(device="cpu")
        train_loader_generator = torch.Generator(device="cpu")
        loader_states = checkpoint.get("loader_states", {})
        saved_sampler_state = loader_states.get("train_sampler")
        if saved_sampler_state is None:
            train_sampler_generator.manual_seed(cfg.model_seed + 17)
            train_sampler = StatefulRandomSampler(len(train_dataset), train_sampler_generator)
        else:
            train_sampler = StatefulRandomSampler(len(train_dataset), train_sampler_generator)
            train_sampler.load_state_dict(saved_sampler_state)
        saved_loader_state = loader_states.get("train_loader")
        if saved_loader_state is None:
            train_loader_generator.manual_seed(cfg.model_seed + 19)
        else:
            train_loader_generator.set_state(saved_loader_state)
        resume_rng = checkpoint.get("rng")
        saved_trainer_state = checkpoint.get("trainer_state", {})
        sampler_position = train_sampler.position
        checkpoint_epoch = int(checkpoint["epoch"])
        start_epoch = checkpoint_epoch if sampler_position < len(train_dataset) else checkpoint_epoch + 1
        global_step = int(checkpoint["global_step"])
        best_val = float(saved_trainer_state.get("best_val", float("inf")))
        best_epoch = int(saved_trainer_state.get("best_epoch", 0))
        best_step = int(saved_trainer_state.get("best_step", 0))
        best_state = checkpoint.get("best_model") or _cpu_state_dict(model)
        history = list(saved_trainer_state.get("history", []))
        eval_count = int(saved_trainer_state.get("eval_count", len(history)))
        samples_seen = int(saved_trainer_state.get("samples_seen", global_step * cfg.train.batch_size))
        last_train_ce = float(saved_trainer_state.get("last_train_ce", float("inf")))
        last_val_ce = float(saved_trainer_state.get("last_val_ce", float("inf")))
        last_val_nll_by_position = list(saved_trainer_state.get("last_val_nll_by_position", []))

    train_loader = make_loader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        seed=cfg.model_seed + 17,
        device=device,
        generator=train_loader_generator,
        sampler=train_sampler,
    )
    train_eval_loader = make_loader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        seed=0,
        device=device,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        seed=0,
        device=device,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        seed=0,
        device=device,
    )

    updates_per_epoch = len(train_loader)
    if resume_rng is not None:
        restore_rng_state(resume_rng)

    warmup_updates = int(math.ceil(cfg.optim.warmup_epochs * updates_per_epoch))

    def trainer_state() -> dict[str, Any]:
        return {
            "best_val": best_val,
            "best_epoch": best_epoch,
            "best_step": best_step,
            "history": history,
            "eval_count": eval_count,
            "samples_seen": samples_seen,
            "tokens_seen": samples_seen * (input_block_size(cfg) - 1),
            "last_train_ce": last_train_ce,
            "last_val_ce": last_val_ce,
            "last_val_nll_by_position": last_val_nll_by_position,
        }

    def loader_states() -> dict[str, Any]:
        return {
            "train_sampler": train_sampler.state_dict(),
            "train_loader": train_loader_generator.get_state(),
        }

    def save_update_checkpoint_if_due(*, epoch: int) -> None:
        """Save a checkpoint at an exact optimizer-step boundary when requested."""
        interval = cfg.train.checkpoint_every_updates
        if (
            out is None
            or not cfg.train.save_checkpoints
            or interval is None
            or global_step % interval != 0
        ):
            return
        if history and int(history[-1]["global_step"]) == global_step:
            metrics = dict(history[-1])
        else:
            metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "samples_seen": samples_seen,
                "tokens_seen": samples_seen * (input_block_size(cfg) - 1),
            }
        save_checkpoint(
            out / "checkpoints" / f"step_{global_step:08d}.pt",
            model=model,
            predictor=predictor,
            optimizer=optimizer,
            cfg=cfg,
            epoch=epoch,
            global_step=global_step,
            metrics=metrics,
            loader_states=loader_states(),
            trainer_state=trainer_state(),
            best_state=best_state,
            rules=rules,
        )

    last_evaluated_step: Optional[int] = None

    def record_evaluation(
        *,
        epoch: int,
        running_train_ce: Optional[float],
        running_aux_loss: Optional[float],
        running_total_loss: Optional[float],
    ) -> None:
        """Evaluate NTP, optionally diagnose, select, and checkpoint one state."""
        nonlocal best_epoch, best_state, best_step, best_val, eval_count
        nonlocal last_evaluated_step, last_train_ce, last_val_ce, last_val_nll_by_position

        was_training = model.training
        predictor_was_training = predictor.training if predictor is not None else None
        try:
            last_train_ce = evaluate(model, train_eval_loader, cfg, device)
            last_val_ce, last_val_nll_by_position = evaluate_with_positions(
                model, val_loader, cfg, device
            )
        finally:
            model.train(was_training)
            if predictor is not None and predictor_was_training is not None:
                predictor.train(predictor_was_training)
        next_eval_count = eval_count + 1
        row: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "running_train_ce": running_train_ce,
            "running_aux_loss": running_aux_loss,
            "running_total_loss": running_total_loss,
            "samples_seen": samples_seen,
            "tokens_seen": samples_seen * (input_block_size(cfg) - 1),
            "train_ce": last_train_ce,
            "val_ce": last_val_ce,
            "val_nll_by_position": [float(value) for value in last_val_nll_by_position],
            "val_last_position_nll": last_val_nll_by_position[-1],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "auxiliary_weight": (
                float(cfg.auxiliary.weight) if cfg.auxiliary.mode != "none" else None
            ),
            "target_layer": cfg.auxiliary.target_layer,
        }

        if (
            cfg.diagnostics.enabled
            and next_eval_count % cfg.diagnostics.every_evals == 0
            and diagnostic_split is not None
            and rules is not None
        ):
            row["diagnostics"] = run_latent_diagnostics(
                model,
                diagnostic_split,
                rules,
                cfg,
                device,
            )

        improved = last_val_ce < best_val
        if improved:
            best_val = last_val_ce
            best_epoch = epoch
            best_step = global_step
            best_state = _cpu_state_dict(model)
        history.append(row)
        eval_count = next_eval_count
        last_evaluated_step = global_step

        if verbose:
            diagnostic_note = " +diagnostics" if "diagnostics" in row else ""
            aux_note = (
                f" aux={running_aux_loss:.6f}" if running_aux_loss is not None else ""
            )
            print(
                f"epoch={epoch:4d} step={global_step:7d} "
                f"train_ce={last_train_ce:.6f} val_ce={last_val_ce:.6f} "
                f"lr={row['lr']:.3e}{aux_note}{diagnostic_note}"
            )

        if out is not None and cfg.train.save_checkpoints:
            if improved:
                save_checkpoint(
                    out / "best.pt",
                    model=model,
                    predictor=predictor,
                    optimizer=optimizer,
                    cfg=cfg,
                    epoch=epoch,
                    global_step=global_step,
                    metrics=row,
                    loader_states=loader_states(),
                    trainer_state=trainer_state(),
                    best_state=best_state,
                    rules=rules,
                )
            if (
                cfg.train.checkpoint_every_updates is None
                and cfg.train.checkpoint_every_evals is not None
                and eval_count % cfg.train.checkpoint_every_evals == 0
            ):
                save_checkpoint(
                    out / "checkpoints" / f"step_{global_step:08d}.pt",
                    model=model,
                    predictor=predictor,
                    optimizer=optimizer,
                    cfg=cfg,
                    epoch=epoch,
                    global_step=global_step,
                    metrics=row,
                    loader_states=loader_states(),
                    trainer_state=trainer_state(),
                    best_state=best_state,
                    rules=rules,
                )

    if resume_from is None and cfg.train.eval_at_start:
        record_evaluation(
            epoch=0,
            running_train_ce=None,
            running_aux_loss=None,
            running_total_loss=None,
        )
        save_update_checkpoint_if_due(epoch=0)

    update_based_evaluation = cfg.train.eval_every_updates is not None
    stop_training = False
    for epoch in range(start_epoch, cfg.train.max_epochs + 1):
        if cfg.train.max_updates is not None and global_step >= cfg.train.max_updates:
            break
        model.train()
        if predictor is not None:
            predictor.train()
        running_ntp = 0.0
        running_aux = 0.0
        running_total = 0.0
        seen = 0
        for tokens in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            _set_warmup_lr(
                optimizer,
                base_lr=cfg.optim.learning_rate,
                update_index=global_step,
                warmup_updates=warmup_updates,
            )
            optimizer.zero_grad(set_to_none=True)
            ntp_loss, aux_loss, total_loss = training_losses(model, predictor, tokens, cfg)
            total_loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    _clip_parameters(model, predictor),
                    cfg.train.grad_clip,
                )
            optimizer.step()

            batch_n = tokens.size(0)
            running_ntp += float(ntp_loss.item()) * batch_n
            if aux_loss is not None:
                running_aux += float(aux_loss.item()) * batch_n
            running_total += float(total_loss.item()) * batch_n
            seen += batch_n
            samples_seen += batch_n
            global_step += 1

            reached_update_budget = (
                cfg.train.max_updates is not None
                and global_step >= cfg.train.max_updates
            )

            if update_based_evaluation:
                update_interval = cfg.train.eval_every_updates
                if global_step % update_interval == 0 or reached_update_budget:
                    record_evaluation(
                        epoch=epoch,
                        running_train_ce=running_ntp / max(seen, 1),
                        running_aux_loss=(
                            running_aux / max(seen, 1) if predictor is not None else None
                        ),
                        running_total_loss=running_total / max(seen, 1),
                    )

            save_update_checkpoint_if_due(epoch=epoch)

            # Stop before DataLoader requests another batch, so sampler state
            # in a mid-epoch checkpoint is exactly reproducible.
            if reached_update_budget:
                stop_training = True
                break

        if update_based_evaluation:
            if epoch == cfg.train.max_epochs and global_step != last_evaluated_step:
                record_evaluation(
                    epoch=epoch,
                    running_train_ce=running_ntp / max(seen, 1),
                    running_aux_loss=(
                        running_aux / max(seen, 1) if predictor is not None else None
                    ),
                    running_total_loss=running_total / max(seen, 1),
                )
            if stop_training:
                break
            continue

        running_train_ce = running_ntp / max(seen, 1)
        should_eval = (
            epoch % cfg.train.eval_every_epochs == 0
            or epoch == cfg.train.max_epochs
            or stop_training
        )
        if should_eval:
            record_evaluation(
                epoch=epoch,
                running_train_ce=running_train_ce,
                running_aux_loss=(running_aux / max(seen, 1) if predictor is not None else None),
                running_total_loss=running_total / max(seen, 1),
            )

        if stop_training:
            break

    if best_state is None:
        raise RuntimeError("no validation evaluation was performed")

    final_epoch = int(history[-1]["epoch"])
    if out is not None and cfg.train.save_checkpoints:
        # Save the true final training state before loading best-validation
        # backbone weights for the one-time test evaluation.
        save_checkpoint(
            out / "last.pt",
            model=model,
            predictor=predictor,
            optimizer=optimizer,
            cfg=cfg,
            epoch=final_epoch,
            global_step=global_step,
            metrics=history[-1],
            loader_states=loader_states(),
            trainer_state=trainer_state(),
            best_state=best_state,
            rules=rules,
        )

    # Test is touched only after NTP validation-based selection is complete.
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    selected_val_ce, selected_val_nll_by_position = evaluate_with_positions(
        model, val_loader, cfg, device
    )
    test_ce, test_nll_by_position = evaluate_with_positions(model, test_loader, cfg, device)

    theory_bounds = None
    if cfg.rhm.m < cfg.rhm.v ** (cfg.rhm.s - 1):
        theory_bounds = loss_upper_bounds(
            cfg.rhm.L, v=cfg.rhm.v, m=cfg.rhm.m, s=cfg.rhm.s
        )

    backbone_params = model.num_parameters()
    aux_params = sum(p.numel() for p in predictor.parameters()) if predictor is not None else 0
    metrics: dict[str, Any] = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "model_seed": cfg.model_seed,
        "rule_seed": cfg.rhm.rule_seed,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "best_val_ce": best_val,
        "selected_val_ce": selected_val_ce,
        "val_ce": selected_val_ce,
        "test_ce": test_ce,
        "val_last_position_nll": selected_val_nll_by_position[-1],
        "test_last_position_nll": test_nll_by_position[-1],
        "uniform_baseline_nll": math.log(cfg.rhm.v),
        "theory_last_token_nll_bounds": theory_bounds,
        "last_train_ce": last_train_ce,
        "last_val_ce": last_val_ce,
        "last_val_last_position_nll": last_val_nll_by_position[-1],
        "global_step": global_step,
        "total_samples_seen": samples_seen,
        "total_tokens_seen": samples_seen * (input_block_size(cfg) - 1),
        "max_updates": cfg.train.max_updates,
        "val_nll_by_position": selected_val_nll_by_position,
        "last_val_nll_by_position": last_val_nll_by_position,
        "test_nll_by_position": test_nll_by_position,
        "num_parameters": backbone_params,
        "backbone_num_parameters": backbone_params,
        "auxiliary_num_parameters": aux_params,
        "total_num_parameters": backbone_params + aux_params,
        "device": str(device),
        "objective": cfg.objective.mode,
        "auxiliary_mode": cfg.auxiliary.mode,
        "auxiliary_target_layer": cfg.auxiliary.target_layer,
        "auxiliary_weight": (
            float(cfg.auxiliary.weight) if cfg.auxiliary.mode != "none" else None
        ),
        "history": history,
    }

    if out is not None:
        import json

        with open(out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return metrics
