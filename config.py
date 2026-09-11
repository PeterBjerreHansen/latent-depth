"""Typed JSON configuration for RHM/nanoGPT experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Optional


@dataclass
class RHMConfig:
    v: int = 32
    n: int = 32
    m: int = 8
    s: int = 2
    L: int = 3
    rule_seed: int = 0
    train_seed: int = 1000
    val_seed: int = 2000
    test_seed: int = 3000

    def validate(self) -> None:
        if min(self.v, self.n, self.m, self.s, self.L) <= 0:
            raise ValueError("RHM dimensions must be positive")
        if self.n * self.m > self.v**self.s:
            raise ValueError("n*m must be <= v**s")
        if self.v * self.m > self.v**self.s:
            raise ValueError("m must be <= v**(s-1) for unambiguous rules")


@dataclass
class DataConfig:
    train_size: int = 131072
    val_size: int = 32768
    test_size: int = 32768

    def validate(self) -> None:
        if min(self.train_size, self.val_size, self.test_size) <= 0:
            raise ValueError("all split sizes must be positive")


@dataclass
class ModelConfig:
    n_layer: int = 3
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = True

    def validate(self) -> None:
        if min(self.n_layer, self.n_head, self.n_embd) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0,1)")


@dataclass
class ObjectiveConfig:
    mode: str = "next_token"

    def validate(self) -> None:
        if self.mode != "next_token":
            raise ValueError("objective mode only supports 'next_token'")


@dataclass
class OptimConfig:
    name: str = "adamw"
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    warmup_epochs: float = 0.0

    def validate(self) -> None:
        if self.name not in {"adam", "adamw"}:
            raise ValueError("optimizer must be 'adam' or 'adamw'")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if len(self.betas) != 2 or not all(0 <= b < 1 for b in self.betas):
            raise ValueError("betas must contain two values in [0,1)")
        if self.weight_decay < 0 or self.warmup_epochs < 0:
            raise ValueError("weight_decay and warmup_epochs must be nonnegative")


@dataclass
class DiagnosticsConfig:
    """Optional observers for the frozen causal NTP representation."""

    enabled: bool = False
    linear_probe: bool = True
    synonym_clustering: bool = True
    every_evals: int = 1
    num_sequences: int = 1024
    probe_steps: int = 300
    probe_lr: float = 1e-3
    seed: int = 12345
    eps: float = 1e-8

    def validate(self) -> None:
        if self.every_evals <= 0:
            raise ValueError("diagnostics.every_evals must be positive")
        if self.num_sequences < 4:
            raise ValueError("diagnostics.num_sequences must be at least 4")
        if self.probe_steps <= 0:
            raise ValueError("diagnostics.probe_steps must be positive")
        if self.probe_lr <= 0:
            raise ValueError("diagnostics.probe_lr must be positive")
        if self.eps <= 0:
            raise ValueError("diagnostics.eps must be positive")
        if self.enabled and not (self.linear_probe or self.synonym_clustering):
            raise ValueError("enabled diagnostics require at least one metric")


@dataclass
class TrainConfig:
    batch_size: int = 256
    max_epochs: int = 100
    max_updates: Optional[int] = None
    grad_clip: float = 1.0
    eval_every_epochs: int = 1
    eval_every_updates: Optional[int] = None
    eval_at_start: bool = False
    num_workers: int = 0
    device: str = "auto"
    deterministic: bool = True
    deterministic_strict: bool = False
    save_checkpoints: bool = True
    checkpoint_every_evals: Optional[int] = None

    def validate(self) -> None:
        if self.batch_size <= 0 or self.max_epochs <= 0 or self.eval_every_epochs <= 0:
            raise ValueError("batch_size, max_epochs, eval_every_epochs must be positive")
        if self.eval_every_updates is not None and self.eval_every_updates <= 0:
            raise ValueError("eval_every_updates must be positive when provided")
        if self.max_updates is not None and self.max_updates <= 0:
            raise ValueError("max_updates must be positive when provided")
        if self.checkpoint_every_evals is not None and self.checkpoint_every_evals <= 0:
            raise ValueError("checkpoint_every_evals must be positive when provided")
        if self.deterministic_strict and not self.deterministic:
            raise ValueError("deterministic_strict requires deterministic=true")
        if self.grad_clip < 0 or self.num_workers < 0:
            raise ValueError("invalid training scalar")


@dataclass
class ExperimentConfig:
    rhm: RHMConfig = field(default_factory=RHMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model_seed: int = 0

    def validate(self) -> None:
        self.rhm.validate()
        self.data.validate()
        self.model.validate()
        self.optim.validate()
        self.diagnostics.validate()
        self.train.validate()
        self.objective.validate()
        if self.diagnostics.enabled and self.diagnostics.synonym_clustering and self.rhm.m < 2:
            raise ValueError("synonym-clustering diagnostics require RHM m >= 2")
        if self.diagnostics.enabled and self.rhm.L < 2:
            raise ValueError("latent diagnostics require RHM depth L >= 2")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        cfg = cls(
            rhm=RHMConfig(**d.get("rhm", {})),
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            objective=ObjectiveConfig(**d.get("objective", {})),
            optim=OptimConfig(**{**d.get("optim", {}), "betas": tuple(d.get("optim", {}).get("betas", (0.9, 0.95)))}),
            diagnostics=DiagnosticsConfig(**d.get("diagnostics", {})),
            train=TrainConfig(**d.get("train", {})),
            model_seed=d.get("model_seed", 0),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SweepConfig:
    experiment: ExperimentConfig
    train_sizes: list[int]
    grammar_seeds: list[int]
    model_seeds: list[int]
    replicates: Optional[list[tuple[int, int]]] = None
    samples_per_example: Optional[float] = None

    @classmethod
    def from_json(cls, path: str | Path) -> "SweepConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        exp = ExperimentConfig.from_dict(d["experiment"])
        train_sizes = [int(x) for x in d["train_sizes"]]
        grammar_seeds = [int(x) for x in d.get("grammar_seeds", [exp.rhm.rule_seed])]
        model_seeds = [int(x) for x in d.get("model_seeds", [exp.model_seed])]
        replicates = None
        if "replicates" in d:
            replicates = [
                (int(r["grammar_seed"]), int(r["model_seed"]))
                for r in d["replicates"]
            ]
            if not replicates or len(set(replicates)) != len(replicates):
                raise ValueError("replicates must be nonempty unique (grammar_seed, model_seed) pairs")
        if not train_sizes or min(train_sizes) <= 0:
            raise ValueError("train_sizes must be nonempty and positive")
        if len(set(train_sizes)) != len(train_sizes):
            raise ValueError("train_sizes must be unique")
        samples_per_example = d.get("samples_per_example")
        if samples_per_example is not None and float(samples_per_example) <= 0:
            raise ValueError("samples_per_example must be positive when provided")
        return cls(
            exp,
            sorted(train_sizes),
            grammar_seeds,
            model_seeds,
            replicates,
            None if samples_per_example is None else float(samples_per_example),
        )

    def replicate_pairs(self) -> list[tuple[int, int]]:
        if self.replicates is not None:
            return list(self.replicates)
        return [(g, m) for g in self.grammar_seeds for m in self.model_seeds]
