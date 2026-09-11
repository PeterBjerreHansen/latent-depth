import copy
from pathlib import Path

import torch

from auxiliary import (
    NextLatentPredictor,
    build_auxiliary_predictor,
    next_latent_loss,
    next_latent_views,
)
from config import ExperimentConfig, TargetDepthSweepConfig
from diagnose import diagnose_checkpoint
from rhm.dataset import LeafSequenceDataset, build_rhm_bundle
from training import (
    _clip_parameters,
    _make_optimizer,
    build_model,
    seed_everything,
    train_model,
    training_losses,
)


def _cfg(*, auxiliary: bool = True, target_layer: int = 1) -> ExperimentConfig:
    payload = {
        "rhm": {
            "v": 8,
            "n": 8,
            "m": 2,
            "s": 2,
            "L": 2,
            "rule_seed": 0,
            "train_seed": 10,
            "val_seed": 20,
            "test_seed": 30,
        },
        "data": {"train_size": 32, "val_size": 32, "test_size": 32},
        "model": {"n_layer": 2, "n_head": 2, "n_embd": 16, "dropout": 0.0},
        "objective": {"mode": "next_token"},
        "auxiliary": {
            "mode": "next_latent" if auxiliary else "none",
            "target_layer": target_layer if auxiliary else None,
            "weight": 0.1,
            "predictor_hidden_mult": 2,
            "seed": 54321,
        },
        "optim": {
            "name": "adamw",
            "learning_rate": 0.001,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "warmup_epochs": 0.0,
        },
        "train": {
            "batch_size": 16,
            "max_epochs": 10,
            "max_updates": 4,
            "grad_clip": 1.0,
            "eval_every_epochs": 1,
            "eval_every_updates": 2,
            "eval_at_start": True,
            "num_workers": 0,
            "device": "cpu",
            "deterministic": True,
            "deterministic_strict": True,
            "save_checkpoints": True,
            "checkpoint_every_updates": 2,
        },
        "model_seed": 5,
    }
    return ExperimentConfig.from_dict(payload)


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
        LeafSequenceDataset(bundle.val.leaves),
        LeafSequenceDataset(bundle.test.leaves),
    )


def _assert_nested_equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_nested_equal(x, y)
    else:
        assert a == b


def test_next_latent_views_use_final_source_t_and_target_t_plus_1():
    # Three residual streams: embedding, block 1, block 2.
    hidden = []
    for layer in range(3):
        values = torch.arange(5, dtype=torch.float32).view(1, 5, 1) + 100 * layer
        hidden.append(values.clone().requires_grad_(True))

    source, target = next_latent_views(hidden, target_layer=1)
    assert source[:, :, 0].tolist() == [[200.0, 201.0, 202.0, 203.0]]
    assert target[:, :, 0].tolist() == [[101.0, 102.0, 103.0, 104.0]]
    assert source.requires_grad
    assert not target.requires_grad


def test_target_branch_is_detached_from_auxiliary_gradient():
    target_stream = torch.randn(2, 5, 4, requires_grad=True)
    source_stream = torch.randn(2, 5, 4, requires_grad=True)
    hidden = [torch.randn(2, 5, 4, requires_grad=True), target_stream, source_stream]
    predictor = NextLatentPredictor(4, hidden_mult=2)
    loss = next_latent_loss(hidden, predictor, target_layer=1)
    loss.backward()
    assert source_stream.grad is not None
    assert source_stream.grad[:, :-1].abs().sum() > 0
    assert target_stream.grad is None
    assert any(parameter.grad is not None for parameter in predictor.parameters())


def test_embedding_and_final_target_layers_select_expected_streams():
    hidden = [torch.full((1, 3, 2), float(layer)) for layer in range(4)]
    _, target0 = next_latent_views(hidden, 0)
    _, target3 = next_latent_views(hidden, 3)
    assert torch.all(target0 == 0)
    assert torch.all(target3 == 3)


def test_predictor_initialization_preserves_global_rng_and_backbone_initialization():
    cfg_aux = _cfg(auxiliary=True, target_layer=1)
    cfg_ntp = _cfg(auxiliary=False)

    seed_everything(cfg_ntp.model_seed, True, True)
    ntp_model = build_model(cfg_ntp)
    ntp_state = {k: v.detach().clone() for k, v in ntp_model.state_dict().items()}

    seed_everything(cfg_aux.model_seed, True, True)
    aux_model = build_model(cfg_aux)
    before_predictor_rng = torch.get_rng_state().clone()
    predictor = build_auxiliary_predictor(cfg_aux, torch.device("cpu"))
    after_predictor_rng = torch.get_rng_state().clone()

    assert predictor is not None
    torch.testing.assert_close(before_predictor_rng, after_predictor_rng, atol=0.0, rtol=0.0)
    for name, value in aux_model.state_dict().items():
        torch.testing.assert_close(value, ntp_state[name], atol=0.0, rtol=0.0)


def test_auxiliary_disabled_training_loss_is_exact_ntp_loss():
    cfg = _cfg(auxiliary=False)
    torch.manual_seed(cfg.model_seed)
    model = build_model(cfg)
    tokens = torch.randint(0, cfg.rhm.v, (4, cfg.rhm.s**cfg.rhm.L))
    ntp, aux, total = training_losses(model, None, tokens, cfg)
    assert aux is None
    torch.testing.assert_close(ntp, total, atol=0.0, rtol=0.0)


def test_predictor_is_in_optimizer_clip_set_and_updates():
    cfg = _cfg(auxiliary=True, target_layer=1)
    seed_everything(cfg.model_seed, True, True)
    model = build_model(cfg)
    predictor = build_auxiliary_predictor(cfg, torch.device("cpu"))
    assert predictor is not None
    optimizer = _make_optimizer(model, cfg, predictor)
    before = {k: v.detach().clone() for k, v in predictor.state_dict().items()}

    tokens = torch.randint(0, cfg.rhm.v, (4, cfg.rhm.s**cfg.rhm.L))
    _, aux_loss, total = training_losses(model, predictor, tokens, cfg)
    assert aux_loss is not None
    optimizer.zero_grad(set_to_none=True)
    total.backward()

    clip_params = list(_clip_parameters(model, predictor))
    predictor_ids = {id(p) for p in predictor.parameters()}
    assert predictor_ids.issubset({id(p) for p in clip_params})
    torch.nn.utils.clip_grad_norm_(clip_params, cfg.train.grad_clip)
    optimizer.step()
    assert any(not torch.equal(value, before[name]) for name, value in predictor.state_dict().items())


def test_auxiliary_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path):
    cfg = _cfg(auxiliary=True, target_layer=1)
    bundle = _bundle(cfg)
    datasets = _datasets(bundle)

    full_dir = tmp_path / "full"
    train_model(cfg, *datasets, output_dir=full_dir, rules=bundle.rules, verbose=False)

    split_cfg = copy.deepcopy(cfg)
    split_cfg.train.max_updates = 2
    split_dir = tmp_path / "split"
    train_model(split_cfg, *datasets, output_dir=split_dir, rules=bundle.rules, verbose=False)

    resumed_dir = tmp_path / "resumed"
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
    _assert_nested_equal(full["model"], resumed["model"])
    _assert_nested_equal(full["predictor"], resumed["predictor"])
    _assert_nested_equal(full["optimizer"], resumed["optimizer"])
    _assert_nested_equal(full["loader_states"]["train_sampler"], resumed["loader_states"]["train_sampler"])
    assert full["global_step"] == resumed["global_step"] == 4


def test_step_zero_backbone_is_identical_between_ntp_and_auxiliary_arms(tmp_path: Path):
    ntp_cfg = _cfg(auxiliary=False)
    aux_cfg = _cfg(auxiliary=True, target_layer=2)
    ntp_cfg.train.max_updates = aux_cfg.train.max_updates = 1
    bundle = _bundle(ntp_cfg)
    datasets = _datasets(bundle)

    train_model(ntp_cfg, *datasets, output_dir=tmp_path / "ntp", rules=bundle.rules, verbose=False)
    train_model(aux_cfg, *datasets, output_dir=tmp_path / "aux", rules=bundle.rules, verbose=False)

    ntp0 = torch.load(
        tmp_path / "ntp" / "checkpoints" / "step_00000000.pt",
        map_location="cpu", weights_only=False,
    )
    aux0 = torch.load(
        tmp_path / "aux" / "checkpoints" / "step_00000000.pt",
        map_location="cpu", weights_only=False,
    )
    _assert_nested_equal(ntp0["model"], aux0["model"])
    assert ntp0["predictor"] is None
    assert aux0["predictor"] is not None


def test_offline_diagnostics_ignore_predictor_and_work_on_auxiliary_checkpoint(tmp_path: Path):
    cfg = _cfg(auxiliary=True, target_layer=1)
    cfg.rhm.L = 3
    cfg.data.train_size = 16
    cfg.data.val_size = 16
    cfg.data.test_size = 16
    cfg.train.batch_size = 8
    cfg.train.max_updates = 1
    cfg.diagnostics.num_sequences = 8
    cfg.diagnostics.probe_steps = 2
    bundle = _bundle(cfg)
    train_model(
        cfg,
        *_datasets(bundle),
        output_dir=tmp_path,
        rules=bundle.rules,
        verbose=False,
    )
    result = diagnose_checkpoint(
        tmp_path / "last.pt",
        split="val",
        device="cpu",
        num_sequences=8,
        probe_steps=2,
    )
    assert result["diagnostics"]["levels"] == [1, 2]


def test_target_depth_sweep_config_accepts_ntp_and_fixed_layers(tmp_path: Path):
    path = tmp_path / "sweep.json"
    payload = {
        "experiment": _cfg(auxiliary=False).to_dict(),
        "target_layers": [None, 0, 1, 2],
        "grammar_seeds": [0],
        "model_seeds": [0],
    }
    import json
    path.write_text(json.dumps(payload), encoding="utf-8")
    sweep = TargetDepthSweepConfig.from_json(path)
    assert sweep.target_layers == [None, 0, 1, 2]

