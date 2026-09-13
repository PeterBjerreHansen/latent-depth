import copy
from pathlib import Path

import pytest
import torch

import training
from config import ExperimentConfig
from diagnose import diagnose_checkpoint
from diagnose_trajectory import diagnose_trajectory
from evaluate_snapshot import evaluate_snapshot
from rhm.dataset import LeafSequenceDataset, build_rhm_bundle, sample_leaf_sequences
from training import train_model, load_model_from_checkpoint, artifact_rules


def _cfg():
    return ExperimentConfig.from_dict({
        'rhm': {'v': 8, 'n': 8, 'm': 2, 's': 2, 'L': 3,
                'train_seed': 10, 'val_seed': 20, 'test_seed': 30},
        'data': {'train_size': 48, 'val_size': 32, 'test_size': 32,
                 'resample_train_each_epoch': True},
        'model': {'n_layer': 2, 'n_head': 2, 'n_embd': 16, 'dropout': .1},
        'train': {'batch_size': 16, 'max_epochs': 4, 'max_updates': 7,
                  'eval_every_updates': 2, 'eval_at_start': True, 'device': 'cpu',
                  'deterministic_strict': True, 'checkpoint_every_updates': 2,
                  'diagnostic_snapshot_every_updates': 1},
    })


def _bundle(cfg):
    return build_rhm_bundle(**vars(cfg.rhm), train_size=cfg.data.train_size,
                            val_size=cfg.data.val_size, test_size=cfg.data.test_size)


def _run(cfg, output_dir=None, **kwargs):
    bundle = _bundle(cfg)
    return train_model(cfg, LeafSequenceDataset(bundle.train.leaves),
                       LeafSequenceDataset(bundle.val.leaves), rules=bundle.rules,
                       output_dir=output_dir, verbose=False, **kwargs)


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def _equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: _equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): _equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('auxiliary', [False, True])
def test_snapshots_are_observational_and_backbone_only(tmp_path, auxiliary):
    cfg = _cfg()
    if auxiliary:
        cfg.auxiliary.mode = 'next_latent'
        cfg.auxiliary.target_layer = 1
    plain_cfg = copy.deepcopy(cfg)
    plain_cfg.train.diagnostic_snapshot_every_updates = None
    plain_metrics = _run(plain_cfg, tmp_path/'plain')
    snapshot_metrics = _run(cfg, tmp_path/'snapshots')
    assert plain_metrics == snapshot_metrics
    plain, observed = _load(tmp_path/'plain/last.pt'), _load(tmp_path/'snapshots/last.pt')
    for key in ['model', 'optimizer', 'predictor', 'loader_states', 'trainer_state']:
        _equal(plain[key], observed[key])
    _equal(plain['rng']['torch'], observed['rng']['torch'])
    files = sorted((tmp_path/'snapshots/diagnostic_snapshots').glob('*.pt'))
    assert len(files) == 8
    state = _load(files[-1])
    assert state['artifact_type'] == 'diagnostic'
    assert not set(state) & {'optimizer', 'predictor', 'rng', 'rules', 'best_model'}
    _equal(state['model'], observed['model'])
    assert not (tmp_path/'snapshots/best.pt').exists()
    assert 'best_model' not in observed


@pytest.mark.parametrize('step', [0, 2, 3, 4])
@pytest.mark.parametrize('auxiliary', [False, True])
def test_exact_resume_preserves_training_and_metrics(tmp_path, step, auxiliary):
    cfg = _cfg()
    cfg.train.checkpoint_every_updates = 1
    if auxiliary:
        cfg.auxiliary.mode = 'next_latent'
        cfg.auxiliary.target_layer = 1
    full_metrics = _run(cfg, tmp_path/'full')
    resumed_metrics = _run(cfg, tmp_path/'resumed',
                           resume_from=tmp_path/'full/checkpoints'/f'step_{step:08d}.pt')
    assert full_metrics == resumed_metrics
    a, b = _load(tmp_path/'full/last.pt'), _load(tmp_path/'resumed/last.pt')
    for key in ['model', 'optimizer', 'predictor', 'loader_states', 'trainer_state']:
        _equal(a[key], b[key])
    _equal(a['rng']['torch'], b['rng']['torch'])


def test_resume_rejects_snapshot_and_wrong_grammar(tmp_path):
    cfg = _cfg()
    _run(cfg, tmp_path/'run')
    with pytest.raises(ValueError, match='continuation checkpoint'):
        _run(cfg, tmp_path/'resume', resume_from=tmp_path/'run/diagnostic_snapshots/step_00000002.pt')
    bundle = _bundle(cfg)
    bundle.rules[1][0, 0, 0] = (bundle.rules[1][0, 0, 0]+1) % cfg.rhm.v
    with pytest.raises(ValueError, match='RHM rules'):
        train_model(cfg, LeafSequenceDataset(bundle.train.leaves), LeafSequenceDataset(bundle.val.leaves),
                    rules=bundle.rules, resume_from=tmp_path/'run/last.pt', verbose=False)


def test_snapshot_grammar_is_verified(tmp_path):
    _run(_cfg(), tmp_path)
    path = tmp_path/'diagnostic_snapshots/step_00000002.pt'
    _, _, state = load_model_from_checkpoint(path)
    assert artifact_rules(path, state)
    (tmp_path/'rules.pt').write_bytes(b'wrong grammar')
    with pytest.raises(ValueError, match='checksum'):
        artifact_rules(path, state)


def test_training_only_evaluates_validation_and_reports_final_state(tmp_path, monkeypatch):
    cfg = _cfg()
    sizes = []
    original = training.evaluate_with_positions
    def record(model, loader, config, device):
        sizes.append(len(loader.dataset))
        return original(model, loader, config, device)
    monkeypatch.setattr(training, 'evaluate_with_positions', record)
    metrics = _run(cfg, tmp_path)
    assert sizes == [cfg.data.val_size]*5
    assert [r['global_step'] for r in metrics['history']] == [0, 2, 4, 6, 7]
    assert not set(metrics) & {'test_ce', 'test_nll_by_position', 'selected_val_ce', 'last_train_ce'}
    assert all('diagnostics' not in r and 'train_ce' not in r for r in metrics['history'])
    evaluated = evaluate_snapshot(tmp_path/'last.pt', device='cpu')
    assert evaluated['ce'] == metrics['val_ce']
    assert evaluated['nll_by_position'] == metrics['val_nll_by_position']
    assert metrics['total_predicted_tokens'] == 7 * 16 * 7
    test = evaluate_snapshot(tmp_path/'last.pt', split='test', device='cpu')
    assert test['split'] == 'test' and test['num_sequences'] == 32


def test_offline_snapshot_and_checkpoint_diagnostics_agree(tmp_path):
    _run(_cfg(), tmp_path)
    paths = [tmp_path/'checkpoints/step_00000002.pt', tmp_path/'diagnostic_snapshots/step_00000002.pt']
    results = [diagnose_checkpoint(p, device='cpu', num_sequences=16, probe_steps=8, controls=True) for p in paths]
    assert results[0]['diagnostics'] == results[1]['diagnostics']
    assert results[0]['diagnostic_config'] == results[1]['diagnostic_config']


def test_sparse_supporting_diagnostics_are_separate_from_probe_trajectory(tmp_path):
    _run(_cfg(), tmp_path/'run')
    probes = diagnose_trajectory(tmp_path/'run', tmp_path/'probe', device='cpu', num_sequences=16, probe_steps=3)
    supporting = diagnose_trajectory(tmp_path/'run', tmp_path/'support', device='cpu',
                                    metric='clustering', num_sequences=16, every_updates=3)
    assert len(probes['checkpoints']) == 8
    assert [r['checkpoint_global_step'] for r in supporting['checkpoints']] == [0, 3, 6]
    assert all('linear_probe' not in r['diagnostics'] for r in supporting['checkpoints'])
    assert all('synonym_clustering' not in r['diagnostics'] for r in probes['checkpoints'])
    with pytest.raises(ValueError, match='missing snapshot'):
        diagnose_trajectory(tmp_path/'run', tmp_path/'bad', steps=[999])


def test_fresh_pools_and_epoch_evaluation_remain_supported(tmp_path, monkeypatch):
    cfg = _cfg()
    cfg.train.eval_every_updates = None
    calls = []
    original = training.sample_leaf_sequences
    def record(n, rules, seed):
        calls.append(seed)
        return original(n, rules, seed)
    monkeypatch.setattr(training, 'sample_leaf_sequences', record)
    metrics = _run(cfg, tmp_path)
    assert calls == [training._resampled_train_seed(10, 2), training._resampled_train_seed(10, 3)]
    assert [r['global_step'] for r in metrics['history']] == [0, 3, 6, 7]


def test_unmeasured_checkpoint_has_its_own_step(tmp_path):
    cfg = _cfg()
    cfg.train.checkpoint_every_updates = 3
    _run(cfg, tmp_path)
    checkpoint = _load(tmp_path/'checkpoints/step_00000003.pt')
    assert checkpoint['metrics']['global_step'] == 3
    assert checkpoint['metrics']['tokens_seen'] == 3*16*7
    assert 'val_ce' not in checkpoint['metrics']


def test_training_without_output_directory():
    assert _run(_cfg())['global_step'] == 7
