import copy
from pathlib import Path

import pytest
import torch

import training
from config import ExperimentConfig, DiagnosticsConfig
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
                  'save_checkpoints': True},
    })


def _bundle(cfg):
    return build_rhm_bundle(**vars(cfg.rhm), train_size=cfg.data.train_size,
                            val_size=cfg.data.val_size, test_size=cfg.data.test_size)


def _run(cfg, output_dir=None, **kwargs):
    bundle = _bundle(cfg)
    return train_model(cfg, LeafSequenceDataset(bundle.train.leaves),
                       bundle.val, rules=bundle.rules,
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
def test_validation_probes_do_not_change_training(tmp_path, auxiliary):
    cfg = _cfg()
    if auxiliary:
        cfg.auxiliary.mode = 'next_latent'
        cfg.auxiliary.target_layer = 1
    plain_cfg = copy.deepcopy(cfg)
    cfg.diagnostics = DiagnosticsConfig(synonym_clustering=False, num_sequences=16, probe_steps=8)
    plain_cfg.diagnostics = None
    plain_metrics = _run(plain_cfg, tmp_path/'plain')
    snapshot_metrics = _run(cfg, tmp_path/'snapshots')
    observed_history = snapshot_metrics.pop('history')
    plain_history = plain_metrics.pop('history')
    assert plain_metrics == snapshot_metrics
    for observed_row, plain_row in zip(observed_history, plain_history):
        assert observed_row.pop('diagnostics')['linear_probe']['fit_examples'] == 8
        observed_row.pop('diagnostic_config')
        assert observed_row == plain_row
    plain, observed = _load(tmp_path/'plain/last.pt'), _load(tmp_path/'snapshots/last.pt')
    for key in ['model', 'optimizer', 'predictor', 'loader_states']:
        _equal(plain[key], observed[key])
    _equal(plain['rng']['torch'], observed['rng']['torch'])
    assert not (tmp_path/'snapshots/diagnostic_snapshots').exists()
    assert 'best_model' not in observed


@pytest.mark.parametrize('step', [0, 2, 3, 4])
@pytest.mark.parametrize('auxiliary', [False, True])
def test_exact_resume_preserves_training_and_metrics(tmp_path, step, auxiliary):
    cfg = _cfg()
    cfg.train.checkpoint_every_updates = 1
    cfg.diagnostics = DiagnosticsConfig(synonym_clustering=False, num_sequences=16, probe_steps=3)
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


def test_resume_rejects_wrong_grammar(tmp_path):
    cfg = _cfg()
    _run(cfg, tmp_path/'run')
    bundle = _bundle(cfg)
    bundle.rules[1][0, 0, 0] = (bundle.rules[1][0, 0, 0]+1) % cfg.rhm.v
    with pytest.raises(ValueError, match='RHM rules'):
        train_model(cfg, LeafSequenceDataset(bundle.train.leaves), bundle.val,
                    rules=bundle.rules, resume_from=tmp_path/'run/last.pt', verbose=False)


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


def test_online_and_offline_diagnostics_agree(tmp_path):
    cfg = _cfg()
    cfg.diagnostics = DiagnosticsConfig(synonym_clustering=False, num_sequences=16, probe_steps=8)
    metrics = _run(cfg, tmp_path)
    offline = diagnose_checkpoint(tmp_path/'last.pt', device='cpu', metric='probe', num_sequences=16, probe_steps=8)
    assert metrics['history'][-1]['diagnostics'] == offline['diagnostics']
    assert metrics['history'][-1]['diagnostic_config'] == offline['diagnostic_config']


def test_offline_measurements_include_final_and_selected_checkpoints(tmp_path):
    cfg = _cfg()
    _run(cfg, tmp_path/'run')
    supporting = diagnose_trajectory(tmp_path/'run', tmp_path/'support', device='cpu',
                                    metric='clustering', num_sequences=16, steps=[0, 4, 7])
    assert [r['global_step'] for r in supporting['history']] == [0, 4, 7]
    assert all('linear_probe' not in r['diagnostics'] for r in supporting['history'])
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


@pytest.mark.parametrize('interval', [None, 1])
def test_default_checkpoint_policy_writes_measurements_only(tmp_path, interval):
    cfg = _cfg()
    cfg.train.save_checkpoints = ExperimentConfig().train.save_checkpoints
    cfg.train.checkpoint_every_updates = interval
    cfg.train.eval_at_start = False
    cfg.diagnostics = DiagnosticsConfig(synonym_clustering=False, num_sequences=16, probe_steps=3)
    metrics = _run(cfg, tmp_path)
    assert [r['global_step'] for r in metrics['history']] == [0, 2, 4, 6, 7]
    assert all('diagnostics' in r for r in metrics['history'])
    assert sorted(p.name for p in tmp_path.iterdir()) == ['config.json', 'metrics.json', 'rules.pt']


@pytest.mark.parametrize('interval', [None, 1])
def test_opt_in_checkpoints_do_not_duplicate_final_state(tmp_path, interval):
    cfg = _cfg()
    cfg.train.checkpoint_every_updates = interval
    _run(cfg, tmp_path)
    assert (tmp_path/'last.pt').exists()
    assert not (tmp_path/'checkpoints/step_00000007.pt').exists()
    assert len(list((tmp_path/'checkpoints').glob('*.pt'))) == (7 if interval else 0)
    assert set(training.saved_checkpoints(tmp_path)) == (set(range(8)) if interval else {7})


def test_failed_probe_keeps_previous_measurements_and_no_completion_marker(tmp_path, monkeypatch):
    import json
    cfg = _cfg()
    cfg.train.save_checkpoints = False
    cfg.diagnostics = DiagnosticsConfig(synonym_clustering=False, num_sequences=16, probe_steps=3)
    original = training.run_latent_diagnostics
    calls = 0
    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('probe failed')
        return original(*args, **kwargs)
    monkeypatch.setattr(training, 'run_latent_diagnostics', fail_second)
    with pytest.raises(RuntimeError, match='probe failed'):
        _run(cfg, tmp_path)
    assert not (tmp_path/'metrics.json').exists()
    assert [r['global_step'] for r in json.loads((tmp_path/'history.json').read_text())['history']] == [0]


def test_resume_can_disable_checkpoint_saving_and_restores_probe_history(tmp_path):
    cfg = _cfg()
    cfg.diagnostics = DiagnosticsConfig(synonym_clustering=False, num_sequences=16, probe_steps=3)
    full = _run(cfg, tmp_path/'full')
    cfg.train.save_checkpoints = False
    resumed = _run(cfg, tmp_path/'resumed', resume_from=tmp_path/'full/checkpoints/step_00000004.pt')
    assert full == resumed
    assert [p.name for p in (tmp_path/'resumed').rglob('*.pt')] == ['rules.pt']
