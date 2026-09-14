# Implementation notes

## Training

The model receives the complete RHM leaf sequence and predicts positions
`1..T-1` from logits at `0..T-2`. Causal attention prevents access to future
tokens. NTP is invariant, with an optional detached next-latent cosine loss.
The predictor reads the final residual stream at `t`; its target is residual
stream `j` at `t+1`. Layer 0 is token-plus-position embeddings; subsequent
layers are post-block streams. Final LayerNorm output is excluded.

All arms construct the same backbone before privately initializing an auxiliary
predictor. A single optimizer builder includes the backbone and optional head.
Gradient clipping includes both. Grammar labels never enter the training loss.

`train_model` consumes a training dataset and the labelled `RHMSplit` for
validation. It records running training losses and fixed-validation NLL. When
`diagnostics` is configured, it also fits fresh probes at each validation event,
including initialization and the final update. It neither rescans the training
pool nor evaluates test data. `best_val_ce` and `best_step` are summary statistics,
not a second selected-model state.

Update-based evaluation takes priority when configured. Epoch-based validation
remains available for small data-size workflows. Fresh training pools use the
deterministic epoch seed schedule; validation remains fixed. Exact continuation
supports `num_workers=0`, avoiding sampler prefetch ambiguity.

## Validation observers

The optional experiment `diagnostics` section uses `DiagnosticsConfig`; `null`
means CE-only validation. Stage 01/02 configs enable linear probes with 16,384
sequences, 3,000 Adam steps, learning rate 0.01 and seed 12345. Smoke configs use
a small fitting budget. `synonym_clustering` is false in the standard workflow.
Probe settings and actual fitting/evaluation counts are stored with each record.

The observer uses the exact validation split already constructed by the caller.
Feature extraction has no gradients; probe fitting has its own optimizer and
uses detached features. Probes are initialized afresh each time. Features, fitted
weights and probe optimizer state are discarded after measurement. Model mode
and Python/NumPy/CPU/CUDA/MPS RNG states are restored, including on failure.
The model's optimizer, auxiliary head and data stream are not observer inputs.

Each feature is standardized using the fitting half only. Evaluation uses the
other half. The first completed constituent at each hierarchy level is measured,
not every position. Accessibility remains the fixed held-out balanced-accuracy
threshold with two consecutive same-layer passes. The validation interval is
part of that measurement definition. C/Q remain optional supporting diagnostics.

## Measurements and state files

Each run stores the exact small grammar in `rules.pt` and the resolved settings
in `config.json`. `history.json` contains a `history` array during training;
`metrics.json` incorporates it when the final validation succeeds. Each row has
`global_step`, exposure counts, validation CE and, when enabled, `diagnostics`
and `diagnostic_config`. JSON files are replaced atomically after serialization.
An interrupted run retains completed measurements but has no completion marker.

The normal workflow has no separate primary `records.jsonl` or `trajectory.json`.
Analysis and plots read the validation history directly from `metrics.json`.
Paired comparisons require matching training settings, diagnostic settings and
observation grids. Historical result bundles retain their original formats.

`train.save_checkpoints` defaults to false and gates all model-state writes,
including `last.pt`, regardless of any interval. When true:

- `checkpoint_every_updates: null` writes only the final `last.pt`.
- A positive interval writes full `checkpoints/step_XXXXXXXX.pt` states at zero
  and periodic updates, plus final `last.pt`. The final state is not duplicated
  under the periodic filename.

There is no diagnostic snapshot writer or `diagnostic_snapshot_every_updates`
setting. Full checkpoints contain backbone, auxiliary head, optimizer, sampler,
loader/global RNG, running losses, validation history and grammar. They are
self-contained and replaced only after serialization succeeds. No probe weights
or features are saved.

`train.py --resume-from PATH` restores full state and measurement history. Only
the update/epoch budget and checkpoint saving policy may differ. A resumed run does not repeat its checkpoint's
validation record. Without a saved checkpoint, completed measurements survive a
failure but training must restart. Use a new directory when restarting from zero.

## Optional offline inspection

`diagnose.py` and `evaluate_snapshot.py` inspect explicitly saved models, including
existing model-only research snapshots. Their read path ignores the retired saving
schedule; new configs and training do not accept it. Test evaluation is explicit
and requires a saved state.

`diagnose_trajectory.py` is optional for remeasurement or supporting diagnostics.
It selects periodic continuation checkpoints and `last.pt`; `--steps` or
`--every-updates` restricts selection. `--snapshot-dir` explicitly selects an
existing directory of named states. Its standalone output has the same `history`
record structure but is separate from primary training measurements.

Offline invariance controls use selected continuation states from `reference_runs`.
Enable saving before training any run whose states will be needed for controls,
probe recalibration, test evaluation, or shared-state continuation experiments.
