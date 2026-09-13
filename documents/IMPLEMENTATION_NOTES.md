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

`train_model` consumes training and validation datasets only. It records running
training losses, fixed-validation mean/per-position NLL, and exposure counts.
It neither rescans the training pool nor runs probes or touches test data.
The final model remains at the final update. `best_val_ce` and `best_step` are
summary statistics over validation history, not a second selected-model state.
Use `evaluate_snapshot.py` for explicit post-training evaluation of a saved
snapshot on validation or test. When selecting a model, choose the saved
validation-measured state with lowest validation NLL; test must not select it.

Update-based evaluation takes priority when configured. Epoch-based validation
remains available for the small data-size workflows. Fresh training pools use
the deterministic epoch seed schedule; validation/test pools remain fixed.
Exact continuation supports `num_workers=0`, avoiding sampler prefetch ambiguity.

## State files

Each run stores its actual grammar in `rules.pt` and resolved training settings
in `config.json`.

- `diagnostic_snapshots/step_XXXXXXXX.pt`: backbone, config, epoch, update, and a
  relative grammar-file reference with SHA-256 verification. No head, optimizer,
  RNG, or second model. Written at step zero, the requested update interval,
  and the final step.
- `checkpoints/step_XXXXXXXX.pt`: full continuation state at the configured
  update interval, including step zero. Contains model, head, optimizer,
  sampler, loader RNG, global RNG, running loss accumulators, history, and grammar.
- `last.pt`: the true final continuation state.

Artifact markers distinguish `diagnostic` from `continuation`. Resuming a
model-only snapshot fails explicitly. Old checkpoint/config formats are not
supported. Full checkpoints are self-contained; diagnostic snapshots travel
with their run's `rules.pt`. There is no `best.pt` or embedded `best_model`.
Files are replaced after successful serialization.

`train.py --resume-from PATH` continues a full state. Only the update/epoch
budget may differ; target switching is a future experimental workflow, not
ordinary resume. Sparse checkpoint restore and snapshot noninterference are
covered by CPU tests with dropout, resampling, and auxiliary heads. MPS replay
is checked with a numerical tolerance.

## Offline diagnostics and analysis

Probe settings are independent `DiagnosticsConfig` values, not training config.
Probes standardize each feature from the fit half and evaluate on the other
half. Seeds, learning rate, fitting steps, and actual sample count are saved
with each measurement. The full configured split is regenerated before slicing,
since sampling a differently sized tensor can produce different examples.

`diagnose.py` measures one snapshot. `diagnose_trajectory.py` defaults to
probe-only snapshots and accepts `--snapshot-dir`, `--steps`, or
`--every-updates`. Run C/Q or shuffled controls into a separate output directory
at selected ages. Do not mix sparsely available supporting metrics into a
dense probe trajectory.

`records.jsonl` is the canonical set of raw measurements. `trajectory.json` is
its derived complete view for analysis; it is removed when a replacement
measurement starts and regenerated only after completion. Training writes a
progress `history.json`; final `metrics.json` incorporates that history and
replaces the progress file.

The probes inspect the first completed constituent at each hierarchy level,
not every position. Accessibility is a descriptive held-out balanced-accuracy
threshold event. The primary rule requires two consecutive same-layer passes;
the earliest observer layer supplies onset. The observation interval is part
of this measurement definition. Clustering and Q remain supporting diagnostics.
Paired comparisons require identical training and diagnostic configurations,
evaluation steps, and snapshot grids. Best CE is secondary to matched-update
validation curves.
