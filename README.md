# Learning to Predict Deeper: RHM × nanoGPT

This repository is an experimental scaffold for studying hierarchical rule
machines (RHM) with a small, standard causal Transformer. It generates fixed
RHM datasets, trains a nanoGPT-style language model with teacher-forced
next-token prediction, and records checkpoints and held-out cross-entropy.

The model is deliberately a causal Transformer rather than a reproduction of
the non-causal encoder architectures used by some related papers. The RHM
data generator and theoretical helpers remain available for experiment design,
but the training code has one objective: predict every next token in a finite
sequence.

## What is implemented

- RHM rule generation and deterministic train/validation/test datasets.
- A nanoGPT-style decoder with learned token and position embeddings,
  pre-normalized causal self-attention blocks, GELU MLPs, final layer norm,
  tied input/output embeddings, and AdamW parameter grouping.
- Standard causal next-token alignment: full `x` input, labels `x[:, 1:]`,
  and loss on logits `logits[:, :-1]`.
- Full-sequence causal forwards with mean and per-position next-token NLL.
- Paper-compatible final-position NLL, uniform/theory reference baselines, and
  total samples seen.
- Hidden-state hooks plus optional linear-probe, synonym-clustering,
  latent-replacement, shuffled-label, and untrained-backbone diagnostics.
- Self-contained checkpoints, validation-based model selection, fixed update
  budgets, deterministic seeds, fixed-exposure training-size sweeps, and an
  implementation-validation report.

The model implementation is in [`nanogpt/model.py`](nanogpt/model.py), and
the training seam is in [`training.py`](training.py).

## Setup

```bash
python -m pip install -r requirements.txt
```

Run the small smoke configuration:

```bash
python train.py \
  --config configs/smoke.json \
  --output-dir runs/smoke
```

Run the diagnostics smoke configuration:

```bash
python train.py \
  --config configs/diagnostics_smoke.json \
  --output-dir runs/diagnostics_smoke
python diagnose.py \
  --checkpoint runs/diagnostics_smoke/last.pt \
  --output runs/diagnostics_smoke/offline_diagnostics.json
python diagnose.py \
  --checkpoint runs/diagnostics_smoke/last.pt \
  --controls \
  --output runs/diagnostics_smoke/controls.json

PYTORCH_ENABLE_MPS_FALLBACK=0 python validate_implementation.py \
  --output runs/implementation_validation/validation.json
```

Run the next-token training-size sweep:

```bash
python sweep.py \
  --config configs/next_token_sweep.json \
  --output-dir runs/next_token_sweep

python sweep.py \
  --config configs/next_token_sweep_fixed_exposure.json \
  --output-dir runs/fixed_exposure_pilot

# Large MPS run; validation is throttled for small training pools.
PYTORCH_ENABLE_MPS_FALLBACK=0 python sweep.py \
  --config configs/next_token_sweep_large_mps.json \
  --output-dir runs/next_token_sweep_large_mps

# Focused developmental L=5 screen.
PYTORCH_ENABLE_MPS_FALLBACK=0 python sweep.py \
  --config configs/developmental_l5_screen.json \
  --output-dir runs/developmental_l5_screen
```

Plot completed sweep results:

```bash
python plot.py \
  --metrics runs/next_token_sweep/metrics.jsonl \
  --config configs/next_token_sweep.json \
  --output runs/next_token_sweep/curve.png

python plot.py \
  --metrics runs/next_token_sweep_large_mps/metrics.jsonl \
  --config configs/next_token_sweep_large_mps.json \
  --metric test_last_position_nll \
  --output runs/next_token_sweep_large_mps/last_token_nll.png
```

Use `--resume` with `sweep.py` to continue an interrupted sweep. Existing
results are checked against the saved sweep configuration before they are
reused.

For one completed L=5 run, diagnose every exact-step checkpoint and plot its
training-age trajectory:

```bash
python diagnose_trajectory.py \
  --run-dir runs/developmental_l5_screen/grammar_0/model_0/P_32768 \
  --output-dir runs/developmental_l5_screen/grammar_0/model_0/P_32768/trajectory \
  --device mps \
  --num-sequences 2048 \
  --probe-steps 500 \
  --controls

python plot_trajectory.py \
  --trajectory runs/developmental_l5_screen/grammar_0/model_0/P_32768/trajectory/trajectory.json \
  --metrics runs/developmental_l5_screen/grammar_0/model_0/P_32768/metrics.json \
  --output runs/developmental_l5_screen/grammar_0/model_0/P_32768/trajectory/trajectory.png
```

## Experimental hygiene

Datasets are generated once per grammar and reused across training-set sizes.
The validation set selects the best checkpoint; the test set is evaluated only
after that selection. The sweep stores one JSON object per run in
`metrics.jsonl`, plus the generated rule table and each run's checkpoints when
enabled.

When `samples_per_example` is present in a sweep configuration, it must be a
positive whole number. Every training-set size then receives that many complete
passes over its dataset, using the effective DataLoader batch size. Each result
records `global_step`, `total_samples_seen`, and `total_tokens_seen`, so reuse is
not confused with a sample-complexity effect.

For comparisons across training-set sizes, set `train.eval_every_updates` to
evaluate at a fixed optimizer-step cadence; it takes precedence over
`eval_every_epochs`. Set `train.eval_at_start` to record an explicit step-zero
baseline. Top-level `val_*` metrics describe the validation-selected model,
while `last_val_*` metrics describe the final training state.

Sweep snapshots record the input-config digest, code revision, worktree state,
and effective per-size training settings. Each sweep run also writes its exact
resolved experiment config beside `metrics.json`.

The current code is a foundation for adding latent prediction or auxiliary
losses later. Such additions should be explicit extensions to the causal
next-token baseline rather than silently changing the training target. Before
implementing that extension, compare the next baseline round with the
[pre-run expectations and go/no-go gate](documents/expectation.md).

Current implementation and experiment summaries are in
[`documents/VALIDATION_RESULTS.md`](documents/VALIDATION_RESULTS.md) and
[`documents/LARGE_SWEEP_RESULTS.md`](documents/LARGE_SWEEP_RESULTS.md).

## Representation diagnostics

Diagnostics are observers only: they run at evaluation points, use their own
probe parameters and local random streams, and do not affect the NTP loss,
optimizer trajectory, or validation checkpoint selection. Linear probes report
latent accessibility at every residual stream (embedding stream and each
post-block stream). Synonym clustering compares an on-grammar forced synonym
realization with an unrelated in-distribution example whose latent differs.
Variable sensitivity adds a matched intervention that changes the latent and
regenerates only its descendant subtree. Diagnostic controls include shuffled
latent labels and an untrained backbone. Probe reports include the theoretical
uniform chance level, an empirical majority-class baseline, and balanced
accuracy so finite-sample class imbalance is visible.

For abstraction level `r`, counted upward from leaves, the first completed
constituent is measured at position `s**r - 1`; `r=1` is the parent of visible
leaves. Therefore the baseline `A(s,j,r)` and `C(s,j,r)` values are explicitly
first-constituent diagnostics, not claims about every constituent position.
The generic comparison is an in-distribution control, so this implementation is
an adaptation for the causal RHM setting rather than a reproduction of a
particular non-causal representation-learning paper.

Every checkpoint stores the RHM grammar when available, along with model,
optimizer, RNG, sampler, history, and best-model state. Resume rejects a
supplied grammar that differs from the checkpoint grammar. Periodic snapshots
are written under `checkpoints/` when `checkpoint_every_evals` or the exact
update-based `checkpoint_every_updates` is configured. The latter is preferred
for training-age studies.

## Provenance

See [`documents/PROVENANCE.md`](documents/PROVENANCE.md) for the relationship
to Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), the local license
text, and the project-specific extensions.
