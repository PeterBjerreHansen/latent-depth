# Validation

## Static checks

The migration to the nanoGPT model should be checked with:

```bash
python -m compileall -q config.py training.py auxiliary.py train.py sweep_data_size.py sweep_target_depth.py plot.py diagnose.py diagnose_trajectory.py summarize_trajectory.py summarize_target_depth.py plot_trajectory.py validate_implementation.py nanogpt rhm diagnostics tests
python -m pytest -q
```

The tests cover causal attention, tied embeddings, hidden-state shapes,
full-sequence next-token indexing, rejection of the removed objective,
per-position and final-position NLL reporting, RHM latent indexing, independent
tree expansion, forced-different synonym and latent-replacement interventions,
synthetic probe recovery, shuffled-label controls, clustering endpoints,
checkpoint round trips, selected-vs-final validation metrics, exact mid-epoch
resume, update-based step-zero evaluation, grammar validation on resume,
complete fixed-exposure budgeting, fit-half probe standardization, token
accounting, sweep resume configuration, exact update-based checkpoint boundaries,
empirical probe baselines, online vs. offline diagnostics parity, fixed-target
auxiliary alignment and stop-gradient behavior, auxiliary checkpoint resume,
and an end-to-end target-depth sweep/resume smoke test. Synthetic acquisition
analysis tests cover same-layer persistence, explicit not-confirmed horizons,
fixed probe milestones, optional controls, Q derivation, paired times with
unavailable censored differences, partial screens, and refusal of mismatched
sweep schedules. Training checkpoints
also carry the optimizer, CPU, CUDA, MPS, grammar, predictor, and
shuffled-loader state needed for continuation. Strict
deterministic mode is available for experiments that require kernel-level
reproducibility.

The executable matrix is:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python validate_implementation.py \
  --output experiments/00_validation/runs/implementation_validation/validation.json
```

It writes independent data/objective checks, CPU replay results, and a
CPU/MPS numerical comparison. Cross-device equality is tolerance-based; exact
replay is guaranteed only for the tested CPU path.

## Runtime note

The repository requires a working PyTorch installation. If test collection
fails before importing the project because the local PyTorch package is absent
or incomplete, repair the environment first; that is an environment failure,
not evidence about the model implementation.

## Reproducibility

Training seeds the Python, NumPy, and PyTorch random generators. RHM rules and
dataset splits are persisted in run directories, and checkpoints include model,
optimizer, configuration, grammar, metrics, sampler state, and RNG state.
