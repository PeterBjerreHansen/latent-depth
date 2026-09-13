# Learning to Predict Deeper: RHM × nanoGPT

This repository studies whether the useful depth of auxiliary latent supervision
changes during causal next-token learning on a Random Hierarchy Model (RHM).
The current experiment calibrates stronger auxiliary weighting before later
shared-state continuation comparisons.

## Workflows

```text
RHM leaves → causal Transformer + optional fixed-target predictor
                  ↓                            ↓
         backbone snapshots          continuation checkpoints
                  ↓
         offline probes → paired accessibility and NTP curves
```

- `training.py` trains and records fixed-validation NLL. It does not run
  diagnostics, select a second model, or evaluate test data.
- `auxiliary.py` defines detached next-position residual targets and the predictor.
- `diagnose.py` / `diagnose_trajectory.py` inspect saved backbone states offline.
- `summarize_target_depth.py` applies the frozen measurement rule to paired arms.
- `plot_trajectory.py` plots accessibility and matched-update validation curves.
- `evaluate_snapshot.py` explicitly evaluates a chosen snapshot on validation/test.
- `sweep_target_depth.py` runs fixed-depth arms; `--arms` selects a subset.
- `sweep_data_size.py` retains small data-size and baseline replication workflows.

## Experiments

- [Stage 00: implementation validation](experiments/00_validation/README.md)
- [Stage 01: NTP developmental baseline](experiments/01_ntp_development/README.md)
- [Stage 02: stronger fixed-target screen](experiments/02_fixed_latent_targets/README.md)

Local run directories contain resolved configs, the exact grammar, measurements,
model-only snapshots, and sparse full continuation states. Large run artifacts
are ignored. Keep compact results and calibration records with each experiment.
Historical formats are not compatibility targets.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

See [implementation notes](documents/IMPLEMENTATION_NOTES.md),
[validation notes](documents/VALIDATION.md), the
[research proposal](documents/writeup_v2.md), and
[origin and licensing](documents/PROVENANCE.md).
