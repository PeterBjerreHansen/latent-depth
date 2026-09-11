# Learning to Predict Deeper: RHM × nanoGPT

This repository studies whether a small causal Transformer develops useful
hierarchical representations while learning to predict fixed finite sequences
from a Random Hierarchy Model (RHM). The exposed baseline objective is ordinary
causal next-token prediction (NTP). Stage 02 adds a separate fixed-target
auxiliary-loss experiment after the Stage-01 baseline gate.

## Architecture

```text
RHM rules and trees → leaf dataset → causal nanoGPT → NTP checkpoints
                                                        ↓
                                      per-position NLL and latent diagnostics
                                                        ↓
                                      fixed residual-target auxiliary arms
```

The reusable seams are deliberately small:

- [`rhm/`](rhm/) generates rules, trees, splits, and theory references.
- [`nanogpt/model.py`](nanogpt/model.py) implements the causal Transformer.
- [`training.py`](training.py) owns evaluation, checkpoints, and training.
- [`auxiliary.py`](auxiliary.py) defines the fixed residual-target predictor
  and cosine auxiliary loss used by Stage 02.
- [`diagnostics/`](diagnostics/) observes hidden states without changing NTP.
- [`sweep_data_size.py`](sweep_data_size.py) runs explicit training-size or
  replicate sweeps.
- [`sweep_target_depth.py`](sweep_target_depth.py) runs the Stage-02 fixed-depth
  target screen.

## Experiments

- [Stage 00: implementation validation](experiments/00_validation/README.md)
- [Stage 01: NTP developmental baseline](experiments/01_ntp_development/README.md)
- [Stage 02: fixed auxiliary target depth](experiments/02_fixed_latent_targets/README.md)

Run outputs belong under `runs/` and are intentionally not tracked. Each run
keeps the resolved config, metrics, and checkpoints needed to inspect or resume
that run. Historical runs are not part of the runnable core protocol.

## Setup and tests

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

For the complete implementation validation matrix, see
[`experiments/00_validation/README.md`](experiments/00_validation/README.md).

## Background

- [Implementation notes](documents/IMPLEMENTATION_NOTES.md)
- [Validation notes](documents/VALIDATION.md)
- [Project proposal](documents/writeup_v2.md)
- [Origin and licensing](documents/PROVENANCE.md)

The code is a causal-Transformer baseline and should not be described as a
reproduction of non-causal encoder or masked teacher/student experiments.
