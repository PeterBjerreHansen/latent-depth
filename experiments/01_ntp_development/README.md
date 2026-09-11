# Stage 01: NTP developmental baseline

This stage asks whether ordinary causal next-token prediction develops a
layerwise, temporally ordered hierarchy of RHM latents. It must be completed
before adding the latent auxiliary loss.

The canonical decision gate is
[`documents/npt_l5_expectation.md`](../../documents/npt_l5_expectation.md). The current
evidence and the reason for the planned rerun are in
[RESULTS.md](RESULTS.md). Compare the completed rerun with the expectation
document before proceeding to any auxiliary objective.

## Protocol

- Binary RHM with `v=n=16`, `m=4`, `s=2`, and `L=5`.
- Ordinary causal NTP only; no latent auxiliary loss.
- Eight Transformer blocks, eight heads, width 256, batch size 256.
- Training sizes `P ∈ {16,384, 32,768, 65,536}` for the regime search.
- Confirmation rerun at `P=65,536` with the full 2 × 2 grammar/model seed
  factorial in `configs/replication.json`.
- Evaluation every 250 optimizer updates, including step zero.
- Exact checkpoint snapshots every 500 updates through step 5,000.
- Offline layer-by-level diagnostics for the embedding stream and all eight
  post-block streams, with shuffled-label and untrained-backbone controls.

The diagnostics report the first completed constituent at positions
`t=1, 3, 7, 15` for `H1` through `H4`. Layer `j=0` is the embedding stream;
`j=1,...,8` are post-block residual streams. A full acquisition claim must
use the same layer for accessibility, synonym invariance, and sensitivity.

## Rerun

Run from the repository root, on MPS when available:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python sweep_data_size.py \
  --config experiments/01_ntp_development/configs/regime_search.json \
  --output-dir runs/01_ntp_development/regime_search

PYTORCH_ENABLE_MPS_FALLBACK=0 python sweep_data_size.py \
  --config experiments/01_ntp_development/configs/replication.json \
  --output-dir runs/01_ntp_development/rerun

python plot.py \
  --metrics runs/01_ntp_development/regime_search/metrics.jsonl \
  --config experiments/01_ntp_development/configs/regime_search.json \
  --metric test_last_position_nll \
  --output runs/01_ntp_development/regime_search/last_token_nll.png
```

After the runs finish, diagnose every exact-step checkpoint for a selected
replicate:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python diagnose_trajectory.py \
  --run-dir runs/01_ntp_development/rerun/grammar_0/model_0/P_65536 \
  --output-dir runs/01_ntp_development/rerun/grammar_0/model_0/P_65536/trajectory \
  --controls

python plot_trajectory.py \
  --trajectory runs/01_ntp_development/rerun/grammar_0/model_0/P_65536/trajectory/trajectory.json \
  --metrics runs/01_ntp_development/rerun/grammar_0/model_0/P_65536/metrics.json \
  --output runs/01_ntp_development/rerun/grammar_0/model_0/P_65536/trajectory/trajectory.png
```

The trajectory plot includes validation NLL by prediction position and the
layer-by-level accessibility, synonym-invariance, and sensitivity heatmaps.

## Decision gate

Do not implement the latent auxiliary loss after a promising screen alone.
First compare the full rerun with [the NTP L5 expectation](../../documents/npt_l5_expectation.md)
and record `go`, `no-go`, or `rerun baseline` in the stage results. Only an
explicit `go` authorizes the next implementation stage.
