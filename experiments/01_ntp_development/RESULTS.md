# Stage 01 NTP developmental baseline results

**Status:** complete
**Decision:** **go** for the fixed-target auxiliary comparison

The confirmation rerun uses ordinary causal NTP with a fresh training pool at
the start of every epoch. Validation and test pools remain fixed. The table
below applies the current fixed balanced-accessibility threshold (`0.75`) and
two-checkpoint same-layer persistence directly to the raw trajectories.

## Required runs

Configuration: [`configs/ntp_l5_m4.json`](configs/ntp_l5_m4.json)
Output: `runs/ntp_l5_m4/`

Each arm used the fresh-pool schedule for 5,000 updates: `1,280,000` sequence
draws and `39,680,000` predicted tokens, with a per-epoch training pool of
`65,536` sequences.

| grammar seed | model seed | best validation CE | best step | H2 onset | H3 onset | decision note |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 1.443791 | 5,000 | 1,500 | 3,500 | |
| 0 | 1 | 1.444331 | 5,000 | 1,500 | 3,500 | |
| 1 | 0 | 1.443546 | 5,000 | 1,000 | 2,500 | |
| 1 | 1 | 1.439367 | 5,000 | 1,000 | 2,500 | |

The mean best validation CE is `1.442759 ± 0.001979`. Validation remains
close to the training cost at the final checkpoint, so the fresh-pool schedule
avoids the earlier finite-pool overfitting confound.

## Layerwise developmental result

The trajectory diagnostics contain all four runs and all 11 checkpoints. The
earliest observer layer for each observed event is listed below; the complete
layerwise curves remain in the raw trajectory files.

| run | H1 | H2 | H3 | H4 |
|---|---|---|---|---|
| grammar 0 / model 0 | step 0, `k=1` | step 1,500, `k=2` | step 3,500, `k=3` | not confirmed by step 5,000 |
| grammar 0 / model 1 | step 0, `k=1` | step 1,500, `k=2` | step 3,500, `k=3` | not confirmed by step 5,000 |
| grammar 1 / model 0 | step 0, `k=1` | step 1,000, `k=2` | step 2,500, `k=5` | not confirmed by step 5,000 |
| grammar 1 / model 1 | step 0, `k=1` | step 1,000, `k=3` | step 2,500, `k=5` | not confirmed by step 5,000 |

H2 precedes H3 in all four fresh runs, with H2 at 1,000--1,500 updates and
H3 at 2,500--3,500 updates. Observer layers vary across seeds, so the result
supports developmental timing more strongly than a one-block-per-level claim.
H1 is already above the primary threshold at step zero and remains supporting
context; H4 is not confirmed within the available budget.

At the first observed H2/H3 layers, final-checkpoint `Q` is positive in all
eight comparisons (approximately `0.54`--`0.97`). This supports the
interpretation that the accessibility pattern is not accompanied by persistent
collapse in the latent-replacement diagnostic.

The baseline meets the core gate: NTP improves with remaining headroom, the
H2-before-H3 ordering replicates across grammar and model seeds, and the raw
intervention contrast is positive. Stage 02 is authorized to compare fixed
target depths; adaptive target selection is not authorized by this result
alone.
