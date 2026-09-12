# Stage 01 NTP developmental baseline results

**Status:** complete
**Decision:** **go** for the fixed-target auxiliary comparison

The current rerun uses ordinary causal NTP with a fresh training pool at the
start of every epoch. Validation and test pools remain fixed. Acquisition
times below were regenerated from the raw trajectories with the frozen rule in
[`../02_fixed_latent_targets/acquisition_rule.json`](../02_fixed_latent_targets/acquisition_rule.json).

## Required runs

Configuration: [`configs/replication.json`](configs/replication.json)

Output: `runs/replication_l5_5000_updates/`

| grammar seed | model seed | best validation CE | best step | H2 A+C onset | H3 A+C onset | decision note |
|---:|---:|---:|---:|---|---|---|
| 0 | 0 | 1.443791 | 5,000 | 1,000 | 2,500 | |
| 0 | 1 | 1.444331 | 5,000 | 1,000 | 2,500 | |
| 1 | 0 | 1.443546 | 5,000 | 500 | 2,000 | |
| 1 | 1 | 1.439367 | 5,000 | 500 | 2,000 | |

The mean best validation CE is `1.442759 ± 0.001979`. In contrast to the
deleted fixed-pool runs, validation is still improving at the final checkpoint
and remains close to training CE. The fresh-pool schedule therefore removes
the earlier finite-pool overfitting confound.

## Layerwise developmental result

The trajectory diagnostics completed for all four runs and all 11 checkpoints.
The first two-checkpoint same-layer A+C windows are shown below. `k` denotes
the observer layer; the listed layer ranges are the layers that first satisfy
the rule at that onset step.

| run | H1 | H2 | H3 | H4 |
|---|---|---|---|---|
| grammar 0 / model 0 | step 500, `k=3--8` | step 1,000, `k=7--8` | step 2,500, `k=3--8` | step 4,500, `k=5--8` |
| grammar 0 / model 1 | step 500, `k=2--8` | step 1,000, `k=4--8` | step 2,500, `k=3--8` | step 4,500, `k=6--8` |
| grammar 1 / model 0 | not acquired | step 500, `k=3--8` | step 2,000, `k=3--8` | step 4,000, `k=8` |
| grammar 1 / model 1 | not acquired | step 500, `k=7--8` | step 2,000, `k=5--8` | step 4,500, `k=6--8` |

Thus H2 precedes H3 in all four fresh runs, with H2 at 500--1,000 updates and
H3 at 2,000--2,500 updates. The earliest observer layer is not perfectly
stable under the frozen rule, so the result supports a developmental timing
ordering more strongly than a one-block-per-level claim. H1 is censored in the
grammar-1 runs, while H4 is observed late at 4,000--4,500 updates.

At the first robust layers, the normalized intervention contrast `Q` is
positive for all eight H2/H3 comparisons (`+0.055` to `+0.135`), so the onset
pattern is not explained by persistent representation collapse.

The baseline meets the core pre-run gate: NTP improves with remaining headroom,
the H2-before-H3 ordering replicates across grammar and model seeds, and the
raw intervention contrast is positive. The observer-onset layers vary across
seeds, so that layerwise qualification should be carried into the Stage-02
comparison. Stage 02 is authorized to compare fixed target depths; adaptive
target selection is not authorized by this result alone.

The complete layerwise probe, invariance, and intervention trajectories are
stored with each run, including the shuffled-label and untrained-backbone
controls. See [README.md](README.md) for the pre-run expectation and decision
criteria.
