# Stage 01: NTP developmental baseline

This stage asks whether ordinary causal next-token prediction develops a
layerwise, temporally ordered hierarchy of RHM latents. It must be completed
before adding the latent auxiliary loss.

This README contains the pre-run expectation and decision gate for the stage.
The paired evidence is in [RESULTS.md](RESULTS.md). Compare the completed
rerun with the criteria below before proceeding to any auxiliary objective.

## Protocol

- Binary RHM with `v=n=16`, `m=4`, `s=2`, and `L=5`.
- Ordinary causal NTP only; no latent auxiliary loss.
- Eight Transformer blocks, eight heads, width 256, batch size 256.
- The required confirmation rerun is fixed at `P=65,536` with the full 2 × 2
  grammar/model seed factorial in `configs/replication.json`.
- `configs/regime_search.json` is retained for optional fresh artifacts only;
  the three-point `P` search is not part of the required decision.
- Evaluation every 250 optimizer updates, including step zero.
- Exact checkpoint snapshots every 500 updates through step 5,000.
- Offline layer-by-level diagnostics for the embedding stream and all eight
  post-block streams, with shuffled-label and untrained-backbone controls.

The diagnostics report the first completed constituent at positions
`t=1, 3, 7, 15` for `H1` through `H4`. Layer `j=0` is the embedding stream;
`j=1,...,8` are post-block residual streams. A full acquisition claim must
use the same layer for accessibility, synonym invariance, and sensitivity.

## Pre-run expectation and decision gate

The ordinary causal next-token prediction (NTP) baseline has one specific job:
show that the model develops a temporally ordered hierarchy of internal
abstractions, with enough separation between transitions that changing the
auxiliary target could plausibly matter.

This section is the pre-run specification for the eventual Stage 01 report.
The report must use the same quantities and acquisition rule, and must state
`go`, `no-go`, or `rerun baseline` against these criteria. New criteria must be
agreed before a rerun, not introduced after inspecting its results.

### Expected training-age pattern

The exact step numbers are not predictions. The important expectation is a
reproducible ordering with meaningful time between transitions while held-out
NTP is still improving.

| training age | NTP | `H1` | `H2` | `H3` | `H4` |
|---|---|---|---|---|---|
| step 0 | baseline | baseline | baseline | baseline | baseline |
| early | improving | emerging | weak | weak | weak |
| early/mid | improving | strong | emerging | weak | weak |
| mid | improving | strong | strong | emerging | weak |
| later | improving | strong | strong | stronger | perhaps emerging |

The ideal checkpoint for a later auxiliary experiment has `H1` established,
`H2` beginning to form, `H3` largely absent, and substantial remaining NTP
headroom. Transition times such as `tau_1 ~= 1000`, `tau_2 ~= 3000`, and
`tau_3 ~= 7000` are useful only as an order-of-magnitude example; the numbers
themselves do not matter.

Layer order is weaker than level order. We do not require block 1 to encode
`H1`, block 2 to encode `H2`, and so on. The useful object is the evolution of
the layer-by-level heatmaps.

### What the run must show

#### Held-out NTP learning

Validation NLL must fall substantially from step zero and remain better than
the relevant uniform and grammar-aware controls. Per-position NLL should show
that learning is not confined to trivial local positions. Training loss alone
is insufficient.

#### Latent accessibility above controls

For each `H_r`, balanced accuracy of a frozen linear probe should rise clearly
above its step-zero value, the shuffled-label probe, and the empirical
majority-class baseline. The theoretical uniform chance value is useful
context, but step-zero accuracy need not equal it because token identity and
position can make information linearly accessible before training.

#### Synonym invariance at the same transition

When a different production rule realizes the same latent, the representation
should become more similar than when the corresponding latent changes. In the
repository's notation,

\[
C_{j,r}=1-\frac{d_{\mathrm{syn}}}{d_{\mathrm{non}}}
\]

should move from its initial baseline toward positive values at approximately
the same training ages at which `H_r` becomes accessible. A probe that rises
while clustering remains flat near zero is evidence for decodability, not yet
for the abstraction of interest. For a full acquisition claim, accessibility,
invariance, and sensitivity must all hold at the same residual-stream layer
`j`; values from different layers may not be combined. Same-layer A+C is the
primary developmental evidence. H1 is useful supporting context, but its probe
signal is confounded by local token identity and is not the core clock.

#### Sensitivity to changing the latent

Synonym invariance must not be explained by representation collapse. Changing
`H_r` should still produce a substantial representation change, measured by
the matched latent-replacement sensitivity control. Report the raw same-layer
values `d_syn`, `d_variable`, and `d_non`, the legacy ratio
`S=d_variable/d_non`, and the normalized contrast

\[
Q_{j,r}=\frac{d_{\mathrm{variable}}-d_{\mathrm{syn}}}
              {d_{\mathrm{non}}+\epsilon}.
\]

The old `S >= 1.005` threshold is retained as a secondary continuity check,
not as an independent acquisition hurdle. Positive `Q` or
`d_variable > d_syn` at the same layer supports the interpretation that latent
replacement matters more than a synonymous surface change. A borderline or
noisy contrast should be reported as such rather than converted into a new
hard threshold after seeing the results.

#### Ordered, replicated transitions

We do not require a mathematically perfect ordering for every seed. We expect
something qualitatively like

\[
\tau_1 < \tau_2 < \tau_3
\]

on the canonical run, with the `H1 -> H2` ordering surviving at least one
additional model seed and one additional grammar realization. `H3` emerging
and `H4` remaining late are supporting evidence; convincingly acquiring `H1`
and `H2` is the minimum useful result.

### Proceed to the latent auxiliary loss only if

- held-out NTP improves strongly and still has headroom after `H1` is learned;
- `H2` is convincingly acquired and the later `H3` transition is visible, with
  the H2-to-H3 ordering replicated across grammar/model seeds;
- acquisition times are visibly separated rather than all appearing at the
  first evaluation;
- each claimed transition is supported primarily by same-layer balanced
  accessibility plus positive synonym invariance; the raw sensitivity contrast
  shows no persistent evidence of collapse;
- the qualitative `H1 -> H2` ordering is not peculiar to one grammar/model
  seed pair; and
- there is a useful transition checkpoint where H2 is established, H3 is
  emerging, and NTP is still improving. H1 invariance is supporting context.

If these conditions hold, freeze the baseline protocol and implement the first
auxiliary comparison as separate from-scratch runs:

\[
\mathcal L=\mathcal L_{\mathrm{NTP}}+
\lambda\mathcal L_{\mathrm{next\ latent}}^{(j)},
\qquad
j\in\{\text{embedding},1,2,4,6,8\}.
\]

The first question is which fixed target minimizes time to acquire `H1`, `H2`,
and `H3`. Adaptive target switching comes only after that fixed-target
comparison produces a meaningful developmental clock.

### Rerun the baseline instead if

- probes become strong but synonym clustering stays near its initial baseline;
- all hierarchy levels rise together at the first evaluation;
- only `H1` develops within the available budget;
- the ordering changes wildly across grammar seeds; or
- NTP saturates before there is a window in which target choices could plausibly
  have different value.

These outcomes are informative baseline failures, not reasons to add the
auxiliary objective anyway. Adjust the dataset size, training budget,
evaluation cadence, or model capacity, then rerun the NTP baseline against this
same expectation. Persistent failure of the raw sensitivity contrast should be
reported as a qualification of the abstraction claim, not hidden by selecting
the largest legacy `S` value from another layer.

### Reporting requirements

The Stage 01 report must include, on the same training-age axis:

- held-out mean and per-position NTP NLL;
- layer-by-level balanced probe accuracy with shuffled-label, majority, and
  untrained-backbone controls;
- synonym-clustering scores and synonym/non-synonym distances;
- latent-replacement sensitivity;
- transition estimates or clearly marked qualitative transition windows; and
- the canonical configuration, grammar/model seeds, checkpoints, total samples
  seen, and device/runtime details.

The report must explicitly state **go**, **no-go**, or **rerun baseline** against
this section. Only a **go** decision authorizes the latent auxiliary-loss
implementation.

## Required rerun

Run from the repository root, on MPS when available:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python sweep_data_size.py \
  --config experiments/01_ntp_development/configs/replication.json \
  --output-dir runs/01_ntp_development/rerun
```

After the sweep finishes, diagnose every exact-step checkpoint for all four
confirmation runs:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python diagnose_trajectory.py \
  --run-dir runs/01_ntp_development/rerun/grammar_0/model_0/P_65536 \
  --output-dir runs/01_ntp_development/rerun/grammar_0/model_0/P_65536/trajectory \
  --controls

PYTORCH_ENABLE_MPS_FALLBACK=0 python diagnose_trajectory.py \
  --run-dir runs/01_ntp_development/rerun/grammar_0/model_1/P_65536 \
  --output-dir runs/01_ntp_development/rerun/grammar_0/model_1/P_65536/trajectory \
  --controls

PYTORCH_ENABLE_MPS_FALLBACK=0 python diagnose_trajectory.py \
  --run-dir runs/01_ntp_development/rerun/grammar_1/model_0/P_65536 \
  --output-dir runs/01_ntp_development/rerun/grammar_1/model_0/P_65536/trajectory \
  --controls

PYTORCH_ENABLE_MPS_FALLBACK=0 python diagnose_trajectory.py \
  --run-dir runs/01_ntp_development/rerun/grammar_1/model_1/P_65536 \
  --output-dir runs/01_ntp_development/rerun/grammar_1/model_1/P_65536/trajectory \
  --controls

python plot_trajectory.py \
  --trajectory runs/01_ntp_development/rerun/grammar_0/model_0/P_65536/trajectory/trajectory.json \
  --metrics runs/01_ntp_development/rerun/grammar_0/model_0/P_65536/metrics.json \
  --output runs/01_ntp_development/rerun/grammar_0/model_0/P_65536/trajectory/trajectory.png

python plot_trajectory.py \
  --trajectory runs/01_ntp_development/rerun/grammar_0/model_1/P_65536/trajectory/trajectory.json \
  --metrics runs/01_ntp_development/rerun/grammar_0/model_1/P_65536/metrics.json \
  --output runs/01_ntp_development/rerun/grammar_0/model_1/P_65536/trajectory/trajectory.png

python plot_trajectory.py \
  --trajectory runs/01_ntp_development/rerun/grammar_1/model_0/P_65536/trajectory/trajectory.json \
  --metrics runs/01_ntp_development/rerun/grammar_1/model_0/P_65536/metrics.json \
  --output runs/01_ntp_development/rerun/grammar_1/model_0/P_65536/trajectory/trajectory.png

python plot_trajectory.py \
  --trajectory runs/01_ntp_development/rerun/grammar_1/model_1/P_65536/trajectory/trajectory.json \
  --metrics runs/01_ntp_development/rerun/grammar_1/model_1/P_65536/metrics.json \
  --output runs/01_ntp_development/rerun/grammar_1/model_1/P_65536/trajectory/trajectory.png
```

The trajectory plot includes validation NLL by prediction position and the
layer-by-level accessibility, synonym-invariance, and sensitivity heatmaps.

## Decision gate

Do not implement the latent auxiliary loss after a promising screen alone.
First compare the full rerun with the expectation in this README and record
`go`, `no-go`, or `rerun baseline` in [RESULTS.md](RESULTS.md). Only an explicit
`go` authorizes the next implementation stage.
