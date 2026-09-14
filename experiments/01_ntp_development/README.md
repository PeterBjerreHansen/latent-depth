# Stage 01: NTP developmental baseline

Stage 01 asks whether ordinary causal next-token prediction develops a
temporally ordered hierarchy of RHM latents. It is the baseline gate for the
later fixed residual-target experiments.

## Protocol

- Binary RHM with `v=n=16`, `m=4`, `s=2`, and `L=5`.
- Ordinary causal NTP only; no latent auxiliary loss.
- Eight Transformer blocks, eight heads, width 256, batch size 256.
- Per-epoch training pool `P=65,536`, fixed validation/test pools, and the
  full 2 × 2 grammar/model seed factorial in `configs/ntp_l5_m4.json`.
- Evaluation every 250 optimizer updates, including step zero.
- Validation probes every 500 updates through step 5,000; no checkpoints by default.
- Offline probes on the embedding stream and all eight post-block streams.
  Shuffled-label controls are optional and do not define acquisition.

The diagnostics inspect the first completed constituent at `H1`--`H4`, at
positions `t=1, 3, 7, 15`. Observer layer `k=0` is the embedding stream and
`k=1,...,8` are post-block residual streams. The latent labels are never used
in the training objective.

## Fixed analysis rule

The primary developmental measurement is balanced probe accuracy at the fixed
threshold in
[`acquisition_rule.json`](acquisition_rule.json):
`0.75`. For each observer layer, the first checkpoint at or above the
threshold is an onset candidate. It is an observed accessibility event only
when the same layer remains at or above the threshold at the next checkpoint.
The earliest observed layer gives the level's `tau_accessibility`; every
layerwise event is retained.

The analysis also reports balanced-probe milestones at `0.50`, `0.75`, and
`0.90`, using the same two-checkpoint persistence rule, plus
synonym-clustering, intervention distances, and `Q`. These thresholds show
whether the timing pattern depends on the chosen accuracy level. Clustering
and `Q` are not additional acquisition hurdles, and values from different
observer layers are not merged into one event. If no same-layer pair is
observed, report “not confirmed by step T”; do not infer a numeric onset or
difference from that censoring.

## Pre-run expectation and decision gate

The baseline should show useful held-out NTP improvement, a reproducible
ordering in which H2 becomes accessible before H3, and enough remaining NTP
progress for a target-depth intervention to matter. A clean one-block-per-level
mapping is not required; level timing is the primary question and layer timing
is descriptive.

Supporting evidence includes positive synonym-invariance scores and a positive
latent-replacement contrast `Q`. A probe rise without a corresponding
supporting representation signal is still a useful accessibility result, but
should not be described as proof of a complete abstraction.

After this baseline is accepted, proceed to the fixed-target screen when the
H2-before-H3 ordering is visible across the canonical runs, NTP has not
saturated before the transition window, and the intervention diagnostics do
not show persistent collapse. Record **go**, **no-go**, or **rerun baseline** in
[RESULTS.md](RESULTS.md). These are
qualitative pre-registered criteria; do not introduce a new hard criterion
after looking at the results.

## Required rerun

Run from the repository root, on MPS when available:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python sweep_data_size.py \
  --config experiments/01_ntp_development/configs/ntp_l5_m4.json \
  --output-dir experiments/01_ntp_development/runs/ntp_l5_m4
```

Each run records probe scores during validation in `metrics.json`. Plot a run with:

```bash
.venv/bin/python plot_trajectory.py --metrics RUN/metrics.json --output RUN/trajectory.png
```

For Stage 01, apply the committed local `acquisition_rule.json` to the
same validation history with the internal analysis helper. Supporting offline
controls require explicitly saved states.

## Reporting requirements

The report must include:

- fixed-validation mean and per-position NTP NLL;
- balanced accessibility curves and fixed-threshold onset/confirmation;
- the persistent 50/75/90% probe milestones;
- layerwise clustering, intervention distances, and `Q`;
- grammar/model seeds, validation schedule and exposure accounting;
- a clear **go**, **no-go**, or **rerun baseline** decision.

The results report should distinguish observed events from levels not confirmed
by the final checkpoint. H1 is useful context because local token information
can make it accessible early; H2 and H3 provide the main developmental gate.

The recorded Stage-01 results predate online validation probes. New runs use
the current configuration and store measurements without model checkpoints.
Historical evidence remains unchanged.
