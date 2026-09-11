# Developmental L5 vanilla-NTP baseline

**Status:** screen, minimal replication, and full 2 × 2 confirmation rerun complete
**Decision:** go for the first fixed-target latent auxiliary-loss comparison; H4 is not acquired and the legacy `S` ratio remains a secondary continuity check
This is the Stage 01 report paired with the expectation in the stage README.
The full factorial results and decision below are evaluated against that
pre-run specification.

This report covers the corrected implementation after the checkpoint-metric,
probe-standardization, and exposure-budget fixes. The summary below retains the
scientific conclusions from the completed screen and partial replication. Their
large raw run directories were intentionally removed during cleanup; the next
rerun is defined by the commands in the stage README and writes to the new
stage-specific `runs/01_ntp_development/` paths.

## Protocol

The screen used ordinary causal next-token prediction only:

- regime search configuration: [`configs/regime_search.json`](configs/regime_search.json);
- replication configuration: [`configs/replication.json`](configs/replication.json).

- `v=n=16`, `m=4`, `s=2`, `L=5`, grammar seed `0`;
- an 8-layer, 8-head, width-256 nanoGPT-style causal Transformer;
- `P ∈ {16,384, 32,768, 65,536}`;
- 5,000 optimizer updates, batch size 256, MPS, and evaluations every 250 updates;
- exact self-contained checkpoints at steps `0, 500, ..., 5,000`;
- 2,048 held-out diagnostic sequences, 500 probe optimization steps;
- shuffled-label and untrained-backbone controls at every checkpoint.

The non-root latent completion positions are `t=1,3,7,15` for `r=1,2,3,4`.
Probe features are standardized using fit-half statistics only.

## NTP learning

| `P` | best validation CE | best step | final validation CE | final-position NLL at best |
|---:|---:|---:|---:|---:|
| 16,384 | 1.6802 | 1,250 | 4.3111 | 0.4535 |
| 32,768 | 1.5450 | 2,250 | 2.1752 | 0.1846 |
| 65,536 | 1.4706 | 4,250 | 1.4806 | 0.1099 |

The smaller runs overfit substantially after their best checkpoints. The
largest run continues to improve through almost the full budget and is the
most useful regime for developmental analysis.

## Screen diagnostics

The table reports the best residual-stream value at selected checkpoints.
`A` is balanced probe accuracy, `C` is the synonym-clustering score, and `S`
is latent-replacement sensitivity. Each metric is maximized independently in
this compact table, so the three values in one cell need not come from the
same layer. The layerwise analysis below is the primary evidence for where a
representation appears. The shuffled-label controls are near the uniform
chance level (`0.0625`). The untrained control has high H1 accuracy (`1.0`)
because the lowest-level label is already exposed by local token identity;
therefore H1 probe accuracy alone is not evidence of learned hierarchy.

| `P` | step | H1 `A/C/S` | H2 `A/C/S` | H3 `A/C/S` | H4 `A/C/S` |
|---:|---:|---:|---:|---:|---:|
| 16,384 | 500 | 1.000 / +.889 / 1.005 | .721 / +.025 / 1.001 | .197 / +.002 / 1.007 | .088 / +.011 / 1.015 |
| 16,384 | 1,500 | 1.000 / +.901 / 1.010 | .872 / +.098 / 1.000 | .245 / +.006 / 1.008 | .100 / +.010 / 1.016 |
| 32,768 | 1,500 | 1.000 / +.965 / 1.021 | .978 / +.267 / 1.028 | .340 / +.016 / 1.015 | .147 / +.010 / 1.006 |
| 32,768 | 2,500 | 1.000 / +.945 / 1.047 | 1.000 / +.550 / 1.021 | .425 / +.028 / 1.008 | .140 / +.010 / 1.007 |
| 65,536 | 1,500 | 1.000 / +.971 / 1.032 | .980 / +.463 / 1.004 | .378 / +.023 / 1.015 | .140 / +.010 / 1.008 |
| 65,536 | 2,500 | 1.000 / +.976 / 1.035 | 1.000 / +.729 / 1.019 | .534 / +.049 / 1.012 | .169 / +.010 / 1.001 |
| 65,536 | 3,000 | 1.000 / +.978 / 1.056 | 1.000 / +.775 / 1.045 | .660 / +.160 / 1.008 | .225 / +.010 / 0.998 |
| 65,536 | 4,000 | 1.000 / +.970 / 1.060 | 1.000 / +.743 / 1.040 | .872 / +.308 / 1.008 | .305 / +.016 / 0.999 |

### Layerwise developmental pattern

The trajectory files retain one value for every residual stream. The layer
convention is `j=0` for the embedding stream and `j=1,...,8` for the
post-Transformer-block streams; the final layer norm is excluded. For every
level and checkpoint, the report can therefore ask two separate questions:

1. When does a probe first decode the latent at each layer?
2. When does that decoded signal also become invariant to a production-rule
   change, while remaining sensitive to a latent replacement?

Those questions should not be collapsed into one `best layer` number. A probe
can decode a label from surface or positional information, and different
metrics can peak at different layers.

The `P=65,536` screen gives a clear layer-and-time pattern:

| signal | first useful location | subsequent layer pattern |
|---|---|---|
| H1 probe | present in the embedding stream and all trained streams at step 0 | confounded by local token identity; H1 probe accuracy is not a learned-hierarchy test |
| H2 probe | strong by step 500 at approximately `j=2` and deeper | by steps 1,500--2,000, accuracy is near one from `j=2` onward and invariance is strong across `j=2,...,8` |
| H3 probe | weakly above the untrained control around step 1,000 at approximately `j=3` | at step 3,000, accuracy is strong from `j=3` onward; by steps 3,500--4,500 the signal is strong across `j=3,...,8` |
| H4 probe | some late probe accessibility appears around `j=3`--`j=4` | clustering remains near zero, so this is not evidence for an acquired H4 abstraction |

Representative balanced-accuracy vectors, ordered by `j=0,...,8`, are:

| step | H2 | H3 |
|---:|---|---|
| 1,500 | `.156, .532, .970, .975, .980, .975, .974, .975, .979` | `.087, .147, .320, .365, .360, .374, .378, .369, .367` |
| 3,000 | `.156, .627, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000` | `.087, .144, .377, .637, .647, .660, .657, .648, .649` |
| 4,000 | `.156, .637, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000` | `.087, .151, .302, .866, .865, .866, .869, .872, .871` |

The corresponding synonym scores provide the stronger abstraction check. For
H2, `C` is already `0.410`--`0.463` across `j=2,...,8` at step 1,500. For
H3, `C` is at most `0.049` at step 2,500, then rises to `0.132`--`0.160`
across `j=3,...,8` at step 3,000 and to roughly `0.30`--`0.33` by steps
4,000--4,500. Thus the useful interpretation is that H2 becomes invariant
earlier and at a shallower residual stream than H3. The H3 signal then extends
through the deeper streams.

This is not a strict one-block-per-level staircase. The metrics are distributed
over residual streams, and their maxima can occur at different layers. In
particular, probe accessibility is observational: it does not establish that
the model computes H2 in block 2 or H3 in block 3, nor that either latent is
used by the next-token head. The intervention and NTP curves are needed to
support the stronger interpretation.

The screen shows a useful ordered pattern in the largest-data regime:

1. H1 is available immediately, but its probe signal is confounded by surface
   information. Its positive and persistent synonym score is the more useful
   result.
2. H2 becomes strongly accessible and invariant around steps 1,500--2,000.
3. H3 becomes clearly accessible and invariant around steps 3,000--4,000.
4. H4 does not become convincingly accessible within 5,000 updates.

This is evidence for an H1-to-H2-to-H3 developmental ordering, not evidence
that the model has learned a complete four-level abstract hierarchy.

The previous raw trajectory plots were exploratory artifacts and were removed
with the old run directories. Fresh plots from the full rerun are stored with
each trajectory under `runs/01_ntp_development/rerun/`.

## Pre-specified acquisition rule

Before evaluating the replication, call a level acquired only if all of the
following hold at the same or adjacent checkpoints and in at least two
consecutive checkpoints:

1. balanced probe accuracy at some residual-stream layer exceeds the
   shuffled-label and majority baselines by at least 0.10 absolute; for H2--H4
   it must also exceed the untrained-backbone control by at least 0.10 when
   that control is below 0.5;
2. `C >= 0.10` and is at least 0.05 above the step-zero value; and
3. For the historical screen, `S >= 1.005` was also recorded as a continuity
   check. The confirmation rerun treats this as secondary and reports the raw
   same-layer `d_syn`, `d_variable`, `d_non`, and `Q` contrast instead.

H1 is exempt from the untrained-accuracy comparison because its untrained
control is already perfect from token exposure. H1 is supporting context, not
the central developmental gate; the primary clock is the replicated H2-to-H3
same-layer A+C ordering.

These thresholds are operational screening criteria, not statistical
significance claims. A confirmatory study should add bootstrap confidence
intervals and retain the full layer-by-level curves rather than selecting only
the maximum layer.

## Criterion status before the full rerun

The screen was promising enough to justify the minimal replication. The
replication was evaluated under the rule above, with the layerwise distinction
made explicit below. Its outcome is not a full go for the auxiliary objective.

## Earlier two-run replication

The earlier two-run replication at `P=65,536` closely matches the screen. The
planned rerun configuration now expands this to the full 2 × 2 grammar/model
seed factorial:
[`configs/replication.json`](configs/replication.json).

| grammar seed | model seed | best validation CE | best step | H2 `A/C` onset | H3 `A/C` onset |
|---:|---:|---:|---:|---|---|
| 0 | 1 | 1.4693 | 4,250 | 1,500--2,000 | 2,500--3,500 |
| 1 | 0 | 1.4685 | 4,250 | 1,500--2,000 | 2,000--3,000 |

Both runs show persistent H1 invariance, strong H2 `A/C` accessibility, later
H3 `A/C` accessibility and invariance, and no convincing H4 acquisition by
step 5,000. For example, at step 3,000 the maximum-layer diagnostics are:

| run | H1 `A/C/S` | H2 `A/C/S` | H3 `A/C/S` | H4 `A/C/S` |
|---|---:|---:|---:|---:|
| grammar 0, model 1 | 1.000 / +.972 / 1.067 | 1.000 / +.795 / 1.048 | .702 / +.250 / 1.010 | .237 / +.011 / .999 |
| grammar 1, model 0 | 1.000 / +.973 / 1.029 | 1.000 / +.853 / 1.056 | .896 / +.391 / 1.002 | .310 / +.021 / 1.013 |

The layerwise replication agrees with the screen. At step 3,000, both runs
first show a strong combined H2 `A/C` signal at `j=2` and a strong combined
H3 `A/C` signal at `j=3`; the signal continues through the deeper streams.
These are `A/C` onset locations, not full `A/C/S` acquisitions, because the
sensitivity ratio is a separate check. The synonym scores at those first
useful layers are:

| run | H2 at `j=2`: `A/C` | H2 at `j=8`: `A/C` | H3 at `j=3`: `A/C` | H3 at `j=8`: `A/C` |
|---|---:|---:|---:|---:|
| grammar 0, model 1 | `1.000 / .752` | `1.000 / .795` | `.620 / .152` | `.692 / .250` |
| grammar 1, model 0 | `1.000 / .839` | `1.000 / .853` | `.895 / .391` | `.888 / .376` |

At step 1,500, H2 is already invariant at `j=2` in both replications, while
H3 clustering remains near zero across the layers. By step 3,000, H3
invariance has appeared at `j=3` and deeper in both runs. This is the useful
depth result: the first robust H3 `A/C` stream is one Transformer block deeper
than the first robust H2 `A/C` stream, and it arrives later in training. It is
a replicated onset pattern, not merely a difference between the independently
selected maximum layers.

The legacy sensitivity check qualified that conclusion. At step 3,000, H2
passed the `S >= 1.005` threshold at some deeper layers in both runs, but H3
did not pass it at the first `A/C` layer (`j=3`) in either run. The
grammar-0/model-1 run passed the old full same-layer rule only at a late `j=8`
window around steps 4,500--5,000; the grammar-1/model-0 run had no H3 layer
satisfying all three thresholds. The confirmation rerun should retain `S` for
continuity but report the raw same-layer distances and `Q` rather than letting
the legacy ratio act as an independent acquisition hurdle.

The H1-to-H2-to-H3 order therefore survives both a model-seed change and a
grammar change. The H3 transition occurs while validation NTP is still
improving, leaving a useful window for a future intervention.

## Decision before the full rerun

The current evidence supports a **rerun baseline** decision, not a go for the
auxiliary objective. The narrow positive result is that vanilla causal NTP
reliably produces a layerwise `A/C` pattern in which H2 appears around `j=2`
before H3 appears around `j=3`, with H3 arriving later in training. The full
claim is not yet established because the same-layer sensitivity criterion for
H3 does not replicate across the two confirmation runs. H4 is also not
acquired.

The rerun should retain the fixed `P=65,536` regime and exact checkpoint
trajectory, but make the layerwise report primary across all four runs. For
every checkpoint and level, retain the complete vectors for `A`, `C`, `S`, and
the raw intervention distances, then report separately:

- the first layer and step for probe-only accessibility;
- the first layer and step for `A+C` evidence; and
- the same-layer `d_variable-d_syn` contrast and normalized `Q`, with the
  legacy `S` threshold shown for continuity.

Do not implement the auxiliary loss until the rerun either satisfies the
pre-specified gate or the gate is deliberately revised before looking at the
new results. The auxiliary objective had not been implemented at that point.

## Full 2 × 2 confirmation rerun

The required confirmation rerun was executed from
[`configs/replication.json`](configs/replication.json) with ordinary causal
NTP only. It used `P=65,536`, four grammar/model seed combinations, 5,000
updates, batch size 256, evaluation every 250 updates, and exact checkpoints
at steps `0, 500, ..., 5,000`. Training ran on MPS with deterministic mode
enabled but non-strict backend behavior. No latent auxiliary loss was used.

Offline diagnostics then evaluated all 11 checkpoints for all four runs on the
validation split, using 1,024 sequences, 300 probe steps, all nine residual
streams (`j=0` embedding and `j=1,...,8` post-block), and both requested probe
controls. The raw artifacts and four five-panel trajectory plots are under
`runs/01_ntp_development/rerun/`.

### NTP results

| grammar seed | model seed | best validation CE | best step | final validation CE | test CE |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1.4709 | 4,250 | 1.4785 | 1.4715 |
| 0 | 1 | 1.4684 | 4,250 | 1.4768 | 1.4686 |
| 1 | 0 | 1.4683 | 4,000 | 1.4843 | 1.4676 |
| 1 | 1 | 1.4689 | 4,250 | 1.4838 | 1.4693 |

All four runs improved substantially from the step-zero validation CE of
approximately `2.86--2.89` and continued improving through the H3 transition
window. The best checkpoints were later than the transition windows, leaving
NTP headroom for a future intervention.

### Where H2 and H3 become decodable

The table below gives the first step of a two-checkpoint consecutive window
for which the pre-specified probe criteria are met at each layer. A dash means
that the criterion was not met twice by step 5,000. The layer order in each
vector is `j=0,1,...,8`.

| run | H2 probe-only onset by layer | H3 probe-only onset by layer |
|---|---|---|
| grammar 0 / model 0 | `--, 2000, 1000, 1000, 1000, 1000, 1000, 1000, 1000` | `--, --, 2000, 2000, 2000, 2000, 2000, 2000, 2000` |
| grammar 0 / model 1 | `--, 3000, 1000, 1000, 1000, 1000, 1000, 1000, 1000` | `--, --, 3000, 2500, 2000, 2000, 2000, 2000, 2000` |
| grammar 1 / model 0 | `--, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000` | `--, --, 3000, 2000, 1500, 1500, 1500, 1500, 1500` |
| grammar 1 / model 1 | `--, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000` | `--, --, --, 2000, 2000, 2000, 2000, 1500, 1500` |

Probe-only accessibility spreads through deeper layers after it first appears,
so it should not be interpreted as a one-block computation claim. The stable
cross-run result is that H2 is first accessible around `j=2`, while H3 is
first accessible around `j=3` or deeper.

### Same-layer A+C acquisition

The primary acquisition test combines balanced probe accuracy and synonym
invariance at the same layer. The table gives the first two-checkpoint window
and the layers that satisfy the criteria throughout that window.

| run | H2 A+C window | H2 layers | H3 A+C window | H3 layers |
|---|---|---|---|---|
| grammar 0 / model 0 | 1,500--2,000 | `j=2--8` | 3,000--3,500 | `j=3--8` |
| grammar 0 / model 1 | 1,500--2,000 | `j=2--8` | 3,000--3,500 | `j=3--8` |
| grammar 1 / model 0 | 1,000--1,500 | `j=2--8` | 2,500--3,000 | `j=3--8` |
| grammar 1 / model 1 | 1,000--1,500 | `j=2--8` | 2,500--3,000 | `j=3--8` |

This is the central result of the rerun. The H2-to-H3 ordering and the
one-block-deeper onset pattern replicate across both grammar realizations and
both model seeds. H3 acquisition occurs later while validation NTP is still
falling: the H3 windows begin at validation CE `1.510--1.526` and the best
validation CE is approximately `1.468--1.471`.

### Raw same-layer intervention checks

The following values are at the first robust layer (`j=2` for H2 and `j=3`
for H3), at the second checkpoint of each A+C window. `d_syn` is the
synonymous surface-change distance, `d_variable` is the latent-replacement
distance, and `d_non` is the unrelated-reference distance. `Q` is the
pre-specified normalized contrast `(d_variable - d_syn) / d_non`.

| run | level/layer | A | C | `d_syn` | `d_variable` | `d_non` | `S` | `Q` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| grammar 0 / model 0 | H2 / `j=2` | .989 | .620 | .799 | 2.116 | 2.105 | 1.0049 | +.625 |
| grammar 0 / model 0 | H3 / `j=3` | .725 | .242 | 1.550 | 2.021 | 2.045 | .9885 | +.230 |
| grammar 0 / model 1 | H2 / `j=2` | .978 | .554 | .942 | 2.072 | 2.113 | .9807 | +.535 |
| grammar 0 / model 1 | H3 / `j=3` | .684 | .293 | 1.446 | 2.011 | 2.044 | .9839 | +.276 |
| grammar 1 / model 0 | H2 / `j=2` | .917 | .571 | .915 | 2.105 | 2.132 | .9870 | +.558 |
| grammar 1 / model 0 | H3 / `j=3` | .868 | .391 | 1.241 | 2.060 | 2.037 | 1.0112 | +.402 |
| grammar 1 / model 1 | H2 / `j=2` | .781 | .156 | 1.767 | 2.023 | 2.094 | .9661 | +.122 |
| grammar 1 / model 1 | H3 / `j=3` | .799 | .268 | 1.488 | 2.024 | 2.031 | .9964 | +.264 |

`Q` is positive at the first robust H2 and H3 layer in every run, and the
latent-replacement distance exceeds the synonymous distance in every one of
these eight comparisons. This provides no evidence of persistent
representation collapse at the claimed transitions. The legacy `S >= 1.005`
continuity check is mixed, especially for H2 and H3 at their first robust
layers; under the pre-run specification it is reported as a secondary ratio,
not used as a second acquisition hurdle. No new hard Q threshold was chosen
after inspecting the results.

H4 does not satisfy the same-layer A+C rule in any run by step 5,000. Some
late H4 probe accessibility is visible, but its synonym score remains near
baseline, so the rerun does not support an H4 abstraction claim.

### Confirmation decision

The full rerun meets the embedded expectation and therefore receives a
**go** decision for the next implementation stage:

- held-out NTP improves strongly and retains headroom after H1, H2, and H3
  transitions;
- H2 is acquired before H3 in all four runs;
- the same-layer onset is consistently `j=2` for H2 and `j=3` for H3;
- the H2-to-H3 timing order replicates across grammar and model seeds;
- the raw same-layer intervention contrast is positive at every claimed H2
  and H3 onset; and
- H4 remains correctly marked as unacquired rather than being inferred from
  probe-only accessibility.

The go authorizes a separate fixed-target auxiliary-loss comparison. It does
not claim that vanilla NTP acquires H4, that the hierarchy is encoded in one
block per level, or that the legacy sensitivity ratio is a calibrated
statistical test. The next experiment should preserve this baseline and
compare fixed target layers before attempting adaptive target switching.
