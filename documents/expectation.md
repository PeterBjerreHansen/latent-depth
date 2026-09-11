# Baseline expectation: establish the developmental clock

**Status:** pre-run expectation for the next experimental round
**Decision:** compare the completed baseline with this document before implementing the latent auxiliary loss

## Purpose

The ordinary causal next-token prediction (NTP) baseline has one specific job:
show that the model develops a temporally ordered hierarchy of internal
abstractions, with enough separation between transitions that changing the
auxiliary target could plausibly matter.

This is a decision gate, not a claim that the baseline must reproduce every
part of the project hypothesis. We should not implement or interpret the
latent auxiliary loss until the baseline results have been compared with these
expectations and the decision has been recorded.

This document is the companion decision gate for the current implementation and experimental work. Read it alongside the [implementation notes](IMPLEMENTATION_NOTES.md), [recommended next runs](RESEARCH_NEXT_RUNS.md), [validation plan](VALIDATION.md), [validation results](VALIDATION_RESULTS.md), and [project proposal](writeup_v2.md). It applies specifically to the ordinary-NTP developmental screen before the latent auxiliary objective is added.

## Expected setting

The intended developmental screen is the binary, depth-five task with
`s=2`, using ordinary causal NTP only. The current starting configuration is
[`configs/developmental_l5_screen.json`](../configs/developmental_l5_screen.json);
the canonical training-pool size `P` should be frozen before confirmation runs
and not chosen separately for each seed.

For the non-root levels, counted upward from the leaves, the first causal
completion positions are:

| level | completion position |
|---:|---:|
| `r=1` | `t=1` |
| `r=2` | `t=3` |
| `r=3` | `t=7` |
| `r=4` | `t=15` |

These are the positions at which the existing probes and intervention
diagnostics should be interpreted. They are not a requirement that the
corresponding abstraction appear in one particular Transformer block.

## Expected training-age pattern

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
headroom. As an order-of-magnitude example, transition times such as
`tau_1 ~= 1000`, `tau_2 ~= 3000`, and `tau_3 ~= 7000` would be useful; the
numbers themselves do not matter.

Layer order is deliberately weaker than level order. We do not require block 1
to encode `H1`, block 2 to encode `H2`, and so on. A result in which `H1` first
becomes strong in blocks 2--4 and `H2` later appears in blocks 4--7 is fully
consistent with the hypothesis. The useful object is the evolution of the
layer-by-level heatmaps.

## What the run must show

### 1. Held-out NTP learning

Validation NLL must fall substantially from step zero and remain better than
the relevant uniform and grammar-aware controls. Per-position NLL should show
that learning is not confined to trivial local positions. Training loss alone
is insufficient.

### 2. Latent accessibility above controls

For each `H_r`, the balanced accuracy of a frozen linear probe should rise
clearly above:

- its step-zero value;
- the shuffled-label probe; and
- the empirical majority-class baseline.

The theoretical uniform chance value is useful context, but step-zero accuracy
need not equal it: token identity and position can make some information
linearly accessible before training.

### 3. Synonym invariance at the same transition

Probe accuracy by itself is not enough. When a different production rule
realizes the same latent, the representation should become more similar than
when the corresponding latent changes. In the repository's notation, the
synonym-clustering score

\[
C_{j,r}=1-\frac{d_{\mathrm{syn}}}{d_{\mathrm{non}}}
\]

should move from its initial baseline toward positive values at approximately
the same training ages at which `H_r` becomes accessible. A probe that rises
while clustering remains flat near zero is evidence for decodability, not yet
for the abstraction of interest.

### 4. Sensitivity to changing the latent

Synonym invariance must not be explained by representation collapse. Changing
`H_r` should still produce a substantial representation change, measured by
the matched latent-replacement sensitivity control. The strongest evidence is
therefore invariance to a rule/surface change together with sensitivity to a
latent change.

### 5. Ordered, replicated transitions

We do not require a mathematically perfect ordering for every seed. We do
expect something qualitatively like

\[
\tau_1 < \tau_2 < \tau_3
\]

on the canonical run, with the `H1 -> H2` ordering surviving at least one
additional model seed and one additional grammar realization. `H3` emerging
and `H4` remaining late are supporting evidence; convincingly acquiring `H1`
and `H2` is the minimum useful result.

## Go/no-go decision

### Proceed to implement the latent auxiliary loss if

The baseline satisfies all of the following:

- held-out NTP improves strongly and still has headroom after `H1` is learned;
- `H1` and `H2` are convincingly acquired, with `H3` at least beginning to
  emerge if the run budget permits;
- the acquisition times are visibly separated rather than all appearing at the
  first evaluation;
- each claimed transition is supported by both balanced linear accessibility
  and positive synonym invariance, with latent-replacement sensitivity still
  present;
- the qualitative `H1 -> H2` ordering is not peculiar to one grammar/model
  seed pair; and
- there is a useful transition checkpoint where `H1` is established, `H2` is
  emerging, `H3` is mostly absent, and NTP is still improving.

If these conditions hold, freeze the baseline protocol and implement the first
auxiliary comparison as separate from-scratch runs:

\[
\mathcal L=\mathcal L_{\mathrm{NTP}}+
\lambda\mathcal L_{\mathrm{next\ latent}}^{(j)},
\qquad
j\in\{\text{embedding},1,2,4,6,8\}.
\]

The first question is simply which fixed target minimizes time to acquire
`H1`, `H2`, and `H3`. Adaptive target switching should come only after that
fixed-target comparison has produced a meaningful developmental clock.

### Do not proceed yet if

- probes become strong but synonym clustering stays near its initial baseline;
- all hierarchy levels rise together at the first evaluation;
- only `H1` develops within the available budget;
- the ordering changes wildly across grammar seeds; or
- NTP saturates before there is a window in which different target choices
  could plausibly have different value.

These outcomes are informative baseline failures, not reasons to add the
auxiliary objective anyway. The next response should adjust the dataset size,
training budget, evaluation cadence, or model capacity, then rerun the NTP
baseline against this same expectation. In particular, if the screen is too
easy, evaluate more frequently or make the regime harder; if it is too hard,
increase `P`, training duration, or capacity before changing the objective.

## Reporting requirements

The baseline report should include, on the same training-age axis:

- held-out mean and per-position NTP NLL;
- layer-by-level balanced probe accuracy with shuffled-label, majority, and
  untrained-backbone controls;
- synonym-clustering scores and their synonym/non-synonym distances;
- latent-replacement sensitivity;
- transition estimates or clearly marked qualitative transition windows; and
- the canonical configuration, grammar/model seeds, checkpoints, total samples
  seen, and device/runtime details.

The final report should explicitly state **go**, **no-go**, or **rerun baseline**
against the criteria above. Only a **go** decision authorizes moving on to the
latent auxiliary-loss implementation.
