# Stage 02: fixed auxiliary target depth

Stage 01 established a reproducible developmental clock: in the canonical
`P=65,536` regime, the first robust same-layer accessibility + synonym-
invariance signal appears for H2 around residual stream `j=2`, and H3 appears
later around `j=3`, while held-out NTP is still improving. Stage 02 asks the
next causal question:

> Does predicting a fixed future Transformer residual stream accelerate those
> hierarchy transitions, and does the useful target depth differ between H2
> and H3?

This is deliberately a fixed-target screen. It does **not** implement adaptive
target selection, switching schedules, target mixtures, or EMA teachers. The
primary screen and the weight-sensitivity follow-ups each use one fixed lambda.

## Training objective

Every arm keeps ordinary causal next-token prediction active. For a target
residual-stream depth `j`, the auxiliary objective is

\[
\mathcal L = \mathcal L_{\mathrm{NTP}}
+ \lambda \mathcal L_{\mathrm{aux}}^{(j)},
\]

\[
\mathcal L_{\mathrm{aux}}^{(j)} =
\frac{1}{T-1}\sum_{t=0}^{T-2}
\left[1-\cos\left(q(h_t^{(D)}),
\operatorname{sg}(h_{t+1}^{(j)})\right)\right].
\]

`D=8` is the final post-block residual stream. The target is detached, so the
auxiliary gradient flows through the predictor and source branch but not
through `h_(t+1)^j`. The hidden-state convention is shared with Stage 01:

- `j=0`: token + positional embedding stream;
- `j=1,...,8`: post-Transformer-block residual streams;
- the final LayerNorm output is not a target layer.

Ground-truth RHM latents H1--H4 are never used in the training loss. They remain
observer diagnostics only. The name “next latent” refers to a model residual
stream target here; this is not direct ground-truth latent supervision. Because
the residual streams also carry token and position information, especially at
`j=0`, interpret a strong shallow-target result cautiously. The first screen
uses the existing shuffled-label and untrained-backbone controls; explicit
surface-token or position controls are a follow-up only if the results motivate
them.

The predictor is identical in every auxiliary arm:

```text
LayerNorm(256)
Linear(256, 512)
GELU
Linear(512, 256)
```

It is initialized from a separate auxiliary seed without advancing the
backbone/training RNG stream. Thus arms with the same grammar/model seed start
from identical GPT weights and receive the same minibatch ordering.

## Screen protocol

The screen is defined by
[`configs/target_depth_screen_lambda_0_1.json`](configs/target_depth_screen_lambda_0_1.json). It
copies the Stage-01 canonical regime:

- RHM: `v=n=16`, `m=4`, `s=2`, `L=5`;
- `P=65,536`, validation/test size `16,384`;
- 8 blocks, 8 heads, width 256, dropout 0;
- AdamW, learning rate `3e-4`, batch size 256;
- 5,000 optimizer updates;
- NTP evaluation every 250 updates, including step zero;
- exact checkpoints every 500 updates;
- grammar seed 0 and model seed 0 for the first screen.

The arms are

\[
\{\mathrm{NTP},\; j=0,1,2,3,4,5,6,7,8\}.
\]

The first screen includes every residual-stream depth so that a sparse result
cannot be mistaken for a complete progressive-depth comparison.

The primary target-depth screen uses `lambda=0.1`. The follow-up screens use
the same complete target-depth grid at fixed `lambda=0.3` and `lambda=1.0`.
These are separate fixed-weight screens, not an adaptive weighting experiment.
Before looking at the results, use these practical margins:

- one 500-update checkpoint is the minimum meaningful acquisition-time
  difference, because the trajectory diagnostics use exact 500-update saves;
- a best-validation-CE increase of more than `0.01` over the paired NTP arm is
  an unacceptable NTP cost.

The `lambda=0.3` and `lambda=1.0` screens are exploratory weight-sensitivity
follow-ups. They do not replace the primary `lambda=0.1` target-depth result
or authorize target selection from validation CE alone.

Validation checkpoint selection remains based **only** on held-out NTP
cross-entropy. `running_aux_loss` and `running_total_loss` are training
observables, not selection criteria.

## Pre-run expectations

This section is the pre-run specification for the Stage-02 report. The report
must compare the completed screens with these expectations. The expectations
are directional: a target residual-stream depth is not the same thing as an
RHM latent level, so we do not require `j=r` or a mathematically monotone
mapping from target depth to latent level.

### Expected target-depth pattern

The main hypothesis is that the useful target depth changes with the level
whose acquisition still offers meaningful NTP progress:

| target depth | expected early levels H1/H2 | expected later levels H3/H4 | main interpretation risk |
|---|---|---|---|
| `j=0` embedding | possible early benefit, especially for surface prediction | little clean benefit expected | token and position information can produce a false shallow win |
| `j=1--2` shallow post-block | strongest candidate for H2 acceleration | neutral or weaker H3/H4 effect | shallow supervision may consume capacity needed for later structure |
| `j=3--4` intermediate | possible H2 benefit | plausible H3 benefit | a broad win may reflect generic regularization rather than target specificity |
| `j=5--8` deep post-block | little early benefit expected | strongest candidate for H3/H4 acceleration | predicting a difficult target may add noisy or conflicting gradients |

The cleanest positive result would therefore be a target-depth crossover: an
early target has a negative `Delta tau` for H2 but not H3, while a later target
has a more negative `Delta tau` for H3 (and H4 if it is reached). A single
intermediate target that improves both levels is also useful, but it weakens
the case for adaptive target selection.

### What counts as evidence for the pattern

For each fixed weight and target depth, compute the same-layer A+C acquisition
time for every level and observer layer. The primary summary is

```text
Delta tau_r(lambda, j) = tau_AC(lambda, j, H_r) - tau_AC(NTP, H_r)
```

Negative values indicate earlier acquisition than the paired NTP control. A
difference of less than one 500-update checkpoint is a practical tie. The
report will show both the minimum onset over observer layers and the complete
layerwise onset matrix, together with the onset layer. Probe-only onset is
secondary because it can precede synonym invariance without showing the
intended abstraction.

The result is consistent with the main hypothesis when:

- early target depths show their largest improvement on H2, with little or no
  corresponding H3 improvement;
- late target depths show their largest improvement on H3, and on H4 if H4 is
  acquired within the budget;
- the same-layer raw intervention contrast does not show persistent collapse;
  and
- the NTP cost remains acceptable: a best-validation-CE increase above `0.01`
  over the paired NTP arm is a failure for that target/weight combination.

The following outcomes are also informative and must be reported explicitly:

- **No depth pattern:** all target depths have similar `tau` values. The
  auxiliary objective may be ineffective or its effect may be generic rather
  than depth-specific.
- **Uniform acceleration:** every target depth advances H2 and H3 by about the
  same amount. This supports an auxiliary-training effect but not the proposed
  developmental specialization.
- **One common winner:** one target depth advances both H2 and H3. This is a
  useful fixed-target result, but provides little motivation for switching.
- **Probe-only acceleration:** accessibility improves while A+C does not.
  This is not evidence that the target speeds acquisition of the abstraction.

### Where the auxiliary loss may do harm

The auxiliary loss is not expected to improve every arm or every training age.
The main failure modes are:

- `j=0` improves early NTP or H1-like probes while synonym invariance and H2/H3
  timing remain unchanged; this is a surface-token or position confound, not a
  clean latent result;
- deep targets delay H1/H2 because the model spends shared capacity predicting
  a difficult future stream before early structure is established;
- a larger weight lowers `running_aux_loss` but delays A+C acquisition or
  worsens validation CE, indicating objective interference rather than useful
  learning;
- an arm reaches a better intermediate validation CE but degrades at later
  checkpoints, indicating that the auxiliary objective has become harmful after
  the useful transition window; or
- the H2-to-H3 order is disrupted, or the raw `Q` contrast becomes persistently
  non-positive, indicating that the auxiliary objective changed the
  representation in a way that is not supporting the intended abstraction.

`lambda=1.0` is therefore a stress test, not an expectation that more
auxiliary pressure must be better. H1 remains supporting evidence because its
signal is confounded, and H4 not being reached by step 5,000 is a censored
observation rather than evidence that no target can accelerate H4.

## Run the target-depth screen

From the repository root:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python sweep_target_depth.py \
  --config experiments/02_fixed_latent_targets/configs/target_depth_screen_lambda_0_1.json \
  --output-dir runs/02_fixed_latent_targets/screen_lambda_0_1
```

The output layout is intentionally explicit:

```text
runs/02_fixed_latent_targets/screen_lambda_0_1/
├── metrics.jsonl
├── sweep_config.json
└── grammar_0/
    ├── rules.pt
    └── model_0/
        ├── ntp/
        ├── target_0/
        ├── target_1/
        ├── target_2/
        ├── target_3/
        ├── target_4/
        ├── target_5/
        ├── target_6/
        ├── target_7/
        └── target_8/
```

Each arm stores its resolved config, ordinary metrics, best/final checkpoints,
and exact-step snapshots. `--resume` skips completed arms only when the saved
sweep configuration exactly matches the requested sweep.

## Run the weight-sensitivity screens

Run the same full target-depth grid at the two larger fixed auxiliary weights:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python sweep_target_depth.py \
  --config experiments/02_fixed_latent_targets/configs/target_depth_screen_lambda_0_3.json \
  --output-dir runs/02_fixed_latent_targets/screen_lambda_0_3

PYTORCH_ENABLE_MPS_FALLBACK=0 python sweep_target_depth.py \
  --config experiments/02_fixed_latent_targets/configs/target_depth_screen_lambda_1_0.json \
  --output-dir runs/02_fixed_latent_targets/screen_lambda_1_0
```

## Diagnose the developmental trajectories

Use exactly the Stage-01 observer pipeline. The final Stage-01 confirmation
used 1,024 validation examples and 300 probe steps, so the first Stage-02
comparison should use the same diagnostic settings:

```bash
for screen in screen_lambda_0_1 screen_lambda_0_3 screen_lambda_1_0; do
  for arm in ntp target_0 target_1 target_2 target_3 target_4 target_5 target_6 target_7 target_8; do
    PYTORCH_ENABLE_MPS_FALLBACK=0 python diagnose_trajectory.py \
      --run-dir runs/02_fixed_latent_targets/$screen/grammar_0/model_0/$arm \
      --output-dir runs/02_fixed_latent_targets/$screen/grammar_0/model_0/$arm/trajectory \
      --device mps \
      --num-sequences 1024 \
      --probe-steps 300 \
      --controls

    python plot_trajectory.py \
      --trajectory runs/02_fixed_latent_targets/$screen/grammar_0/model_0/$arm/trajectory/trajectory.json \
      --metrics runs/02_fixed_latent_targets/$screen/grammar_0/model_0/$arm/metrics.json \
      --output runs/02_fixed_latent_targets/$screen/grammar_0/model_0/$arm/trajectory/trajectory.png
  done
done
```

On a non-MPS machine, replace `--device mps` and the training config device as
appropriate; do not mix devices within a paired comparison unless necessary.

## Primary result

Do not invent a new hierarchy-acquisition definition for Stage 02. Reuse the
Stage-01 same-layer rule: balanced probe accessibility plus positive synonym
invariance over a two-checkpoint window, with the raw variable-vs-synonym
intervention contrast used as a collapse check.

The report should reduce the screens to a table of this form, with entries for
each weight/target combination:

| lambda | fixed target | H2 onset `tau_2` | H3 onset `tau_3` | best validation CE | CE cost vs NTP |
|---:|---|---:|---:|---:|---:|
| `0.1`, `0.3`, or `1.0` | NTP or `j=0,...,8` | | | | |

The scientific question is the **ranking across target depths**, not merely
whether an auxiliary loss can lower its own training objective.

The pre-declared margins above define practical ties and costs; they are not a
new hierarchy-acquisition threshold. A developmental crossover would look like
different target depths minimizing
`tau_2` and `tau_3`. If one target wins both transitions, that weakens the need
for adaptive target selection but is still a useful positive auxiliary result.
If no target beats NTP, first assess the common auxiliary weight before adding
more complicated machinery.

## What happens after the screen

Do not replicate all ten arms automatically. Select NTP plus only the target
layers that are scientifically competitive (for example, a shallow H2 winner,
a deeper H3 winner, and perhaps one intermediate control), then run that subset
on the same 2 x 2 grammar/model-seed factorial used in Stage 01.

Only after a replicated fixed-target comparison shows that the locally useful
target changes with developmental stage should the repository add switching or
adaptive target selection.
