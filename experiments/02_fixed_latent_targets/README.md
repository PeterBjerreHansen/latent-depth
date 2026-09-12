# Stage 02: fixed auxiliary target depth

Stage 01 supplies the baseline developmental clock for the canonical
per-epoch training-pool regime. Stage 02 asks the next causal question:

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
- per-epoch training pool `P=65,536`, validation/test size `16,384`;
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
The immediate follow-up is the fresh L5, `lambda=0.1` screen. The larger-weight
screens are conditional follow-ups after that screen has been analyzed.

## Frozen acquisition analysis

Raw trajectories are collected by `diagnose_trajectory.py`; that command makes
no developmental judgments. The committed
[`acquisition_rule.json`](acquisition_rule.json) is the single source of truth
for acquisition analysis. It requires balanced accessibility and synonym
invariance at the **same observer layer** for two consecutive diagnostic
checkpoints:

\[
A_{r,k}(t) \ge \max\{A_{r,k}(0)+\delta_0,
A^{\rm shuffled}_{r,k}(t)+\delta_{\rm shuf},
A^{\rm balanced\ baseline}_{r}+\delta_{\rm base}\},
\]

\[
C_{r,k}(t) \ge \max\{C_{r,k}(0)+\delta_C,C_{\min}\}.
\]

The first checkpoint in the persistent pair is `onset_step`; the second is
`confirmed_step`. The level onset `tau_r` is the earliest same-layer onset over
observer layers. Every layerwise onset is retained, and an unreached level or
layer is explicitly marked `censored` through the final diagnostic step.

The rule also reports independent balanced-probe milestones (`0.50`, `0.75`,
and `0.90`) and the normalized intervention contrast

\[
Q_{r,k}(t)=\frac{d_{\rm variable}-d_{\rm syn}}
                 {d_{\rm non}+\epsilon}.
\]

`Q` is a reported representation diagnostic, not an additional acquisition
hurdle. Detectable emergence and high decoding accuracy are separate claims.

Apply the rule with:

```bash
python summarize_trajectory.py \
  --trajectory RUN/trajectory/trajectory.json \
  --rule experiments/02_fixed_latent_targets/acquisition_rule.json \
  --output RUN/trajectory/acquisition.json
```

Do not change the rule after inspecting auxiliary-arm summaries. If its margins
need calibration, inspect the fresh NTP trajectory and its controls first, then
edit and freeze the committed rule before summarizing any target arm.

Before looking at results, use these practical margins:

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

## Training-data schedule

The default configuration uses one fixed finite training pool. The developmental
screens below explicitly set `data.resample_train_each_epoch` to `true`. Their
validation and test pools remain fixed, while the training pool is sampled
afresh from the same grammar before each epoch. Epoch 1 uses the configured
`rhm.train_seed`; epoch `e` uses the deterministic seed
`train_seed + 1,000,003 * (e - 1)` modulo `2^63 - 1`. This gives every arm in
the paired sweep the same epoch-specific examples without requiring duplicate
checking. A resumed run regenerates the pool for the checkpoint's current
epoch while preserving the checkpoint's sampler position.

With epoch-wise resampling, `65,536` is the size of each fresh training pool,
not the total finite dataset seen during training. At batch size 256, the L5
screen processes `5,000 × 256 = 1,280,000` sequence draws and `39,680,000`
predicted tokens per arm. The L6, 10,000-update follow-up would process
`2,560,000` sequence draws and `161,280,000` predicted tokens per arm. Record
the per-epoch pool, optimizer updates, sequence draws, predicted tokens, and
`resample_train_each_epoch=true` for every result. Training CE is evaluated on
the current epoch's pool; model selection and the final test remain held out.

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
- the raw `Q` contrast becomes persistently non-positive or other controls show
  representation collapse, indicating that the auxiliary objective changed the
  representation in a way that is not supporting the intended abstraction.

An auxiliary arm showing H3 before or near H2 is not automatically harmful.
Earlier H3 accessibility with healthy same-layer A+C/Q controls and good NTP
is potentially the effect of interest. The comparison must still report both
absolute times: a shorter H2-to-H3 interval caused by delaying H2 is not H3
acceleration.

`lambda=1.0` is therefore a stress test, not an expectation that more
auxiliary pressure must be better. H1 remains supporting evidence because its
signal is confounded, and H4 not being reached by step 5,000 is a censored
observation rather than evidence that no target can accelerate H4.

## Run the target-depth screen

From the repository root:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python sweep_target_depth.py \
  --config experiments/02_fixed_latent_targets/configs/target_depth_screen_lambda_0_1.json \
  --output-dir experiments/02_fixed_latent_targets/runs/target_depth_l5_lambda_0_1_5000_updates
```

The output layout is intentionally explicit:

```text
experiments/02_fixed_latent_targets/runs/target_depth_l5_lambda_0_1_5000_updates/
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

### Checkpoint storage

Set `train.save_checkpoints` to `false` for a metrics-only sweep. The sweep
still writes `metrics.jsonl`, each arm's `metrics.json`, resolved configs, and
the sweep manifest, but it does not write `best.pt`, `last.pt`, or exact-step
model files. Completed arms can still be skipped with `--resume`; an
interrupted arm must be restarted because there is no state checkpoint.

Keep `train.save_checkpoints: true` for this Stage-02 target-depth study until
the trajectory diagnostics are complete. The offline trajectory pipeline reads
the exact-step snapshots to measure layerwise H1--H5 acquisition. After those
diagnostics and the report are finished, the step snapshots can be removed if
the retained metrics and diagnostic JSON are sufficient.

## Conditional weight-sensitivity screens

After the fresh L5/`lambda=0.1` screen has been diagnosed, summarized, and
reviewed, run the same full target-depth grid at the two larger fixed auxiliary
weights if that follow-up is justified:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python sweep_target_depth.py \
  --config experiments/02_fixed_latent_targets/configs/target_depth_screen_lambda_0_3.json \
  --output-dir experiments/02_fixed_latent_targets/runs/target_depth_l5_lambda_0_3_5000_updates

PYTORCH_ENABLE_MPS_FALLBACK=0 python sweep_target_depth.py \
  --config experiments/02_fixed_latent_targets/configs/target_depth_screen_lambda_1_0.json \
  --output-dir experiments/02_fixed_latent_targets/runs/target_depth_l5_lambda_1_0_5000_updates
```

## Diagnose and summarize the developmental trajectories

Use exactly the Stage-01 observer pipeline. The final Stage-01 confirmation
used 1,024 validation examples and 300 probe steps, so the first Stage-02
comparison should use the same diagnostic settings:

```bash
for arm in ntp target_0 target_1 target_2 target_3 target_4 target_5 target_6 target_7 target_8; do
  screen=target_depth_l5_lambda_0_1_5000_updates
  PYTORCH_ENABLE_MPS_FALLBACK=0 python diagnose_trajectory.py \
    --run-dir experiments/02_fixed_latent_targets/runs/$screen/grammar_0/model_0/$arm \
    --output-dir experiments/02_fixed_latent_targets/runs/$screen/grammar_0/model_0/$arm/trajectory \
    --device mps \
    --num-sequences 1024 \
    --probe-steps 300 \
    --controls

  python summarize_trajectory.py \
    --trajectory experiments/02_fixed_latent_targets/runs/$screen/grammar_0/model_0/$arm/trajectory/trajectory.json \
    --rule experiments/02_fixed_latent_targets/acquisition_rule.json \
    --output experiments/02_fixed_latent_targets/runs/$screen/grammar_0/model_0/$arm/trajectory/acquisition.json

  python plot_trajectory.py \
    --trajectory experiments/02_fixed_latent_targets/runs/$screen/grammar_0/model_0/$arm/trajectory/trajectory.json \
    --metrics experiments/02_fixed_latent_targets/runs/$screen/grammar_0/model_0/$arm/metrics.json \
    --output experiments/02_fixed_latent_targets/runs/$screen/grammar_0/model_0/$arm/trajectory/trajectory.png
done

python summarize_target_depth.py \
  --screen-dir experiments/02_fixed_latent_targets/runs/target_depth_l5_lambda_0_1_5000_updates \
  --rule experiments/02_fixed_latent_targets/acquisition_rule.json \
  --output-dir experiments/02_fixed_latent_targets/runs/target_depth_l5_lambda_0_1_5000_updates/analysis

python plot_trajectory.py \
  --comparison experiments/02_fixed_latent_targets/runs/target_depth_l5_lambda_0_1_5000_updates/analysis/comparison.json \
  --output-dir experiments/02_fixed_latent_targets/runs/target_depth_l5_lambda_0_1_5000_updates/analysis/plots
```

The sweep summarizer writes absolute `tau_2`/`tau_3`, paired `Delta tau`, the
H2-to-H3 interval, explicit censoring bounds, matched-update validation CE, and
`validation_ce_by_step.csv`. It refuses to compare arms with different
hierarchy configurations, diagnostic schedules, or validation checkpoint
steps. It also refuses a partial target-depth screen: each paired group must
contain NTP and every `target_0` through `target_8` arm.

The larger-weight screens are conditional. After the fresh `lambda=0.1` L5
comparison is reviewed, repeat the same diagnosis, summary, and plot workflow
for a pre-approved weight follow-up if needed:

```bash
for screen in target_depth_l5_lambda_0_3_5000_updates target_depth_l5_lambda_1_0_5000_updates; do
  for arm in ntp target_0 target_1 target_2 target_3 target_4 target_5 target_6 target_7 target_8; do
    PYTORCH_ENABLE_MPS_FALLBACK=0 python diagnose_trajectory.py \
      --run-dir experiments/02_fixed_latent_targets/runs/$screen/grammar_0/model_0/$arm \
      --output-dir experiments/02_fixed_latent_targets/runs/$screen/grammar_0/model_0/$arm/trajectory \
      --device mps \
      --num-sequences 1024 \
      --probe-steps 300 \
      --controls

    python summarize_trajectory.py \
      --trajectory experiments/02_fixed_latent_targets/runs/$screen/grammar_0/model_0/$arm/trajectory/trajectory.json \
      --rule experiments/02_fixed_latent_targets/acquisition_rule.json \
      --output experiments/02_fixed_latent_targets/runs/$screen/grammar_0/model_0/$arm/trajectory/acquisition.json

    python plot_trajectory.py \
      --trajectory experiments/02_fixed_latent_targets/runs/$screen/grammar_0/model_0/$arm/trajectory/trajectory.json \
      --metrics experiments/02_fixed_latent_targets/runs/$screen/grammar_0/model_0/$arm/metrics.json \
      --output experiments/02_fixed_latent_targets/runs/$screen/grammar_0/model_0/$arm/trajectory/trajectory.png
  done
done

for screen in target_depth_l5_lambda_0_3_5000_updates target_depth_l5_lambda_1_0_5000_updates; do
  python summarize_target_depth.py \
    --screen-dir experiments/02_fixed_latent_targets/runs/$screen \
    --rule experiments/02_fixed_latent_targets/acquisition_rule.json \
    --output-dir experiments/02_fixed_latent_targets/runs/$screen/analysis

  python plot_trajectory.py \
    --comparison experiments/02_fixed_latent_targets/runs/$screen/analysis/comparison.json \
    --output-dir experiments/02_fixed_latent_targets/runs/$screen/analysis/plots
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

| lambda | fixed target | H2 `tau_2` / `Delta tau_2` | H3 `tau_3` / `Delta tau_3` | `tau_3 - tau_2` | best validation CE / matched CE |
|---:|---|---:|---:|---:|---:|
| `0.1`, `0.3`, or `1.0` | NTP or `j=0,...,8` | | | | |

The scientific question is the **ranking across target depths**, not merely
whether an auxiliary loss can lower its own training objective.

The pre-declared margins above define practical ties and costs; they are not a
new hierarchy-acquisition threshold. A developmental crossover would look like
different target depths minimizing
`tau_2` and `tau_3`. If one target wins both transitions, that weakens the need
for adaptive target selection but is still a useful positive auxiliary result.
If no target beats NTP, first assess the common auxiliary weight and probe
budget before adding more complicated machinery.

## Conditional follow-up: extended-hierarchy target-depth sweep

Do not run this screen until the fresh resampled L5/`lambda=0.1` NTP baseline
and complete target-depth comparison have been summarized and reviewed. The old
fixed-data failure to acquire H4 is not a calibration for the fresh-data regime.
If the L5 result leaves a justified late-level question, this conditional
follow-up uses `L=6`, `lambda=1.0`, and 10,000 updates. It sweeps the complete
target grid and includes an NTP arm in the same sweep.

Use [`configs/target_depth_screen_l6_lambda_1_0.json`](configs/target_depth_screen_l6_lambda_1_0.json)
and write the result under
`runs/target_depth_l6_lambda_1_0_10000_updates_resampled_train/`:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python sweep_target_depth.py \
  --config experiments/02_fixed_latent_targets/configs/target_depth_screen_l6_lambda_1_0.json \
  --output-dir experiments/02_fixed_latent_targets/runs/target_depth_l6_lambda_1_0_10000_updates_resampled_train
```

The run keeps the Stage-02 model, seeds, data sizes, optimizer, and MPS
settings, but uses `L=6` and 10,000 updates. The sequence has 64 tokens and
63 predicted tokens, so each arm processes about 16,128 predicted tokens per
update and 161.3 million over the full budget. Report onset in both updates
and predicted tokens.

The arms are exactly:

```text
NTP, target_0, target_1, ..., target_8
```

Run the existing trajectory diagnostics for every arm after training:

```bash
for arm in ntp target_0 target_1 target_2 target_3 target_4 target_5 target_6 target_7 target_8; do
  PYTORCH_ENABLE_MPS_FALLBACK=0 python diagnose_trajectory.py \
    --run-dir experiments/02_fixed_latent_targets/runs/target_depth_l6_lambda_1_0_10000_updates_resampled_train/grammar_0/model_0/$arm \
    --output-dir experiments/02_fixed_latent_targets/runs/target_depth_l6_lambda_1_0_10000_updates_resampled_train/grammar_0/model_0/$arm/trajectory \
    --device mps \
    --num-sequences 1024 \
    --probe-steps 300 \
    --controls

  python summarize_trajectory.py \
    --trajectory experiments/02_fixed_latent_targets/runs/target_depth_l6_lambda_1_0_10000_updates_resampled_train/grammar_0/model_0/$arm/trajectory/trajectory.json \
    --rule experiments/02_fixed_latent_targets/acquisition_rule.json \
    --output experiments/02_fixed_latent_targets/runs/target_depth_l6_lambda_1_0_10000_updates_resampled_train/grammar_0/model_0/$arm/trajectory/acquisition.json

  python plot_trajectory.py \
    --trajectory experiments/02_fixed_latent_targets/runs/target_depth_l6_lambda_1_0_10000_updates_resampled_train/grammar_0/model_0/$arm/trajectory/trajectory.json \
    --metrics experiments/02_fixed_latent_targets/runs/target_depth_l6_lambda_1_0_10000_updates_resampled_train/grammar_0/model_0/$arm/metrics.json \
    --output experiments/02_fixed_latent_targets/runs/target_depth_l6_lambda_1_0_10000_updates_resampled_train/grammar_0/model_0/$arm/trajectory/trajectory.png
done
```

Then run `summarize_target_depth.py --rule experiments/02_fixed_latent_targets/acquisition_rule.json`
and the comparison plotter on the L6 screen as shown for L5 above. H4 should have a
realistic chance to
appear, while H5 may remain censored. If H4 remains censored, the sweep can
still describe effects on H2/H3, but it cannot test late-level specialization
directly. Do not add another auxiliary weight or adaptive target schedule
until this complete `L=6` result has been reviewed.

## What happens after the screen

The `L=6` run intentionally includes all ten arms because the complete
target-depth pattern is the current question. After it completes, select NTP
plus only the scientifically competitive target layers for replication on the
same 2 x 2 grammar/model-seed factorial used in Stage 01.

Only after a replicated fixed-target comparison shows that the locally useful
target changes with developmental stage should the repository add switching or
adaptive target selection.
