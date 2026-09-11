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
target selection, switching schedules, target mixtures, EMA teachers, or a
lambda grid.

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
[`configs/target_depth_screen.json`](configs/target_depth_screen.json). It
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

The first auxiliary weight is fixed at `lambda=0.1`. Do not sweep target depth
and lambda simultaneously. Before looking at the screen, fix these practical
margins:

- one 500-update checkpoint is the minimum meaningful acquisition-time
  difference, because the trajectory diagnostics use exact 500-update saves;
- a best-validation-CE increase of more than `0.01` over the paired NTP arm is
  an unacceptable NTP cost.

A target is competitive when it advances H2 or H3 by at least one checkpoint
without exceeding that NTP cost. If no target is competitive and no target
exceeds the cost margin, run one pre-declared follow-up at `lambda=0.3`. If all
auxiliary targets exceed the cost margin, run one at `lambda=0.03`. Otherwise
freeze `0.1` and study target depth; do not choose lambda after inspecting
individual target winners.

Validation checkpoint selection remains based **only** on held-out NTP
cross-entropy. `running_aux_loss` and `running_total_loss` are training
observables, not selection criteria.

## Run the target-depth screen

From the repository root:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python sweep_target_depth.py \
  --config experiments/02_fixed_latent_targets/configs/target_depth_screen.json \
  --output-dir runs/02_fixed_latent_targets/screen
```

The output layout is intentionally explicit:

```text
runs/02_fixed_latent_targets/screen/
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

## Diagnose the developmental trajectories

Use exactly the Stage-01 observer pipeline. The final Stage-01 confirmation
used 1,024 validation examples and 300 probe steps, so the first Stage-02
comparison should use the same diagnostic settings:

```bash
for arm in ntp target_0 target_1 target_2 target_3 target_4 target_5 target_6 target_7 target_8; do
  PYTORCH_ENABLE_MPS_FALLBACK=0 python diagnose_trajectory.py \
    --run-dir runs/02_fixed_latent_targets/screen/grammar_0/model_0/$arm \
    --output-dir runs/02_fixed_latent_targets/screen/grammar_0/model_0/$arm/trajectory \
    --device mps \
    --num-sequences 1024 \
    --probe-steps 300 \
    --controls

  python plot_trajectory.py \
    --trajectory runs/02_fixed_latent_targets/screen/grammar_0/model_0/$arm/trajectory/trajectory.json \
    --metrics runs/02_fixed_latent_targets/screen/grammar_0/model_0/$arm/metrics.json \
    --output runs/02_fixed_latent_targets/screen/grammar_0/model_0/$arm/trajectory/trajectory.png
done
```

On a non-MPS machine, replace `--device mps` and the training config device as
appropriate; do not mix devices within a paired comparison unless necessary.

## Primary result

Do not invent a new hierarchy-acquisition definition for Stage 02. Reuse the
Stage-01 same-layer rule: balanced probe accessibility plus positive synonym
invariance over a two-checkpoint window, with the raw variable-vs-synonym
intervention contrast used as a collapse check.

The first report should reduce the screen to a table of this form:

| fixed target | H2 onset `tau_2` | H3 onset `tau_3` | best validation CE |
|---|---:|---:|---:|
| NTP | | | |
| embedding (`j=0`) | | | |
| `j=1` | | | |
| `j=2` | | | |
| `j=3` | | | |
| `j=4` | | | |
| `j=5` | | | |
| `j=6` | | | |
| `j=7` | | | |
| `j=8` | | | |

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
