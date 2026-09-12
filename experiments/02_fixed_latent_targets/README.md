# Stage 02: fixed auxiliary target depth

Stage 01 supplies the NTP baseline. Stage 02 asks the first empirical
comparison question:

> At the same training exposure, do different auxiliary targets make
> particular hierarchy levels accessible earlier?

The runnable experiment is a fixed-target screen. It does not implement
adaptive target switching, target mixtures, EMA teachers, or a universal
“solved” label.

## Training objective

Every arm keeps ordinary causal next-token prediction active. For a fixed
residual-stream target depth `j`,

\[
\mathcal L = \mathcal L_{\mathrm{NTP}} +
\lambda\mathcal L_{\mathrm{aux}}^{(j)}.
\]

The auxiliary predictor maps the final residual stream at position `t` to the
detached residual stream at position `t+1`. `j=0` is the token-plus-position
embedding stream; `j=1,...,8` are post-block streams; the final LayerNorm
output is not a target. Ground-truth RHM latents remain observer diagnostics,
not training targets. Because residual streams carry token and position
information, especially at `j=0`, interpret a shallow-target result
cautiously.

## Runnable L5 screen

The committed screen is
[`configs/target_depth_screen_lambda_0_1.json`](configs/target_depth_screen_lambda_0_1.json).
It uses the Stage-01 exposure regime:

- `v=n=16`, `m=4`, `s=2`, `L=5`;
- fresh `P=65,536` training pool each epoch and fixed validation/test pools;
- eight blocks, width 256, batch size 256, AdamW with learning rate `3e-4`;
- 5,000 optimizer updates, evaluation every 250 updates including step zero;
- exact diagnostic checkpoints every 500 updates; and
- NTP plus fixed target depths `j=0,...,8`.

The screen uses `lambda=0.1`, grammar seed 0, and model seed 0. All arms share
the same grammar, data schedule, and training exposure. Higher weights are
exploratory follow-ups only; add their explicit configs after the L5 result is
reviewed if the result warrants a weight-sensitivity question. A deeper `L=6`
screen is likewise a later, separately justified experiment rather than part
of this runnable core.

## Primary analysis

The committed rule in [`acquisition_rule.json`](acquisition_rule.json) uses
balanced probe accuracy with one preselected primary threshold, currently
`0.75`. An accessibility event is the first checkpoint at or above that
threshold followed by a second checkpoint at or above it at the same observer
layer. The earliest observed layer supplies `tau_accessibility`; the analysis
retains the full layerwise curves and events. The 50%, 75%, and 90% probe
milestones use the same two-checkpoint persistence rule and are reported to
show threshold dependence.

Synonym clustering and the latent-replacement contrast `Q` are explanatory
diagnostics. Shuffled-label probes are optional controls. Neither clustering,
`Q`, nor a control margin is an acquisition hurdle. If an event is not observed
by the final checkpoint, report “not confirmed by step T”. Numeric onset
differences are available only when both paired events are observed; otherwise
the difference is explicitly unavailable. Do not derive censoring bounds from
backdated onset candidates.

Validation cross-entropy is reported at matched optimizer updates and remains
the model-selection quantity. Auxiliary and total losses are training
observables. The analysis does not declare an arm failed because its best CE
differs by a fixed post hoc amount; interpret matched costs alongside the
accessibility result.

Partial screens are valid. A comparison must contain the NTP arm and at least
one target arm, and records `targets_present` so missing targets are visible.
It need not contain the full target-depth grid.

## Run and analyze

From the repository root:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python sweep_target_depth.py \
  --config experiments/02_fixed_latent_targets/configs/target_depth_screen_lambda_0_1.json \
  --output-dir experiments/02_fixed_latent_targets/runs/target_depth_l5_lambda_0_1_5000_updates
```

Diagnose each arm from its exact-step checkpoints. The controls are optional;
when requested, `--controls` adds the trained-backbone shuffled-label probe.

```bash
for arm in ntp target_0 target_1 target_2 target_3 target_4 target_5 target_6 target_7 target_8; do
  screen=target_depth_l5_lambda_0_1_5000_updates
  PYTORCH_ENABLE_MPS_FALLBACK=0 python diagnose_trajectory.py \
    --run-dir experiments/02_fixed_latent_targets/runs/$screen/grammar_0/model_0/$arm \
    --output-dir experiments/02_fixed_latent_targets/runs/$screen/grammar_0/model_0/$arm/trajectory \
    --device mps --num-sequences 1024 --probe-steps 300 --controls
done
```

Run the single comparison analysis entry point. It loads the raw
`trajectory.json` files, applies the rule in memory, writes a compact table,
and does not create or consume per-arm acquisition summaries:

```bash
python summarize_target_depth.py \
  --screen-dir experiments/02_fixed_latent_targets/runs/target_depth_l5_lambda_0_1_5000_updates \
  --rule experiments/02_fixed_latent_targets/acquisition_rule.json \
  --output-dir experiments/02_fixed_latent_targets/runs/target_depth_l5_lambda_0_1_5000_updates/analysis

python plot_trajectory.py \
  --comparison experiments/02_fixed_latent_targets/runs/target_depth_l5_lambda_0_1_5000_updates/analysis/comparison.json \
  --output-dir experiments/02_fixed_latent_targets/runs/target_depth_l5_lambda_0_1_5000_updates/analysis/plots
```

The comparison contains absolute accessibility times, exact paired differences
when available, matched-update validation CE, exposure accounting, and the
targets present in each group. The combined timing figure shows absolute times
and paired differences; there is no duplicate delta figure. Layerwise curves
are plotted in observer-layer panels for every available arm, so a
not-confirmed NTP event cannot hide auxiliary movement at another layer.

The controls are optional; when requested, `--controls` adds the trained-backbone
shuffled-label probe.

The comparison contains absolute accessibility times, exact paired differences
when available, matched-update validation CE, exposure accounting, and the
targets present in each group.

## Interpretation and follow-up

The main comparison is the ranking of fixed target depths at the same exposure.
A target that advances H2 but not H3, or one that advances H3 without delaying
H2, would support a depth-specific developmental effect. A common winner or
uniform acceleration is still a useful fixed-target result but gives weaker
motivation for adaptive switching. A probe milestone without persistent
threshold accessibility is reported as a probe result, not as an acquisition
claim.

After the L5 screen is summarized and reviewed, a separate weight-sensitivity
screen or an extended hierarchy can be justified if the scientific question
requires it. Keep those follow-ups small and explicit. Do not add adaptive
target policies until a fixed-target effect is replicated across the intended
seed comparison.
