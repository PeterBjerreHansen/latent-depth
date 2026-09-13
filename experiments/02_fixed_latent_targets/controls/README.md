# Stage-02 controls: recoverability versus abstraction

These controls test the interpretation of the step-zero H2 result. They do
not train a Transformer and do not change the Stage-02 acquisition rule. The
main question is whether a frozen random feature map makes the grammar label
easy for a supervised decoder, while synonym invariance is still absent.

## Frozen protocol

The default protocol uses the Stage-02 L5 architecture and grammar, the first
16,384 validation derivations, a fixed 8,192-example evaluation partition, and
nested class-balanced fitting subsets of 64 through 8,192 examples. H2 uses
the first four visible leaves and H3 the first eight. Linear probes use the
main experiment's 3,000 Adam steps at learning rate 0.01. Model seed 0 gets
the full sample-complexity curve; model seeds 1--9 replicate the 1,024- and
8,192-example points. Probe seeds measure fitting variability.

The controls include:

- linear probes on frozen random Transformer features;
- exact visible-prefix lookup, additive one-hot linear, and fixed MLP surface
  baselines;
- a privileged exact grammar lookup from the true child latent pair;
- shuffled-label probes as a probe-capacity negative control.

The optional invariance run repeats synonym and latent-replacement C/Q
measurements at saved step-zero and step-5,000 snapshots. It requires the
local Stage-02 model-only snapshots, which are intentionally not committed.

## Run

From the repository root:

```bash
.venv/bin/python experiments/02_fixed_latent_targets/controls/run.py \
  --config experiments/02_fixed_latent_targets/controls/config.json \
  --output-dir experiments/02_fixed_latent_targets/controls/results/lambda_1_0

.venv/bin/python experiments/02_fixed_latent_targets/controls/summarize.py \
  --input experiments/02_fixed_latent_targets/controls/results/lambda_1_0/raw_controls.json \
  --output-dir experiments/02_fixed_latent_targets/controls/results/lambda_1_0/summary
```

Add `--include-invariance` to the first command after the referenced local
snapshots exist. A small CPU smoke configuration is provided in
`config_smoke.json`.

Only compact JSON/CSV summaries and plots belong in this directory's committed
results. Do not commit checkpoints, hidden features, or generated tensor
captures.

## Interpretation

These are controls, not replacement outcomes. High step-zero probe accuracy
does not establish an abstract representation. C/Q is a geometric invariance
diagnostic under specified interventions, not a complete information measure.
Any abstraction-acquisition threshold must be declared prospectively after
these estimator checks.
