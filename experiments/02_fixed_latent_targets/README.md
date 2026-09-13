# Stage 02: stronger fixed auxiliary target depth

The current screen establishes whether stronger auxiliary weighting makes
target-depth differences resolvable for later continuation experiments.
The old lambda=0.3 run is retired; its compact factual record is
[retired_lambda_0_3.json](retired_lambda_0_3.json). Its large artifacts are removed.

The six-arm lambda=1.0 screen is complete. See [RESULTS.md](RESULTS.md) for the
findings and [portable evidence](results/lambda_1_0/README.md) for measurements,
checks and figures.

## Frozen training protocol

Use [target_depth_screen_lambda_1_0.json](configs/target_depth_screen_lambda_1_0.json):

- NTP and fixed residual targets j=0,1,2,4,6, with auxiliary weight 1.0.
- RHM v=n=16, m=4, s=2, L=5; grammar/model seed 0.
- Eight blocks, eight heads, width 256, dropout zero.
- Fresh 65,536-sequence pool each epoch; fixed validation/test pools of 16,384.
- Batch 256, AdamW learning rate 3e-4, existing betas and gradient clipping.
- 5,000 updates; validation and model-only snapshots every 100, including zero.
- Full continuation checkpoints every 500 updates, plus the final state.

All arms share initialization and data ordering. The target is the detached
next-position residual stream; the source is the final residual stream at the
preceding position. Grammar latents are diagnostic labels only.

## Frozen diagnostic protocol

The calibration record is [probe_calibration.json](probe_calibration.json).
The initial 1,024/300 budget was materially sensitive to fitting and sample size.
The chosen settings in [probe_settings.json](probe_settings.json) use 16,384
validation sequences (8,192 fit / 8,192 evaluation), 3,000 fitting steps, learning
rate 0.01 and probe seed 12345. These settings agreed closely with much longer
fits at the old learning rate on the checked H3 states. Split-seed uncertainty
remains; grid spacing is not a statistical confidence interval.

The primary [acquisition rule](acquisition_rule.json) uses balanced accuracy
0.75 and two consecutive same-layer observations. On the new 100-update grid,
confirmation spans 100 updates. Report 50/75/90% milestones as sensitivity
summaries, not alternative primary outcomes. Do not treat differences from the
retired 250-update/old-probe screen as a pure effect of lambda.

The primary trajectory is probe-only. Run C/Q and shuffled controls separately
at selected ages. Missing supporting diagnostics are not acquisition failures.
Compare matched-update validation NLL curves; best validation CE is secondary.
Test evaluation is explicit and should follow development/model selection.

## Run one arm through the full workflow, then the rest

From the repository root:

```bash
screen=experiments/02_fixed_latent_targets/runs/target_depth_l5_lambda_1_0_sparse_5000_updates
config=experiments/02_fixed_latent_targets/configs/target_depth_screen_lambda_1_0.json
for arm in ntp target_0 target_1 target_2 target_4 target_6; do
  PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -u sweep_target_depth.py \
    --config "$config" --output-dir "$screen" --arms "$arm" --resume
  run=$screen/grammar_0/model_0/$arm
  PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -u diagnose_trajectory.py \
    --run-dir "$run" --output-dir "$run/trajectory" --device mps \
    --metric probe --num-sequences 16384 --probe-steps 3000 \
    --probe-lr 0.01 --probe-seed 12345
  PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -u diagnose_trajectory.py \
    --run-dir "$run" --output-dir "$run/supporting" --device mps \
    --metric clustering --num-sequences 1024 --steps 0 2500 5000
done
.venv/bin/python summarize_target_depth.py --screen-dir "$screen" \
  --rule experiments/02_fixed_latent_targets/acquisition_rule.json \
  --output-dir "$screen/analysis"
.venv/bin/python plot_trajectory.py --comparison "$screen/analysis/comparison.json" \
  --output-dir "$screen/analysis/plots"
```

Use `set -e` when executing the sequence as a script so a failed arm or diagnostic
stops the screen. Inspect the first arm's metrics and complete probe output
before moving on. Sweep `--resume` skips complete arms; interrupted training can
be continued explicitly with `train.py --resume-from` and its resolved config.

For a selected shuffled control, run `diagnose.py --controls --metric probe`
with the same probe budget into a separate file. Avoid fitting controls at every
age. `records.jsonl` contains raw measurements; trajectory/analysis files are
derived views. Keep compact final evidence and config with the experiment;
model-state files remain local.

## Next decisions

Clear separation with useful NTP learning motivates replication of an
informative shallow/deeper subset on new seeds. Weak separation with reliable
probes and stable training motivates another increase in lambda on a reduced
subset. Disrupted NTP learning motivates an intermediate lambda. Extend the
paired budget only if relevant transitions remain unresolved at the endpoint.

Shared-state continuation experiments follow calibration and replication.
Predictor preparation and optimizer handling must be specified there. Adaptive
policies, freezing, drift controllers and automated parameter searches are deferred.
