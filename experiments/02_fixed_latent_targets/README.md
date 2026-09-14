# Stage 02: fixed auxiliary target depth

Stage 02 contains two paired screens. Each compares ordinary NTP with the same
fixed next-latent auxiliary loss at Transformer target layers `0,2,4,8`.
Results are currently cleared so both screens can be rerun from zero.

## Protocol

Use [aux_depth_screen_l5_m4_lambda_1_0.json](configs/aux_depth_screen_l5_m4_lambda_1_0.json)
for the L5/m=4 screen and
[aux_depth_screen_l8_m2_lambda_1_0.json](configs/aux_depth_screen_l8_m2_lambda_1_0.json)
for the L8/m=2 screen.

Both screens use:

- NTP plus fixed residual targets `j=0,2,4,8`, with auxiliary weight 1.0;
- grammar/model seed 0 and eight Transformer blocks, heads, and width 256;
- a fresh 65,536-sequence training pool each epoch and fixed validation/test
  pools of 16,384;
- AdamW with learning rate `3e-4`, existing betas, and gradient clipping;
- 5,000 optimizer updates, with validation and probes every 100 updates,
  including step zero;
- no model checkpoints by default.

The L5/m=4 screen uses batch size 256. The L8/m=2 screen uses batch size 32.
The detached next-position residual stream is the auxiliary target, and grammar
latents are diagnostic labels only.

## Diagnostics

Each config enables linear probes on 16,384 validation sequences (8,192 fit /
8,192 evaluation), with 3,000 fitting steps, learning rate 0.01 and probe seed
12345. Probes run during validation and are stored alongside CE in each arm's
`metrics.json` history. Synonym clustering is disabled for these primary
screens.

## Run both screens

From the repository root:

```bash
set -e
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -u sweep_target_depth.py \
  --config experiments/02_fixed_latent_targets/configs/aux_depth_screen_l5_m4_lambda_1_0.json \
  --output-dir experiments/02_fixed_latent_targets/runs/aux_depth_screen_l5_m4_lambda_1_0_5000_updates
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -u sweep_target_depth.py \
  --config experiments/02_fixed_latent_targets/configs/aux_depth_screen_l8_m2_lambda_1_0.json \
  --output-dir experiments/02_fixed_latent_targets/runs/aux_depth_screen_l8_m2_lambda_1_0_5000_updates
```

After both screens finish, summarize each one:

```bash
.venv/bin/python summarize_target_depth.py \
  --screen-dir experiments/02_fixed_latent_targets/runs/aux_depth_screen_l5_m4_lambda_1_0_5000_updates \
  --rule experiments/02_fixed_latent_targets/acquisition_rule.json \
  --output-dir experiments/02_fixed_latent_targets/runs/aux_depth_screen_l5_m4_lambda_1_0_5000_updates/analysis
.venv/bin/python summarize_target_depth.py \
  --screen-dir experiments/02_fixed_latent_targets/runs/aux_depth_screen_l8_m2_lambda_1_0_5000_updates \
  --rule experiments/02_fixed_latent_targets/acquisition_rule.json \
  --levels 3 4 5 6 7 \
  --output-dir experiments/02_fixed_latent_targets/runs/aux_depth_screen_l8_m2_lambda_1_0_5000_updates/analysis
```

If a run must be restarted, remove its generated run directory and launch it
again; no continuation state is expected.
