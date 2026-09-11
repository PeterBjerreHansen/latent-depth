# Stage 00: implementation validation

This stage checks the data generator, causal next-token objective, checkpoint
and resume behavior, device behavior, and optional diagnostics. It is a small
validation suite, not evidence for the latent-depth hypothesis.

Run commands from the repository root. Outputs belong under `runs/`, which is
ignored by Git; the condensed record is [RESULTS.md](RESULTS.md).

## Checks

```bash
python -m pytest -q

PYTORCH_ENABLE_MPS_FALLBACK=0 python validate_implementation.py \
  --output runs/00_validation/implementation_validation/validation.json
```

## Small training and diagnostics

```bash
python train.py \
  --config experiments/00_validation/configs/smoke.json \
  --output-dir runs/00_validation/smoke

python train.py \
  --config experiments/00_validation/configs/diagnostics_smoke.json \
  --output-dir runs/00_validation/diagnostics_smoke

python diagnose.py \
  --checkpoint runs/00_validation/diagnostics_smoke/last.pt \
  --controls \
  --output runs/00_validation/diagnostics_smoke/controls.json
```

## Small sweeps and controls

```bash
python sweep_data_size.py \
  --config experiments/00_validation/configs/data_sweep.json \
  --output-dir runs/00_validation/data_sweep

python sweep_data_size.py \
  --config experiments/00_validation/configs/fixed_exposure.json \
  --output-dir runs/00_validation/fixed_exposure

python sweep_data_size.py \
  --config experiments/00_validation/configs/grammar_replication.json \
  --output-dir runs/00_validation/grammar_replication

python sweep_data_size.py \
  --config experiments/00_validation/configs/saturated_null.json \
  --output-dir runs/00_validation/saturated_null

python plot.py \
  --metrics runs/00_validation/fixed_exposure/metrics.jsonl \
  --config experiments/00_validation/configs/fixed_exposure.json \
  --metric test_last_position_nll \
  --output runs/00_validation/fixed_exposure/last_token_nll.png
```

The sweep writes a simple `sweep_config.json` and one resolved `config.json`
per run. Use `--resume` only for an interrupted run with the same sweep
configuration; start a new output directory for a new experiment.
