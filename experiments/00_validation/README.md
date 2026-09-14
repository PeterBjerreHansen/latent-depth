# Stage 00: implementation validation

These small workflows validate the implementation, not the scientific hypothesis.
Run from the repository root with `.venv/bin/python`.

```bash
.venv/bin/python -m pytest -q
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python validate_implementation.py \
  --output experiments/00_validation/runs/implementation_validation/validation.json

.venv/bin/python train.py --config experiments/00_validation/configs/smoke.json \
  --output-dir experiments/00_validation/runs/smoke

.venv/bin/python sweep_target_depth.py \
  --config experiments/00_validation/configs/target_depth_smoke.json \
  --output-dir experiments/00_validation/runs/target_depth_smoke

.venv/bin/python summarize_target_depth.py \
  --screen-dir experiments/00_validation/runs/target_depth_smoke \
  --rule experiments/00_validation/acquisition_rule.json \
  --output-dir experiments/00_validation/runs/target_depth_smoke/analysis
.venv/bin/python plot_trajectory.py \
  --comparison experiments/00_validation/runs/target_depth_smoke/analysis/comparison.json \
  --output-dir experiments/00_validation/runs/target_depth_smoke/analysis/plots
```

The data-size and fixed-exposure workflows remain available through
`sweep_data_size.py` and their existing configs. They now report final validation
metrics. `plot.py --metric val_last_position_nll` plots the corresponding
last-token measurement. Test evaluation is explicit via `evaluate_snapshot.py`.

Use new output directories for new configurations. Sweep `--resume` skips
completed arms with the same config; `train.py --resume-from` restores an explicitly saved
training state and can extend its update budget.
