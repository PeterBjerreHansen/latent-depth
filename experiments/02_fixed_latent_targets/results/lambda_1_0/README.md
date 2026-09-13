# Lambda=1.0 L5 screen evidence

Six paired arms, grammar/model seed 0, 5,000 updates each. See the experiment's
[results report](../../RESULTS.md) for conclusions and qualifications.

This portable evidence bundle retains resolved training configs and metrics,
consolidated raw probe measurements, sparse supporting diagnostics, the actual
grammar, the acquisition rule, frozen probe settings, checks, and derived tables
and figures. It excludes backbone and optimizer states, which remain in the
local run directory. Consolidated trajectory files contain the same records as
the original run's `records.jsonl`; the duplicate files are omitted here.

The `checkpoint` fields identify the original local model artifacts; analysis
uses the saved measurements and does not need to load those model files.

From the repository root, rebuild the registered H2/H3 comparison:

```bash
root=experiments/02_fixed_latent_targets/results/lambda_1_0
.venv/bin/python summarize_target_depth.py --screen-dir "$root" \
  --rule "$root/acquisition_rule.json" --output-dir "$root/analysis"
.venv/bin/python plot_trajectory.py --comparison "$root/analysis/comparison.json" \
  --output-dir "$root/analysis/plots"
```

H4 is a supporting result, not a retrospectively substituted primary endpoint.
For an additional descriptive comparison that includes H4, use
`--primary-levels 2 3 4 --output-dir "$root/descriptive_analysis"` in the
summarizer and plot that separate comparison.

`summary.json` retains events, milestone confirmations, and validation curves.
`analysis/plots/overview.png` shows maximum accuracy across observer layers;
registered event times still require two consecutive passes at the same layer.
The layerwise plots preserve where each effect occurs.

The second-seed checks inspect local crossing windows only. They do not replace
the frozen main trajectories or represent independent grammar/model replication.
