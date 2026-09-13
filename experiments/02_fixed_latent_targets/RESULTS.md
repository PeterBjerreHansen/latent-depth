# Stage 02 results

## Current status

The lambda=1.0 six-arm L5 screen is running, beginning with NTP, following
probe calibration and the snapshot refactor. Live run output belongs in
`runs/target_depth_l5_lambda_1_0_sparse_5000_updates/`.
No lambda=1.0 scientific result is claimed until training and diagnostics finish.

## Probe calibration

The original 1,024-sequence / 300-step budget materially underestimated held-out
accessibility relative to larger samples and better fitted probes. Across the
six original H2/H3 transition checkpoints, the stronger 4,096/1,000 setting
raised maximum balanced accuracy by approximately 0.06–0.20.

Further checks used earlier H2/H3 checkpoints, up to the full 16,384-example
validation split. The chosen probe has 8,192 fitting and 8,192 evaluation
examples, 3,000 Adam steps at learning rate 0.01, and seed 12345. On the three
H3 convergence checks, maximum balanced accuracy differed by less than 0.005
from 12,000 steps at learning rate 0.001. A second split seed changed that
maximum by up to approximately 0.025. These are calibration checks, not error
bars for the upcoming trajectory. Borderline conclusions need sensitivity checks.

Full measurement records and the frozen settings are in
[probe_calibration.json](probe_calibration.json) and
[probe_settings.json](probe_settings.json).

## Retired lambda=0.3 screen

The preliminary single-pair screen showed auxiliary acceleration but insufficient
target separation for the intended next experiments. The old fitting budget also
materially affected accessibility measurements. This motivates stronger lambda
and improved probes rather than interpreting the preliminary ranking as final.

The compact original training config, timing table, and matched-update CE values
are preserved in [retired_lambda_0_3.json](retired_lambda_0_3.json). The old run,
its active config, and redundant outputs were removed after calibration.
