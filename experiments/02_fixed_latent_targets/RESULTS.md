# Stage 02 results

## Completed lambda=1.0 screen

**Status:** all six arms, 306 probe measurements and 18 supporting diagnostic
measurements completed. Integrity checks passed. No additional training
experiment has been launched.

Training used grammar/model seed 0, the frozen L5 configuration, and 5,000
updates per arm: 1,280,000 sequence draws and 39,680,000 prediction tokens.
Every arm has 51 snapshots/probe records at steps 0..5,000 by 100, with the
same 16,384-sequence / 3,000-step / learning-rate-0.01 probe configuration.
Initial backbone weights and grammars match exactly across arms; final sampler
states match; final diagnostic snapshots match full continuation backbones.
The complete training/diagnostic pipeline took approximately 3 hours 4 minutes.

Portable evidence is in [results/lambda_1_0](results/lambda_1_0/README.md).
The [overview figure](results/lambda_1_0/analysis/plots/overview.png) shows
accessibility curves and matched-update validation CE differences. The
[summary](results/lambda_1_0/summary.json) retains all level events and milestones.

| Target | H2 onset | H3 onset | H3 difference vs NTP | H4 onset (supporting) | H3→H4 interval | Validation CE at 5,000 |
|---|---:|---:|---:|---:|---:|---:|
| NTP | 0 | 2,000 | 0 | not confirmed by 5,000 | unavailable | 1.443452 |
| j=0 | 0 | 1,400 | −600 | 4,200 | 2,800 | 1.435348 |
| j=1 | 0 | 1,700 | −300 | 4,800 | 3,100 | 1.438046 |
| j=2 | 0 | 1,600 | −400 | 4,800 | 3,200 | 1.437304 |
| j=4 | 0 | 1,600 | −400 | 4,000 | 2,400 | 1.435715 |
| j=6 | 0 | 1,700 | −300 | 3,900 | 2,200 | 1.435390 |

Onsets follow the unchanged registered 0.75 threshold with two consecutive
same-layer observations on the new 100-update grid. Confirmation occurs 100
updates after each listed onset. NTP's unavailable H4 time is not assigned a
numeric bound or used to manufacture an onset difference. All arms' best
validation measurement occurs at step 5,000.

## Interpretation

The screen resolves a useful contrast: j=0 reaches H3 earliest, while j=6
reaches H4 earliest and j=4 follows closely. The H3→H4 interval is 2,200 updates
for j=6 versus 2,800 for j=0. This is a descriptive timing contrast between
whole-run treatments, not a shared-state continuation-value crossover.

The H4 curves support the timing result: final maximum observer-layer balanced
accuracy is 0.9146 for j=6, 0.8938 for j=0 and j=4, and 0.7476 for NTP. However,
the 50% H4 milestone favors j=0 (2,400) over j=6 (2,500), while the 75% milestone
favors j=6. The difference concerns the later part of H4 development rather
than uniform deep-target superiority. No H4 90% milestone is confirmed within
the saved horizon; exceeding 90% only at the final point is insufficient.

At step 3,000, j=0 validation CE is 1.458786 and j=6 is 1.466052, compared with
NTP's 1.487987. By step 5,000, j=0 and j=6 differ by only 0.000041 CE, both
about 0.0081 below NTP. Thus the deeper arm's H4 advantage does not imply that
it is a better NTP training recipe over the entire run. Comparisons match
updates/exposure, not wall-clock compute. Final Q is positive at all observed
H3/H4 event layers in the sparse supporting diagnostics.

H4 was supporting in this screen. H2/H3 were the registered primary levels;
the H4 finding is worth confirming with a prospectively declared endpoint.
This is one grammar/model pair. The new probe budget and temporal grid also
prevent interpreting a difference from the retired lambda=0.3 table as a pure
lambda effect.

## H2 and diagnostic checks

H2 is already nearly perfectly decodable from the untrained backbone: maximum
balanced accuracy is 0.9984 at step zero. At the earliest passing observer
layer, k=1, it is 0.9821. Every arm retains some observer layer above 0.90
throughout the saved trajectory. Consequently, H2 onset is zero under the
frozen rule and cannot measure developmental timing in this setting.

The step-zero shuffled-label H2 control ranges from 0.0582 to 0.0623 across
layers, close to the 1/16 reference. H3 and H4 real-label maxima at step zero
are only 0.2398 and 0.1119. This supports interpreting H2 as accessibility in
the random backbone rather than training-induced acquisition. It does not
justify changing the threshold after looking at these results.

Seventeen additional measurements with probe seed 12346 checked local H3/H4
crossing windows on the same backbones and validation pool. Each window includes
a point below threshold followed by two passing observations at the same layer.
The [sensitivity record](results/lambda_1_0/checks/probe_seed_sensitivity.json)
and raw measurements are separate from the frozen main trajectories.

| Arm / level | Primary seed 12345 onset | Seed 12346 local onset |
|---|---:|---:|
| NTP / H3 | 2,000 | 2,100 |
| j=0 / H3 | 1,400 | 1,400 |
| j=6 / H3 | 1,700 | 1,700 |
| j=0 / H4 | 4,200 | 4,100 |
| j=6 / H4 | 3,900 | 3,900 |

The shallow/deeper ordering survives this check. The H4 gap shrinks from 300
to 200 updates, while the H3 gap remains 300. These are local checks, not full
second-seed trajectories or independent grammar/model replications; they do
not establish the absence of earlier crossings outside the checked windows.

## Recommended next experiment

Keep L5 and lambda=1.0. Replicate an informative NTP/j=0/j=1/j=6 subset on new
grammar/model seeds, prospectively using H3 and H4 accessibility alongside
matched-update NTP curves. Retain H2 as an initialization/control measurement.
The present screen provides a concrete shallow/deeper contrast to confirm;
another lambda or a deeper grammar need not precede that confirmation.

If the contrast survives, the next causal experiment should branch shallow and
deep targets from shared early/late states, including a shallow-trained history.
No replication, new lambda, branch training, or controller was started here.

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
bars for the full trajectory. Borderline conclusions need sensitivity checks.

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
