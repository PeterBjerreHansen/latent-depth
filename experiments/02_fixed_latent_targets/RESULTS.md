# Stage 02 fixed auxiliary target-depth results

**Status:** pending fresh rerun
**Decision:** pending comparison with the pre-run expectation in [README.md](README.md)

The previous generated artifacts and result summary were removed. The current
developmental screens use fresh training pools at every epoch from the same
grammar, with fixed validation and test pools. The immediate analysis target is
the complete L5/`lambda=0.1` target-depth grid. The larger-weight L5 screens and
the L6/`lambda=1.0` screen are conditional follow-ups, not fresh-data
calibrations that can be interpreted before L5.

Acquisition results must be generated with the frozen rule in
[`acquisition_rule.json`](acquisition_rule.json) and the two summarizer scripts.
No result is entered here until the raw trajectories have been converted to
same-layer A+C onset/confirmation records with explicit censoring.

## L5 target-depth screens

| lambda | output |
|---:|---|
| 0.1 | `runs/target_depth_l5_lambda_0_1_5000_updates/` |
| 0.3 | `runs/target_depth_l5_lambda_0_3_5000_updates/` |
| 1.0 | `runs/target_depth_l5_lambda_1_0_5000_updates/` |

For each screen, report NTP and targets `j=0,...,8` using same-layer A+C
acquisition, layerwise onset matrices, matched-update validation CE versus NTP,
and accessibility, clustering, and `Q` curves. Pending summary:

| lambda | target | H2 `tau_2` / `Delta tau_2` | H3 `tau_3` / `Delta tau_3` | `tau_3 - tau_2` | best CE / matched CE |
|---:|---|---:|---:|---:|---:|
| 0.1 | NTP, `j=0,...,8` | pending | pending | pending | pending |
| 0.3 | NTP, `j=0,...,8` | pending | pending | pending | pending |
| 1.0 | NTP, `j=0,...,8` | pending | pending | pending | pending |

## Conditional extended L6 screen

Configuration:
[`configs/target_depth_screen_l6_lambda_1_0.json`](configs/target_depth_screen_l6_lambda_1_0.json)

Output: `runs/target_depth_l6_lambda_1_0_10000_updates_resampled_train/`

| arm | H2 `tau_2` / `Delta tau_2` | H3 `tau_3` / `Delta tau_3` | H4 onset | `tau_3 - tau_2` | best CE / matched CE |
|---|---:|---:|---:|---:|---:|
| NTP | pending | pending | pending | pending | baseline |
| `target_0` | pending | pending | pending | pending | pending |
| `target_1` | pending | pending | pending | pending | pending |
| `target_2` | pending | pending | pending | pending | pending |
| `target_3` | pending | pending | pending | pending | pending |
| `target_4` | pending | pending | pending | pending | pending |
| `target_5` | pending | pending | pending | pending | pending |
| `target_6` | pending | pending | pending | pending | pending |
| `target_7` | pending | pending | pending | pending | pending |
| `target_8` | pending | pending | pending | pending | pending |

The final report must compare the observed target-depth pattern with the
pre-declared expectations before any adaptive target schedule is considered.
Absolute acquisition times must remain next to the H2-to-H3 interval, and
censored paired differences must be reported as bounds rather than omitted.
