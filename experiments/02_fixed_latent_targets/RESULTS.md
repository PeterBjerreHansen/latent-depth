# Stage 02 fixed auxiliary target-depth results

**Status:** pending fresh rerun
**Decision:** pending comparison with the pre-run expectation in [README.md](README.md)

The Stage-02 result will use the fresh-training-pool L5 screen and the fixed
balanced-accessibility threshold in [`acquisition_rule.json`](acquisition_rule.json).
The comparison must be made at matched optimizer updates and must retain the
targets actually present in each paired group.

## L5 target-depth screen

| lambda | output |
|---:|---|
| 0.1 | `runs/target_depth_l5_lambda_0_1_5000_updates/` |

Pending summary:

| lambda | target | H2 `tau_accessibility` / `Delta tau_2` | H3 `tau_accessibility` / `Delta tau_3` | `tau_3 - tau_2` | best CE / matched CE |
|---:|---|---:|---:|---:|---:|
| 0.1 | NTP and targets present in the screen | pending | pending | pending | pending |

The final report will include the 50/75/90% probe milestones, layerwise
accessibility curves, optional shuffled-label controls, clustering and `Q` as
supporting diagnostics, and matched validation cross-entropy. If an event is
not observed by the final checkpoint, it will be reported as “not confirmed by
step T”; paired time differences involving an unconfirmed event will be
explicitly unavailable rather than converted into a numeric bound.

Higher auxiliary weights or an extended hierarchy are conditional follow-ups.
They are not part of the current runnable result and should not be interpreted
before the L5 screen is reviewed.
