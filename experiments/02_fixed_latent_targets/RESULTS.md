# Stage 02 fixed auxiliary target-depth results

**Status:** complete for the active `lambda=0.3` screen
**Decision:** proceed to seed replication before making a depth-specific claim

The Stage-02 result uses the fresh-training-pool L5 screen and the fixed
balanced-accessibility threshold in [`acquisition_rule.json`](acquisition_rule.json).
The comparison must be made at matched optimizer updates and must retain the
targets actually present in each paired group.

## L5 target-depth screen

| lambda | output |
|---:|---|
| 0.3 | `runs/target_depth_l5_lambda_0_3_10000_updates/` |

Summary:

| target | H2 `tau` / `Delta` | H3 `tau` / `Delta` | H4 `tau` / `Delta` | `tau_3 - tau_2` | best validation CE / `Delta` |
|---|---:|---:|---:|---:|---:|
| NTP | 1,250 / 0 | 3,250 / 0 | 6,250 / 0 | 2,000 | 1.400051 / 0 |
| `j=0` | 1,000 / -250 | 2,500 / -750 | 5,250 / -1,000 | 1,500 | 1.398966 / -0.001085 |
| `j=1` | 1,250 / 0 | 2,750 / -500 | 5,750 / -500 | 1,500 | 1.399097 / -0.000954 |
| `j=2` | 1,250 / 0 | 3,000 / -250 | 6,000 / -250 | 1,750 | 1.399560 / -0.000491 |
| `j=3` | 1,250 / 0 | 3,000 / -250 | 5,750 / -500 | 1,750 | 1.399324 / -0.000727 |
| `j=4` | 1,250 / 0 | 3,000 / -250 | 5,750 / -500 | 1,750 | 1.399499 / -0.000552 |
| `j=5` | 1,250 / 0 | 3,000 / -250 | 5,750 / -500 | 1,750 | 1.399143 / -0.000908 |
| `j=6` | 1,250 / 0 | 2,750 / -500 | 5,500 / -750 | 1,500 | 1.399275 / -0.000776 |
| `j=7` | 1,250 / 0 | 3,000 / -250 | 5,750 / -500 | 1,750 | 1.399006 / -0.001045 |
| `j=8` | 1,250 / 0 | 3,000 / -250 | 5,500 / -750 | 1,750 | 1.399001 / -0.001050 |

The primary H2/H3 timing result shows the largest acceleration for `j=0`:
250 updates earlier for H2 and 750 updates earlier for H3. Later targets also
advance H3 and H4 in this seed, but there is no monotone target-depth pattern.
All H4 events are observed within the extended 10,000-update budget. The
best-validation CE values are within 0.0011 of the paired NTP value; the full
matched-update CE curves are in the archived analysis CSV outputs.

The final report includes the 50/75/90% probe milestones, layerwise
accessibility curves, optional shuffled-label controls, clustering and `Q` as
supporting diagnostics, and matched validation cross-entropy. If an event is
not observed by the final checkpoint, it will be reported as “not confirmed by
step T”; paired time differences involving an unconfirmed event will be
explicitly unavailable rather than converted into a numeric bound.

`lambda=1.0` and an extended hierarchy are conditional follow-ups. They are
not part of the current result.
