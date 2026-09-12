# Stage 01 NTP developmental baseline results

**Status:** pending fresh rerun
**Decision:** pending comparison with the pre-run expectation in [README.md](README.md)

The previous generated artifacts and result summary were removed. The current
rerun uses ordinary causal NTP with a fresh training pool at the start of every
epoch. Validation and test pools remain fixed.

## Required runs

Configuration: [`configs/replication.json`](configs/replication.json)

Output: `runs/replication_l5_5000_updates/`

| grammar seed | model seed | best validation CE | best step | H2 A+C onset | H3 A+C onset | decision note |
|---:|---:|---:|---:|---|---|---|
| 0 | 0 | pending | pending | pending | pending | |
| 0 | 1 | pending | pending | pending | pending | |
| 1 | 0 | pending | pending | pending | pending | |
| 1 | 1 | pending | pending | pending | pending | |

The completed report must include the complete layerwise probe, invariance,
and intervention trajectories, with the shuffled-label and untrained-backbone
controls. It must state **go**, **no-go**, or **rerun baseline** against the
expectation before Stage 02 is interpreted as a causal follow-up.
