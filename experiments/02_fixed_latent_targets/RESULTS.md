# Stage 02 results: fixed auxiliary target depth

**Status:** not yet run  
**Decision:** pending the fixed-target screen

The runnable protocol is defined in [README.md](README.md), and the initial
screen configuration is
[`configs/target_depth_screen_lambda_0_1.json`](configs/target_depth_screen_lambda_0_1.json).

## Screen summary

Fill this table after running all ten paired arms and the Stage-01 trajectory
diagnostics:

| fixed target | H2 onset `tau_2` | H3 onset `tau_3` | best validation CE | note |
|---|---:|---:|---:|---|
| NTP | | | | |
| embedding (`j=0`) | | | | |
| `j=1` | | | | |
| `j=2` | | | | |
| `j=3` | | | | |
| `j=4` | | | | |
| `j=5` | | | | |
| `j=6` | | | | |
| `j=7` | | | | |
| `j=8` | | | | |

The report should also record the NTP, auxiliary, and total training-loss
curves so an apparent acceleration can be distinguished from globally damaged
language-model optimization. Apply the README's pre-declared 500-update and
`0.01` validation-CE margins without changing them after seeing the results.

## Decision after the screen

Record one of:

- **replicate fixed targets**: at least one auxiliary target advances H2 or H3
  by at least one checkpoint without exceeding the `0.01` validation-CE cost;
- **adjust lambda once**: all target depths are uniformly too weak or uniformly
  harmful, motivating the pre-declared `0.3` or `0.03` follow-up;
- **negative fixed-target result**: after the one justified weight adjustment,
  no target improves the developmental transitions.

Do not implement adaptive switching from this screen alone. Replicate only the
scientifically competitive fixed targets first.
