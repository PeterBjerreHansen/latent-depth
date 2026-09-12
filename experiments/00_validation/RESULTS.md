# Stage 00 validation results

**Status:** complete

The validation protocol was rerun from the current code after the previous
generated artifacts were removed.

## Required checks

- `pytest -q`
- `validate_implementation.py`
- smoke training and diagnostics
- data-size, fixed-exposure, grammar-replication, and saturated-null sweeps

The test suite passed with 80 tests, including the fixed accessibility-analysis
tests. The implementation validation also passed
the data oracle, shifted-loss reference, CPU checkpoint replay, CPU/MPS device
comparison, and MPS checkpoint replay:

| check | result |
|---|---|
| data oracle | passed |
| causal objective | passed |
| CPU checkpoint replay | passed exactly |
| CPU/MPS validation comparison | passed; CE difference `9.9e-9` |
| MPS checkpoint replay | passed; max parameter difference `1.7e-7` |

## Learning controls

The fixed-exposure sweep improves monotonically with more data. Mean final-token
test NLL across two model seeds falls from `1.255` at `P=32` to `0.248` at
`P=1,024`.

The grammar replication also improves with more data:

| P | final-token test NLL, mean ± SD |
|---:|---:|
| 256 | `0.895 ± 0.186` |
| 1,024 | `0.378 ± 0.143` |

The saturated null remains at the uniform baseline. Across four arms, test CE
is about `2.081` and final-token NLL about `2.084`, close to `log(8)=2.0794`.

The data-size sweep shows the expected finite-size behavior: small pools
(`P=128`–`512`) overfit badly within 1,000 updates, while larger pools improve
steadily. At `P=262,144`, mean best validation CE is `2.597` and mean final-
token test NLL is `1.832`.

These are implementation and control results, not evidence for the
latent-depth hypothesis. The developmental result is in [Stage 01](../01_ntp_development/RESULTS.md).
