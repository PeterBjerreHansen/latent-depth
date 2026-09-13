# Stage 00 validation results

**Status:** complete

The snapshot refactor was validated on 13 September 2026:

- 79 tests passed, including snapshot noninterference and exact continuation
  through mid-epoch and epoch boundaries with dropout and auxiliary heads.
- Independent data and causal-objective checks passed.
- CPU checkpoint replay was bit-identical.
- CPU/MPS validation CE differed by `4.97e-8`.
- MPS checkpoint replay differed by at most `1.14e-7` in model parameters and
  `2.98e-8` in validation CE, within the declared `1e-4` tolerance.
- The single-run training smoke and three-arm training → model snapshots →
  offline probes → separate C/Q → comparison → plotting workflow passed.

The machine-readable validation is in
`runs/snapshot_refactor/validation.json`; the smoke screen is in
`runs/snapshot_refactor/screen/`. Training now reports validation only; test
measurement is an explicit post-training operation.

The following learning-control results are historical checks from before this
refactor. They were not rerun as scientific experiments in this change.

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
