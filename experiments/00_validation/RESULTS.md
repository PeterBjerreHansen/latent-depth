# Validation results

These results were run on an Apple Silicon Mac with PyTorch 2.14.0 and MPS,
using `PYTORCH_ENABLE_MPS_FALLBACK=0` for the MPS runs.

The reported learning and large-sweep artifacts are historical runs from
before fixed update-based evaluation was added to the configs. The current
configs use that schedule for future reruns; existing output directories should
not be resumed or mixed with those reruns.

## Implementation gates

```text
40 tests passed
compileall passed
data expansion oracle passed
shifted-loss reference passed
CPU checkpoint replay passed exactly
CPU/MPS forward and training comparison passed
MPS checkpoint replay passed within 1e-4
```

The machine-readable report is
[`runs/implementation_validation/validation.json`](../../runs/implementation_validation/validation.json).
The CPU/MPS comparison produced a validation-CE difference of approximately
`7e-8` and a maximum parameter difference of approximately `7e-7`. MPS
checkpoint replay produced a maximum parameter difference of approximately
`8e-8` and a validation-CE difference of approximately `1e-8`.

These are numerical comparisons on this PyTorch/macOS installation, not a
claim of bitwise cross-device reproducibility.

## Fixed-exposure learning pilot

Configuration:
[`configs/fixed_exposure.json`](configs/fixed_exposure.json)

This used `v=8,m=2,s=2,L=3`, two model seeds, one grammar seed, and eight
dataset exposures at each training size. The mean held-out results were:

| P | total samples seen | full-sequence test CE | final-token test NLL |
|---:|---:|---:|---:|
| 32 | 128 | 1.696 | 1.423 |
| 64 | 512 | 1.568 | 1.222 |
| 128 | 1,024 | 1.427 | 1.036 |
| 256 | 2,048 | 1.257 | 0.781 |
| 512 | 4,096 | 1.091 | 0.518 |
| 1,024 | 8,192 | 0.953 | 0.248 |

The uniform baseline is `log(8)=2.079`. Both metrics improve monotonically in
this pilot. The results show that the model learns the fixed grammar; they do
not yet establish a scaling exponent or a representation hierarchy.

## Grammar and model-seed replication

Configuration:
[`configs/grammar_replication.json`](configs/grammar_replication.json)

This used three grammar seeds and two model seeds at `P=256` and `P=1024`.
Mean final-token NLL across model seeds was:

| P | grammar 0 | grammar 1 | grammar 2 | all-run mean ± SD |
|---:|---:|---:|---:|---:|
| 256 | 0.781 | 0.750 | 1.155 | 0.895 ± 0.186 |
| 1,024 | 0.248 | 0.311 | 0.576 | 0.378 ± 0.143 |

All three grammars improve with more data, although grammar 2 is visibly harder
at this budget. This is why a single-grammar learning curve would be
insufficient evidence.

## Saturated hierarchy null

Configuration:
[`configs/saturated_null.json`](configs/saturated_null.json)

This used `m=v^(s-1)=8`, where the unambiguous grammar is saturated and the
hierarchical correlation denominator vanishes. Across two grammar and two
model seeds, held-out CE remained approximately `2.08`, close to
`log(8)=2.079`, with no improving final-token signal. The run also verified
that the reporting code correctly omits theory bounds when the hierarchy is
saturated.

## Diagnostics smoke

The tracked `diagnostics_smoke.json` configuration exercises online
layer-by-level probes and synonym clustering on a small CPU run. The same
checkpoint can be passed to `diagnose.py --controls` to verify the shuffled-label
and untrained-backbone controls, and to `diagnose_trajectory.py` for the full
checkpoint trajectory. These are implementation checks, not a developmental
claim.

## Current conclusion

The implementation is supported by independent data, objective, device,
replay, learning, null-control, and diagnostic checks. The current evidence
supports “the causal nanoGPT baseline learns the small fixed RHM grammar” and
does not support a claim that it has already learned the full hierarchy or
reproduced either paper's sample-complexity result. The developmental question
belongs to [Stage 01](../01_ntp_development/README.md).
