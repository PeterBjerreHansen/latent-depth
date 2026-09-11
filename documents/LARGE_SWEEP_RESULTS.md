# Large MPS sweep results

The larger causal nanoGPT sweep completed on MPS with CPU fallback disabled.
It contains 36 runs: one grammar seed, three model seeds, and twelve training
pool sizes. Each run used 1,000 optimizer updates and selected the best model
by validation CE.

These artifacts are the historical epoch-cadence run. The current sweep config
also supports fixed update-based evaluation and step-zero baselines; reruns
using it should use a new output directory rather than mixing with these
results.

Raw results:
[`runs/next_token_sweep_large_mps/metrics.jsonl`](../runs/next_token_sweep_large_mps/metrics.jsonl)

Plots:
[`test_ce.png`](../runs/next_token_sweep_large_mps/test_ce.png) and
[`last_token_nll.png`](../runs/next_token_sweep_large_mps/last_token_nll.png)

| P | test CE mean ± SD | final-token NLL mean ± SD |
|---:|---:|---:|
| 128 | 5.006 ± 0.020 | 4.508 ± 0.040 |
| 256 | 4.658 ± 0.021 | 4.267 ± 0.022 |
| 512 | 4.597 ± 0.009 | 4.274 ± 0.010 |
| 1,024 | 4.710 ± 0.012 | 4.606 ± 0.038 |
| 2,048 | 4.844 ± 0.013 | 4.927 ± 0.016 |
| 4,096 | 4.680 ± 0.022 | 4.858 ± 0.075 |
| 8,192 | 3.427 ± 0.009 | 3.587 ± 0.034 |
| 16,384 | 2.681 ± 0.005 | 2.107 ± 0.015 |
| 32,768 | 2.611 ± 0.005 | 1.881 ± 0.024 |
| 65,536 | 2.604 ± 0.012 | 1.849 ± 0.035 |
| 131,072 | 2.599 ± 0.004 | 1.848 ± 0.019 |
| 262,144 | 2.597 ± 0.006 | 1.832 ± 0.021 |

The uniform final-token baseline is `log(32)=3.466`. The large run therefore
shows a pronounced transition between `P=8,192` and `P=16,384`, followed by
continued improvement toward the larger-pool regime. The small-pool runs
overfit strongly under the fixed 1,000-update budget, which explains why their
selected held-out metrics are worse than the uniform baseline and why the
curve is not monotonic at small `P`.

This is evidence for a sharp data-size transition in this fixed-grammar causal
baseline, not evidence that successive latent levels were acquired at distinct
training ages. It is also not a reproduction of the 2025 or 2026 paper
protocols. The run uses a full-sequence causal objective, fixed offline data,
and a nanoGPT-style decoder; the reported final-token metric is provided for
the closest comparison to last-token RHM curves.
