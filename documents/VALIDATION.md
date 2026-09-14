# Validation

Use the repository environment:

```bash
.venv/bin/python -m pytest -q
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python validate_implementation.py \
  --output experiments/00_validation/runs/implementation_validation/validation.json
```

The tests protect causal indexing, tied embeddings, detached latent targets,
paired initialization, data generation, fresh-pool seeding, exposure accounting,
probe standardization, held-out controls, and accessibility analysis.

Training tests verify that validation probes leave parameters, auxiliary heads,
optimizer and data-stream states, RNG, and NTP measurements unchanged with dropout
and resampling. Online and offline probe scores must agree on the same state.
Exact continuation restores measurement history without duplicate observations.

Default runs must write no model states, even if a checkpoint interval is set.
Opt-in saving supports final-only and periodic states without duplicating the
final state. Failed probes must retain earlier measurements without marking the
run complete. CE-only validation and explicit test evaluation remain supported.

The executable validation matrix checks the data/objective oracle, exact CPU
replay, CPU/MPS agreement, MPS replay, and observer noninterference plus online/
offline agreement on CPU and MPS. CPU equality is exact; MPS checks use a stated
tolerance and verify RNG equality.

The checkpoint-free training → validation probes → paired summary → plot smoke
workflow is documented in `experiments/00_validation/README.md`.
