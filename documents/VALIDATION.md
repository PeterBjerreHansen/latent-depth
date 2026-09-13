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

Training tests additionally verify that dense snapshots change neither
parameters nor numerical metrics; resume preserves the model, head, optimizer,
sampler, loader state, running losses and history across mid-epoch and epoch
boundaries; diagnostic snapshots reject resume; substituted grammar files fail
checksum validation; and training evaluates only the validation split.
Offline probes from a full checkpoint and its model-only snapshot must agree.

The executable validation matrix checks the independent data/objective oracle,
exact CPU replay, and tolerance-based CPU/MPS agreement and MPS replay.
Do not interpret CPU bit equality as an MPS guarantee.

The full train → snapshot → offline probes → paired summary → plot smoke
workflow is documented in `experiments/00_validation/README.md`.
