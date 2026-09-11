# Agent coding policy

## Optimize for the final research repository

This project is actively being reshaped. Optimize for a clean, simple codebase
that exposes the experiments we ultimately intend to run and report.

- Backward compatibility is not a goal unless the user explicitly requests it.
- Breaking old command paths, config formats, output formats, or internal
  interfaces is acceptable when the replacement is simpler and clearer.
- Historical runs, intermediate sweeps, obsolete configs, and bookkeeping may
  be deleted when they are not part of the eventual core experiments.
- Do not preserve machinery merely because it was useful for an earlier phase.
  Prefer removing it and fixing the remaining callers.
- Prefer a small number of direct, stage-specific runnable workflows over a
  universal experiment framework or compatibility layer.

## Make focused architectural choices

- Keep reusable modules generic only when they already have a real reuse case.
- Prefer deep modules with small interfaces and clear responsibilities.
- Avoid speculative abstractions, duplicate configuration systems, and manual
  bookkeeping that does not materially improve the experiments.
- Organize code and experiment documentation around the final experiment ladder
  rather than around every historical development step.

## When changing the repository

- Inspect the current callers, tests, configs, and documents before moving or
  deleting anything.
- When a refactor changes paths or interfaces, update all remaining callers and
  documentation in the same change. Do not add compatibility shims unless they
  make the final design simpler.
- Keep only the validation and tests that protect the current intended design.
- Run the full test suite and the relevant experiment smoke command after code
  changes. Fix failures caused by the refactor rather than restoring old
  behavior solely for compatibility.
- Use the simplest provenance and output bookkeeping that the scale of the
  experiment actually needs. Reproducibility of the final runnable workflows
  matters; preserving every historical run does not.
