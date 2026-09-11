# Provenance and licensing

## nanoGPT model

`nanogpt/model.py` is adapted from the public model structure in
[Karpathy/nanoGPT](https://github.com/karpathy/nanoGPT), especially its
`model.py` implementation. The local version keeps the core causal-Decoder
design and adds project-specific configuration validation, hidden-state return
hooks, and a small training API.

The upstream MIT license is included at
[`third_party/NANOGPT_LICENSE`](../third_party/NANOGPT_LICENSE).

## Project code

RHM generation, dataset construction, theory helpers, configuration, training,
checkpointing, plotting, implementation validation, representation diagnostics,
and tests are project code. They are not presented as part of nanoGPT or as an
exact reproduction of any external paper's model.

## Research references

The conceptual documents in this repository discuss hierarchical prediction and
related literature. The executable baseline intentionally remains explicit:
fixed RHM sequences, a causal next-token objective, and a nanoGPT-style
Transformer. The optional latent probes, synonym-clustering score, and
latent-replacement sensitivity controls are project-specific causal adaptations
inspired by the cited representation analysis, not claims of architectural or
experimental reproduction. The shuffled-label and untrained-backbone controls
are implementation checks rather than paper-reproduction claims.
