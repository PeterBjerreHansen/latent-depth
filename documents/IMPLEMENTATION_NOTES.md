# Implementation notes

## Current baseline

The repository trains a small nanoGPT-style causal Transformer on fixed RHM
leaf sequences. For a token sequence

```text
x_1, x_2, ..., x_T
```

the model receives the full sequence `x_1, ..., x_T` and is trained against
`x_2, ..., x_T` using logits from positions `1, ..., T-1`. Causality prevents
the prediction at position `t` from seeing `x_{t+1}`. This is the only
objective exposed by the configuration and training APIs.

The inference-only forward path projects only the final input position, as in
nanoGPT's autoregressive generation loop. Supplying targets, or requesting
hidden states, returns logits for every position.

## Model correspondence

`nanogpt/model.py` follows the structure of Karpathy's public nanoGPT model:

- learned token and positional embeddings;
- pre-layer-normalized causal multi-head self-attention;
- four-times-width GELU MLP blocks;
- residual dropout and optional bias terms;
- final layer normalization;
- tied token embedding and language-model head;
- AdamW decay/no-decay parameter groups;
- autoregressive generation and block-size cropping.

The local model also returns the embedding stream and post-block hidden states
when `return_hidden=True`. This hook is intended for later latent-depth
experiments and does not alter the baseline loss.

## RHM interface

`input_block_size(cfg)` is always `s**L`, so the complete RHM sequence is
available for hidden-state indexing. `objective_inputs(tokens)` returns the
complete input and its one-position-shifted labels. Evaluation reports both
mean NTP NLL and a per-position NLL vector. Dataset generation, split seeding,
checkpoint selection, and sweep bookkeeping are otherwise independent of the
model.

Evaluation also reports final-position NLL (the closest analogue to the
last-token curves in the RHM literature), the uniform `log(v)` baseline, the
last-token theory bounds when the hierarchy is non-saturated, and cumulative
samples seen. A fixed-exposure sweep can set `samples_per_example` so
different training-set sizes receive matched data exposure rather than a
matched update count.

When `train.eval_every_updates` is set, evaluation and checkpoint selection
use a fixed optimizer-step cadence instead of `eval_every_epochs`. The optional
`train.eval_at_start` flag records the untrained step-zero baseline. Final
top-level `val_*` fields are recomputed on the validation-selected model;
`last_val_*` fields retain the final training-state measurements.

Training is controlled by a fixed `max_updates` budget when one is provided.
The old near-zero training-CE stopping threshold is intentionally absent: full
sequence NTP has irreducible conditional uncertainty and can contain repeated
prefixes with different continuations.

`deterministic_strict=true` switches PyTorch from warning-only deterministic
checks to errors. It is intended for paired continuation experiments where
small numerical differences matter; the default remains warning-only so
hardware-specific attention kernels can be used for baseline runs.

## Representation diagnostics

`diagnostics/latent.py` contains optional observers for the frozen NTP
backbone. The linear probe reads layer 0 (the embedding stream) and each
post-block residual stream. Synonym clustering forces a different production
rule for an on-grammar realization of the same latent and compares that change
with an in-distribution example whose corresponding latent is different.
Variable sensitivity performs the complementary latent-replacement
intervention. Offline diagnostics can additionally fit shuffled-label probes
and probes on an untrained backbone.

Abstraction level `r` is counted upward from the leaves. The implementation
uses `trees[L-r]` and the first constituent completion position `s**r - 1`.
Thus all baseline `A(s,j,r)` and `C(s,j,r)` values refer to the first
completed constituent at each level; they do not silently average over all
constituent positions. The non-synonym reference is a generic control, and the
causal adaptation should not be described as reproducing a specific
non-causal data2vec experiment.

Diagnostics preserve the model's parameters, train/eval mode, and global RNG
states. Probe reports retain the theoretical uniform chance level and also
report empirical majority and balanced-accuracy baselines. `diagnose.py`
regenerates the selected validation/test split from the checkpoint's stored
grammar and provides an offline parity path.

## Checkpoint semantics

`last.pt` records the true final optimizer/model state, while `best.pt` records
the validation-selected model state. Both include the RHM grammar when the
caller supplies it, plus optimizer state, global RNG state, sampler position,
loader RNG state, history, and best-model weights. Periodic evaluation
snapshots use the same self-contained format. A resume request may change only
the update/epoch budget; if it supplies a grammar, it must match the stored
grammar exactly. The sampler stops immediately after the requested update so a
mid-epoch resume does not skip a prefetched batch.

## Scope

This is a causal-Transformer baseline for the project. It should not be read as
a reproduction of papers whose core model is a non-causal encoder, masked
teacher/student system, or other architecture. Those comparisons belong in a
separate experiment configuration and model implementation.
