# Recommended next runs

## Scope and current context

The upcoming developmental baseline must be evaluated against the
repository-level [baseline expectation and go/no-go gate](expectation.md)
before any latent auxiliary loss is implemented. The baseline's purpose is to
establish a stable, observable developmental clock; a positive NTP curve alone
does not authorize moving to the auxiliary experiment.

This note uses only the requested primary sources: upstream
[nanoGPT `model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py), the
upstream [RHM repository](https://github.com/pcsl-epfl/hierarchy-learning)
([dataset implementation](https://github.com/pcsl-epfl/hierarchy-learning/blob/master/datasets/hierarchical.py)
and [GPT implementation](https://github.com/pcsl-epfl/hierarchy-learning/blob/master/models/gpt.py)),
[arXiv:2505.07070](https://arxiv.org/html/2505.07070v1),
[arXiv:2605.27734](https://arxiv.org/html/2605.27734v1), and official
[PyTorch MPS documentation/source](https://docs.pytorch.org/docs/stable/notes/mps.html)
([MPS Python source](https://github.com/pytorch/pytorch/blob/main/torch/mps/__init__.py)).

For local context, the README and current configs/tests describe a causal,
full-sequence next-token baseline, fixed train/validation/test splits, separate
grammar/data/model seeds, checkpointed optimizer/RNG/sampler state, and causal
diagnostics at first constituent-completion positions. The existing
`next_token_sweep` is therefore a useful starting point, but its fixed grammar,
offline data reuse, and fixed update budget are not the same protocol as the
online curves in arXiv:2505.07070.

## Findings that determine the experiment design

1. **Validate the data generator against the RHM contract before interpreting a
   curve.** The upstream RHM samples `v m` distinct child tuples and partitions
   them into `v` parent groups of size `m`; a tuple has one parent, and each
   level is expanded recursively. Its reference implementation also separates
   rule generation, train/test selection, and within-hierarchy sampling seeds
   ([RHM dataset code](https://github.com/pcsl-epfl/hierarchy-learning/blob/master/datasets/hierarchical.py#L11-L54),
   [#L149-L177](https://github.com/pcsl-epfl/hierarchy-learning/blob/master/datasets/hierarchical.py#L149-L177)).
   A tiny exhaustive run is the fastest way to catch a swapped level, wrong
   child order, invalid rule reuse, or a split that accidentally changes the
   grammar.

2. **Validate next-token alignment independently of the model.** Upstream
   nanoGPT feeds the full causal sequence through the decoder, computes
   cross-entropy on the supplied target sequence, and only shortens the output
   projection at inference time ([nanoGPT `model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py#L155-L175)).
   For this repo, the validation invariant is: input `x=tokens[:, :-1]`, target
   `y=tokens[:, 1:]`, logits at the corresponding input positions, and no loss
   contribution from an unpaired final input. Also retain the causal-prefix
   test: changing a suffix must not change earlier-position logits.

3. **Use both a random baseline and grammar-aware controls.** For a uniform
   visible vocabulary, random-choice NLL is `log(v)`. The 2025 paper defines
   `s^ell`-gram controls for last-token prediction and shows stagewise movement
   between them; it computes the controls from the grammar and averages over 64
   RHM instances, while evaluating the model on 8 independent test
   realisations ([arXiv:2505.07070, §VII.1](https://arxiv.org/html/2505.07070v1#S7.SS1)).
   The repo should report, separately, full-sequence mean NLL, final-position
   NLL, `log(v)`, and exact fixed-grammar conditional controls. Do not compare a
   full-sequence mean directly with the paper's last-token curve.

4. **The data-size sweep must expose offline/online and exposure effects.** The
   2025 paper uses a fresh batch at every step, so samples seen grow linearly
   with steps ([§VI.2](https://arxiv.org/html/2505.07070v1#S6.SS2) and
   [§VII](https://arxiv.org/html/2505.07070v1#S7)). The 2026 paper explicitly
   distinguishes online fresh-sample training from offline reuse of a fixed set
   ([§5](https://arxiv.org/html/2605.27734v1#S5)). A fixed `max_updates` across
   train-set sizes is therefore not, by itself, a sample-complexity curve:
   small sets receive more reuse. Record both `P` and total samples seen, and
   run either a fixed-exposure offline curve or a separate online control.

5. **Use paper thresholds as anchors, not as predictions for this exact model.**
   For token-level learning, the 2026 paper gives a first latent-recovery scale
   of order `v m^3` and higher-level scales of order `v m^(ell+2)`; the 2025
   paper gives transformer transition scales of order
   `(1-f)^(-1) v m^(2 ell+1)`, with `f=m/v^(s-1)`
   ([arXiv:2605.27734, §2](https://arxiv.org/html/2605.27734v1#S2) and
   [arXiv:2505.07070, §V.3](https://arxiv.org/html/2505.07070v1#S5.SS3)).
   For the current `v=32,m=8,s=2,L=3` sweep, useful log-spaced anchors are
   `P={256, 1024, 4096, 16384, 65536, 131072, 262144}`; the `v m^3` and
   `v m^4` anchors are 16,384 and 131,072. These are order-of-magnitude
   waypoints, not a claim that this smaller MLP-equipped causal decoder will
   reproduce either paper's exponent.

6. **Representation diagnostics need matched interventions, not probes alone.**
   The 2025 paper distinguishes replacing a hidden variable, which changes its
   whole subtree, from replacing only its production rule, which preserves the
   hidden symbol; it standardises hidden activations and measures cosine
   invariance. Sensitivity to variable replacement plus invariance to rule
   replacement is the relevant signature ([§VIII](https://arxiv.org/html/2505.07070v1#S8)).
   The repo's synonym counterfactual should be paired with a same-level latent
   replacement and a marginal-matched unrelated negative. Evaluate at the
   causal completion position for each level, and at every residual stream.
   Linear-probe accuracy is evidence that a latent is accessible; it is not
   evidence that the model used that latent for prediction.

7. **The latent-prediction paper is a control objective, not an interpretation
   of this baseline.** It studies predictor-clusterer modules and data2vec-style
   teacher targets, with a teacher EMA and latent targets ([arXiv:2605.27734,
   §4](https://arxiv.org/html/2605.27734v1#S4) and [§5](https://arxiv.org/html/2605.27734v1#S5)).
   Its `v m^3` depth-independent result is relevant as a future comparison, but
   a causal next-token run cannot establish latent-prediction sample efficiency.
   The paper's own limitations also restrict the result to fixed-topology,
   unambiguous, non-recursive RHM grammars ([§Limitations](https://arxiv.org/html/2605.27734v1#S6)).

8. **Treat MPS as a separate numerical-validation target.** PyTorch documents
   that reproducibility is not guaranteed across releases or platforms and
   recommends explicit seeding and deterministic-algorithm checks where
   available ([randomness note](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md#reproducibility)).
   The MPS source exposes a device RNG state, `manual_seed`, and an explicit
   `synchronize()` that waits for all MPS kernels ([MPS source](https://github.com/pytorch/pytorch/blob/main/torch/mps/__init__.py#L752-L809)).
   Therefore compare CPU and MPS numerically within a documented tolerance,
   synchronize before copying metrics/checkpoints to CPU, and test MPS
   checkpoint replay with the MPS RNG state included in the recorded state. Run
   once with `PYTORCH_ENABLE_MPS_FALLBACK=0` to expose unsupported operations;
   the official environment-variable documentation says that setting `1`
   enables CPU fallback ([MPS environment variables](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)).

## Recommended minimal run matrix

The matrix below is intentionally small. Reuse the same fixed split and saved
grammar within a row; never regenerate a grammar when changing only model seed.
Use `dropout=0`, `num_workers=0`, fixed batch size, and strict deterministic
mode for validation. For the learning curves, log checkpoints at steps
`0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000` (or the nearest available
evaluation points), not only the best checkpoint.

| ID | Purpose | RHM / data | Model and budget | Seeds / outputs |
|---|---|---|---|---|
| V1 | Exhaustive data/indexing oracle | `v=n=4,m=2,s=2,L=2`; enumerate all root/rule-choice paths for one grammar | No training; compare stored trees, choices, leaves, valid-tuple uniqueness, and per-level shapes against an independent enumerator and the upstream RHM contract | Grammar seeds `0,1`; save a compact JSON digest of rules, first samples, counts, and split disjointness |
| V2 | Objective/causality/reference forward | `v=8,m=2,s=2,L=2`, fixed 32-example batch containing hand-checkable token sequences | One initialization and one update on CPU; verify shifted-target loss manually, per-position NLL length, causal-prefix invariance, tied embedding/output weights, and (if practical) logits against an independently loaded upstream nanoGPT-equivalent forward | Model seeds `0,1`; require finite values and exact CPU replay |
| V3 | Checkpoint and sampler replay | Same tiny setup as V2; fixed train/val/test tensors | Uninterrupted 4 updates vs. 2 updates + resume to 4; compare model, optimizer, sampler/order, RNG, validation history, and final per-position NLL | One grammar seed, model seed `0`; repeat on MPS if available, with `torch.mps.synchronize()` before reads and explicit MPS RNG capture |
| V4 | MPS backend smoke | `v=16,m=4,s=2,L=3`; use `configs/next_token_smoke.json` scale or smaller | CPU and MPS forward, one-update gradient, and 20-update train/replay; run with fallback disabled, then record a fallback-enabled comparison only if needed for diagnosis | Same grammar/model/data seeds; report max logit delta, loss delta, parameter delta, wall time, PyTorch/macOS/device versions |
| C1 | Learning/control curve | Current `v=32,m=8,s=2,L=3`; `P={256,1024,4096,16384,65536,131072,262144}`; same val/test sizes | `n_layer=3,n_head=8,n_embd=256`, `dropout=0`; 1000 updates at fixed batch size, plus a fixed-exposure variant; report full-sequence and final-position curves | First pass: grammar seed `0`, model seeds `0,1,2`. Include `log(v)`, exact grammar conditional controls, and an iid-token control with matched marginals |
| C2 | Grammar/seed replication | Two anchor sizes around transitions: `P=16384` and `131072` | Same budget as C1; do not tune separately per seed | Grammar seeds `0,1,2` × model seeds `0,1` at each anchor; report mean, standard deviation, and per-grammar curves |
| D1 | Representation dynamics | Reuse C1 checkpoints for grammar seed `0`; diagnostic set at least 2048 held-out sequences | At steps `0,16,64,256,1000`, run linear probes and matched interventions for levels `r=1,2` (and `r=3` if `L` is increased); use a fixed probe budget and separate probe seed | Report probe accuracy/CE by stream, `q_rule`, `q_variable`, synonym-vs-negative clustering, and NTP loss on the same checkpoints |
| D2 | Diagnostic robustness | Reuse C2 anchor checkpoints | Repeat D1 for the two anchor sizes; include shuffled latent labels and untrained-backbone controls | Two diagnostic/probe seeds; no gradient or RNG effect on the backbone; keep the held-out diagnostic set fixed |

### Practical budget choices

- For validation, prefer a few updates with exact replay over a long run.
- For C1, the existing 1000-update budget is adequate to reveal whether the
  curve is flat, stagewise, or numerically unstable, but it is not evidence of
  convergence. Add a longer continuation only after V1--V4 pass.
- For D1/D2, probe only saved checkpoints. A fixed linear-probe budget of
  300--1000 optimizer steps is enough for a first diagnostic; increase it only
  after the shuffled-label control stays at chance. The 2026 paper used much
  larger models and 2000 probe steps in its data2vec study, so those settings
  should not be transplanted uncritically to this small decoder
  ([Appendix D](https://arxiv.org/html/2605.27734v1#A4)).
- Keep grammar seed, train/validation/test data seeds, model seed, probe seed,
  `P`, batch size, updates, samples seen, device, PyTorch version, and fallback
  mode in every run record. Three independent grammar realisations are the
  minimum sensible replication for a curve; the 2026 paper likewise averages
  key results over three independent RHM instantiations
  ([Appendix C.4](https://arxiv.org/html/2605.27734v1#A3.SS4)).

## Claims the matrix can support

After V1--V4 pass, C1/C2 can support: “under this fixed RHM parameterisation,
this causal nanoGPT-style decoder exhibits [or does not exhibit] stagewise
next-token learning, with the stated dependence on train-set size, exposure,
grammar seed, and model seed.” D1/D2 can support: “the frozen residual stream
contains linearly accessible latent labels and shows [or does not show]
rule-invariance/variable-sensitivity at the measured causal completion
positions.”

The matrix cannot, by itself, support any of the following:

- a reproduction of the 2025 paper's scaling exponent, because the repo uses a
  full-sequence causal objective and fixed offline data, whereas that paper's
  transformer curves use last-token prediction, online batches, and grammar-
  averaged controls ([§VI.2](https://arxiv.org/html/2505.07070v1#S6.SS2),
  [§VII.1](https://arxiv.org/html/2505.07070v1#S7.SS1));
- the 2026 paper's latent-prediction sample-complexity result, because this
  repo does not train a teacher-student latent target or a predictor-clusterer
  objective ([§4--§5](https://arxiv.org/html/2605.27734v1#S4));
- a claim that probe accuracy proves the model uses a latent, or that a single
  grammar generalises across grammars; or
- bitwise CPU/MPS equivalence across devices or PyTorch releases, which PyTorch
  explicitly does not guarantee ([randomness note](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md#reproducibility)).

The recommended order is therefore **V1--V4 → C1 → C2 → D1/D2**. Stop and
repair the relevant implementation seam if a validation run fails; do not use
the learning-curve or representation results to diagnose a failing oracle.
After the developmental screen and confirmation runs, compare the complete
trajectory with [`expectation.md`](expectation.md) and record **go**,
**no-go**, or **rerun baseline**. Implement the latent auxiliary loss only
after an explicit **go** decision.
