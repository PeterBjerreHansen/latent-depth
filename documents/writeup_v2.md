# Learning to Predict Deeper

*A developmental curriculum for autoregressive latent self-supervision*

*Revised conceptual proposal and experimental plan — 10 September 2026*

## Abstract

Deep networks progressively construct representations that make structured properties of their inputs accessible to prediction. This suggests that the representations most useful as self-supervised targets should change during learning. We propose to test a developmental curriculum for autoregressive models: maintain next-token prediction throughout training, while moving auxiliary supervision from shallow contextual representations toward deeper representations as their predictive structure becomes useful to the learner. The central hypothesis is a crossover in marginal target value, driven by the emergence of task-relevant predictive features and diminishing returns from already acquired features. We connect this hypothesis to empirical work on progressive latent prediction in vision, ordered layer convergence, and linear representations of concepts. A simple feature-mode model explains how a crossover can arise and identifies quantities that should predict it. Experiments on ordinary causal Transformers trained on hierarchical synthetic languages will measure target value, identify its contributing feature modes, and test interventions that advance or delay the crossover. A practical controller will estimate marginal value with short validation lookahead and allocate auxiliary weight accordingly. The intended result is a demonstrated principle for when to advance latent supervision, supported by both causal experiments and complete training runs at matched compute.

## 1. The motivating hypothesis

The proposal begins with a constructive empirical bet:

> **As a model learns hierarchical structure, deeper contextual targets become better auxiliary teachers because they expose predictive features that the learner has not yet fully acquired. Allocating supervision according to this changing opportunity should improve training over keeping the target fixed.**

There are two developmental processes in this statement. The teacher representation becomes more informative about structured features. Meanwhile, the learner becomes better at predicting features that earlier targets already exposed. Their interaction creates a reason to change the target: a useful target should offer something both available and still worth learning.

Consider a grammar in which local token patterns identify a constituent, and combinations of constituents reveal a broader category that governs subsequent material. Early in training, local regularities are the most reliable self-supervised signal. Later, a deeper representation may express the broader category in a form that an auxiliary prediction head can directly supervise. Continuing to emphasize local features may then offer less benefit than predicting that newly available structure. The architecture need not receive constituent boundaries or category labels. They are properties to be discovered through training and inspected afterward.

The prediction concerns a movement from shallow toward deeper **contextual** representations over a meaningful training interval. It does not require the last Transformer block to become the best target or the preferred depth to increase at every checkpoint. The final blocks may specialize toward the token output. But the hypothesis does require a reproducible shallow-to-deeper crossover; a consistently fixed optimum would count against this developmental explanation in the tested setting.

This restores the original motivation for the project. The mathematical analysis should explain a useful empirical regularity and guide a training signal. Counterexamples establish the scope of that analysis and belong mainly in an appendix.

## 2. Why this is a reasonable empirical prior

### 2.1 Progressive targets and ordered convergence

LayerLock supplies a direct precedent in vision. It observes ordered convergence in video masked-autoencoder layers and progressively freezes lower blocks while advancing the prediction target to deeper activations. The authors report benefits on semantic and dense prediction tasks, and also study application to V-JEPA. Its freezing and target progression operate together; our first language experiments will therefore compare target advancement with and without freezing to identify which part transfers. [Erdogan et al., 2025](https://arxiv.org/html/2509.10156v3)

Ordered convergence has earlier support in SVCCA analyses, which found that networks in the studied settings approach their final representations from the bottom up. This motivates measuring when a representation has become sufficiently settled to provide sustained supervision. Stability is an experimentally motivated baseline signal, rather than a requirement that a target must stop changing completely. [Raghu et al., 2017](https://arxiv.org/abs/1706.05806)

In autoregressive language modeling, NITP reports strong results from shallow contextual targets, with its tested depth sweeps favoring a region near one fifth of network depth. A successful curriculum must improve on that strong fixed-target recipe. The open question is whether a target selected for an entire training run remains best at each stage, particularly after a shallow-target training history. [Zhang et al., 2026](https://arxiv.org/html/2605.24956v1)

### 2.2 Depth supplies successive computations

The original motivation for depth is relevant: compositions of transformations can efficiently express structured functions. Expressivity analyses make depth-dependent increases in computational complexity precise for particular network families. In trained language models, probing studies also find ordered locations for different linguistic computations; BERT's observed progression from local syntactic information toward more relational tasks is a concrete example. [Raghu et al., 2017](https://arxiv.org/abs/1606.05336), [Tenney et al., 2019](https://arxiv.org/abs/1905.05950)

The relevant notion of complexity is the **structure made accessible by successive computation**. It need not mean increasing activation dimension or entropy. Intrinsic-dimensionality studies find expansion followed by compression, while recent language-model analyses find that intermediate representations can outperform final representations on embedding tasks. These observations suggest a useful contextual region through which a curriculum might advance, followed by an output-specialized region. [Ansuini et al., 2019](https://arxiv.org/abs/1905.12784), [Skean et al., 2025](https://arxiv.org/abs/2502.02013)

We will measure three separate empirical axes: hierarchy level, the network depth where that structure becomes accessible, and training age. Their relationship is the phenomenon under investigation.

### 2.3 Linear features provide a way to study the mechanism

The linear representation hypothesis gives a useful operational view: concepts can be expressed through directions or low-dimensional subspaces of a representation. Park, Choe, and Veitch connect concept representations to probing and interventions under an appropriate geometry. Their subsequent work studies categorical and hierarchical concepts, with experiments on Gemma and LLaMA-3. These results motivate identifying feature subspaces and testing their effects rather than treating hidden states as uninterpreted vectors. [Park et al., 2024](https://arxiv.org/abs/2311.03658), [Park et al., 2025](https://arxiv.org/html/2406.01506v3)

There is also a link to the training objective: Jiang et al. show in a latent-variable model how next-token prediction and gradient-descent bias promote linear concept representations, with supporting experiments. That makes linear structure a plausible consequence of autoregressive training, rather than solely an analysis convenience. [Jiang et al., 2024](https://arxiv.org/abs/2403.03867)

Our extension is explicitly empirical: **do intermediate layers increasingly expose useful hierarchical features as predictable supervision, and does that explain when those layers should receive auxiliary weight?** Existing concept-geometry results principally concern the final embedding/unembedding interface. Applying the idea to intermediate layers and their learning trajectories is part of the proposed work.

### 2.4 Hierarchical data and developmental learning

Random Hierarchy Model analyses connect the acquisition of hidden grammatical variables to the statistical resolution of increasingly long-range correlations. Latent-prediction theory supplies a mechanism by which learned intermediate variables can expose those correlations more efficiently. These works justify using hierarchical grammars as a controlled test of developmental self-supervision. [Cagnetta and Wyart, 2024](https://arxiv.org/abs/2406.00048), [Korchinski et al., 2026](https://arxiv.org/abs/2605.27734)

Deep-linear learning theory offers another constructive precedent. Saxe, McClelland, and Ganguli derive developmental transitions in the acquisition of semantic structure from mode-dependent learning dynamics. This supports studying the interaction of signal strength and learning stage. Their ordering of semantic distinctions is not automatically the same as bottom-up recovery in an RHM; the data statistics determine which modes emerge first. [Saxe et al., 2019](https://arxiv.org/abs/1810.10531)

Together, these literatures motivate a research hypothesis. The combination of ordered convergence, hierarchical computation, and linear feature structure is strong enough to warrant a direct experiment on progressive target weighting.

## 3. From target features to a developmental opportunity

Let $C=X_{\leq t}$ be a prefix, $Y=X_{t+1}$ the next token, and $s$ a training checkpoint. A causal Transformer supplies a final source state $S_\theta(C)$ and a detached target from block $j$ at the next position. Write the normalized target as $Z_j(C,Y)$ and the normalized predictor as $U_j(C)$. The main training objective is

$$
\mathcal L=\mathcal L_{\mathrm{NTP}}+
\sum_{j\in\mathcal J}\lambda_j(s),
\mathbb E[1-U_j(C)^\top Z_j(C,Y)].
$$

The candidate set consists of ordinary network layers. Ground-truth grammar variables do not enter this loss or the model architecture. Targets use stop-gradient; statements about an individual update condition on a fixed teacher.

### 3.1 Predictive exposure

Define

$$
m_j(C)=\mathbb E[Z_j(C,Y)\mid C].
$$

For cosine prediction, the population loss is exactly

$$
\mathcal L_j=1-\mathbb E[U_j(C)^\top m_j(C)].
$$

Thus $m_j$ is the coherent target signal exposed to prediction from the prefix. On a small RHM, it can be computed by enumerating next tokens under the exact grammar conditional. This lets us study predictive content directly, without estimating it from the auxiliary loss alone.

Let $\Pi_{j,r}$ denote a fixed, independently estimated projection associated with hierarchy level $r$ in target layer $j$. With $\widetilde m_j=m_j-\mathbb E[m_j]$, define

$$
P_{j,r}(s)=\mathbb E\|\Pi_{j,r}\widetilde m_j(C)\|_2^2.
$$

Centering prevents a constant target offset from being counted as a predictive distinction. The project hypothesizes that deeper layers eventually expose higher-level distinctions through larger, more selective $P_{j,r}$. These are layer-dependent learned subspaces: knowing a grammar label does not supply its direction in a neural representation.

A simple example illustrates the purpose of this measurement. Let the prefix determine a grammatical feature $A\in\{-1,+1\}$, and let the next token contain an independent fair-sign variation $Y$. For

$$
Z_\gamma=(\gamma A,\sqrt{1-\gamma^2},Y),
\qquad 0<\gamma\leq1,
$$

the grammatical feature is perfectly decodable for every positive $\gamma$, but its predictive exposure is $\gamma^2$. Increasing $\gamma$ makes that feature a stronger component of the supervision. This is the one counterexample retained in the main narrative: its purpose is to motivate measuring the signal the objective actually presents.

### 3.2 The opportunity includes what remains to be learned

Feature exposure alone would continue rewarding a target after the learner had mastered it. For the normalized predictor, the negative gradient with respect to its position on the unit sphere is

$$
d_j(C)=(I-U_j(C)U_j(C)^\top)m_j(C).
$$

Its squared norm measures the coherent angular correction still requested by the target. The population quantity

$$
R_j=\mathbb E\|d_j(C)\|_2^2
$$

is a useful diagnostic of remaining predictive supervision. It is small when the conditional target signal is weak and when the prediction is already aligned with that signal. This naturally expresses the idea of a target that is available but still useful to learn. A perfectly anti-aligned prediction is also stationary in this spherical parameterization; loss values and actual gradients distinguish that exceptional configuration.

The backbone receives this signal through the predictor Jacobian. Writing $\mathsf J_j(C)=D_\theta U_j(C)$,

$$
g_j=-\mathbb E[\mathsf J_j(C)^\top m_j(C)].
$$

Since $\mathsf J_j^\top U_j=0$, the same expression holds with $m_j$ replaced by $d_j$. A remaining predictive distinction creates a useful training opportunity when the learner's parameter derivatives can act on it in a direction that improves the primary objective.

The positive mechanism is therefore: **emerging predictive features, remaining learning error, and access through shared computation**. The next model makes their interaction explicit.

## 4. A constructive model of a target crossover

Consider an idealized regression problem in which features are represented in orthogonal coordinates. Let $A_r$ be a centered target feature, $\mu_r(C)=\mathbb E[A_r\mid C]$, and $p_r=\mathbb E[\mu_r(C)^2]$. At a fixed checkpoint, target $j$ exposes feature $r$ with amplitude $b_{j,r}$. The teacher coordinate is $b_{j,r}A_r$ and the student's corresponding prediction is $b_{j,r}w_r\mu_r(C)$.

The coefficient $w_r$ measures progress on the predictive feature; $w_r=1$ is its population optimum. Holding the teacher amplitudes and predictor calibration fixed, squared auxiliary regression has the form

$$
\mathcal L_j(w)
=\frac12\sum_r b_{j,r}^2p_r(1-w_r)^2
+\text{irreducible variance}.
$$

Let the local primary objective be

$$
J(w)=\frac12\sum_r\omega_r(1-w_r)^2,
\qquad \omega_r\geq0,
$$

and let $\kappa_r\geq0$ be a fixed diagonal mobility representing access to the feature through the current parameterization and preconditioner. An auxiliary update changes the remaining error $e_r=1-w_r$ to

$$
e_r^+=(1-a_{j,r})e_r,
\qquad
a_{j,r}=\eta\lambda\kappa_r b_{j,r}^2p_r.
$$

**Proposition: mode-wise developmental value.** Under these assumptions, the exact primary-loss improvement from that auxiliary update is

$$
J(w)-J(w^+)
=\sum_r\omega_r e_r^2
\left(a_{j,r}-\frac12a_{j,r}^2\right).
$$

*Proof.* Substitute $e_r^+=(1-a_{j,r})e_r$ into the quadratic primary objective and expand $1-(1-a)^2=2a-a^2$.

For sufficiently small steps, the target ranking is governed by

$$
\boxed{
V_j(s)=\sum_r
\underbrace{\omega_r}_{\text{task relevance}}
\underbrace{\kappa_r(s)}_{\text{access through learning}}
\underbrace{b_{j,r}(s)^2p_r}_{\text{predictive exposure}}
\underbrace{e_r(s)^2}_{\text{remaining error}}.
}
$$

This model gives a positive reason for progression. Early targets are useful when they expose features on which the learner can make progress. Their contribution declines as the corresponding errors shrink. A deeper target overtakes them when it exposes additional predictive structure while substantial error remains on that structure. Target quality and learner progress evolve together.

For illustration, set task relevance and mobility to one, and absorb $p_r$ into exposure. The following numbers are hypothetical, not experimental results:

| Stage | Remaining errors $e_r^2$ | Shallow exposure | Deeper exposure | $V_{\mathrm{shallow}}$ | $V_{\mathrm{deeper}}$ |
|---|---|---|---|---:|---:|
| Early | $(0.5,1)$ | $(0.8,0.02)$ | $(0.1,0.03)$ | 0.420 | 0.080 |
| Later | $(0.01,0.5)$ | $(0.8,0.05)$ | $(0.15,0.7)$ | 0.033 | 0.3515 |

The shallow target remains informative later. Its most strongly exposed feature has simply become less valuable to supervise again. This is the intended explanation of the moving opportunity.

The proposition is an analytical model of aligned feature learning, not a theorem about general Transformer optimization. Orthogonal modes, aligned teacher and task coordinates, and diagonal mobility are explicit simplifying assumptions. The main empirical test is whether the measured Transformer dynamics exhibit this structure approximately. The native cosine identities in Section 3 connect the same idea to the actual training loss; the squared-loss model supplies a tractable developmental explanation.

## 5. Three claims that the experiments can reject

### Claim H1: useful target depth exhibits a developmental crossover

At checkpoint $s$, define the paired continuation value

$$
G_{j,\lambda}(s;k)
=\mathbb E[J(\theta^{0}_{s+k})-J(\theta^{j,\lambda}_{s+k})],
$$

where $\theta^0$ continues with NTP only and the other arm adds the candidate target. All arms begin with the same complete training state and use paired minibatches. The main outer loss is held-out autoregressive NLL.

For shallow and deeper targets chosen on development grammars, define

$$
D(s)=G_{\mathrm{deeper},\lambda}(s;k)
-G_{\mathrm{shallow},\lambda}(s;k).
$$

H1 predicts negative $D$ early and positive $D$ later on independent grammars and model seeds, with effects exceeding a preregistered practical margin. Repeat over a small auxiliary-weight range and at two continuation horizons. Also report the full layer-value map and uncertainty.

**What would reject it:** sufficiently precise estimates favor a fixed optimum throughout the declared development interval, or the apparent crossover disappears with equal predictor preparation. A positive trend in hand-selected layers after inspecting the test data is not confirmation.

### Claim H2: emerging hierarchical modes explain the crossover

H2 predicts that the deeper target's advantage is associated with increased exposure of a higher-level predictive distinction on which meaningful learning error remains. That distinction should become identifiable before, or within the temporal resolution of, the increase in target value. Removing it should reduce the deeper target's advantage; injecting a controlled version should advance the onset of that advantage.

Measure decodability, predictive exposure, synonym invariance, and the contribution to shared-parameter gradients separately. Compare a development-fitted model using exposure and remaining error with baselines using depth, training age, target loss, and target stability. Evaluate prediction of finite-horizon $G$ on held-out grammars and training histories.

**What would reject it:** target value changes without the predicted feature development, identified feature interventions fail to affect it selectively, or simpler nonsemantic controls explain the effect equally well. The exact local gradient identity is an analysis tool; restating that identity is not evidence that hierarchical modes explain developmental value.

### Claim H3: signals for current value improve progressive weighting

H3 predicts that an adaptive weighting rule can detect useful shifts and outperform the best development-tuned fixed target at the same operational compute budget. It should also outperform, or be more robust across changed learning rates, data quantities, and depths than, a predetermined shallow-to-deep schedule.

**What would reject it:** full adaptive trajectories provide no practically meaningful improvement after counting head calibration, scoring, and discarded updates. A visible crossover alone does not establish an efficient algorithm.

These claims form the intended paper: a developmental phenomenon, its feature-level explanation, and its practical use. H1 is a real commitment. A fixed optimum would motivate a different paper about target choice; it would not confirm the developmental curriculum proposed here.

## 6. Identifying and intervening on feature modes

The RHM supplies latent labels and generative interventions. At a target position, identify its relevant ancestor or completed constituent and use the exact posterior when the prefix does not fully determine its identity. Fit linear encoding models from categorical feature contrasts to target activations on an independent diagnostic split. Confirm the resulting subspaces with held-out decoding and with synonym-preserving versus parent-changing interventions.

For additive gradient accounting, construct mutually orthogonal blocks $\Pi_{j,r}$ plus a residual block that sum to the identity in the native loss geometry. Because hierarchical features can overlap, assign shared directions explicitly and test sensitivity to the preregistered residualization order. Report joint-subspace results alongside level-specific assignments. Concept geometry motivates these measurements; it does not supply canonical orthogonal coordinates for every internal layer.

With that declared decomposition, the exact population gradient can be written

$$
g_j=\sum_r g_{j,r}+g_{j,\perp},
\qquad
g_{j,r}=-\mathbb E[\mathsf J_j^\top\Pi_{j,r}m_j].
$$

The residual term includes the unassigned directions. If centering is used for the exposure statistic, retain the mean component in this gradient accounting; a constant target can still affect optimization.

For a controlled intervention, use

$$
Z_j^{(-r)}=(I-\Pi_{j,r})Z_j
$$

inside the linear target loss $1-U_j^\top Z_j^{(-r)}$, without renormalizing the projected target. At the same checkpoint with a fixed teacher and projection, this removes exactly the corresponding auxiliary semi-gradient component. Re-normalized cosine ablations are useful robustness experiments but are separate interventions.

For a small common-baseline SGD intervention, the first-order difference in primary value is $\eta\lambda\,\nabla J^\top g_{j,r}$. This yields a signed local prediction. The substantive experiment asks whether its direction and the identified mode predict effects over longer training continuations, including the timing of the crossover.

Match the removed dimension and energy with a random-subspace control and, where available, an unrelated feature subspace. Add an oracle feature-injection experiment using the same magnitude as a control injection. Grammar labels in these interventions are explanatory instruments; they do not appear in the deployed controller.

## 7. A concrete signal for adaptive weighting

The first implementable method will estimate primary-task value directly. This keeps the algorithm grounded while the feature measurements explain its decisions.

Maintain weights $w_j$ over candidate targets and a null action $j=0$, with $w_j\geq0$ and $\sum_jw_j=1$. For a total auxiliary scale $\Lambda$,

$$
\mathcal L=\mathcal L_{\mathrm{NTP}}
+\Lambda\sum_{j\neq0}w_j\mathcal L_j.
$$

Every $M$ updates, take a training batch and an independent selection batch. From the same optimizer state, construct one baseline update and candidate updates with the same declared auxiliary scale. Use the actual optimizer, including moment updates and clipping. Score each candidate by

$$
\widehat v_j
=J_{\mathrm{select}}(\theta_0^+)
-J_{\mathrm{select}}(\theta_j^+),
\qquad \widehat v_0=0.
$$

A cheaper score uses the actual parameter difference:

$$
\widehat v_j^{\mathrm{lin}}
=-\nabla J_{\mathrm{select}}(\theta_0^+)^\top
(\theta_j^+-\theta_0^+).
$$

A concrete smooth allocation is

$$
w_j^{\mathrm{new}}
=(1-\rho)w_j^{\mathrm{old}}
+\rho\frac{\exp(\widehat v_j/T)}{\sum_i\exp(\widehat v_i/T)},
$$

where $T$ controls concentration and $\rho$ controls how quickly the allocation changes. Scores use a common evaluation scale; tune $T$, $\rho$, and $\Lambda$ on development runs. The null action allows total latent weight to fall when target supervision has little value. Initially score the full candidate set. A neighboring-layer search is an efficiency variant to test after ordering is established.

This rule can express a band of useful targets and a gradual transfer of weight. It does not force depth to increase; a rising weighted mean depth is a prediction of the developmental hypothesis. Pure-action scores approximate marginal allocation value only locally, so evaluate the resulting mixtures as actual interventions and use complete trajectories for the algorithmic claim. A single-target selection version is a required comparison.

Validation-based auxiliary weighting has established precedents, including Auto-Lambda. The proposed contribution is its application to a verified developmental mechanism in latent targets, with evidence that the resulting progression improves training. We should not present generic lookahead as a new optimization principle. [Liu et al., 2022](https://arxiv.org/abs/2202.03091)

Two cheaper signals deserve explicit comparison: held-out auxiliary learning progress on a fixed teacher, and temporal representation stability. A third diagnostic, $R_j$ from Section 3, isolates remaining coherent supervision on the synthetic grammar. Test each against actual continuation value before combining signals. In particular, if stability alone works almost as well, the practical result may be a simple language analogue of progressive visual prediction.

Candidate heads require equal preparation. Periodically refresh them on detached states, including inactive targets, and count that work. One-step scores must be validated against longer continuations before they control substantial training intervals. All selection examples and discarded updates are part of the operational training budget.

## 8. Implementation and experimental milestones

### Milestone 1: Establish a useful experimental regime and run the motivating comparison

Build a raw-leaf RHM generator, exact small-grammar inference, and a standard decoder-only Transformer with target-layer hooks. A starting configuration is binary hierarchy depth 5, vocabulary 16, four productions per parent, sequence length 32, and eight blocks of width 256. Use these as development defaults, then freeze the chosen configuration before confirmation on new grammars.

Run NTP, a fixed shallow target, a fixed deeper target, and a predetermined shallow-to-deep schedule early in the project. Keep NTP active in all latent runs. Add a matched progressive-freezing arm on a subset to connect directly to the vision precedent. These runs test whether the setting contains a promising developmental phenomenon; theory diagnostics should support this comparison rather than delay it.

Validate causal indexing, stop-gradient behavior, and grammar conditionals. Keep optimization time separate from unique-sample exposure by studying a fixed dataset first, then varying dataset size. Record ordinary-layer probes and target stability throughout training.

**Output:** reproducible pilot learning curves and representation trajectories. **Decision:** select an observable regime on development data; retain unfavorable confirmation results.

### Milestone 2: Measure the shallow-to-deeper crossover

Create paired branches from both NTP-only and fixed-shallow-target histories. The latter is central: the practical question is when to advance after beginning with useful shallow supervision. Continue NTP, the current target, and alternative targets from each complete checkpoint. Match minibatches, head calibration, optimizer state, and target policy.

Use at least early, middle, and late checkpoints, a modest depth grid, and two horizons such as 32 and 128 updates, with a longer subset. Select the shallow/deeper comparison, stage windows, and meaningful effect margin before evaluating held-out grammars. Sweep auxiliary strength on a subset and include a matched-gradient-norm comparison.

As an illustrative budget, three grammars, two seeds, five checkpoints, seven arms, and 128 updates require 26,880 shadow updates per checkpoint-history family and weight. Base trajectories, head fitting, and diagnostic costs are additional. Three grammars are a pilot; the number needed for confirmation depends on measured between-grammar variability.

**Output:** the causal value heatmap and the preregistered crossover curve $D(s)$. **Decision:** assess H1 with interval estimates and a practical effect margin; a flat or fixed optimum is an informative rejection.

### Milestone 3: Explain and shift the crossover with feature interventions

Fit layer-specific feature subspaces on a diagnostic split. Compute exact $m_j$, predictive exposure $P_{j,r}$, coherent residual $R_j$, and held-out feature error. Use native layer geometry for loss accounting and declare any transformed-geometry analysis separately.

Fit the constructive exposure-and-error model to development grammars; predict target rankings on held-out grammars. Then remove the hypothesized emerging mode, remove matched control directions, and inject a controlled predictive feature. Test whether removal delays or eliminates the deeper target's advantage and whether injection advances it. Report both immediate gradient effects and finite-horizon NLL effects.

Branch averaging and next-token conditioning are secondary controls if needed to explain an observed target difference. Branch averaging preserves the expected auxiliary semi-gradient without changing its conditional-mean cancellation; token conditioning changes the task. They should not displace the main developmental experiment.

**Output:** one or two feature interventions with predicted effects on crossover timing. **Decision:** assess H2 through held-out prediction and selective causal effects, rather than probe correlations alone.

### Milestone 4: Validate signals and run adaptive weighting

Implement actual-update lookahead and its linear approximation. Compare them with fixed-teacher learning progress and temporal stability. Measure selection regret against the branch values from Milestone 2, using separate data to choose and judge the action. Test whether a signal anticipates the deeper target's advantage and whether it remains useful at the training interval it will control.

Run the proposed smooth weighting rule and a single-target version. Start the main trajectory with NTP plus shallow latent supervision; retain exploration of alternative targets during periodic scoring. Restore every discarded model, optimizer, head, and teacher state before continuing. Log layer weights, total latent strength, score uncertainty, and actual overhead.

**Output:** a validated switching signal and complete adaptive trajectories. **Decision:** retain a signal only if its decisions predict meaningful continuation value and its cost is plausible; a simple signal that works is preferable to an elaborate unvalidated composite.

### Milestone 5: Establish practical value and developmental specificity

At matched total FLOPs, compare NTP, the best development-tuned fixed target, a uniform mixture, the predetermined progressive schedule, stability-triggered progression, validation-based weighting, and the proposed adaptive recipe. Where the proposed recipe is the validation-based rule itself, use an existing meta-weighting implementation as the comparison and report the overlap plainly. Include a reverse-depth schedule as a test of developmental ordering.

Match tuning resources and report operational costs separately from the larger scientific diagnostic campaign. The NTP and fixed-target baselines receive the training that fits within the same total compute. Run both token-matched and compute-matched comparisons.

Change learning rate, training-set size, and network depth on held-out settings. The signal-driven schedule should track changes in development better than a wall-clock schedule. Add mixture-versus-single-target tests and freezing-versus-unfrozen tests so that any benefit can be assigned to the relevant design choice.

**Output:** efficiency curves and adaptation under altered developmental speed. **Decision:** assess H3 against the tuned fixed and scheduled alternatives, including selection overhead.

### Milestone 6: Test the same principle in language models

Use ordinary target layers and the validated label-free signal in a modest language-model pretraining pilot, for example 100–300M parameters. Keep the architecture and NTP objective fixed. Repeat at a second depth or scale before claiming broad transfer.

Test whether target weights move from shallow to deeper contextual layers and whether that progression improves held-out NLL at matched FLOPs. Add tasks with meaningful signal at the pilot scale and fixed-prefix future probes whose inputs exclude intermediate true tokens. Use representation complexity and concept probes to interpret the progression; neither substitutes for the causal and training outcomes.

Complete a full-text comparison with nearby latent-language methods before finalizing novelty. HiLP is relevant to hierarchical latent prediction, but its reported higher-level rollout mechanism is a different design axis from choosing ordinary target depth during training; the comparison here is based on its abstract. [Shi et al., 2026](https://arxiv.org/abs/2608.05806)

**Output:** a language-model test of the developmental prediction and its training benefit. **Decision:** distinguish successful synthetic mechanism, successful language transfer, and practical efficiency as separately evidenced claims.

## 9. Intended paper structure and contribution

The main paper should open with progressive visual prediction and the fixed-target question in language modeling. Its first major result should be the measured shallow-to-deeper crossover. The feature-mode analysis then explains that result and motivates the weighting signal. The final result is the complete adaptive learning curve.

Four central figures would present: (1) target value over training age and network depth; (2) predictive feature exposure and remaining error around the crossover; (3) feature interventions that change crossover timing; and (4) adaptive versus fixed and scheduled training at matched compute.

The earlier mathematical counterexamples, detailed channel-noise bounds, optimizer qualifications, and estimator proofs can be retained as supporting material. Their role is to clarify assumptions and select controls. The main text should spend its explanatory effort on the regularity we expect to find: **the best self-supervised target advances when a deeper representation exposes predictive structure whose acquisition still offers useful progress.**

The developmental crossover, its feature-level causes, and the effectiveness of a practical weighting signal are the proposed discoveries. The literature and analytical model make them plausible; the experiments decide whether they hold.
