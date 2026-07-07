# Recoverable Is Not the Same as Useful: A Multi-Domain Pre-Registered Study of Small Next-Token Transformers

Author: [Raghavan Muthuregunathan](https://www.linkedin.com/in/raghavanmit/) 

<span style="font-size: 5px;">
This is an independent work and does not represent any of my past or present employers.</span>

## Abstract

We introduce a protocol for pre-registered representational analysis in
next-token transformers. The protocol combines (i) multi-seed probes and
activation patching, (ii) destroyed-structure corpus controls (real,
within-shuffled, global-shuffled), (iii) position-controlled probing, and
(iv) git-audited ex-ante predictions whose commits predate model training.
We apply it across seven domains: cities, Othello, music, flight phases,
symmetric-group walks, maze navigation, and HTTP log sequences. The maze
and HTTP experiments are run against quantitative predictions committed
to a public repository before any data was collected or any model was
trained, with commit hashes verifiable via `git log --diff-filter=A` on
the predictions files.

We test a candidate hypothesis called the N-criterion: encoding in the
residual stream is driven by predictive relevance for next-token
prediction. Both pre-registered experiments falsify the strict
biconditional form. Features predicted to be absent were recovered above
the untrained baseline at +0.15 (maze starting cell) and +0.22 (HTTP
cumulative count, after position-control). The one mechanism that
survives ex-ante testing is *architectural carry-through*: features at
positionally distinct input slots persist in the residual stream via the
side effect of self-attention copying. Carry-through was introduced
after the maze falsification and independently confirmed by the HTTP
first-request prediction (+0.17 gap), making it the single substantive
ex-ante claim the framework passed on two domains of different shape.
Three post-lock HTTP follow-ups further show the effect is not confined
to first slots: same-request path is recoverable at the size token
(+0.81 MLP gap), previous-request path is recoverable across a record
boundary (+0.67), and the path of the most recent earlier large response
remains recoverable even when the source event is content-selected
(+0.62 full lag; +0.51 with lag >= 3).

The HTTP Feature B result also surfaces a methodological failure mode
that probe-based work should control for: target features correlated
with token position inherit positional encoding's gap. We propose two
diagnostics (within-position probing at fixed k, and
residual-after-position regression) and show Feature B remains encoded
above the locked threshold even after both controls.

Contributions: (i) a pre-registered protocol for
emergent-representation experiments; (ii) one ex-ante-validated
mechanism (architectural carry-through) confirmed on two domains;
(iii) one identified probe-confound mode with two controllable
diagnostics future work should adopt. We do not claim a strong
predictive theory of which features emerge as representations.

Scope: all models are 0.27M to 13M parameters on synthetic or
semi-synthetic data. We do not claim that any findings extend to
frontier-scale models on natural-language data.

---

## 1 Introduction

### 1.1 The question

A transformer language model trained only on next-token prediction is not
given access to the structure of its domain: it does not see board states
in Othello, latitude/longitude in cities, or chord labels in music. It
only sees tokens. Prior work nevertheless shows that such a model can
develop, in its residual stream, a linearly recoverable representation of
domain state. Li et al. (2022) and Nanda et al. (2023) demonstrated this
for Othello board state. The natural follow-up is whether and when this
generalizes, and what predicts which features get encoded.

### 1.2 Three methodological gaps in prior work

Most mechanistic-interpretability work on emergent representations shares
three limitations.

1. **Single domain, post-hoc**: a typical study probes one domain, reports
   what worked, and leaves the reader unable to distinguish a genuine
   pattern from a fishing expedition.
2. **Single seed**: probe accuracies are reported as point estimates rather
   than mean ± std, so the noise floor of the measurement is hidden.
3. **No ex-ante predictions**: published probe studies almost never commit
   in writing to what the outcome should be before the experiment is run,
   so it is difficult to distinguish a framework that *predicts* the data
   from one that *fits* the data after the fact.

This paper addresses all three: comparative (seven domains), multi-seed
(five seeds, mean ± std), and pre-registered (two domains with
git-audited predictions committed before training).

### 1.3 Contributions

This paper makes three contributions.

**First, a methodological protocol**. We combine multi-seed reporting,
probe and activation-patching convergence, per-layer ablation,
destroyed-structure corpus controls, position-controlled probing, and
pre-registered ex-ante predictions with a git audit trail. None of these
elements is individually novel; the combination is rarer than the
literature suggests and is the principal contribution of this paper.

**Second, an ex-ante-validated mechanism**. We test the N-criterion
hypothesis on two pre-registered domains. Both falsify its strict form.
The graded form, refined to include a specific mechanism we call
*architectural carry-through*, was introduced after the maze
falsification and confirmed by the HTTP first-request prediction. It is
the one substantive ex-ante claim that survived on two domains of
different shape.

**Third, an identified methodological failure mode**. The HTTP Feature B
result surfaces position-correlation as a probe confound. We propose two
controllable diagnostics (within-position probing at fixed k and
residual-after-position regression) and recommend them as default for
future probe-based work whenever the target feature correlates with
token position.

We do not claim a strong predictive theory of which features emerge.
The pre-registered tests have demonstrably weakened any such claim.
What remains is a methodology, one validated mechanism, and one
identified failure mode.

### 1.4 What this paper is not

We do not study frontier-scale language models. We do not study
natural-language data. We do not claim that any of the representations
recovered here are a "world model" in a philosophically loaded sense.
We study whether, when, and where small transformers represent
task-relevant structure in their residual streams, with care taken about
what counts as evidence.

### 1.5 Visual companions

Eight figures support this paper and are available in the
[`figures/`](../../figures/) directory of the project repository. Readers
who prefer a visual summary before the prose may find them useful.

- **Figure 1**: [`figures/01_experimental_arc.md`](../../figures/01_experimental_arc.md).
  Mermaid flowchart of the pre-register, falsify, revise, re-test arc,
  with commit hashes and gap numbers inline. Renders directly on
  GitHub. The single best entry point for readers who want to see the
  scientific timeline at a glance before reading the experimental
  sections.
- **Figure 2**: [`figures/02_results_matrix.md`](../../figures/02_results_matrix.md).
  Color-coded outcome matrix across every (domain, feature, condition)
  cell tested in the paper. Includes a summary scorecard.
- **Figure 3**: [`figures/03_per_layer_ablation.png`](../../figures/03_per_layer_ablation.png).
  Per-layer encoding strength across domains. Embedded in §5.
- **Figure 4**: [`figures/04_cross_condition_gradient.png`](../../figures/04_cross_condition_gradient.png).
  Cross-condition gradient comparison across domains. Embedded in §5.
- **Figure 5**: [`figures/05_maze_shuffle_comparison.png`](../../figures/05_maze_shuffle_comparison.png).
  Maze-only real / within-shuffled / global-shuffled comparison for the
  pre-registered maze claims.
- **Figure 6**: [`figures/06_http_shuffle_comparison.png`](../../figures/06_http_shuffle_comparison.png).
  HTTP-only real / within-shuffled / global-shuffled comparison,
  including the Feature B position-control follow-ups.
- **Figure 7**: [`figures/07_maze_http_layer_ablation.png`](../../figures/07_maze_http_layer_ablation.png).
  Blog-focused layer-by-layer probe gaps for the two pre-registered
  Maze and HTTP domains.
- **Figure 8**: [`figures/08_http_carrythrough_followups.png`](../../figures/08_http_carrythrough_followups.png).
  HTTP carry-through follow-up ladder contrasting decodability with
  corruption-intervention usefulness.

Figure 1 is the recommended starting point. It is referenced again from
§5 (alongside the embedded Figures 3 and 4) and from §7.1, where the
prose narrative of the same arc lives. Figures 5-8 are focused companions
for the Maze/HTTP story emphasized in the LessWrong post; the broader
paper keeps the cross-domain Figures 2-4 as the main visual summary.

---

## 2 Framing: The N-Criterion

### 2.1 What we are trying to make precise

The intuition behind Othello-GPT and its successors is that a transformer
needs to know the board state to predict the next legal move, and that
this need pressures gradient descent to construct an internal
representation. Generalizing this cleanly is harder than it sounds. The
word "need" can mean information-theoretically required, required for
optimal prediction, required for *achievable* prediction at this
capacity, or merely useful. Almost no real feature is cleanly required
or cleanly irrelevant; predictive relevance is continuous.

### 2.2 The strict form (as originally tested)

The strict form, useful as a sharp target:

> **Strict N-criterion.** A feature F is encoded as a linearly recoverable
> representation in the residual stream of a trained next-token
> transformer **iff** F is required for next-token prediction on this
> corpus, where "required" is defined model-independently from the task's
> information structure.

This biconditional is falsifiable in both directions: a required feature
not encoded falsifies the "if"; an irrelevant feature found encoded
falsifies the "only if". Both directions of falsification occur in our
pre-registered tests (§5.7, §5.8).

### 2.3 The graded form (post-experiment refinement)

The graded form was adopted after the maze starting-cell falsification
made the strict iff untenable. It retains predictive relevance as the
dominant driver of encoding but adds a second mechanism for input-borne
features:

> **Graded N-criterion.** Predictive relevance of a candidate feature F
> drives whether F emerges as a linearly recoverable representation in
> the residual stream. Predictively-relevant features tend to be encoded;
> predictively-irrelevant features tend to be absent, *except* when F is
> *architecturally carry-through-accessible*, that is, when F is present
> in the input at a positionally distinct slot and self-attention can
> copy it forward at essentially zero cost.

The graded form is weaker, more defensible, and (as we show) only half
correct. The carry-through *positive* direction confirms ex-ante on two
domains. The graded form's *null* direction (computed features should be
absent) is also falsified in HTTP. We discuss the mechanistic intuition
behind carry-through in §7, where we also tell the full arc of how the
framework changed during the study.

### 2.4 What we explicitly do not claim

The N-criterion is not a claim about whether probes or activation
patching work; those methods are how we test it.

It is not a claim that features absent from the residual stream are not
used by the model. A feature might be computed on the fly in attention
without ever appearing as a clean residual direction. The N-criterion is
about *what is represented in the residual stream*, not *what the model
uses*. These can come apart.

It is not a claim about frontier-scale language models on natural
language. Whether they follow this pattern is an empirical question on
which our small-model results provide weak indirect evidence at best.

### 2.5 Background and related work

**Othello-GPT.** Li et al. (2022) trained a transformer on random Othello
games and showed that an MLP probe recovers per-cell board state at
approximately 94% accuracy. Nanda et al. (2023) showed the representation
is approximately linear under the mine/yours/empty basis.

**Basis dependence.** Linear decodability depends on basis choice. We
adopt a single per-domain parameterization (defined from the task) and
acknowledge this as a choice.

**Superposition.** Elhage et al. (2022) showed that small transformers
pack more features than residual dimensions into overlapping directions.
This is directly relevant to our negative predictions: superposition
predicts that features not strictly required can still appear. The
strict N-criterion is in tension with this; the graded form
accommodates it.

**Probe-causality convergence.** A probe finding that a feature is
decodable is consistent with the probe computing the feature rather
than the model representing it. The standard defense is activation
patching, with convergence (positive probe AND positive transplant) as
the standard for representational claims. We adopt this.

---

## 3 Core Concepts and Assumptions

### 3.1 Features as probe targets

A feature F is a function of the input prefix yielding a discrete label
at each probe position. Examples: per-cell board state (Othello), grid
cell (cities), beat-in-measure (music), flight phase (ADS-B), cumulative
large-response count (HTTP), starting cell (maze). Probe accuracy at
layer L is the test accuracy of a linear or two-layer MLP regressor
trained on the residual at L. We report both linear and MLP probes;
the linear-vs-MLP gap is itself diagnostic of whether the encoding is
linearly accessible.

**THE ONE RULE.** The probe target's true value must not appear in the
model's input stream. Tokens are arbitrary identifiers; coordinates,
phases, beat positions, cumulative counts live only in side tables that
the probe reads but the model never sees.

### 3.2 The carry-through mechanism

When a feature sits at a positionally distinct slot in the input (such
as the first non-BOS token), self-attention has a low-cost path to copy
that slot's content forward. The model has no incentive to *discard*
such features even when they are predictively irrelevant, because their
persistence is essentially free. The mechanism predicts: features whose
value is fixed early and retrievable in a single attention hop should
be encoded above untrained baseline, regardless of relevance.

### 3.3 Destroyed-structure controls

We construct two per-domain controls:

- **Within-shuffled**: token order permuted within each sequence; the
  per-sequence token set is preserved.
- **Global-shuffled**: tokens reassigned uniformly across the corpus;
  per-sequence set membership is destroyed too.

A trained-vs-untrained probe gap that disappears under these controls
is evidence that the encoding depends on genuine task structure, not on
a memorization artifact.

### 3.4 Pre-registration as anti-hindsight

For the maze and HTTP domains, we commit *quantitative* predictions
(probe accuracy bands, falsification thresholds, condition-by-condition
expectations) to the project repository before any model is trained.
The commit timestamp is the audit trail. The predictions file is
append-only after the commit. This converts the framework from "explains
what we already saw" to "predicts what we have not seen yet." It is the
central scientific discipline of this paper.

The complementary methodological tool we introduce in this paper is
position-controlled probing (§4.10), developed in response to the HTTP
Feature B result. It is independent of the N-criterion: any probe whose
target correlates with token position should be run with one of the
position-control designs as a default check.

---

## 4 Core Methods

### 4.1 Models

All models are small decoder-only transformers (nanoGPT-style),
0.27M-13M parameters depending on domain. Architecture details are
tabulated in Appendix B.

### 4.2 Three corpus conditions per domain

For each domain we train three models with identical hyperparameters,
one per corpus condition (real, within-shuffled, global-shuffled).
A representation that survives under global-shuffled is almost
certainly a probe-capacity artifact rather than a genuine learned
feature.

### 4.3 Multi-seed protocol

Every probe and transplant number is the mean ± std across five
independent seeds, where each seed simultaneously varies the
untrained-control random initialization, the positions sampled for
the probe dataset, and the probe-training optimizer's random state.

This protocol caught at least one bug during the study: four probe
scripts were initializing both trained and untrained models with random
weights, silently masking the gap on the affected runs. The multi-seed
variance pattern on those runs disagreed with the others. The fix is
committed and the affected numbers were rerun.

### 4.4 Probes

For each probe target we train two probes from each layer's residual
stream: a single dense linear probe with cross-entropy loss, and a
one-hidden-layer ReLU MLP probe. The linear-vs-MLP gap tests the
linear-encoding hypothesis (Nanda et al. 2023) in the chosen basis.

### 4.5 Splits

Position-level (random partition of probe positions) is the weak
baseline; a node-level, maze-level, piece-level, or flight-level split
is the honest test. We report both and treat the honest split as the
primary number.

### 4.6 Activation patching (transplant)

For each (domain × layer) we run a causal intervention: cache B's
residual at layer L, substitute it into A's forward pass at L, and
measure the change in the next-token distribution toward B's expected
continuation. The standard summaries are "P(B-continuation) under
transplant minus under unpatched" and "under transplant minus under
random control." Positive transplant lift with a positive probe at the
same layer is the convergent evidence on which a representational claim
can rest.

### 4.7 Per-layer ablation

We run probe and transplant at *every* layer. Convergence at a specific
depth (probe and transplant both peaking at the same layer) is the
strongest layerwise evidence.

### 4.8 Pre-registration protocol

For maze and HTTP we wrote quantitative predictions to the repository
before any model was trained. The commit timestamp is the audit trail;
the predictions file is append-only after commit. This does not fully
remove unconscious analysis bias, but it makes post-hoc threshold
tuning and silent prediction revision verifiable from outside.

### 4.9 Power for nulls

A null probe result is intrinsically weaker evidence than a positive:
the probe might lack capacity to recover an actually-encoded feature.
We partially mitigate via (i) using the same probe family and budget
that recovers positive controls in the same domain, and (ii) corroborating
nulls with transplant where possible. We do not offer a formal power
analysis; we acknowledge this as the most defensible criticism of our
null results (§8.3).

### 4.10 Position-controlled probing

This is the second methodological contribution of the paper, alongside
pre-registration. When a probe target correlates with token position,
the trained-vs-untrained gap can be inflated by positional encoding
alone. We use two complementary controls.

**Design A: Within-position probing.** Restrict probe data to a single
fixed position k. The positional embedding contribution is constant
across examples, so any trained-vs-untrained gap cannot be attributed
to positional encoding alone.

**Design B3: Residual-after-position probing.** Fit a per-position
empirical baseline P(F | k) on the probe-train split (Laplace-smoothed).
Compute the residual `y_onehot - baseline(k)` on both train and test.
Train a regression probe (linear and MLP) to predict the residual from
activations; report R² on the test residual.

Agreement between A and B3 yields a strong joint verdict. A and B3 were
developed post-hoc for the HTTP Feature B result (§5.8.4), but the
diagnostic is general: any probe with a position-correlated target
should run at least one of A or B3 as a default. We commend this as a
recommended practice for future probe-based interpretability work.

---

## 5 Cross-Domain Evidence

We present results in the order of how cleanly they instantiate the
N-criterion. Each subsection opens with a one-sentence verdict followed
by the numbers and the discussion.

Visual companions to this section:
[`figures/02_results_matrix.md`](../../figures/02_results_matrix.md) gives an
at-a-glance outcome matrix across every (domain, feature, condition) cell.
Figures 3 and 4 below show the per-layer ablation and the cross-condition
gradient, respectively. Figure 1, the experimental arc flowchart, is in
[`figures/01_experimental_arc.md`](../../figures/01_experimental_arc.md) and is
also referenced from §7.

![Figure 3: Per-layer ablation](../../figures/03_per_layer_ablation.png)

**Figure 3**: Per-layer ablation across six (domain, feature) pairs
reported in this section. Othello and music are shown via causal
transplant lift; maze starting cell and HTTP Features A and B are shown
via probe gap. The bottom-right panel is the HTTP Feature B probe at
fixed position k=5 (Design A, §5.8.4), showing the trained and untrained
accuracy lines separately.

![Figure 4: Cross-condition gradient](../../figures/04_cross_condition_gradient.png)

**Figure 4**: Real / within-shuffled / global-shuffled gap bars for six
(domain, feature) pairs. The framework's destroyed-structure prediction
is real > within > global. The note below each panel indicates whether
the prediction holds (cities London, maze starting cell, and Feature A
all deviate from monotonicity in ways discussed in the corresponding
subsections).

### 5.1 Othello (positive control, independent reproduction)

**Verdict**: confirmed positive control. Board state is required for
legal-move prediction and is recovered linearly at 94% accuracy,
matching Li et al. (2022) within 0.01.

We trained a transformer (4M parameters) on 50,000 random Othello games.
Per-cell occupancy probe (3 classes):

| | Trained MLP | Untrained MLP | Gap |
|---|---|---|---|
| Best layer (L4), per-cell mean | **0.9399 ± 0.0012** | 0.5963 ± 0.0058 | **+0.344** |

Linear probe at L4: 0.8093 ± 0.0059 (gap +0.255). Linear-to-MLP gap at
L4: 0.131, the upper end of what we call "approximately linear" (§8.4).

Per-layer transplant lift peaks at L3 (+0.296). The probe peaks at L4.
Convergent: both descriptive and causal evidence locate board state in
the same depth band.

### 5.2 Music (one domain, both directions of the criterion)

**Verdict**: cleanest positive plus null pair in the paper. Voice-leading
is encoded as predicted; chord is weakly encoded as predicted;
beat-in-measure is null on both probe AND transplant, as predicted.

We trained on Bach chorales and related Renaissance polyphony (1.4M
parameters). Probe targets and verdicts on the honest piece-level
split:

| Target | Trained MLP | Untrained MLP | Gap | Verdict |
|---|---|---|---|---|
| Voice-leading (transplant lift at L2) | +0.889 ± 0.007 | (n/a) | (n/a) | encoded + causally used |
| Chord | 0.3035 ± 0.0202 | 0.2147 ± 0.0285 | +0.089 | weakly encoded |
| Beat-in-measure | 0.2798 ± 0.0075 | 0.2740 ± 0.0077 | +0.006 | **null** |

The beat null is doubly corroborated. Beat-transplant on
matched-voice-leading donors moves predictions *less* than the
random-direction control (max|Δp| 0.273 ± 0.005 vs 0.493 ± 0.007;
argmax-changed 0.470 vs 0.965; KL 0.69 vs 3.98). This is the strict
N-criterion's negative direction verified by both probe and causal
experiment.

The beat null was identified post-hoc. It motivated the design of the
pre-registered maze starting-cell prediction (§5.7), where the
analogous irrelevance prediction was committed to writing before any
data was collected.

### 5.3 Cities (partial confound, downplayed)

**Verdict**: present but driven primarily by the embedding table, not by
the transformer. Qualitatively different evidence than Othello or music.

We trained transformers (11M parameters) on tokenized street-walk
sequences from real cities. Probe target: 10×10 grid classification of
intersection location on the honest split.

| City × Condition | Trained MLP | Untrained MLP | Gap |
|---|---|---|---|
| London real | 0.6423 ± 0.0545 | 0.0927 ± 0.0130 | +0.550 |
| London within-shuffled | 0.7393 ± 0.0505 | 0.0673 ± 0.0170 | +0.672 |
| London global-shuffled | 0.0993 ± 0.0248 | 0.0870 ± 0.0359 | +0.012 |
| Manhattan real | 0.6092 ± 0.0141 | 0.1033 ± 0.0078 | +0.506 |
| Boston real | 0.6667 ± 0.0205 | 0.1117 ± 0.0042 | +0.555 |

Two observations qualify the headline. First, within-shuffled London
scores higher than real London (a non-monotonic finding we do not have
a clean test for; see §8.5). Second, per-layer transplant lift on
London real is +0.735 at L0 already: more than three quarters of the
eventual signal is in the embedding table itself, before any
transformer block has acted. We therefore treat cities as evidence that
*embedding-table structure can mimic the surface phenomenology of an
emergent world model* rather than as a third clean example of one.

### 5.4 Flight phases (ADS-B, moderate signal)

**Verdict**: graded N-criterion working at moderate magnitude.

Real ADS-B trajectories (238 flights, 46k tokens, 0.27M-param model).
Flight phase probe on the honest flight-level split:

| Condition | Trained MLP | Untrained MLP | Gap |
|---|---|---|---|
| Real | 0.8817 ± 0.0792 | 0.7765 ± 0.1043 | +0.105 |
| Within-shuffled | 0.6785 ± 0.1834 | 0.5783 ± 0.1715 | +0.100 |
| Global-shuffled | 0.3766 ± 0.1512 | 0.3236 ± 0.1080 | +0.053 |

Per-layer transplant lift on real: +0.47 at L1 (peak in a 2-layer
model); within-shuffled +0.30; global-shuffled +0.00. The probe-side
untrained baselines are high because flight phase is heavily determined
by single-token statistics. The transplant gradient is the more
interpretable evidence.

### 5.5 Symmetric-group walks (partial signal)

**Verdict**: trained beats untrained statistically but at small absolute
magnitude. Useful as a partial data point, not a clean positive control.

S_8 self-avoiding walks (10.7M-param model). Per-element partial-product
probe on the honest word-level split:

| Variant | Trained MLP | Untrained MLP | Gap |
|---|---|---|---|
| sa real | 0.3599 ± 0.0050 | 0.3074 ± 0.0039 | +0.053 |
| sa within-shuffled | 0.2879 ± 0.0018 | 0.2778 ± 0.0039 | +0.010 |
| sa global-shuffled | 0.2913 ± 0.0040 | 0.2774 ± 0.0034 | +0.014 |

Destroyed-structure controls null out as predicted on the honest split.

### 5.6 Per-layer transplant patterns

Per-layer transplant lift shows a domain-distinctive pattern:

| Domain (real) | embed | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| Othello | +0.040 | +0.062 | +0.108 | **+0.296** |
| Music (voice-leading) | +0.035 | +0.813 | **+0.889** | (n/a) |

Othello builds the representation across depth, peaking at the deepest
block. Music shows a striking L0 → L1 jump: voice-leading state is
absent from the embedding table and constructed by the first transformer
block from surrounding context. This is the canonical
"transformer-computed world state" pattern, in contrast to cities
(§5.3), where the embedding table itself already carries most of the
relevant structure.

### 5.7 Pre-registered: maze navigation

**Verdict**: P4 starting-cell NULL prediction *falsified* at +0.15 gap
(threshold +0.10); P3 distance-to-goal positive prediction also
falsified at +0.01 gap (threshold +0.20); P1/P2 row/col predictions
were predictor's error (token-identity confound). The strict
N-criterion fails on this pre-registered domain.

The predictions were committed to the project repository before any
maze data was generated or any maze model was trained. Commit hash
`aa025b1`; audit via
`git log --diff-filter=A predictions/predictions_maze_navigation.md`.

#### Setup and the four predictions

8×8 mazes, 100k mazes, 1.5M training tokens, 67-token vocab, 6-layer
transformer (2M params). Honest split is maze-level (test mazes whose
walls the model never saw during training). Five seeds.

| # | Probe target | N-criterion verdict | Pre-registered band |
|---|---|---|---|
| P1 | Current cell row | encoded | trained MLP 0.70-0.97; gap ≥ 0.40 |
| P2 | Current cell col | encoded | same as row |
| P3 | Manhattan distance to goal | encoded | trained MLP 0.35-0.75; gap ≥ 0.20 |
| P4 | Starting cell ID | **NULL** | gap ≤ 0.10 (load-bearing risky claim) |

P4 was the load-bearing prediction: the next path step does not require
knowing where the path started, only the current cell and the goal.

#### Results

| # | Trained MLP | Untrained MLP | Gap | Outcome |
|---|---|---|---|---|
| P1 (row) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.000 | predictor's error (token-identity) |
| P2 (col) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.000 | predictor's error |
| P3 (distance, best layer) | 0.1462 ± 0.0064 | 0.1310 ± 0.0115 | +0.015 | **band missed** (predicted ≥ 0.20) |
| P4 (start cell, L5 honest) | 0.2024 ± 0.0166 | 0.0504 ± 0.0082 | **+0.152** | **NULL FALSIFIED** |

Each maze cell is a unique token (token IDs 3..66, row = (id − 3) // 8,
col = (id − 3) mod 8), so row and column are deterministic functions of
token identity. P1 and P2 were never meaningful tests; we flag this as
a predictor's error and discuss in §8.7.

P3: the trained MLP reaches 0.1462, barely above untrained 0.1310. The
model can predict the next path step locally without ever computing
distance-to-goal as an explicit residual feature.

P4: the trained model's L5 residual carries the starting-cell information
at probe accuracy 0.20, well above untrained 0.05 and chance 0.015. The
gap of +0.152 exceeds the locked falsification threshold of 0.10. The
signal builds across depth (L1: +0.04 → L2: +0.12 → L5: +0.15),
suggesting transformer-constructed rather than embedding-table artifact.

Per-layer transplant lift on real: +0.155 at L5; within-shuffled +0.017;
global-shuffled +0.155. Real and global-shuffled produce nearly
identical transplant profiles. We treat this as a finding about what
the transplant metric measures (sequential co-occurrence structure, not
spatial structure) rather than as evidence about the model.

P4 is the most important single result in the paper. The strict
N-criterion's biconditional does not survive: an irrelevant feature,
predicted absent in advance via an audit-trail mechanism, was recovered
with substantial signal. The graded form was proposed in response;
§7 tells the full arc.

### 5.8 Pre-registered: HTTP log sequences

**Verdict**: Feature A (carry-through) confirmed at +0.17 gap, replicating
the maze carry-through finding on a different domain shape. Feature B
(computed null) apparently falsified at +0.29 gap; *also falsified at
+0.22 even after position-control*. The graded form's null direction
fails on this pre-registered domain.

The predictions were committed to the repository before any NASA-HTTP
data was downloaded. Commit hash `3b25ed3`; audit via
`git log --diff-filter=A predictions/predictions_http_log_sequences.md`.

#### Setup and the two predictions

NASA-HTTP July + August 1995 archives from the
[Internet Traffic Archive](https://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html)
(`NASA_access_log_Jul95.gz` and `NASA_access_log_Aug95.gz`; 3.46M
requests, 226k retained sessions, 8.08M training tokens). Per-field
tokenization: each request is 4 tokens (`method`, `path_category`,
`status_bucket`, `size_bin`). Vocab 52, model n_layer=4 / n_head=4 /
n_embd=128 (0.81M params).

| # | Probe target | Category per graded framework | Predicted outcome |
|---|---|---|---|
| P1 | Feature A: `size_bin` of FIRST request | In-input at distinct slot | **Encoded** via carry-through (gap ≥ 0.10) |
| P2 | Feature B: cumulative count of large responses, binned | Must be actively computed | **Null** (gap ≤ 0.10) |

Cross-feature prediction: A gap ≥ B gap.

#### Results (initial probe)

Session-level honest split, 5-seed mean ± std, best layer:

| | Trained MLP | Untrained MLP | Gap | Verdict |
|---|---|---|---|---|
| Feature A (L3) | 0.410 ± 0.04 | 0.243 ± 0.03 | **+0.168** | **P1 confirmed** |
| Feature B (L2) | 0.888 ± 0.004 | 0.597 ± 0.01 | **+0.291** | **P2 apparently falsified** |

Cross-feature ordering: A < B. Cross-feature ordering also falsified.

#### Why Feature B has two readings

Position-in-session correlates with cumulative-large-response count
(later positions tend to have higher counts). The trained model
develops sharper positional representations than untrained (it must
predict which of `{method, path, status, size_bin}` comes next, which
requires knowing position mod 4). The probe may be reading position
rather than aggregation.

Two pieces of evidence consistent with this reading:

- Untrained baseline is 0.60 on Feature B (well above the 0.33
  3-class chance), so positional info alone gives substantial signal.
- Global-shuffled gap (+0.30) is at least as large as real (+0.29). If
  the trained model were actually identifying "large-response tokens"
  and counting them, global-shuffled (which permutes token identities)
  should collapse the gap. It does not. Position-as-proxy is the more
  consistent explanation.

We did not pre-register a position-control. It was identified post-hoc
as the natural follow-up. The methodological lesson is captured in §4.10.

#### Position-control follow-up

Three follow-up probes, all post-hoc:

**Position-control probe.** Target: `request_idx_in_session`, binned
4-class (idx ≤ 3 → 0; ≤ 6 → 1; ≤ 12 → 2; else 3). A purely positional
deterministic function with no content-dependent component.

| Condition | Trained MLP @ L2 | Untrained MLP best | Gap |
|---|---|---|---|
| Real | 0.836 ± 0.009 | 0.409 ± 0.014 | **+0.427** |
| Within-shuffled | 0.945 ± 0.007 | 0.404 ± 0.026 | **+0.541** |
| Global-shuffled | 0.828 ± 0.011 | 0.424 ± 0.011 | **+0.404** |

Pure position information is recovered at +0.43 by the trained model;
Feature B at +0.29. Position alone exceeds Feature B in every condition,
confirming positional encoding accounts for part of the apparent
Feature B signal.

**Within-position probe at fixed k=5 (Design A).**

| Condition | Trained MLP @ L2 | Untrained MLP @ L2 | Gap |
|---|---|---|---|
| Real | 0.904 ± 0.006 | 0.685 ± 0.014 | **+0.220** |
| Within-shuffled | 0.884 ± 0.003 | 0.685 ± 0.014 | **+0.199** |
| Global-shuffled | 0.825 ± 0.008 | 0.685 ± 0.014 | **+0.140** |

After holding position literally constant, Feature B retains a positive
gap in all three conditions, with a monotonic real > within > global
gradient.

**Residual-after-position probe (Design B3).** R² on the test residual:

| Condition | Trained R² @ L2 | Untrained R² | Gap |
|---|---|---|---|
| Real | +0.678 ± 0.026 | +0.210 ± 0.012 | **+0.468** |
| Within-shuffled | +0.639 ± 0.024 | +0.210 ± 0.012 | **+0.429** |
| Global-shuffled | +0.431 ± 0.022 | +0.210 ± 0.012 | **+0.222** |

After statistically controlling for the per-position baseline, Feature B
retains a substantial R² gap in all three conditions with the same
monotonic gradient.

**Joint reading.** Position-correlation accounts for approximately +0.07
of the +0.29 headline gap. The remaining +0.22 is still well above the
locked falsification threshold of +0.10. The graded form's null direction
is falsified even under the most conservative control.

The cross-condition gradient (real > within > global) survives both
controls. Feature B encoding strength is structure-dependent, not a
pure position artifact.

#### Summary

Feature A (carry-through prediction) confirmed at +0.17 gap. Combined
with the maze starting-cell carry-through finding, this is the only
substantive ex-ante claim that the framework passed on two domains of
different shape.

Feature B falsified at +0.29 unconditional; +0.22 even at fixed
position. The graded N-criterion is wrong about even simple aggregations
being absent.

The position-correlation issue is a methodological contribution (§4.10).

#### Post-lock carry-through follow-ups

After the original HTTP result, we locked three narrower follow-up
predictions to test whether carry-through was only first-slot copying.
All three reuse the trained HTTP checkpoint and probe at the current
request's size token `sz_k`.

| Follow-up | Source feature | Best MLP gap | Intervention reading |
|---|---|---:|---|
| Same-request path | `p_k` | **+0.809** | Strongly useful for predicting `sz_k`; weak after `sz_k`. |
| Previous-request path | `p_{k-1}` | **+0.674** | Modestly useful for predicting `sz_k`; tiny after `sz_k`. |
| Recent-large-response path | most recent earlier `p_j` with `size_bin_j >= 5` | **+0.621** full lag; **+0.514** for lag `>= 3` | Strongly decodable, but weakly useful under lag-filtered corruption. |

The third test is the useful LessWrong result: the source event is not a
first slot and not a fixed relative offset; it is selected by content.
The probe result is strong, but the intervention result is cautious. This
is evidence for residual persistence of structured recent context, not
proof of a robust semantic memory circuit.

### 5.9 Joint reading of the two pre-registered tests

| Pre-registered claim | Maze | HTTP |
|---|---|---|
| Carry-through (input-slot feature encoded) | ✓ (starting cell encoded) | ✓ (Feature A encoded) |
| Null on computed irrelevant feature | n/a | ✗ falsified (even at fixed position) |
| Specific positive (predictively-relevant) | ✗ falsified (distance) | n/a |

**Carry-through: 2-for-2 on the original pre-registered tests, with three
HTTP follow-up confirmations.** Confirmed when introduced as the explanatory
mechanism for the maze result, again as a forward-looking prediction on a
different domain shape (HTTP Feature A), and then in a ladder of HTTP
follow-ups that moves from local fixed-offset copying to content-selected
earlier-event recoverability.

**Broader "predictive relevance drives encoding": 0-for-3 on risky
predictions.** Maze distance was predicted encoded but is not. HTTP
Feature B was predicted null but is encoded (even after position-control).

**Position-correlation as a probe confound.** The HTTP Feature B result
surfaces this as a methodological issue: features correlated with
positional information inherit positional encoding's gap. We capture
this as the second methodological contribution (§4.10) and recommend
both Design A and B3 as default for future probe-based work whenever
the target feature correlates with token position.

---

## 6 Relevance and Broader Implications

### 6.1 What the cities case does and does not show

The cities Procrustes-aligned overlay is visually striking but, by our
own analysis, the weakest support for an emergent computed world model:
most of the geographic structure is in the embedding table before any
transformer block acts. Embedding-table structure arising from token
co-occurrence statistics is interesting but qualitatively different
from the Othello and music cases, where most encoding is built up
across transformer layers.

### 6.2 Why pre-registration matters even when the prediction loses

The maze experiment cost us our central scientific bet. We argue it is
*more* valuable for having falsified than it would have been with a
confirm: the audit trail makes retroactive reinterpretation impossible
and forces an honest revision (the graded form). The falsification
also pointed at a specific mechanism (carry-through) that the strict
framework would have dismissed.

### 6.3 Beat null and starting-cell encoding are jointly informative

Music beat-in-measure and maze starting cell are both predictively
irrelevant features. The strict form predicts both null;
beat is null, starting cell is encoded. Under the strict form this is
an inconsistency. Under the graded form they are *jointly informative*:
beat is not at any single positional slot of the input and would
require active computation, so carry-through does not apply (null
predicted, null observed); starting cell sits at the first non-BOS
token of the sequence, so carry-through applies (encoded predicted,
encoded observed). The two results together identify the mechanism
that the strict framework lacked.

### 6.4 Methodological implications for the field

The combination of multi-seed reporting, probe and transplant
convergence, per-layer ablation, destroyed-structure controls,
pre-registration with a git audit trail, AND position-controlled
probing constitutes a reasonable bar for representational-content
claims about small transformers. None of these elements is individually
novel; the combination is rarer than the literature suggests.

We make two specific recommendations:

1. **Pre-register null predictions with thresholds.** A null prediction
   tested without a written threshold is not falsifiable. The
   pre-registered threshold (gap ≤ 0.10 for maze, ≤ 0.10 for HTTP) is
   what allowed us to call the falsifications cleanly.

2. **Run position-controlled probing whenever the target correlates
   with token position.** Either Design A (fixed k) or Design B3
   (residual after position) suffices; running both is best. Without
   such a control, position-correlated targets will produce a
   trained-vs-untrained gap that reads as encoding but is the
   model's positional representation being more refined than random.

---

## 7 Discussion

### 7.1 The experimental arc as it actually unfolded

The scientific timeline can be seen at a glance in
[`figures/01_experimental_arc.md`](../../figures/01_experimental_arc.md). The
prose summary, with the specific decisions made at each step:

The work began with the strict N-criterion as the falsifiable target. Its
positive direction (Othello board state required and recovered) had been
established by prior work. The risky direction was the null: features
predicted to be irrelevant should not appear in the residual stream
even if they are decodable from the input. The music beat null (§5.2),
identified post-hoc, gave us preliminary confidence that the null
direction was real but did not pre-register the claim. We chose maze
navigation as the first pre-registered test specifically because
"required" is task-structurally definable and the destroyed-structure
controls are unambiguous to construct.

The maze predictions were locked at commit `aa025b1` (2026-05-27). P4
(starting cell NULL) was the load-bearing prediction. After training
and probing, P4 returned a +0.15 gap, falsifying the strict form at
above the locked +0.10 threshold. We then asked the question: why does
an irrelevant feature persist? The simplest reading was that the
starting cell sits at a positionally distinct input slot, self-attention
can copy it forward at low cost, and the next-token objective never
penalizes carrying it. We named this mechanism architectural
carry-through and adopted the graded form of the N-criterion (§2.3).

HTTP was the second pre-registered test, designed to span the graded
form's two-category split. Predictions locked at commit `3b25ed3`
(2026-05-31): Feature A (first request's size_bin) predicted encoded
via carry-through; Feature B (cumulative large-response count) predicted
null because it would require active aggregation. Feature A confirmed at
+0.17 gap, validating carry-through as a forward-looking prediction on
a domain it was not designed against. Feature B apparently falsified at
+0.29 gap. The graded form's null direction failed on its first
ex-ante test.

We then asked whether the Feature B falsification was a true encoding
finding or a probe confound. Position-in-session correlates with the
cumulative count, and the trained model develops sharp positional
representations. We ran three post-hoc controls: a position-control
probe (gap +0.43, larger than Feature B), a within-position probe at
fixed k=5 (gap +0.22), and a residual-after-position probe (gap +0.47
in R²). All three agreed that Feature B is encoded above untrained
baseline even after controlling for position. The falsification stands.
But it is reduced in magnitude (+0.22 not +0.29), and the controls
yielded a methodological contribution we believe is reusable.

Where we land: architectural carry-through is the one substantive
predictive claim that survived ex-ante testing on two domains. The
broader N-criterion in either form does not survive. The
position-control diagnostics are the second methodological contribution
of the paper.

### 7.2 What the strict N-criterion does not survive

The strict biconditional, F is encoded iff next-token prediction
requires F, does not hold. A plainly irrelevant feature, predicted absent
in advance via the audit trail, was recovered with substantial signal.
The biconditional framing predicted this should not happen.

### 7.3 What the graded form does say, and where it also fails

The graded form retains predictive relevance as the dominant driver of
encoding with carry-through as a second mechanism for input-borne
features. The positive direction (carry-through encoded) is confirmed
on both pre-registered domains. The null direction (computed irrelevant
features are absent) is falsified on HTTP Feature B even after
position-control. We do not have a way to rescue the null direction in
the graded form without further weakening it past the point of
informativeness.

The trade-off is plain. The strict form is more informative and more
falsifiable. The graded form is more defensible on the positive
direction (and confirmed ex-ante on two domains there) but still
incorrect on the null direction. We adopt the graded form's positive
half (carry-through) as the working empirical claim and treat its null
half as falsified by this paper's pre-registered tests.

---

## 8 Challenges and Limitations

### 8.1 Scale and external validity

All models are 0.27M-13M parameters on synthetic or semi-synthetic data.
Whether the carry-through claim, or any other claim, extends to
frontier-scale models on natural language is not established by this
study. Our framing scopes the question to small models on structured
tasks accordingly.

### 8.2 "Required" remains imperfectly operationalized

We define "required" from task structure (model-independently) where we
can. This still leaves room for proxies and partial reliance. The graded
form absorbs this at the cost of being weaker and less surprising.

### 8.3 Probe nulls have weak power

A null probe is consistent with (a) feature unencoded, (b) feature
encoded non-linearly or in attention not the residual stream, or
(c) probe lacking capacity. We mitigate via (i) the same probe family
that recovers positive controls in the same domain, and (ii) transplant
corroboration where possible. We do not offer formal power analysis;
this is the single most defensible criticism of our null results.

### 8.4 Linear-vs-MLP gap at the upper end

We report a maximum linear-vs-MLP gap of 0.13 across positive cases.
For the strict Nanda strong claim, 0.131 absolute accuracy on a
0.94-trained probe is approximately 14% of the available headroom. We
do not claim our results establish linear encoding in the strict Nanda
sense; we claim approximate linear recoverability in the chosen basis.

### 8.5 The within-shuffled cities oddity

London within-shuffled scores higher on the cities probe than London
real (§5.3). This contradicts the naive monotonic-destruction prediction
and we do not have a clean test for the proposed explanation
(within-shuffling frees the model to specialize on geographic clustering
by removing the graph-topology constraint).

### 8.6 Limited number of pre-registered domains

Two pre-registered tests are not many. The strong form of the
pre-registration argument requires multiple independent tests; we view
the maze and HTTP experiments as initial prospective tests and a model
for future pre-registrations rather than as a conclusive demonstration.

### 8.7 P1 and P2 of the maze predictions were methodologically flawed

Two of the four maze predictions (row and column) were not meaningful
tests because the probe target was a deterministic function of token
identity. This is a predictor's error, caught on inspection of the data
rather than by the audit-trail mechanism itself. Pre-registration is
only as good as the prediction's construct validity.

### 8.8 The position-control diagnostic was post-hoc

The position-control probe and Designs A and B3 were all run after the
Feature B falsification surfaced. In future pre-registered domains we
would commit in writing to running a position-control alongside any
computed-feature probe whose target correlates with token position.

### 8.9 Mechanistic depth is coarse

We use residual-stream probes and full-residual transplant. We do not
perform path patching, composition-score analysis, or attention-head
decomposition. Going from "the feature is represented" to "this
specific circuit computes it" requires finer tools we have not applied.
We treat this as a scope choice, not an oversight.

### 8.10 Carry-through risks being "obvious in hindsight"

A reasonable critic could say carry-through is a fairly obvious
architectural side effect: of course an input-slot token persists
through layers of self-attention. We acknowledge this. What gives the
carry-through claim its empirical weight is not its novelty but the
fact that it was committed in writing as the Feature A HTTP prediction
*before* any HTTP data was downloaded, and was then confirmed at the
predicted magnitude. Without the pre-registration, carry-through is
post-hoc rationalization of the maze result. With pre-registration on
a second domain it is an ex-ante validated mechanism. The discipline
is what converts an obvious-sounding intuition into a falsifiable claim.

---

## 9 Future Directions

### 9.1 A new risky prediction the revised framework should make

The natural next test, following the audit-trail protocol used here,
is a domain in which we pre-register predictions for *two*
predictively-irrelevant features chosen deliberately to span the
carry-through split: one present at a positionally distinct input slot
(predicted encoded) and one requiring active computation (predicted
null). A TCP-state-recovery task offers source IP (first packet,
distinct slot) and total retransmission count (must be computed across
packets); a multi-turn dialog-state task could be structured similarly.
A confirm on the carry-through prediction and falsification on the
null prediction would earn the graded framework the next layer of
credibility. Either outcome is more useful than not running the test.

### 9.2 Position-controlled probing as default protocol

We recommend that future probe-based interpretability work include a
position-control probe by default whenever the target feature has any
correlation with token position. Designs A and B3 are cheap (one
additional probe run per condition) and convert ambiguous gaps into
interpretable verdicts. The HTTP Feature B case shows the value
concretely: the unconditional falsification headline is +0.29; the
position-controlled headline is +0.22. The qualitative verdict is
unchanged, but the magnitude attributable to genuine encoding versus
position confound is now separable. This separation should be
default-on rather than opt-in.

### 9.3 Mechanistic decomposition of carry-through

The carry-through mechanism as currently stated is a high-level claim
about self-attention behavior. A natural follow-up is to localize the
relevant attention heads, characterize the copying pattern across layers,
and verify that ablating those heads collapses the encoding. The HTTP
recent-large-response case is the most interesting target because it is
strongly decodable but only weakly causal under lag-filtered corruption.
The maze and HTTP models studied here are small enough that head-level
decomposition is tractable.

### 9.4 Scaling

Re-running the maze, HTTP, and music experiments at 100M+ parameters
would test whether the calibration and pre-registered findings hold at
scale. The methodology is scale-portable.

---

## 10 Conclusion

This paper offers three contributions: a methodological protocol for
pre-registered representational analysis in small next-token
transformers, one ex-ante-validated mechanism (architectural
carry-through), and one identified methodological failure mode
(position-correlation as a probe confound) with two controllable
diagnostics.

The N-criterion as a strong predictive theory of which features emerge
does not survive its own pre-registered tests. The mechanism that does
survive, carry-through, is a simple architectural side effect of
self-attention. Its empirical weight comes not from novelty but from
having been written down before the second pre-registered domain was
trained and then confirmed at the predicted magnitude, then stress-tested
by three locked HTTP follow-ups beyond first-slot copying. We treat this
not as a victory for the framework but as a demonstration of what
pre-registered, audit-trailed work makes possible: even an
obvious-in-hindsight claim earns scientific status only when its
prediction is committed in writing in advance.

We hope the combined protocol, multi-seed reporting plus
destroyed-structure controls plus probe-and-transplant convergence plus
pre-registration plus position-controlled probing, becomes a reasonable
bar for the small-model branch of mechanistic interpretability.

---

## References

- Alain, G., and Bengio, Y. (2016). *Understanding Intermediate Layers
  Using Linear Classifier Probes.* arXiv:1610.01644.
  <https://arxiv.org/abs/1610.01644>
- Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T.,
  Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C.,
  Grosse, R., McCandlish, S., Kaplan, J., Amodei, D., Wattenberg, M.,
  and Olah, C. (2022). *Toy Models of Superposition.*
  <https://transformer-circuits.pub/2022/toy_model/index.html>
- Hewitt, J., and Liang, P. (2019). *Designing and Interpreting Probes
  with Control Tasks.* EMNLP-IJCNLP 2019. arXiv:1909.03368.
  <https://arxiv.org/abs/1909.03368>
- Karpathy, A. (2022). *nanoGPT.* <https://github.com/karpathy/nanoGPT>
- Li, K., Hopkins, A. K., Bau, D., Viégas, F., Pfister, H., and
  Wattenberg, M. (2022). *Emergent World Representations: Exploring a
  Sequence Model Trained on a Synthetic Task.* arXiv:2210.13382.
  <https://arxiv.org/abs/2210.13382>
- Nanda, N., Lee, A., and Wattenberg, M. (2023). *Emergent Linear
  Representations in World Models of Self-Supervised Sequence Models.*
  arXiv:2309.00941. <https://arxiv.org/abs/2309.00941>
- Park, K., Choe, Y. J., and Veitch, V. (2023). *The Linear
  Representation Hypothesis and the Geometry of Large Language Models.*
  arXiv:2311.03658. <https://arxiv.org/abs/2311.03658>
- Internet Traffic Archive. *NASA-HTTP: Two Months of HTTP Logs from the
  KSC-NASA WWW Server.* <https://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html>

---

<small>This is an independent work and does not represent any of my past
or present employers.</small>

AI-assisted implementation disclosure: the implementation was heavily
AI-assisted. Data pipelines, probe and transplant scripts, training
configs, figure code, and iterative debugging were developed with
Claude/Codex assistance. The claims, experiment framing, pre-registered
predictions, and interpretation are the author's responsibility.

## Appendix A: Reproducibility and Data Sources

All code, data preparation pipelines, training configurations, probe
and transplant scripts, multi-seed runners, and the pre-registered
prediction files are publicly versioned at the project repository:
<https://github.com/Raghavan1988/LatentWorldsGPT>.

Each headline number in the pre-registered Maze and HTTP sections can be
reproduced end-to-end on an Apple MPS laptop in under 8 hours. Larger
scale reruns discussed in §8.1 would require GPU rental.

The pre-registration audit trail is verifiable via:

```bash
git log --diff-filter=A predictions/predictions_maze_navigation.md
git log --diff-filter=A predictions/predictions_http_log_sequences.md
```

These commands show commit hashes `aa025b1` (maze) and `3b25ed3`
(HTTP), predating the corresponding model training, data generation, and
probe runs. The post-lock HTTP follow-ups are similarly auditable at
commits `88155b4`, `0d115c2`, and `8d671ca`; the Maze C3 follow-up is
auditable at `d7dabf6`.

| Domain | Data source / construction | Notes |
|---|---|---|
| Cities | OpenStreetMap-derived route corpora via OSMnx (<https://www.openstreetmap.org/>, <https://osmnx.readthedocs.io/>) | Used as a calibration domain; not pre-registered. |
| Othello | Generated random legal games | Positive control against Othello-GPT-style prior work. |
| Music | music21 Bach chorale corpus and related symbolic-music tooling (<https://web.mit.edu/music21/>) | Used for voice-leading/chord/beat probes and transplant tests. |
| Flight | ADS-B trajectories from the `traffic` quickstart/sample tooling, with phase labels from OpenAP (<https://traffic-viz.github.io/>, <https://openap.dev/>) | Smallest model/data regime in the study. |
| Symmetric group | Generated walks on finite symmetric groups | Synthetic algebraic calibration domain. |
| Maze | Generated 8x8 recursive-backtracking mazes with shortest paths | First pre-registered test. |
| HTTP | NASA-HTTP July/August 1995 logs from the Internet Traffic Archive | Uses `NASA_access_log_Jul95.gz` and `NASA_access_log_Aug95.gz`. |

## Appendix B: Per-Domain Configurations

| Domain | Model | Params | Train tokens | Vocab | Main evidence type |
|---|---|---:|---:|---:|---|
| Cities (London) | nanoGPT small | 5.3M | 1.13M | 666 | Probe + shuffle controls |
| Othello | medium_othello | 4.0M | 2.5M | 67 | Probe + transplant |
| Music (expanded) | small | 1.4M | 358k | 60 | Probe + transplant |
| Flight (ADS-B) | tiny_flight | 0.27M | 46k | approximately 80 | Probe + transplant |
| Symmetric group (S_5) | small | 1.5M | approximately 200k | approximately 125 | Probe |
| Maze (8x8) | maze_2M | 2.0M | 1.5M | 67 | Pre-registered probe + transplant |
| HTTP (NASA, Jul+Aug 1995) | small_http | 0.81M | 8.08M | 52 | Pre-registered probe + interventions |

## Appendix C: Locked Prediction Audit

| Prediction file | Commit | Locked prediction | Observed result | Verdict |
|---|---|---|---|---|
| `predictions_maze_navigation.md` | `aa025b1` | Distance-to-goal encoded; starting cell null at late positions | Distance gap `+0.009`; starting-cell gap `+0.152` | Both substantive claims falsified |
| `predictions_http_log_sequences.md` | `3b25ed3` | Feature A encoded via carry-through; Feature B null | Feature A `+0.168`; Feature B `+0.291` raw and `+0.220` fixed-position | Feature A confirmed; Feature B falsified |
| `predictions_http_same_request_path_carrythrough.md` | `88155b4` | Same-request path `p_k` recoverable at `sz_k` | MLP gap `+0.809` | Confirmed |
| `predictions_http_previous_request_path_carrythrough.md` | `0d115c2` | Previous-request path `p_{k-1}` recoverable at `sz_k` | MLP gap `+0.674` | Confirmed |
| `predictions_http_recent_large_response_path_carrythrough.md` | `8d671ca` | Most recent earlier large-response path `p_j` recoverable at `sz_k` | MLP gap `+0.621`; lag `>= 3` gap `+0.514` | Confirmed as recoverability; weak causal-usefulness at long lag |
| `predictions_maze_c3_carrythrough.md` | `d7dabf6` | Fourth maze path cell `C3` recoverable at late positions | Primary MLP gap `+0.053`; late-only MLP gap `+0.032` | Ambiguous / not confirmed |

## Appendix D: Headline Result Tables

### D.1 Main Pre-Registered Maze and HTTP Results

| Domain | Target | Prediction | Main metric | Verdict |
|---|---|---|---:|---|
| Maze | Distance to goal | Encoded, gap `>= +0.20` | `+0.009` MLP gap | Falsified |
| Maze | Starting cell | Null, gap `<= +0.10` | `+0.152` MLP gap | Falsified |
| HTTP | Feature A: first request `size_bin` | Encoded, gap `>= +0.10` | `+0.168` MLP gap | Confirmed |
| HTTP | Feature B: cumulative large-response count | Null, gap `<= +0.10` | `+0.291` raw; `+0.220` fixed-position | Falsified |

### D.2 Real / Within-Shuffled / Global-Shuffled Comparisons

| Test | Real | Within-shuffled | Global-shuffled | Reading |
|---|---:|---:|---:|---|
| Maze distance-to-goal probe gap | `+0.009` | `+0.011` | `+0.007` | Flat null |
| Maze starting-cell probe gap | `+0.152` | `+0.031` | `+0.154` | Real approximately global > within |
| Maze starting-cell transplant lift | `+0.155` | `+0.017` | `+0.155` | Mirrors probe pattern |
| HTTP Feature A probe gap | `+0.168` | `+0.134` | `+0.163` | Carry-through confirmed |
| HTTP Feature B raw probe gap | `+0.291` | `+0.236` | `+0.303` | Null falsified; position confound suspected |
| HTTP position-control gap | `+0.427` | `+0.541` | `+0.404` | Position strongly encoded |
| HTTP Feature B fixed `k=5` gap | `+0.220` | `+0.199` | `+0.140` | Feature B remains after literal position control |
| HTTP Feature B residual-after-position R2 gap | `+0.468` | `+0.429` | `+0.222` | Residual signal remains after statistical position control |

### D.3 HTTP Carry-Through Follow-Ups

| Follow-up | Best linear gap | Best MLP gap | Corrupt-source delta NLL on `sz_k` | Delta NLL after `sz_k` |
|---|---:|---:|---:|---:|
| Same-request path `p_k` | `+0.787` | `+0.809` | `+4.029` | `+0.083` |
| Previous-request path `p_{k-1}` | `+0.535` | `+0.674` | `+0.187` | `+0.024` |
| Recent-large-response path `p_j`, lag `>= 1` | `+0.558` | `+0.621` | `+0.084` | `+0.009` |
| Recent-large-response path `p_j`, lag `>= 3` | `+0.460` | `+0.514` | `+0.013` | `+0.003` |

### D.4 Calibration-Domain Summary

| Domain / feature | Headline result | Reading |
|---|---|---|
| Othello board state | approximately 94% per-cell probe accuracy; transplant lift positive | Positive control reproduced |
| Music voice-leading | transplant lift `+0.889` at L2 | Clean positive causal signal |
| Music chord | probe gap approximately `+0.09` | Weak positive |
| Music beat-in-measure | probe gap approximately `+0.006`; transplant null | Clean null |
| Cities geographic location | real `+0.55`, within `+0.67`, global `+0.01` | Useful but confounded by token/co-occurrence structure |
| Flight phase | real probe gap approximately `+0.11` | Moderate signal |
| Symmetric-group partial product | real probe gap approximately `+0.05` | Partial signal |

Full layer-by-layer and seed-level result tables remain in
[`docs/results/results_maze_navigation.md`](../results/results_maze_navigation.md),
[`docs/results/results_http_log_sequences.md`](../results/results_http_log_sequences.md),
and the per-domain history files, but the tables above contain the
numbers needed to understand the main claims without opening those files.
