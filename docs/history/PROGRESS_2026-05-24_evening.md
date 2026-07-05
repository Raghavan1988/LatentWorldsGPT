# Progress note — 2026-05-24 evening session

Work done after the earlier PROGRESS_2026-05-24.md snapshot.

## Summary

Two new evaluation scripts written and ran across three cities. PLAN.md Phase 3
(baselines, except LSTM) and Phase 4 (probe suite, except destroyed-structure
control) are implemented. The cross-city probe results support the central
hypothesis (the trained model contains a geographically-decodable representation)
but with important methodological caveats that emerged during the work.

---

## What got built

### `eval/baselines.py` — Phase 3

Analytical baselines (no training needed), plus a long-range coherence metric:

- **Uniform random** — `P(t) = 1/vocab_size`
- **Unigram frequency** — empirical token frequency from `train.bin`
- **1st-order Markov over the graph** — `P(B|A)` from bigram counts, Laplace smoothing
- **2nd-order Markov** — `P(C|A,B)` from trigram counts, with backoff to bigram, then unigram
- **GPT cross-entropy on the same real → real protocol** so the comparison is apples-to-apples
- **Long-range coherence metric**: for true val routes, take a short prefix, generate K more
  hops with the model and with Markov, measure graph-shortest-path distance to the true
  destination. Markov has no notion of destination; GPT can infer it from the trajectory.

Deferred (PLAN.md Phase 3 still requires this):
- Same-parameter-count **LSTM** baseline. Requires training another model per city —
  ~20–30 min of MPS time per city. Not started.

### `eval/probe.py` — Phase 4

- **Linear probe** (effectively ridge regression via weight decay), recovers (x_m, y_m).
- **MLP probe** — 2 hidden layers × 256 units, GELU. For the linear-vs-nonlinear comparison.
- **Per-layer sweep** — residual stream at the input embedding (layer 0) and after each
  transformer block (L1..L6).
- **Untrained-model control** — same architecture, random init.
- **Position-level split** — standard probe methodology: 80% of probe POSITIONS for
  training, 20% for test. Tokens may appear in both sets.
- **Node-level split** — the probe-capacity-controlled test: 80% of unique TOKENS for
  training, the other 20% only at test time. The probe is asked to localize tokens it has
  never seen during training. A probe that just memorized a per-token lookup fails this
  test; a probe that learned a continuous geographic map holds up.
- Reports R², median Euclidean error in meters, p90 error in meters.

**Loss function choice.** MSE on **standardized planar (x_m, y_m)** coordinates. Raw
(lat, lon) inputs are projected to local Cartesian (x_m, y_m) via equirectangular
projection centered on the city's mean (lat, lon). This is the intellectually honest
choice for city-scale (≤ 50 km) geographic regression — distortion < 0.5% — and makes
"meters of error" the directly-interpretable physical unit. Naive MSE on raw (lat, lon)
would over-weight latitude (1° lat ≈ 111 km, but 1° lon ≈ 70 km at London's 51°N) and
produce a non-physical loss.

Deferred (PLAN.md Phase 4 still requires this):
- **Destroyed-structure control** — train a separate model on identity-shuffled routes
  and re-run the probe. If R² stays high on that model, the geographic signal we see was
  a token-co-occurrence artifact, not the graph's geometry. The most important missing
  control.

---

## Phase 3 results — London

Real → real transition cross-entropy (filtering both sides to be real intersections,
so the comparison protocol is identical for every model):

| Baseline | val CE | val ppl | gen CE | gen ppl |
|---|---:|---:|---:|---:|
| Uniform random | 6.497 | 663.0 | 6.497 | 663.0 |
| Unigram (token frequency) | 5.966 | 389.9 | 5.974 | 393.1 |
| **1st-order Markov** | 0.910 | 2.48 | 0.918 | 2.50 |
| **2nd-order Markov** | 0.381 | 1.46 | 0.391 | 1.48 |
| **LatentCityGPT (small, smoke-trained)** | **0.207** | **1.23** | **0.275** | **1.32** |

LatentCityGPT clearly beats both Markov baselines on perplexity — the acceptance
criterion ("matches/beats Markov") is met. Note how much of the available signal Markov
already captures: from random's CE 6.5 to 1st-order Markov's 0.91 is most of it. The
remaining gap is what LatentCityGPT closes plus more.

Long-range coherence — 50 val routes, 4-token prefix, generate 20 more tokens, measure
graph-distance from the final generated node to the true destination:

| | LatentCityGPT | 1st-order Markov |
|---|---:|---:|
| Median hops to dest | **14.0** | 15.0 |
| Mean hops to dest | 15.2 | 15.8 |
| p90 hops to dest | **22.0** | 26.0 |
| Validity (final pos is a real node) | 50/50 | 50/50 |

GPT does beat Markov, but the median advantage is small (1 hop). The clearer
advantage is in the **p90** (worst cases): 22 vs 26 hops. Probably bigger advantage
on Manhattan/Boston where routes are longer.

---

## Phase 4 results — London, Manhattan, Boston

### Linear probe (the probe-capacity-controlled metric)

| | London | Manhattan | Boston |
|---|---:|---:|---:|
| City extent (diagonal) | ~3 km | ~22 km | ~24 km |
| Training visits per node | ~1,600 | ~590 | ~260 |
| **Position-level best R² (layer)** | **0.867** (embed) | 0.409 (L4) | 0.345 (L2) |
| Position-level untrained best R² | 0.428 (embed) | 0.142 (L6) | 0.127 (L6) |
| Position-level emergence gap | +0.44 | +0.27 | +0.22 |
| **Node-level best R² (layer)** | **0.639 (L3)** | **0.255 (L5)** | **0.233 (L3)** |
| Node-level untrained best R² | −0.053 | 0.074 | 0.058 |
| **Node-level emergence gap** | **+0.69** | **+0.18** | **+0.18** |
| Median error (m), node-level | 318 | 2244 | 2930 |
| Normalized error / city diagonal | ~10% | ~10% | ~12% |

### MLP probe (contaminated by lookup-memorization)

| | London | Manhattan | Boston |
|---|---:|---:|---:|
| Position-level trained R² | 0.998 | 0.949 | 0.882 |
| **Position-level untrained R² (memorization signal)** | **0.966** | **0.480** | **0.324** |
| Node-level trained R² | 0.775 | 0.688 | 0.570 |
| Node-level untrained R² | −0.10 | 0.088 | 0.104 |
| **Untrained MLP collapse on node-level** | **−1.07** | **−0.39** | **−0.22** |

---

## Key methodological discovery — MLP probe contamination

The MLP probe achieves R² 0.998 on the trained model at London (position-level). The
natural reading is "near-perfect recovery — the map is encoded." But the same probe also
gets R² 0.966 on the **untrained** model at the same split. The MLP is not measuring the
model's representation; it is doing per-token lookup memorization, exploiting the fact
that there are only 663 unique tokens and ~24 training positions per token.

The node-level split exposes this cleanly: trained on a disjoint set of tokens, the
untrained MLP collapses to R² ≈ 0 in all three cities. This is **not** the model failing;
it is the probe being denied the lookup it was using.

The trained MLP also drops on node-level (0.998 → 0.775 for London), but holds well
above its untrained counterpart. So the trained MLP DID find some real geographic
structure on top of the lookup — just not as much as the position-level R² implied.

**Implication for PLAN.md Phase 4's "linear ≈ MLP → linearly encoded" criterion:** the
criterion does not apply cleanly in our setting. Othello-GPT's 60 squares × 3 states
= 180 discrete labels make the MLP probe capacity-limited; our 663 continuous targets
do not. We cannot conclude "linear" or "nonlinear" encoding from this comparison alone.

---

## Cross-city pattern — the deeper layers matter for generalization

For each city, where in the network does the probe perform best?

| City | Position-level peak (linear) | **Node-level peak (linear)** |
|---|---|---|
| London | embed (L0) | **L3** |
| Manhattan | L4 | **L5** |
| Boston | L2 | **L3** |

On the position-level split (which the probe can shortcut via lookup), London's input
embedding wins. On the **node-level split, the best layer is always inside the
transformer** — including London's. The transformer is doing geometric work that
generalizes to unseen tokens. The embedding by itself does not.

This is the closest analogue to Othello-GPT's "world model is built in the residual
stream" finding that the experiment has produced so far.

---

## Honest summary statement

> Across London, Manhattan, and Boston, the trained model's residual stream contains
> a linearly-decodable geographic representation that generalizes to held-out tokens
> the probe was never trained on. The trained-vs-untrained linear-probe gap is positive
> in all three cities on the held-out-token test (+0.69, +0.18, +0.18 R²), with the
> geographic signal peaking in mid-transformer layers rather than in the raw token
> embedding. The MLP probe's higher headline numbers do not support a stronger claim
> — they substantially reflect probe lookup-memorization, as shown by the untrained-MLP
> collapse on the node-level test. Pending controls — destroyed-structure model, LSTM
> baseline, geographic-region holdout — are needed before a quantitative claim about
> probe accuracy is final.

---

## Still ahead (work not started this session)

- **Destroyed-structure control** — Phase 4. Adds a `--shuffle_nodes` flag to
  `prepare_city.py`, trains an identical-architecture model on shuffled routes, runs
  the probe. Required to fully establish that the recovered map comes from the graph's
  geometry rather than per-token co-occurrence frequencies.
- **LSTM baseline** — Phase 3. Same-parameter-count LSTM on the same corpus, same
  protocol. Confirms whether the result is "transformer-specific" or "any sequence
  model."
- **Causal patching** — Phase 5. Take the residual direction encoding location, patch
  in "I'm at node B" while the model is at node A, measure the shift in next-hop
  predictions. This is what would close the loop from "we found a representation" to
  "the model uses this representation."
- **Geographic-region holdout** — Phase 0 follow-up + Phase 2. Hold out a contiguous
  lat/lon sub-region instead of scattered destinations. The stronger generalization
  claim.
- **Proper-scale training** — see `STATUS_vs_OTHELLO-GPT.md` for what full training is
  expected to change.
