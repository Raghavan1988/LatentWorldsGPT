# LatentCityGPT vs Othello-GPT — current status and what proper training will show

This is a stock-take comparing the experiment's progress to the Othello-GPT lineage
it is modeled on, and a calibrated prediction of what changes when we move from
the current smoke-trained models to proper (medium-config, full-corpus) training.

> **Update — 2026-05-24 evening.** Phase 5 (`eval/causal.py`) and the
> destroyed-structure control (`--shuffle_routes` in `prepare_city.py`) were
> implemented and run. The destroyed-structure control surfaced an unexpected
> result that forced a reframe: **probe-decoded geographic clustering does NOT
> specifically require sequence learning** — a model trained on the same routes
> with token-order randomly shuffled has higher probe R² than the real-trained
> model. The thing that *does* require sequence training is graph-adjacency
> knowledge (P(A's real neighbors) = 0.984 real vs 0.063 shuffled). The
> claim-by-claim table and the bottom line have been revised accordingly. See
> `update_may24_final.md` for the full session narrative.

---

## Othello-GPT in one paragraph

Li et al. (2022, *Emergent World Representations*) trained a GPT to predict the next
move in random Othello games — token sequences with no board-state annotation. They
found that an MLP probe could recover the current board state (which squares are
occupied by which color) from the model's residual-stream activations with high
accuracy. Nanda's 2023 follow-up showed the representation is **linearly** decodable
once you frame the basis correctly ("I just moved here" basis change), and that
*patching* the residual direction encoding the board state causally shifts the
model's move predictions in the patched direction. Together those three findings —
**(1) a world state is recoverable from the residual stream, (2) it is encoded
linearly, (3) it is the actual cause of the model's outputs** — make Othello-GPT
a load-bearing interpretability result.

LatentCityGPT asks the same question with a continuous, real-world substrate:
streets instead of squares, (lat, lon) instead of {empty, black, white}.

---

## Claim-by-claim status

| # | Othello-GPT claim | LatentCityGPT counterpart | Status (post-destroyed-structure update) |
|---:|---|---|---|
| 1 | Trained transformer learns the rules of next-token prediction (legal moves) | Trained transformer learns the city graph (real neighbors) | ✅ **Verified, clean.** London valid-edge rate 99.7% on val, 99.3% on gen. Untrained baseline ≈ 0.3%. **Destroyed-structure control: 0.063** — separates cleanly. |
| 2 | Beats simple baselines on the next-token task | Beats Markov-2 on perplexity (1.23 val vs 1.46) and beats Markov-1 (modestly) on long-range coherence | ✅ Verified for London. Margins are real but smaller than one might hope on coherence — see `PROGRESS_2026-05-24_evening.md`. |
| 3 | World state recoverable from residual stream activations | (lat, lon) recoverable from residual stream activations | 🟡 **Verified but reframed.** Linear probe trained R² 0.87 (London) / 0.41 / 0.34 on position-level; node-level 0.64 / 0.26 / 0.23. **However:** the destroyed-structure model achieves an *even higher* R² (0.97 node-level on shuffled London) — so the recoverability source is route co-occurrence (set-membership), not sequence learning per se. |
| 4 | World state survives held-out conditions | (lat, lon) recoverable for held-out **tokens** the probe never saw during training | 🟡 Verified at the node-level split, but with the same caveat as row 3: the shuffled-trained model passes node-level too (R² 0.965), so this is testing probe extrapolation over co-occurrence-clusters, not a sequence-specific representation. |
| 5 | World state is encoded *linearly* (Nanda's strong claim) | Linear probe matches MLP probe → encoding is linear | ❌ **Not cleanly testable in our setting.** MLP probe is contaminated by lookup memorization (untrained-MLP node-level collapse). The standard linear-vs-MLP criterion does not apply when targets are continuous and tokens are few. |
| 6 | World state is computed by the transformer (deeper layers better) | Map signal is built up in the transformer's middle layers | 🟡 **Mixed.** On node-level split, peak signal is in L2–L5 across all cities — the transformer matters. But on position-level the embedding is best. The shuffled-trained model's peak is also in L2 with very high R², so "transformer enriches the cluster" is happening for any geographic-co-occurrence training, not just sequence training. |
| 7 | Activation patching the world state changes the model's outputs (causal claim) | Patching the probe-direction toward target B shifts P(B's neighbors) above random-direction patches | ❌ **Implemented; current design does NOT support the causal claim.** Target-vs-random patching shows 79.5% target-beats-random on the real model, but 75.0% on the shuffled-trained model that cannot actually route. The pseudoinverse-direction patch operates on the model's clustering geometry, which exists in both real and shuffled models. A cleaner test (e.g., activation-transplant with a real `a_B` from a B-position) is needed. |
| 7′ | (Phase 5′ — adjacency-as-causal-prerequisite) | The model uses graph-adjacency knowledge to predict next hops | 🟢 **Strongly suggested by training-data ablation.** P(A's neighbors) is 0.984 on the real model and 0.063 on the shuffled-trained model — only sequence-order training produces graph-adjacency-using behavior. This is correlational with training data, not yet a residual-stream causal intervention. |

---

## What we can say with high confidence (post-destroyed-structure-update)

### Positive

- **The model genuinely learns the graph.** 99.7% next-step valid-edge rate on real London;
  generalizes to routes whose destinations were never training endpoints (99.3% on gen).
  **Destroyed-structure ablation makes this clean: a model trained on the same routes with
  token-order randomly shuffled achieves only 0.063 mass on a node's real neighbors.** This
  is the cleanest separator we have and the most defensible claim of the project.
- **A representation of geographic location is decodable from the residual stream.** Linear
  probe trained-vs-untrained gap is positive in all three cities on the held-out-token test.
- **The transformer enriches the cluster** (probe peak is mid-network on the held-out-token
  test, not at the embedding) — across both real and destroyed-structure-trained models.

### Negative / important caveats

- **The "linearly encoded" claim cannot be made with this setup.** MLP probe is contaminated
  by lookup memorization.
- **The simple "an emergent metric map" framing was too coarse.** The destroyed-structure
  control showed that geographic clustering of activations emerges from *any* training signal
  with geographic co-occurrence — including a permutation-invariant one (within-route shuffle).
  Sequence learning is *not* a prerequisite for geographic clustering in activations; it is
  a prerequisite only for graph-adjacency-using behavior. These are two distinct phenomena.
- **The causal Phase 5 claim is NOT supported by the current data.** The target-vs-random
  patching signal we see on the real model is statistically indistinguishable from the same
  signal on the destroyed-structure-trained model (79.5% vs 75.0% target-beats-random). The
  pseudoinverse-direction patch operates on the probe's clustering geometry, which both
  models have. A cleaner test (activation-transplant style) is needed.
- **Probe quality degrades with vocab size** at the smoke-training scale. London 0.64
  → Manhattan 0.26 → Boston 0.23 (node-level linear R²). Almost certainly under-training
  rather than architecture-limited, but unverified.

### Difference vs Othello-GPT worth noting

Othello-GPT's headline interpretability story is *"the model computes the world state inside
the transformer; we can decode it linearly from the right intermediate layer, and patching
that direction causally controls the next move."* After the destroyed-structure control, our
honest story is two-part: *"(a) The model contains a geographically-decodable representation
that emerges from training, but this emerges from any geographic co-occurrence signal — not
specifically from learning to predict sequences. (b) Graph-adjacency-using behavior IS
specifically sequence-learned: only models trained on real routes achieve high probability on
a node's real graph neighbors."* Part (a) is the weaker analogue; part (b) is genuinely
mechanistically interesting but is shown by training-data ablation rather than by causal
residual-stream intervention.

---

## What proper (full-scale) training will likely show

"Proper" here means: medium-config model (~25–30M params, `block_size=512, n_layer=8,
n_head=8, n_embd=512`), full corpus (`n_shortest=200k --n_walks=50k --n_gen=20k`),
`max_iters=30,000`, on a CUDA GPU.

### Calibrated predictions

| Quantity | Smoke (current) | Proper-training prediction | Confidence |
|---|---:|---:|---|
| Valid-edge rate on val | 99.7% (London) | 99.95%+ — essentially saturated | High |
| Val perplexity | 1.27–1.65 | 1.05–1.15 — close to graph's per-step entropy floor | High |
| **Linear probe R², node-level, best layer** | London 0.64, Manhattan 0.26, Boston 0.23 | London 0.85+, Manhattan 0.70+, Boston 0.55+ | Medium — assumes the smoke-level result is genuinely under-trained, not architecturally limited. |
| Median meters error | London 318 m, Manhattan 2244 m, Boston 2930 m | London <100 m, Manhattan <500 m, Boston <1000 m | Medium |
| Trained-vs-untrained linear gap (node-level) | +0.18 to +0.69 R² | Larger across all cities | High — the untrained baseline doesn't move; the trained number is what improves |
| Peak layer (node-level) | L3–L5 | Probably similar; possibly shifts later if the model uses depth more effectively | Low — current evidence is consistent with several outcomes |

### What proper training will NOT change

- **The MLP-vs-linear contamination problem.** This is a methodological artifact of our
  continuous target + few-tokens setting, not a training artifact. We need a different
  test (e.g., probe with held-out *node positions*, or a probe with parameter count tied
  to the linear probe's) regardless of how well we train the model.
- **The need for the destroyed-structure control.** Even at proper scale, we can't claim
  the recovered map comes from graph geometry rather than co-occurrence statistics until
  we train the shuffled-route control.
- **The causal claim.** Phase 5 activation patching is a separate experiment; better-trained
  models don't get us closer to it without the actual patching code.

### What proper training MIGHT change

- The **peak-layer story**. With more training, the transformer might compose a sharper,
  later-layer geographic representation, moving the peak deeper (more Othello-GPT-like).
  Or it might not — the embedding-table-clustering pattern could be a stable attractor.
  Genuinely unclear from current data; this is one of the more interesting open questions.
- The **long-range coherence advantage** over Markov. Smoke training gave a modest median
  advantage (1 hop). Proper training might extend the gap substantially as the model gets
  better at inferring destinations from longer prefixes.

---

## What's still ahead for a complete project

Re-ordered after the destroyed-structure finding. Priorities have shifted:

1. **A STRONGER destroyed-structure control: globally shuffled tokens.** The
   current `--shuffle_routes` preserves route SET-MEMBERSHIP, which itself
   carries geographic signal. A global shuffle across the whole stream should
   break that too. Predicted to make the probe finally fail. ~30 min code +
   one training run. The most informative immediate experiment.

2. **An activation-transplant-style Phase 5.** The current pseudoinverse-
   direction patch is contaminated by the probe's clustering geometry.
   Replacing it with `a_B := residual at a real position where the model
   processes node B` would test the model's own representation rather than
   the probe's. Standard mech-interp practice. Half a day of code.

3. **LSTM baseline.** Same-param-count LSTM on real London; reports
   perplexity and probe results. Establishes whether the findings are
   transformer-specific or sequence-model-general. ~half day code + run.

4. **Proper-scale training run on Manhattan.** Tightens every number to
   publishable values. ~$2 GPU rental. Useful but no longer the most
   urgent thing — the methodological items above matter more.

5. **Geographic-region holdout.** Hold out a contiguous lat/lon sub-region
   rather than scattered destinations. Stronger generalization test, and
   independently interesting given that route SET-MEMBERSHIP carries
   geographic signal.

6. **Visualization** (`viz/overlay.py`). The visual documentation. Best
   done after items 1-2 so the overlay reflects the cleaned-up
   methodology.

The biggest near-term value is items 1-2, both M1-feasible.

---

## Bottom line

After the destroyed-structure finding, the calibrated state of the project is:

- **What is rock-solid**: the model trained on real route sequences learns the
  city graph (99.7% valid-edge rate; only 0.063 mass on real neighbors after
  the destroyed-structure ablation). This is unambiguous.
- **What is real but reframed**: the model's activations are linearly decodable
  to (lat, lon), but the source of this decodability is route SET-MEMBERSHIP
  co-occurrence — not sequence-order learning. The destroyed-structure-trained
  model has even higher probe R². So our "emergent metric map" is, more
  precisely, "an emergent clustering of node embeddings by geographic
  neighborhood, which arises whenever training data has geographic
  co-occurrence at any granularity."
- **What is NOT yet supported**: the causal claim that the model *uses* the
  location representation for next-hop prediction. The Phase 5 experiment as
  designed shows the same statistical signal in both real and destroyed-
  structure models, so it can't isolate the causal effect of the
  representation.

This is real progress, with honest accounting of what is and isn't yet shown.
The destroyed-structure control turned out to be more informative than the
original framing anticipated; it forced a precise reframe that makes the
project's contributions cleaner to state. Next steps are concrete and
bounded.
