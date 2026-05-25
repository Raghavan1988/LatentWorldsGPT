# LatentCityGPT vs Othello-GPT — current status and what proper training will show

This is a stock-take comparing the experiment's progress to the Othello-GPT lineage
it is modeled on, and a calibrated prediction of what changes when we move from
the current smoke-trained models to proper (medium-config, full-corpus) training.

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

| # | Othello-GPT claim | LatentCityGPT counterpart | Status |
|---:|---|---|---|
| 1 | Trained transformer learns the rules of next-token prediction (legal moves) | Trained transformer learns the city graph (real neighbors) | ✅ **Verified** for all 3 trained cities. London valid-edge rate 99.7% on val, 99.3% on gen. Untrained baseline ≈ 0.3%. |
| 2 | Beats simple baselines on the next-token task | Beats Markov-2 on perplexity (1.23 val vs 1.46) and beats Markov-1 (modestly) on long-range coherence | ✅ Verified for London. Margins are real but smaller than one might hope on coherence — discussed in `PROGRESS_2026-05-24_evening.md`. |
| 3 | World state recoverable from residual stream activations | (lat, lon) recoverable from residual stream activations | ✅ **Verified** for all 3 cities. Linear probe trained R² 0.87 / 0.41 / 0.34 (position) and 0.64 / 0.26 / 0.23 (node-level). Substantial gap to untrained baseline in every condition. |
| 4 | World state survives held-out conditions | (lat, lon) recoverable for held-out **tokens** the probe never saw during training | ✅ Verified at the node-level split. Important because it rules out per-token lookup memorization for the linear probe. |
| 5 | World state is encoded *linearly* (Nanda's strong claim) | Linear probe matches MLP probe → encoding is linear | ❌ **Not cleanly testable in our setting**. MLP probe is contaminated by lookup memorization (proved by the untrained-MLP node-level collapse). The standard linear-vs-MLP criterion does not apply when targets are continuous and tokens are few. Needs a different test. |
| 6 | World state is computed by the transformer (deeper layers better) | Map signal is built up in the transformer's middle layers | 🟡 **Mixed**. On the node-level (probe-capacity-controlled) split, peak signal is in L3–L5 across all three cities — the transformer matters. But on position-level, London's peak is at the input embedding (where lookup helps), so the "transformer builds it" story depends on the split. |
| 7 | Activation patching the world state changes the model's outputs (causal claim) | Patching the location direction changes the next-hop predictions | ❌ **Not tested**. Phase 5 of the plan. No `eval/causal.py` written yet. This is the load-bearing missing experiment. |

---

## What we can say with high confidence

### Positive

- **The model genuinely learns the graph.** 99.7% next-step valid-edge rate, generalizes to routes
  whose destinations were never training endpoints (99.3% on gen). This is unambiguous and the
  effect size dwarfs any baseline noise.
- **Geographic structure emerged from training.** Across all three cities, the linear probe's
  trained-vs-untrained gap is positive on the node-level (lookup-controlled) split, with
  meaningful effect sizes (London +0.69, Manhattan +0.18, Boston +0.18 R²). This is the rung-3
  claim from the README ("a map is decodable") supported under controls.
- **The transformer is doing geometric work**, not just hosting an organized embedding table.
  On the held-out-token test, the linear probe peaks in mid-network layers in every city —
  the input embedding alone does not give the best probe performance under that protocol.

### Negative (or weak)

- **We cannot yet claim the representation is *linearly encoded* (Nanda's strong claim).** The
  MLP-vs-linear comparison breaks because the MLP has enough capacity to memorize per-token
  lookups in our setting. A different test methodology is needed.
- **We cannot yet claim the model *uses* the recovered representation.** The probe finds a
  decodable signal; that is correlation. Activation patching (Phase 5) is the only experiment
  that turns this into causation.
- **The destroyed-structure control is missing.** Until we train a parallel model on shuffled
  routes and show its probe fails, we cannot fully rule out the alternative that the geometry
  is an artifact of token-co-occurrence statistics rather than the real graph's structure.
- **Probe quality degrades with vocab size.** Boston's node-level R² (0.23) is far below
  London's (0.64). This is likely because smoke-trained Boston only has ~260 visits per node
  vs London's ~1600. Almost certainly an under-training artifact rather than a fundamental
  limitation, but the proper-training results will determine that.

### Difference vs Othello-GPT worth noting

Othello-GPT's headline interpretability story is *"the model computes the world state
inside the transformer; we can decode it linearly from the right intermediate layer."* Our
current story is *"the model contains a geographic representation that generalizes to
unseen tokens, located primarily in mid-to-late transformer layers."* That is structurally
similar but quantitatively weaker (R² 0.18–0.69 on the controlled test vs Othello-GPT's
near-perfect board recovery) because (a) our model is smoke-trained and (b) our target is
continuous in 2D rather than binary across discrete squares.

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

In priority order (highest scientific value first):

1. **Phase 5 — Activation patching** (`eval/causal.py`). The single most important missing
   piece. Without it, the project's strongest claim is "we found a probe that works" —
   not "the model uses an internal map." Estimated effort: ~half day to implement, quick to run.

2. **Destroyed-structure control**. Add `--shuffle_nodes` to `prepare_city.py`, train one
   parallel city, re-run the probe. Required to rule out the artifact-explanation. Effort:
   ~half day code + one training run.

3. **LSTM baseline**. Trains a same-parameter-count LSTM, reports perplexity and the
   probe result on its activations. Tests "does the architecture matter" — if LSTM gets
   a similar probe R², the result is about sequence models in general, not transformers
   specifically. Effort: ~half day code + one training run.

4. **Proper-scale training run on Manhattan**. Tightens every number in the result table
   to publishable values. Single A100 hour on a rental platform; ~$1–3 GPU cost.

5. **Geographic-region holdout** (`--split geographic` in `prepare_city.py`). Strong-form
   generalization: holds out a contiguous sub-region of the map. The probe is asked to
   predict locations for nodes from a part of the city the model never trained on routes
   ending in. Effort: small code change + one corpus regeneration + one training run.

6. **Visualization** (`viz/overlay.py`). Procrustes-align the probe-recovered coordinates
   to the true ones; render the recovered-vs-true overlay. The visual evidence that a
   street network reassembles itself from the model's internals.

After 1–5, the project has the structure of a complete interpretability result. After 6,
it has the visual documentation. The smoke runs we've done are sufficient to validate
the pipeline; the items above are what would turn this into a defensible claim.

---

## Bottom line

We are roughly **at Rung 2** of the README's three-rung hypothesis ladder (it learns the
graph; a map is decodable; the map emerged from training). Specifically: rungs 1 and 3
are essentially verified for the linear probe across three cities. Rung 2 (the existence
of the map) is also verified, but with diminishing absolute R² as cities grow — a smoke
artifact that proper training should largely fix.

The bonus rungs (linearly encoded; causally steers predictions) are not yet defensible
from the data on hand. Closing them requires the missing controls and Phase 5.

This is real progress, with honest accounting of what is and isn't yet shown. The
methodology is in place; the remaining work is concrete and bounded.
