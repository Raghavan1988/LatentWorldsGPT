# Pre-registered: when does next-token prediction force a world model?

**Epistemic status**: Small multi-domain study (7 toy domains, 0.27M to 13M-param transformers). For two of the seven domains, I committed quantitative predictions to a public repo before training the models. The commits are auditable via git. Both pre-registered predictions failed. This post is about what those failures taught me. I put roughly 30 to 40% on the residual claim ("architectural carry-through") surviving the next adversarial test someone runs against it.

## TL;DR

1. Othello-GPT (Li 2022, Nanda 2023) showed a transformer can linearly encode board state. The natural follow-up: when does this generalize?
2. I wrote down a sharp hypothesis I called the **N-criterion**. A feature is in the residual stream iff predicting the next token requires it. I locked two derived predictions into git before running the experiments:
   - Maze navigation, commit `aa025b1`, 2026-05-27
   - HTTP log sequences, commit `3b25ed3`, 2026-05-31

   Audit either with `git log --diff-filter=A predictions/predictions_maze_navigation.md` (use the HTTP filename for the other). Both commits predate the corresponding data being generated or downloaded.
3. **Both predictions failed.** Features the N-criterion said should be absent came in at +0.15 (maze starting cell) and +0.22 (HTTP cumulative count, after a position-control diagnostic). What survived the falsifications is a weaker architectural observation I'm calling **carry-through**. It's explained below.

## Why I think this might interest LessWrong

Most mechanistic-interpretability work on emergent representations is single-domain, single-seed, and post-hoc. So we can't tell a framework that predicts the data from one that fits it after the fact. Committing a `predictions.md` file to a public repo before you train anything is a cheap fix and I think it's unreasonably underused. If 10% of probe papers started doing this, the field gets noticeably better. (Confidence ~60%.) The interesting part of this post isn't the toy-scale results. It's the audit trail showing the hypothesis change shape under contact with pre-registered evidence.

## Setup, in one paragraph

Same nanoGPT-shaped architecture across all seven domains. Only the vocabulary and corpus change. For every domain I trained three models on three corpora: **real** (the actual structured data), **within-shuffled** (tokens permuted inside each sequence, so set membership is preserved but structure is destroyed), and **global-shuffled** (tokens shuffled across the entire corpus, so set membership is destroyed too). For each (domain × condition × seed) I ran linear and MLP probes, activation transplant (splice in another prefix's residual stream and see whether the model behaves as if it's in that state), and per-layer ablation. The shuffled controls give every probe a built-in null. A probe that scores 85% on the real corpus and 80% on within-shuffled is mostly reading lexical statistics, not world state.

![Cross-condition gradient across domains](figures/04_cross_condition_gradient.png)

## All seven domains vs the N-criterion

All numbers are 5-seed mean at the honest split (node-, piece-, flight-, maze-, or session-level). Pre-registered rows are marked ⚓ with the lockdown commit.

| # | Domain | Feature | N-criterion predicted | Result (gap vs untrained) | Verdict |
|---|---|---|---|---|---|
| 1 | **Othello** | Board state (per-cell) | Encoded | MLP +0.344 at L4; transplant +0.296 at L3 | ✓ Confirmed (matches Li/Nanda within 0.01) |
| 2a | **Music** (Bach) | Voice-leading | Encoded | Transplant +0.889 at L2; real / within / global gradient 96% / 64% / 56% | ✓ Cleanest cross-condition gradient |
| 2b | **Music** | Chord identity | Weak | MLP +0.089 (~3σ) | ✓ Weakly confirmed |
| 2c | **Music** | Beat-in-measure | **NULL** | MLP +0.006; beat-matched transplant moves predictions less than a random control | ✓ Strict-N negative direction (probe and causal) |
| 3 | **Cities** (London, Manhattan, Boston) | Grid cell | Encoded | MLP +0.51 to +0.55 real; global-shuffled +0.01 | ⚠ Surface confirm, but about 0.74 of the +0.94 transplant lift is already in the embedding table at L0. Co-occurrence structure, not transformer-computed world state |
| 4 | **Flight phase** (ADS-B) | Climb / cruise / descent | Encoded | Transplant +0.47 real / +0.31 within / +0.00 global | ✓ Clean monotonic gradient |
| 5 | **Symmetric-group walks** (S₈) | Partial product | Encoded | MLP +0.053 real / +0.014 global | ⚠ Statistically significant, small in absolute terms. Methodology calibration, not a clean positive |
| 6 | **Maze navigation** ⚓ `aa025b1` | **Starting cell at late steps** | **NULL** | **MLP +0.152** at L5 (threshold +0.10) | ✗ **Falsified** |
| 7a | **HTTP logs** (NASA July+Aug 1995) ⚓ `3b25ed3` | Feature A: first request's size-bin | Encoded via carry-through | **MLP +0.168** at L3 (predicted ≥ 0.10) | ✓ Carry-through confirmed ex-ante on a new domain |
| 7b | **HTTP logs** ⚓ `3b25ed3` | Feature B: cumulative large-response count | **NULL** (must be aggregated) | MLP +0.291 raw, **+0.220** after position-control | ✗ **Falsified** even at fixed position |

A color-coded version of the matrix is in [`figures/02_results_matrix.md`](figures/02_results_matrix.md).

The per-layer view tells the second half of the story. Where in the stack does each domain's representation actually live?

![Per-layer encoding strength across domains](figures/03_per_layer_ablation.png)

Cities is mostly L0 (the embedding table). Music's voice-leading jumps from L0 to L1, meaning it's built by the first transformer block from context rather than read off the embedding. Othello builds gradually and peaks at L4. The shape of these curves is what made the cities case feel qualitatively different from Othello and music, and it's what eventually pushed me to take pre-registration more seriously.

## How the N-criterion evolved

The interesting structural thing about this project isn't any single result. It's that the hypothesis changed shape twice under contact with pre-registered evidence, and the audit trail makes the changes hard to launder retroactively.

**Version 0: strict biconditional.** F is encoded in the residual stream iff F is predictive of the next token under training. I chose this form because it generates predictions in both directions. Irrelevant features should be absent, not just present-but-noisy. The music beat null gave it preliminary support: beat-in-measure isn't needed for next-pitch prediction, and the probe and transplant both came in null. But that was post-hoc.

**Falsification 1: maze, commit `aa025b1` (2026-05-27).** I picked maze navigation as the first ex-ante test because "required for next-token" is structurally definable. Predicting the next path step needs the current cell and the goal, not the starting cell. So I locked the prediction: starting-cell probe gap should be ≤ 0.10. After lockdown I trained the maze model (100k 8×8 mazes, 1.5M tokens, 2M-param 6-layer transformer) and ran the probe at the maze-level honest split. Trained MLP at L5: 0.20. Untrained: 0.05. Gap: **+0.152**. Above threshold. The audit trail made retroactive reinterpretation impossible. Either drop the strict form or weaken it.

**Version 1: graded form plus architectural carry-through.** The minimal revision that absorbs the maze result without becoming vacuous. Predictive relevance is still the dominant driver of learned encoding, but a second mechanism is also at work: self-attention copies input-slot tokens forward, so positionally-distinct features end up in the late-layer residual stream whether they're useful or not. This became a falsifiable claim on its own. Input-slot features should be carried. Mid-sequence computed features that aren't useful should still be absent.

**Falsification 2: HTTP, commit `3b25ed3` (2026-05-31).** Two ex-ante predictions, designed to span exactly that split. Feature A was the size-bin of the first request in a session, which is an input-slot feature, predicted encoded via carry-through. Feature B was the cumulative count of large responses in the session so far, which has to be aggregated across packets, predicted null.

- **Feature A: +0.168 gap. Confirmed.** Carry-through made a successful out-of-sample call on a domain that didn't exist when the mechanism was proposed.
- **Feature B: +0.29 gap raw, +0.22 after the strictest position-control.** Still above the locked +0.10 threshold. The graded form's null direction also fails.

The position-control was post-hoc but load-bearing. Position-in-session correlates with the cumulative count, and the trained model develops sharper positional representations than the untrained one. So I had to separate position-as-proxy from genuine encoding. Two diagnostics, within-position probing at fixed k=5 and residual-after-position regression, both kept the gap above threshold. Worth carrying forward as a default for any probe whose target correlates with token position.

**Version 2: the deflated working claim.** Carry-through survives 2-for-2 on ex-ante tests. The broader "predictive relevance drives encoding" claim is 0-for-3 on risky pre-registered predictions. Maze distance: predicted encoded, wasn't. Maze starting cell: predicted null, wasn't. HTTP Feature B: predicted null, wasn't. What's left:

- Carry-through is an architectural claim, not a theory of learning. It says features at positionally distinct input slots persist in the late-layer residual stream by default. It does not say which learned features the model will construct.
- Anything stronger than that, including any version of "the next-token objective shapes which abstract features get represented", does not have an audit-trail confirmation in this work. Music voice-leading and Othello board state are consistent with such a claim, but the music finding was post-hoc and the Othello result is independent reproduction of prior work.
- The methodology residue is at least as load-bearing as the substantive claim. Multi-seed reporting, destroyed-structure controls, probe-transplant convergence at the same layer, pre-registration with audit hashes, and position-controlled probing whenever the target correlates with position.

That's the honest summary. I started with a sharp falsifiable claim, ran two pre-registered tests, watched the strict and then the graded form fail, and ended with a much weaker architectural observation plus a methodology I'd recommend more confidently than the theory.

## What I'd most like pushback on

1. Is "architectural carry-through" basically a restatement of the residual stream being a copy bus? In which case the one out-of-sample confirmation is interesting only as a check on the obvious. I'm at about 40% that this is the right critique.
2. Are the destroyed-structure controls (within- vs global-shuffle) doing what I think they're doing? Music's voice-leading gradient (96% / 64% / 56%) is the cleanest case. I'd like someone to try to break it.
3. The git-audited predictions setup is cheap. What failure modes am I missing? Implicit researcher degrees of freedom in what to commit to is the obvious one.

All seven domains, the locked predictions files (verifiable at `aa025b1` and `3b25ed3`), the checkpoints, and the probe and transplant code are in a single small repo. If you want to take a swing at the carry-through claim adversarially, I'd be happy about that.

---

Standard caveat. None of this is claimed to extend to frontier-scale models on natural language, and I'm not claiming it does. What I'd actually defend is the methodology: multi-seed, destroyed-structure controls, ex-ante git-audited predictions, position-correlation diagnostics. Not any specific result. The N-criterion is interesting mostly because it failed cleanly, and the version-by-version trail of how it failed is the part I think is most worth sharing.
