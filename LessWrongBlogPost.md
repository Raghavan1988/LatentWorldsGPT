# Pre-registering "when does next-token prediction force a world model?" — and watching the prediction fail

**Epistemic status**: Small multi-domain study (7 toy domains, 0.27M–13M-param transformers). Two of the seven domains had quantitative predictions committed to a public repo *before training the models* — verifiable via git. Both pre-registered predictions failed. This post is about what those failures taught me. Roughly 30–40% the residual claim ("architectural carry-through") survives the next adversarial test.

## TL;DR

1. Othello-GPT (Li 2022, Nanda 2023) showed a transformer can linearly encode board state. The natural follow-up: *when does this generalize?*
2. I wrote down a sharp hypothesis — the **N-criterion**: a feature is in the residual stream iff predicting the next token requires it. I locked two derived predictions into git **before** running the experiments:
   - Maze navigation: commit `aa025b1` (2026-05-27)
   - HTTP log sequences: commit `3b25ed3` (2026-05-31)

   Audit either via `git log --diff-filter=A predictions/predictions_maze_navigation.md` (replace filename for HTTP). Both commits predate the corresponding data being generated or downloaded.
3. **Both predictions failed.** Features the N-criterion said should be *absent* came in at +0.15 (maze starting cell) and +0.22 (HTTP cumulative count, post-control). What survived the falsifications is a weaker architectural observation — **carry-through** — which I explain below.

## Why I think this might interest LessWrong

Most mechanistic-interpretability work on emergent representations is single-domain, single-seed, post-hoc. So we can't tell a framework that *predicts* the data from one that *fits* it after the fact. Committing a `predictions.md` file to a public repo before you train anything is a cheap fix that I think is unreasonably underused. If 10% of probe papers did this, the field gets noticeably better. (Confidence ~60%.) The interesting thing about this post is not the toy-scale results — it's the audit trail showing the hypothesis evolving under contact with pre-registered evidence.

## Setup, in one paragraph

Same nanoGPT-shaped architecture across all seven domains; only the vocabulary and corpus change. For every domain I trained three models on three corpora: **real** (the actual structured data), **within-shuffled** (tokens permuted inside each sequence; set membership preserved, structure destroyed), and **global-shuffled** (tokens shuffled across the corpus; set membership destroyed too). For each (domain × condition × seed), I ran linear+MLP probes, activation transplant ("splice in another prefix's residual stream — does the model behave as if it's in that state?"), and per-layer ablation. The shuffled controls give every probe a built-in null: a probe that scores 85% on real and 80% on within-shuffled is mostly reading lexical statistics, not world state.

![Cross-condition gradient across domains](figures/04_cross_condition_gradient.png)

## All seven domains vs the N-criterion

All numbers are 5-seed mean at the honest split (node-, piece-, flight-, maze-, or session-level). Pre-registered rows are marked ⚓ with the lockdown commit.

| # | Domain | Feature | N-criterion predicted | Result (gap vs untrained) | Verdict |
|---|---|---|---|---|---|
| 1 | **Othello** | Board state (per-cell) | Encoded | MLP +0.344 at L4; transplant +0.296 at L3 | ✓ Confirmed (matches Li/Nanda within 0.01) |
| 2a | **Music** (Bach) | Voice-leading | Encoded | Transplant +0.889 at L2; clean real/within/global gradient (96% / 64% / 56%) | ✓ Cleanest cross-condition gradient |
| 2b | **Music** | Chord identity | Weak | MLP +0.089 (~3σ) | ✓ Weakly confirmed |
| 2c | **Music** | Beat-in-measure | **NULL** | MLP +0.006; beat-matched transplant moves predictions *less* than random | ✓ Strict-N negative direction (probe + causal) |
| 3 | **Cities** (London/Manhattan/Boston) | Grid cell | Encoded | MLP +0.51 to +0.55 real; global-shuffled +0.01 | ⚠ Surface confirm — but ~0.74 of the +0.94 transplant lift is already in the embedding table at L0. Co-occurrence structure, not transformer-computed world state |
| 4 | **Flight phase** (ADS-B) | Climb/cruise/descent | Encoded | Transplant +0.47 real / +0.31 within / +0.00 global | ✓ Clean monotonic gradient |
| 5 | **Symmetric-group walks** (S₈) | Partial product | Encoded | MLP +0.053 real / +0.014 global | ⚠ Statistically significant but small absolute — methodology calibration, not a clean positive |
| 6 | **Maze navigation** ⚓ `aa025b1` | **Starting cell at late steps** | **NULL** | **MLP +0.152** at L5 (threshold +0.10) | ✗ **Falsified** |
| 7a | **HTTP logs** (NASA July+Aug 1995) ⚓ `3b25ed3` | Feature A: first request's size-bin | Encoded via carry-through | **MLP +0.168** at L3 (predicted ≥ 0.10) | ✓ Carry-through confirmed ex-ante on a new domain |
| 7b | **HTTP logs** ⚓ `3b25ed3` | Feature B: cumulative large-response count | **NULL** (must be aggregated) | MLP +0.291 raw; **+0.220** after position-control | ✗ **Falsified** even at fixed position |

(A color-coded version is in [`figures/02_results_matrix.md`](figures/02_results_matrix.md).)

The per-layer view tells the second half of the story: where in the stack each domain's representation lives.

![Per-layer encoding strength across domains](figures/03_per_layer_ablation.png)

Cities is mostly L0 (embedding table). Music's voice-leading jumps from L0 → L1 (built by the first transformer block from context). Othello builds gradually and peaks at L4. The shape of these curves is what makes the cities case feel qualitatively different from Othello/music, and it's what eventually pushed me to take pre-registration more seriously.

## How the N-criterion evolved

The interesting structural thing about this project isn't any single result. It's that the hypothesis changed shape twice under contact with pre-registered evidence, and the audit trail makes the changes hard to launder.

**Version 0 — strict biconditional.** *F is encoded in the residual stream iff F is predictive of the next token under training.* I chose this form because it generates predictions in *both* directions — irrelevant features should be *absent*, not just present-but-noisy. The music beat null (post-hoc) gave it preliminary support. But it was post-hoc.

**Falsification 1 — Maze, commit `aa025b1` (2026-05-27).** I picked maze navigation as the first ex-ante test because "required for next-token" is structurally definable: predicting the next path step needs the current cell and the goal, not the starting cell. So I locked the prediction: starting-cell probe gap should be ≤ 0.10. After lockdown I trained the maze model (100k 8×8 mazes, 1.5M tokens, 2M-param transformer), ran the probe at the maze-level honest split. **Trained MLP at L5: 0.20. Untrained: 0.05. Gap: +0.152.** Above threshold. The audit trail made retroactive reinterpretation impossible — drop the strict form or weaken it.

**Version 1 — graded form + architectural carry-through.** The minimal revision that absorbs the maze result without becoming vacuous: predictive relevance is still the dominant driver of *learned* encoding, but a *second* mechanism — self-attention copying input-slot tokens forward — encodes positionally-distinct features regardless of whether they're useful. This became a falsifiable claim of its own: input-slot features should be carried, mid-sequence-computed features should still be absent.

**Falsification 2 — HTTP, commit `3b25ed3` (2026-05-31).** Two ex-ante predictions, designed to span exactly that split. Feature A (size-bin of the first request — input-slot) predicted encoded via carry-through. Feature B (cumulative count of large responses — must be aggregated) predicted null.

- **Feature A: +0.168 gap. Confirmed.** Carry-through made a successful out-of-sample call on a domain that didn't exist when the mechanism was proposed.
- **Feature B: +0.29 gap raw, +0.22 after the strictest position-control.** Still above the locked +0.10 threshold. The graded form's null direction *also* fails.

(The position-control was post-hoc but load-bearing: position-in-session correlates with cumulative count, and the trained model has sharper positional reps than untrained, so I had to separate position-as-proxy from true encoding. Two diagnostics — within-position probing at fixed k=5, residual-after-position regression — both kept the gap above threshold. Worth carrying forward as a default for any probe whose target correlates with token position.)

**Version 2 — the deflated working claim.** Carry-through survives 2-for-2 on ex-ante tests. The broader "predictive relevance drives encoding" claim is 0-for-3 on risky pre-registered predictions (maze distance: predicted encoded, wasn't; maze starting cell: predicted null, wasn't; HTTP Feature B: predicted null, wasn't). What's left to defend:

- Carry-through is an *architectural* claim, not a theory of learning. It says "features at positionally distinct input slots persist in the late-layer residual stream by default." It does *not* say which learned features the model will construct.
- Anything stronger — any version of "the next-token objective shapes which abstract features get represented" — does not have an audit-trail confirmation in this work. Music voice-leading and Othello board state are *consistent* with it, but the first was post-hoc and the second is independent reproduction of prior work.
- The methodology residue is at least as load-bearing as the substantive claim: multi-seed reporting, destroyed-structure controls, probe/transplant convergence, pre-registration with audit hashes, and position-controlled probing whenever the target correlates with position.

I started with a sharp falsifiable claim, ran two pre-registered tests, watched the strict and then the graded form fail, and ended with a much weaker architectural observation plus a methodology I'd recommend more confidently than the theory.

## What I'd most like pushback on

1. Is "architectural carry-through" basically a restatement of the residual stream being a copy bus? In which case the one out-of-sample confirmation is interesting only as a check on the obvious. (I'm at ~40% that this is the right critique.)
2. Are the destroyed-structure controls (within- vs global-shuffle) doing what I think they're doing? Music's voice-leading gradient (96% / 64% / 56%) is the cleanest case — I'd like someone to try to break it.
3. The git-audited predictions setup is cheap. What failure modes am I missing? Implicit researcher degrees of freedom in *what* to commit to is the obvious one.

All seven domains, the locked predictions files (verifiable at `aa025b1` and `3b25ed3`), the checkpoints, and the probe/transplant code are in a single small repo. If you want to take a swing at carry-through adversarially, I'd genuinely like that.

---

Standard caveat in line with site norms: none of this is claimed to extend to frontier-scale models on natural language, and I'm not claiming it does. What I'd actually defend is *the methodology* — multi-seed, destroyed-structure controls, ex-ante git-audited predictions, position-correlation diagnostics — not any specific result. The N-criterion is interesting mostly because it failed cleanly, and the version-by-version trail of how it failed is what I think is most useful to share.
