# Two failed pre-registered predictions about when transformers form world models


**Epistemic status**: Small multi-domain study (7 toy domains, 0.27M to 13M-param transformers). For two of the seven domains, I committed quantitative predictions to a public repo before training the models. The commits are auditable via git. Both pre-registered predictions failed. This post is about what those failures taught me. After the original HTTP result I also locked three narrower HTTP carry-through follow-ups before running them; those now give the cleanest small story. I put roughly 30 to 40% on the residual claim ("architectural carry-through") surviving the next adversarial test someone runs against it. I'd point readers more at the audit trail (how the hypothesis changed shape against pre-registered evidence) than at the toy-scale numbers themselves. Most mechanistic-interpretability work on emergent representations is single-domain, single-seed, and post-hoc, so it's hard to separate a framework that predicts the data from one that fits it after the fact. Committing a `predictions.md` file to a public repo before training takes very little overhead, and I think this discipline is unreasonably underused. If 10% of probe papers started doing this, the field gets noticeably better.**AI assistance disclosure**: The implementation was very heavily AI-assisted and Claude-enabled. Data pipelines for each domain, probe and transplant scripts, training configs, figure code, and a lot of iterative debugging were done with Claude.


## TL;DR

1. Othello-GPT (Li 2022, Nanda 2023) showed a transformer can linearly encode board state. The natural follow-up: when does this generalize?
2. I wrote down a sharp hypothesis I called the **N-criterion** (N for next-token necessity): a feature is in the residual stream iff predicting the next token requires it. The residual stream is the running hidden-state vector at each token position; every attention and MLP block reads from it and writes back to it, and by the final layer it is what the model uses to pick the next token. When I say "encoded in the residual stream" I mean a linear or MLP probe can recover the feature from this vector. I locked two derived predictions into git before running the experiments:
   - Maze navigation, commit `aa025b1`, 2026-05-27
   - HTTP log sequences, commit `3b25ed3`, 2026-05-31

   Audit either with `git log --diff-filter=A predictions/predictions_maze_navigation.md` (use the HTTP filename for the other). Both commits predate the corresponding data being generated or downloaded.
3. **Both predictions failed.** Features the N-criterion said should be absent came in at +0.15 (maze starting cell) and +0.22 (HTTP cumulative count, after a position-control diagnostic). The residual after both falsifications is a weaker architectural observation I'm calling **carry-through**. The July 4 HTTP follow-ups make that weaker claim less trivial: carry-through appears for same-request path, previous-request path, and a content-selected earlier large-response path.

## Setup, in one paragraph

Same nanoGPT-shaped architecture across all seven domains; only the vocabulary and corpus change. For every domain I trained three models on three corpora. The **real** corpus is the actual structured data. The **within-shuffled** corpus permutes tokens inside each sequence, so set membership is preserved but structure is destroyed. The **global-shuffled** corpus shuffles tokens across the entire corpus, destroying set membership too. For each combination of domain, condition, and seed I ran three things on the trained checkpoint: linear and MLP probes; an activation transplant, where I splice another prefix's residual stream into the position under test and check whether the model now behaves as if it were in that state; and a layer-by-layer ablation. The shuffled controls give every probe a built-in null. A probe that scores 85% on the real corpus and 80% on within-shuffled is mostly reading lexical statistics, not world state.

![Cross-condition gradient across domains](../../figures/04_cross_condition_gradient.png)

## All seven domains vs the N-criterion

All numbers below are 5-seed means at the honest split for that domain (node level for cities, piece level for music, flight level for ADS-B, maze level for maze, session level for HTTP). Pre-registered rows are marked ⚓ with the lockdown commit.

| # | Domain | Feature | N-criterion predicted | Result (gap vs untrained) | Verdict |
|---|---|---|---|---|---|
| 1 | **Othello** | Board state (per-cell) | Encoded | MLP +0.344 at L4; transplant +0.296 at L3 | ✓ Confirmed (matches Li/Nanda within 0.01) |
| 2a | **Music** (Bach) | Voice-leading | Encoded | Transplant +0.889 at L2; real / within / global gradient 96% / 64% / 56% | ✓ Cleanest cross-condition gradient |
| 2b | **Music** | Chord identity | Weak | MLP +0.089 (~3σ) | ✓ Weakly confirmed |
| 2c | **Music** | Beat-in-measure | **NULL** | MLP +0.006; beat-matched transplant moves predictions less than a random control | ✓ Strict-N negative direction (probe and causal) |
| 3 | **Cities** (London, Manhattan, Boston) | Grid cell | Encoded | MLP +0.51 to +0.55 real; global-shuffled +0.01 | ⚠ Surface confirm, but about 0.74 of the +0.94 transplant lift is already in the embedding table at L0. Co-occurrence structure, not transformer-computed world state |
| 4 | **Flight phase** (ADS-B) | Climb, cruise, or descent | Encoded | Transplant +0.47 real / +0.31 within / +0.00 global | ✓ Monotonic gradient across conditions |
| 5 | **Symmetric-group walks** (S₈) | Partial product | Encoded | MLP +0.053 real / +0.014 global | ⚠ Statistically significant, small in absolute terms. Methodology calibration, not a robust positive |
| 6 | **Maze navigation** ⚓ `aa025b1` | **Starting cell at late steps** | **NULL** | **MLP +0.152** at L5 (threshold +0.10) | ✗ **Falsified** |
| 7a | **HTTP logs** (NASA July+Aug 1995) ⚓ `3b25ed3` | Feature A: first request's size-bin | Encoded via carry-through | **MLP +0.168** at L3 (predicted ≥ 0.10) | ✓ Carry-through confirmed ex-ante on a new domain |
| 7b | **HTTP logs** ⚓ `3b25ed3` | Feature B: cumulative large-response count | **NULL** (must be aggregated) | MLP +0.291 raw, **+0.220** after position-control | ✗ **Falsified** even at fixed position |
| 7c | **HTTP follow-up** ⚓ `88155b4` | Same-request path `p_k` at size token `sz_k` | Encoded | **MLP +0.809** | ✓ Local repeated-layout carry-through |
| 7d | **HTTP follow-up** ⚓ `0d115c2` | Previous-request path `p_{k-1}` at `sz_k` | Encoded | **MLP +0.674** | ✓ Cross-record carry-through |
| 7e | **HTTP follow-up** ⚓ `8d671ca` | Most recent earlier large-response path `p_j` at `sz_k` | Weakly encoded | **MLP +0.621** full lag, **+0.514** at lag ≥3 | ✓ Content-selected earlier-event recoverability |

A color-coded version of the matrix is in [`figures/02_results_matrix.md`](../../figures/02_results_matrix.md).

### Smoke fit: did the transformer actually learn the target?

The probe results above are only meaningful if the model fit the task in the first place. Each domain has a step-by-step validity (or perplexity) check, run before any probe was wired up. For the four domains where a "valid-next-step" rate is well-defined, the smoke check itself produces the same real-within-global gradient that the rest of the paper rests on.

| Domain | Smoke check | Real | Within-shuffled | Global-shuffled | Comparison |
|---|---|---|---|---|---|
| Othello | next-token is a legal Othello move | 82.2% | n/a | n/a | Li/Nanda ~95%; ours within range |
| Music (Bach) | next-pitch within ±7 semitones (voice-leading validity) | 96% | 64% | 56% | Strongest cross-condition gradient in the table |
| Cities (London) | next-token is an adjacent graph edge | 99.7% | 0.06% | 0.006% | Strict destroyed-structure separation |
| Flight (ADS-B) | next-token physics-plausible (alt/vr/spd deltas in band) | 94.0% | 39.8% | 16.5% | Monotonic gradient |
| Symmetric-group (S₈) | val perplexity | 5.90 | n/a | n/a | Uniform baseline 6.82; modest improvement |
| Maze (8×8) | val perplexity | ~1.55 | ~1.55 | ~1.55 | Train/val/gen tightly matched, no overfit (no step-by-step validity script yet) |
| HTTP logs | val perplexity | 1.54-1.58 | 1.54-1.58 | 1.54-1.58 | Perplexity tightly matched across conditions, no overfit |

The two pre-registered domains (maze and HTTP) are the weakest cells in this column. Neither has a `valid_next_step` script in the repo yet, only a perplexity-and-no-overfit check. For maze the probe verdict (P4 falsified at +0.152 above untrained baseline) is robust to this gap because the comparison is *trained vs untrained on the same task*, so probe headroom is the relevant signal regardless of absolute task-fit. For HTTP the position-control diagnostic plays the same role. But a step-by-step validity check is worth writing for both before any second pre-registered domain goes in.

### Encoding strength by layer

The layer-by-layer view tells the second half of the story. Where in the stack does each domain's representation actually live?

![Encoding strength by layer across domains](../../figures/03_per_layer_ablation.png)

Cities is mostly L0 (the embedding table). Music's voice-leading jumps from L0 to L1, meaning it's built by the first transformer block from context rather than read off the embedding. Othello builds gradually and peaks at L4. The shape of these curves is what made the cities case feel qualitatively different from Othello and music, and it's what eventually pushed me to take pre-registration more seriously.

## How the N-criterion evolved

The hypothesis changed shape twice under pre-registered evidence. The audit hashes make those revisions hard to relabel after the fact.

My first version was a strict biconditional: F is encoded in the residual stream iff F is predictive of the next token under training. I chose this form because it generates predictions in both directions, so irrelevant features should be absent rather than just present-but-noisy. The music beat null gave it preliminary support (beat-in-measure isn't needed for next-pitch prediction, and the probe and transplant both came in null), but that was post-hoc.

I picked maze navigation as the first ex-ante test because "required for next-token" is structurally definable there: predicting the next path step needs the current cell and the goal, not the starting cell. I locked the prediction in commit `aa025b1` on 2026-05-27 (starting-cell probe gap should be ≤ 0.10), then trained the maze model (100k 8×8 mazes, 1.5M tokens, 2M-param 6-layer transformer) and ran the probe at the maze-level honest split. The trained MLP scored 0.20 at L5 against 0.05 for untrained, for a gap of +0.152, above the locked threshold. With the audit trail in place there was no retroactive reinterpretation available, which left two options: drop the strict form or weaken it.

I went with weakening, in the smallest way I could see that absorbed the maze result without becoming vacuous. Predictive relevance is still the dominant driver of learned encoding, but a second mechanism is also at work: self-attention copies input-slot tokens forward, so positionally-distinct features end up in the late-layer residual stream whether they're useful or not. I named that carry-through and turned it into its own falsifiable claim: input-slot features should be carried, while mid-sequence computed features that aren't useful should still be absent.

The HTTP-logs test (commit `3b25ed3` on 2026-05-31) was designed to span exactly that split, with two pre-registered features. Feature A was the size-bin of the first request in a session, an input-slot feature, predicted encoded via carry-through. Feature B was the cumulative count of large responses so far in the session, which has to be aggregated across packets, predicted null.

- **Feature A: +0.168 gap, confirmed.** Carry-through made a successful out-of-sample call on a domain that didn't exist when the mechanism was proposed.
- **Feature B: +0.29 gap raw, +0.22 after the strictest position-control,** still above the locked +0.10 threshold. The graded form's null direction therefore also fails.

The position-control was post-hoc but load-bearing. Position-in-session correlates with the cumulative count, and the trained model develops sharper positional representations than the untrained one. So I had to separate position-as-proxy from genuine encoding. Two diagnostics, within-position probing at fixed k=5 and residual-after-position regression, both kept the gap above threshold. Worth carrying forward as a default for any probe whose target correlates with token position.

Then I ran three locked HTTP follow-ups because "first request size-bin" still felt too easy. They form a useful ladder:

| Test | Probe result | Intervention reading |
|---|---:|---|
| Same-request path `p_k` at `sz_k` | MLP gap +0.809 | Very useful for predicting `sz_k`; weak after `sz_k`. |
| Previous-request path `p_{k-1}` at `sz_k` | MLP gap +0.674 | Mildly useful for `sz_k`; tiny after `sz_k`. |
| Most recent earlier large-response path `p_j` at `sz_k` | MLP gap +0.621; lag ≥3 gap +0.514 | Strongly decodable, but weakly useful under lag-filtered corruption. |

The third one is the part I now find most interesting. The source event is not the first slot and not a fixed relative offset. It is "the most recent earlier request whose response was large." That does not prove semantic memory. But it does show the residual can preserve a strong recoverable trace of a content-selected earlier event even when corrupting that trace has only a weak immediate next-token effect.

Where that leaves the hypothesis: carry-through survives 2-for-2 on the original ex-ante tests and gets three narrower HTTP follow-up confirmations, while the broader "predictive relevance drives encoding" claim is 0-for-3. Maze distance was predicted encoded and wasn't; maze starting cell and HTTP Feature B were predicted null and both came in above threshold. The remaining claims:

- Carry-through is an architectural claim, not a theory of learning. It only covers features at positionally distinct input slots persisting in the late-layer residual stream by default; which learned features the model actually constructs is a different question.
- Anything stronger than that, including any version of "the next-token objective shapes which abstract features get represented", does not have an audit-trail confirmation in this work. Music voice-leading and Othello board state are consistent with such a claim, but the music finding was post-hoc and the Othello result is independent reproduction of prior work.
- The methodology residue is at least as load-bearing as the substantive claim. Multi-seed reporting, destroyed-structure controls, probe-transplant convergence at the same layer, pre-registration with audit hashes, and position-controlled probing whenever the target correlates with position.

## Where I want pushback

1. Is "architectural carry-through" basically a restatement of the residual stream being a copy bus? The recent-large-response result makes this less obviously first-slot copying, but maybe still not deep. I'm at about 40% that this is the right critique.
2. Are the destroyed-structure controls (within- vs global-shuffle) doing what I think they're doing? Music's voice-leading gradient (96% / 64% / 56%) is the cleanest case. I'd like someone to try to break it.
3. Pre-committing predictions to git is low-overhead in practice. What failure modes am I missing? Implicit researcher degrees of freedom in what to commit to is the obvious one.

All seven domains, the locked predictions files (verifiable at `aa025b1` and `3b25ed3`), the checkpoints, and the probe and transplant code are in a single small repo: https://github.com/Raghavan1988/LatentWorldsGPT. If you want to take a swing at the carry-through claim, I'd be happy about that.

---

To be clear about scope: none of this is claimed to transfer to frontier-scale models on natural language. The methodology I'd recommend to someone else with no real hedging is multi-seed reporting, destroyed-structure controls, ex-ante git-audited predictions, and position-correlation diagnostics. The substantive N-criterion claim is more useful as something that failed sharply and traceably than as a working theory.
