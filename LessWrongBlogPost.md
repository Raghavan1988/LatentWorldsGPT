# Pre-registering "when does next-token prediction force a world model?" — and watching the prediction fail

**Epistemic status**: I ran a small, multi-domain study (seven toy domains, 0.27M–13M-parameter transformers). For two of the seven domains I committed quantitative ex-ante predictions to a public git repo *before training the models*, with commit hashes that predate every training run. Both pre-registered predictions failed. This post is mostly about what that failure taught me, and what I now believe instead — at modest confidence. I think there is roughly a 30–40% chance the residual claim ("architectural carry-through") survives the next adversarial test someone runs against it. I'd update fast on a clean counterexample.

## TL;DR

1. Li et al. (2022) and Nanda et al. (2023) showed Othello-GPT linearly encodes board state. The obvious follow-up is: *when does this generalize?* What predicts which features get encoded in a next-token transformer's residual stream?
2. I formulated a sharp hypothesis I called the **N-criterion**: a feature is linearly encoded in the residual stream iff predicting the next token requires it. This is the strongest version of the "predictive relevance" intuition I could state precisely.
3. I committed two quantitative ex-ante predictions derived from the N-criterion to a public repo. The lockdown commits are reproducible:
   - Maze navigation: `aa025b1` (2026-05-27, before maze data was generated).
   - HTTP log sequences: `3b25ed3` (2026-05-31, before NASA-HTTP was downloaded).
   - Audit either via `git log --diff-filter=A predictions/predictions_maze_navigation.md` (or `..._http_log_sequences.md`).
4. Both predictions failed. Features the N-criterion said should be **absent** from the residual stream were recovered above the untrained baseline (+0.15 for maze starting cell, +0.22 for HTTP cumulative request count after position-control).
5. The weaker mechanism that survived both falsifications is what I'm calling **architectural carry-through**: features at positionally distinct input slots persist in later residual states essentially as a side effect of self-attention copying earlier tokens forward. It is not a theory of *which* features get *learned*; it is an observation about which features get *carried*.

## Why I think this is worth your time

Most mechanistic-interpretability work on emergent representations has three structural problems that compound:

1. **Single domain, post-hoc.** One study probes one domain, reports what worked, and the reader can't distinguish a real pattern from a fishing expedition.
2. **Single seed.** Probe accuracies appear as point estimates, so the measurement noise floor is invisible. Multi-seed retrofits in this project moved several of my own headline numbers by 5–15 points and flipped one conclusion from "the model encodes it" to "this is at chance."
3. **No ex-ante predictions.** Published probe studies almost never commit in writing to what the outcome should be before the run. So we can't tell a framework that *predicts* the data from one that *fits* the data after the fact.

I think the cheapest fix for problem 3 is unreasonably underused. You can just commit a `predictions.md` file to a public repo before you train anything. If you're wrong, you have to say so, with a hash that predates your run. I'd guess the field improves a lot if 10% of probe papers started doing this. (Confidence ~60%.)

## Setup

Same nanoGPT-shaped architecture across domains; only vocabulary and corpus change. For every domain I trained three models on three corpora:

- **Real**: the actual structured data (routes through real cities, real Othello games, real Bach chorales, etc.).
- **Within-shuffled**: tokens within each sequence are permuted. Set membership preserved, structure destroyed.
- **Global-shuffled**: tokens shuffled across the entire corpus. Set membership destroyed too.

For each (domain × condition × seed) triple I ran (a) linear and MLP probes for domain features, (b) activation transplant ("if I splice the residual stream from a different prefix, does the model behave as if it's in that other state?"), and (c) per-layer ablation.

The "destroyed-structure" controls are load-bearing. They give every probe a built-in null. A probe that scores 85% on the real corpus and 80% on within-shuffled is mostly reading lexical statistics, not "world state."

The headline cross-condition pattern (real > within > global) for the four domains where it applies cleanly:

![Cross-condition gradient across domains](figures/04_cross_condition_gradient.png)

Per-layer encoding strength by domain (where the signal lives in the stack):

![Per-layer ablation across domains](figures/03_per_layer_ablation.png)

## All seven domains and how each fared against the N-criterion

The seven domains span four roles: independent positive controls (Othello, music voice-leading), partial / confounded positives (cities, flight, sym-group), a strict-N negative direction (music beat), and the two pre-registered tests (maze, HTTP). All numbers are 5-seed mean ± std at the honest split (node-, piece-, flight-, maze-, or session-level).

| # | Domain | Feature | N-criterion says | Result (gap vs untrained) | Verdict |
|---|---|---|---|---|---|
| 1 | **Othello** | Board state (per-cell, 3-class) | Encoded — required for legal-move prediction | MLP +0.344 at L4; linear +0.255; transplant +0.296 at L3 | Confirmed (matches Li/Nanda within 0.01) |
| 2a | **Music (Bach)** | Voice-leading (next-pitch) | Encoded — required for next-pitch prediction | Transplant +0.889 at L2; real 96% / within 64% / global 56% strict valid-step | Confirmed; cleanest cross-condition gradient |
| 2b | **Music (Bach)** | Chord identity | Weakly relevant | MLP +0.089 (~3σ) | Weakly confirmed |
| 2c | **Music (Bach)** | **Beat-in-measure** | **NULL** — voice-leading is locally predictable; beat not needed | MLP +0.006 (within 1σ of zero); beat-controlled transplant moves predictions *less* than random | Strict-N negative direction confirmed (probe + causal) |
| 3 | **Cities** (London / Manhattan / Boston) | 10×10 grid cell | Encoded — predicting next intersection requires geography | MLP +0.51 to +0.55 real; global-shuffled +0.01 | Surface result confirmed, but per-layer ablation shows ~+0.74 of the +0.94 transplant lift is already in the *embedding table* at L0 — mostly co-occurrence structure, not transformer-computed world state |
| 4 | **Flight phase (ADS-B)** | Phase label (climb/cruise/descent/etc.) | Encoded — phase governs next-token altitude/speed dynamics | MLP +0.105 real / +0.100 within / +0.053 global; transplant +0.47 real / +0.31 within / +0.00 global | Graded N working; clean monotonic gradient on transplant |
| 5 | **Symmetric-group walks (S₈ self-avoiding)** | Partial-product element | Encoded under task structure | MLP +0.053 real / +0.010 within / +0.014 global | Partial signal — trained beats untrained at p<0.01 but absolute magnitude small; useful as a methodology calibration, not a clean positive control |
| 6 | **Maze navigation** ⚓ pre-registered at `aa025b1` | **Starting cell at late steps** | **NULL** — next path step depends on current cell + goal, not where the path started | MLP **+0.152** at L5 (threshold gap < 0.10) | **Falsified** — strict N is wrong |
| 7a | **HTTP logs (NASA July+Aug 1995)** ⚓ pre-registered at `3b25ed3` | Feature A: size-bin of *first* request | Encoded via carry-through | MLP **+0.168** at L3 (predicted band ≥ 0.10) | Carry-through confirmed ex-ante on second domain |
| 7b | **HTTP logs** ⚓ pre-registered at `3b25ed3` | Feature B: cumulative large-response count (binned) | **NULL** — must be actively aggregated | MLP +0.291 unconditional; **+0.220** after position-control (threshold gap < 0.10) | **Falsified** even after the most conservative control |

The matrix-style verdicts (color-coded green/yellow/red) live in [`figures/02_results_matrix.md`](figures/02_results_matrix.md) for whoever wants the at-a-glance version.

## What the pre-registered failures actually showed

The two key falsifications:

**Maze (commit `aa025b1`, 2026-05-27).** P4 — the load-bearing prediction — said "starting cell ID is irrelevant to next-token prediction; gap should be ≤ 0.10." After lockdown I trained an 8×8 maze model on 100k mazes (1.5M training tokens, 6-layer / 2M-param transformer) and ran the probe at the honest *maze-level* split (test mazes whose walls the model had never seen). The trained MLP at L5 hit 0.20, untrained 0.05, **gap +0.152**. Above threshold. Strict N falsified.

**HTTP (commit `3b25ed3`, 2026-05-31).** Two predictions. Feature A (first-request size-bin) was the carry-through positive: predicted encoded, observed gap **+0.168**, confirmed. Feature B (cumulative count of large responses) was the computed-feature null: predicted gap ≤ 0.10, observed gap +0.29 unconditional. I worried this was a probe confound — position-in-session correlates with cumulative count, and the trained model has sharper positional reps than untrained — so I ran post-hoc position controls (within-position probing at fixed k, residual-after-position regression). Even after the strictest control, **gap +0.22**. Still above threshold. The graded form's null direction also fails.

## What I now believe instead, weakly

The residual signal in both falsifying cases looks like it traces back to a much more boring mechanism than "the model needs this feature, therefore it learns it." Specifically: features that *appear at a positionally distinct input slot* (the start cell at t=0; the first request in a session) persist in the residual stream at later positions because self-attention is *built* to copy earlier tokens forward. The feature is not learned because it's useful; it's carried because the architecture has nowhere else to put earlier-position information.

I'm calling this **architectural carry-through**. Two things to flag:

1. This was introduced *after* the maze falsification. It then made a clean prediction for HTTP (Feature A: first-request identity should be carried; +0.17 gap predicted, +0.17 gap measured). That's one out-of-sample confirmation, which counts for something but not a lot.
2. The HTTP run also surfaced an independent methodological issue: when a target feature is correlated with token position, probes inherit positional encoding's signal and look like they're recovering the feature when they're really recovering position. I had to add a within-position diagnostic and a residual-after-position regression to control for this. Feature B survives both controls, but the diagnostics matter — I'd guess this is a confound several published probe results have.

So my honest summary, post-falsification, is roughly:

- Strict "predictive relevance ⇒ encoding" is wrong, on at least two domains.
- A weaker mechanism (carry-through) is consistent with the data on those domains and made one correct out-of-sample call.
- I don't have a strong predictive theory of which features get *learned* into the residual stream beyond carry-through. The pre-registered tests have demonstrably weakened any such claim from my own work.

## How the N-criterion evolved (the arc of the hypothesis)

The interesting part of this project isn't any single domain. It's that the hypothesis changed shape twice under contact with pre-registered evidence, and the audit trail makes the changes hard to launder.

**Version 0 — strict biconditional.** *F is encoded in the residual stream iff F is predictive of the next token under the training distribution.* I picked this because it has the rare property of generating crisp predictions in *both* directions: irrelevant features should be *absent*, not just present-but-noisy. The music beat null (post-hoc) gave it preliminary support: beat-in-measure isn't required for next-pitch prediction, and the probe + transplant *both* came in null. That was encouraging. But it was post-hoc.

**Falsification 1 — maze (commit `aa025b1`).** P4 said starting cell should be absent at late timesteps. It wasn't. The audit-trail mechanism made retroactive reinterpretation impossible. I had to either drop the strict form or weaken it.

**Version 1 — graded form + architectural carry-through.** The minimal revision that absorbs the maze result without becoming vacuous: predictive relevance is still the dominant driver of *learned* encoding, but a *second* mechanism — self-attention copying input-slot tokens forward — encodes positionally distinct features regardless of relevance. This made carry-through a positive prediction on its own. The null direction (computed irrelevant features still absent) was retained.

**Falsification 2 — HTTP Feature B (commit `3b25ed3`).** Two ex-ante predictions on one domain to test exactly this split. Feature A (input-slot, carry-through) confirmed at +0.17. Feature B (cumulative count, must be computed) predicted null, observed +0.22 even after the strictest position-control. The graded form's null direction also fails.

**Version 2 — the deflated working claim.** Carry-through survives 2-for-2 on ex-ante tests. The broader "predictive relevance drives encoding" claim is 0-for-3 on risky pre-registered predictions (maze distance predicted encoded — wasn't; HTTP Feature B predicted null — wasn't; maze starting cell predicted null — wasn't). What's left:

- Carry-through is an architectural claim, not a theory of *learning*. It says "features at positionally distinct input slots will be in the late-layer residual stream by default." It does not say *which* learned features the model will construct.
- Anything stronger than carry-through — any version of "the next-token objective shapes which abstract features the model represents" — does not have an audit-trail confirmation in this work. Music voice-leading (transplant +0.89) is consistent with it but was a post-hoc finding. Othello board state is consistent with it but is an independent reproduction of prior work.
- The methodological residue is at least as load-bearing as any positive claim: multi-seed reporting, destroyed-structure controls, per-layer convergence between probe and transplant, pre-registration with audit hashes, and position-controlled probing for any target that correlates with token position. The HTTP Feature B run is the case study for that last one.

The honest summary of the evolution: I started with a sharp falsifiable claim, ran two pre-registered tests, watched the strict and then the graded form fail, and ended with a much weaker architectural observation plus a methodology I'd recommend more confidently than the substantive theory. I'd rather have a sharp wrong hypothesis publicly than a vague right one privately.

## What I'd most like LessWrong readers to push back on

1. Is "architectural carry-through" basically a restatement of the residual stream being a copy bus, in which case my one out-of-sample confirmation is uninteresting? (I'm at ~40% that this is the right critique.)
2. Are the destroyed-structure controls (within- vs global-shuffle) doing what I think they're doing? In music specifically the gradient comes out very clean (voice-leading: 96% / 64% / 56% strict valid-step rate across real / within / global). I'd like someone to try to break it.
3. The git-audited predictions setup is cheap. Are there obvious failure modes (e.g., implicit researcher degrees of freedom in *what* to commit to) that I'm not controlling for?

All seven domains, all model checkpoints, the two locked predictions files (verifiable at commits `aa025b1` and `3b25ed3` respectively), and the probe/transplant code are in a single small repo. If anyone wants to take a swing at the carry-through claim adversarially, I'd genuinely like that.

---

I want to flag, in line with site norms about new users on AI topics: I don't think any of this extends to frontier-scale models on natural-language data, and I'm not claiming it does. The contribution I'd actually defend is *the methodology* (multi-seed, destroyed-structure controls, ex-ante git-audited predictions, position-correlation diagnostics), not any specific result. The N-criterion is interesting mostly because it failed cleanly, and the version-by-version trail of how it failed is what I think is most useful to share.
