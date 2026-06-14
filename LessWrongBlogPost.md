# Pre-registering "when does next-token prediction force a world model?" — and watching the prediction fail

**Epistemic status**: I ran a small, multi-domain study (seven toy domains, 0.27M–13M-parameter transformers). I committed quantitative ex-ante predictions to a public git repo before training the models for two of the seven domains. Both pre-registered predictions failed. This post is mostly about what that failure taught me, and what I now believe instead — at modest confidence. I think there is roughly a 30–40% chance the residual claim ("architectural carry-through") survives the next adversarial test someone runs against it. I'd update fast on a clean counterexample.

## TL;DR

1. Li et al. (2022) and Nanda et al. (2023) showed Othello-GPT linearly encodes board state. The obvious follow-up is: *when does this generalize?* What predicts which features get encoded in a next-token transformer's residual stream?
2. I formulated a sharp hypothesis I called the **N-criterion**: a feature is linearly encoded in the residual stream iff predicting the next token requires it. This is the strongest version of the "predictive relevance" intuition I could state precisely.
3. I committed two quantitative ex-ante predictions derived from the N-criterion to a public repo (maze navigation, then HTTP log sequences). The commit hashes predate any model training and are verifiable via `git log --diff-filter=A` on the predictions files.
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

## The N-criterion and how it failed

**Strict N-criterion**: feature F is encoded in the residual stream iff F is predictive of the next token under the training distribution.

This was the strongest predictive statement I could derive. It also gives crisp predictions in both directions — features that *can't* matter for the next token shouldn't show up.

I picked two domains where the strict version gave a confident "should be absent" answer and committed to those predictions before training:

| Domain | Feature predicted absent | Ex-ante locked threshold | Result |
|---|---|---|---|
| Maze | Starting-cell identity at late timesteps | trained–untrained gap < 0.05 | **+0.15** |
| HTTP logs | Cumulative request count (Feature B) | gap < 0.05 after position-control | **+0.22** |

Both are above the locked threshold. The strict N-criterion is wrong, at least for these toy transformers.

## What I now believe instead, weakly

The residual signal in both falsifying cases looks like it traces back to a much more boring mechanism than "the model needs this feature, therefore it learns it." Specifically: features that *appear at a positionally distinct input slot* (the start cell at t=0; the first request in a session) persist in the residual stream at later positions because self-attention is *built* to copy earlier tokens forward. The feature is not learned because it's useful; it's carried because the architecture has nowhere else to put earlier-position information.

I'm calling this **architectural carry-through**. Two things to flag:

1. This was introduced *after* the maze falsification. It then made a clean prediction for HTTP (Feature A: first request identity should be carried; +0.17 gap predicted, +0.17 gap measured). That's one out-of-sample confirmation, which counts for something but not a lot.
2. The HTTP run also surfaced an independent methodological issue: when a target feature is correlated with token position, probes inherit positional encoding's signal and look like they're recovering the feature when they're really recovering position. I had to add a within-position diagnostic and a residual-after-position regression to control for this. Feature B survives both controls, but the diagnostics matter — I'd guess this is a confound several published probe results have.

So my honest summary, post-falsification, is roughly:

- Strict "predictive relevance ⇒ encoding" is wrong, on at least two domains.
- A weaker mechanism (carry-through) is consistent with the data on those domains and made one correct out-of-sample call.
- I don't have a strong predictive theory of which features get *learned* into the residual stream beyond carry-through. The pre-registered tests have demonstrably weakened any such claim from my own work.

## What I'd most like LessWrong readers to push back on

1. Is "architectural carry-through" basically a restatement of the residual stream being a copy bus, in which case my one out-of-sample confirmation is uninteresting? (I'm at ~40% that this is the right critique.)
2. Are the destroyed-structure controls (within- vs global-shuffle) doing what I think they're doing? In music specifically the gradient comes out very clean (voice-leading: 96% / 64% / 56% strict valid-step rate across real / within / global). I'd like someone to try to break it.
3. The git-audited predictions setup is cheap. Are there obvious failure modes (e.g., implicit researcher degrees of freedom in *what* to commit to) that I'm not controlling for?

All seven domains, all model checkpoints, the predictions files, and the probe/transplant code are in a single small repo. If anyone wants to take a swing at the carry-through claim adversarially, I'd genuinely like that.

---

I want to flag, in line with site norms about new users on AI topics: I don't think any of this extends to frontier-scale models on natural-language data, and I'm not claiming it does. The contribution I'd actually defend is *the methodology* (multi-seed, destroyed-structure controls, ex-ante git-audited predictions, position-correlation diagnostics), not any specific result. The N-criterion is interesting mostly because it failed cleanly. I'd rather have a sharp wrong hypothesis publicly than a vague right one privately.
