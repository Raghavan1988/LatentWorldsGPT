# Plan: Weeks 3–6 (writeup-track refinements)

After the Phase 1+Phase 2 rigor pass (multi-seed probes, multi-seed transplants,
per-layer transplant ablation, linear-vs-MLP, symgroup methodology
resolution), the remaining work is structured around six themes:

1. **Sharpen what we already have** (per-domain ablations, scale on cities)
2. **One ex-ante predictive demonstration** on a not-yet-tested task
3. **Pre-registration of predictions** at the start of Phase 4
4. **Complementary causal-interp analyses** (DLA, logit lens, zero-ablation per-layer) — fills in the conventional mech-interp toolkit alongside transplant
5. **Methodology-first writeup framing**
6. **Draft + revise**

These map to weeks 3–6 below. Lower-leverage items (LSTM baseline,
adversarial probes, path patching, composition scores, causal
scrubbing) are explicitly de-scoped; they don't move the quality bar
enough to justify the work, given the macro-level encoding-existence
claim we're making.

## Phase 3 — Scale + pre-register

**Goal**: lock in the existing claims at higher confidence and set up
ex-ante predictions for Phase 4.

| Task | What | ETA | Marginal value |
|---|---|---|---|
| Phase 3-a | Pre-registration protocol setup. `predictions/` directory with README explaining the lockdown rules + confirm/falsify criteria, a `predictions_TEMPLATE.md` that future predictions copy from, and a worked `predictions_EXAMPLE.md`. ✓ DONE 2026-05-27. The ex-ante predictions file lands in Phase 4-b under a descriptive filename keyed to the experiment (`predictions_<task>.md`, e.g. `predictions_maze_navigation.md`). | 2 h | +2-3% |
| Phase 3-b | Othello championship-games retrain (Task 56). Replace 50k random uniform games with championship games; push trained MLP from 0.94 toward published 0.95+. | ~1 day training (GPU rental) | small (tightens a specific comparison) |
| Phase 3-c | Cities scale demonstration. Train one larger model (~50M params) on full Manhattan; show probe + transplant patterns persist at scale. | ~2-3 days GPU rental, $20-50 | +3-4% |
| Phase 3-d | **Direct Logit Attribution per-layer per-domain**. For each trained model, decompose the residual into per-component contributions to the logit for the correct next-move / next-pitch / next-phase token, summed at each layer. Output: `eval/dla.py` + `figs/phase3_dla_per_domain.png` showing each layer's direct contribution to the valid-next-token logit. Pairs naturally with transplant (DLA = component-wise direct contribution, transplant = causal effect of replacement). | ½ day | +2% (adds a conventional complementary view) |
| Phase 3-e | **Logit lens curve per-domain**. Apply the trained unembed to each layer's residual stream (skipping subsequent blocks); plot per-layer next-token accuracy. Output: `eval/logit_lens.py` + `figs/phase3_logit_lens_per_domain.png`. Single figure makes it visible at a glance where the model commits to its answer per layer. | ½ day | +2% |
| Phase 3-f | **Zero-ablation per-layer per-domain**. Standard complement to activation patching: for each layer, zero the residual output of that layer's blocks and measure the drop in valid-next-token rate. Output: `eval/zero_ablation.py` + `figs/phase3_zero_ablation_per_domain.png`. Pairs with transplant as the destruction-vs-restoration mirror image. | ½ day | +2% |

## Phase 4 — Ex-ante prediction experiment

**Goal**: pick a 7th domain we haven't tried; write down N-criterion
predictions BEFORE running anything; run; record confirms/falsifies.
This is the highest-leverage single item in the Phase 3-Phase 6 plan.

| Task | What | ETA | Marginal value |
|---|---|---|---|
| Phase 4-a | Pick the target task. Candidates (decide later): TCP-state corpus, simple code-execution traces, maze-path corpus, MultiWOZ dialog state. Pick by: (1) probe target is unambiguous, (2) destroyed-structure controls are buildable, (3) training fits 1 day on MPS. | ½ day choosing | — |
| Phase 4-b | Write predictions to `predictions_<task>.md` (descriptive filename, e.g. `predictions_maze_navigation.md` — not `predictions_w4.md`). For each chosen feature: predicted probe accuracy band, predicted gap-vs-untrained band, predicted per-layer peak location, predicted transplant lift band, expected destroyed-structure-control shape. Commit BEFORE any experiment runs. | ½ day | (zero alone; required for Phase 4-c to count) |
| Phase 4-c | Build data pipeline + train model (3 conditions: real, within-shuf, global-shuf). | 1-2 days | — |
| Phase 4-d | Run probe + transplant + per-layer ablation. Multi-seed protocol. | 1 day compute | — |
| Phase 4-e | Confirm/falsify table. For each prediction in `predictions_<task>.md`, write the matching `results_<task>.md` recording: observed value, predicted band, verdict (✓ confirmed / ◐ ambiguous / ✗ falsified). If falsified, document what the failure implies for the N-criterion framework. | 1 day analysis + writeup | **+7-10%** |

**Risk**: predictions may be wrong on something. That's not a failure
mode — a falsified prediction is a more useful paper than a confirmed
one. The only failure mode here is choosing a task where the
prediction itself is ambiguous or untestable.

## Phase 5 — Methodology-first writeup + visualization

**Goal**: rewrite the contribution as the N-criterion framework, not
the per-domain results.

| Task | What | ETA |
|---|---|---|
| Phase 5-a | Procrustes-aligned cities map overlay (`viz/overlay.py`). Real vs decoded vs destroyed-structure side-by-side. The visual hook. | 1 day |
| Phase 5-b | Per-domain transplant lift figure consolidating  Phase 2 +  Phase 4 results. Already started in `figs/phase2_transplant_lift.png`; expand with  Phase 4 data. | ½ day |
| Phase 5-c | "Complementary causal-interp analyses" section in the writeup. Show DLA, logit lens, and zero-ablation per-layer figures (from Phase 3-d/e/f) alongside the transplant per-layer figure as a 4-panel consolidated view. Explain explicitly: "Our claim is at the encoding-existence level, not the circuit-decomposition level. We use probing for descriptive evidence and activation patching for causal evidence; finer-grained tools (path patching, causal scrubbing, composition scores) target a different question." | 1 day |
| Phase 5-d | First-draft writeup with methodology-first framing. Lead the abstract with the N-criterion as the contribution. Domains are validation. Ex-ante prediction is the closing argument. | 3-4 days |
| Phase 5-e | Internal review: get 1-2 outside readers on the draft, incorporate edits. | 1-2 days |

## Phase 6 — Final draft + final revisions

**Goal**: polished writeup, all figures locked, reproducibility tested
end-to-end.

| Task | What | ETA |
|---|---|---|
| Phase 6-a | End-to-end reproducibility test. Fresh checkout, run every script, regenerate every figure. Document any missing dependencies. | 1 day |
| Phase 6-b | Limitations section. Explicit, complete enumeration of what we did and did NOT show. What would change the conclusions. Where the framework predicts encoding but doesn't show. | 1 day |
| Phase 6-c | Final pass on writing. Make sure the abstract, intro, contribution-table, conclusion all tell the same story. | 2 days |
| Phase 6-d | Final figures, supplementary materials, README updates. | 1 day |

## What is explicitly de-scoped

- **LSTM baseline**. Destroyed-structure controls + 5-domain evidence
  already do the architectural-specificity work. Adding LSTM is
  defensive, not load-bearing.
- **Path patching, composition scores, eigenvalue scores**. These
  target circuit-structure questions (how a known feature is
  computed), not encoding-existence questions (whether the feature
  is encoded at all). Different research program.
- **SAEs / causal scrubbing / fancier interp methods**. The simplicity
  of classical methods is a feature here — it lets the comparison span
  domains cleanly.
- **Max-activating dataset examples, feature visualization**. Single-
  feature interpretation tools; not load-bearing for the cross-domain
  encoding claim.
- **Mainline-LLM probe** (Llama-class). Out of compute budget; the
  scale-demo in Phase 3-c on cities does the necessary scaling work.
- **Maze, TCP, code-exec all three** — pick ONE for  Phase 4 ex-ante.

## Critical path

Phase 3-a (pre-registration setup) → Phase 4-b (write predictions) → Phase 4-c-d-e
(run experiments and record outcomes) → Phase 5-d (writeup framing) →
Phase 6-c (final pass).

Phase 3-b (Othello retrain) and Phase 3-c (cities scale) are parallel-able with
the  Phase 4 work if compute budget allows.

Phase 3-d/e/f (DLA, logit lens, zero-ablation) operate on the existing
Phase 1+Phase 2 checkpoints. No new training needed; ~1.5 days total. They feed
directly into Phase 5-c (complementary causal-interp analyses section).

## Acceptance criteria for "ready to share"

- All Phase 3-Phase 4 experiments complete with documented predictions and
  outcomes
- Methodology framing crisp in the abstract and intro
- Limitations section complete
- End-to-end reproducibility passes
- Pre-registration file untouched since lockdown (verifiable in git
  history)
