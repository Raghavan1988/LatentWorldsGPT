# Plan: Weeks 3–6 (writeup-track refinements)

After the W1+W2 rigor pass (multi-seed probes, multi-seed transplants,
per-layer transplant ablation, linear-vs-MLP, symgroup methodology
resolution), the remaining work is structured around five themes:

1. **Sharpen what we already have** (per-domain ablations, scale on cities)
2. **One ex-ante predictive demonstration** on a not-yet-tested task
3. **Pre-registration of predictions** at the start of W4
4. **Methodology-first writeup framing**
5. **Draft + revise**

These map to weeks 3–6 below. Lower-leverage items (LSTM baseline,
adversarial probes) are explicitly de-scoped; they don't move the
quality bar enough to justify the work.

## Week 3 — Scale + pre-register

**Goal**: lock in the existing claims at higher confidence and set up
ex-ante predictions for W4.

| Task | What | ETA | Marginal value |
|---|---|---|---|
| W3-a | Pre-registration document. Create `predictions.md` listing, per upcoming W4 experiment, the exact prediction (probe accuracy band, layer-of-peak, transplant lift band) and the criterion for confirm/falsify. Timestamped, git-committed. Read-only after lockdown. | 2 h | +2-3% |
| W3-b | Othello championship-games retrain (Task 56). Replace 50k random uniform games with championship games; push trained MLP from 0.94 toward published 0.95+. | ~1 day training (GPU rental) | small (tightens a specific comparison) |
| W3-c | Cities scale demonstration. Train one larger model (~50M params) on full Manhattan; show probe + transplant patterns persist at scale. | ~2-3 days GPU rental, $20-50 | +3-4% |

## Week 4 — Ex-ante prediction experiment

**Goal**: pick a 7th domain we haven't tried; write down N-criterion
predictions BEFORE running anything; run; record confirms/falsifies.
This is the highest-leverage single item in the W3-W6 plan.

| Task | What | ETA | Marginal value |
|---|---|---|---|
| W4-a | Pick the target task. Candidates (decide later): TCP-state corpus, simple code-execution traces, maze-path corpus, MultiWOZ dialog state. Pick by: (1) probe target is unambiguous, (2) destroyed-structure controls are buildable, (3) training fits 1 day on MPS. | ½ day choosing | — |
| W4-b | Write predictions to `predictions_w4.md`. For each chosen feature: predicted probe accuracy band, predicted gap-vs-untrained band, predicted per-layer peak location, predicted transplant lift band, expected destroyed-structure-control shape. Commit BEFORE any experiment runs. | ½ day | (zero alone; required for W4-c to count) |
| W4-c | Build data pipeline + train model (3 conditions: real, within-shuf, global-shuf). | 1-2 days | — |
| W4-d | Run probe + transplant + per-layer ablation. Multi-seed protocol. | 1 day compute | — |
| W4-e | Confirm/falsify table. For each prediction in `predictions_w4.md`: did the experiment confirm or falsify? If falsified, what does the failure imply for the N-criterion framework? | 1 day analysis + writeup | **+7-10%** |

**Risk**: predictions may be wrong on something. That's not a failure
mode — a falsified prediction is a more useful paper than a confirmed
one. The only failure mode here is choosing a task where the
prediction itself is ambiguous or untestable.

## Week 5 — Methodology-first writeup + visualization

**Goal**: rewrite the contribution as the N-criterion framework, not
the per-domain results.

| Task | What | ETA |
|---|---|---|
| W5-a | Procrustes-aligned cities map overlay (`viz/overlay.py`). Real vs decoded vs destroyed-structure side-by-side. The visual hook. | 1 day |
| W5-b | Per-domain transplant lift figure consolidating W2 + W4 results. Already started in `figs/week2_transplant_lift.png`; expand with W4 data. | ½ day |
| W5-c | First-draft writeup with methodology-first framing. Lead the abstract with the N-criterion as the contribution. Domains are validation. Ex-ante prediction is the closing argument. | 3-4 days |
| W5-d | Internal review: get 1-2 outside readers on the draft, incorporate edits. | 1-2 days |

## Week 6 — Final draft + final revisions

**Goal**: polished writeup, all figures locked, reproducibility tested
end-to-end.

| Task | What | ETA |
|---|---|---|
| W6-a | End-to-end reproducibility test. Fresh checkout, run every script, regenerate every figure. Document any missing dependencies. | 1 day |
| W6-b | Limitations section. Explicit, complete enumeration of what we did and did NOT show. What would change the conclusions. Where the framework predicts encoding but doesn't show. | 1 day |
| W6-c | Final pass on writing. Make sure the abstract, intro, contribution-table, conclusion all tell the same story. | 2 days |
| W6-d | Final figures, supplementary materials, README updates. | 1 day |

## What is explicitly de-scoped

- **LSTM baseline**. Destroyed-structure controls + 5-domain evidence
  already do the architectural-specificity work. Adding LSTM is
  defensive, not load-bearing.
- **SAE / causal scrubbing / fancier interp methods**. The simplicity
  of classical methods is a feature here — it lets the comparison span
  domains cleanly.
- **Mainline-LLM probe** (Llama-class). Out of compute budget; the
  scale-demo in W3-c on cities does the necessary scaling work.
- **Maze, TCP, code-exec all three** — pick ONE for W4 ex-ante.

## Critical path

W3-a (pre-registration setup) → W4-b (write predictions) → W4-c-d-e
(run experiments and record outcomes) → W5-c (writeup framing) →
W6-c (final pass).

W3-b and W3-c are parallel-able with the W4 work if compute budget
allows.

## Acceptance criteria for "ready to share"

- All W3-W4 experiments complete with documented predictions and
  outcomes
- Methodology framing crisp in the abstract and intro
- Limitations section complete
- End-to-end reproducibility passes
- Pre-registration file untouched since lockdown (verifiable in git
  history)
