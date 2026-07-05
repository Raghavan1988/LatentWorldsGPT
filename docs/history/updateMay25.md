# Session update — 2026-05-25

Pivot.md Milestone 2 (music) first-pass result. Covers everything from the
docs-reframing of M2 (commit `1a95659`) through the data pipeline, eval
suite, and three-condition gradient run on Bach chorales.

---

## Bottom line up front

The cities → music pivot **partially landed and partially did not**. What
worked and what didn't:

- **Voice-leading rate (the music-domain analogue of cities' valid-edge
  rate) produced a clean three-condition gradient as predicted.** Real
  Bach model: 96.25% strict (≤7st) / 98.33% loose (≤12st); within-piece-
  shuffled: 64.33% / 86.80%; global-shuffled: 55.91% / 78.74%. Median
  same-voice interval: 2 / 5 / 7 semitones across the gradient. Per-voice
  rates differentiated only in the real model (S=98, A=97, T=97, B=93 —
  bass slightly looser, musically correct).
- **Perplexity gradient also clean and predicted:** val_ppl 3.84 / 22.16 /
  27.27 across the three conditions. Same monotonic destruction shape as
  cities (1.65 / 25.0 / 313).
- **Mode probe (cities-analogue prediction): partial confirmation.** Real
  60% / within 60% / global 55% on the PIECE-LEVEL (held-out pieces)
  split, with chance ~50% and majority-class baseline 58%. Real and within
  tied → set-membership leak as the cities-analogue predicted; global
  drops to near chance.
- **Beat probe (predicted Othello-positive): inconclusive.** Real,
  within-shuffled, global-shuffled all sit at ~26–27% on PIECE-LEVEL
  (chance 25%). A heavier sanity run on the real model (n=10k positions,
  100 epochs) found beat MLP 32% — barely above chance. **Either (a) the
  model genuinely doesn't encode beat at all because Bach voice-leading
  doesn't require it; or (b) the model is undertrained at 154 visits/token
  (below CLAUDE.md's 200-visit threshold); or (c) the probe is too thin.**
  Until we resolve this, pivot.md Milestone 2 outcome A (predicted mixed
  verdict) cannot be declared.

**Net verdict against the M2 outcome matrix:** closer to a hybrid of A and
C than any single named outcome. Mode behaves cities-like (the A
prediction). Beat does NOT cleanly hit the Othello-like positive that A
needed. Voice-leading (which the outcome matrix didn't enumerate, because
it wasn't framed as a probe) is the strongest, cleanest result of the
session.

The honest reframe: **valid-voice-step rate, not the three classification
probes, is the load-bearing positive in the music domain.** It is the
direct analogue of cities valid-edge rate and produces the same shape of
result. The probe results are secondary diagnostics.

---

## What was implemented this session

### Pipeline

| File | Purpose |
|---|---|
| `data/prepare_music.py` | music21 Bach chorale → 4-voice quarter-note token stream + key/chord/beat CSV side tables. THE ONE RULE invariant asserted on token streams. `--shuffle_within_piece` and `--shuffle_globally` flags (two-tier destroyed-structure template). |
| `tests/test_prepare_music.py` | 5 offline tests on synthetic 4-voice scores: tokenizer bijection, encoding layout, no-probe-target-leakage, shuffle correctness, dump roundtrip. |
| `eval/probe_music.py` | Per-layer linear + MLP classification probes (mode 2-class, chord ~150-class, beat 4-class), position-level + piece-level (held-out-pieces) splits, untrained-model control. Mirrors `eval/probe.py` architecture. |
| `eval/valid_voice_step.py` | Music-domain valid-edge analogue: next-step voice-leading validity + full-piece generation validity. Strict/loose semitone bands (±7 / ±12). |
| `requirements.txt` | `music21>=10.0` added. |
| `.gitignore` | Extended for music probe-target CSVs and `cache/`. |

### Corpora and checkpoints

| Corpus | Train tokens | Val ppl (best) | Best ckpt iter |
|---|---:|---:|---:|
| `data/music_bach` (real) | 52,902 | **3.84** | 500 |
| `data/music_bach_within_shuffled` | 52,902 | **22.16** | 250 |
| `data/music_bach_global_shuffled` | 52,902 | **27.27** | 250 |

All three models: 313 chorales after 4/4+SATB filter; transposed to C major
/ A minor; vocab=60 (pitch tokens + 4 reserved). Same nanoGPT small.py
config (10.74M params), same training schedule.

All three overfit hard with the small.py config (designed for cities, with
1M+ token corpora). Music has only ~53k train tokens — ~154 visits per
token, below the 200-visit threshold from CLAUDE.md. Best checkpoints
were saved at first val improvement and then val perplexity climbed
monotonically. This is a sizing issue, not a methodology issue — but it
limits how confidently we can read the beat probe's null result.

---

## Results

### Three-condition gradient (Bach chorales)

| Metric | Real | Within-shuffled | Global-shuffled | Predicted? |
|---|---:|---:|---:|---|
| Best val perplexity | **3.84** | 22.16 | 27.27 | ↑ (✓ monotonic) |
| Voice-leading rate ≤7st (gen) | **96.25%** | 64.33% | 55.91% | ↓ (✓ clean) |
| Voice-leading rate ≤12st (gen) | **98.33%** | 86.80% | 78.74% | ↓ (✓) |
| Median \|interval\| (semitones) | 2 | 5 | 7 | ↑ (✓) |
| Per-voice differentiation (S/A/T/B) | yes | no | no | ✓ |
| Probe: Mode PIECE-LEVEL | 60% | 60% | 55% | flat real/within (✓ cities-analogue leak), drop on global |
| Probe: Mode POSITION-LEVEL | 88% | 88% | 81% | high all — memorization artifact |
| Probe: Beat PIECE-LEVEL | 26–32% | 27% | 27% | flat ~chance (25%) → **does NOT match Othello-positive prediction** |

Probe runs: n_positions=5,000, epochs=50 for the consistent three-way
comparison. A heavier sanity run on the real model alone (n=10,000,
epochs=100) brought beat probe to 32% (MLP) on PIECE-LEVEL — still
weak. Mode probe to 70% PIECE-LEVEL — clear signal but doesn't change the
real/within tied-at-60 finding.

### Comparison to pivot.md Milestone 2 outcome matrix

| Joint outcome | Predicted shape | Observed? |
|---|---|---|
| **A. Predicted mixed verdict** (key+chord survive, beat collapses) | mode survives shuffle, beat HIGH on real, LOW on shuffle | **No.** Mode survives (✓) but beat doesn't show HIGH on real. |
| **B. Universal cities-like failure** (all survive) | all probes HIGH on real AND shuffle | Partial. Mode is cities-like; beat is uniformly low. |
| **C. Universal Othello positive** (all collapse) | all probes HIGH on real, LOW on shuffle | No. Beat isn't HIGH on real. |
| **D. Graded leak** (partial-mixed) | varies | Best fit: mode = cities-like, beat = no encoding, voice-leading = clean gradient. |

The observed pattern doesn't sit cleanly in any of A–D. The actual story
is more like:

> *Voice-leading (a structural metric, not a probe) shows the clean
> three-condition gradient pivot.md hoped for. Mode probe reproduces the
> cities-analogue co-occurrence leak. Beat probe shows neither the
> Othello-positive nor the cities-analogue — it shows essentially no
> encoding, possibly because beat is not necessary for the model's
> next-pitch prediction objective in this tokenization, or because the
> model is undertrained at this scale.*

---

## Methodology assets graduated this session

- **Valid-voice-step rate as the music-domain valid-edge analogue.** This
  is now the single clearest comparable metric across cities and music.
  Pattern: cities valid-edge (99.7% → 0.06% → 0.006%), music valid-voice-
  step (96.25% → 64.33% → 55.91%). Same shape, different absolute
  ceilings (music can't reach 100% because vocal style allows variation).
- **Piece-level split as the per-domain analogue of cities' node-level
  split.** Probe trained on positions from one set of pieces, tested on
  positions from disjoint pieces — the capacity-controlling generalization
  test. Worked: it reveals that the mode-probe POSITION-LEVEL signal (88%)
  is largely memorization artifact, with the real PIECE-LEVEL signal at 60%.
- **Two-tier destroyed-structure controls (within-piece + global) transfer
  cleanly from cities to music.** The same `--shuffle_within_piece` / 
  `--shuffle_globally` flag pattern, with the same predictable two-step
  destruction of voice-leading + perplexity.

---

## Open questions and decision points (framing next)

### 1. Is the beat-probe null real, or is it an artifact?

Three candidate explanations, each with a different next step:

- **(a) Real — beat is not needed for Bach voice-leading.** If the model
  predicts S/A/T/B's next pitch via local 4-back context, beat-in-measure
  is dispensable. To test: train a larger / better-regularized model on
  the same data; if beat probe still fails, the null is genuine and we
  have an interesting *negative* result — "beat is the kind of structural
  property the model doesn't need, so doesn't encode."
- **(b) Undertraining — 154 visits/token is below threshold.** Train on
  a corpus where vocab × visits/token works out: either larger corpus
  (add Renaissance and Romantic 4-voice writing) or smaller / more
  regularized model. medium.py is one option; a custom config with more
  dropout + smaller n_embd is another.
- **(c) Probe too thin.** Run with n_positions=20k+, epochs=200+, and
  add the untrained-model control to all three conditions so we can read
  the trained-vs-untrained gap directly.

Recommended next step: **(c) first** (cheap, ~30 min compute), then **(b)
if (c) doesn't move the needle** (1 day of work to retrain).

### 2. Should the music-domain framing center on probes or on valid-voice-step?

Currently pivot.md frames Milestone 2 around three probe targets. The
session result suggests **valid-voice-step is the strongest single
contribution** — the cleanest gradient, the most directly comparable to
cities, the easiest to interpret. The probe story is a secondary diagnostic.

If we accept this reframe, M2's "load-bearing positive" becomes
voice-leading instead of beat-probe. The within-domain mixed-verdict
ambition has to wait for either heavier probing or a different corpus.

### 3. Status of the joint-outcome decision

Per pivot.md: "The framing decision (which of A–D the paper centers on)
is made after the three-probe × three-condition table is in hand, not
before." The table is now in hand. None of A–D fit cleanly. Options:

- **Defer the decision** until we run option-(c) heavier probes and
  option-(b) retrain.
- **Reframe to a 5th case** that wasn't anticipated: "graded leak across
  metrics, with valid-voice-step as the load-bearing structural metric
  and mode-probe as the cities-analogue confirmation; beat probe is
  inconclusive at our training budget."
- **Fall back to outcome B** and treat music as a cities-analogue
  cautionary tale plus a working valid-voice-step demonstration.

---

## What's next, concretely

| Effort | What | Outcome |
|---|---|---|
| **30 min** | Run heavier probes (n=20k, epochs=200, all three conditions, with untrained control) | Resolves whether the beat-probe null is genuine vs sampling artifact. |
| **1 day** | Retrain with smaller / better-regularized model (e.g. n_embd=192, dropout=0.3, 2000 iters with weight_decay=0.3); if data is still small, expand corpus to include Palestrina + Lassus 4-voice writing. | Resolves the undertraining hypothesis. |
| **2 days** | Run Symmetric-group-GPT (pivot.md Milestone 1) as the methodology calibration we deferred. | Independent sanity check: does `eval/probe_music.py` correctly recover a probe target on a domain where the answer is known by construction? |

The Milestone 1 (sym-group) detour is now more attractive than it was
before this session: if sym-group's probe trivially recovers the
permutation target (>0.9 accuracy as `pivot.md` predicts) then the
beat-probe null on music is genuinely about the music domain, not about
the probe code. If sym-group also fails, there's a bug in the probe.

---

## Confidence summary (updated)

| Claim | Confidence (delta from pivot.md) |
|---|---|
| Valid-voice-step gradient is reproducible | **~95% (NEW)** — cleanly observed first try |
| Mode probe shows cities-analogue leakage | **~75%** — observed at 60%/60%/55% with reasonable separation |
| Beat probe encodes beat in a probe-recoverable form | **~25%** (was 55% before observation) — the null is real but explanations (a/b/c) not yet differentiated |
| Music joint outcome lands as A (predicted mixed verdict) | **~15%** (was 40%) — beat probe doesn't carry it |
| Music joint outcome is A or D | **~50%** (was 60%) |
| Music delivers SOMETHING publishable | **~85%** (was 90%) — valid-voice-step is the safety net |
| The within-domain mixed-verdict thesis survives M2 as stated | **~30%** (was 70%) — needs heavier probes + retraining to claim it |

---

## Pointers

- `data/prepare_music.py`, `eval/probe_music.py`, `eval/valid_voice_step.py`
  for the implementation.
- `checkpoints/music_bach/best.pt`,
  `checkpoints/music_bach_within_shuffled/best.pt`,
  `checkpoints/music_bach_global_shuffled/best.pt` for the trained models.
- `pivot.md` Milestone 2 outcome matrix for the framing context.
- `update_may24_final.md` for the prior session's cities decomposition
  result that this work builds on.
