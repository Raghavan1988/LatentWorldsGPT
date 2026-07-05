# Predictions — maze C3 carry-through

**Task tag**: `maze_c3_carrythrough`
**Parent domain**: `maze_navigation`
**Predictions committed (timestamp)**: <filled at commit time by `git log`>
**Predictions author**: Raghavan
**Status**: locked

## Question

The original maze pre-registration showed that the starting cell remains
recoverable late in the path, contrary to the strict N-criterion's null
prediction. A natural criticism is that this is only first-token copying:

> The model remembers `c0` because `c0` is the first path cell.

This follow-up asks for the smallest maze-side check of that criticism:

> If `c3`, the fourth path cell, appears early in the sequence, is it still
> recoverable from late residual-stream positions?

Use zero-indexed path-cell notation:

```text
[BOS, c0=start, c1, c2, c3, c4, ..., goal, EOS]
                       ^              ^
                    source         probe later
```

This is intentionally not the strongest possible memory test. It is a
non-first, fixed-early-slot carry-through test in the same maze domain. The
HTTP recent-large-response follow-up remains the stronger content-selected
test. This maze test exists to check whether the original maze carry-through
result was uniquely first-slot-specific.

## Setup

No model training is required. Use the already-trained maze checkpoints and
existing maze corpora:

- Primary checkpoint: `checkpoints/maze_8x8/best.pt`
- Primary data: `data/maze_8x8`
- Optional controls:
  - `checkpoints/maze_8x8_within_shuffled/best.pt`
  - `data/maze_8x8_within_shuffled`
  - `checkpoints/maze_8x8_global_shuffled/best.pt`
  - `data/maze_8x8_global_shuffled`
- Model: same as original maze run, `n_layer=6`, `n_head=6`,
  `n_embd=192`, `block_size=64`, about 2M parameters.
- Probe family: linear probe and one-hidden-layer MLP probe at every layer
  of the residual stream, matching `eval/probe_maze.py`.
- Seeds: 5 seeds, outermost loop varying activation sampling and probe
  training RNG. If runtime forces a quicker run, 3 seeds is acceptable only
  if explicitly labeled as a quick check.

The existing `eval/probe_maze.py` is hardcoded for row/col/distance/start.
This follow-up will either extend that script or add a small dedicated script
that constructs the `c3_cell` target.

## Target definition

Group `mazes.csv` rows by `(split, maze_idx)` and sort by `token_pos` within
each maze. Let the sorted path-cell rows be:

```text
c0, c1, c2, c3, ..., c_{L-1}
```

For every maze with length `L >= 10`:

- **Source feature**: `c3_cell`, the cell token at path index `3`.
- **Probe positions**: path indices `t >= 9`.
- **Label at every probe position**: the source cell token `c3_cell`.
- **Headline split**: train the probe on held-out `val` mazes and evaluate
  on `gen` mazes, or equivalently use a maze-level split with no maze shared
  between probe train/test. The key requirement is maze-level separation, not
  position-level separation.

The `t >= 9` filter creates at least a six-cell gap between source and probe:

```text
source index = 3
first probe index = 9
lag = 6 path cells
```

This avoids counting the source token itself or its immediate neighbors as
evidence for carry-through.

For optional adversarial reporting, also run a stricter subset:

- **Late-only subset**: path indices `t >= 13`.

The late-only subset is not the primary verdict, but it is a useful check
that the result is not driven by short-range local copying.

## Prediction

**Verdict**: encoded.

Rationale: `c3` is an early, positionally identifiable input token. It is not
the first path token, but it is still easy for a causal transformer to attend
to and preserve. If architectural carry-through is not uniquely first-slot
specific, the residual stream at late path positions should retain a
recoverable trace of `c3`.

Primary quantitative prediction on real maze data:

- Best-layer trained-vs-untrained MLP gap: **>= +0.08**.
- Best-layer trained-vs-untrained linear gap: **>= +0.04**.
- Strong confirmation: best-layer MLP gap **>= +0.12**.
- Best layer: likely L3-L5, but any layer counts for the primary verdict.

Expected magnitude:

- The C3 signal may be weaker than the original starting-cell signal
  (`+0.152` MLP gap), because `c3` is not the path anchor and appears after
  several previous cells.
- The C3 signal should still be clearly above the distance-to-goal signal
  from the original maze result (`+0.009` MLP gap), because `c3` is a concrete
  earlier token and distance-to-goal was not cleanly represented.

Qualitative expected ordering:

```text
start_cell gap >= c3_cell gap >> distance_to_goal gap
```

This ordering is not a falsifier by itself, but it is the intended reading.

## Falsification thresholds

Primary real-data probe:

- **Confirmed**: best-layer MLP gap `>= +0.08`.
- **Strongly confirmed**: best-layer MLP gap `>= +0.12`.
- **Ambiguous**: best-layer MLP gap in `[+0.04, +0.08)`.
- **Null / surprising**: best-layer MLP gap `< +0.04` at every layer.

A null result would narrow the maze-side carry-through story:

> The original maze evidence supports starting-cell persistence, but not
> general early-slot persistence within maze paths.

It would not falsify architectural carry-through generally, because the HTTP
follow-ups already test non-first slots in a richer repeated-record setting.

## Controls and caveats

### Not first-token copying

The source is `c3`, not `c0`. A positive result therefore rebuts the narrowest
maze-specific criticism that the starting-cell result is only a first-path-cell
artifact.

### Still fixed early-slot copying

This is not content-addressed retrieval. The source is at a fixed early path
index. A positive result should be described as non-first fixed-slot
carry-through, not semantic memory.

### Current-token leakage

The probe positions are restricted to `t >= 9`, so the current token is never
the source `c3`, and the probe is at least six path cells after the source.

### Cell-token identity

The target is a 64-class cell token. As with the original starting-cell probe,
the relevant comparison is trained vs untrained at the same target and split.
High absolute accuracy is not meaningful by itself; the load-bearing number is
the trained-vs-untrained gap.

### Destroyed-structure controls

Optional controls should define `c3` as the fourth observed path-cell token in
that condition's token stream:

- **Within-shuffled**: since the observed fourth cell is still at a fixed
  source slot, a carry-through signal may remain. Prediction: MLP gap
  `>= +0.04`, possibly weaker than real.
- **Global-shuffled**: token identities are remapped but the slot structure is
  preserved. Prediction: MLP gap `>= +0.04`, possibly similar to real.

These controls are secondary. The primary question is whether the real maze
model carries a non-first early path cell forward.

## Activation-patching / intervention plan

Probe recoverability is the primary result.

Activation patching is useful only if it is C3-specific. The existing
`eval/transplant_maze.py` transplants an entire donor residual state and
measures whether the model moves toward the donor's next step. That is a good
path-state intervention, but it does not isolate C3 carry-through.

If time permits, run one of these secondary checks:

1. **Source-token corruption**: replace `c3` in the prefix with a random
   different cell token at evaluation time and measure natural dNLL at late
   positions `t >= 9`. Prediction: small or modest effect. A weak effect is
   compatible with carry-through, because the feature may be recoverable
   without being strongly used for next-token prediction.
2. **C3 source-position activation patch**: patch the residual at the `c3`
   source position from a donor maze into a recipient prefix and measure
   whether late-position logits or C3-probe predictions shift toward the donor
   label. Prediction: if implemented cleanly, the C3 probe prediction should
   shift more than natural next-token logits.

These interventions should be reported as causal/use diagnostics, not as the
main confirm/falsify criterion.

## Interpretation

If confirmed, the result should be stated narrowly:

> Maze carry-through is not only first-path-cell persistence. A non-first early
> path cell remains recoverable from later residual-stream positions.

If strongly confirmed and the late-only subset also clears threshold, it
supports the LessWrong claim that architectural carry-through is a general
early-slot persistence effect, not merely a first-token artifact.

If null, the LessWrong post should not hide it. The correct update would be:

> HTTP provides the stronger non-first carry-through ladder; the maze evidence
> remains limited to starting-cell persistence.

## Amendments (post-lockdown)

[empty until first amendment]
