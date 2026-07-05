# Results — maze C3 carry-through

Companion to `predictions/predictions_maze_c3_carrythrough.md`, locked at
commit `d7dabf6` on 2026-07-04 before the C3 probe was run.

## Setup as run

- **Checkpoint**: `checkpoints/maze_8x8/best.pt`
- **Data**: `data/maze_8x8`
- **Model**: original maze checkpoint, `n_layer=6`, `n_head=6`,
  `n_embd=192`, `block_size=64`, about 2.68M parameters.
- **Target**: `c3_cell`, the fourth path cell in each maze path.
- **Probe positions**: late path positions.
- **Split**: train probes on `val` mazes, evaluate on `gen` mazes.
- **Seeds**: 5.
- **Sample size**: 20,000 train examples and 20,000 test examples per seed.
- **Probe family**: linear and MLP probes at every layer.
- **Baseline**: untrained model with the same architecture.

## Primary run: path index `t >= 9`

This is the pre-registered headline condition. It requires at least a six-cell
gap between `c3` and the probe position.

Eligible examples:

| Split | Examples | Classes | Majority |
|---|---:|---:|---:|
| val | 71,389 | 64 | 0.021 |
| gen | 71,343 | 64 | 0.020 |

Aggregate over 5 seeds:

| Layer | Trained linear | Trained MLP | Untrained linear | Untrained MLP | Linear gap | MLP gap |
|---|---:|---:|---:|---:|---:|---:|
| embed | 0.0234 | 0.0233 | 0.0228 | 0.0232 | +0.0005 | +0.0001 |
| L0 | 0.0250 | 0.0298 | 0.0252 | 0.0397 | -0.0002 | -0.0100 |
| L1 | 0.0373 | 0.0688 | 0.0272 | 0.0451 | +0.0102 | +0.0238 |
| L2 | 0.0487 | 0.0841 | 0.0283 | 0.0482 | +0.0205 | +0.0359 |
| L3 | 0.0525 | 0.0847 | 0.0295 | 0.0485 | +0.0230 | +0.0362 |
| L4 | 0.0574 | 0.0881 | 0.0298 | 0.0507 | +0.0276 | +0.0374 |
| L5 | 0.0645 | 0.1032 | 0.0302 | 0.0501 | **+0.0344** | **+0.0532** |

Headline:

- Best linear gap: `+0.0344` at L5.
- Best MLP gap: `+0.0532` at L5.
- Locked-threshold verdict: **ambiguous**.

The primary prediction was not confirmed. The MLP gap lands inside the
pre-registered ambiguous band `[+0.04, +0.08)`, and the linear gap is below
the predicted `+0.04` threshold.

## Late-only run: path index `t >= 13`

This was the pre-registered adversarial subset. It asks whether recoverability
survives farther from the source token.

Eligible examples:

| Split | Examples | Classes | Majority |
|---|---:|---:|---:|
| val | 44,957 | 64 | 0.021 |
| gen | 44,837 | 64 | 0.020 |

Aggregate over 5 seeds:

| Layer | Trained linear | Trained MLP | Untrained linear | Untrained MLP | Linear gap | MLP gap |
|---|---:|---:|---:|---:|---:|---:|
| embed | 0.0266 | 0.0265 | 0.0260 | 0.0267 | +0.0006 | -0.0002 |
| L0 | 0.0276 | 0.0315 | 0.0286 | 0.0408 | -0.0010 | -0.0092 |
| L1 | 0.0367 | 0.0522 | 0.0292 | 0.0458 | +0.0074 | +0.0064 |
| L2 | 0.0453 | 0.0599 | 0.0308 | 0.0476 | +0.0146 | +0.0123 |
| L3 | 0.0485 | 0.0644 | 0.0313 | 0.0491 | +0.0172 | +0.0153 |
| L4 | 0.0516 | 0.0676 | 0.0324 | 0.0497 | +0.0192 | +0.0179 |
| L5 | 0.0583 | 0.0816 | 0.0321 | 0.0494 | **+0.0262** | **+0.0322** |

Headline:

- Best linear gap: `+0.0262` at L5.
- Best MLP gap: `+0.0322` at L5.
- Locked-threshold verdict: **null / surprising**.

## Verdict

The C3 carry-through prediction was **not confirmed**.

The primary condition shows weak recoverability above untrained baseline, but
not enough to clear the locked confirmation threshold. The late-only subset
falls below even the ambiguous band.

## Interpretation

This narrows the maze-side carry-through story. The original maze result still
shows starting-cell persistence, but this follow-up does not support a strong
claim that arbitrary early maze path cells are carried forward with similar
strength.

The HTTP follow-ups remain the cleaner evidence that carry-through can happen
outside first slots:

- same-request path: `+0.809`
- previous-request path: `+0.674`
- recent-large-response path: `+0.621` full lag, `+0.514` for lag `>= 3`

For the LessWrong post, this should be reported as a useful failed/ambiguous
follow-up, not hidden. It weakens the maze-specific version of the claim while
leaving the HTTP carry-through ladder intact.
