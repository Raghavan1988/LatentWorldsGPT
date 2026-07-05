# Audit Trail

This file summarizes the evidence chain for the two pre-registered
experiments. It is intentionally short; the detailed locked predictions and
result tables remain in `predictions/` and the `results_*.md` files.

## Locked Prediction Commits

| Experiment | Prediction file | Locked commit | Date | Result file |
|---|---|---|---|---|
| Maze navigation | `predictions/predictions_maze_navigation.md` | `aa025b1` | 2026-05-27 | `results_maze_navigation.md` |
| HTTP log sequences | `predictions/predictions_http_log_sequences.md` | `3b25ed3` | 2026-05-31 | `results_http_log_sequences.md` |
| HTTP same-request path follow-up | `predictions/predictions_http_same_request_path_carrythrough.md` | `88155b4` | 2026-07-04 | `results_http_log_sequences.md` |
| HTTP previous-request path follow-up | `predictions/predictions_http_previous_request_path_carrythrough.md` | `0d115c2` | 2026-07-04 | `results_http_log_sequences.md` |
| HTTP recent-large-response path follow-up | `predictions/predictions_http_recent_large_response_path_carrythrough.md` | `8d671ca` | 2026-07-04 | `results_http_log_sequences.md` |

Verify locally:

```bash
git log --diff-filter=A predictions/predictions_maze_navigation.md
git log --diff-filter=A predictions/predictions_http_log_sequences.md
git log --diff-filter=A predictions/predictions_http_same_request_path_carrythrough.md
git log --diff-filter=A predictions/predictions_http_previous_request_path_carrythrough.md
git log --diff-filter=A predictions/predictions_http_recent_large_response_path_carrythrough.md
```

## Maze Navigation

Locked claim:

- The starting cell ID was predicted to be null: trained-vs-untrained MLP
  gap `<= 0.10`.
- The explicit falsifier in the locked file was gap `> 0.15`.

Observed:

| Target | Layer | Trained MLP | Untrained MLP | Gap | Verdict |
|---|---:|---:|---:|---:|---|
| Starting cell ID | L5 | `0.2024 +/- 0.0166` | `0.0504 +/- 0.0082` | `+0.152` | Null band violated; explicit falsifier crossed |

Interpretation:

- The strict N-criterion predicted absence and failed.
- The result motivated the later input-slot persistence / carry-through
  hypothesis.
- Maze row/column predictions were marked predictor error because row and
  column were deterministic functions of token ID.

## HTTP Log Sequences

Locked claims:

- Feature A (`first_request_size_bin`) was predicted encoded via
  input-slot carry-through, gap `>= 0.10`.
- Feature B (`cumulative_large_response_binned`) was predicted null,
  gap `<= 0.10`, with falsifier gap `> 0.15`.
- Cross-feature ordering predicted Feature A gap `>=` Feature B gap.

Observed initial probe:

| Target | Layer | Trained MLP | Untrained MLP | Gap | Verdict |
|---|---:|---:|---:|---:|---|
| Feature A | L3 | `0.410 +/- 0.04` | `0.243 +/- 0.03` | `+0.168` | Confirmed |
| Feature B | L2 | `0.888 +/- 0.004` | `0.597 +/- 0.01` | `+0.291` | Null falsified |

Post-hoc position controls:

| Control | Real-condition headline | Interpretation |
|---|---:|---|
| Pure position probe | MLP gap `+0.427` | Position is strongly encoded and can inflate Feature B |
| Fixed-position Feature B (`k=5`) | MLP gap `+0.220` | Feature B remains above threshold with position held constant |
| Residual-after-position Feature B | R2 gap `+0.468` | Feature B remains recoverable after subtracting per-position baseline |

Interpretation:

- Feature A is the one forward-looking carry-through prediction that
  confirmed on HTTP.
- Feature B falsified the graded framework's null direction.
- The position-control diagnostics were post-hoc and should be
  pre-registered in future experiments where target labels correlate with
  token position.

## HTTP Carry-Through Follow-Ups

The July 4 follow-ups tested whether carry-through extends beyond fixed
first-request slots. All three were locked before their corresponding evals
were run.

| Follow-up | Probe target | Headline probe | Intervention reading |
|---|---|---:|---|
| Same-request path | recover `p_k` at `sz_k` | MLP gap `+0.809` | Strongly useful for predicting `sz_k`; weak after `sz_k`. |
| Previous-request path | recover `p_{k-1}` at `sz_k` | MLP gap `+0.674` | Modestly useful for predicting `sz_k`; tiny after `sz_k`. |
| Recent-large-response path | recover most recent earlier large-response `p_j` at `sz_k` | MLP gap `+0.621`; lag `>= 3` gap `+0.514` | Strongly decodable but weakly useful under lag-filtered corruption. |
