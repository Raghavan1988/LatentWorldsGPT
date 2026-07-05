# Session update — 2026-07-04

Today we turned the "architectural carry-through" idea into a cleaner
LessWrong-sized story by running three HTTP follow-ups beyond first-slot
copying.

## What changed today

- Locked and pushed the third HTTP follow-up prediction:
  `predictions/predictions_http_recent_large_response_path_carrythrough.md`
  at commit `8d671ca`.
- Added and pushed the eval scripts for the recent-large-response path
  experiment at commit `6a13599`:
  - `eval/probe_http_recent_large_response_path.py`
  - `eval/intervene_http_recent_large_response_path.py`
- Ran the recent-large-response probe at full lag `>= 1` and at the stricter
  lag filter `>= 3`.
- Ran corruption interventions for the same target at lag `>= 1` and `>= 3`.
- Updated the durable docs so the repo no longer tells the older "two
  follow-ups" story.

## Three HTTP carry-through follow-ups

All three use the already-trained NASA-HTTP model and probe at the current
request's `size_bin` token `sz_k`.

| Follow-up | Source feature | Best MLP gap | What it tests |
|---|---|---:|---|
| Same-request path | `p_k` | `+0.809` | local repeated-layout carry-through |
| Previous-request path | `p_{k-1}` | `+0.674` | cross-record fixed-offset carry-through |
| Recent-large-response path | `p_j`, where `j < k` and `size_bin_j >= 5` | `+0.621` full lag; `+0.514` for lag `>= 3` | content-selected earlier-event recoverability |

The ordering matched the pre-registered qualitative prediction:

```text
same-request path > previous-request path > recent-large-response path
```

This matters because the third case is not the first request, not the first
path token, and not a fixed relative offset. The source request is selected by
content: "the most recent earlier request whose size bin was large."

## Intervention reading

The intervention results are more cautious than the probe results.

| Follow-up | Main intervention result |
|---|---|
| Same-request path | Corrupting `p_k` strongly hurts prediction of `sz_k` (`+4.03` natural dNLL), but only weakly affects the token after `sz_k` (`+0.083`). |
| Previous-request path | Corrupting `p_{k-1}` mildly hurts prediction of `sz_k` (`+0.187` natural dNLL), and barely affects the token after `sz_k` (`+0.024`). |
| Recent-large-response path, lag `>= 1` | Corrupting selected `p_j` weakly hurts prediction of `sz_k` (`+0.084` natural dNLL), with little after-size effect (`+0.009`). |
| Recent-large-response path, lag `>= 3` | The causal effect becomes tiny: `+0.013` natural dNLL on `sz_k`, `+0.003` after `sz_k`. |

So the best interpretation is:

> The residual stream strongly preserves recoverable traces of structured
> recent context, including a content-selected earlier event. But the model
> only weakly uses that trace for the immediate next-token prediction once the
> source event is farther away.

## LessWrong story after today

The cleaner post is not "I found semantic memory." It is:

> I expected next-token transformers to mostly represent features they need.
> That was too simple. In a tiny HTTP transformer, information remains strongly
> decodable in the residual stream even after its direct next-token usefulness
> has mostly faded. Attention and residual streams make structured context easy
> to carry forward, and the training objective does not strongly penalize
> carrying it.

The post should emphasize the ladder:

1. same-request path: easy, locally useful, huge effect;
2. previous-request path: cross-record, still strongly decodable, weaker use;
3. recent-large-response path: content-selected, strongly decodable, weakly
   useful under lag.

That ladder is the nontrivial update. It makes "architectural carry-through"
less like a first-slot artifact and more like residual persistence of recent
structured context.

## What to avoid overclaiming

- Do not claim this is a robust semantic memory system.
- Do not claim the probe result alone proves the model uses the feature.
- Do not claim the N-criterion is rescued. The broader predictive-relevance
  theory remains weakened by the maze and HTTP failures.
- Do not call the recent-large-response result fully causal: the probe is
  strong, but the lag-filtered intervention is weak.

## Files updated after this result

- `README.md`
- `paper_draft.md`
- `report.md`
- `results_http_log_sequences.md`
- `REPRODUCIBILITY.md`
- `docs/AUDIT_TRAIL.md`
- `figures/02_results_matrix.md`
- `figures/README.md`
- `LessWrongBlogPost.md`
- `lesswrong_draft.md`
- `predictions/README.md`
- `predictions/predictions_http_log_sequences.md`
- `predictions/predictions_http_same_request_path_carrythrough.md`
- `predictions/predictions_http_previous_request_path_carrythrough.md`
- `predictions/predictions_http_recent_large_response_path_carrythrough.md`

Older May planning files remain historical records. Where they presented
themselves as current guidance, they now point here for the July 4 state.
