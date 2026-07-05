# Predictions — HTTP recent-large-response path carry-through

**Task tag**: `http_recent_large_response_path_carrythrough`
**Parent domain**: `http_log_sequences`
**Predictions committed (timestamp)**: <filled at commit time by `git log`>
**Predictions author**: Raghavan
**Status**: locked

## Question

The strongest critique of "architectural carry-through" is that the existing
positives are too easy:

- first path token in maze
- first request size-bin in HTTP
- possibly fixed-relative local copying inside repeated HTTP request blocks

This follow-up asks for a harder, content-addressed variant:

> At the current request's `size_bin` token, can the residual recover the path
> category of the most recent earlier large response?

A "large response" is a request with `size_bin >= 5` (size >= 100,000 bytes),
matching the earlier HTTP Feature B definition.

This is not a fixed first slot. It is not a fixed relative offset. The source
request depends on the content of earlier size-bin tokens:

    find largest j < k such that size_bin_j >= 5
    target = path_category_j
    probe  = residual at sz_k

This is the most interesting of the three quick follow-ups and the least
guaranteed. It tests whether carry-through extends from regular slot copying
toward content-selected event memory.

## Setup

Use the existing NASA-HTTP per-field corpus:

    [BOS, m_1, p_1, s_1, sz_1, m_2, p_2, s_2, sz_2, ..., EOS]

- **Model**: existing HTTP checkpoint/config, `model/configs/small_http.py`.
- **Checkpoint**: `checkpoints/http_real/best.pt` for the primary result.
- **Controls**: if time permits, also run `http_within_shuffled` and
  `http_global_shuffled` with their existing checkpoints.
- **Probe family**: linear and MLP probes at each layer.
- **Split**: session-level split is the headline.
- **Seeds**: 5 seeds preferred; 3 seeds acceptable for a quick follow-up if
  clearly labeled.

## Target definition

For each request `k` in held-out val/gen sessions:

1. Find the most recent earlier request `j < k` such that `size_bin_j >= 5`.
2. If no such `j` exists, skip this probe example.
3. Probe at the current request's `size_bin` field token `sz_k`.
4. Label the example with `path_category_j`.

Primary analysis:

- Include all examples with lag `k - j >= 1`.

Adversarial lag-filtered analysis:

- Also report the subset with lag `k - j >= 3`.
- If runtime forces only one analysis, use the full lag `>= 1` version as
  primary and report the `>= 3` class distribution as future work.

Expected empirical distribution from the existing val/gen corpus:

Full lag `>= 1`:

- About 267k examples.
- About 8 target classes.
- Majority class about 37.6%.
- Mean lag about 2.8 requests.

Lag `>= 3` subset:

- About 109k examples.
- About 8 target classes.
- Majority class about 40%.

Class-balanced sampling is allowed and preferred for the primary probe. The
class distribution before and after sampling must be printed.

## Prediction

**Verdict**: weakly encoded in the full-lag analysis; ambiguous-to-weakly
encoded in the lag-filtered analysis.

Rationale: the model sees a repeated record structure and can attend to recent
large-response tokens. The source event is often close to the probe position,
so residual directions from the most recent large event can plausibly persist.
However, this feature requires content selection by `size_bin >= 5`, so it is
harder than same-request or previous-request copying and should be treated as
an adversarial test of the carry-through story.

Primary quantitative prediction, full lag `>= 1`:

- Best-layer trained-vs-untrained MLP gap: **>= +0.05**.
- Best-layer trained-vs-untrained linear gap: **>= +0.02**.
- Best-layer MLP gap may be smaller than same-request and previous-request
  path carry-through.

Stronger threshold:

- MLP gap >= +0.10 would count as strong evidence that carry-through extends
  beyond fixed-relative copying.

Lag `>= 3` subset prediction:

- Best-layer trained-vs-untrained MLP gap: **>= +0.03** is weak evidence.
- MLP gap >= +0.05 is the pre-specified confirmation threshold for the
  lag-filtered subset.
- MLP gap < +0.03 is ambiguous/null.

Expected ordering:

    same-request path gap
      >= previous-request path gap
      >= recent-large-response path gap

This ordering is qualitative but important. If recent-large-response is
stronger than the fixed-relative probes, that would suggest the probe is
picking up target distribution or position/session-phase artifacts.

## Falsification thresholds

Full lag `>= 1`:

- **Confirmed**: best-layer MLP gap >= +0.05.
- **Strongly confirmed**: best-layer MLP gap >= +0.10.
- **Ambiguous**: best-layer MLP gap in [+0.03, +0.05).
- **Null**: best-layer MLP gap < +0.03 at every layer.

Lag `>= 3` subset:

- **Confirmed**: best-layer MLP gap >= +0.05.
- **Weak evidence**: best-layer MLP gap in [+0.03, +0.05).
- **Null**: best-layer MLP gap < +0.03 at every layer.

A null here does not falsify architectural carry-through generally. It narrows
the claim to fixed-slot and fixed-relative repeated-layout persistence. A
positive result is much more interesting than the earlier first-slot positives.

## Controls and caveats

### Position/session-phase confounding

The existence and identity of the most recent large response correlate with
request index and session phase. Therefore:

- Session-level split is the headline.
- Print lag distribution and target distribution.
- If quick, report a request-index or lag-only baseline.
- Do not claim "the model computes semantic large-response memory" from a
  probe alone; phrase as residual recoverability after content-selected event
  filtering.

### Current-token leakage

The probe is at `sz_k`, while the target is `path_category_j` for `j < k`. The
current token is not itself the label. Still, nearby path categories may be
correlated within sessions, so compare against previous-request path results.

### Destroyed-structure conditions

Predictions for controls:

- **Within-shuffled request-blocks**: the "most recent previous large
  response" is computed in shuffled block order. Since block content remains
  intact but chronology is destroyed, a signal may remain if the mechanism is
  purely architectural over the observed token stream. Prediction: gap in
  [0.00, +0.10], likely weaker than real.
- **Global-shuffled token alphabet**: labels are original feature values while
  token identities are scrambled. Prediction: gap in [0.00, +0.08], weaker and
  less stable than real.

## Interpretation

If confirmed, this gives the LessWrong post a genuinely nontrivial update:

> carry-through is not just first-slot copying or fixed-relative previous-field
> copying; the residual can preserve information about a content-selected
> earlier event.

If null, that is still useful:

> carry-through appears real for fixed slots and repeated-layout offsets, but I
> do not yet have evidence that it supports content-addressed event memory.

This result should be described as a quick adversarial follow-up, not as a new
full theory of transformer memory.

## Amendments (post-lockdown)

[empty until first amendment]
