# Predictions — HTTP same-request path carry-through

**Task tag**: `http_same_request_path_carrythrough`
**Parent domain**: `http_log_sequences`
**Predictions committed (timestamp)**: <filled at commit time by `git log`>
**Predictions author**: Raghavan
**Status**: locked

## Question

The earlier HTTP pre-registration tested whether the first request's
`size_bin` persists later in the residual stream. A natural criticism is
that this is only fixed-position copying from the first request.

This follow-up asks for a smaller but cleaner non-first-token carry-through
case:

> At a request's `size_bin` token, can the residual stream recover the
> `path_category` token from the same request?

The source token is not the first request and not at a fixed absolute
position across the full stream. It is a repeated-layout field two tokens
earlier in the same four-token request block:

    [m_k, p_k, s_k, sz_k]
              ^       ^
           source   probe

This is the easiest non-first-slot carry-through test. It should be treated
as a sanity check, not the most interesting substantive claim.

## Setup

Use the existing NASA-HTTP per-field corpus:

    [BOS, m_1, p_1, s_1, sz_1, m_2, p_2, s_2, sz_2, ..., EOS]

- **Model**: existing HTTP checkpoint/config, `model/configs/small_http.py`.
- **Checkpoint**: `checkpoints/http_real/best.pt` for the primary result.
- **Controls**: if time permits, also run `http_within_shuffled` and
  `http_global_shuffled` with their existing checkpoints.
- **Probe family**: linear and MLP probes at each layer, same basic code path
  as `eval/probe_http.py`.
- **Split**: session-level split is the headline. Position-level split may be
  reported as a weak diagnostic only.
- **Seeds**: 5 seeds if runtime is acceptable; 3 seeds acceptable for a quick
  LessWrong addendum, explicitly labeled as such.

## Target definition

For each request `k >= 1` in held-out val/gen sessions:

- **Probe position**: the `size_bin` field token `sz_k`.
- **Source feature**: the same request's `path_category` field `p_k`.
- **Label**: integer path category token class, e.g. `p_0`, `p_1`, ...

Request `k=0` may be included or excluded. The preferred primary analysis uses
`k >= 1` so the test is not accidentally dominated by first-request behavior.

Expected empirical distribution from the existing val/gen corpus:

- About 432k probe examples.
- About 15 path classes.
- Majority class about 22%.

Class-balanced sampling is allowed and preferred for the primary probe, so the
trained-vs-untrained gap is not hidden by path-category imbalance. The sampled
class distribution must be printed in the result log.

## Prediction

**Verdict**: encoded.

Rationale: by the time the model reaches `sz_k`, the path token `p_k` is only
two tokens back. Self-attention has a short, regular, repeated-layout route
from `p_k` to `sz_k`. If architectural carry-through applies beyond first
tokens at all, this should be the easiest positive case.

Primary quantitative prediction:

- Best-layer trained-vs-untrained MLP gap: **>= +0.10**.
- Best-layer trained-vs-untrained linear gap: **>= +0.05**.
- Best layer: any layer, including embedding, is acceptable.

Expected range, not required for verdict:

- Trained MLP accuracy: 0.45 to 0.90 under class-balanced sampling.
- Trained linear accuracy: 0.35 to 0.85 under class-balanced sampling.

## Falsification thresholds

- **Confirmed**: best-layer MLP gap >= +0.10.
- **Ambiguous**: best-layer MLP gap in [+0.05, +0.10).
- **Falsified / surprising null**: best-layer MLP gap < +0.05 at every layer.

A null here would be strong evidence that the current probe construction or
checkpoint is not sensitive enough for the harder follow-ups. If this test is
null, do not over-interpret nulls on previous-request or content-addressed
carry-through.

## Controls and caveats

### Same-token lexical leakage

The target `p_k` is not the current token. The current token is `sz_k`. So a
probe cannot solve this by reading the current token identity directly.

### Fixed relative offset

This test is still easy because the source is always two tokens earlier. A
positive result is evidence for non-first repeated-layout carry-through, but
not evidence for content-addressed retrieval.

### Position confounding

`path_category` may correlate with request index. Therefore:

- Report a position-only or request-index baseline if quick.
- Prefer session-level split as the headline.
- If a fixed-k analysis is run, label it as a diagnostic, not the primary
  verdict.

### Destroyed-structure conditions

Predictions for controls:

- **Within-shuffled request-blocks**: same-request field layout is preserved,
  so the carry-through route `p_k -> sz_k` should still exist. Prediction:
  MLP gap >= +0.05, possibly similar to real.
- **Global-shuffled token alphabet**: field layout remains but token
  identities are scrambled. Prediction: MLP gap >= +0.05 if the model learns
  the field layout; weaker than real is acceptable.

## Interpretation

If confirmed, this result rebuts the narrowest critique:

> carry-through only works for the first request or first path token.

It does **not** show that the model can find an arbitrary earlier slot by
content. It only shows that repeated-layout, non-first, non-absolute-position
features can persist across a short attention route.

This is a sanity check for the next two follow-ups.

## Amendments (post-lockdown)

[empty until first amendment]
