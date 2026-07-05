# Predictions — HTTP previous-request path carry-through

**Task tag**: `http_previous_request_path_carrythrough`
**Parent domain**: `http_log_sequences`
**Predictions committed (timestamp)**: <filled at commit time by `git log`>
**Predictions author**: Raghavan
**Status**: locked

## Question

The earlier carry-through positives used fixed first slots:

- maze: first path token / starting cell
- HTTP: first request's `size_bin`

This follow-up tests a more interesting repeated-record case:

> At the current request's `size_bin` token, can the residual recover the
> previous request's `path_category`?

The source is not the first request. It is not at a fixed absolute position.
It is at a fixed relative record offset:

    [m_{k-1}, p_{k-1}, s_{k-1}, sz_{k-1}, m_k, p_k, s_k, sz_k]
              ^                                           ^
           source                                       probe

This is still much easier than arbitrary content-addressed retrieval, but it
is a better test than first-token copying. Attention has a simple mechanical
route: learn to attend back one request block, then copy or preserve the path
field direction.

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

For each request `k >= 1` in held-out val/gen sessions:

- **Probe position**: the current request's `size_bin` field token `sz_k`.
- **Source feature**: the previous request's `path_category` field
  `p_{k-1}`.
- **Label**: integer path category token class, e.g. `p_0`, `p_1`, ...

Expected empirical distribution from the existing val/gen corpus:

- About 432k probe examples.
- About 15 path classes.
- Majority class about 22.5%.

Class-balanced sampling is allowed and preferred for the primary probe. The
class distribution before and after sampling must be printed.

## Prediction

**Verdict**: encoded.

Rationale: this is a fixed-relative-offset memory task inside a highly regular
four-token record structure. The transformer can route information from
`p_{k-1}` to `sz_k` with attention over a distance of six tokens. The source
slot varies across absolute sequence positions, so a positive result is not
just first-token copying.

Primary quantitative prediction:

- Best-layer trained-vs-untrained MLP gap: **>= +0.10**.
- Best-layer trained-vs-untrained linear gap: **>= +0.05**.
- The same-request path probe should be at least as strong as this probe:
  `gap(same_request_path) >= gap(previous_request_path) - 0.05`.

Expected range, not required for verdict:

- Trained MLP accuracy: 0.35 to 0.85 under class-balanced sampling.
- Trained linear accuracy: 0.25 to 0.75 under class-balanced sampling.

Layer prediction:

- The best layer is likely not earlier than the same-request path probe's best
  layer, because the information must persist across a longer span. This is a
  qualitative expectation, not a strict falsifier.

## Falsification thresholds

- **Confirmed**: best-layer MLP gap >= +0.10.
- **Ambiguous**: best-layer MLP gap in [+0.05, +0.10).
- **Falsified / surprising null**: best-layer MLP gap < +0.05 at every layer.

If same-request path is positive but previous-request path is null, the
carry-through story should be narrowed to local repeated-layout copying rather
than cross-record persistence.

## Controls and caveats

### Not first-token copying

The source token appears at many absolute positions. The prediction does not
depend on the first request, first path token, or fixed absolute position.

### Still fixed-relative-offset

This is not yet content-addressed retrieval. The source is always six tokens
back from the probe position in the unshuffled corpus. A positive result says
attention can preserve a repeated prior-record field; it does not show the
model can search for an event by value.

### Position confounding

Path categories may correlate with request index and local session phase.
Therefore:

- Session-level split is the headline.
- A position/request-index baseline should be reported if quick.
- A fixed-k diagnostic may be run, but should not replace the session-level
  headline.

### Destroyed-structure conditions

Predictions for controls:

- **Within-shuffled request-blocks**: the meaning of "previous request" after
  shuffling is the previous block in the shuffled order, not the chronological
  previous request. Since block layout remains intact, a previous-block
  carry-through signal may still appear. Prediction: MLP gap >= +0.05, but
  possibly weaker or less stable than real.
- **Global-shuffled token alphabet**: field layout remains but token
  identities are scrambled. Prediction: MLP gap >= +0.05 if the model learns
  the field layout; weaker than real is acceptable.

## Interpretation

If confirmed, this becomes the cleanest quick LessWrong result:

> architectural carry-through is not only first-token persistence; it also
> appears in repeated-record layouts where the relevant source slot is a
> non-first, variable-absolute-position previous field.

If null while same-request path is positive, the claim should be weakened:

> the current evidence supports short local carry-through, not robust
> previous-record memory.

## Amendments (post-lockdown)

- 2026-07-04: Confirmed. Five-seed class-balanced probe at `sz_k` recovered
  previous-request `p_{k-1}` with best linear gap `+0.535` at L2 and best MLP
  gap `+0.674` at L2. Paired corruption showed modest usefulness for
  predicting `sz_k` (natural dNLL `+0.187`) and very small usefulness after
  `sz_k` (natural dNLL `+0.024`).
