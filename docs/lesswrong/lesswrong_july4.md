# I was wrong about when transformers represent features

**Epistemic status:** small-model mechanistic interpretability work. Seven
toy or semi-toy domains, 0.27M to 13M parameter decoder-only transformers.
The strongest part of this work is not the object-level theory. The strongest
part is the audit trail: I wrote down quantitative predictions before running
two of the experiments, committed them to git, and then had to live with the
results. Both pre-registered tests weakened my original theory.

**AI assistance disclosure:** the implementation was heavily AI-assisted.
Data pipelines, training scripts, probe code, activation-patching scripts,
and debugging were done with Claude/Codex assistance. I am writing this post
because I think the experiment design and the failures are worth inspecting,
not because I think toy-scale numbers should be taken as strong evidence about
frontier models.

I started with a simple hypothesis about when next-token transformers form
internal representations. The hypothesis was too simple.

The motivating example is Othello-GPT. Li et al. and Nanda et al. showed that
a transformer trained only to predict legal Othello moves develops a
recoverable internal representation of board state. That is a beautiful
result, but it raises the obvious next question: when should we expect this to
happen? If I train a small transformer on another structured domain, which
latent features end up represented in the residual stream?

By "residual stream" I mean the running hidden-state vector at each token
position. Every attention and MLP block reads from it and writes back to it.
By "represented" I mean something modest: a linear probe or small MLP probe
can recover the feature from that vector, above an untrained-model baseline.
This is not the same as proving the model uses the feature. It is just
evidence that the information is present in a usable internal format.

My first hypothesis was what I called the **N-criterion**, where N stands for
next-token necessity:

> A feature is represented in the residual stream iff the feature is needed
> for next-token prediction.

I liked this hypothesis because it made predictions in both directions. It
says not only "useful things are represented," but also "irrelevant things are
absent." The second half is the risky half. It is easy to find a useful
feature after the fact and say the model should have learned it. It is harder
to say in advance: this feature should not be there.

I tested this across seven domains: Othello, music, cities, flight phases,
symmetric-group walks, maze navigation, and HTTP logs. Most of those were
exploratory calibration. The two real tests were maze navigation and HTTP
logs, where I committed prediction files before running the corresponding
models/probes.

The maze prediction was committed on 2026-05-27, commit `aa025b1`. The HTTP
prediction was committed on 2026-05-31, commit `3b25ed3`. You can audit them
with `git log --diff-filter=A` on the prediction files.

## The maze test

The maze task used 8x8 mazes. Each training example was a shortest path from
start to goal, tokenized as cells:

`[BOS, start_cell, step_1, step_2, ..., goal_cell, EOS]`

The model was trained only on next-cell prediction. My prediction was
straightforward: the current cell and distance-to-goal should matter; the
starting cell should not matter once the path has moved on. If you are halfway
through a maze path, the next step depends on where you are and where the goal
is, not where you started.

So I predicted that the starting cell would be null at late positions:
trained-vs-untrained MLP probe gap at most `+0.10`, with `> +0.15` as an
explicit falsifier.

It falsified by a hair, but a real hair. At layer 5, the trained MLP recovered
starting cell with accuracy `0.2024 +/- 0.0166`; the untrained model was
`0.0504 +/- 0.0082`. Gap: `+0.152`.

The more embarrassing result was distance-to-goal. I predicted distance would
be encoded. It was not. The trained-vs-untrained MLP gap was only about
`+0.009`.

So the maze test failed in both directions. A feature I thought should be
represented was absent. A feature I thought should be absent was represented.

That forced the first update: "needed for next-token prediction" and
"recoverable from the residual stream" are different claims. A model can solve
the task without building the named feature I had in mind. And a model can
preserve information that is not needed, because nothing in the objective
directly punishes harmless extra information.

## Architectural carry-through

The starting-cell result suggested a weaker mechanism.

The starting cell is not some hidden latent variable. It is literally the
first path token. A transformer can attend back to a positionally distinct
earlier token and route information forward. The residual stream is not a
minimal sufficient statistic for the next token. It is also a workspace where
information can hang around if the architecture makes it cheap to keep.

I called this **architectural carry-through**.

The idea is not profound. It is close to saying "attention plus residual
streams make copying easy." But I think the distinction matters. Next-token
training rewards information that helps prediction. It does not impose a
strong penalty for retaining extra information, unless that information
interferes with something else. So some features may remain decodable not
because the model still needs them, but because they were easy to carry.

After the maze result, I weakened the theory:

- features required for prediction may be represented, but need not be
  represented in the clean feature basis I expected;
- features not required for prediction may still be represented if they are
  already present in an input slot and easy for attention to carry forward;
- therefore, the negative direction of the N-criterion is suspect.

This would be cheap hindsight unless it made a new prediction. So I used HTTP
logs as the next test.

## The HTTP test

The HTTP task used NASA HTTP logs. Each request was represented as four
tokens:

`[method, path_category, status_bucket, size_bin]`

A session looked like:

`[BOS, m_1, p_1, s_1, sz_1, m_2, p_2, s_2, sz_2, ...]`

I pre-registered two feature probes.

**Feature A:** the `size_bin` of the first request in the session. This is not
obviously useful later, but it is present at a positionally distinct input
slot. Carry-through predicts it should be recoverable.

**Feature B:** the cumulative count of large responses so far, binned as
`{0, 1, 2+}`. This is not sitting at one slot. It requires aggregating across
previous request sizes. I predicted it would be null.

Feature A confirmed. Best MLP gap: `+0.168` at layer 3.

Feature B falsified. Raw MLP gap: `+0.291`.

This was not a clean victory for anything. Carry-through made one correct
forward-looking prediction on a new domain, which was good. But the broader
"computed irrelevant features are absent" claim failed badly.

There was also an important confound. Cumulative count correlates with
position in session. Later request positions tend to have larger counts. The
trained model encodes position much better than the untrained model, so a
probe might recover cumulative count partly by reading position.

I ran post-hoc position controls. A pure position probe had MLP gap `+0.427`,
so the confound was real. But holding request index fixed at `k=5`, Feature B
still had MLP gap `+0.220`. A residual-after-position control also left
substantial signal. So position explained part of the result, but not all of
it.

This is one of the useful methodological residues of the project: if a probe
target correlates with token position, I now think position controls should be
default, preferably pre-registered.

## The July 4 follow-ups

At this point a fair criticism of carry-through was:

> You have only shown first-slot copying.

Maze starting cell is the first path token. HTTP Feature A is the first
request's size bin. Maybe "architectural carry-through" just means "the model
remembers the first token."

So I locked three smaller HTTP follow-ups before running them. All use the
already-trained HTTP model. All probe at the current request's size token,
`sz_k`.

First, same-request path:

`[m_k, p_k, s_k, sz_k]`

Can the residual at `sz_k` recover `p_k`, the path category two tokens
earlier? This is the easiest non-first-slot case. It is a local
repeated-layout carry-through test.

Result: MLP gap `+0.809`.

The intervention result was also strong locally: corrupting `p_k` caused
`+4.03` natural dNLL on predicting `sz_k`, but only `+0.083` after `sz_k`.

Second, previous-request path:

Can the residual at `sz_k` recover `p_{k-1}`? This is not a first token and
not a fixed absolute position. It is a fixed relative offset across request
records.

Result: MLP gap `+0.674`.

Intervention was weaker: `+0.187` dNLL on `sz_k`, and `+0.024` after `sz_k`.

Third, the more interesting one: recent-large-response path.

For each current request `k`, find the most recent earlier request `j < k`
such that `size_bin_j >= 5`. Then ask whether the residual at `sz_k` can
recover `p_j`, the path category of that earlier large response.

This is not first-slot copying. It is not fixed-relative-offset copying. The
source event is selected by content: "most recent earlier request whose
response was large."

Result: MLP gap `+0.621` for all lags, and `+0.514` when requiring lag
`>= 3`.

The ordering matched the pre-registered qualitative prediction:

`same-request > previous-request > recent-large-response`

But the intervention was cautious. For lag `>= 3`, corrupting the selected
earlier path caused only `+0.013` natural dNLL on `sz_k`, and `+0.003` after
`sz_k`.

So I do not want to call this a semantic memory circuit. The honest version is
weaker and, I think, more interesting: the residual stream can preserve a
strongly decodable trace of a content-selected earlier event, even when the
model only weakly uses that trace for immediate next-token prediction.

One follow-up went the other way. After writing the HTTP ladder, I also tested
a maze-side version of the same criticism: instead of asking whether the model
remembers the first path cell `c0`, ask whether it remembers `c3`, the fourth
path cell, from later positions. This did not confirm. The primary probe was
only ambiguous, with best MLP gap `+0.053`, and a stricter late-only subset was
null, with best MLP gap `+0.032`. So the maze result should stay narrow:
starting-cell persistence is real in this setup, but I do not have evidence
that arbitrary early maze path cells are carried forward. The stronger
non-first-slot evidence is the HTTP ladder, especially the recent-large-response
path result.

## What this does not show

There are several claims I am not making.

First, I am not claiming that these small models have human-like world models,
or that the results transfer to frontier language models. The models here are
tiny, and the domains are deliberately narrow. The point of the setup is that
the features are clean enough to make predictions about, not that the domains
are realistic.

Second, I am not claiming that a probe result proves causal use. The July 4
follow-ups are useful partly because they separate those ideas. Same-request
path is both strongly recoverable and strongly useful for predicting the
current size token. Recent-large-response path is strongly recoverable but
only weakly useful under the lag-filtered intervention. Those should not be
collapsed into the same statement.

Third, I am not claiming that carry-through is the only source of residual
information. The HTTP cumulative-count result is a warning in the other
direction: simple aggregations and positional proxies can also create probe
signals. If I were starting this project again, I would pre-register
position-matched controls much earlier.

The narrow claim is this: when a fact is already present in the token stream
and the architecture gives attention a simple route to preserve it, I should
not expect next-token training to erase it just because it is no longer
immediately useful.

## Where I currently land

The N-criterion, as a strong theory, is dead for me.

Predictive relevance is not enough to predict what is represented. Maze
distance was predicted useful and was not cleanly represented. Maze starting
cell was predicted irrelevant and was represented. HTTP cumulative count was
predicted null and was represented even after position controls.

What survived is narrower:

1. **Architectural carry-through seems real in these small models.**
   Information present at regular earlier slots can remain decodable later.
   The July 4 HTTP follow-ups make this less trivially "first-token copying,"
   especially the recent-large-response case.

2. **Probe recoverability and causal use come apart.** The recent-large-response
   feature is strongly recoverable but weakly causal under the lag-filtered
   intervention. I should have been more careful about this distinction
   earlier.

3. **Pre-registration helped mostly because it made failure legible.** Without
   the locked prediction files, I could easily have told a smoother story after
   the fact. The git commits made that harder.

4. **Position controls matter.** The HTTP cumulative-count result would have
   been overstated without them. The qualitative verdict remained, but the
   interpretation changed.

The residual stream is not a minimal database of just what the model needs
next. It is more like a historically contaminated workspace. Some contents are
there because they help prediction. Some are there because attention made them
cheap to preserve. Some may be probe artifacts unless you control carefully.

## What I want pushback on

I especially want criticism on three points.

First: is "architectural carry-through" too trivial? Maybe it is just the
residual stream being a copy bus. My current view is that this is partly true,
but still worth naming because it breaks a tempting interpretation of probe
results: recoverable does not imply selected-for-use.

Second: is the recent-large-response follow-up actually content-selected
memory, or is it still mostly session-phase, lag, or path-distribution
structure? I do not think the current result fully resolves that. A
matched-lag or matched-index adversarial control would be a good next test.

Third: is git pre-registration enough to be useful? It does not remove
researcher degrees of freedom. I chose which predictions to write down. I
chose thresholds. I ran post-hoc diagnostics. But it did force me to record
the theory before seeing the result, and in this project that changed the
scientific meaning of the failures.

I am much less confident in my original theory than when I started. But I am
more confident that pre-registered small-model interpretability experiments
are worth doing. A failed prediction with an audit trail taught me more than
several clean-looking post-hoc confirmations.
