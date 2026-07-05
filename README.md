# LatentWorldsGPT

Pre-registered study of emergent representations in small next-token
transformers. Seven domains. One ex-ante-validated mechanism (architectural
carry-through). One identified probe confound (position-correlation) with
two controllable diagnostics. July 4 follow-ups show the HTTP carry-through
story is not limited to first slots: same-request path, previous-request path,
and the path of the most recent earlier large response are all recoverable at
the current request's size token.

The paper is at [`docs/paper/paper_draft.md`](docs/paper/paper_draft.md).
The longer working draft is at [`docs/paper/report.md`](docs/paper/report.md).
Visual companions are in [`figures/`](figures/). The reorganized document
index is at [`docs/README.md`](docs/README.md).

## What this is in one paragraph

We study whether and when small next-token transformers (0.27M to 13M
parameters) develop linearly recoverable representations of latent task
structure. We propose a candidate hypothesis called the N-criterion
(encoding driven by predictive relevance for next-token prediction) and
test it across seven domains. The maze and HTTP domains are tested via
quantitative predictions committed to this repository before any data
was collected or any model was trained, with the commit hashes
verifiable from `git log --diff-filter=A` on the predictions files. Both
pre-registered experiments falsify the strict form of the N-criterion.
One mechanism survives ex-ante testing on both domains (architectural
carry-through of features at positionally distinct input slots), and
three locked HTTP follow-ups show the same mechanism is not limited to
first-slot copying. One methodological failure mode is identified
(position-correlation as a probe confound) with two controllable
diagnostics that future probe-based work should adopt by default.

## Seven domains

| # | Domain | Verdict |
|:--|---|---|
| 1 | Othello | Positive control reproduced (94% per-cell probe; within 0.01 of Li et al. 2022) |
| 2 | Music | Voice-leading encoded; chord weakly encoded; beat-in-measure null on both probe and transplant |
| 3 | Cities | Geographic structure recovered but driven primarily by embedding-table co-occurrence, not transformer computation |
| 4 | Flight phases (ADS-B) | Moderate signal; monotone destroyed-structure gradient |
| 5 | Symmetric-group walks | Partial signal; useful as a data point, not a clean positive control |
| 6 | Maze navigation (pre-registered, commit `aa025b1`) | Starting cell NULL **falsified** (+0.15 gap, threshold +0.10) |
| 7 | HTTP log sequences (pre-registered, commit `3b25ed3`) | Feature A input-slot persistence confirmed (+0.17); Feature B computed null **falsified** (+0.22 even after position control) |

## The protocol

1. Multi-seed mean ± std reporting (5 seeds, each varying untrained
   init, sampling positions, and probe-training RNG)
2. Probe and activation-patching convergence at the same layer
3. Per-layer ablation
4. Destroyed-structure corpus controls (real, within-shuffled,
   global-shuffled)
5. Pre-registered ex-ante predictions with a git audit trail
6. Position-controlled probing (Design A: fixed k; Design B3:
   residual-after-position regression)

Items 5 and 6 are the methodological contributions of the paper.

## Pre-registration audit trail

Two pre-registered domains, both with locked predictions before any
model was trained or any data was collected:

```bash
git log --diff-filter=A predictions/predictions_maze_navigation.md
# returns commit aa025b1 (2026-05-27)

git log --diff-filter=A predictions/predictions_http_log_sequences.md
# returns commit 3b25ed3 (2026-05-31)
```

The predictions files are append-only after commit. Post-hoc amendments
appear as clearly-marked additions.

## Figures

![Per-layer ablation](figures/03_per_layer_ablation.png)
*Figure 3: Per-layer ablation across six (domain, feature) pairs.
Othello and music shown via causal transplant lift; maze starting cell
and HTTP Features A and B shown via probe gap. The bottom-right panel
is the HTTP Feature B probe at fixed position k=5 (Design A).*

![Cross-condition gradient](figures/04_cross_condition_gradient.png)
*Figure 4: Real / within-shuffled / global-shuffled gap bars across six
(domain, feature) pairs. The note below each panel indicates where the
destroyed-structure monotonicity prediction holds and where it does
not.*

Two more visual companions:

- [`figures/01_experimental_arc.md`](figures/01_experimental_arc.md):
  Mermaid flowchart of the pre-register, falsify, revise, re-test arc
  with commit hashes inline.
- [`figures/02_results_matrix.md`](figures/02_results_matrix.md):
  color-coded cross-domain outcome matrix.

## Quickstart

```bash
pip install -r requirements.txt

# Pick a domain. Each domain has a prepare script and a model config.
python -u data/prepare_city.py     --place "City of London, ..." --out_dir data/london_city
python -u data/prepare_maze.py     --n_mazes 100000 --grid 8 --out_dir data/maze_8x8
python -u data/prepare_http.py     --raw_dir data/nasa_http_raw --out_dir data/nasa_http
python -u data/prepare_othello.py  --out_dir data/othello_50k
python -u data/prepare_music.py    --out_dir data/music_bach
python -u data/prepare_adsb.py     --out_dir data/flight_quickstart
python -u data/prepare_symgroup.py --out_dir data/symgroup_s5

# Destroyed-structure controls (per domain)
python -u data/prepare_city.py --place "City of London, ..." --shuffle_routes \
  --out_dir data/london_shuffled
python -u data/prepare_city.py --place "City of London, ..." --shuffle_globally \
  --out_dir data/london_global_shuffled

# Train a small model
python model/train.py --config model/configs/small.py --data_dir data/london_city

# Standard eval suite
python eval/probe.py      --ckpt checkpoints/best.pt --data_dir data/london_city
python eval/transplant.py --ckpt checkpoints/best.pt --data_dir data/london_city
python eval/valid_edge.py --ckpt checkpoints/best.pt --data_dir data/london_city

# Position-controlled probes (HTTP example)
python eval/probe_http_position_control.py --ckpt CKPT --data_dir data/nasa_http
python eval/probe_http_within_position.py  --ckpt CKPT --data_dir data/nasa_http --fixed_k 5
python eval/probe_http_residual.py         --ckpt CKPT --data_dir data/nasa_http
```

## Probe-Target Rules

The project uses two kinds of probe targets. Keeping them separate is
important for interpreting the results.

1. **Withheld latent targets.** Coordinates, phases, beat positions,
   chord labels, cumulative counts, and similar labels live only in
   side tables read by probe/eval code. They are never given to the
   model as labels or metadata. Data preparation scripts assert this
   at the boundary where it is cheap.
2. **Input-slot memory targets.** Some pre-registered tests ask whether
   information that is already present in the token stream remains
   recoverable later. Examples: the maze starting cell (the first
   path token) and HTTP Feature A (the first request's `size_bin`
   token). These are not hidden labels; they test input-slot
   persistence / carry-through.

The general rule is narrower than "no target value ever appears in
input": no *withheld latent* target value may appear in the model's
input. Input-slot memory targets are explicitly labeled as such in the
prediction files.

## Documents

| File | What |
|---|---|
| [`paper_draft.md`](docs/paper/paper_draft.md) | Tight version of the paper (about 12 pages excluding references) |
| [`report.md`](docs/paper/report.md) | Longer working draft in original prose |
| [`predictions/`](predictions/) | Locked ex-ante predictions for maze and HTTP; template and example |
| [`results_maze_navigation.md`](docs/results/results_maze_navigation.md) | Confirm/falsify result tables for the maze experiment |
| [`results_http_log_sequences.md`](docs/results/results_http_log_sequences.md) | Confirm/falsify result tables for the HTTP experiment |
| [`results_maze_c3_carrythrough.md`](docs/results/results_maze_c3_carrythrough.md) | C3 follow-up result showing the maze non-first-slot prediction did not confirm |
| [`update_july4.md`](docs/history/update_july4.md) | July 4 HTTP carry-through follow-up log and LessWrong story |
| [`docs/lesswrong/`](docs/lesswrong/) | LessWrong-oriented drafts and source material |
| [`REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Canonical commands and expected artifacts for the pre-registered maze/HTTP runs |
| [`docs/AUDIT_TRAIL.md`](docs/AUDIT_TRAIL.md) | Commit-level audit trail for locked predictions and post-hoc follow-ups |
| [`figures/`](figures/) | Four visual companions; see Figures section above |
| [`CLAUDE.md`](CLAUDE.md) | Operational guide for development; per-domain conventions and commands |
| [`CONTEXT.md`](docs/planning/CONTEXT.md) | Scientific framing of the project |
| [`PLAN.md`](docs/planning/PLAN.md) | Phased build plan; status by phase |

## Cities sub-project context

The project began as a single replication on real-city routes
("a small GPT trained only on intersection IDs recovers a metric map of
the city from its activations"). The destroyed-structure control surfaced
something unexpected: within-shuffled London scored higher on the
geographic probe than real London. The cities domain became the worked
example of decomposition (real, within-shuffled, and global-shuffled
each producing a distinct internal structure) and the project pivoted
to a multi-domain comparative study. The cities case is §5.3 of the
paper, intentionally downplayed because most of the cities encoding
lives in the embedding table rather than in the transformer's
computation.

History is preserved in [`docs/planning/pivot.md`](docs/planning/pivot.md)
and [`docs/history/update_may24_final.md`](docs/history/update_may24_final.md). The
cities-specific THE ONE RULE (no coordinate, distance, or direction in
input tokens) generalizes per-domain to "no probe-target value may
appear in the model's input."

## Not to be confused with

There is a separate, unrelated CityGPT (Feng et al., KDD 2025,
[arXiv:2406.13948](https://arxiv.org/abs/2406.13948)). That work
fine-tunes large LLMs on text instructions to improve performance on
urban benchmarks; it *gives* the model spatial knowledge in language
and measures task scores. The cities sub-project of LatentWorldsGPT is
the inverse: it *withholds* all spatial information and trains a small
model from scratch on bare intersection IDs, then asks whether (and
what) spatial structure emerges unsupervised. Different question,
different method, different evidence.

## References

- Li, K., et al. (2022). *Emergent World Representations: Exploring a
  Sequence Model Trained on a Synthetic Task* (Othello-GPT).
- Nanda, N. (2023). *Actually, Othello-GPT Has a Linear Emergent World
  Representation*.
- Karvonen, A. (2024). *Emergent World Models and Latent Variable
  Estimation in Chess-Playing Language Models*.
- Elhage, N., et al. (2022). *Toy Models of Superposition*.
- Karpathy, A. *nanoGPT*: the model and training scaffold.
- Hewitt, J. and Liang, P. (2019). *Designing and Interpreting Probes
  with Control Tasks*.
- Park, K., et al. (2023). *The Linear Representation Hypothesis and
  the Geometry of Large Language Models*.
