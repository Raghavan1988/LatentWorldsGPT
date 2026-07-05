# Reproducibility

This file is the canonical entry point for reproducing the two
pre-registered experiments in this repository: maze navigation and HTTP log
sequences.

The exact headline numbers in the paper were produced with 5-seed probe and
transplant runs. A full rerun may take several hours on Apple MPS or CUDA.
The commands below show the intended pipeline and artifact layout.

## Environment

```bash
pip install -r requirements.txt
```

The training and evaluation scripts choose `cuda`, then `mps`, then `cpu`
when available.

## Audit The Locked Predictions

```bash
git log --diff-filter=A predictions/predictions_maze_navigation.md
# aa025b1 2026-05-27 LOCKDOWN: predictions_maze_navigation.md

git log --diff-filter=A predictions/predictions_http_log_sequences.md
# 3b25ed3 2026-05-31 LOCKDOWN: predictions_http_log_sequences.md
```

The prediction files were committed before the corresponding data/model/probe
runs. See `docs/AUDIT_TRAIL.md` for the compact timeline.

## Maze Navigation

### Data

```bash
python data/prepare_maze.py \
  --n_mazes 100000 \
  --grid 8 \
  --out_dir data/maze_8x8 \
  --seed 0
```

Destroyed-structure variants:

```bash
python data/prepare_maze.py --n_mazes 100000 --grid 8 \
  --out_dir data/maze_8x8_within_shuffled --shuffle_within_path --seed 0

python data/prepare_maze.py --n_mazes 100000 --grid 8 \
  --out_dir data/maze_8x8_global_shuffled --shuffle_globally --seed 0
```

### Train

```bash
python model/train.py \
  --config model/configs/small_maze.py \
  --data_dir data/maze_8x8 \
  --out_dir checkpoints/maze_8x8 \
  --seed 0
```

### Evaluate

```bash
python eval/probe_maze.py \
  --ckpt checkpoints/maze_8x8/best.pt \
  --data_dir data/maze_8x8 \
  --seeds 0 1 2 3 4

python eval/transplant_maze.py \
  --ckpt checkpoints/maze_8x8/best.pt \
  --data_dir data/maze_8x8 \
  --layer 5 \
  --seed 0
```

The headline maze result is the starting-cell probe gap at L5:
trained MLP `0.2024 +/- 0.0166`, untrained MLP `0.0504 +/- 0.0082`,
gap `+0.152`.

## HTTP Log Sequences

The HTTP experiment expects NASA-HTTP raw logs in `data/nasa_http_raw/`.
The project does not vendor the raw logs or generated corpora.

### Data

```bash
python data/prepare_http.py \
  --raw_dir data/nasa_http_raw \
  --out_dir data/nasa_http
```

Destroyed-structure variants use the same script with the relevant shuffle
flags; see `python data/prepare_http.py --help` for the exact options in
the current checkout.

### Train

```bash
python model/train.py \
  --config model/configs/small_http.py \
  --data_dir data/nasa_http \
  --out_dir checkpoints/http_real \
  --seed 0
```

### Evaluate

```bash
python eval/probe_http.py \
  --ckpt checkpoints/http_real/best.pt \
  --data_dir data/nasa_http \
  --seeds 0 1 2 3 4

python eval/probe_http_position_control.py \
  --ckpt checkpoints/http_real/best.pt \
  --data_dir data/nasa_http \
  --seeds 0 1 2 3 4

python eval/probe_http_within_position.py \
  --ckpt checkpoints/http_real/best.pt \
  --data_dir data/nasa_http \
  --fixed_k 5 \
  --seeds 0 1 2 3 4

python eval/probe_http_residual.py \
  --ckpt checkpoints/http_real/best.pt \
  --data_dir data/nasa_http \
  --seeds 0 1 2 3 4
```

The headline HTTP results are:

| Target | Headline |
|---|---:|
| Feature A (`first_request_size_bin`) | MLP gap `+0.168` |
| Feature B (`cumulative_large_response_binned`) | raw MLP gap `+0.291` |
| Feature B at fixed `k=5` | MLP gap `+0.220` |
| Feature B residual-after-position | R2 gap `+0.468` |

### HTTP Carry-Through Follow-Ups

```bash
python eval/probe_http_same_request_path.py \
  --per_class_n 1000 \
  --epochs 8 \
  --seeds 0 1 2 3 4

python eval/intervene_http_path_token.py \
  --natural_n 50000 \
  --per_class_n 1000 \
  --batch_size 512

python eval/probe_http_previous_request_path.py \
  --per_class_n 1000 \
  --epochs 8 \
  --seeds 0 1 2 3 4

python eval/intervene_http_previous_path_token.py \
  --natural_n 50000 \
  --per_class_n 1000 \
  --batch_size 512

python eval/probe_http_recent_large_response_path.py \
  --min_lag 1 \
  --per_class_n 1000 \
  --epochs 8 \
  --seeds 0 1 2 3 4

python eval/probe_http_recent_large_response_path.py \
  --min_lag 3 \
  --per_class_n 1000 \
  --epochs 8 \
  --seeds 0 1 2 3 4

python eval/intervene_http_recent_large_response_path.py \
  --min_lag 1 \
  --natural_n 50000 \
  --per_class_n 1000 \
  --batch_size 512

python eval/intervene_http_recent_large_response_path.py \
  --min_lag 3 \
  --natural_n 50000 \
  --per_class_n 1000 \
  --batch_size 512
```

Headline follow-up results:

| Target | Headline |
|---|---:|
| Same-request path `p_k` at `sz_k` | MLP gap `+0.809` |
| Previous-request path `p_{k-1}` at `sz_k` | MLP gap `+0.674` |
| Recent-large-response path, lag `>= 1` | MLP gap `+0.621` |
| Recent-large-response path, lag `>= 3` | MLP gap `+0.514` |
