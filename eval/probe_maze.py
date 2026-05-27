"""
LatentWorldsGPT — Phase 4 maze navigation probe.

Tests the four probe targets locked in
`predictions/predictions_maze_navigation.md`:

  1. current cell row (8 classes; chance 0.125)              → encoded
  2. current cell col (8 classes; chance 0.125)              → encoded
  3. Manhattan distance to goal (15 classes; chance ~0.067)  → encoded
  4. starting cell id (64 classes; chance 0.0156)            → NULL

Multi-seed protocol: outermost loop varies untrained init, activation
sampling, and probe-training RNG together (matches probe_cities_grid.py
and probe_music.py).

Two splits:
  - position-level: random partition of probe positions (weak baseline)
  - maze-level: partition mazes into disjoint sets — held-out mazes the
    model never saw during training. THE HONEST TEST.

Usage:
    python eval/probe_maze.py \\
        --ckpt checkpoints/maze_8x8/best.pt \\
        --data_dir data/maze_8x8 \\
        --seeds 0 1 2 3 4
"""
import argparse
import csv
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "model"))

from model import GPT, GPTConfig  # noqa: E402
from probe import cache_layer_activations  # noqa: E402

PAD, BOS, EOS = 0, 1, 2

TARGET_NAMES = ("row", "col", "distance", "start_cell")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load probe targets from mazes.csv
# ─────────────────────────────────────────────────────────────────────────────

def load_targets(data_dir: Path, splits=("val", "gen")):
    """Returns:
       targets[target_name][(split, token_pos)] = label_int
       maze_idx_of[(split, token_pos)] = maze_idx
    """
    targets = {t: {} for t in TARGET_NAMES}
    maze_idx_of = {}
    with open(data_dir / "mazes.csv") as f:
        for row in csv.DictReader(f):
            if row["split"] not in splits:
                continue
            key = (row["split"], int(row["token_pos"]))
            targets["row"][key] = int(row["row"])
            targets["col"][key] = int(row["col"])
            targets["distance"][key] = int(row["distance_to_goal"])
            targets["start_cell"][key] = int(row["start_cell"])
            maze_idx_of[key] = int(row["maze_idx"])
    return targets, maze_idx_of


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build probe dataset
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_probe_dataset(model, streams, targets, maze_idx_of,
                        block_size, n_positions, device, rng_seed=0):
    """Sample positions, cache activations across all layers, return
    X (n_layers+1, n_positions, n_embd), y_dict (target → labels),
    maze_idx (n_positions,).
    """
    model.eval()
    rng = np.random.default_rng(rng_seed)

    # Stitch streams into one array; remember offsets to map back to (split, pos)
    split_order = list(streams.keys())
    offsets, parts, cursor = {}, [], 0
    for s in split_order:
        offsets[s] = cursor
        parts.append(streams[s])
        cursor += len(streams[s])
    combined = np.concatenate(parts).astype(np.int64)

    def to_split_pos(gp):
        for s in reversed(split_order):
            if gp >= offsets[s]:
                return s, gp - offsets[s]
        return None, None

    # Sample positions with a valid maze_idx label
    n_layer = model.config.n_layer
    pos_indices = []
    attempts = 0
    while len(pos_indices) < n_positions and attempts < n_positions * 20:
        gp = int(rng.integers(0, len(combined) - block_size - 1))
        split, pos = to_split_pos(gp)
        if split is None:
            attempts += 1
            continue
        key = (split, pos)
        if key not in maze_idx_of:
            attempts += 1
            continue
        pos_indices.append(gp)
        attempts += 1

    pos_indices = np.array(pos_indices[:n_positions], dtype=np.int64)

    # For each sampled position, build the context (block_size tokens up to position)
    X_layers = [[] for _ in range(n_layer + 1)]  # embed + n_layer
    y_dict = {t: [] for t in TARGET_NAMES}
    maze_indices = []
    batch_size = 16
    for batch_start in range(0, len(pos_indices), batch_size):
        batch_gps = pos_indices[batch_start:batch_start + batch_size]
        batch_idxs = []
        valid_in_batch = []
        for gp in batch_gps:
            split, pos = to_split_pos(int(gp))
            ctx_start = max(0, gp - block_size + 1)
            ctx = combined[ctx_start:gp + 1]
            if len(ctx) < block_size:
                ctx = np.concatenate([np.full(block_size - len(ctx), PAD), ctx])
            batch_idxs.append(ctx)
            valid_in_batch.append((gp, split, pos))
        idx_tensor = torch.from_numpy(np.stack(batch_idxs)).to(device)
        # Cache activations at the last position of each sequence
        with torch.no_grad():
            layer_acts = cache_layer_activations(model, idx_tensor)
        # layer_acts: list of (B, T, n_embd); take the last position
        for L, act in enumerate(layer_acts):
            X_layers[L].append(act[:, -1, :].cpu().numpy())
        for gp, split, pos in valid_in_batch:
            key = (split, pos)
            for t in TARGET_NAMES:
                y_dict[t].append(targets[t][key])
            maze_indices.append(maze_idx_of[key])

    X = np.stack([np.concatenate(X_layers[L], axis=0) for L in range(n_layer + 1)], axis=0)
    y_dict = {t: np.array(y_dict[t], dtype=np.int64) for t in TARGET_NAMES}
    maze_indices = np.array(maze_indices, dtype=np.int64)
    return X, y_dict, maze_indices


# ─────────────────────────────────────────────────────────────────────────────
# 3. Probes
# ─────────────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.lin = nn.Linear(in_dim, n_classes)
    def forward(self, x):
        return self.lin(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x):
        return self.net(x)


def train_eval(probe, Xtr, ytr, Xte, yte, device, epochs=30, lr=3e-3):
    probe = probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    Xtr_t, ytr_t = torch.from_numpy(Xtr).float().to(device), torch.from_numpy(ytr).long().to(device)
    Xte_t, yte_t = torch.from_numpy(Xte).float().to(device), torch.from_numpy(yte).long().to(device)
    n = len(Xtr_t)
    batch = min(256, n)
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            ix = idx[s:s + batch]
            logits = probe(Xtr_t[ix])
            loss = F.cross_entropy(logits, ytr_t[ix])
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        preds = probe(Xte_t).argmax(dim=-1)
        acc = (preds == yte_t).float().mean().item()
    return acc


def position_split(n, train_frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ntr = int(train_frac * n)
    return perm[:ntr], perm[ntr:]


def maze_level_split(maze_indices, train_frac, seed):
    """Split by maze_idx — all positions from a given maze go to one side."""
    unique = np.unique(maze_indices)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique)
    n_train = int(train_frac * len(unique))
    train_mazes = set(perm[:n_train].tolist())
    train_ix = np.array([i for i, m in enumerate(maze_indices) if m in train_mazes])
    test_ix = np.array([i for i, m in enumerate(maze_indices) if m not in train_mazes])
    return train_ix, test_ix


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-seed layer sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_seed(model, target_name, X_layers, y, n_classes, train_ix, test_ix,
             device, epochs, label):
    """For one (target, split, seed), run probe at each layer; return list of
    (lin_acc, mlp_acc)."""
    if len(train_ix) == 0 or len(test_ix) == 0:
        return []
    rows = []
    n_layer = X_layers.shape[0]
    print(f"\n  {label}  [{target_name}]  n_classes={n_classes}")
    print(f"    Layer       Lin      MLP")
    for L in range(n_layer):
        Xtr, ytr = X_layers[L, train_ix], y[train_ix]
        Xte, yte = X_layers[L, test_ix], y[test_ix]
        in_dim = Xtr.shape[1]
        lin = LinearProbe(in_dim, n_classes)
        mlp = MLPProbe(in_dim, n_classes)
        lin_acc = train_eval(lin, Xtr, ytr, Xte, yte, device, epochs=epochs)
        mlp_acc = train_eval(mlp, Xtr, ytr, Xte, yte, device, epochs=epochs)
        lab = "embed" if L == 0 else f"L{L - 1}"
        print(f"    {lab:<8}  {lin_acc:.4f}   {mlp_acc:.4f}")
        rows.append((lin_acc, mlp_acc))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--n_positions", type=int, default=10_000)
    p.add_argument("--probe_train_frac", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--targets", nargs="+", default=list(TARGET_NAMES))
    p.add_argument("--skip_untrained", action="store_true")
    p.add_argument("--skip_maze_split", action="store_true")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    print(f"running {len(args.seeds)} seed(s): {args.seeds}")
    print(f"targets: {args.targets}")

    data_dir = Path(args.data_dir)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    trained = GPT(config).to(device); trained.load_state_dict(ckpt["model_state"])
    trained.eval()
    print(f"loaded ckpt: iter={ckpt.get('iter', '?')}  val_ppl={ckpt.get('val_perplexity', float('nan')):.4f}")

    print(f"\nLoading probe targets ...")
    targets, maze_idx_of = load_targets(data_dir, splits=("val", "gen"))
    n_mazes_per = {s: len({m for (s2, _), m in maze_idx_of.items() if s2 == s})
                    for s in ("val", "gen")}
    print(f"  labeled positions: {len(maze_idx_of):,}  (val mazes: {n_mazes_per['val']}, gen mazes: {n_mazes_per['gen']})")

    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dtype = np.dtype(meta["dtype"])
    streams = {s: np.asarray(np.memmap(data_dir / f"{s}.bin", dtype=dtype, mode="r"))
               for s in ("val", "gen")}

    # all_results[target][cond][L] = list of (lin_acc, mlp_acc) per-seed
    all_results = {
        t: {
            "trained_pos": defaultdict(list),
            "trained_maze": defaultdict(list),
            "untrained_pos": defaultdict(list),
            "untrained_maze": defaultdict(list),
        }
        for t in args.targets
    }

    for seed in args.seeds:
        print(f"\n{'#' * 78}\n# SEED {seed}\n{'#' * 78}")
        torch.manual_seed(seed); np.random.seed(seed)

        untrained = None if args.skip_untrained else GPT(config).to(device).eval()

        print(f"\nBuilding probe dataset for TRAINED (seed={seed}) ...")
        t0 = time.time()
        X_t, y_t_dict, mazes_t = build_probe_dataset(
            trained, streams, targets, maze_idx_of,
            config.block_size, args.n_positions, device, rng_seed=seed)
        print(f"  collected {X_t.shape[1]:,} positions × {X_t.shape[0]} layers ({time.time() - t0:.1f}s)")

        if untrained is not None:
            print(f"\nBuilding probe dataset for UNTRAINED (seed={seed}) ...")
            t0 = time.time()
            X_u, y_u_dict, mazes_u = build_probe_dataset(
                untrained, streams, targets, maze_idx_of,
                config.block_size, args.n_positions, device, rng_seed=seed)
            print(f"  collected {X_u.shape[1]:,} positions × {X_u.shape[0]} layers ({time.time() - t0:.1f}s)")

        for tgt in args.targets:
            y_t = y_t_dict[tgt]
            if untrained is not None:
                y_u = y_u_dict[tgt]
            n_classes = int(max(y_t.max(), y_u.max() if untrained is not None else 0)) + 1
            pos_train_t, pos_test_t = position_split(len(y_t), args.probe_train_frac, seed)
            maze_train_t, maze_test_t = ((np.array([]), np.array([])) if args.skip_maze_split
                                          else maze_level_split(mazes_t, args.probe_train_frac, seed))

            rows = run_seed(trained, tgt, X_t, y_t, n_classes,
                            pos_train_t, pos_test_t, device, args.epochs,
                            f"TRAINED POSITION-LEVEL (seed {seed})")
            for L, (lin, mlp) in enumerate(rows):
                all_results[tgt]["trained_pos"][L].append((lin, mlp))

            if not args.skip_maze_split and len(maze_train_t) > 0:
                rows = run_seed(trained, tgt, X_t, y_t, n_classes,
                                maze_train_t, maze_test_t, device, args.epochs,
                                f"TRAINED MAZE-LEVEL (seed {seed})")
                for L, (lin, mlp) in enumerate(rows):
                    all_results[tgt]["trained_maze"][L].append((lin, mlp))

            if untrained is not None:
                pos_train_u, pos_test_u = position_split(len(y_u), args.probe_train_frac, seed)
                rows = run_seed(untrained, tgt, X_u, y_u, n_classes,
                                pos_train_u, pos_test_u, device, args.epochs,
                                f"UNTRAINED POSITION-LEVEL (seed {seed})")
                for L, (lin, mlp) in enumerate(rows):
                    all_results[tgt]["untrained_pos"][L].append((lin, mlp))
                if not args.skip_maze_split:
                    maze_train_u, maze_test_u = maze_level_split(mazes_u, args.probe_train_frac, seed)
                    rows = run_seed(untrained, tgt, X_u, y_u, n_classes,
                                    maze_train_u, maze_test_u, device, args.epochs,
                                    f"UNTRAINED MAZE-LEVEL (seed {seed})")
                    for L, (lin, mlp) in enumerate(rows):
                        all_results[tgt]["untrained_maze"][L].append((lin, mlp))

    # ─────────────────────────────────────────────────────────────────
    # Aggregate + HEADLINE
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 78}\nAGGREGATE — mean ± std over {len(args.seeds)} seeds\n{'=' * 78}")
    for tgt in args.targets:
        print(f"\n  [{tgt}]")
        for cond in ("trained_pos", "trained_maze", "untrained_pos", "untrained_maze"):
            results = all_results[tgt][cond]
            if not results:
                continue
            print(f"\n  {cond}")
            print(f"    Layer       LinAcc(mean±std)     MLPAcc(mean±std)")
            for L in sorted(results.keys()):
                lins = np.array([r[0] for r in results[L]])
                mlps = np.array([r[1] for r in results[L]])
                lab = "embed" if L == 0 else f"L{L - 1}"
                lin_s = f"{lins.std(ddof=1):.4f}" if len(lins) > 1 else "0.0000"
                mlp_s = f"{mlps.std(ddof=1):.4f}" if len(mlps) > 1 else "0.0000"
                print(f"    {lab:<8}    {lins.mean():.4f} ± {lin_s}      {mlps.mean():.4f} ± {mlp_s}")

    print(f"\n{'=' * 78}\nHEADLINE — best layer by mean over {len(args.seeds)} seed(s)\n{'=' * 78}")
    for tgt in args.targets:
        print(f"\n  [{tgt}]")
        for cond in ("trained_pos", "trained_maze", "untrained_pos", "untrained_maze"):
            results = all_results[tgt][cond]
            if not results:
                continue
            lin_means = {L: np.mean([r[0] for r in results[L]]) for L in results}
            mlp_means = {L: np.mean([r[1] for r in results[L]]) for L in results}
            best_lin_L = max(lin_means, key=lin_means.get)
            best_mlp_L = max(mlp_means, key=mlp_means.get)
            lin_s = np.std([r[0] for r in results[best_lin_L]], ddof=1) if len(results[best_lin_L]) > 1 else 0
            mlp_s = np.std([r[1] for r in results[best_mlp_L]], ddof=1) if len(results[best_mlp_L]) > 1 else 0
            lab_lin = "embed" if best_lin_L == 0 else f"L{best_lin_L - 1}"
            lab_mlp = "embed" if best_mlp_L == 0 else f"L{best_mlp_L - 1}"
            print(f"    {cond:<22} linear best {lab_lin}: {lin_means[best_lin_L]:.4f} ± {lin_s:.4f}")
            print(f"    {cond:<22} MLP    best {lab_mlp}: {mlp_means[best_mlp_L]:.4f} ± {mlp_s:.4f}")


if __name__ == "__main__":
    main()
