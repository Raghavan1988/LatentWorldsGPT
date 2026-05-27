"""
LatentWorldsGPT — cities probe with grid-classification reframing.

The original `eval/probe.py` does REGRESSION on continuous (x_m, y_m)
coordinates. Continuous regression with ~10³ unique tokens suffers
MLP-lookup-memorization contamination: an untrained model's MLP probe
can also achieve high R² by memorizing the (token → coord) mapping.
The "linear ≈ MLP → linearly encoded" criterion from Nanda 2023
cannot be cleanly applied.

This script reframes the same probe target as CLASSIFICATION:
  - Discretize (lat, lon) into a G x G spatial grid (default G=10 = 100 cells)
  - Each token (= each city node) belongs to exactly one cell
  - Train a classification probe (100-class) on activations → cell label

Classification probes don't suffer the same memorization vulnerability
because a linear classifier has to project the residual onto specific
one-hot directions for each class. Lookup memorization helps less.

The result is directly comparable to Othello's 64-cell board-state
probe (`eval/probe_othello.py`).

THE ONE RULE
============
This script reads coords.csv (the probe-side ground truth) — same as
`eval/probe.py`. The grid bucketing is computed FROM coords; coords
never enter the model.

Usage:
    python eval/probe_cities_grid.py --ckpt checkpoints/best.pt \
        --data_dir data/london_city --grid_size 10
"""

import argparse
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "model"))

from model import GPT, GPTConfig  # noqa: E402
from probe import cache_layer_activations, load_coords_planar, N_RESERVED  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# 1. Build grid label per token
# ─────────────────────────────────────────────────────────────────────────────

def build_grid_labels(coords_xy: torch.Tensor, grid_size: int):
    """For each token id (row in coords_xy), compute its grid-cell label
    (0..grid_size**2 - 1). Returns a long tensor (max_token_id + 1,) with
    -1 at control-token rows.
    """
    valid_mask = ~torch.isnan(coords_xy[:, 0])
    valid_xy = coords_xy[valid_mask]
    if len(valid_xy) == 0:
        raise ValueError("No valid coords")
    x_min, x_max = valid_xy[:, 0].min().item(), valid_xy[:, 0].max().item()
    y_min, y_max = valid_xy[:, 1].min().item(), valid_xy[:, 1].max().item()

    # Compute bin edges with a tiny pad so the max value lands in the last bin
    x_edges = np.linspace(x_min, x_max + 1e-3, grid_size + 1)
    y_edges = np.linspace(y_min, y_max + 1e-3, grid_size + 1)

    labels = torch.full((coords_xy.shape[0],), -1, dtype=torch.long)
    for i in range(coords_xy.shape[0]):
        if torch.isnan(coords_xy[i, 0]):
            continue
        x, y = coords_xy[i, 0].item(), coords_xy[i, 1].item()
        xi = int(np.searchsorted(x_edges, x, side="right") - 1)
        yi = int(np.searchsorted(y_edges, y, side="right") - 1)
        xi = max(0, min(grid_size - 1, xi))
        yi = max(0, min(grid_size - 1, yi))
        labels[i] = yi * grid_size + xi
    return labels, (x_edges, y_edges)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build probe dataset
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_probe_dataset(model: GPT, stream: np.ndarray,
                        grid_labels: torch.Tensor, block_size: int,
                        n_positions: int, device: str, rng_seed: int = 0):
    """Sample positions from `stream`; cache activations per layer; pair with
    each position's current-token grid label."""
    model.eval()
    rng = np.random.default_rng(rng_seed)
    batch_size = 32
    all_X = None
    all_y, all_tokens = [], []
    n_collected = 0
    while n_collected < n_positions:
        starts = rng.integers(0, len(stream) - block_size - 1, size=batch_size)
        windows = [np.asarray(stream[s : s + block_size]) for s in starts]
        idx_batch = torch.from_numpy(np.stack(windows).astype(np.int64)).to(device)
        acts = cache_layer_activations(model, idx_batch)
        if all_X is None:
            all_X = [[] for _ in range(len(acts))]
        idx_np = idx_batch.cpu().numpy()
        for b in range(idx_np.shape[0]):
            for t in range(idx_np.shape[1]):
                tok = int(idx_np[b, t])
                if tok < N_RESERVED:
                    continue
                if tok >= grid_labels.shape[0]:
                    continue
                lab = int(grid_labels[tok].item())
                if lab < 0:
                    continue
                for L in range(len(acts)):
                    all_X[L].append(acts[L][b, t].cpu().numpy())
                all_y.append(lab)
                all_tokens.append(tok)
                n_collected += 1
                if n_collected >= n_positions:
                    break
            if n_collected >= n_positions:
                break
    X = [np.stack(layer_list) for layer_list in all_X]
    y = np.array(all_y, dtype=np.int64)
    tokens = np.array(all_tokens, dtype=np.int64)
    return X, y, tokens


# ─────────────────────────────────────────────────────────────────────────────
# 3. Probes
# ─────────────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, in_dim, n):
        super().__init__()
        self.lin = nn.Linear(in_dim, n)
    def forward(self, x):
        return self.lin(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim, n, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n),
        )
    def forward(self, x):
        return self.net(x)


def train_eval(probe, Xtr, ytr, Xte, yte, n_classes, device,
               lr=1e-3, wd=1e-3, epochs=50, batch_size=512):
    probe = probe.to(device)
    Xtr = torch.from_numpy(Xtr).float().to(device)
    ytr = torch.from_numpy(ytr).long().to(device)
    Xte = torch.from_numpy(Xte).float().to(device)
    yte = torch.from_numpy(yte).long().to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    nrows = Xtr.shape[0]
    best = -1.0
    for _ in range(epochs):
        perm = torch.randperm(nrows, device=device)
        for i in range(0, nrows, batch_size):
            ix = perm[i : i + batch_size]
            loss = F.cross_entropy(probe(Xtr[ix]), ytr[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (probe(Xte).argmax(-1) == yte).float().mean().item()
            best = max(best, acc)
    return best


def node_level_split(tokens, train_frac, seed):
    uniq = np.unique(tokens)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_train_nodes = int(len(uniq) * train_frac)
    train_nodes = set(int(t) for t in uniq[perm[:n_train_nodes]])
    train_mask = np.array([int(t) in train_nodes for t in tokens])
    return np.nonzero(train_mask)[0], np.nonzero(~train_mask)[0]


def position_split(n, train_frac, seed):
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(n * train_frac)
    return perm[:n_train], perm[n_train:]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────────────

def run_layer_sweep(X_layers, y, n_classes, train_ix, test_ix, epochs,
                    device, label):
    if len(train_ix) == 0 or len(test_ix) == 0:
        print(f"\n{label}: empty split, skipping"); return []
    print(f"\n{'─'*78}\n{label}\n{'─'*78}")
    print(f"  {'Layer':<8}{'LinAcc':>10}{'MLPAcc':>10}")
    rows = []
    for L, Xl in enumerate(X_layers):
        Xtr, Xte = Xl[train_ix], Xl[test_ix]
        ytr, yte = y[train_ix], y[test_ix]
        a_lin = train_eval(LinearProbe(Xl.shape[1], n_classes),
                           Xtr, ytr, Xte, yte, n_classes, device,
                           lr=1e-3, wd=1e-3, epochs=epochs)
        a_mlp = train_eval(MLPProbe(Xl.shape[1], n_classes),
                           Xtr, ytr, Xte, yte, n_classes, device,
                           lr=1e-3, wd=1e-5, epochs=epochs)
        lab = "embed" if L == 0 else f"L{L}"
        print(f"  {lab:<8}{a_lin:>10.4f}{a_mlp:>10.4f}")
        rows.append((L, a_lin, a_mlp))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--grid_size", type=int, default=10,
                   help="grid side length; total cells = grid_size**2")
    p.add_argument("--n_positions", type=int, default=20_000)
    p.add_argument("--probe_train_frac", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seeds", type=int, default=1,
                   help="number of seeds to run; uses seeds 0..N-1")
    p.add_argument("--seed", type=int, default=None,
                   help="single seed shorthand; if set, overrides --seeds")
    p.add_argument("--skip_untrained", action="store_true")
    p.add_argument("--skip_node_split", action="store_true")
    args = p.parse_args()

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(range(args.seeds))
    print(f"running {len(seeds)} seed(s): {seeds}")

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    data_dir = Path(args.data_dir)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    trained = GPT(config).to(device); trained.load_state_dict(ckpt["model_state"])
    trained.eval()
    print(f"loaded ckpt: iter={ckpt.get('iter','?')}  val_ppl={ckpt.get('val_perplexity',float('nan')):.4f}")

    coords_xy, center_lat, center_lon = load_coords_planar(data_dir)
    n_nodes = (~torch.isnan(coords_xy[:, 0])).sum().item()
    grid_labels, _ = build_grid_labels(coords_xy, args.grid_size)
    n_classes = args.grid_size ** 2
    nonempty_labels = grid_labels[grid_labels >= 0]
    cell_counts = np.bincount(nonempty_labels.numpy(), minlength=n_classes)
    print(f"  {n_nodes} nodes binned into {args.grid_size}x{args.grid_size}={n_classes} cells")
    print(f"  per-cell node-count: min={cell_counts.min()} "
          f"median={int(np.median(cell_counts))} max={cell_counts.max()} "
          f"empty={(cell_counts==0).sum()}")

    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dtype = np.dtype(meta["dtype"])
    val_stream = np.asarray(np.memmap(data_dir / "val.bin", dtype=dtype, mode="r"))
    gen_stream = np.asarray(np.memmap(data_dir / "gen.bin", dtype=dtype, mode="r"))
    combined = np.concatenate([val_stream, gen_stream])
    print(f"  val+gen stream: {len(combined):,} tokens")

    # all_results[condition][L] = list of (lin_acc, mlp_acc) tuples, one per seed
    all_results = {
        "trained_pos": defaultdict(list),
        "trained_node": defaultdict(list),
        "untrained_pos": defaultdict(list),
        "untrained_node": defaultdict(list),
    }
    majority = None

    for seed in seeds:
        print(f"\n{'#'*78}\n# SEED {seed}\n{'#'*78}")
        torch.manual_seed(seed); np.random.seed(seed)

        untrained = None if args.skip_untrained else GPT(config).to(device).eval()

        def build(model, lab):
            print(f"\nBuilding probe dataset for {lab} (seed={seed}) ...")
            t0 = time.time()
            X, y, tokens = build_probe_dataset(
                model, combined, grid_labels, config.block_size,
                args.n_positions, device, rng_seed=seed,
            )
            print(f"  collected {len(y):,} positions ({time.time()-t0:.1f}s)")
            return X, y, tokens

        X_t, y_t, tokens_t = build(trained, "TRAINED")
        if untrained is not None:
            X_u, y_u, tokens_u = build(untrained, "UNTRAINED")

        pos_train_t, pos_test_t = position_split(len(y_t), args.probe_train_frac, seed)
        if untrained is not None:
            pos_train_u, pos_test_u = position_split(len(y_u), args.probe_train_frac, seed)

        if not args.skip_node_split:
            node_train_t, node_test_t = node_level_split(tokens_t, args.probe_train_frac, seed)
            if untrained is not None:
                node_train_u, node_test_u = node_level_split(tokens_u, args.probe_train_frac, seed)

        if majority is None:
            cnt = np.bincount(y_t, minlength=n_classes)
            majority = cnt.max() / len(y_t)
            print(f"\nMajority-class baseline: {majority:.4f}  (chance = 1/{n_classes} = {1/n_classes:.4f})")

        trained_pos = run_layer_sweep(X_t, y_t, n_classes, pos_train_t, pos_test_t,
                                       args.epochs, device,
                                       f"TRAINED — POSITION-LEVEL (seed {seed})")
        for L, lin, mlp in trained_pos:
            all_results["trained_pos"][L].append((lin, mlp))

        if not args.skip_node_split:
            trained_node = run_layer_sweep(X_t, y_t, n_classes, node_train_t, node_test_t,
                                            args.epochs, device,
                                            f"TRAINED — NODE-LEVEL (held-out tokens, seed {seed})")
            for L, lin, mlp in trained_node:
                all_results["trained_node"][L].append((lin, mlp))

        if untrained is not None:
            untrained_pos = run_layer_sweep(X_u, y_u, n_classes, pos_train_u, pos_test_u,
                                             args.epochs, device,
                                             f"UNTRAINED — POSITION-LEVEL (seed {seed})")
            for L, lin, mlp in untrained_pos:
                all_results["untrained_pos"][L].append((lin, mlp))

            if not args.skip_node_split:
                untrained_node = run_layer_sweep(X_u, y_u, n_classes, node_train_u, node_test_u,
                                                  args.epochs, device,
                                                  f"UNTRAINED — NODE-LEVEL (seed {seed})")
                for L, lin, mlp in untrained_node:
                    all_results["untrained_node"][L].append((lin, mlp))

    # ─────────────────────────────────────────────────────────────────────────
    # Aggregate across seeds
    # ─────────────────────────────────────────────────────────────────────────
    n_s = len(seeds)

    def aggregate(layer_dict, label):
        if not layer_dict: return None
        print(f"\n{'='*78}\n{label}  (mean ± std over {n_s} seed(s))\n{'='*78}")
        print(f"  {'Layer':<8}{'LinAcc (mean±std)':>22}{'MLPAcc (mean±std)':>22}")
        rows = []
        for L in sorted(layer_dict.keys()):
            lins = np.array([v[0] for v in layer_dict[L]])
            mlps = np.array([v[1] for v in layer_dict[L]])
            lin_m = lins.mean(); lin_s = lins.std(ddof=1) if len(lins) > 1 else 0.0
            mlp_m = mlps.mean(); mlp_s = mlps.std(ddof=1) if len(mlps) > 1 else 0.0
            lab = "embed" if L == 0 else f"L{L}"
            print(f"  {lab:<8}{lin_m:>10.4f}±{lin_s:.4f}    {mlp_m:>10.4f}±{mlp_s:.4f}")
            rows.append((L, lin_m, lin_s, mlp_m, mlp_s))
        return rows

    agg_tpos  = aggregate(all_results["trained_pos"],   "TRAINED — POSITION-LEVEL")
    agg_tnode = aggregate(all_results["trained_node"],  "TRAINED — NODE-LEVEL (held-out tokens)")
    agg_upos  = aggregate(all_results["untrained_pos"], "UNTRAINED — POSITION-LEVEL")
    agg_unode = aggregate(all_results["untrained_node"], "UNTRAINED — NODE-LEVEL")

    print(f"\n{'═'*78}\nHEADLINE — best layer by MEAN over {n_s} seed(s)\n{'═'*78}")
    def show_best(rows, mean_ix, std_ix, lab):
        if not rows: print(f"  {lab:<32}    —"); return
        b = max(rows, key=lambda r: r[mean_ix])
        L = b[0]; layer = "embed" if L == 0 else f"L{L}"
        m = b[mean_ix]; s = b[std_ix]
        print(f"  {lab:<32}  {layer:>5}   acc={m:.4f}±{s:.4f}")

    print("\n  POSITION-LEVEL:")
    show_best(agg_tpos, 1, 2, "trained  linear")
    show_best(agg_tpos, 3, 4, "trained  MLP")
    if agg_upos:
        show_best(agg_upos, 1, 2, "untrained linear")
        show_best(agg_upos, 3, 4, "untrained MLP")
    if agg_tnode:
        print("\n  NODE-LEVEL (held-out tokens):")
        show_best(agg_tnode, 1, 2, "trained  linear")
        show_best(agg_tnode, 3, 4, "trained  MLP")
        if agg_unode:
            show_best(agg_unode, 1, 2, "untrained linear")
            show_best(agg_unode, 3, 4, "untrained MLP")

    if n_s > 1:
        print(f"\n{'═'*78}\nPER-SEED MAX-LAYER INFLATION CHECK\n{'═'*78}")
        print("(Single-seed reporting maximizes across layers; this shows what that gives.)")
        def per_seed_max(layer_dict, lab):
            if not layer_dict: return
            layers = sorted(layer_dict.keys())
            seed_lin_maxes, seed_mlp_maxes = [], []
            for s_i in range(n_s):
                lins = [layer_dict[L][s_i][0] for L in layers]
                mlps = [layer_dict[L][s_i][1] for L in layers]
                seed_lin_maxes.append(max(lins))
                seed_mlp_maxes.append(max(mlps))
            lin_m = np.mean(seed_lin_maxes); lin_s = np.std(seed_lin_maxes, ddof=1) if n_s > 1 else 0.0
            mlp_m = np.mean(seed_mlp_maxes); mlp_s = np.std(seed_mlp_maxes, ddof=1) if n_s > 1 else 0.0
            print(f"  {lab:<36}  lin: {lin_m:.4f}±{lin_s:.4f}    mlp: {mlp_m:.4f}±{mlp_s:.4f}")
        per_seed_max(all_results["trained_pos"],    "trained POS   (max-layer per seed)")
        per_seed_max(all_results["trained_node"],   "trained NODE  (max-layer per seed)")
        per_seed_max(all_results["untrained_pos"],  "untrained POS  (max-layer per seed)")
        per_seed_max(all_results["untrained_node"], "untrained NODE (max-layer per seed)")


if __name__ == "__main__":
    main()
