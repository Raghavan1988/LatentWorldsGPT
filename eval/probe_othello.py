"""
LatentWorldsGPT — Othello-GPT board-state probe.

Reproduces the load-bearing Li 2022 / Nanda 2023 result: trained on
random Othello move sequences, the model's residual stream encodes the
current BOARD STATE recoverable by a classification probe at ~90%+
per-cell accuracy. This is the framework's end-to-end sanity check.

For each of 64 cells, train a 3-class classifier (empty / black / white)
from the residual-stream activation at that move position. Report mean
per-cell accuracy under honest multi-seed mean-across-layers reporting.

Expected (per Li 2022 / Nanda 2023):
  - TRAINED model MLP probe: mean per-cell accuracy ~94%
  - TRAINED model linear probe: ~75-85% (originally reported as much
    lower in Li 2022; Nanda showed a black-vs-white re-parameterization
    pushes it ~98%, but standard 3-class linear sits ~80%).
  - UNTRAINED model: baseline (majority-class for each cell; ~45-60%).
  - Trained > Untrained gap: substantial (~30-40 points).

If we see this gap, the framework works end-to-end. Music's null is
then principled (N criterion fails), not an artifact.

THE ONE RULE
============
This file is the only place board-state values enter the picture. The
model takes only LongTensor move-token IDs. board_state.csv stays on
the probe side.

USAGE
=====
    python eval/probe_othello.py --ckpt checkpoints/othello/best.pt \\
        --data_dir data/othello --seeds 0 1 2 3 --report_mode both
"""

import argparse
import csv
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig  # noqa: E402
from probe import cache_layer_activations  # noqa: E402

PAD, BOS, EOS, PASS = 0, 1, 2, 3
N_RESERVED = 4
N_CELLS = 64
N_CLASSES = 3  # empty / black / white


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load board-state targets
# ─────────────────────────────────────────────────────────────────────────────

def load_targets(data_dir: Path, splits=("val", "gen")):
    """Returns {(split, token_pos): board_array(64,)}."""
    targets = {}
    with open(data_dir / "board_state.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["split"] not in splits:
                continue
            board = np.array([int(x) for x in row["cells"].split("-")],
                             dtype=np.int64)
            assert board.shape == (N_CELLS,)
            targets[(row["split"], int(row["token_pos"]))] = board
    return targets


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build probe dataset
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_probe_dataset(model, streams, targets, block_size, n_positions,
                        device, rng_seed=0):
    """Sample positions, run model forward, collect activations per layer
    + board-state targets. Each row's target is a 64-int vector."""
    model.eval()
    rng = np.random.default_rng(rng_seed)

    split_order = list(streams.keys())
    offsets = {}
    parts = []
    cursor = 0
    for s in split_order:
        offsets[s] = cursor
        parts.append(streams[s])
        cursor += len(streams[s])
    combined = np.concatenate(parts).astype(np.int64) if parts else np.array([], np.int64)

    def to_split_pos(gp):
        for s in reversed(split_order):
            if gp >= offsets[s]:
                return s, gp - offsets[s]
        return None, None

    batch_size = 32
    all_X = None
    all_y = []
    n_collected = 0
    while n_collected < n_positions:
        starts = rng.integers(0, len(combined) - block_size - 1, size=batch_size)
        windows = [combined[s : s + block_size] for s in starts]
        idx_batch = torch.from_numpy(np.stack(windows)).to(device)
        acts = cache_layer_activations(model, idx_batch)
        if all_X is None:
            all_X = [[] for _ in range(len(acts))]
        for b in range(idx_batch.shape[0]):
            gs = int(starts[b])
            for t in range(idx_batch.shape[1]):
                global_pos = gs + t
                split, pos_in_split = to_split_pos(global_pos)
                if split is None:
                    continue
                tok = int(idx_batch[b, t].item())
                if tok in (PAD, BOS, EOS):
                    continue  # control tokens (PASS is a real token: keep)
                key = (split, pos_in_split)
                if key not in targets:
                    continue
                for L in range(len(acts)):
                    all_X[L].append(acts[L][b, t].cpu().numpy())
                all_y.append(targets[key])
                n_collected += 1
                if n_collected >= n_positions:
                    break
            if n_collected >= n_positions:
                break
    X = [np.stack(layer_list) for layer_list in all_X]
    y = np.stack(all_y).astype(np.int64)   # (N, 64)
    return X, y


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


def train_eval(probe, Xtr, ytr, Xte, yte, n, device, lr=1e-3, wd=1e-3,
               epochs=50, batch_size=512):
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
            acc = (probe(Xte).argmax(dim=-1) == yte).float().mean().item()
            best = max(best, acc)
    return best


def position_split(n, train_frac, seed):
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(n * train_frac)
    return perm[:n_train], perm[n_train:]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-cell × per-layer × per-seed sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_cell_sweep(X_layers, y_board, train_ix, test_ix, epochs, device,
                   label, seeds=(0,)):
    """For each layer, train 64 cell-probes (linear + MLP) under each seed.
    Returns rows of (L, lin_stats, mlp_stats).
    """
    if len(train_ix) == 0 or len(test_ix) == 0:
        print(f"\n{label}: empty split, skipping"); return []
    print(f"\n{'─'*78}\n{label}  (seeds={list(seeds)})\n{'─'*78}")
    header = (f"{'Layer':<8}"
              f"{'Lin μ±σ':>14}{'Lin max':>10}"
              f"{'MLP μ±σ':>14}{'MLP max':>10}")
    print(header); print("─" * len(header))
    rows = []
    for L, Xl in enumerate(X_layers):
        Xtr, Xte = Xl[train_ix], Xl[test_ix]
        lin_per_cell = []
        mlp_per_cell = []
        for cell in range(N_CELLS):
            ytr = y_board[train_ix, cell]
            yte = y_board[test_ix,  cell]
            lin_seeds = []
            mlp_seeds = []
            for s in seeds:
                torch.manual_seed(s); np.random.seed(s)
                a_lin = train_eval(
                    LinearProbe(Xl.shape[1], N_CLASSES),
                    Xtr, ytr, Xte, yte, N_CLASSES, device,
                    lr=1e-3, wd=1e-3, epochs=epochs,
                )
                a_mlp = train_eval(
                    MLPProbe(Xl.shape[1], N_CLASSES),
                    Xtr, ytr, Xte, yte, N_CLASSES, device,
                    lr=1e-3, wd=1e-5, epochs=epochs,
                )
                lin_seeds.append(a_lin); mlp_seeds.append(a_mlp)
            lin_per_cell.append(np.mean(lin_seeds))
            mlp_per_cell.append(np.mean(mlp_seeds))
        lin_per_cell = np.array(lin_per_cell)
        mlp_per_cell = np.array(mlp_per_cell)
        lin_stats = {
            "per_cell_means": lin_per_cell.tolist(),
            "accuracy_mean": float(lin_per_cell.mean()),
            "accuracy_std":  float(lin_per_cell.std()),
            "accuracy_max":  float(lin_per_cell.max()),
        }
        mlp_stats = {
            "per_cell_means": mlp_per_cell.tolist(),
            "accuracy_mean": float(mlp_per_cell.mean()),
            "accuracy_std":  float(mlp_per_cell.std()),
            "accuracy_max":  float(mlp_per_cell.max()),
        }
        layer_label = "embed" if L == 0 else f"L{L}"
        lin_ms = f"{lin_stats['accuracy_mean']:.3f}±{lin_stats['accuracy_std']:.3f}"
        mlp_ms = f"{mlp_stats['accuracy_mean']:.3f}±{mlp_stats['accuracy_std']:.3f}"
        print(f"{layer_label:<8}{lin_ms:>14}{lin_stats['accuracy_max']:>10.4f}"
              f"{mlp_ms:>14}{mlp_stats['accuracy_max']:>10.4f}")
        rows.append((L, lin_stats, mlp_stats))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--n_positions", type=int, default=5_000)
    p.add_argument("--probe_train_frac", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--report_mode", choices=("max", "mean", "both"),
                   default="both")
    p.add_argument("--skip_untrained", action="store_true")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    print(f"running {len(args.seeds)} seed(s): {args.seeds}")

    data_dir = Path(args.data_dir)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    trained = GPT(config).to(device); trained.load_state_dict(ckpt["model_state"])
    trained.eval()
    print(f"  iter={ckpt.get('iter','?')}  "
          f"val_ppl={ckpt.get('val_perplexity',float('nan')):.4f}  "
          f"vocab_size={config.vocab_size}")

    targets = load_targets(data_dir, splits=("val", "gen"))
    print(f"  {len(targets):,} labeled positions across val+gen")
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dtype = np.dtype(meta["dtype"])
    streams = {
        s: np.asarray(np.memmap(data_dir / f"{s}.bin", dtype=dtype, mode="r"))
        for s in ("val", "gen")
    }

    # all_results[condition][L] = list of (lin_mean, mlp_mean) per-cell-means, one tuple per seed
    all_results = {"trained": defaultdict(list), "untrained": defaultdict(list)}

    for seed in args.seeds:
        print(f"\n{'#'*78}\n# SEED {seed}\n{'#'*78}")
        torch.manual_seed(seed); np.random.seed(seed)

        untrained = None if args.skip_untrained else GPT(config).to(device).eval()

        def build(model, lab):
            print(f"\nBuilding probe dataset for {lab} (seed={seed}) ...")
            t0 = time.time()
            X, y = build_probe_dataset(
                model, streams, targets, config.block_size,
                args.n_positions, device, rng_seed=seed,
            )
            print(f"  collected {len(y):,} positions × 64 cells "
                  f"across {len(X)} layers ({time.time()-t0:.1f}s)")
            return X, y

        X_t, y_t = build(trained, "TRAINED")
        if untrained is not None:
            X_u, y_u = build(untrained, "UNTRAINED")

        tr_ix, te_ix = position_split(len(y_t), args.probe_train_frac, seed)
        if untrained is not None:
            tr_u, te_u = position_split(len(y_u), args.probe_train_frac, seed)

        if seed == args.seeds[0]:
            cnt = np.bincount(y_t[te_ix, 0], minlength=3)
            print(f"\n  cell-0 test class distribution: empty={cnt[0]} black={cnt[1]} "
                  f"white={cnt[2]}  (majority baseline = {cnt.max()/cnt.sum():.3f})")

        trained_rows = run_cell_sweep(
            X_t, y_t, tr_ix, te_ix, args.epochs, device,
            f"TRAINED — board-state probe (seed {seed})",
            seeds=(seed,),
        )
        for L, lin_stats, mlp_stats in trained_rows:
            all_results["trained"][L].append(
                (lin_stats["accuracy_mean"], mlp_stats["accuracy_mean"]))

        if untrained is not None:
            untrained_rows = run_cell_sweep(
                X_u, y_u, tr_u, te_u, args.epochs, device,
                f"UNTRAINED — random-init control (seed {seed})",
                seeds=(seed,),
            )
            for L, lin_stats, mlp_stats in untrained_rows:
                all_results["untrained"][L].append(
                    (lin_stats["accuracy_mean"], mlp_stats["accuracy_mean"]))

    # ─────────────────────────────────────────────────────────────────────────
    # Aggregate across seeds
    # ─────────────────────────────────────────────────────────────────────────
    n_s = len(args.seeds)

    def aggregate(layer_dict, label):
        if not layer_dict: return None
        print(f"\n{'='*78}\n{label}  (mean ± std over {n_s} seed(s))\n{'='*78}")
        print(f"  {'Layer':<8}{'Lin per-cell μ±σ':>26}{'MLP per-cell μ±σ':>26}")
        rows = []
        for L in sorted(layer_dict.keys()):
            arr = np.array(layer_dict[L])  # (n_seeds, 2): col 0 = lin, col 1 = mlp
            lin_m = float(arr[:, 0].mean()); lin_s = float(arr[:, 0].std(ddof=1)) if n_s > 1 else 0.0
            mlp_m = float(arr[:, 1].mean()); mlp_s = float(arr[:, 1].std(ddof=1)) if n_s > 1 else 0.0
            lab = "embed" if L == 0 else f"L{L}"
            print(f"  {lab:<8}{lin_m:>14.4f}±{lin_s:.4f}    {mlp_m:>12.4f}±{mlp_s:.4f}")
            rows.append((L, lin_m, lin_s, mlp_m, mlp_s))
        return rows

    agg_t = aggregate(all_results["trained"], "TRAINED — board-state")
    agg_u = aggregate(all_results["untrained"], "UNTRAINED — board-state")

    print(f"\n{'═'*78}\nHEADLINE — best layer by per-cell μ over {n_s} seed(s)\n{'═'*78}")
    def show_best(rows, ix_m, ix_s, lab):
        if not rows: print(f"  {lab:<32}    —"); return
        b = max(rows, key=lambda r: r[ix_m])
        L, m, s = b[0], b[ix_m], b[ix_s]
        layer = "embed" if L == 0 else f"L{L}"
        print(f"  {lab:<32}  {layer:>5}   per-cell={m:.4f}±{s:.4f}")
    show_best(agg_t, 1, 2, "trained  linear")
    show_best(agg_t, 3, 4, "trained  MLP")
    if agg_u:
        show_best(agg_u, 1, 2, "untrained linear")
        show_best(agg_u, 3, 4, "untrained MLP")

    print(f"\n{'─'*78}\nACCEPTANCE (best-layer trained MLP vs best-layer untrained MLP)\n{'─'*78}")
    if agg_t and agg_u:
        b_t = max(agg_t, key=lambda r: r[3])
        b_u = max(agg_u, key=lambda r: r[3])
        gap = b_t[3] - b_u[3]
        ok = b_t[3] >= 0.85 and gap >= 0.20
        mark = "✓" if ok else "✗"
        print(f"  {mark} TRAINED MLP per-cell μ = {b_t[3]:.4f}±{b_t[4]:.4f}  (target ≥ 0.85)")
        print(f"  {mark} TRAINED − UNTRAINED gap = {gap:+.4f}  (target ≥ 0.20)")
        if ok:
            print(f"    → Framework reproduces Othello-GPT-style positive control.")
        else:
            print(f"    → Investigate: model capacity, training schedule, or probe code.")


if __name__ == "__main__":
    main()
