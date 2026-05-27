"""
LatentWorldsGPT — Milestone 4 (flight-phase) probe.

Mirrors `eval/probe_music.py` and `eval/probe_othello.py` patterns for
the flight domain. Classification probe for the per-position flight
phase (~5-6 classes per Sun et al. fuzzy logic: GND/CL/CR/DE/LVL/NA).

Two splits in the spirit of cities' node-level / music's piece-level:
  - position-level: random partition of probe positions (cheap baseline;
    may suffer per-flight memorization at high accuracy)
  - flight-level: partition flights into disjoint sets — the probe trains
    on positions from one set of flights and tests on positions from a
    disjoint set, mirroring cities node-level / music piece-level.

Honest reporting with multi-seed + mean-vs-max headline (matching
probe_music.py infrastructure).
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
N_RESERVED = 3


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load probe targets
# ─────────────────────────────────────────────────────────────────────────────

def load_targets(data_dir: Path, splits=("val", "gen")):
    """Returns {(split, token_pos): phase_str} and {(split, token_pos): flight_idx}."""
    phases = {}
    flight_idx_of = {}
    with open(data_dir / "flight_phase.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["split"] not in splits:
                continue
            key = (row["split"], int(row["token_pos"]))
            phases[key] = row["phase"]
            flight_idx_of[key] = int(row["flight_idx"])
    return phases, flight_idx_of


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build probe dataset
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_probe_dataset(model, streams, targets, flight_idx_of, block_size,
                        n_positions, device, rng_seed=0):
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
    all_flights = []
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
                gp = gs + t
                split, pos = to_split_pos(gp)
                if split is None:
                    continue
                tok = int(idx_batch[b, t].item())
                if tok in (PAD, BOS, EOS):
                    continue
                key = (split, pos)
                if key not in targets:
                    continue
                for L in range(len(acts)):
                    all_X[L].append(acts[L][b, t].cpu().numpy())
                all_y.append(targets[key])
                all_flights.append(flight_idx_of[key])
                n_collected += 1
                if n_collected >= n_positions:
                    break
            if n_collected >= n_positions:
                break
    X = [np.stack(layer_list) for layer_list in all_X]
    return X, np.array(all_y), np.array(all_flights, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Probes
# ─────────────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, in_dim, n): super().__init__(); self.lin = nn.Linear(in_dim, n)
    def forward(self, x): return self.lin(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim, n, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n),
        )
    def forward(self, x): return self.net(x)


def train_eval(probe, Xtr, ytr, Xte, yte, device, epochs=30):
    probe = probe.to(device)
    Xtr = torch.from_numpy(Xtr).float().to(device)
    ytr = torch.from_numpy(ytr).long().to(device)
    Xte = torch.from_numpy(Xte).float().to(device)
    yte = torch.from_numpy(yte).long().to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-3)
    best = -1.0
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(perm), 512):
            ix = perm[i : i + 512]
            loss = F.cross_entropy(probe(Xtr[ix]), ytr[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (probe(Xte).argmax(-1) == yte).float().mean().item()
            best = max(best, acc)
    return best


def position_split(n, train_frac, seed):
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(n * train_frac)
    return perm[:n_train], perm[n_train:]


def flight_split(flights, train_frac, seed):
    uniq = np.unique(flights)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_train = int(len(uniq) * train_frac)
    train_flights = set(int(f) for f in uniq[perm[:n_train]])
    train_mask = np.array([int(f) in train_flights for f in flights])
    return np.nonzero(train_mask)[0], np.nonzero(~train_mask)[0]


def run_layer_sweep(X_layers, y, n_classes, train_ix, test_ix, epochs, device,
                    label, seeds=(0,)):
    if len(train_ix) == 0 or len(test_ix) == 0:
        print(f"\n{label}: empty split, skipping"); return []
    print(f"\n{'─'*78}\n{label}  (n_classes={n_classes}, seeds={list(seeds)})\n{'─'*78}")
    print(f"  {'Layer':<8}{'Lin μ±σ':>14}{'Lin max':>10}{'MLP μ±σ':>14}{'MLP max':>10}")
    rows = []
    for L, Xl in enumerate(X_layers):
        Xtr, Xte = Xl[train_ix], Xl[test_ix]
        ytr, yte = y[train_ix], y[test_ix]
        lin_accs = []; mlp_accs = []
        for s in seeds:
            torch.manual_seed(s); np.random.seed(s)
            a_lin = train_eval(LinearProbe(Xl.shape[1], n_classes),
                               Xtr, ytr, Xte, yte, device, epochs)
            a_mlp = train_eval(MLPProbe(Xl.shape[1], n_classes),
                               Xtr, ytr, Xte, yte, device, epochs)
            lin_accs.append(a_lin); mlp_accs.append(a_mlp)
        lin_mean = float(np.mean(lin_accs)); lin_std = float(np.std(lin_accs))
        mlp_mean = float(np.mean(mlp_accs)); mlp_std = float(np.std(mlp_accs))
        lin_max = float(max(lin_accs)); mlp_max = float(max(mlp_accs))
        lab = "embed" if L == 0 else f"L{L}"
        print(f"  {lab:<8}{lin_mean:.3f}±{lin_std:.3f}".ljust(22)
              + f"{lin_max:>10.4f}{mlp_mean:.3f}±{mlp_std:.3f}".ljust(24)
              + f"{mlp_max:>10.4f}")
        rows.append((L, lin_mean, lin_std, lin_max, mlp_mean, mlp_std, mlp_max))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--n_positions", type=int, default=5000)
    p.add_argument("--probe_train_frac", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--skip_untrained", action="store_true")
    p.add_argument("--skip_flight_split", action="store_true")
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
    print(f"loaded ckpt: iter={ckpt.get('iter','?')}  val_ppl={ckpt.get('val_perplexity',float('nan')):.4f}")

    print(f"\nLoading probe targets ...")
    targets, flight_idx_of = load_targets(data_dir, splits=("val", "gen"))
    n_flights = len(set(flight_idx_of.values()))
    print(f"  {len(targets):,} labeled positions across {n_flights} flights")

    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dtype = np.dtype(meta["dtype"])
    streams = {s: np.asarray(np.memmap(data_dir / f"{s}.bin", dtype=dtype, mode="r"))
               for s in ("val", "gen")}

    # all_results[cond][L] = list of (lin_acc, mlp_acc) per-seed
    all_results = {
        "trained_pos": defaultdict(list),
        "trained_flt": defaultdict(list),
        "untrained_pos": defaultdict(list),
        "untrained_flt": defaultdict(list),
    }
    class_info_printed = False

    for seed in args.seeds:
        print(f"\n{'#'*78}\n# SEED {seed}\n{'#'*78}")
        torch.manual_seed(seed); np.random.seed(seed)

        untrained = None if args.skip_untrained else GPT(config).to(device).eval()

        def build(model, lab):
            print(f"\nBuilding probe dataset for {lab} (seed={seed}) ...")
            t0 = time.time()
            X, y, flights = build_probe_dataset(
                model, streams, targets, flight_idx_of,
                config.block_size, args.n_positions, device, rng_seed=seed)
            print(f"  collected {len(y):,} positions ({time.time() - t0:.1f}s)")
            return X, y, flights

        X_t, y_t_str, flights_t = build(trained, "TRAINED")
        if untrained is not None:
            X_u, y_u_str, flights_u = build(untrained, "UNTRAINED")

        # Class encoding (stable across seeds since classes are deterministic)
        all_classes = sorted({c for c in y_t_str.tolist()
                              + (y_u_str.tolist() if untrained is not None else [])})
        cls_to_idx = {c: i for i, c in enumerate(all_classes)}
        y_t = np.array([cls_to_idx[c] for c in y_t_str], dtype=np.int64)
        n_classes = len(all_classes)
        if untrained is not None:
            y_u = np.array([cls_to_idx[c] for c in y_u_str], dtype=np.int64)

        if not class_info_printed:
            print(f"\nclasses ({n_classes}): {all_classes}")
            print(f"class distribution: {dict(Counter(y_t_str.tolist()))}")
            majority = max(Counter(y_t_str.tolist()).values()) / len(y_t_str)
            print(f"majority baseline: {majority:.4f}  (chance: {1/n_classes:.4f})")
            class_info_printed = True

        pos_train_t, pos_test_t = position_split(len(y_t), args.probe_train_frac, seed)
        if untrained is not None:
            pos_train_u, pos_test_u = position_split(len(y_u), args.probe_train_frac, seed)
        if not args.skip_flight_split:
            flt_train_t, flt_test_t = flight_split(flights_t, args.probe_train_frac, seed)
            if untrained is not None:
                flt_train_u, flt_test_u = flight_split(flights_u, args.probe_train_frac, seed)

        trained_pos = run_layer_sweep(
            X_t, y_t, n_classes, pos_train_t, pos_test_t, args.epochs, device,
            f"TRAINED — POSITION-LEVEL (seed {seed})", seeds=(seed,))
        for L, lin_mean, _, _, mlp_mean, _, _ in trained_pos:
            all_results["trained_pos"][L].append((lin_mean, mlp_mean))

        if not args.skip_flight_split:
            trained_flt = run_layer_sweep(
                X_t, y_t, n_classes, flt_train_t, flt_test_t, args.epochs, device,
                f"TRAINED — FLIGHT-LEVEL (seed {seed})", seeds=(seed,))
            for L, lin_mean, _, _, mlp_mean, _, _ in trained_flt:
                all_results["trained_flt"][L].append((lin_mean, mlp_mean))

        if untrained is not None:
            untrained_pos = run_layer_sweep(
                X_u, y_u, n_classes, pos_train_u, pos_test_u, args.epochs, device,
                f"UNTRAINED — POSITION-LEVEL (seed {seed})", seeds=(seed,))
            for L, lin_mean, _, _, mlp_mean, _, _ in untrained_pos:
                all_results["untrained_pos"][L].append((lin_mean, mlp_mean))

            if not args.skip_flight_split:
                untrained_flt = run_layer_sweep(
                    X_u, y_u, n_classes, flt_train_u, flt_test_u, args.epochs, device,
                    f"UNTRAINED — FLIGHT-LEVEL (seed {seed})", seeds=(seed,))
                for L, lin_mean, _, _, mlp_mean, _, _ in untrained_flt:
                    all_results["untrained_flt"][L].append((lin_mean, mlp_mean))

    # ─────────────────────────────────────────────────────────────────────────
    # Aggregate
    # ─────────────────────────────────────────────────────────────────────────
    n_s = len(args.seeds)

    def aggregate(layer_dict, label):
        if not layer_dict: return None
        print(f"\n{'='*78}\n{label}  (mean ± std over {n_s} seed(s))\n{'='*78}")
        print(f"  {'Layer':<8}{'LinAcc (mean±std)':>22}{'MLPAcc (mean±std)':>22}")
        rows = []
        for L in sorted(layer_dict.keys()):
            arr = np.array(layer_dict[L])
            lin_m = float(arr[:, 0].mean()); lin_s = float(arr[:, 0].std(ddof=1)) if n_s > 1 else 0.0
            mlp_m = float(arr[:, 1].mean()); mlp_s = float(arr[:, 1].std(ddof=1)) if n_s > 1 else 0.0
            lab = "embed" if L == 0 else f"L{L}"
            print(f"  {lab:<8}{lin_m:>10.4f}±{lin_s:.4f}    {mlp_m:>10.4f}±{mlp_s:.4f}")
            rows.append((L, lin_m, lin_s, mlp_m, mlp_s))
        return rows

    agg_tpos  = aggregate(all_results["trained_pos"],   "TRAINED — POSITION-LEVEL")
    agg_tflt  = aggregate(all_results["trained_flt"],   "TRAINED — FLIGHT-LEVEL")
    agg_upos  = aggregate(all_results["untrained_pos"], "UNTRAINED — POSITION-LEVEL")
    agg_uflt  = aggregate(all_results["untrained_flt"], "UNTRAINED — FLIGHT-LEVEL")

    print(f"\n{'═'*78}\nHEADLINE — best layer by mean over {n_s} seed(s)\n{'═'*78}")
    def show_best(rows, ix_m, ix_s, lab):
        if not rows: print(f"  {lab:<32}    —"); return
        b = max(rows, key=lambda r: r[ix_m])
        L, m, s = b[0], b[ix_m], b[ix_s]
        layer = "embed" if L == 0 else f"L{L}"
        print(f"  {lab:<32}  {layer:>5}   acc={m:.4f}±{s:.4f}")

    print("\n  POSITION-LEVEL:")
    show_best(agg_tpos, 1, 2, "trained  linear")
    show_best(agg_tpos, 3, 4, "trained  MLP")
    show_best(agg_upos, 1, 2, "untrained linear")
    show_best(agg_upos, 3, 4, "untrained MLP")
    if agg_tflt:
        print("\n  FLIGHT-LEVEL (held-out flights):")
        show_best(agg_tflt, 1, 2, "trained  linear")
        show_best(agg_tflt, 3, 4, "trained  MLP")
        show_best(agg_uflt, 1, 2, "untrained linear")
        show_best(agg_uflt, 3, 4, "untrained MLP")


if __name__ == "__main__":
    main()
