"""
LatentWorldsGPT — Milestone 2 (music) probe suite.

Mirrors eval/probe.py's architecture but for classification targets:
    mode   : 2 classes  (major / minor) — piece-level constant
    chord  : ~50–200 classes (Roman-numeral figures from music21)
    beat   : 4 classes  (beat-in-measure 1..4) — load-bearing Othello-positive

WHAT WE ARE TESTING
====================
The pivot.md Milestone-2 outcome matrix: each probe is an independent bet on
whether sequence order is load-bearing for that target in tonal music.

  Real model + within-piece-shuffled model + global-shuffled model -> the
  three-condition gradient. This script handles a single (model, data_dir)
  pair; run it three times and compare.

THE ONE RULE for music
=======================
The model takes only LongTensor pitch token IDs as input. Probe targets
(mode / chord / beat) live in key.csv / chord.csv / beat.csv side tables
and are read only by this file.

SPLITS
======
We use TWO splits, in deliberate analogy to cities:

  position-level : random partition of probe positions. Cheap baseline.
                   For beat / mode / chord, the same target value appears
                   in train AND test (4 beats, 2 modes, ~50-200 chords);
                   a probe can pass by learning per-position artifacts.

  PIECE-level    : the meaningful capacity test. Probe trained on positions
                   from a set of pieces, tested on positions from a DISJOINT
                   set of pieces. A probe that learned a real representation
                   should generalize across pieces; a probe that memorized
                   piece-specific artifacts cannot.

USAGE
=====
    python eval/probe_music.py --ckpt checkpoints/music_bach/best.pt \
        --data_dir data/music_bach

    # restrict to one target (faster)
    python eval/probe_music.py --ckpt checkpoints/music_bach/best.pt \
        --data_dir data/music_bach --targets beat
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

# Reuse the model package + the cities probe's activation cacher
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig  # noqa: E402
from probe import cache_layer_activations  # noqa: E402

# Mirror data/prepare_music.py reserved indices.
PAD, BOS, EOS, REST = 0, 1, 2, 3
N_RESERVED = 4

TARGET_NAMES = ("mode", "chord", "beat")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load probe targets from CSV side tables
# ─────────────────────────────────────────────────────────────────────────────

def load_probe_targets(data_dir: Path, splits: tuple[str, ...] = ("val", "gen")):
    """Return per-target dicts: {target_name: {(split, token_pos): label}}.

    Restricted to rows whose split is in `splits`. We also collect piece_idx
    per (split, token_pos) so the piece-level split can be built later.
    """
    targets = {"mode": {}, "chord": {}, "beat": {}}
    piece_idx_of = {}  # (split, token_pos) -> piece_idx

    for tgt, fname, value_caster in [
        ("mode",  "key.csv",   lambda v: int(v)),
        ("chord", "chord.csv", lambda v: v),                # keep as string label
        ("beat",  "beat.csv",  lambda v: int(v)),
    ]:
        with open(data_dir / fname) as f:
            r = csv.DictReader(f)
            for row in r:
                split = row["split"]
                if split not in splits:
                    continue
                key = (split, int(row["token_pos"]))
                targets[tgt][key] = value_caster(row[tgt if tgt != "chord" else "roman"])
                if tgt == "mode":  # only build piece_idx_of once
                    piece_idx_of[key] = int(row["piece_idx"])
    return targets, piece_idx_of


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build the probe dataset
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_probe_dataset(model: GPT, streams: dict[str, np.ndarray],
                        targets: dict, piece_idx_of: dict,
                        block_size: int, n_positions: int, device: str,
                        rng_seed: int = 0):
    """Sample positions across `streams` (dict split -> np array). For each
    kept position record activations per layer + the three target labels +
    the source piece_idx (for the piece-level split).

    A "kept position" is one with a real-pitch token (id >= N_RESERVED) AND
    valid target labels at that (split, token_pos). REST positions are kept
    if labels exist — REST is a valid music event but the target tables emit
    -1 for BOS/EOS only; rests get real labels.
    """
    model.eval()
    rng = np.random.default_rng(rng_seed)

    # Concatenate streams with a marker of which split each position came from.
    # Position index inside this concatenation maps back to (split, pos_in_split)
    # via the cumulative offsets.
    split_order = list(streams.keys())
    offsets = {}
    parts = []
    cursor = 0
    for s in split_order:
        offsets[s] = cursor
        parts.append(streams[s])
        cursor += len(streams[s])
    combined = np.concatenate(parts).astype(np.int64) if parts else np.array([], dtype=np.int64)

    def to_split_pos(global_idx: int):
        for s in reversed(split_order):
            if global_idx >= offsets[s]:
                return s, global_idx - offsets[s]
        return None, None

    batch_size = 32
    all_X = None
    all_y = {"mode": [], "chord": [], "beat": []}
    all_pieces = []
    n_collected = 0

    while n_collected < n_positions:
        # Sample block starts that don't fall off the end.
        starts = rng.integers(0, len(combined) - block_size - 1, size=batch_size)
        windows = [combined[s : s + block_size] for s in starts]
        idx_batch = torch.from_numpy(np.stack(windows)).to(device)
        assert idx_batch.dtype in (torch.long, torch.int64)
        acts = cache_layer_activations(model, idx_batch)

        if all_X is None:
            n_layers = len(acts)
            all_X = [[] for _ in range(n_layers)]

        for b in range(idx_batch.shape[0]):
            global_start = int(starts[b])
            for t in range(idx_batch.shape[1]):
                global_pos = global_start + t
                split, pos_in_split = to_split_pos(global_pos)
                if split is None:
                    continue
                tok = int(idx_batch[b, t].item())
                # skip control tokens
                if tok in (PAD, BOS, EOS):
                    continue
                key = (split, pos_in_split)
                if key not in targets["beat"]:
                    continue
                # We have valid labels — collect.
                for L in range(len(acts)):
                    all_X[L].append(acts[L][b, t].cpu().numpy())
                all_y["mode"].append(targets["mode"][key])
                all_y["chord"].append(targets["chord"][key])
                all_y["beat"].append(targets["beat"][key])
                all_pieces.append(piece_idx_of[key])
                n_collected += 1
                if n_collected >= n_positions:
                    break
            if n_collected >= n_positions:
                break

    X = [np.stack(layer_list) for layer_list in all_X]  # each (N, n_embd)
    y_mode = np.array(all_y["mode"], dtype=np.int64)
    y_chord = np.array(all_y["chord"])      # object dtype (strings)
    y_beat = np.array(all_y["beat"], dtype=np.int64)
    pieces = np.array(all_pieces, dtype=np.int64)
    return X, y_mode, y_chord, y_beat, pieces


# ─────────────────────────────────────────────────────────────────────────────
# 3. Probes (linear + MLP, classification)
# ─────────────────────────────────────────────────────────────────────────────

class LinearClassifier(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_and_eval_classifier(probe, X_train, y_train, X_test, y_test,
                              n_classes, device, lr=1e-3, weight_decay=1e-3,
                              epochs=100, batch_size=512):
    probe = probe.to(device)
    Xtr = torch.from_numpy(X_train).float().to(device)
    ytr = torch.from_numpy(y_train).long().to(device)
    Xte = torch.from_numpy(X_test).float().to(device)
    yte = torch.from_numpy(y_test).long().to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    n = Xtr.shape[0]
    best_acc = -1.0
    best_state = None
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        probe.train()
        for i in range(0, n, batch_size):
            ix = perm[i : i + batch_size]
            logits = probe(Xtr[ix])
            loss = F.cross_entropy(logits, ytr[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            pred = probe(Xte).argmax(dim=-1)
            acc = (pred == yte).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        pred = probe(Xte).argmax(dim=-1).cpu().numpy()
    # Macro F1 — handles class imbalance (chord probe will be heavily skewed).
    macro_f1 = _macro_f1(y_test, pred, n_classes)
    return {"accuracy": best_acc, "macro_f1": macro_f1}


def _macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp == 0 or tp + fn == 0:
            continue
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        if p + r == 0:
            continue
        f1s.append(2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Splits
# ─────────────────────────────────────────────────────────────────────────────

def position_split(n, train_frac, seed):
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(n * train_frac)
    return perm[:n_train], perm[n_train:]


def piece_split(pieces, train_frac, seed):
    """Partition by piece_idx. Train and test pieces are disjoint — this is
    the music-domain analogue of cities' node-level split.
    """
    uniq = np.unique(pieces)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_train_pieces = int(len(uniq) * train_frac)
    train_pieces = set(int(p) for p in uniq[perm[:n_train_pieces]])
    train_mask = np.array([int(p) in train_pieces for p in pieces])
    return (
        np.nonzero(train_mask)[0],
        np.nonzero(~train_mask)[0],
        {"n_train_pieces": n_train_pieces,
         "n_test_pieces": len(uniq) - n_train_pieces},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Layer sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_layer_sweep(X_layers, y, n_classes, train_ix, test_ix, epochs, device,
                    label, target_name, seeds=(0,)):
    """For each layer, train a Linear + MLP classifier under each seed in
    `seeds`. Returns rows of (L, lin_stats, mlp_stats) where each stats dict
    has 'accuracies' (list), 'accuracy_mean', 'accuracy_std', 'accuracy_max',
    and 'macro_f1_mean'.

    seeds = (0,) reproduces the single-seed behavior. Multiple seeds
    re-initialize the classifier and re-shuffle the train/test split
    within the already-sampled probe positions, providing a more honest
    estimate of probe accuracy under noise.
    """
    if len(train_ix) == 0 or len(test_ix) == 0:
        print(f"\n{label}: empty split, skipping")
        return []
    print(f"\n{'─'*78}")
    print(f"{label}  —  target: {target_name}  "
          f"(n_classes={n_classes}, seeds={list(seeds)})")
    print(f"{'─'*78}")
    header = (f"{'Layer':<8}"
              f"{'Lin μ±σ':>14}{'Lin max':>10}"
              f"{'MLP μ±σ':>14}{'MLP max':>10}")
    print(header); print("─" * len(header))
    rows = []
    for L, Xl in enumerate(X_layers):
        Xtr, Xte = Xl[train_ix], Xl[test_ix]
        ytr, yte = y[train_ix],  y[test_ix]
        lin_accs = []; mlp_accs = []
        lin_f1s = []; mlp_f1s = []
        for s in seeds:
            torch.manual_seed(s); np.random.seed(s)
            lin = train_and_eval_classifier(
                LinearClassifier(Xl.shape[1], n_classes),
                Xtr, ytr, Xte, yte, n_classes, device,
                lr=1e-3, weight_decay=1e-3, epochs=epochs,
            )
            mlp = train_and_eval_classifier(
                MLPClassifier(Xl.shape[1], n_classes),
                Xtr, ytr, Xte, yte, n_classes, device,
                lr=1e-3, weight_decay=1e-5, epochs=epochs,
            )
            lin_accs.append(lin["accuracy"]); lin_f1s.append(lin["macro_f1"])
            mlp_accs.append(mlp["accuracy"]); mlp_f1s.append(mlp["macro_f1"])

        lin_stats = {
            "accuracies":     lin_accs,
            "accuracy_mean":  float(np.mean(lin_accs)),
            "accuracy_std":   float(np.std(lin_accs)),
            "accuracy_max":   float(np.max(lin_accs)),
            "macro_f1_mean":  float(np.mean(lin_f1s)),
        }
        mlp_stats = {
            "accuracies":     mlp_accs,
            "accuracy_mean":  float(np.mean(mlp_accs)),
            "accuracy_std":   float(np.std(mlp_accs)),
            "accuracy_max":   float(np.max(mlp_accs)),
            "macro_f1_mean":  float(np.mean(mlp_f1s)),
        }
        layer_label = "embed" if L == 0 else f"L{L}"
        lin_ms = f"{lin_stats['accuracy_mean']:.3f}±{lin_stats['accuracy_std']:.3f}"
        mlp_ms = f"{mlp_stats['accuracy_mean']:.3f}±{mlp_stats['accuracy_std']:.3f}"
        print(f"{layer_label:<8}{lin_ms:>14}{lin_stats['accuracy_max']:>10.4f}"
              f"{mlp_ms:>14}{mlp_stats['accuracy_max']:>10.4f}")
        rows.append((L, lin_stats, mlp_stats))
    return rows


def encode_chord_labels(y_chord_str):
    """Map chord string labels to integer class indices. '?' (unanalyzable)
    is class 0, real chord labels start at 1.
    """
    classes = ["?"] + sorted({c for c in y_chord_str if c != "?"})
    chord_to_idx = {c: i for i, c in enumerate(classes)}
    return np.array([chord_to_idx[c] for c in y_chord_str], dtype=np.int64), classes


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--n_positions", type=int, default=20_000)
    p.add_argument("--probe_train_frac", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=None,
                   help="legacy single-seed flag; use --seeds instead")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="one or more probe seeds (default [0]). Each seed "
                        "re-runs untrained init + probe-data sampling + "
                        "classifier training; results aggregated across seeds.")
    p.add_argument("--targets", nargs="+", default=list(TARGET_NAMES),
                   choices=TARGET_NAMES, help="which probe targets to run")
    p.add_argument("--skip_untrained", action="store_true")
    p.add_argument("--skip_piece_split", action="store_true")
    args = p.parse_args()

    if args.seeds is None:
        args.seeds = [args.seed] if args.seed is not None else [0]
    if args.seed is None:
        args.seed = args.seeds[0]

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"device: {device}")
    print(f"running {len(args.seeds)} seed(s): {args.seeds}")

    data_dir = Path(args.data_dir)

    print(f"\nLoading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    trained_model = GPT(config).to(device); trained_model.load_state_dict(ckpt["model_state"])
    trained_model.eval()
    print(f"  iter={ckpt.get('iter','?')}  val_ppl={ckpt.get('val_perplexity',float('nan')):.4f}"
          f"  vocab_size={config.vocab_size}")

    print(f"\nLoading probe-target side tables ...")
    targets, piece_idx_of = load_probe_targets(data_dir, splits=("val", "gen"))
    n_target_rows = len(targets["beat"])
    n_pieces = len(set(piece_idx_of.values()))
    print(f"  {n_target_rows:,} labeled positions across {n_pieces} pieces (val+gen)")

    # Pre-compute chord vocab from the side table (deterministic across seeds).
    all_chord_strs = set(targets["chord"].values())
    chord_classes = ["?"] + sorted({c for c in all_chord_strs if c != "?"})
    chord_to_idx = {c: i for i, c in enumerate(chord_classes)}
    n_chord_classes = len(chord_classes)
    print(f"  chord vocab: {n_chord_classes} classes (incl. '?' unanalyzable)")

    print(f"\nLoading val + gen streams ...")
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dtype = np.dtype(meta["dtype"])
    streams = {
        s: np.asarray(np.memmap(data_dir / f"{s}.bin", dtype=dtype, mode="r"))
        for s in ("val", "gen")
    }
    for s, arr in streams.items():
        print(f"  {s}.bin: {len(arr):,} tokens")

    # Accumulator: all_results[tgt][cond][L] = list of (lin_mean, mlp_mean) tuples per seed
    conditions = ["pos_trained", "piece_trained", "pos_untrained", "piece_untrained"]
    all_results = {t: {c: defaultdict(list) for c in conditions} for t in args.targets}
    majorities = {}  # per target; computed once on first seed

    for seed in args.seeds:
        print(f"\n{'#'*78}\n# SEED {seed}\n{'#'*78}")
        torch.manual_seed(seed); np.random.seed(seed)

        untrained_model = None if args.skip_untrained else GPT(config).to(device).eval()

        def build(model, label):
            print(f"\nBuilding probe dataset for {label} (seed={seed}) ...")
            t0 = time.time()
            out = build_probe_dataset(
                model, streams, targets, piece_idx_of,
                config.block_size, args.n_positions, device, rng_seed=seed,
            )
            print(f"  collected {len(out[1]):,} positions across {len(out[0])} layers"
                  f"  ({time.time()-t0:.1f}s)")
            return out

        X_t, y_mode_t, y_chord_t, y_beat_t, pieces_t = build(trained_model, "TRAINED")
        if untrained_model is not None:
            X_u, y_mode_u, y_chord_u, y_beat_u, pieces_u = build(untrained_model, "UNTRAINED")

        pos_train_t, pos_test_t = position_split(
            len(y_beat_t), args.probe_train_frac, seed)
        if untrained_model is not None:
            pos_train_u, pos_test_u = position_split(
                len(y_beat_u), args.probe_train_frac, seed)

        if not args.skip_piece_split:
            piece_train_t, piece_test_t, _ = piece_split(
                pieces_t, args.probe_train_frac, seed)
            if untrained_model is not None:
                piece_train_u, piece_test_u, _ = piece_split(
                    pieces_u, args.probe_train_frac, seed)

        # Encode chord targets using the pre-computed deterministic vocab
        y_chord_t_int = np.array(
            [chord_to_idx.get(c, 0) for c in y_chord_t], dtype=np.int64)
        if untrained_model is not None:
            y_chord_u_int = np.array(
                [chord_to_idx.get(c, 0) for c in y_chord_u], dtype=np.int64)

        target_specs_t = {
            "mode":  (y_mode_t,    2),
            "chord": (y_chord_t_int, n_chord_classes),
            "beat":  (y_beat_t,    5),
        }
        if untrained_model is not None:
            target_specs_u = {
                "mode":  (y_mode_u,    2),
                "chord": (y_chord_u_int, n_chord_classes),
                "beat":  (y_beat_u,    5),
            }

        for tgt in args.targets:
            y_t, ncls = target_specs_t[tgt]
            if seed == args.seeds[0]:
                cnts = Counter(y_t.tolist())
                majorities[tgt] = max(cnts.values()) / len(y_t)
                print(f"\n[{tgt}] majority-class baseline: {majorities[tgt]:.4f}")

            rows = run_layer_sweep(
                X_t, y_t, ncls, pos_train_t, pos_test_t, args.epochs, device,
                label=f"TRAINED — POSITION-LEVEL (seed {seed})", target_name=tgt,
                seeds=(seed,),
            )
            for L, lin_stats, mlp_stats in rows:
                all_results[tgt]["pos_trained"][L].append(
                    (lin_stats["accuracy_mean"], mlp_stats["accuracy_mean"]))

            if not args.skip_piece_split:
                rows = run_layer_sweep(
                    X_t, y_t, ncls, piece_train_t, piece_test_t, args.epochs, device,
                    label=f"TRAINED — PIECE-LEVEL (seed {seed})", target_name=tgt,
                    seeds=(seed,),
                )
                for L, lin_stats, mlp_stats in rows:
                    all_results[tgt]["piece_trained"][L].append(
                        (lin_stats["accuracy_mean"], mlp_stats["accuracy_mean"]))

            if untrained_model is not None:
                y_u, _ = target_specs_u[tgt]
                rows = run_layer_sweep(
                    X_u, y_u, ncls, pos_train_u, pos_test_u, args.epochs, device,
                    label=f"UNTRAINED — POSITION-LEVEL (seed {seed})", target_name=tgt,
                    seeds=(seed,),
                )
                for L, lin_stats, mlp_stats in rows:
                    all_results[tgt]["pos_untrained"][L].append(
                        (lin_stats["accuracy_mean"], mlp_stats["accuracy_mean"]))

                if not args.skip_piece_split:
                    rows = run_layer_sweep(
                        X_u, y_u, ncls, piece_train_u, piece_test_u, args.epochs, device,
                        label=f"UNTRAINED — PIECE-LEVEL (seed {seed})", target_name=tgt,
                        seeds=(seed,),
                    )
                    for L, lin_stats, mlp_stats in rows:
                        all_results[tgt]["piece_untrained"][L].append(
                            (lin_stats["accuracy_mean"], mlp_stats["accuracy_mean"]))

    # ─────────────────────────────────────────────────────────────────────────
    # Aggregate across seeds
    # ─────────────────────────────────────────────────────────────────────────
    n_s = len(args.seeds)

    def aggregate(layer_dict, label):
        if not layer_dict: return None
        print(f"\n{'-'*78}\n{label}  (mean ± std over {n_s} seed(s))\n{'-'*78}")
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

    print(f"\n{'═'*78}\nAGGREGATE TABLES (mean ± std over {n_s} seed(s))\n{'═'*78}")
    aggregated = {t: {} for t in args.targets}
    for tgt in args.targets:
        print(f"\n[{tgt}]  majority baseline = {majorities.get(tgt, float('nan')):.4f}")
        aggregated[tgt]["pos_trained"]    = aggregate(all_results[tgt]["pos_trained"],    f"TRAINED — POSITION-LEVEL [{tgt}]")
        aggregated[tgt]["piece_trained"]  = aggregate(all_results[tgt]["piece_trained"],  f"TRAINED — PIECE-LEVEL [{tgt}]")
        aggregated[tgt]["pos_untrained"]  = aggregate(all_results[tgt]["pos_untrained"],  f"UNTRAINED — POSITION-LEVEL [{tgt}]")
        aggregated[tgt]["piece_untrained"] = aggregate(all_results[tgt]["piece_untrained"], f"UNTRAINED — PIECE-LEVEL [{tgt}]")

    print(f"\n{'═'*78}\nHEADLINE — best layer by mean over {n_s} seed(s)\n{'═'*78}")
    def show_best(rows, ix_m, ix_s, lab):
        if not rows: print(f"  {lab:<32}    —"); return
        b = max(rows, key=lambda r: r[ix_m])
        L, m, s = b[0], b[ix_m], b[ix_s]
        layer = "embed" if L == 0 else f"L{L}"
        print(f"  {lab:<32}  {layer:>5}   acc={m:.4f}±{s:.4f}")

    for tgt in args.targets:
        print(f"\n  [{tgt}]")
        print("    POSITION-LEVEL:")
        show_best(aggregated[tgt]["pos_trained"],   1, 2, "trained  linear")
        show_best(aggregated[tgt]["pos_trained"],   3, 4, "trained  MLP")
        show_best(aggregated[tgt]["pos_untrained"], 1, 2, "untrained linear")
        show_best(aggregated[tgt]["pos_untrained"], 3, 4, "untrained MLP")
        if aggregated[tgt]["piece_trained"]:
            print("    PIECE-LEVEL (held-out pieces):")
            show_best(aggregated[tgt]["piece_trained"],   1, 2, "trained  linear")
            show_best(aggregated[tgt]["piece_trained"],   3, 4, "trained  MLP")
            show_best(aggregated[tgt]["piece_untrained"], 1, 2, "untrained linear")
            show_best(aggregated[tgt]["piece_untrained"], 3, 4, "untrained MLP")


if __name__ == "__main__":
    main()
