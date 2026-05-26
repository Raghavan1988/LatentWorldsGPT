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
from collections import Counter
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
                    label, target_name):
    if len(train_ix) == 0 or len(test_ix) == 0:
        print(f"\n{label}: empty split, skipping")
        return []
    print(f"\n{'─'*78}")
    print(f"{label}  —  target: {target_name}  (n_classes={n_classes})")
    print(f"{'─'*78}")
    header = f"{'Layer':<8}{'LinAcc':>10}{'LinF1':>10}{'MLPAcc':>10}{'MLPF1':>10}"
    print(header); print("─" * len(header))
    rows = []
    for L, Xl in enumerate(X_layers):
        Xtr, Xte = Xl[train_ix], Xl[test_ix]
        ytr, yte = y[train_ix],  y[test_ix]
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
        layer_label = "embed" if L == 0 else f"L{L}"
        print(f"{layer_label:<8}{lin['accuracy']:>10.4f}{lin['macro_f1']:>10.4f}"
              f"{mlp['accuracy']:>10.4f}{mlp['macro_f1']:>10.4f}")
        rows.append((L, lin, mlp))
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--targets", nargs="+", default=list(TARGET_NAMES),
                   choices=TARGET_NAMES, help="which probe targets to run")
    p.add_argument("--skip_untrained", action="store_true")
    p.add_argument("--skip_piece_split", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"device: {device}")

    data_dir = Path(args.data_dir)

    print(f"\nLoading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    trained_model = GPT(config).to(device); trained_model.eval()
    untrained_model = None if args.skip_untrained else GPT(config).to(device).eval()
    print(f"  iter={ckpt.get('iter','?')}  val_ppl={ckpt.get('val_perplexity',float('nan')):.4f}"
          f"  vocab_size={config.vocab_size}")

    print(f"\nLoading probe-target side tables ...")
    targets, piece_idx_of = load_probe_targets(data_dir, splits=("val", "gen"))
    n_target_rows = len(targets["beat"])
    n_pieces = len(set(piece_idx_of.values()))
    print(f"  {n_target_rows:,} labeled positions across {n_pieces} pieces (val+gen)")

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

    def build(model, label):
        print(f"\nBuilding probe dataset for {label} model ...")
        t0 = time.time()
        out = build_probe_dataset(
            model, streams, targets, piece_idx_of,
            config.block_size, args.n_positions, device, rng_seed=args.seed,
        )
        print(f"  collected {len(out[1]):,} positions across {len(out[0])} layers"
              f"  ({time.time()-t0:.1f}s)")
        return out

    X_t, y_mode_t, y_chord_t, y_beat_t, pieces_t = build(trained_model, "TRAINED")
    if untrained_model is not None:
        X_u, y_mode_u, y_chord_u, y_beat_u, pieces_u = build(untrained_model, "UNTRAINED")

    # Splits — built on trained-model positions; untrained re-derives its own.
    pos_train_t, pos_test_t = position_split(
        len(y_beat_t), args.probe_train_frac, args.seed)
    print(f"\nPOSITION-LEVEL split: train={len(pos_train_t):,}  test={len(pos_test_t):,}")
    if untrained_model is not None:
        pos_train_u, pos_test_u = position_split(
            len(y_beat_u), args.probe_train_frac, args.seed)

    piece_train_t = piece_test_t = pi_t_stats = None
    if not args.skip_piece_split:
        piece_train_t, piece_test_t, pi_t_stats = piece_split(
            pieces_t, args.probe_train_frac, args.seed)
        print(f"PIECE-LEVEL split:  train_pieces={pi_t_stats['n_train_pieces']}"
              f"  test_pieces={pi_t_stats['n_test_pieces']}  "
              f"(train_pos={len(piece_train_t):,}  test_pos={len(piece_test_t):,})")
        if untrained_model is not None:
            piece_train_u, piece_test_u, _ = piece_split(
                pieces_u, args.probe_train_frac, args.seed)

    # Encode chord-string labels to class IDs (shared mapping across models).
    chord_strs = list(y_chord_t)
    if untrained_model is not None:
        chord_strs += list(y_chord_u)
    classes = ["?"] + sorted({c for c in chord_strs if c != "?"})
    chord_to_idx = {c: i for i, c in enumerate(classes)}
    y_chord_t_int = np.array([chord_to_idx[c] for c in y_chord_t], dtype=np.int64)
    n_chord_classes = len(classes)
    if untrained_model is not None:
        y_chord_u_int = np.array([chord_to_idx[c] for c in y_chord_u], dtype=np.int64)

    target_specs = {
        "mode":  (y_mode_t,    2),
        "chord": (y_chord_t_int, n_chord_classes),
        "beat":  (y_beat_t,    5),   # values are 1..4; using n_classes=5 covers it
    }

    summaries = {}  # target -> dict of (split -> {linear,mlp})
    for tgt in args.targets:
        y, ncls = target_specs[tgt]
        # class-balance sanity
        cnts = Counter(y.tolist())
        majority = max(cnts.values()) / len(y)
        print(f"\n[{tgt}] majority-class baseline (random guess of most common): {majority:.4f}")

        summaries[tgt] = {"pos_trained": None, "piece_trained": None,
                          "pos_untrained": None, "piece_untrained": None}

        summaries[tgt]["pos_trained"] = run_layer_sweep(
            X_t, y, ncls, pos_train_t, pos_test_t, args.epochs, device,
            label=f"TRAINED — POSITION-LEVEL", target_name=tgt,
        )
        if not args.skip_piece_split:
            summaries[tgt]["piece_trained"] = run_layer_sweep(
                X_t, y, ncls, piece_train_t, piece_test_t, args.epochs, device,
                label=f"TRAINED — PIECE-LEVEL (held-out pieces)", target_name=tgt,
            )
        if untrained_model is not None:
            y_u = {"mode": y_mode_u, "chord": y_chord_u_int, "beat": y_beat_u}[tgt]
            summaries[tgt]["pos_untrained"] = run_layer_sweep(
                X_u, y_u, ncls, pos_train_u, pos_test_u, args.epochs, device,
                label=f"UNTRAINED — POSITION-LEVEL (random-init control)",
                target_name=tgt,
            )
            if not args.skip_piece_split:
                summaries[tgt]["piece_untrained"] = run_layer_sweep(
                    X_u, y_u, ncls, piece_train_u, piece_test_u, args.epochs, device,
                    label=f"UNTRAINED — PIECE-LEVEL", target_name=tgt,
                )

    # ── headline ──
    print(f"\n{'═'*78}\nHEADLINE\n{'═'*78}")

    def best(rows, probe_ix):
        if not rows:
            return None
        return max(rows, key=lambda r: r[probe_ix]["accuracy"])

    def show(rows, probe_ix, label):
        b = best(rows, probe_ix)
        if b is None:
            print(f"  {label:<32}    —")
            return
        L, _, _ = b
        res = b[probe_ix]
        layer = "embed" if L == 0 else f"L{L}"
        print(f"  {label:<32}  {layer:>5}   "
              f"acc={res['accuracy']:>+.4f}   F1={res['macro_f1']:.4f}")

    for tgt in args.targets:
        print(f"\n  [{tgt}]")
        s = summaries[tgt]
        print("    POSITION-LEVEL:")
        show(s["pos_trained"], 1, "trained  linear")
        show(s["pos_trained"], 2, "trained  MLP")
        show(s["pos_untrained"], 1, "untrained linear")
        show(s["pos_untrained"], 2, "untrained MLP")
        if not args.skip_piece_split:
            print("    PIECE-LEVEL (held-out pieces):")
            show(s["piece_trained"], 1, "trained  linear")
            show(s["piece_trained"], 2, "trained  MLP")
            show(s["piece_untrained"], 1, "untrained linear")
            show(s["piece_untrained"], 2, "untrained MLP")

    print(f"\n{'─'*78}\nINTERPRETATION GUIDE\n{'─'*78}")
    print("  Beat probe is the load-bearing Othello-positive prediction:")
    print("    real model:        high accuracy on PIECE-LEVEL split")
    print("    within-shuffled:   should COLLAPSE on PIECE-LEVEL split")
    print("    global-shuffled:   should COLLAPSE to chance (~0.25)")
    print("")
    print("  Mode / chord probes test the cities-analogue prediction:")
    print("    real model:        high acc on PIECE-LEVEL")
    print("    within-shuffled:   should STAY HIGH (set-membership leak)")
    print("    global-shuffled:   should collapse")


if __name__ == "__main__":
    main()
