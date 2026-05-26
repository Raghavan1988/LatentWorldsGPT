"""
LatentWorldsGPT — Milestone 1 (sym-group) probe.

Methodology calibration: probe the trained model's residual stream for the
partial product of the word seen so far. The target is a permutation of n
elements; we factor it into n separate "where does position i go?"
classifiers (each is n-class) and report mean accuracy across the n
element-probes.

Pivot.md acceptance: node-level (here: word-level) linear-probe mean
accuracy > 0.9 for n ≤ 8 on the real model; destroyed-structure control
drops to chance. If the real model passes and the shuffled model fails,
the probe code is sound and any null result in other domains (music's
beat probe) reflects the domain, not the probe.

THE ONE RULE
============
This file is the only place permutation values enter the picture. The
model takes only LongTensor generator-token IDs; permutation values live
in partial_product.csv side table.

USAGE
=====
    python eval/probe_symgroup.py --ckpt checkpoints/symgroup_s8/best.pt \
        --data_dir data/symgroup_s8
"""

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig  # noqa: E402
from probe import cache_layer_activations  # noqa: E402

PAD, BOS, EOS = 0, 1, 2
N_RESERVED = 3


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load partial-product targets
# ─────────────────────────────────────────────────────────────────────────────

def load_targets(data_dir: Path, splits=("val", "gen")):
    """Returns {(split, token_pos): (word_idx, perm_list)}.

    perm_list is a list[int] of length n.
    """
    targets = {}
    with open(data_dir / "partial_product.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["split"] not in splits:
                continue
            perm = [int(x) for x in row["perm"].split("-")]
            targets[(row["split"], int(row["token_pos"]))] = (
                int(row["word_idx"]), perm,
            )
    return targets


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build probe dataset
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_probe_dataset(model, streams, targets, block_size, n_positions, n,
                        device, rng_seed=0):
    """Sample positions across val+gen streams; collect activations + targets."""
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

    def to_split_pos(global_idx):
        for s in reversed(split_order):
            if global_idx >= offsets[s]:
                return s, global_idx - offsets[s]
        return None, None

    batch_size = 32
    all_X = None
    all_perms = []   # list of perm lists
    all_words = []
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
                if tok < N_RESERVED:
                    continue  # PAD/BOS/EOS — partial-product undefined for boundary
                key = (split, pos_in_split)
                if key not in targets:
                    continue
                word_idx, perm = targets[key]
                for L in range(len(acts)):
                    all_X[L].append(acts[L][b, t].cpu().numpy())
                all_perms.append(perm)
                all_words.append(word_idx)
                n_collected += 1
                if n_collected >= n_positions:
                    break
            if n_collected >= n_positions:
                break

    X = [np.stack(layer_list) for layer_list in all_X]
    # Build a (N, n) integer target matrix: row k is the perm
    perms_mat = np.array(all_perms, dtype=np.int64)
    words = np.array(all_words, dtype=np.int64)
    return X, perms_mat, words


# ─────────────────────────────────────────────────────────────────────────────
# 3. Probes
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


def train_eval(probe, Xtr, ytr, Xte, yte, n, device, lr=1e-3, wd=1e-3,
               epochs=100, batch_size=512):
    probe = probe.to(device)
    Xtr = torch.from_numpy(Xtr).float().to(device)
    ytr = torch.from_numpy(ytr).long().to(device)
    Xte = torch.from_numpy(Xte).float().to(device)
    yte = torch.from_numpy(yte).long().to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    nrows = Xtr.shape[0]
    best_acc = -1.0
    best_state = None
    for ep in range(epochs):
        perm = torch.randperm(nrows, device=device)
        probe.train()
        for i in range(0, nrows, batch_size):
            ix = perm[i : i + batch_size]
            loss = F.cross_entropy(probe(Xtr[ix]), ytr[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            pred = probe(Xte).argmax(dim=-1)
            acc = (pred == yte).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state); probe.eval()
    return best_acc


def position_split(n_rows, train_frac, seed):
    perm = np.random.default_rng(seed).permutation(n_rows)
    n_train = int(n_rows * train_frac)
    return perm[:n_train], perm[n_train:]


def word_split(words, train_frac, seed):
    uniq = np.unique(words)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_train = int(len(uniq) * train_frac)
    train_words = set(int(w) for w in uniq[perm[:n_train]])
    train_mask = np.array([int(w) in train_words for w in words])
    return np.nonzero(train_mask)[0], np.nonzero(~train_mask)[0]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-element probe sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_element_probes(X_layers, perms_mat, n, train_ix, test_ix, epochs,
                       device, label):
    """For each element index i in [0, n), train a probe to predict
    perms_mat[:, i] from activations. Report per-layer mean accuracy
    across the n element-probes.
    """
    print(f"\n{'─'*72}\n{label}\n{'─'*72}")
    print(f"{'Layer':<8}{'LinAcc(mean)':>16}{'MLPAcc(mean)':>16}{'BestLin':>10}{'BestMLP':>10}")
    print("─" * 60)
    rows = []
    for L, Xl in enumerate(X_layers):
        lin_accs, mlp_accs = [], []
        for i in range(n):
            yi = perms_mat[:, i]
            Xtr = Xl[train_ix]; ytr = yi[train_ix]
            Xte = Xl[test_ix];  yte = yi[test_ix]
            a_lin = train_eval(LinearClassifier(Xl.shape[1], n),
                               Xtr, ytr, Xte, yte, n, device,
                               lr=1e-3, wd=1e-3, epochs=epochs)
            a_mlp = train_eval(MLPClassifier(Xl.shape[1], n),
                               Xtr, ytr, Xte, yte, n, device,
                               lr=1e-3, wd=1e-5, epochs=epochs)
            lin_accs.append(a_lin); mlp_accs.append(a_mlp)
        lab = "embed" if L == 0 else f"L{L}"
        lin_mean = float(np.mean(lin_accs))
        mlp_mean = float(np.mean(mlp_accs))
        print(f"{lab:<8}{lin_mean:>16.4f}{mlp_mean:>16.4f}"
              f"{max(lin_accs):>10.4f}{max(mlp_accs):>10.4f}")
        rows.append((L, lin_mean, mlp_mean, max(lin_accs), max(mlp_accs)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--n_positions", type=int, default=20_000)
    p.add_argument("--probe_train_frac", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_untrained", action="store_true")
    p.add_argument("--skip_word_split", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() \
             else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    data_dir = Path(args.data_dir)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    trained_model = GPT(config).to(device); trained_model.eval()
    untrained_model = None if args.skip_untrained else GPT(config).to(device).eval()
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    n = meta["n"]
    print(f"  iter={ckpt.get('iter','?')}  val_ppl={ckpt.get('val_perplexity', float('nan')):.4f}  n={n}")

    targets = load_targets(data_dir, splits=("val", "gen"))
    print(f"  {len(targets):,} labeled positions across val+gen")
    dtype = np.dtype(meta["dtype"])
    streams = {
        s: np.asarray(np.memmap(data_dir / f"{s}.bin", dtype=dtype, mode="r"))
        for s in ("val", "gen")
    }

    def build(model, lab):
        print(f"\nBuilding probe dataset for {lab} ...")
        t0 = time.time()
        X, perms, words = build_probe_dataset(
            model, streams, targets, config.block_size, args.n_positions,
            n, device, rng_seed=args.seed,
        )
        print(f"  {len(perms):,} positions across {len(X)} layers ({time.time()-t0:.1f}s)")
        return X, perms, words

    X_t, perms_t, words_t = build(trained_model, "TRAINED")
    if untrained_model is not None:
        X_u, perms_u, words_u = build(untrained_model, "UNTRAINED")

    pos_train_t, pos_test_t = position_split(len(perms_t), args.probe_train_frac, args.seed)
    word_train_t = word_test_t = None
    if not args.skip_word_split:
        word_train_t, word_test_t = word_split(words_t, args.probe_train_frac, args.seed)
        print(f"\nPOSITION-LEVEL split: train={len(pos_train_t):,}  test={len(pos_test_t):,}")
        print(f"WORD-LEVEL split:     train={len(word_train_t):,}  test={len(word_test_t):,}")
    if untrained_model is not None:
        pos_train_u, pos_test_u = position_split(len(perms_u), args.probe_train_frac, args.seed)
        if not args.skip_word_split:
            word_train_u, word_test_u = word_split(words_u, args.probe_train_frac, args.seed)

    print(f"\nChance baseline: 1/{n} = {1/n:.4f}")

    summary = {}
    summary["pos_trained"] = run_element_probes(
        X_t, perms_t, n, pos_train_t, pos_test_t, args.epochs, device,
        "TRAINED — POSITION-LEVEL",
    )
    if not args.skip_word_split:
        summary["word_trained"] = run_element_probes(
            X_t, perms_t, n, word_train_t, word_test_t, args.epochs, device,
            "TRAINED — WORD-LEVEL (held-out words; capacity test)",
        )
    if untrained_model is not None:
        summary["pos_untrained"] = run_element_probes(
            X_u, perms_u, n, pos_train_u, pos_test_u, args.epochs, device,
            "UNTRAINED — POSITION-LEVEL (random-init control)",
        )
        if not args.skip_word_split:
            summary["word_untrained"] = run_element_probes(
                X_u, perms_u, n, word_train_u, word_test_u, args.epochs, device,
                "UNTRAINED — WORD-LEVEL",
            )

    print(f"\n{'═'*72}\nHEADLINE\n{'═'*72}")
    def best(rows, ix):
        if rows is None or not rows: return None
        return max(rows, key=lambda r: r[ix])
    def show(rows, ix, lab):
        b = best(rows, ix)
        if b is None: print(f"  {lab:<32}    —"); return
        L, lin_m, mlp_m, lin_max, mlp_max = b
        layer = "embed" if L == 0 else f"L{L}"
        if ix == 1: val, max_ = lin_m, lin_max
        else:       val, max_ = mlp_m, mlp_max
        print(f"  {lab:<32} {layer:>5}  mean_acc={val:.4f}  best_element={max_:.4f}")
    print("  POSITION-LEVEL:")
    show(summary.get("pos_trained"), 1, "trained linear (mean)")
    show(summary.get("pos_trained"), 2, "trained MLP (mean)")
    show(summary.get("pos_untrained"), 1, "untrained linear (mean)")
    show(summary.get("pos_untrained"), 2, "untrained MLP (mean)")
    if not args.skip_word_split:
        print("  WORD-LEVEL (held-out words):")
        show(summary.get("word_trained"), 1, "trained linear (mean)")
        show(summary.get("word_trained"), 2, "trained MLP (mean)")
        show(summary.get("word_untrained"), 1, "untrained linear (mean)")
        show(summary.get("word_untrained"), 2, "untrained MLP (mean)")

    print(f"\n{'─'*72}\nACCEPTANCE (pivot.md M1)\n{'─'*72}")
    b_word_lin = best(summary.get("word_trained"), 1) if not args.skip_word_split else None
    if b_word_lin and b_word_lin[1] > 0.9:
        print(f"  ✓ Word-level mean linear probe > 0.9 ({b_word_lin[1]:.3f}) at "
              f"{'embed' if b_word_lin[0] == 0 else f'L{b_word_lin[0]}'}")
        print(f"    → probe code is sound; any null result in other domains "
              f"reflects the domain, not the probe.")
    else:
        b_lin = best(summary.get("pos_trained"), 1)
        achieved = b_lin[1] if b_lin else float('nan')
        print(f"  ✗ Mean linear probe accuracy {achieved:.3f} (best layer) "
              f"did NOT reach the 0.9 threshold.")
        print(f"    → investigate: probe code, training schedule, or target choice.")


if __name__ == "__main__":
    main()
