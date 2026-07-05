"""
Probe maze C3 carry-through.

Locked prediction:
  predictions/predictions_maze_c3_carrythrough.md

Target:
  [BOS, c0=start, c1, c2, c3, c4, ..., goal, EOS]
                         ^
                       source

At later path positions t >= min_probe_index, recover the fourth path cell
c3 from the residual stream. The headline split trains probes on val mazes
and evaluates on gen mazes.

Usage:
  python eval/probe_maze_c3_carrythrough.py \
    --ckpt checkpoints/maze_8x8/best.pt \
    --data_dir data/maze_8x8 \
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

PAD = 0
N_RESERVED = 3


def compact_cell_label(token_id):
    return int(token_id) - N_RESERVED


def load_c3_examples(data_dir: Path, min_probe_index: int):
    """Return examples by split.

    Each example is a dict with split, token_pos, maze_idx, path_index, and
    compact c3 label in 0..63.
    """
    by_maze = defaultdict(list)
    with open(data_dir / "mazes.csv") as f:
        for row in csv.DictReader(f):
            split = row["split"]
            if split not in ("val", "gen"):
                continue
            by_maze[(split, int(row["maze_idx"]))].append(row)

    examples = {"val": [], "gen": []}
    c3_by_maze = {}
    for (split, maze_idx), rows in by_maze.items():
        rows.sort(key=lambda r: int(r["token_pos"]))
        if len(rows) <= min_probe_index:
            continue
        c3_token = int(rows[3]["current_cell"])
        c3_label = compact_cell_label(c3_token)
        c3_by_maze[(split, maze_idx)] = c3_token
        for path_index in range(min_probe_index, len(rows)):
            r = rows[path_index]
            examples[split].append({
                "split": split,
                "maze_idx": maze_idx,
                "token_pos": int(r["token_pos"]),
                "path_index": path_index,
                "label": c3_label,
            })
    return examples, c3_by_maze


def sample_examples(examples, n, seed):
    rng = np.random.default_rng(seed)
    if n is None or n <= 0 or n >= len(examples):
        ix = np.arange(len(examples))
    else:
        ix = rng.choice(len(examples), size=n, replace=False)
    return [examples[int(i)] for i in ix]


def describe_examples(name, examples):
    labels = [ex["label"] for ex in examples]
    paths = [ex["path_index"] for ex in examples]
    counts = Counter(labels)
    maj_label, maj_n = counts.most_common(1)[0]
    print(f"{name}: {len(examples):,} examples")
    print(f"  labels: {len(counts)} classes; majority={maj_label} ({maj_n / len(labels):.3f})")
    print(f"  path_index: min={min(paths)} median={np.median(paths):.1f} max={max(paths)}")


def build_context(stream, token_pos, block_size):
    start = max(0, token_pos - block_size + 1)
    ctx = stream[start:token_pos + 1]
    if len(ctx) < block_size:
        ctx = np.concatenate([np.full(block_size - len(ctx), PAD, dtype=ctx.dtype), ctx])
    return ctx


@torch.no_grad()
def cache_examples(model, stream, examples, block_size, device, batch_size=32):
    n_layer = model.config.n_layer
    X_layers = [[] for _ in range(n_layer + 1)]
    y = np.array([ex["label"] for ex in examples], dtype=np.int64)

    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        contexts = np.stack([build_context(stream, ex["token_pos"], block_size) for ex in batch])
        idx = torch.from_numpy(contexts).long().to(device)
        layer_acts = cache_layer_activations(model, idx)
        for layer, act in enumerate(layer_acts):
            X_layers[layer].append(act[:, -1, :].cpu().numpy())

    X = np.stack([np.concatenate(parts, axis=0) for parts in X_layers], axis=0)
    return X, y


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
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_eval(probe, Xtr, ytr, Xte, yte, device, epochs=30, lr=3e-3):
    probe = probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    Xte_t = torch.from_numpy(Xte).float().to(device)
    yte_t = torch.from_numpy(yte).long().to(device)
    batch = min(256, len(Xtr_t))
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr_t), device=device)
        for start in range(0, len(Xtr_t), batch):
            ix = perm[start:start + batch]
            loss = F.cross_entropy(probe(Xtr_t[ix]), ytr_t[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
    probe.eval()
    with torch.no_grad():
        pred = probe(Xte_t).argmax(dim=-1)
        return (pred == yte_t).float().mean().item()


def run_probe_sweep(X_train, y_train, X_test, y_test, device, epochs, seed, label):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rows = []
    print(f"\n{label}")
    print("  Layer       Lin      MLP")
    for layer in range(X_train.shape[0]):
        Xtr = X_train[layer]
        Xte = X_test[layer]
        in_dim = Xtr.shape[1]
        lin = LinearProbe(in_dim, 64)
        mlp = MLPProbe(in_dim, 64)
        lin_acc = train_eval(lin, Xtr, y_train, Xte, y_test, device, epochs=epochs)
        mlp_acc = train_eval(mlp, Xtr, y_train, Xte, y_test, device, epochs=epochs)
        lab = "embed" if layer == 0 else f"L{layer - 1}"
        print(f"  {lab:<8}  {lin_acc:.4f}   {mlp_acc:.4f}")
        rows.append((lin_acc, mlp_acc))
    return rows


def aggregate(all_rows, n_layers):
    out = {}
    for layer in range(n_layers):
        vals = np.array([rows[layer] for rows in all_rows], dtype=np.float64)
        out[layer] = {
            "lin_mean": float(vals[:, 0].mean()),
            "lin_std": float(vals[:, 0].std(ddof=1)) if len(vals) > 1 else 0.0,
            "mlp_mean": float(vals[:, 1].mean()),
            "mlp_std": float(vals[:, 1].std(ddof=1)) if len(vals) > 1 else 0.0,
        }
    return out


def layer_label(layer):
    return "embed" if layer == 0 else f"L{layer - 1}"


def print_aggregate(trained_rows, untrained_rows, n_layers, n_seeds):
    trained = aggregate(trained_rows, n_layers)
    untrained = aggregate(untrained_rows, n_layers)

    print("\n" + "=" * 78)
    print(f"AGGREGATE - mean +/- std over {n_seeds} seeds")
    print("=" * 78)
    print("Layer     TrLin        TrMLP        UnLin        UnMLP        LinGap   MLPGap")
    best_mlp_gap = (-999.0, None)
    best_lin_gap = (-999.0, None)
    for layer in range(n_layers):
        tr = trained[layer]
        un = untrained[layer]
        lin_gap = tr["lin_mean"] - un["lin_mean"]
        mlp_gap = tr["mlp_mean"] - un["mlp_mean"]
        if mlp_gap > best_mlp_gap[0]:
            best_mlp_gap = (mlp_gap, layer)
        if lin_gap > best_lin_gap[0]:
            best_lin_gap = (lin_gap, layer)
        print(
            f"{layer_label(layer):<8} "
            f"{tr['lin_mean']:.4f}+/-{tr['lin_std']:.4f}  "
            f"{tr['mlp_mean']:.4f}+/-{tr['mlp_std']:.4f}  "
            f"{un['lin_mean']:.4f}+/-{un['lin_std']:.4f}  "
            f"{un['mlp_mean']:.4f}+/-{un['mlp_std']:.4f}  "
            f"{lin_gap:+.4f}  {mlp_gap:+.4f}"
        )

    print("\nHEADLINE")
    print(f"  Best linear gap: {best_lin_gap[0]:+.4f} at {layer_label(best_lin_gap[1])}")
    print(f"  Best MLP gap:    {best_mlp_gap[0]:+.4f} at {layer_label(best_mlp_gap[1])}")
    verdict = (
        "STRONGLY CONFIRMED" if best_mlp_gap[0] >= 0.12 else
        "CONFIRMED" if best_mlp_gap[0] >= 0.08 else
        "AMBIGUOUS" if best_mlp_gap[0] >= 0.04 else
        "NULL / SURPRISING"
    )
    print(f"  Locked-threshold verdict: {verdict}")
    return best_lin_gap, best_mlp_gap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--min_probe_index", type=int, default=9)
    p.add_argument("--n_train", type=int, default=20_000)
    p.add_argument("--n_test", type=int, default=20_000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    print(f"seeds: {args.seeds}")
    print(f"min_probe_index: {args.min_probe_index}")

    data_dir = Path(args.data_dir)
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dtype = np.dtype(meta["dtype"])
    streams = {
        "val": np.asarray(np.memmap(data_dir / "val.bin", dtype=dtype, mode="r")),
        "gen": np.asarray(np.memmap(data_dir / "gen.bin", dtype=dtype, mode="r")),
    }

    all_examples, _ = load_c3_examples(data_dir, args.min_probe_index)
    describe_examples("all val", all_examples["val"])
    describe_examples("all gen", all_examples["gen"])

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    trained = GPT(config).to(device)
    trained.load_state_dict(ckpt["model_state"])
    trained.eval()
    print(f"loaded ckpt: iter={ckpt.get('iter', '?')} val_ppl={ckpt.get('val_perplexity', float('nan')):.4f}")

    trained_rows = []
    untrained_rows = []
    for seed in args.seeds:
        print("\n" + "#" * 78)
        print(f"# SEED {seed}")
        print("#" * 78)
        train_ex = sample_examples(all_examples["val"], args.n_train, seed)
        test_ex = sample_examples(all_examples["gen"], args.n_test, seed + 10_000)
        describe_examples("sampled train(val)", train_ex)
        describe_examples("sampled test(gen)", test_ex)

        print("\nCaching TRAINED activations ...")
        t0 = time.time()
        Xtr_t, ytr = cache_examples(
            trained, streams["val"], train_ex, config.block_size,
            device, batch_size=args.batch_size,
        )
        Xte_t, yte = cache_examples(
            trained, streams["gen"], test_ex, config.block_size,
            device, batch_size=args.batch_size,
        )
        print(f"  trained cache done in {time.time() - t0:.1f}s")

        untrained = GPT(config).to(device).eval()
        print("\nCaching UNTRAINED activations ...")
        t0 = time.time()
        Xtr_u, ytr_u = cache_examples(
            untrained, streams["val"], train_ex, config.block_size,
            device, batch_size=args.batch_size,
        )
        Xte_u, yte_u = cache_examples(
            untrained, streams["gen"], test_ex, config.block_size,
            device, batch_size=args.batch_size,
        )
        print(f"  untrained cache done in {time.time() - t0:.1f}s")
        assert np.array_equal(ytr, ytr_u)
        assert np.array_equal(yte, yte_u)

        trained_rows.append(
            run_probe_sweep(Xtr_t, ytr, Xte_t, yte, device, args.epochs, seed,
                            "TRAINED val -> gen")
        )
        untrained_rows.append(
            run_probe_sweep(Xtr_u, ytr_u, Xte_u, yte_u, device, args.epochs, seed,
                            "UNTRAINED val -> gen")
        )

    print_aggregate(trained_rows, untrained_rows, config.n_layer + 1, len(args.seeds))


if __name__ == "__main__":
    main()
