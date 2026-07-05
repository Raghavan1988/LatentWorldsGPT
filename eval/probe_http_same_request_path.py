#!/usr/bin/env python3
"""Probe HTTP same-request path carry-through.

Pre-registration:
    predictions/predictions_http_same_request_path_carrythrough.md

At each request's size-bin token ``sz_k`` for ``k >= 1``, train probes to
recover the same request's path-category token ``p_k`` from residual-stream
activations. The headline metric is trained-vs-untrained probe accuracy gap
under a session-level split.

Usage:
    python eval/probe_http_same_request_path.py \
        --ckpt checkpoints/http_real/best.pt \
        --data_dir data/nasa_http \
        --per_class_n 1000 \
        --epochs 20 \
        --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
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
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "model"))
sys.path.insert(0, str(HERE))

from model import GPT, GPTConfig  # noqa: E402
from probe import cache_layer_activations  # noqa: E402

PAD, BOS, EOS = 0, 1, 2


def resolve_repo_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def load_streams(data_dir: Path):
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dtype = np.dtype(meta["dtype"])
    streams = {
        split: np.asarray(np.memmap(data_dir / f"{split}.bin", dtype=dtype, mode="r"))
        for split in ("val", "gen")
    }
    return meta, streams


def collect_targets(streams, itos):
    """Return labels keyed by ``(split, size_token_position)``."""
    raw_targets: dict[tuple[str, int], str] = {}
    session_of: dict[tuple[str, int], str] = {}
    distribution = Counter()

    for split, arr in streams.items():
        i = 0
        session_idx = 0
        while i < len(arr):
            if int(arr[i]) != BOS:
                i += 1
                continue

            start = i
            j = i + 1
            while j < len(arr) and int(arr[j]) != EOS:
                j += 1
            if j >= len(arr):
                break

            n_reqs = (j - start - 1) // 4
            for req_idx in range(1, n_reqs):
                path_pos = start + 1 + req_idx * 4 + 1
                size_pos = start + 1 + req_idx * 4 + 3
                if size_pos >= j:
                    continue

                label_name = itos[int(arr[path_pos])]
                current_name = itos[int(arr[size_pos])]
                if not label_name.startswith("p_") or not current_name.startswith("sz_"):
                    continue

                key = (split, size_pos)
                raw_targets[key] = label_name
                session_of[key] = f"{split}:{session_idx}"
                distribution[label_name] += 1

            session_idx += 1
            i = j + 1

    label_names = sorted(distribution, key=lambda name: int(name.split("_")[1]))
    label_to_id = {name: i for i, name in enumerate(label_names)}
    targets = {key: label_to_id[name] for key, name in raw_targets.items()}
    return targets, session_of, label_names, distribution


def class_balanced_sample(targets, per_class_n: int, seed: int):
    by_class = defaultdict(list)
    for key, label in targets.items():
        by_class[label].append(key)

    rng = np.random.default_rng(seed)
    sampled = []
    for label in sorted(by_class):
        keys = list(by_class[label])
        rng.shuffle(keys)
        sampled.extend(keys[: min(per_class_n, len(keys))])
    rng.shuffle(sampled)
    return sampled


@torch.no_grad()
def build_dataset(model, streams, sampled_keys, targets, session_of, block_size, device):
    model.eval()
    split_order = list(streams.keys())
    offsets = {}
    parts = []
    cursor = 0
    for split in split_order:
        offsets[split] = cursor
        parts.append(streams[split])
        cursor += len(streams[split])
    combined = np.concatenate(parts).astype(np.int64)

    n_layer = model.config.n_layer
    x_layers = [[] for _ in range(n_layer + 1)]
    y_list = []
    session_list = []

    batch_size = 64
    for batch_start in range(0, len(sampled_keys), batch_size):
        batch_keys = sampled_keys[batch_start: batch_start + batch_size]
        batch_idxs = []
        for split, pos in batch_keys:
            global_pos = offsets[split] + pos
            ctx_start = max(0, global_pos - block_size + 1)
            ctx = combined[ctx_start: global_pos + 1]
            if len(ctx) < block_size:
                ctx = np.concatenate([np.full(block_size - len(ctx), PAD), ctx])
            batch_idxs.append(ctx)

        idx = torch.from_numpy(np.stack(batch_idxs)).to(device)
        acts = cache_layer_activations(model, idx)
        for layer, act in enumerate(acts):
            x_layers[layer].append(act[:, -1, :].cpu().numpy())
        for key in batch_keys:
            y_list.append(targets[key])
            session_list.append(session_of[key])

    x = np.stack([np.concatenate(parts_for_layer, axis=0) for parts_for_layer in x_layers], axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    return x, y, np.asarray(session_list, dtype=object)


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.lin(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_eval(probe, x_train, y_train, x_test, y_test, device, epochs: int, lr: float = 3e-3):
    probe = probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    xtr = torch.from_numpy(x_train).float().to(device)
    ytr = torch.from_numpy(y_train).long().to(device)
    xte = torch.from_numpy(x_test).float().to(device)
    yte = torch.from_numpy(y_test).long().to(device)
    batch = min(256, len(xtr))

    for _ in range(epochs):
        order = torch.randperm(len(xtr), device=device)
        for start in range(0, len(xtr), batch):
            idx = order[start: start + batch]
            loss = F.cross_entropy(probe(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(xte).argmax(dim=-1)
        return (pred == yte).float().mean().item()


def session_split(session_ids, train_frac: float, seed: int):
    unique = np.unique(session_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique)
    n_train = int(train_frac * len(unique))
    train_sessions = set(perm[:n_train].tolist())
    train_idx = np.asarray([i for i, sid in enumerate(session_ids) if sid in train_sessions])
    test_idx = np.asarray([i for i, sid in enumerate(session_ids) if sid not in train_sessions])
    return train_idx, test_idx


def sweep_layers(x, y, session_ids, n_classes: int, device, epochs: int, seed: int, label: str):
    train_idx, test_idx = session_split(session_ids, 0.8, seed)
    print(f"\n{label}: train={len(train_idx):,} test={len(test_idx):,} sessions={len(np.unique(session_ids)):,}")
    rows = []
    for layer in range(x.shape[0]):
        xtr, ytr = x[layer, train_idx], y[train_idx]
        xte, yte = x[layer, test_idx], y[test_idx]
        lin = train_eval(LinearProbe(xtr.shape[1], n_classes), xtr, ytr, xte, yte, device, epochs)
        mlp = train_eval(MLPProbe(xtr.shape[1], n_classes), xtr, ytr, xte, yte, device, epochs)
        layer_name = "embed" if layer == 0 else f"L{layer - 1}"
        print(f"  {layer_name:<5} lin={lin:.4f} mlp={mlp:.4f}")
        rows.append((lin, mlp))
    return np.asarray(rows)


def summarize(results, metric_idx: int, metric_name: str):
    trained = np.stack([result["trained"][:, metric_idx] for result in results])
    untrained = np.stack([result["untrained"][:, metric_idx] for result in results])
    gap = trained - untrained
    labels = ["embed"] + [f"L{i}" for i in range(trained.shape[1] - 1)]

    print(f"\nSUMMARY {metric_name} mean +/- std over {len(results)} seeds")
    print("layer    trained          untrained        gap")
    for i, label in enumerate(labels):
        print(
            f"{label:<7} "
            f"{trained[:, i].mean():.4f} +/- {trained[:, i].std(ddof=0):.4f}   "
            f"{untrained[:, i].mean():.4f} +/- {untrained[:, i].std(ddof=0):.4f}   "
            f"{gap[:, i].mean():+.4f} +/- {gap[:, i].std(ddof=0):.4f}"
        )

    best = int(np.argmax(gap.mean(axis=0)))
    print(f"BEST {metric_name} gap: {labels[best]} {gap[:, best].mean():+.4f}")
    return labels[best], float(gap[:, best].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/http_real/best.pt")
    parser.add_argument("--data_dir", default="data/nasa_http")
    parser.add_argument("--per_class_n", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ckpt_path = resolve_repo_path(args.ckpt)
    data_dir = resolve_repo_path(args.data_dir)
    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")
    print(f"per_class_n: {args.per_class_n}; epochs: {args.epochs}; seeds: {args.seeds}")

    meta, streams = load_streams(data_dir)
    targets, session_of, label_names, distribution = collect_targets(streams, meta["itos"])
    total = sum(distribution.values())
    print(f"\nTarget distribution before sampling: {total:,} examples; {len(label_names)} classes")
    for name in label_names:
        print(f"  {name:<5} {distribution[name]:>7,} ({distribution[name] / total:.4f})")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    trained = GPT(config).to(device)
    trained.load_state_dict(ckpt["model_state"])
    trained.eval()

    all_results = []
    for seed in args.seeds:
        print(f"\n# seed {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        sampled = class_balanced_sample(targets, args.per_class_n, seed)
        sampled_counts = Counter(targets[key] for key in sampled)
        print(f"sampled: {len(sampled):,} examples")
        print("sampled counts:", {label_names[k]: sampled_counts[k] for k in sorted(sampled_counts)})

        t0 = time.time()
        x_trained, y, session_ids = build_dataset(
            trained, streams, sampled, targets, session_of, config.block_size, device,
        )
        print(f"built trained activations: {x_trained.shape} in {time.time() - t0:.1f}s")

        untrained = GPT(config).to(device).eval()
        t0 = time.time()
        x_untrained, y_untrained, session_ids_untrained = build_dataset(
            untrained, streams, sampled, targets, session_of, config.block_size, device,
        )
        print(f"built untrained activations: {x_untrained.shape} in {time.time() - t0:.1f}s")
        assert np.array_equal(y, y_untrained)
        assert np.array_equal(session_ids, session_ids_untrained)

        trained_rows = sweep_layers(
            x_trained, y, session_ids, len(label_names), device, args.epochs, seed, "TRAINED",
        )
        untrained_rows = sweep_layers(
            x_untrained, y_untrained, session_ids_untrained, len(label_names), device, args.epochs, seed, "UNTRAINED",
        )
        all_results.append({"trained": trained_rows, "untrained": untrained_rows})

    best_lin_layer, best_lin_gap = summarize(all_results, 0, "linear")
    best_mlp_layer, best_mlp_gap = summarize(all_results, 1, "MLP")

    print("\nVERDICT")
    if best_mlp_gap >= 0.10:
        print(f"CONFIRMED: best-layer MLP gap {best_mlp_gap:+.4f} at {best_mlp_layer} >= +0.10")
    elif best_mlp_gap >= 0.05:
        print(f"AMBIGUOUS: best-layer MLP gap {best_mlp_gap:+.4f} at {best_mlp_layer} in [+0.05, +0.10)")
    else:
        print(f"FALSIFIED/SURPRISING NULL: best-layer MLP gap {best_mlp_gap:+.4f} at {best_mlp_layer} < +0.05")
    print(f"Linear best-layer gap: {best_lin_gap:+.4f} at {best_lin_layer}")


if __name__ == "__main__":
    main()
