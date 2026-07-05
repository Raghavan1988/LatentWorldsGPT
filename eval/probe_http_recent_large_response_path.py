#!/usr/bin/env python3
"""Probe HTTP recent-large-response path carry-through.

Pre-registration:
    predictions/predictions_http_recent_large_response_path_carrythrough.md

At each current request's size-bin token ``sz_k``, find the most recent
earlier request ``j < k`` whose size bin is large (``size_bin_j >= 5``), and
train probes to recover that earlier request's path-category token ``p_j``.

Usage:
    python eval/probe_http_recent_large_response_path.py \
        --ckpt checkpoints/http_real/best.pt \
        --data_dir data/nasa_http \
        --min_lag 1 \
        --per_class_n 1000 \
        --epochs 8 \
        --seeds 0 1 2 3 4

For the adversarial lag-filtered analysis:
    python eval/probe_http_recent_large_response_path.py --min_lag 3
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "model"))
sys.path.insert(0, str(HERE))

from model import GPT, GPTConfig  # noqa: E402
from probe_http_same_request_path import (  # noqa: E402
    BOS,
    EOS,
    build_dataset,
    class_balanced_sample,
    load_streams,
    resolve_repo_path,
    summarize,
    sweep_layers,
)

LARGE_RESPONSE_BIN_THRESHOLD = 5


def size_bin_from_name(name: str) -> int | None:
    if not name.startswith("sz_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return None


def collect_targets(streams, itos, min_lag: int):
    """Return recent-large-response path labels keyed by current ``sz_k``."""
    raw_targets = {}
    session_of = {}
    distribution = Counter()
    lag_distribution = Counter()

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
            most_recent_large_req = None
            most_recent_large_path_name = None
            for req_idx in range(n_reqs):
                path_pos = start + 1 + req_idx * 4 + 1
                size_pos = start + 1 + req_idx * 4 + 3
                if size_pos >= j:
                    continue

                current_name = itos[int(arr[size_pos])]
                size_bin = size_bin_from_name(current_name)

                if most_recent_large_req is not None:
                    lag = req_idx - most_recent_large_req
                    if lag >= min_lag:
                        key = (split, size_pos)
                        raw_targets[key] = most_recent_large_path_name
                        session_of[key] = f"{split}:{session_idx}"
                        distribution[most_recent_large_path_name] += 1
                        lag_distribution[lag] += 1

                if size_bin is not None and size_bin >= LARGE_RESPONSE_BIN_THRESHOLD:
                    path_name = itos[int(arr[path_pos])]
                    if path_name.startswith("p_"):
                        most_recent_large_req = req_idx
                        most_recent_large_path_name = path_name

            session_idx += 1
            i = j + 1

    label_names = sorted(distribution, key=lambda name: int(name.split("_")[1]))
    label_to_id = {name: i for i, name in enumerate(label_names)}
    targets = {key: label_to_id[name] for key, name in raw_targets.items()}
    return targets, session_of, label_names, distribution, lag_distribution


def print_counter_distribution(title: str, counter: Counter, label_lookup=None, limit: int | None = None):
    total = sum(counter.values())
    print(f"\n{title}: {total:,} examples; {len(counter)} classes")
    rows = counter.most_common(limit)
    for key, count in rows:
        label = label_lookup[key] if label_lookup is not None else key
        print(f"  {str(label):<5} {count:>7,} ({count / total:.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/http_real/best.pt")
    parser.add_argument("--data_dir", default="data/nasa_http")
    parser.add_argument("--min_lag", type=int, default=1)
    parser.add_argument("--per_class_n", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=8)
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
    print(
        f"min_lag: {args.min_lag}; per_class_n: {args.per_class_n}; "
        f"epochs: {args.epochs}; seeds: {args.seeds}"
    )

    meta, streams = load_streams(data_dir)
    targets, session_of, label_names, distribution, lag_distribution = collect_targets(
        streams, meta["itos"], args.min_lag,
    )
    print_counter_distribution("Target distribution before sampling", distribution)
    print_counter_distribution("Lag distribution", lag_distribution, limit=12)

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
    if args.min_lag >= 3:
        if best_mlp_gap >= 0.05:
            print(f"CONFIRMED: lag-filtered MLP gap {best_mlp_gap:+.4f} at {best_mlp_layer} >= +0.05")
        elif best_mlp_gap >= 0.03:
            print(f"WEAK EVIDENCE: lag-filtered MLP gap {best_mlp_gap:+.4f} at {best_mlp_layer} in [+0.03, +0.05)")
        else:
            print(f"NULL: lag-filtered MLP gap {best_mlp_gap:+.4f} at {best_mlp_layer} < +0.03")
    else:
        if best_mlp_gap >= 0.10:
            print(f"STRONGLY CONFIRMED: best-layer MLP gap {best_mlp_gap:+.4f} at {best_mlp_layer} >= +0.10")
        elif best_mlp_gap >= 0.05:
            print(f"CONFIRMED: best-layer MLP gap {best_mlp_gap:+.4f} at {best_mlp_layer} >= +0.05")
        elif best_mlp_gap >= 0.03:
            print(f"AMBIGUOUS: best-layer MLP gap {best_mlp_gap:+.4f} at {best_mlp_layer} in [+0.03, +0.05)")
        else:
            print(f"NULL: best-layer MLP gap {best_mlp_gap:+.4f} at {best_mlp_layer} < +0.03")
    print(f"Linear best-layer gap: {best_lin_gap:+.4f} at {best_lin_layer}")


if __name__ == "__main__":
    main()
