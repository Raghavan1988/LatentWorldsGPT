#!/usr/bin/env python3
"""Probe HTTP previous-request path carry-through.

Pre-registration:
    predictions/predictions_http_previous_request_path_carrythrough.md

At each current request's size-bin token ``sz_k`` for ``k >= 1``, train probes
to recover the previous request's path-category token ``p_{k-1}`` from
residual-stream activations.

Usage:
    python eval/probe_http_previous_request_path.py \
        --ckpt checkpoints/http_real/best.pt \
        --data_dir data/nasa_http \
        --per_class_n 1000 \
        --epochs 20 \
        --seeds 0 1 2 3 4
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


def collect_targets(streams, itos):
    """Return previous-request path labels keyed by current ``sz_k`` position."""
    raw_targets = {}
    session_of = {}
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
                prev_path_pos = start + 1 + (req_idx - 1) * 4 + 1
                size_pos = start + 1 + req_idx * 4 + 3
                if size_pos >= j:
                    continue

                label_name = itos[int(arr[prev_path_pos])]
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
