#!/usr/bin/env python3
"""Corrupt recent-large-response path tokens and measure usefulness.

For each current request ``k``, find the most recent earlier request ``j < k``
whose size bin is large (``size_bin_j >= 5``). Replace ``p_j`` with a different
observed path-category token and measure paired clean/corrupted NLL on:

1. ``sz_k``, predicted from context ending at ``s_k``.
2. the token after ``sz_k``, predicted from context ending at ``sz_k``.

Usage:
    python eval/intervene_http_recent_large_response_path.py \
        --ckpt checkpoints/http_real/best.pt \
        --data_dir data/nasa_http \
        --min_lag 1 \
        --natural_n 50000 \
        --per_class_n 1000
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "model"))
sys.path.insert(0, str(HERE))

from intervene_http_path_token import (  # noqa: E402
    BOS,
    EOS,
    choose_replacements,
    describe,
    eval_target,
    load_streams,
    resolve_repo_path,
    sample_balanced,
    sample_natural,
)
from model import GPT, GPTConfig  # noqa: E402
from probe_http_recent_large_response_path import (  # noqa: E402
    LARGE_RESPONSE_BIN_THRESHOLD,
    size_bin_from_name,
)


def collect_examples(streams, itos, min_lag: int):
    """Collect examples as ``(split, source_path_pos, size_pos, after_pos, source_path_tok)``."""
    examples = []
    path_counts = Counter()
    lag_counts = Counter()

    for split, arr in streams.items():
        i = 0
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
            most_recent_large_path_pos = None
            most_recent_large_path_tok = None
            for req_idx in range(n_reqs):
                path_pos = start + 1 + req_idx * 4 + 1
                size_pos = start + 1 + req_idx * 4 + 3
                after_pos = size_pos + 1
                if after_pos > j:
                    continue

                current_size_name = itos[int(arr[size_pos])]
                size_bin = size_bin_from_name(current_size_name)

                if most_recent_large_req is not None:
                    lag = req_idx - most_recent_large_req
                    if lag >= min_lag:
                        examples.append(
                            (
                                split,
                                most_recent_large_path_pos,
                                size_pos,
                                after_pos,
                                most_recent_large_path_tok,
                            )
                        )
                        path_counts[most_recent_large_path_tok] += 1
                        lag_counts[lag] += 1

                if size_bin is not None and size_bin >= LARGE_RESPONSE_BIN_THRESHOLD:
                    path_name = itos[int(arr[path_pos])]
                    if path_name.startswith("p_"):
                        most_recent_large_req = req_idx
                        most_recent_large_path_pos = path_pos
                        most_recent_large_path_tok = int(arr[path_pos])

            i = j + 1
    return examples, path_counts, lag_counts


def run_condition(name, model, streams, examples, path_tokens, block_size: int, args, sample_mode: str):
    if sample_mode == "natural":
        sampled = sample_natural(examples, args.natural_n, args.seed)
    else:
        sampled = sample_balanced(examples, args.per_class_n, args.seed)

    replacements = choose_replacements(sampled, path_tokens, args.seed + 17)
    counts = Counter(ex[4] for ex in sampled)
    count_values = sorted(counts.values())
    median_count = count_values[len(count_values) // 2]
    print(f"\n# {name} ({sample_mode})")
    print(f"examples: {len(sampled):,}; path classes: {len(counts)}")
    print(f"path-count min/median/max: {min(count_values)}/{median_count}/{max(count_values)}")

    size_stats = eval_target(
        model, streams, sampled, replacements, block_size, "size", args.device, args.batch_size,
    )
    after_stats = eval_target(
        model, streams, sampled, replacements, block_size, "after_size", args.device, args.batch_size,
    )
    describe("Predict sz_k after corrupting recent-large p_j", size_stats)
    describe("Predict token after sz_k after corrupting recent-large p_j", after_stats)
    print("\nKEY RATIO")
    denom = abs(size_stats["delta"].mean()) + 1e-9
    print(f"  after-size dNLL / size dNLL = {after_stats['delta'].mean() / denom:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/http_real/best.pt")
    parser.add_argument("--data_dir", default="data/nasa_http")
    parser.add_argument("--min_lag", type=int, default=1)
    parser.add_argument("--natural_n", type=int, default=50_000)
    parser.add_argument("--per_class_n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    args.device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {args.device}")
    print(f"seed: {args.seed}; min_lag: {args.min_lag}")

    data_dir = resolve_repo_path(args.data_dir)
    ckpt_path = resolve_repo_path(args.ckpt)
    meta, streams = load_streams(data_dir)
    examples, path_counts, lag_counts = collect_examples(streams, meta["itos"], args.min_lag)
    print(f"collected examples: {len(examples):,}")
    print(f"path classes: {len(path_counts)}")
    for tok, count in path_counts.most_common():
        print(f"  {meta['itos'][tok]:<5} {count:>7,} ({count / len(examples):.4f})")
    print("lag distribution:", dict(lag_counts.most_common(12)))

    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    path_tokens = set(path_counts)
    run_condition(
        "HTTP recent-large-response path corruption",
        model,
        streams,
        examples,
        path_tokens,
        config.block_size,
        args,
        "natural",
    )
    run_condition(
        "HTTP recent-large-response path corruption",
        model,
        streams,
        examples,
        path_tokens,
        config.block_size,
        args,
        "balanced",
    )


if __name__ == "__main__":
    main()
