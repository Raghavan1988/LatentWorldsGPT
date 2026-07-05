#!/usr/bin/env python3
"""Corrupt HTTP path tokens to test local usefulness vs carry-through.

For held-out HTTP requests with layout ``[m_k, p_k, s_k, sz_k]``, this script
replaces ``p_k`` with a different observed path-category token and measures
paired clean/corrupted negative log-likelihood on:

1. ``sz_k``, predicted from context ending at ``s_k``.
2. the token after ``sz_k``, predicted from context ending at ``sz_k``.

This separates "the path feature is useful for predicting the current size
token" from "the path feature remains useful after the size token has already
been observed."

Usage:
    python eval/intervene_http_path_token.py \
        --ckpt checkpoints/http_real/best.pt \
        --data_dir data/nasa_http \
        --natural_n 50000 \
        --per_class_n 1000
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "model"))

from model import GPT, GPTConfig  # noqa: E402

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


def collect_examples(streams, itos):
    """Collect examples as ``(split, path_pos, size_pos, after_pos, path_tok)``."""
    examples = []
    path_counts = Counter()
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
            for req_idx in range(1, n_reqs):
                path_pos = start + 1 + req_idx * 4 + 1
                size_pos = start + 1 + req_idx * 4 + 3
                after_pos = size_pos + 1
                if after_pos > j:
                    continue

                path_tok = int(arr[path_pos])
                size_tok = int(arr[size_pos])
                if not itos[path_tok].startswith("p_") or not itos[size_tok].startswith("sz_"):
                    continue

                examples.append((split, path_pos, size_pos, after_pos, path_tok))
                path_counts[path_tok] += 1
            i = j + 1
    return examples, path_counts


def sample_natural(examples, n: int, seed: int):
    rng = np.random.default_rng(seed)
    if n >= len(examples):
        return list(examples)
    idx = rng.choice(len(examples), size=n, replace=False)
    return [examples[int(i)] for i in idx]


def sample_balanced(examples, per_class: int, seed: int):
    by_path = defaultdict(list)
    for ex in examples:
        by_path[ex[4]].append(ex)
    rng = np.random.default_rng(seed)
    sampled = []
    for tok in sorted(by_path):
        rows = list(by_path[tok])
        rng.shuffle(rows)
        sampled.extend(rows[: min(per_class, len(rows))])
    rng.shuffle(sampled)
    return sampled


def build_context(arr, end_pos: int, block_size: int):
    start = max(0, end_pos - block_size + 1)
    ctx = np.asarray(arr[start: end_pos + 1], dtype=np.int64).copy()
    if len(ctx) < block_size:
        ctx = np.concatenate([np.full(block_size - len(ctx), PAD, dtype=np.int64), ctx])
    return ctx, start, block_size - (end_pos - start + 1)


def corrupt_context(ctx, global_start: int, pad_offset: int, path_pos: int, replacement: int):
    out = ctx.copy()
    local = pad_offset + (path_pos - global_start)
    if not (0 <= local < len(out)):
        raise ValueError("path position missing from context")
    out[local] = replacement
    return out


def choose_replacements(sampled, path_tokens, seed: int):
    rng = np.random.default_rng(seed)
    path_tokens = np.asarray(sorted(path_tokens), dtype=np.int64)
    replacements = []
    for *_, path_tok in sampled:
        choices = path_tokens[path_tokens != path_tok]
        replacements.append(int(rng.choice(choices)))
    return replacements


@torch.no_grad()
def eval_target(model, streams, sampled, replacements, block_size: int, target_kind: str, device, batch_size: int):
    clean_nll = []
    corrupt_nll = []
    clean_acc = []
    corrupt_acc = []
    clean_batch = []
    corrupt_batch = []
    targets = []

    def flush():
        if not clean_batch:
            return
        clean = torch.from_numpy(np.stack(clean_batch)).to(device)
        corrupt = torch.from_numpy(np.stack(corrupt_batch)).to(device)
        y = torch.tensor(targets, dtype=torch.long, device=device)
        clean_logits = model(clean)[0][:, -1, :]
        corrupt_logits = model(corrupt)[0][:, -1, :]
        clean_loss = F.cross_entropy(clean_logits, y, reduction="none")
        corrupt_loss = F.cross_entropy(corrupt_logits, y, reduction="none")
        clean_nll.extend(clean_loss.cpu().numpy().tolist())
        corrupt_nll.extend(corrupt_loss.cpu().numpy().tolist())
        clean_acc.extend((clean_logits.argmax(dim=-1) == y).cpu().numpy().tolist())
        corrupt_acc.extend((corrupt_logits.argmax(dim=-1) == y).cpu().numpy().tolist())
        clean_batch.clear()
        corrupt_batch.clear()
        targets.clear()

    for ex, replacement in zip(sampled, replacements):
        split, path_pos, size_pos, after_pos, _ = ex
        arr = streams[split]
        if target_kind == "size":
            end_pos = size_pos - 1
            target_pos = size_pos
        elif target_kind == "after_size":
            end_pos = size_pos
            target_pos = after_pos
        else:
            raise ValueError(target_kind)

        ctx, global_start, pad_offset = build_context(arr, end_pos, block_size)
        clean_batch.append(ctx)
        corrupt_batch.append(corrupt_context(ctx, global_start, pad_offset, path_pos, replacement))
        targets.append(int(arr[target_pos]))
        if len(clean_batch) >= batch_size:
            flush()
    flush()

    clean_nll = np.asarray(clean_nll)
    corrupt_nll = np.asarray(corrupt_nll)
    clean_acc = np.asarray(clean_acc, dtype=np.float64)
    corrupt_acc = np.asarray(corrupt_acc, dtype=np.float64)
    return {
        "clean_nll": clean_nll,
        "corrupt_nll": corrupt_nll,
        "delta": corrupt_nll - clean_nll,
        "clean_acc": clean_acc,
        "corrupt_acc": corrupt_acc,
    }


def describe(name: str, stats):
    delta = stats["delta"]
    print(f"\n{name}")
    print(f"  clean NLL:   {stats['clean_nll'].mean():.4f}")
    print(f"  corrupt NLL: {stats['corrupt_nll'].mean():.4f}")
    print(f"  delta NLL:   {delta.mean():+.4f} +/- {delta.std(ddof=0):.4f}")
    print(f"  median dNLL: {np.median(delta):+.4f}")
    print(f"  clean acc:   {stats['clean_acc'].mean():.4f}")
    print(f"  corrupt acc: {stats['corrupt_acc'].mean():.4f}")
    print(f"  acc delta:   {stats['corrupt_acc'].mean() - stats['clean_acc'].mean():+.4f}")


def run_condition(name, model, streams, examples, path_tokens, block_size: int, args, sample_mode: str):
    if sample_mode == "natural":
        sampled = sample_natural(examples, args.natural_n, args.seed)
    else:
        sampled = sample_balanced(examples, args.per_class_n, args.seed)

    replacements = choose_replacements(sampled, path_tokens, args.seed + 17)
    counts = Counter(ex[4] for ex in sampled)
    print(f"\n# {name} ({sample_mode})")
    print(f"examples: {len(sampled):,}; path classes: {len(counts)}")
    print(f"path-count min/median/max: {min(counts.values())}/{int(np.median(list(counts.values())))}/{max(counts.values())}")

    size_stats = eval_target(
        model, streams, sampled, replacements, block_size, "size", args.device, args.batch_size,
    )
    after_stats = eval_target(
        model, streams, sampled, replacements, block_size, "after_size", args.device, args.batch_size,
    )
    describe("Predict sz_k from context ending at s_k", size_stats)
    describe("Predict token after sz_k from context ending at sz_k", after_stats)
    print("\nKEY RATIO")
    denom = abs(size_stats["delta"].mean()) + 1e-9
    print(f"  after-size dNLL / size dNLL = {after_stats['delta'].mean() / denom:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/http_real/best.pt")
    parser.add_argument("--data_dir", default="data/nasa_http")
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
    print(f"seed: {args.seed}")

    data_dir = resolve_repo_path(args.data_dir)
    ckpt_path = resolve_repo_path(args.ckpt)
    meta, streams = load_streams(data_dir)
    examples, path_counts = collect_examples(streams, meta["itos"])
    print(f"collected examples: {len(examples):,}")
    print(f"path classes: {len(path_counts)}")
    for tok, count in path_counts.most_common():
        print(f"  {meta['itos'][tok]:<5} {count:>7,} ({count / len(examples):.4f})")

    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    path_tokens = set(path_counts)
    run_condition("HTTP path-token corruption", model, streams, examples, path_tokens, config.block_size, args, "natural")
    run_condition("HTTP path-token corruption", model, streams, examples, path_tokens, config.block_size, args, "balanced")


if __name__ == "__main__":
    main()
