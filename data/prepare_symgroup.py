"""
LatentWorldsGPT — Milestone 1 (sym-group) dataset preparation.

Synthetic generator for words in the symmetric group Sₙ. Each "word" is a
sequence of adjacent transposition generators g_1, ..., g_{n-1} where g_k
swaps positions k-1 and k. The model's task is next-token prediction over
these generators; the probe target at each position is the **partial
product** — the permutation accumulated from the start of the word up
through (and including) the current generator.

WHY THIS DOMAIN
================
Sym-group is the guaranteed-positive methodology calibration. The probe
target (a permutation of n elements) is non-trivial to recover from raw
tokens (you have to actually compose generators step by step), but is
recoverable in principle — there is no co-occurrence shortcut. If
`eval/probe_*.py` correctly recovers the partial-product target here,
the probe code is sound; if it can't, there's a bug to fix before
trusting the cities or music probes.

THE ONE RULE (sym-group edition)
=================================
The model's input is a stream of GENERATOR token IDs. The probe target
(the partial product) lives only in the side table partial_product.csv
and is read only by the probe code. Token streams never contain
permutation indices.

Outputs (all in --out_dir):
    train.bin                  uint16 token stream (80% of words)
    val.bin                    uint16 (10% — in-distribution validation)
    gen.bin                    uint16 (10% — held-out test words)
    meta.pkl                   {vocab_size, stoi, itos, dtype, ...}
    partial_product.csv        split,word_idx,token_pos,partial_perm (csv-encoded list)

Token convention: 0=PAD, 1=BOS, 2=EOS; real generators start at 3.
Sequence layout per word: [BOS, g_{i1}, g_{i2}, ..., g_{iL}, EOS]
"""

import argparse
import csv
import pickle
import random
from pathlib import Path

import numpy as np

PAD, BOS, EOS = 0, 1, 2
N_RESERVED = 3


# ─────────────────────────────────────────────────────────────────────────────
# 1. Generators and permutation arithmetic
# ─────────────────────────────────────────────────────────────────────────────

def identity(n: int) -> list[int]:
    return list(range(n))


def apply_generator(perm: list[int], gen_idx: int) -> list[int]:
    """Apply adjacent-transposition generator g_k to `perm`.

    g_k swaps positions k-1 and k. gen_idx ranges from 1 to n-1.
    Returns a new list; does not mutate input.
    """
    i = gen_idx - 1
    out = list(perm)
    out[i], out[i + 1] = out[i + 1], out[i]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Word sampling + encoding
# ─────────────────────────────────────────────────────────────────────────────

def sample_word(n: int, length: int, rng: random.Random) -> list[int]:
    """Sample a random word of `length` generators, each uniform in [1, n-1].

    Note: uniform random words give next-token uniform-over-generators, so the
    next-token loss converges to log(n-1) and the model has no signal to learn
    the partial product. Use sample_self_avoiding_word() for the Othello-GPT
    analog: each next-generator must avoid revisiting an earlier state, so
    legal moves depend on the partial product the model must track.
    """
    return [rng.randint(1, n - 1) for _ in range(length)]


def sample_self_avoiding_word(n: int, length: int,
                              rng: random.Random) -> tuple[list[int], list[list[int]]]:
    """Sample a self-avoiding walk of `length` generators starting from
    identity on the Cayley graph of Sₙ. At each step, the next generator is
    chosen uniformly among the ones whose result has NOT been visited yet.

    Returns (word, perms) where perms[k] is the partial product after k
    generators (perms[0] = identity, perms[len(word)] = final state).

    If no legal extension exists before reaching `length`, the walk ends early
    and a shorter word is returned. For S_8 with length ≤ 32 this is rare
    (degree 7, 40320 vertices).
    """
    cur = identity(n)
    visited = {tuple(cur)}
    word = []
    perms = [list(cur)]
    for _ in range(length):
        candidates = []
        for g in range(1, n):
            nxt = apply_generator(cur, g)
            if tuple(nxt) not in visited:
                candidates.append((g, nxt))
        if not candidates:
            break
        g, nxt = candidates[rng.randrange(len(candidates))]
        visited.add(tuple(nxt))
        word.append(g)
        cur = nxt
        perms.append(list(cur))
    return word, perms


def encode_word(word: list[int], n: int, stoi: dict) -> tuple[list[int], list[list[int]]]:
    """Encode a word into (tokens, perms_per_position).

    tokens: [BOS, stoi[g_1], ..., stoi[g_L], EOS]
    perms[k]: partial product after applying word[:k]
              perms[0] = identity (corresponds to BOS)
              perms[L] = full product (corresponds to last real token)
              perms[L+1] = perms[L] (corresponds to EOS)
    """
    tokens = [BOS]
    perms = [identity(n)]
    cur = identity(n)
    for g in word:
        cur = apply_generator(cur, g)
        tokens.append(stoi[g])
        perms.append(cur)
    tokens.append(EOS)
    perms.append(cur)
    return tokens, perms


def build_tokenizer(n: int):
    """Tokenizer: PAD/BOS/EOS reserved, generators 1..n-1 mapped to N_RESERVED..."""
    stoi = {g: g - 1 + N_RESERVED for g in range(1, n)}
    itos = {idx: g for g, idx in stoi.items()}
    vocab_size = (n - 1) + N_RESERVED
    return stoi, itos, vocab_size


# ─────────────────────────────────────────────────────────────────────────────
# 3. Destroyed-structure controls
# ─────────────────────────────────────────────────────────────────────────────

def shuffle_within_word(tokens: list[int], rng: random.Random) -> list[int]:
    """Weak control: shuffle generator tokens within each [BOS..EOS] span.

    Preserves per-word generator multiset; destroys ordered composition. A
    probe of the partial product should COLLAPSE under this shuffle (since
    permutation composition is non-commutative; the same multiset of
    generators produces different products depending on order).
    """
    out = []
    i = 0
    while i < len(tokens):
        if tokens[i] == BOS:
            j = i + 1
            while j < len(tokens) and tokens[j] != EOS:
                j += 1
            interior = list(tokens[i + 1 : j])
            rng.shuffle(interior)
            out.append(BOS)
            out.extend(interior)
            if j < len(tokens):
                out.append(EOS)
            i = j + 1
        else:
            out.append(tokens[i])
            i += 1
    return out


def shuffle_globally(tokens: list[int], rng: random.Random) -> list[int]:
    """Strict control: shuffle real-generator tokens across the entire stream;
    BOS/EOS positions remain.
    """
    real_positions = [i for i, t in enumerate(tokens) if t >= N_RESERVED]
    real_values = [tokens[i] for i in real_positions]
    rng.shuffle(real_values)
    out = list(tokens)
    for pos, val in zip(real_positions, real_values):
        out[pos] = val
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Split + dump
# ─────────────────────────────────────────────────────────────────────────────

def split_words(n_words: int, val_frac: float, gen_frac: float,
                rng: random.Random) -> tuple[set[int], set[int], set[int]]:
    indices = list(range(n_words))
    rng.shuffle(indices)
    n_gen = int(n_words * gen_frac)
    n_val = int(n_words * val_frac)
    gen = set(indices[:n_gen])
    val = set(indices[n_gen : n_gen + n_val])
    train = set(indices[n_gen + n_val :])
    return train, val, gen


def pick_dtype(vocab_size):
    return np.uint16 if vocab_size < 2**16 else np.uint32


def dump(out_dir, splits, stoi, itos, vocab_size, n):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = pick_dtype(vocab_size)

    for name, payload in splits.items():
        arr = np.array(payload["tokens"], dtype=dtype)
        if arr.size:
            assert int(arr.min()) >= 0 and int(arr.max()) < vocab_size
        arr.tofile(out / f"{name}.bin")

    with open(out / "meta.pkl", "wb") as f:
        pickle.dump({
            "vocab_size": vocab_size, "stoi": stoi, "itos": itos,
            "dtype": np.dtype(dtype).name,
            "pad": PAD, "bos": BOS, "eos": EOS,
            "n": n,
        }, f)

    # Probe target side table: partial permutation at every real-token position.
    with open(out / "partial_product.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "word_idx", "token_pos", "perm"])
        for name, payload in splits.items():
            for word_idx, start, end, perms in payload["word_starts"]:
                # perms has len = (end - start) = 1 BOS + L word + 1 EOS
                for local_pos in range(end - start):
                    global_pos = start + local_pos
                    perm = perms[local_pos]
                    # write perm as a hyphen-joined string for compactness
                    w.writerow([name, word_idx, global_pos,
                                "-".join(str(x) for x in perm)])
    return dtype


# ─────────────────────────────────────────────────────────────────────────────
# 5. Driver
# ─────────────────────────────────────────────────────────────────────────────

def build_corpus(args, rng):
    print(f"[1/4] sampling {args.n_words:,} words from S_{args.n} "
          f"(length in [{args.min_len}, {args.max_len}], "
          f"mode={'self-avoiding' if args.self_avoiding else 'uniform'}) ...")
    words = []
    precomputed_perms = []  # only used when self_avoiding
    for _ in range(args.n_words):
        L = rng.randint(args.min_len, args.max_len)
        if args.self_avoiding:
            w, perms = sample_self_avoiding_word(args.n, L, rng)
            words.append(w)
            precomputed_perms.append(perms)
        else:
            words.append(sample_word(args.n, L, rng))

    stoi, itos, vocab_size = build_tokenizer(args.n)
    print(f"[2/4] vocab_size = {vocab_size} ({args.n - 1} generators + "
          f"{N_RESERVED} reserved)")

    train_set, val_set, gen_set = split_words(
        len(words), args.val_frac, args.gen_frac, rng,
    )
    print(f"[3/4] split: train={len(train_set):,}  val={len(val_set):,}  "
          f"gen={len(gen_set):,}")

    splits = {
        "train": {"tokens": [], "word_starts": []},
        "val":   {"tokens": [], "word_starts": []},
        "gen":   {"tokens": [], "word_starts": []},
    }
    for word_idx, word in enumerate(words):
        if word_idx in train_set:
            name = "train"
        elif word_idx in val_set:
            name = "val"
        else:
            name = "gen"
        toks, perms = encode_word(word, args.n, stoi)
        start = len(splits[name]["tokens"])
        splits[name]["tokens"].extend(toks)
        end = start + len(toks)
        splits[name]["word_starts"].append((word_idx, start, end, perms))

    if args.shuffle_within_word:
        print("[3b/4] applying within-word shuffle (weak destroyed-structure) ...")
        for name in splits:
            splits[name]["tokens"] = shuffle_within_word(
                splits[name]["tokens"], rng,
            )
    if args.shuffle_globally:
        print("[3c/4] applying GLOBAL shuffle (strict destroyed-structure) ...")
        for name in splits:
            splits[name]["tokens"] = shuffle_globally(
                splits[name]["tokens"], rng,
            )

    return splits, stoi, itos, vocab_size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/symgroup_s8")
    p.add_argument("--n", type=int, default=8,
                   help="size of the symmetric group Sₙ (default 8)")
    p.add_argument("--n_words", type=int, default=10_000,
                   help="number of random words to sample")
    p.add_argument("--min_len", type=int, default=16)
    p.add_argument("--max_len", type=int, default=32)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--gen_frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle_within_word", action="store_true")
    p.add_argument("--shuffle_globally", action="store_true")
    p.add_argument("--self_avoiding", action="store_true",
                   help="generate self-avoiding walks on the Cayley graph of "
                        "Sₙ — the Othello-GPT analog. Each next generator must "
                        "produce a state not seen before in the word, forcing "
                        "the model to track partial product to predict legal "
                        "next moves.")
    args = p.parse_args()

    rng = random.Random(args.seed)
    splits, stoi, itos, vocab_size = build_corpus(args, rng)

    print(f"[4/4] writing to {args.out_dir} ...")
    dtype = dump(args.out_dir, splits, stoi, itos, vocab_size, args.n)

    print("\ndone.")
    for name in ("train", "val", "gen"):
        n_tok = len(splits[name]["tokens"])
        print(f"  {name}.bin: {n_tok:>10,} tokens "
              f"({n_tok / max(1, vocab_size):.0f} visits/token)")
    print(f"  dtype     : {np.dtype(dtype).name}")


if __name__ == "__main__":
    main()
