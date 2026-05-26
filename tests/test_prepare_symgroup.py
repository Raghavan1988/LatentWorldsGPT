"""
Smoke test for data/prepare_symgroup.py — offline, deterministic.

Verifies tokenizer bijection, generator-application correctness, encoded
layout, no-probe-target-leakage (partial-product values never appear as
token IDs), shuffle correctness, and dump roundtrip.
"""

import argparse
import importlib.util
import pickle
import random
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPARE = REPO_ROOT / "data" / "prepare_symgroup.py"

spec = importlib.util.spec_from_file_location("prepare_symgroup", PREPARE)
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)


def test_tokenizer_bijection():
    stoi, itos, vocab_size = ps.build_tokenizer(n=8)
    assert ps.N_RESERVED == 3
    # 7 generators (g_1..g_7) + 3 reserved
    assert vocab_size == 10
    assert all(idx >= ps.N_RESERVED for idx in stoi.values())
    assert {stoi[g] for g in range(1, 8)} == set(range(ps.N_RESERVED, vocab_size))
    assert all(itos[stoi[g]] == g for g in range(1, 8))


def test_generator_application():
    # Identity stays identity if we apply g and then g again (involution).
    perm = ps.identity(n=8)
    once = ps.apply_generator(perm, gen_idx=3)
    twice = ps.apply_generator(once, gen_idx=3)
    assert once != perm and twice == perm

    # g_3 specifically swaps positions 2 and 3
    perm = ps.identity(4)
    out = ps.apply_generator(perm, gen_idx=3)
    assert out == [0, 1, 3, 2]


def test_encode_word_partial_products():
    n = 5
    stoi, _, _ = ps.build_tokenizer(n)
    # word = [g_1, g_3, g_2] applied to identity
    #   identity        [0,1,2,3,4]
    #   after g_1       [1,0,2,3,4]   (swap 0,1)
    #   after g_3       [1,0,3,2,4]   (swap 2,3)
    #   after g_2       [1,3,0,2,4]   (swap 1,2)
    word = [1, 3, 2]
    tokens, perms = ps.encode_word(word, n, stoi)
    assert tokens[0] == ps.BOS
    assert tokens[-1] == ps.EOS
    assert tokens[1:-1] == [stoi[g] for g in word]
    assert perms[0] == [0, 1, 2, 3, 4]
    assert perms[1] == [1, 0, 2, 3, 4]
    assert perms[2] == [1, 0, 3, 2, 4]
    assert perms[3] == [1, 3, 0, 2, 4]
    # EOS gets the same partial product as the last real token
    assert perms[4] == perms[3]


def test_no_probe_target_leakage():
    """Partial-product values (small integers in [0, n)) must not appear
    in the token stream. Tokens are PAD/BOS/EOS/generator-IDs, all
    distinct from the n probe-target class IDs.

    With n=8 and N_RESERVED=3, generator tokens span [3, 9]. Partial-
    product values span [0, 7]. They overlap on {3,4,5,6,7} as integers
    but never as IDs in the model's vocabulary — the model never SEES
    a "permutation value" because we never write one to a .bin.
    """
    n = 8
    stoi, _, vocab_size = ps.build_tokenizer(n)
    tokens, perms = ps.encode_word([1, 2, 3, 4, 5, 6, 7] * 3, n, stoi)
    for t in tokens:
        assert t == ps.PAD or t == ps.BOS or t == ps.EOS or \
               (ps.N_RESERVED <= t < vocab_size)


def test_shuffles_preserve_what_they_should():
    from collections import Counter
    rng = random.Random(0)
    n = 5
    stoi, _, _ = ps.build_tokenizer(n)
    # Build a small stream of two words
    toks1, _ = ps.encode_word([1, 2, 3, 4], n, stoi)
    toks2, _ = ps.encode_word([2, 4, 1], n, stoi)
    stream = toks1 + toks2

    out = ps.shuffle_within_word(stream, rng)
    # BOS/EOS positions unchanged
    assert [i for i, t in enumerate(out) if t == ps.BOS] == \
           [i for i, t in enumerate(stream) if t == ps.BOS]
    assert [i for i, t in enumerate(out) if t == ps.EOS] == \
           [i for i, t in enumerate(stream) if t == ps.EOS]
    # per-word multiset preserved
    assert Counter(out[1:5]) == Counter(stream[1:5])
    assert Counter(out[7:10]) == Counter(stream[7:10])

    rng2 = random.Random(1)
    out_g = ps.shuffle_globally(stream, rng2)
    assert Counter(out_g) == Counter(stream)


def test_dump_roundtrip(tmp_path):
    args = argparse.Namespace(
        out_dir=str(tmp_path), n=5, n_words=50,
        min_len=8, max_len=12, val_frac=0.1, gen_frac=0.1, seed=0,
        shuffle_within_word=False, shuffle_globally=False,
    )
    rng = random.Random(args.seed)
    splits, stoi, itos, vocab_size = ps.build_corpus(args, rng)
    ps.dump(tmp_path, splits, stoi, itos, vocab_size, args.n)

    meta = pickle.loads((tmp_path / "meta.pkl").read_bytes())
    dtype = np.dtype(meta["dtype"])
    assert meta["n"] == 5
    assert meta["vocab_size"] == vocab_size

    train_arr = np.fromfile(tmp_path / "train.bin", dtype=dtype)
    assert train_arr.size > 0
    assert int(train_arr.min()) >= 0
    assert int(train_arr.max()) < vocab_size

    # partial_product.csv has the right header and at least one real row
    lines = (tmp_path / "partial_product.csv").read_text().splitlines()
    assert lines[0] == "split,word_idx,token_pos,perm"
    assert len(lines) > 1
    # each perm row should be a valid permutation of [0, n)
    for line in lines[1:10]:
        split_name, word_idx, token_pos, perm_s = line.split(",")
        perm = [int(x) for x in perm_s.split("-")]
        assert sorted(perm) == list(range(5))


if __name__ == "__main__":
    import tempfile
    test_tokenizer_bijection()
    test_generator_application()
    test_encode_word_partial_products()
    test_no_probe_target_leakage()
    test_shuffles_preserve_what_they_should()
    with tempfile.TemporaryDirectory() as d:
        test_dump_roundtrip(Path(d))
    print("ok")
