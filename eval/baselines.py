"""
LatentCityGPT — baselines (PLAN.md Phase 3).

WHAT THIS FILE DOES, IN ONE PICTURE
====================================

  train.bin (counts)                   val.bin / gen.bin (evaluate)
        │                                    │
        ▼                                    │
   uniform / unigram / 1st-order Markov     │
   / 2nd-order Markov tables                 │
        │                                    │
        └────────────────┬───────────────────┘
                         ▼
                  cross-entropy + perplexity (per baseline, per split)

                  ┌─────────────────────────────────────────────┐
                  │  Long-range coherence metric (the harder    │
                  │  ground for which Markov should lose):      │
                  │                                             │
                  │  For each true route in val.bin:            │
                  │    1. Take a short prefix (start + few hops)│
                  │    2. Generate K more hops with each model  │
                  │    3. Measure graph-distance from generated │
                  │       end-position to the true destination  │
                  │                                             │
                  │  → Markov has no notion of destination — it │
                  │    only knows "where I am now" — so it drifts│
                  │    randomly. LatentCityGPT can infer the    │
                  │    destination from the trajectory.         │
                  └─────────────────────────────────────────────┘


WHY THIS PHASE EXISTS — what PLAN.md / CONTEXT.md call "the trap"
================================================================

"LatentCityGPT beats the baselines at next-token prediction" is a TRAP. A
1st-order Markov chain over the street graph IS the adjacency matrix — at
short range, it's competitive on perplexity because the next intersection is
already very constrained by the current one.

The real claim of LatentCityGPT is NOT "I'm a better next-token predictor."
The real claim is "I contain an emergent metric map of the city — a
representation Markov structurally cannot have."

This file's job is to EARN THE RIGHT to that claim, not to make it. Two things
to establish here:
  1. On perplexity, LatentCityGPT is at LEAST as good as Markov (sanity).
  2. On long-range coherence, LatentCityGPT clearly beats Markov (it stays
     goal-directed where Markov drifts) — but this is still a route-model
     comparison. The actual contribution lives in `probe.py` (Phase 4).


BASELINES IMPLEMENTED
=====================

  - UNIFORM RANDOM      — P(t) = 1/vocab_size                 (trivial floor)
  - UNIGRAM             — P(t) = count(t) / total_count       (token frequency only)
  - 1st-ORDER MARKOV    — P(B|A) from bigram counts, Laplace smoothing
  - 2nd-ORDER MARKOV    — P(C|A,B) from trigram counts, backoff to 1st-order
                          and unigram on unseen contexts

DEFERRED:
  - Same-parameter-count LSTM. Same training-loop pattern as the GPT;
    requires ~20-30 min of MPS time per city. Will add after the current
    Manhattan/Boston training jobs finish so it doesn't contend for the GPU.


PROTOCOL
========

Cross-entropy is computed only on REAL → REAL transitions (token at position t
and target at position t+1 both in [3, vocab_size)). Reasons:
  - Markov over the graph is defined for real-node transitions, not BOS/EOS.
  - It's the metric that actually matters: "given the model is at a real
    intersection, how confident is it about the next intersection?"
  - For the same-protocol GPT number we re-compute the GPT's CE on the same
    filter (just real-real transitions), so the comparison is apples-to-apples.


USAGE
=====
    python eval/baselines.py --data_dir data/london_city

    # Include GPT comparison (loads a checkpoint and computes CE on the same
    # real-real-transition filter):
    python eval/baselines.py --data_dir data/london_city --ckpt checkpoints/best.pt

    # Long-range coherence metric (slower; needs the checkpoint):
    python eval/baselines.py --data_dir data/london_city --ckpt checkpoints/best.pt \\
        --coherence --n_routes 100 --prefix_len 4 --gen_steps 20
"""

import argparse
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import networkx as nx
import torch

# Allow importing the model package when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
from model import GPT, GPTConfig, BOS, EOS, PAD  # noqa: E402


N_RESERVED = 3   # PAD=0, BOS=1, EOS=2; real intersections start at 3


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_streams(data_dir: Path) -> tuple[dict, dict]:
    """Load meta.pkl + train/val/gen .bin streams. Returns (meta, streams)."""
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dtype = np.dtype(meta["dtype"])
    streams = {}
    for split in ("train", "val", "gen"):
        streams[split] = np.asarray(np.memmap(data_dir / f"{split}.bin",
                                              dtype=dtype, mode="r"))
    return meta, streams


def real_real_transitions(stream: np.ndarray) -> np.ndarray:
    """Return the indices t where BOTH stream[t] and stream[t+1] are real tokens
    (>= N_RESERVED). Used as the filter for every baseline's evaluation so the
    comparison is apples-to-apples across models."""
    s = stream.astype(np.int64)
    return np.nonzero((s[:-1] >= N_RESERVED) & (s[1:] >= N_RESERVED))[0]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Count tables from train.bin
# ─────────────────────────────────────────────────────────────────────────────

def count_unigrams(stream: np.ndarray, vocab_size: int) -> np.ndarray:
    """Count occurrences of each real token in `stream`. Returns counts (vocab_size,)."""
    counts = np.zeros(vocab_size, dtype=np.int64)
    s = stream.astype(np.int64)
    s = s[s >= N_RESERVED]                       # ignore control tokens
    np.add.at(counts, s, 1)
    return counts


def count_bigrams(stream: np.ndarray) -> defaultdict:
    """Count A → B transitions over real-real positions. Returns nested dict
    {a: {b: count}} so unseen (a, b) returns 0 without huge dense arrays.
    Sparse is essential at South Bay's 46k vocab (~2 billion possible bigrams).
    """
    counts = defaultdict(lambda: defaultdict(int))
    s = stream.astype(np.int64)
    for t in real_real_transitions(stream):
        a, b = int(s[t]), int(s[t + 1])
        counts[a][b] += 1
    return counts


def count_trigrams(stream: np.ndarray) -> defaultdict:
    """Count (A, B) → C transitions where all three are real tokens.
    Sparse keyed by (a, b)."""
    counts = defaultdict(lambda: defaultdict(int))
    s = stream.astype(np.int64)
    n = len(s)
    for t in range(n - 2):
        a, b, c = int(s[t]), int(s[t + 1]), int(s[t + 2])
        if a < N_RESERVED or b < N_RESERVED or c < N_RESERVED:
            continue
        counts[(a, b)][c] += 1
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# 3. Probability lookups with smoothing / backoff
# ─────────────────────────────────────────────────────────────────────────────

def uniform_logp(target_count: int, vocab_size: int) -> float:
    """Log probability for the uniform-over-vocab baseline. Returns the per-
    position CE contribution times target_count (caller divides by N)."""
    return -math.log(vocab_size) * target_count


def unigram_logp_sum(targets: np.ndarray, unigram_counts: np.ndarray,
                    total_real_tokens: int, alpha: float = 1.0) -> float:
    """Sum of log P_unigram(target) over targets, with add-α Laplace smoothing.

    P(t) = (count(t) + α) / (total + α * vocab_size_real)
    Only real-token vocabulary is in scope here.
    """
    vocab_real = (unigram_counts.shape[0] - N_RESERVED)
    denom = total_real_tokens + alpha * vocab_real
    # log P for each target; pull from numpy for speed
    smoothed = (unigram_counts[targets] + alpha) / denom
    return float(np.log(smoothed).sum())


def markov1_logp(prev: int, target: int, bigrams: dict,
                 unigram_counts: np.ndarray, total_real_tokens: int,
                 alpha: float = 1.0) -> float:
    """log P(target | prev) under a 1st-order Markov chain with add-α smoothing.

    P(B|A) = (count(A→B) + α) / (sum_b count(A→b) + α * vocab_size_real)
    Unseen (A→B) for an unseen A backs off to unigram(B).
    """
    if prev not in bigrams:
        # A never seen — back off to unigram on B
        vocab_real = unigram_counts.shape[0] - N_RESERVED
        denom = total_real_tokens + alpha * vocab_real
        return math.log((unigram_counts[target] + alpha) / denom)

    row = bigrams[prev]
    row_total = sum(row.values())
    vocab_real = unigram_counts.shape[0] - N_RESERVED
    denom = row_total + alpha * vocab_real
    num = row.get(target, 0) + alpha
    return math.log(num / denom)


def markov2_logp(prev2: int, prev1: int, target: int,
                 trigrams: dict, bigrams: dict,
                 unigram_counts: np.ndarray, total_real_tokens: int,
                 alpha: float = 0.1) -> float:
    """log P(target | prev2, prev1) — trigram with backoff to bigram, then unigram.

    Backoff: if (prev2, prev1) not seen in trigrams, fall back to P(target | prev1)
    via the 1st-order Markov model. This is a simple Katz-style backoff (no
    discounting). Sufficient for our purposes given the smallish vocab.
    """
    key = (prev2, prev1)
    if key not in trigrams:
        return markov1_logp(prev1, target, bigrams, unigram_counts,
                            total_real_tokens, alpha=1.0)
    row = trigrams[key]
    row_total = sum(row.values())
    vocab_real = unigram_counts.shape[0] - N_RESERVED
    denom = row_total + alpha * vocab_real
    num = row.get(target, 0) + alpha
    return math.log(num / denom)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluate baselines on a split
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_baselines_on_split(stream: np.ndarray, split_name: str,
                                vocab_size: int, unigram_counts: np.ndarray,
                                bigrams: dict, trigrams: dict,
                                total_real_train_tokens: int) -> dict:
    """Compute CE + perplexity on the split for each analytical baseline.
    Operates only on real → real transitions (apples-to-apples)."""
    s = stream.astype(np.int64)
    transitions = real_real_transitions(stream)
    N = len(transitions)
    if N == 0:
        return {}

    targets = s[transitions + 1]
    prevs = s[transitions]

    out = {}

    # Uniform: every real-vocab token gets equal probability.
    vocab_real = vocab_size - N_RESERVED
    ce_uniform = math.log(vocab_real)
    out["uniform"] = {"ce": ce_uniform, "ppl": math.exp(ce_uniform), "n": N}

    # Unigram.
    logp_unigram_sum = unigram_logp_sum(targets, unigram_counts, total_real_train_tokens)
    ce_unigram = -logp_unigram_sum / N
    out["unigram"] = {"ce": ce_unigram, "ppl": math.exp(ce_unigram), "n": N}

    # 1st-order Markov.
    total = 0.0
    for a, b in zip(prevs.tolist(), targets.tolist()):
        total += markov1_logp(a, b, bigrams, unigram_counts, total_real_train_tokens)
    ce_m1 = -total / N
    out["markov1"] = {"ce": ce_m1, "ppl": math.exp(ce_m1), "n": N}

    # 2nd-order Markov — need (prev2, prev1, target) where prev2 is also real.
    # We re-derive transitions where positions t-1, t, t+1 are all real.
    n2 = 0
    total2 = 0.0
    for t in range(1, len(s) - 1):
        if s[t - 1] < N_RESERVED or s[t] < N_RESERVED or s[t + 1] < N_RESERVED:
            continue
        total2 += markov2_logp(int(s[t - 1]), int(s[t]), int(s[t + 1]),
                                trigrams, bigrams, unigram_counts,
                                total_real_train_tokens)
        n2 += 1
    ce_m2 = -total2 / max(n2, 1)
    out["markov2"] = {"ce": ce_m2, "ppl": math.exp(ce_m2), "n": n2}

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. GPT cross-entropy on the SAME real→real filter (apples-to-apples)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def gpt_ce_on_real_real(model: GPT, stream: np.ndarray, block_size: int,
                         device: str, n_positions: int = 30_000) -> dict:
    """Compute the GPT's average log P(target | context) at real → real positions,
    using random windows so we don't have to score the full stream."""
    model.eval()
    rng = np.random.default_rng(0)
    s = stream.astype(np.int64)
    n_scored = 0
    log_p_sum = 0.0
    batch_size = 32

    while n_scored < n_positions:
        starts = rng.integers(0, len(s) - block_size - 1, size=batch_size)
        windows = [s[i : i + block_size + 1] for i in starts]
        batch = np.stack(windows)
        x = torch.from_numpy(batch[:, :-1]).long().to(device)
        y = torch.from_numpy(batch[:, 1:]).long().to(device)

        logits, _ = model(x)                   # (B, T, V)
        log_probs = torch.log_softmax(logits, dim=-1)
        # gather log_p at each (t, target)
        log_p_t = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (B, T)

        # mask: position is real-real iff x[b,t] >= 3 and y[b,t] >= 3
        mask = (x >= N_RESERVED) & (y >= N_RESERVED)
        log_p_real = log_p_t[mask]
        log_p_sum += log_p_real.sum().item()
        n_scored += int(mask.sum().item())

    ce = -log_p_sum / max(n_scored, 1)
    return {"ce": ce, "ppl": math.exp(ce), "n": n_scored}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Long-range coherence metric
# ─────────────────────────────────────────────────────────────────────────────

def extract_full_routes(stream: np.ndarray, min_len: int = 8) -> list[list[int]]:
    """Walk the stream and return all routes [n1, n2, ...] (real-node tokens
    only) with length >= min_len. A route is the run of real tokens between BOS
    and EOS markers."""
    out = []
    cur = []
    for tok in stream.tolist():
        if tok == BOS:
            cur = []
        elif tok == EOS:
            if len(cur) >= min_len:
                out.append(cur)
            cur = []
        elif tok >= N_RESERVED:
            cur.append(tok)
    return out


def sample_markov1(prev: int, bigrams: dict, vocab_size: int,
                   unigram_counts: np.ndarray,
                   rng: np.random.Generator) -> int:
    """Sample next token from the 1st-order Markov chain. If `prev` unseen, falls
    back to unigram sampling."""
    if prev in bigrams and bigrams[prev]:
        row = bigrams[prev]
        toks = list(row.keys())
        probs = np.array([row[t] for t in toks], dtype=np.float64)
        probs /= probs.sum()
        return int(rng.choice(toks, p=probs))
    # backoff: sample from unigram
    probs = unigram_counts.astype(np.float64)
    probs[:N_RESERVED] = 0
    s = probs.sum()
    if s == 0:
        return N_RESERVED  # unreachable in practice
    probs /= s
    return int(rng.choice(len(probs), p=probs))


@torch.no_grad()
def coherence_metric(routes: list[list[int]], itos: dict, stoi: dict,
                     G: nx.MultiDiGraph,
                     model: GPT, bigrams: dict, vocab_size: int,
                     unigram_counts: np.ndarray,
                     device: str, prefix_len: int, gen_steps: int,
                     n_routes: int) -> dict:
    """For up to `n_routes` true routes from val.bin:
      1. Use the first `prefix_len` tokens as the prefix.
      2. Generate `gen_steps` more tokens with GPT (greedy → deterministic).
      3. Generate `gen_steps` more tokens with 1st-order Markov (also greedy).
      4. Measure graph shortest-path distance from the final generated real-node
         to the route's true destination (last node).

    Returns dict with median, mean, p90 of the distance for each model, plus
    the count of routes where the final generation was a real node (validity).

    Greedy is used for both models so the difference is about world-model
    capability, not sampling variance.
    """
    rng = np.random.default_rng(0)
    rng.shuffle(routes)
    routes = routes[:n_routes]

    model.eval()

    gpt_dists = []
    markov_dists = []
    gpt_valid = 0
    markov_valid = 0
    n_used = 0

    for route in routes:
        # need: prefix_len < len(route) - 1 so we have at least one true continuation
        if len(route) < prefix_len + 1:
            continue
        prefix_tokens = route[:prefix_len]            # list of real-token ids
        true_dest_tok = route[-1]
        true_dest_node = itos[true_dest_tok]

        # ── GPT generation (greedy from BOS + prefix) ──
        ctx = [BOS] + prefix_tokens
        for _ in range(gen_steps):
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            x = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]
            logits, _ = model(x)
            nxt = int(logits[0, -1].argmax().item())
            ctx.append(nxt)
            if nxt == EOS:
                break
        # take the last real-node token in ctx
        gpt_last = next((t for t in reversed(ctx) if t >= N_RESERVED), None)
        if gpt_last is not None and gpt_last in itos:
            gpt_valid += 1
            try:
                d = nx.shortest_path_length(G, itos[gpt_last], true_dest_node)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                d = -1
            if d >= 0:
                gpt_dists.append(d)

        # ── Markov 1st-order generation (greedy = argmax bigram) ──
        cur = prefix_tokens[-1]
        markov_ctx = list(prefix_tokens)
        for _ in range(gen_steps):
            if cur in bigrams and bigrams[cur]:
                row = bigrams[cur]
                nxt = max(row.items(), key=lambda kv: kv[1])[0]
            else:
                # back off to unigram argmax (the most common real node)
                probs = unigram_counts.copy().astype(np.int64)
                probs[:N_RESERVED] = 0
                nxt = int(np.argmax(probs))
            markov_ctx.append(nxt)
            cur = nxt
        markov_last = next((t for t in reversed(markov_ctx) if t >= N_RESERVED), None)
        if markov_last is not None and markov_last in itos:
            markov_valid += 1
            try:
                d = nx.shortest_path_length(G, itos[markov_last], true_dest_node)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                d = -1
            if d >= 0:
                markov_dists.append(d)

        n_used += 1

    def stats(arr):
        if not arr:
            return {"median": float("nan"), "mean": float("nan"),
                    "p90": float("nan"), "n": 0}
        a = np.array(arr)
        return {"median": float(np.median(a)), "mean": float(a.mean()),
                "p90": float(np.percentile(a, 90)), "n": len(a)}

    return {
        "routes_attempted": n_used,
        "gpt":    {**stats(gpt_dists),    "valid": gpt_valid},
        "markov": {**stats(markov_dists), "valid": markov_valid},
        "prefix_len": prefix_len,
        "gen_steps": gen_steps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────────────

def fmt_row(name: str, ce: float, ppl: float, n: int) -> str:
    return f"  {name:<22}  CE={ce:8.4f}   ppl={ppl:10.2f}   n={n:,}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", default=None,
                   help="optional checkpoint for the apples-to-apples GPT comparison")
    p.add_argument("--n_positions", type=int, default=30_000,
                   help="GPT CE estimation positions (when --ckpt given)")
    p.add_argument("--coherence", action="store_true",
                   help="also run the long-range coherence metric (needs --ckpt)")
    p.add_argument("--n_routes", type=int, default=100,
                   help="(coherence) number of true routes from val.bin to use")
    p.add_argument("--prefix_len", type=int, default=4,
                   help="(coherence) tokens of true route used as prefix")
    p.add_argument("--gen_steps", type=int, default=20,
                   help="(coherence) tokens to generate beyond the prefix")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")

    # Load data
    meta, streams = load_streams(data_dir)
    vocab_size = meta["vocab_size"]
    itos = meta["itos"]
    stoi = meta["stoi"]
    n_real = vocab_size - N_RESERVED
    print(f"data_dir: {data_dir}   vocab={vocab_size}   real_nodes={n_real}")

    # Build count tables from train.bin
    t0 = time.time()
    unigram_counts = count_unigrams(streams["train"], vocab_size)
    total_real_train = int(unigram_counts.sum())
    bigrams = count_bigrams(streams["train"])
    trigrams = count_trigrams(streams["train"])
    print(f"counts built in {time.time()-t0:.1f}s   "
          f"(real-train tokens={total_real_train:,}, "
          f"unique bigrams={sum(len(v) for v in bigrams.values()):,}, "
          f"unique trigrams={sum(len(v) for v in trigrams.values()):,})")

    # Evaluate analytical baselines on val + gen
    print(f"\n{'─'*78}")
    print(f"Analytical baselines — real → real transitions only")
    print(f"{'─'*78}")

    for split in ("val", "gen"):
        results = evaluate_baselines_on_split(
            streams[split], split, vocab_size, unigram_counts,
            bigrams, trigrams, total_real_train,
        )
        print(f"\n[{split}.bin]")
        for name in ("uniform", "unigram", "markov1", "markov2"):
            r = results[name]
            print(fmt_row(name, r["ce"], r["ppl"], r["n"]))

    # Optionally compare to GPT
    if args.ckpt:
        print(f"\n{'─'*78}")
        print(f"GPT (same protocol: real → real transitions only)")
        print(f"{'─'*78}")
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        config = GPTConfig(**ckpt["config"])
        model = GPT(config).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        for split in ("val", "gen"):
            r = gpt_ce_on_real_real(model, streams[split], config.block_size,
                                     device, n_positions=args.n_positions)
            print(f"\n[{split}.bin]")
            print(fmt_row("LatentCityGPT", r["ce"], r["ppl"], r["n"]))

    # Long-range coherence
    if args.coherence:
        if not args.ckpt:
            print("\n--coherence requires --ckpt; skipping.")
            return
        print(f"\n{'─'*78}")
        print(f"Long-range coherence — graph distance to true destination")
        print(f"  (lower is better; Markov should drift, GPT should stay closer)")
        print(f"{'─'*78}")
        G = pickle.loads((data_dir / "graph.gpickle").read_bytes())
        routes = extract_full_routes(streams["val"], min_len=args.prefix_len + 2)
        print(f"\nFound {len(routes):,} eligible val routes (length ≥ {args.prefix_len+2}).")
        print(f"Using up to {args.n_routes} for the test "
              f"(prefix_len={args.prefix_len}, gen_steps={args.gen_steps}).")

        res = coherence_metric(routes, itos, stoi, G, model, bigrams,
                               vocab_size, unigram_counts, device,
                               args.prefix_len, args.gen_steps, args.n_routes)
        print(f"\nroutes used: {res['routes_attempted']}")
        for k in ("gpt", "markov"):
            s = res[k]
            label = "LatentCityGPT" if k == "gpt" else "1st-order Markov"
            print(f"\n  {label}")
            print(f"     final-position validity : {s['valid']:>3d} / {res['routes_attempted']}")
            print(f"     median hops to dest     : {s['median']:>6.2f}")
            print(f"     mean   hops to dest     : {s['mean']:>6.2f}")
            print(f"     p90    hops to dest     : {s['p90']:>6.2f}")


if __name__ == "__main__":
    main()
