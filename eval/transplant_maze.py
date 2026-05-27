"""
LatentWorldsGPT — Phase 4 maze transplant.

Activation transplant for the maze navigation domain. At a chosen
layer L, we replace the recipient's residual stream with the donor's
and measure whether the model's next-token distribution shifts toward
the donor's expected next step.

Donor (B) and recipient (A) are positions on different maze paths.
For each pair we measure three quantities at the next-token output:

  P(A's next-step token)  : probability of A's actual next path step
  P(B's next-step token)  : probability of B's actual next path step
  P(C's next-step token)  : random-control next-step from a third maze

unp = unpatched (run on A's prefix unchanged)
trp = transplanted (run on A's prefix with B's residual at layer L)
rnd = random control (run on A's prefix with C's residual at layer L
       where C is a random other position)

The headline metric is:
  P(B's next-step) under trp − P(B's next-step) under unp

A positive lift means the transplant successfully steered the model
toward B's continuation; a near-zero lift means the residual stream
isn't causally encoding B's path state at this layer.

Usage:
    python eval/transplant_maze.py \\
        --ckpt checkpoints/maze_8x8/best.pt \\
        --data_dir data/maze_8x8 \\
        --layer 2 --n_donors 500 --n_pairs 200 --seed 0
"""
import argparse
import csv
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "model"))

from model import GPT, GPTConfig  # noqa: E402

PAD, BOS, EOS = 0, 1, 2


def load_maze_records(data_dir, split="gen"):
    """Return per-position records: list of dicts with current_cell,
    distance_to_goal, and an inferred next_cell (computed from the
    next row of the same maze in the side table)."""
    rows = []
    with open(data_dir / "mazes.csv") as f:
        for r in csv.DictReader(f):
            if r["split"] != split:
                continue
            rows.append({
                "maze_idx": int(r["maze_idx"]),
                "token_pos": int(r["token_pos"]),
                "current_cell": int(r["current_cell"]),
                "start_cell": int(r["start_cell"]),
                "goal_cell": int(r["goal_cell"]),
                "distance_to_goal": int(r["distance_to_goal"]),
            })
    # Compute next_cell for each row by looking at the next row in the same maze
    by_maze = defaultdict(list)
    for r in rows:
        by_maze[r["maze_idx"]].append(r)
    for maze_rows in by_maze.values():
        maze_rows.sort(key=lambda r: r["token_pos"])
        for i, r in enumerate(maze_rows):
            r["next_cell"] = maze_rows[i + 1]["current_cell"] if i + 1 < len(maze_rows) else None
    # Flatten back
    return [r for maze_rows in by_maze.values() for r in maze_rows
            if r["next_cell"] is not None]


@torch.no_grad()
def cache_residual_at_layer(model, idx_batch, layer):
    """Run forward through `layer` blocks; return the residual stream
    after that block. Shape: (B, T, n_embd)."""
    B, T = idx_batch.shape
    tok_emb = model.transformer.wte(idx_batch)
    pos = torch.arange(0, T, dtype=torch.long, device=idx_batch.device)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)
    for i, block in enumerate(model.transformer.h):
        x = block(x)
        if i + 1 == layer:
            return x
    return x  # if layer == n_layer, return final residual


@torch.no_grad()
def forward_with_replacement(model, idx_batch, layer, replacement_residual):
    """Run forward, but at the chosen layer, replace the residual at
    the last position with `replacement_residual` (B, n_embd)."""
    B, T = idx_batch.shape
    tok_emb = model.transformer.wte(idx_batch)
    pos = torch.arange(0, T, dtype=torch.long, device=idx_batch.device)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)
    for i, block in enumerate(model.transformer.h):
        x = block(x)
        if i + 1 == layer:
            x = x.clone()
            x[:, -1, :] = replacement_residual
    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)
    return logits


def build_context(stream, token_pos, block_size):
    start = max(0, token_pos - block_size + 1)
    ctx = stream[start:token_pos + 1]
    if len(ctx) < block_size:
        ctx = np.concatenate([np.full(block_size - len(ctx), PAD, dtype=ctx.dtype), ctx])
    return ctx


def build_donors(model, stream, records, block_size, layer, device,
                 n_donors, rng_seed=0, split_offset=0):
    """Sample `n_donors` records; cache the residual at `layer` at the
    last position of each context."""
    rng = np.random.default_rng(rng_seed)
    sample_ix = rng.choice(len(records), size=min(n_donors, len(records)), replace=False)
    donors = []
    contexts = []
    for i in sample_ix:
        r = records[i]
        global_pos = split_offset + r["token_pos"]
        if global_pos + 1 >= len(stream) or global_pos < 0:
            continue
        ctx = build_context(stream, global_pos, block_size)
        contexts.append(ctx)
        donors.append(dict(r))
    if not contexts:
        return []
    batch_size = 32
    for start in range(0, len(contexts), batch_size):
        batch = contexts[start:start + batch_size]
        idx_t = torch.from_numpy(np.stack(batch)).long().to(device)
        res = cache_residual_at_layer(model, idx_t, layer)  # (B, T, n_embd)
        last = res[:, -1, :].cpu().numpy()
        for k, d in enumerate(donors[start:start + len(batch)]):
            d["residual"] = last[k]
            d["context"] = batch[k]
    return [d for d in donors if "residual" in d]


def run_transplant(model, donors, layer, n_pairs, device, rng_seed=0,
                   batch_size=16):
    """Sample (A, B, C) triples; run unp, trp, rnd forwards; record
    P(A_next), P(B_next), P(C_next) under each."""
    rng = np.random.default_rng(rng_seed)
    n = len(donors)
    if n < 3:
        return None
    # Sample triples
    triples = []
    while len(triples) < n_pairs and len(triples) < n * n:
        a, b, c = rng.choice(n, 3, replace=False)
        if (donors[a]["maze_idx"] == donors[b]["maze_idx"]
            or donors[a]["maze_idx"] == donors[c]["maze_idx"]
            or donors[b]["maze_idx"] == donors[c]["maze_idx"]):
            continue
        triples.append((a, b, c))

    results = defaultdict(list)
    for start in range(0, len(triples), batch_size):
        batch = triples[start:start + batch_size]
        a_ctxs = np.stack([donors[a]["context"] for a, _, _ in batch])
        b_residuals = np.stack([donors[b]["residual"] for _, b, _ in batch])
        c_residuals = np.stack([donors[c]["residual"] for _, _, c in batch])
        a_next = np.array([donors[a]["next_cell"] for a, _, _ in batch])
        b_next = np.array([donors[b]["next_cell"] for _, b, _ in batch])
        c_next = np.array([donors[c]["next_cell"] for _, _, c in batch])

        a_ctxs_t = torch.from_numpy(a_ctxs).long().to(device)
        b_res_t = torch.from_numpy(b_residuals).float().to(device)
        c_res_t = torch.from_numpy(c_residuals).float().to(device)

        # unpatched
        logits_unp = forward_with_replacement(model, a_ctxs_t, layer + 1,
                                              # Use layer + 1 to write AFTER block `layer`;
                                              # then the replacement happens at the same point cache_residual_at_layer reads.
                                              # Simpler: dummy replacement at layer+1 that doesn't exist:
                                              torch.zeros_like(b_res_t))
        # Re-do unpatched cleanly without replacement
        logits_unp = forward_no_replacement(model, a_ctxs_t)
        # transplanted
        logits_trp = forward_with_replacement(model, a_ctxs_t, layer, b_res_t)
        # random control
        logits_rnd = forward_with_replacement(model, a_ctxs_t, layer, c_res_t)

        # Read prob at last position for the specific next-step tokens
        for label, logits in (("unp", logits_unp), ("trp", logits_trp), ("rnd", logits_rnd)):
            p = torch.softmax(logits[:, -1, :], dim=-1).cpu().numpy()
            for i in range(len(batch)):
                results[f"{label}_PA"].append(float(p[i, a_next[i]]))
                results[f"{label}_PB"].append(float(p[i, b_next[i]]))
                results[f"{label}_PC"].append(float(p[i, c_next[i]]))

    return results, len(triples)


@torch.no_grad()
def forward_no_replacement(model, idx_batch):
    logits, _ = model(idx_batch)
    return logits


def summarize(results, n_pairs):
    print("")
    print("═" * 78)
    print("ACTIVATION TRANSPLANT — SUMMARY (maze)")
    print("═" * 78)
    print(f"n_pairs: {n_pairs}")
    print("")
    print("  condition       P(A nxt)    P(B nxt)    P(C nxt)")
    for label in ("unp", "trp", "rnd"):
        a = np.mean(results[f"{label}_PA"])
        b = np.mean(results[f"{label}_PB"])
        c = np.mean(results[f"{label}_PC"])
        print(f"  {label:<10}    {a:.4f}      {b:.4f}      {c:.4f}")
    print("")
    print("Effect sizes:")
    d_PB_trp_unp = np.mean(results["trp_PB"]) - np.mean(results["unp_PB"])
    d_PB_trp_rnd = np.mean(results["trp_PB"]) - np.mean(results["rnd_PB"])
    d_PA_trp_unp = np.mean(results["trp_PA"]) - np.mean(results["unp_PA"])
    rate = np.mean(np.array(results["trp_PB"]) > np.array(results["rnd_PB"]))
    print(f"  Δ P(B's next-step)  transplant − unpatched : {d_PB_trp_unp:+.4f}")
    print(f"  Δ P(B's next-step)  transplant − random    : {d_PB_trp_rnd:+.4f}")
    print(f"  Δ P(A's next-step)  transplant − unpatched : {d_PA_trp_unp:+.4f}")
    print(f"  rate(transplant > random on P(B's next-step)) : {rate*100:.1f}%")
    print("")
    print("─" * 78)
    print("Effect-size summary")
    print(f"  Δ P(B's next-step) transplant − unpatched : {d_PB_trp_unp:+.4f}")
    print(f"  Δ P(B's next-step) transplant − random    : {d_PB_trp_rnd:+.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", choices=("val", "gen"), default="gen")
    p.add_argument("--layer", type=int, default=2,
                   help="0..n_layer-1 — block index AFTER which we transplant.")
    p.add_argument("--n_donors", type=int, default=500)
    p.add_argument("--n_pairs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device); model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded ckpt: iter={ckpt.get('iter', '?')} val_ppl={ckpt.get('val_perplexity', float('nan')):.4f}")
    print(f"layer = {args.layer} (of {config.n_layer} blocks)")

    data_dir = Path(args.data_dir)
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dtype = np.dtype(meta["dtype"])
    # Stitch the bins so token_pos values from mazes.csv map correctly.
    val_stream = np.asarray(np.memmap(data_dir / "val.bin", dtype=dtype, mode="r"))
    gen_stream = np.asarray(np.memmap(data_dir / "gen.bin", dtype=dtype, mode="r"))
    if args.split == "val":
        stream = val_stream
        offset = 0  # token_pos in mazes.csv is offset within its own split
    else:
        stream = gen_stream
        offset = 0  # gen.csv is its own split; offsets are within gen

    print(f"{args.split}.bin: {len(stream):,} tokens")

    records = load_maze_records(data_dir, split=args.split)
    print(f"loaded {len(records):,} {args.split} records with valid next_cell")

    print(f"building donor bank (n_donors={args.n_donors}) ...")
    donors = build_donors(model, stream, records, config.block_size,
                          args.layer, device, args.n_donors,
                          rng_seed=args.seed, split_offset=offset)
    print(f"  built {len(donors)} donors")
    if len(donors) < 50:
        print("not enough donors; exit"); return

    print(f"\nrunning transplant interventions (n_pairs={args.n_pairs}) ...")
    t0 = time.time()
    result = run_transplant(model, donors, args.layer, args.n_pairs,
                            device, rng_seed=args.seed,
                            batch_size=args.batch_size)
    print(f"  done in {time.time() - t0:.1f}s")

    if result is None:
        return
    metrics, n_pairs = result
    summarize(metrics, n_pairs)


if __name__ == "__main__":
    main()
