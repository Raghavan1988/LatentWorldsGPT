"""
LatentCityGPT — Phase 5: causal activation patching.

WHAT THIS FILE DOES, IN ONE PICTURE
====================================

  trained model (frozen)            coords.csv (probe side; not model input)
        │                                   │
        │ forward pass on a prefix          │
        ▼                                   │
  residual stream at chosen layer L         │
        │                                   │
        │ train linear probe                │
        │   y = W·a + b                     │
        ▼                                   │
   probe(a_t) = current location ───────────┤
                                            │
        │                                   ▼  pick random target node B,
        │                                  look up coords[B] in coords.csv
        │                                   │
        │ minimal-norm patch:               │
        │   Δ = W⁺ · (coords[B] - probe(a_t))
        ▼
  patched activation a_t + Δ at layer L
        │
        │ continue forward pass from L+1 → final layer
        ▼
  next-token logits (over vocab)
        │
        ▼
  measure shifts:
      P(A's neighbors): unpatched vs target-patched vs random-patched
      P(B's neighbors): unpatched vs target-patched vs random-patched


HYPOTHESIS (Phase 5)
====================
The model causally USES its emergent location representation to drive next-token
predictions. If true:

  patched-toward-B  ⇒  P(B's neighbors) RISES
                       P(A's neighbors) FALLS
  random-direction patch (same L2)  ⇒  no systematic shift toward any node's neighbors

NULL HYPOTHESIS
===============
The probe finds a location signal in activations, but the model doesn't actually
use that signal for prediction. In that case the patched and random-patched
distributions would shift similarly (only the model's general sensitivity to
perturbations, no location-specific effect).


WHY THE PSEUDOINVERSE
=====================
For a linear probe y = W·a + b (W shape (2, n_embd)), the patch
Δ = pinv(W) · (y_target - y_current) is the *minimum-L2-norm* perturbation of `a`
that achieves probe(a + Δ) = y_target. We want minimum norm because we want to
change the location signal and as little else as possible. Larger patches would
push the model out of its training distribution, contaminating the test.


THE ONE RULE — clarification
============================
This script reads coords.csv to compute patch directions. Like probe.py, the
coordinates flow into the PATCH (a probe-side operation in residual space), NOT
into the model's input. The model still only ever sees token IDs.


USAGE
=====
    # Default: probe at layer 3, 200 test positions, both target-patched and random controls.
    python eval/causal.py --ckpt checkpoints/best.pt --data_dir data/london_city

    # Tune which layer to patch.
    python eval/causal.py --ckpt checkpoints/manhattan/best.pt \\
        --data_dir data/manhattan --layer 5 --n_positions 300
"""

import argparse
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

# Repo layout: eval/causal.py and eval/probe.py are siblings; model/ is at the
# repo root next to eval/. Make both importable when run from anywhere.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                          # eval/probe.py reuse
sys.path.insert(0, str(HERE.parent / "model"))         # model/model.py

from model import GPT, GPTConfig, BOS, EOS, PAD                            # noqa: E402
from probe import (                                                         # noqa: E402
    load_coords_planar,
    LinearProbe,
    train_and_evaluate_probe,
    cache_layer_activations,
)

N_RESERVED = 3
R_EARTH_M = 6_371_008.8


# ─────────────────────────────────────────────────────────────────────────────
# 1. Build probe-training dataset at one layer only (lighter than probe.py's)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_layer_acts_for_probe(model: GPT, stream: np.ndarray,
                                 coords_xy: torch.Tensor, block_size: int,
                                 layer: int, n_positions: int, device: str,
                                 rng_seed: int = 0):
    """Sample n_positions positions whose current token is a real intersection;
    cache residual-stream activations at exactly `layer`, paired with that
    position's ground-truth (x_m, y_m)."""
    model.eval()
    rng = np.random.default_rng(rng_seed)
    batch_size = 32
    X_list, y_list, tok_list = [], [], []
    n_collected = 0

    while n_collected < n_positions:
        starts = rng.integers(0, len(stream) - block_size - 1, size=batch_size)
        windows = [np.asarray(stream[s : s + block_size]) for s in starts]
        idx_batch = torch.from_numpy(np.stack(windows).astype(np.int64)).to(device)

        acts_all = cache_layer_activations(model, idx_batch)   # list of n_layer+1
        acts_L = acts_all[layer]                                # (B, T, n_embd)

        idx_np = idx_batch.cpu().numpy()
        for b in range(idx_np.shape[0]):
            for t in range(idx_np.shape[1]):
                tok = int(idx_np[b, t])
                if tok < N_RESERVED:
                    continue
                xy = coords_xy[tok]
                if torch.isnan(xy).any():
                    continue
                X_list.append(acts_L[b, t].cpu().numpy())
                y_list.append(xy.numpy())
                tok_list.append(tok)
                n_collected += 1
                if n_collected >= n_positions:
                    break
            if n_collected >= n_positions:
                break

    return (np.stack(X_list).astype(np.float32),
            np.stack(y_list).astype(np.float32),
            np.array(tok_list, dtype=np.int64))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Forward pass with a residual-stream patch applied at one layer
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def patched_forward(model: GPT, idx_batch: torch.Tensor, layer: int,
                    patch_positions: list[int], patch_vectors: torch.Tensor):
    """Run the model forward, ADDING patch_vectors[i] to the residual stream at
    (batch_row=i, seq_pos=patch_positions[i]) after layer L.

    Args:
        idx_batch       : LongTensor (B, T) of tokens
        layer           : 0 = patch after embedding; k = patch after block k
        patch_positions : list of length B; the seq-position to patch in row i
        patch_vectors   : FloatTensor (B, n_embd) — one patch per row

    Returns logits (B, T, vocab_size).
    """
    B, T = idx_batch.shape
    device = idx_batch.device
    tok_emb = model.transformer.wte(idx_batch)
    pos = torch.arange(0, T, dtype=torch.long, device=device)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)

    if layer == 0:
        for b in range(B):
            x[b, patch_positions[b]] = x[b, patch_positions[b]] + patch_vectors[b]

    for i, block in enumerate(model.transformer.h, start=1):
        x = block(x)
        if i == layer:
            for b in range(B):
                x[b, patch_positions[b]] = x[b, patch_positions[b]] + patch_vectors[b]

    x = model.transformer.ln_f(x)
    return model.lm_head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. The causal intervention experiment
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_intervention(model: GPT, probe: LinearProbe, layer: int,
                     stream: np.ndarray, block_size: int, G: nx.MultiDiGraph,
                     itos: dict, stoi: dict, coords_xy: torch.Tensor,
                     n_positions: int, device: str, rng_seed: int = 0,
                     batch_size: int = 16, patch_scale: float = 1.0) -> dict:
    """For each of n_positions positions:
      - read the residual at `layer`
      - pick a random target node B that's NOT A and NOT a neighbor of A (so the
        patched direction is unambiguously toward a different location)
      - compute Δ_target  = W⁺ · (coords[B] - probe(a))  (min-norm patch)
      - compute Δ_random  = a random-direction vector with the same L2 norm
      - run THREE forward passes per position: unpatched, target-patched, random-patched
      - score the next-token distribution at the patched position:
          P_A_nbrs       = sum of probs over A's real graph neighbors
          P_B_nbrs       = sum of probs over B's real graph neighbors
    """
    model.eval()
    probe.eval()
    rng = np.random.default_rng(rng_seed)

    # ── Probe weight matrix and pseudoinverse ──
    W = probe.linear.weight                            # (2, n_embd)
    b_bias = probe.linear.bias                         # (2,)
    W_pinv = torch.linalg.pinv(W).to(device)           # (n_embd, 2)

    # ── Cache: every real token's set of graph neighbors (as token ids) ──
    # Builds quickly; lets us score "P(A's neighbors)" without re-traversing G.
    print("  building neighbor cache...", flush=True)
    real_tokens = [tok for tok in stoi.values() if tok >= N_RESERVED]
    nbr_tokens_of: dict[int, set[int]] = {}
    for tok in real_tokens:
        node = itos[tok]
        nbrs = set()
        if G.has_node(node):
            for n in G.successors(node):
                if n in stoi:
                    nbrs.add(stoi[n])
        nbr_tokens_of[tok] = nbrs

    # ── Sample positions to test ──
    # We need positions where the current token is a real node AND we have at least
    # one valid target node B (any other real node not in A's neighborhood).
    print(f"  sampling {n_positions} test positions...", flush=True)
    test_records = []  # each: (window: np.ndarray, pos_t: int, tok_A: int, tok_B: int)
    attempts = 0
    while len(test_records) < n_positions and attempts < n_positions * 50:
        attempts += 1
        start = int(rng.integers(0, len(stream) - block_size - 1))
        window = np.asarray(stream[start : start + block_size]).copy()
        # pick a position inside the window — prefer non-edge positions so the
        # model has both prefix context and a meaningful next token.
        t = int(rng.integers(8, block_size - 2))
        tok_A = int(window[t])
        if tok_A < N_RESERVED:
            continue
        if not nbr_tokens_of.get(tok_A):
            continue   # no neighbors; can't measure
        # Pick B: any real token with valid coords, not A and not in A's neighborhood,
        # so the patch direction is clearly *different from* A's local geography.
        for _ in range(20):
            tok_B = int(rng.choice(real_tokens))
            if tok_B == tok_A or tok_B in nbr_tokens_of[tok_A]:
                continue
            if not nbr_tokens_of.get(tok_B):
                continue
            break
        else:
            continue
        test_records.append((window, t, tok_A, tok_B))

    if len(test_records) < n_positions:
        print(f"  WARNING: only collected {len(test_records)}/{n_positions} valid positions")

    # ── Run interventions in mini-batches ──
    results = {
        "unpatched_PA": [],   # P(A's neighbors) in unpatched run
        "unpatched_PB": [],   # P(B's neighbors) in unpatched run
        "target_PA":    [],   # P(A's neighbors) in target-patched run
        "target_PB":    [],   # P(B's neighbors) in target-patched run
        "random_PA":    [],   # P(A's neighbors) in random-patched run
        "random_PB":    [],   # P(B's neighbors) in random-patched run
        "patch_norm":   [],
    }

    print(f"  running interventions ({batch_size} positions per batch)...", flush=True)
    t0 = time.time()
    for batch_start in range(0, len(test_records), batch_size):
        batch = test_records[batch_start : batch_start + batch_size]
        B = len(batch)
        idx_batch = torch.from_numpy(
            np.stack([w for w, _, _, _ in batch]).astype(np.int64)
        ).to(device)
        positions = [t for _, t, _, _ in batch]

        # 1) Unpatched forward
        logits_unp = patched_forward(model, idx_batch, layer=layer,
                                      patch_positions=positions,
                                      patch_vectors=torch.zeros(B, model.config.n_embd, device=device))

        # 2) Get current activation at (b, t) for each row to compute patch Δ
        acts_all = cache_layer_activations(model, idx_batch)
        acts_L = acts_all[layer]                                # (B, T, n_embd)
        cur_acts = torch.stack([acts_L[b, positions[b]] for b in range(B)])  # (B, n_embd)
        y_current = (W @ cur_acts.T).T + b_bias                  # (B, 2)
        y_targets = torch.stack(
            [coords_xy[tB] for (_, _, _, tB) in batch]
        ).to(device).float()                                      # (B, 2)
        delta_y = y_targets - y_current                           # (B, 2)
        patch_target_full = (W_pinv @ delta_y.T).T                # (B, n_embd) -- minimum-norm
        # Scale the patch. patch_scale=1.0 = pseudoinverse-exact (probe(a+Δ) = y_target).
        # Smaller values intentionally undershoot the location shift but disturb the
        # rest of the residual less, which is often the cleaner experimental regime.
        patch_target = patch_target_full * patch_scale            # (B, n_embd)

        # 3) Target-patched forward
        logits_tgt = patched_forward(model, idx_batch, layer=layer,
                                      patch_positions=positions,
                                      patch_vectors=patch_target)

        # 4) Random-direction patch with the same per-row L2 norm as the SCALED target patch
        rand = torch.randn(B, model.config.n_embd, device=device)
        rand = rand / rand.norm(dim=1, keepdim=True)
        rand = rand * patch_target.norm(dim=1, keepdim=True)
        logits_rand = patched_forward(model, idx_batch, layer=layer,
                                       patch_positions=positions,
                                       patch_vectors=rand)

        # 5) Score: probability mass on A's neighbors and B's neighbors
        for b in range(B):
            _, t, tok_A, tok_B = batch[b]
            nbrs_A = list(nbr_tokens_of[tok_A])
            nbrs_B = list(nbr_tokens_of[tok_B])
            if not nbrs_A or not nbrs_B:
                continue
            for label, logits in (("unpatched", logits_unp),
                                  ("target",    logits_tgt),
                                  ("random",    logits_rand)):
                probs = F.softmax(logits[b, t], dim=-1).cpu().numpy()
                p_A = float(probs[nbrs_A].sum())
                p_B = float(probs[nbrs_B].sum())
                results[f"{label}_PA"].append(p_A)
                results[f"{label}_PB"].append(p_B)
            results["patch_norm"].append(float(patch_target[b].norm().item()))
            results.setdefault("act_norm", []).append(float(cur_acts[b].norm().item()))

    print(f"  done in {time.time()-t0:.1f}s", flush=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def summarize(results: dict) -> dict:
    """Aggregate per-position scores into a small numeric summary."""
    out = {}
    n = len(results["unpatched_PA"])
    out["n"] = n
    for label in ("unpatched", "target", "random"):
        out[f"{label}_PA_mean"]   = float(np.mean(results[f"{label}_PA"]))
        out[f"{label}_PB_mean"]   = float(np.mean(results[f"{label}_PB"]))
        out[f"{label}_PA_median"] = float(np.median(results[f"{label}_PA"]))
        out[f"{label}_PB_median"] = float(np.median(results[f"{label}_PB"]))

    # The key effect-size: how much did patching toward B move mass to B's neighbors,
    # relative to (a) no patch, and (b) random patch?
    out["delta_PB_target_vs_unpatched"] = out["target_PB_mean"] - out["unpatched_PB_mean"]
    out["delta_PB_random_vs_unpatched"] = out["random_PB_mean"] - out["unpatched_PB_mean"]
    out["delta_PA_target_vs_unpatched"] = out["target_PA_mean"] - out["unpatched_PA_mean"]

    # Per-position: how often does target-patch increase P(B) more than random-patch does?
    n_target_beats_random = sum(1 for i in range(n)
                                 if results["target_PB"][i] > results["random_PB"][i])
    out["frac_target_beats_random_on_PB"] = n_target_beats_random / max(n, 1)
    if "patch_norm" in results and results["patch_norm"]:
        out["patch_norm_mean"] = float(np.mean(results["patch_norm"]))
    if "act_norm" in results and results["act_norm"]:
        out["act_norm_mean"]   = float(np.mean(results["act_norm"]))
    return out


def print_report(layer: int, probe_r2: float, summary: dict):
    print("")
    print("═" * 78)
    print("CAUSAL INTERVENTION — SUMMARY")
    print("═" * 78)
    print(f"  Patch layer:                       L{layer}")
    print(f"  Probe R² (this layer, held-out):   {probe_r2:.4f}")
    print(f"  Test positions:                    {summary['n']}")
    if "patch_norm_mean" in summary:
        ratio = summary["patch_norm_mean"] / max(summary["act_norm_mean"], 1e-9)
        print(f"  Mean ||patch|| / ||activation||:   "
              f"{summary['patch_norm_mean']:.2f} / {summary['act_norm_mean']:.2f}  =  "
              f"{ratio:.3f}")
        if ratio > 1.0:
            print(f"    NOTE: patch L2 exceeds the activation's L2 — the patch likely")
            print(f"    overwhelms the rest of the residual. Try --patch_scale < 1.")
    print("")
    print("  Mean probability mass over GRAPH NEIGHBORS of:")
    print(f"    P(A's neighbors)  unpatched        : {summary['unpatched_PA_mean']:.4f}")
    print(f"    P(A's neighbors)  target-patched   : {summary['target_PA_mean']:.4f}   "
          f"Δ = {summary['delta_PA_target_vs_unpatched']:+.4f}")
    print(f"    P(A's neighbors)  random-patched   : {summary['random_PA_mean']:.4f}")
    print("")
    print(f"    P(B's neighbors)  unpatched        : {summary['unpatched_PB_mean']:.4f}")
    print(f"    P(B's neighbors)  target-patched   : {summary['target_PB_mean']:.4f}   "
          f"Δ = {summary['delta_PB_target_vs_unpatched']:+.4f}")
    print(f"    P(B's neighbors)  random-patched   : {summary['random_PB_mean']:.4f}   "
          f"Δ = {summary['delta_PB_random_vs_unpatched']:+.4f}")
    print("")
    print("  Effect-size summary:")
    print(f"    target-patch lift on P(B's nbrs) over random patch: "
          f"{summary['target_PB_mean'] - summary['random_PB_mean']:+.4f}")
    print(f"    target-patch lift on P(B's nbrs) over unpatched   : "
          f"{summary['delta_PB_target_vs_unpatched']:+.4f}")
    print(f"    per-position: target_PB > random_PB in            : "
          f"{summary['frac_target_beats_random_on_PB']*100:.1f}% of cases")
    print("")
    print("  Interpretation:")
    print("    If target_PB rises and random_PB does not, the model causally USES")
    print("    its emergent location representation for next-hop prediction. The")
    print("    bigger the gap (target Δ ≫ random Δ), the stronger the causal link.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--layer", type=int, default=3,
                   help="residual-stream layer to probe + patch at (0 = embed, "
                        "1..n_layer = after that transformer block)")
    p.add_argument("--n_positions", type=int, default=200,
                   help="number of (position, target_B) trials to evaluate")
    p.add_argument("--probe_train_positions", type=int, default=15_000,
                   help="positions used to TRAIN the linear probe whose W defines "
                        "the patch direction")
    p.add_argument("--probe_epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16,
                   help="positions per intervention batch (3 forward passes each)")
    p.add_argument("--patch_scale", type=float, default=1.0,
                   help="scalar multiplier for the pseudoinverse-derived patch. "
                        "1.0 = make the probe decode exactly y_target. <1 = "
                        "undershoot the location shift but disturb less of the "
                        "residual; useful when patch_scale=1.0 produces patches "
                        "larger than the activation's L2 norm.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"device: {device}")

    data_dir = Path(args.data_dir)

    # ── Trained model from checkpoint ──
    print(f"\nLoading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  iter={ckpt.get('iter','?')}  val_ppl={ckpt.get('val_perplexity', float('nan')):.4f}  "
          f"vocab_size={config.vocab_size}  n_layer={config.n_layer}")

    assert 0 <= args.layer <= config.n_layer, \
        f"--layer {args.layer} out of range [0, {config.n_layer}]"

    # ── Coords + meta + graph ──
    coords_xy, center_lat, center_lon = load_coords_planar(data_dir)
    print(f"\ncoords: {(~torch.isnan(coords_xy[:,0])).sum().item():,} nodes "
          f"centered at ({center_lat:.4f}, {center_lon:.4f})")

    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dtype = np.dtype(meta["dtype"])
    itos = meta["itos"]
    stoi = meta["stoi"]

    G = pickle.loads((data_dir / "graph.gpickle").read_bytes())
    print(f"graph: {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges")

    # ── Build probe-training dataset at this layer ──
    val_stream = np.asarray(np.memmap(data_dir / "val.bin", dtype=dtype, mode="r"))
    gen_stream = np.asarray(np.memmap(data_dir / "gen.bin", dtype=dtype, mode="r"))
    combined   = np.concatenate([val_stream, gen_stream])

    print(f"\nBuilding probe dataset at layer L{args.layer} "
          f"({args.probe_train_positions:,} positions) ...")
    t0 = time.time()
    X_probe, y_probe, _ = collect_layer_acts_for_probe(
        model, combined, coords_xy.cpu(),
        config.block_size, args.layer, args.probe_train_positions, device,
        rng_seed=args.seed,
    )
    print(f"  done ({time.time()-t0:.1f}s)")

    # ── Train linear probe at this layer ──
    n = len(y_probe)
    perm = np.random.default_rng(args.seed).permutation(n)
    n_train = int(n * 0.8)
    Xtr, Xte = X_probe[perm[:n_train]], X_probe[perm[n_train:]]
    ytr, yte = y_probe[perm[:n_train]], y_probe[perm[n_train:]]

    print(f"\nTraining linear probe at L{args.layer} for patch directions...")
    t0 = time.time()
    probe = LinearProbe(X_probe.shape[1])
    res = train_and_evaluate_probe(
        probe, Xtr, ytr, Xte, yte, device,
        lr=1e-3, weight_decay=1e-3, epochs=args.probe_epochs,
    )
    print(f"  done ({time.time()-t0:.1f}s)  "
          f"held-out R²={res['r2']:.4f}  median_m={res['median_m']:.1f}")

    # ── Run the causal intervention ──
    print(f"\nRunning causal intervention (n_positions={args.n_positions}, "
          f"layer L{args.layer})...")
    results = run_intervention(
        model, probe, args.layer, combined, config.block_size, G, itos, stoi,
        coords_xy.to(device).float(),
        n_positions=args.n_positions, device=device, rng_seed=args.seed,
        batch_size=args.batch_size, patch_scale=args.patch_scale,
    )

    summary = summarize(results)
    print_report(args.layer, res["r2"], summary)


if __name__ == "__main__":
    main()
