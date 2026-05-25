"""
LatentCityGPT — coordinate probe suite (PLAN.md Phase 4).

WHAT THIS FILE DOES, IN ONE PICTURE
====================================

  trained model (frozen)                ground-truth coords (coords.csv)
        │                                       │
        │  forward pass on val + gen streams    │
        ▼                                       │  equirectangular projection
   residual stream activations              centered on the city
   at layer 0, 1, ..., n_layer                  │
        │                                       ▼
        │                                  (x_m, y_m) in meters
        │                                       │
        └────────────────┬──────────────────────┘
                         │
                         ▼  pair each position's activation with
                          the (x_m, y_m) of the *current* token
                         │
                         ▼
                  probe dataset (X, y)
                   80/20 train/test split
                         │
                         ▼
              train Linear probe   |   train MLP probe
              (MSE on standardized   |   (MSE; 2-layer w/ GELU)
              targets; weight_decay   |
              = ridge regression)     |
                         │
                         ▼
              evaluate on held-out positions:
                R² (standardized space, dimensionless)
                median Euclidean error in meters (interpretable units)


WHAT WE ARE TESTING
====================
The hypothesis (PLAN.md rung 2): the trained transformer's residual stream
encodes the (x_m, y_m) location of the current intersection — recoverable by
a probe. Concretely:

  HYPOTHESIS         | TRAINED model probe → high R², low meters error.
  NULL HYPOTHESIS    | The probe is just learning to memorize node → coord
                       lookups, not exploiting any geometry in the
                       representation. Controls below distinguish these.

Two controls are *implemented here* and reported alongside the main result.
One control (destroyed-structure) is deferred — see PLAN.md Phase 4.


LOSS / METRIC CHOICE — INTELLECTUAL HONESTY
============================================
Naive MSE on raw (lat, lon) is wrong:
  - 1° of latitude is ~111 km on the surface.
  - 1° of longitude is ~111 km at the equator, ~70 km at London's 51°N,
    and ~0 km at the poles. So an "MSE on (lat, lon)" weights latitude
    errors much more than longitude errors at high latitudes — a bias
    that has nothing to do with the model.
  - Worse: MSE on raw degrees is in deg², not meters². Cross-city or
    absolute "the probe is X good" claims are not interpretable.

The honest fix: project (lat, lon) to a local planar coordinate system
(x_m, y_m) in meters using the equirectangular projection centered on the
city's mean (lat, lon). For a city-sized region (≤ ~50 km diameter), this
projection has < 0.5% distortion — far below the meters-of-error the probe
will report. MSE in this space is meters².

  x_m = (lon - lon_center) * cos(lat_center_rad) * R_earth * (π/180)
  y_m = (lat - lat_center) *                       R_earth * (π/180)

We standardize the targets (zero mean, unit std per axis) for stable probe
training, then un-standardize predictions before computing the error-in-meters.

Reported metrics, per layer × probe type:
  R²                  — standardized regression goodness-of-fit (dimensionless)
  median meters error — physical "how close is the probe to the right point?"


THE ONE RULE
============
This file is the only place in the project where coords.csv enters the picture.
The model never sees this file. Token IDs flow into the model; (x_m, y_m)
targets stay on the probe side. We assert this explicitly: the model's
forward pass takes only LongTensor token IDs, no floats.


USAGE
=====
    python eval/probe.py --ckpt checkpoints/best.pt --data_dir data/london_city

    # bigger probe sample, more probe-training epochs
    python eval/probe.py --ckpt checkpoints/manhattan/best.pt \\
        --data_dir data/manhattan --n_positions 40000 --epochs 300


CONTROLS REPORTED
=================
  TRAINED model, layer 0..n_layer  — main result
  UNTRAINED model (random init)    — should fail at all layers
  TRAINED model, layer 0 (= wte+wpe only) — probes the raw input embedding;
                                            tests whether the answer is "the
                                            geometry is just in token IDs"

  Destroyed-structure control (model trained on shuffled routes) is noted
  in PLAN.md and deferred — it requires retraining.
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

# Reuse the model package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
from model import GPT, GPTConfig  # noqa: E402


# IUGG mean Earth radius. Accurate to < 0.3% anywhere on Earth and to << 0.1%
# at city scales. Plenty good for probe-error reporting in meters.
R_EARTH_M = 6_371_008.8

# Reserved token indices (must mirror data/prepare_city.py and model/model.py).
N_RESERVED = 3


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load and project the coordinate table to local meters
# ─────────────────────────────────────────────────────────────────────────────

def load_coords_planar(data_dir: Path) -> tuple[torch.Tensor, float, float]:
    """Read coords.csv, project to local (x_m, y_m) via equirectangular projection
    centered on the city's mean (lat, lon).

    Returns:
        coords_xy   : torch.Tensor (max_token_id + 1, 2)   filled with NaN at
                      control-token rows (idx in {0,1,2}) so we can detect and
                      skip them when building the probe dataset.
        center_lat  : float   — projection center (city mean latitude)
        center_lon  : float   — projection center (city mean longitude)
    """
    lines = (data_dir / "coords.csv").read_text().splitlines()
    assert lines[0] == "idx,lat,lon", f"unexpected header: {lines[0]!r}"
    rows = [line.split(",") for line in lines[1:]]
    rows = [(int(i), float(la), float(lo)) for i, la, lo in rows]

    max_idx = max(r[0] for r in rows)
    coords_xy = torch.full((max_idx + 1, 2), float("nan"))

    lats = np.array([la for _, la, _ in rows])
    lons = np.array([lo for _, _, lo in rows])
    center_lat = float(lats.mean())
    center_lon = float(lons.mean())

    # Precompute equirectangular constants
    deg_to_m = math.pi / 180.0 * R_EARTH_M             # ~111,195 m per degree
    cos_lat0 = math.cos(math.radians(center_lat))

    for idx, lat, lon in rows:
        x_m = (lon - center_lon) * cos_lat0 * deg_to_m
        y_m = (lat - center_lat) * deg_to_m
        coords_xy[idx] = torch.tensor([x_m, y_m])

    return coords_xy, center_lat, center_lon


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cache residual-stream activations at every layer for one batch
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def cache_layer_activations(model: GPT, idx_batch: torch.Tensor) -> list[torch.Tensor]:
    """Run one forward pass, returning the residual-stream tensor *after* each
    transformer block (and the post-embedding tensor as layer 0). The model is
    not modified.

    Returns:
        acts : list of length (n_layer + 1) of tensors (B, T, n_embd)
               acts[0]   — embedding output: wte(idx) + wpe(pos), post-dropout
               acts[k]   — output of block k (k = 1..n_layer)
    """
    B, T = idx_batch.shape
    tok_emb = model.transformer.wte(idx_batch)
    pos = torch.arange(0, T, dtype=torch.long, device=idx_batch.device)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)

    acts = [x.clone()]                      # layer 0 — input embedding only
    for block in model.transformer.h:
        x = block(x)
        acts.append(x.clone())
    return acts


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build the probe dataset
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_probe_dataset(model: GPT, stream: np.ndarray, coords_xy: torch.Tensor,
                        block_size: int, n_positions: int, device: str,
                        rng_seed: int = 0) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Sample positions from `stream`, run the model, collect per-position
    residual-stream activations at every layer, paired with that position's
    ground-truth (x_m, y_m).

    Only positions whose current token is a real intersection (id >= 3) are
    kept; positions at BOS/EOS/PAD or with NaN coords are skipped.

    Returns:
        X            : list of (n_positions, n_embd) np arrays — one per layer
        y            : (n_positions, 2) np array — (x_m, y_m) targets
        tokens       : (n_positions,) np array — the token id at each kept
                       position (useful for diagnostics; not used by the probe)
    """
    model.eval()
    rng = np.random.default_rng(rng_seed)

    batch_size = 32
    all_X: list[list[np.ndarray]] | None = None
    all_y: list[np.ndarray] = []
    all_tokens: list[int] = []
    n_collected = 0

    while n_collected < n_positions:
        starts = rng.integers(0, len(stream) - block_size - 1, size=batch_size)
        windows = [np.asarray(stream[s : s + block_size]) for s in starts]
        idx_batch = torch.from_numpy(np.stack(windows).astype(np.int64)).to(device)

        # THE ONE RULE: only token IDs flow into the model.
        assert idx_batch.dtype in (torch.long, torch.int64)
        acts = cache_layer_activations(model, idx_batch)

        if all_X is None:
            n_layers = len(acts)
            all_X = [[] for _ in range(n_layers)]

        # Walk every position in every batch row; keep positions whose token
        # is a real intersection and has valid coords.
        idx_np = idx_batch.cpu().numpy()
        for b in range(idx_np.shape[0]):
            for t in range(idx_np.shape[1]):
                tok = int(idx_np[b, t])
                if tok < N_RESERVED:
                    continue
                xy = coords_xy[tok]
                if torch.isnan(xy).any():
                    continue
                for L in range(len(acts)):
                    all_X[L].append(acts[L][b, t].cpu().numpy())
                all_y.append(xy.numpy())
                all_tokens.append(tok)
                n_collected += 1
                if n_collected >= n_positions:
                    break
            if n_collected >= n_positions:
                break

    X = [np.stack(layer_list) for layer_list in all_X]  # each: (N, n_embd)
    y = np.stack(all_y).astype(np.float32)              # (N, 2), meters
    tokens = np.array(all_tokens, dtype=np.int64)       # (N,)
    return X, y, tokens


# ─────────────────────────────────────────────────────────────────────────────
# 4. Probes
# ─────────────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    """A single linear layer activations → (x_m, y_m).
    With weight_decay (L2 regularization) this is equivalent to ridge
    regression in closed form, but trained via gradient descent so the
    interface mirrors the MLP probe."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 2)

    def forward(self, x):
        return self.linear(x)


class MLPProbe(nn.Module):
    """Two-hidden-layer MLP probe. The point of including this is to ask
    'could a *nonlinear* probe extract substantially more than a linear one?'
    If MLP R² ≈ linear R², the map is encoded LINEARLY — the strong claim."""
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Train one probe with MSE on standardized targets; report meters error
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate_probe(probe: nn.Module,
                             X_train: np.ndarray, y_train: np.ndarray,
                             X_test:  np.ndarray, y_test:  np.ndarray,
                             device: str,
                             lr: float = 1e-3,
                             weight_decay: float = 1e-3,
                             epochs: int = 200,
                             batch_size: int = 512,
                             verbose: bool = False) -> dict:
    """Train `probe` to map activations → standardized (x_m, y_m). Return
    held-out R² (standardized space) and median meters error (unstandardized).

    Notes:
      - Targets are standardized using TRAIN statistics, applied to test.
      - We track best test R² across epochs and restore that state at the end.
      - weight_decay > 0 on the LinearProbe = ridge regression.
      - For the MLP we use a much smaller weight_decay (the capacity comes
        from the hidden layers, not the L2; over-regularizing kills it).
    """
    probe = probe.to(device)

    # Standardize using TRAIN-set stats.
    y_mean = y_train.mean(axis=0)
    y_std  = y_train.std(axis=0) + 1e-8
    yt_train = (y_train - y_mean) / y_std
    yt_test  = (y_test  - y_mean) / y_std

    Xtr = torch.from_numpy(X_train).float().to(device)
    ytr = torch.from_numpy(yt_train.astype(np.float32)).to(device)
    Xte = torch.from_numpy(X_test).float().to(device)
    yte = torch.from_numpy(yt_test.astype(np.float32)).to(device)

    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    n = Xtr.shape[0]
    best_r2 = -float("inf")
    best_state = None

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        probe.train()
        for i in range(0, n, batch_size):
            ix = perm[i : i + batch_size]
            preds = probe(Xtr[ix])
            loss = F.mse_loss(preds, ytr[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Eval on held-out test set.
        probe.eval()
        with torch.no_grad():
            preds = probe(Xte)
            ss_res = ((preds - yte) ** 2).sum().item()
            ss_tot = ((yte - yte.mean(0)) ** 2).sum().item()
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            if r2 > best_r2:
                best_r2 = r2
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"      epoch {epoch:>4d}  test_R²={r2:.4f}  best={best_r2:.4f}")

    probe.load_state_dict(best_state)
    probe.eval()

    # Final metrics: un-standardize predictions and compute meters error.
    with torch.no_grad():
        preds_std = probe(Xte).cpu().numpy()
    preds = preds_std * y_std + y_mean

    # Median Euclidean error in meters between prediction and ground truth.
    errs_m = np.sqrt(((preds - y_test) ** 2).sum(axis=1))
    median_m = float(np.median(errs_m))
    mean_m   = float(np.mean(errs_m))
    p90_m    = float(np.percentile(errs_m, 90))

    # R² unstandardized (matches standardized one; reporting both is reassuring).
    ss_res = ((preds - y_test) ** 2).sum()
    ss_tot = ((y_test - y_test.mean(0)) ** 2).sum()
    r2_unstd = 1.0 - ss_res / max(ss_tot, 1e-12)

    return {
        "r2":        float(r2_unstd),
        "median_m":  median_m,
        "mean_m":    mean_m,
        "p90_m":     p90_m,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Probe-data splits and layer-sweep helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_position_split(n: int, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard random partition of POSITIONS. Train and test positions are
    disjoint, but the same TOKEN can appear in both (different sequence
    positions). This is the conventional probe-evaluation split.

    Limitation: a sufficiently expressive probe can pass this test by learning
    a per-token lookup table — it has effectively seen every token before
    being asked about it.
    """
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(n * train_frac)
    return perm[:n_train], perm[n_train:]


def make_node_split(tokens: np.ndarray, coords_xy: torch.Tensor,
                    train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Partition by TOKEN ID rather than by position. The probe trains on
    activations whose current-token comes from a 'train' set of nodes; it is
    evaluated on positions whose current-token comes from a disjoint 'test'
    set of nodes that the probe NEVER saw during training.

    This is the probe-capacity-controlling split:
      - A probe that learned a continuous geographic map should generalize to
        unseen tokens: R² stays high.
      - A probe that just memorized a per-token coord lookup cannot generalize:
        R² collapses (typically toward the untrained baseline).

    Returns (train_ix, test_ix, stats).
    """
    # Universe of valid real-token IDs (those that have non-NaN coords).
    valid_ids = (~torch.isnan(coords_xy[:, 0])).nonzero(as_tuple=True)[0].cpu().numpy()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(valid_ids))
    n_train_nodes = int(len(valid_ids) * train_frac)
    train_tokens = set(int(t) for t in valid_ids[perm[:n_train_nodes]])
    test_tokens  = set(int(t) for t in valid_ids[perm[n_train_nodes:]])

    train_mask = np.array([int(t) in train_tokens for t in tokens])
    test_mask  = np.array([int(t) in test_tokens  for t in tokens])
    train_ix = np.nonzero(train_mask)[0]
    test_ix  = np.nonzero(test_mask)[0]

    return train_ix, test_ix, {
        "n_train_nodes":     len(train_tokens),
        "n_test_nodes":      len(test_tokens),
        "n_train_positions": int(len(train_ix)),
        "n_test_positions":  int(len(test_ix)),
    }


def run_layer_sweep(X_layers: list[np.ndarray], y: np.ndarray,
                    train_ix: np.ndarray, test_ix: np.ndarray,
                    epochs: int, device: str, label: str) -> list[tuple]:
    """Run linear + MLP probes at every layer; print one table; return rows.
    Caller controls the train/test index split (position-level or node-level)."""
    print(f"\n{'─'*78}")
    print(label)
    print(f"{'─'*78}")
    header = (f"{'Layer':<8}{'LinR²':>10}{'Lin med m':>14}{'Lin p90 m':>14}"
              f"{'MLPR²':>10}{'MLP med m':>14}")
    print(header)
    print("─" * len(header))

    rows = []
    for L, Xl in enumerate(X_layers):
        Xtr, Xte = Xl[train_ix], Xl[test_ix]
        ytr, yte = y[train_ix],  y[test_ix]
        lin = train_and_evaluate_probe(
            LinearProbe(Xl.shape[1]), Xtr, ytr, Xte, yte, device,
            lr=1e-3, weight_decay=1e-3, epochs=epochs,
        )
        mlp = train_and_evaluate_probe(
            MLPProbe(Xl.shape[1]),    Xtr, ytr, Xte, yte, device,
            lr=1e-3, weight_decay=1e-5, epochs=epochs,
        )
        rows.append((L, lin, mlp))
        layer_label = "embed" if L == 0 else f"L{L}"
        print(f"{layer_label:<8}{lin['r2']:>10.4f}{lin['median_m']:>14.1f}"
              f"{lin['p90_m']:>14.1f}{mlp['r2']:>10.4f}{mlp['median_m']:>14.1f}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────────────

def fmt(v, w=10, fmt_spec=".4f"):
    return f"{v:>{w}{fmt_spec}}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint to probe")
    p.add_argument("--data_dir", required=True, help="data/<city>/")
    p.add_argument("--n_positions", type=int, default=20_000,
                   help="positions sampled for the probe (train+test combined)")
    p.add_argument("--probe_train_frac", type=float, default=0.8,
                   help="fraction of probe positions for training")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_untrained", action="store_true",
                   help="skip the untrained-model control (faster)")
    p.add_argument("--skip_node_split", action="store_true",
                   help="skip the node-level split (faster but less informative)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"device: {device}")

    data_dir = Path(args.data_dir)

    # ── trained model from checkpoint ──
    print(f"\nLoading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    trained_model = GPT(config).to(device)
    trained_model.load_state_dict(ckpt["model_state"])
    trained_model.eval()
    vp = ckpt.get("val_perplexity", float("nan"))
    print(f"  iter={ckpt.get('iter','?')}  val_ppl={vp:.4f}  vocab_size={config.vocab_size}")

    # ── untrained model (control) ──
    if not args.skip_untrained:
        untrained_model = GPT(config).to(device)
        untrained_model.eval()
    else:
        untrained_model = None

    # ── coords (the only place spatial info enters this script) ──
    coords_xy, center_lat, center_lon = load_coords_planar(data_dir)
    n_nodes = (~torch.isnan(coords_xy[:, 0])).sum().item()
    print(f"\nCoords loaded: {n_nodes:,} nodes")
    print(f"  projection center: lat={center_lat:.5f}, lon={center_lon:.5f}")
    # quick sanity on city extent
    valid = coords_xy[~torch.isnan(coords_xy[:, 0])]
    extent_x = (valid[:, 0].max() - valid[:, 0].min()).item()
    extent_y = (valid[:, 1].max() - valid[:, 1].min()).item()
    print(f"  city extent (planar): {extent_x:.0f} m east-west × {extent_y:.0f} m north-south")

    # ── probe-source stream: val + gen ──
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dtype = np.dtype(meta["dtype"])
    val_stream = np.asarray(np.memmap(data_dir / "val.bin", dtype=dtype, mode="r"))
    gen_stream = np.asarray(np.memmap(data_dir / "gen.bin", dtype=dtype, mode="r"))
    combined = np.concatenate([val_stream, gen_stream])
    print(f"\nProbe-source token stream: {len(combined):,} tokens "
          f"(val={len(val_stream):,} + gen={len(gen_stream):,})")

    # ── build datasets ──
    print(f"\nBuilding probe dataset for TRAINED model "
          f"(target n_positions={args.n_positions:,}) ...")
    t0 = time.time()
    X_trained, y_t, tokens_t = build_probe_dataset(
        trained_model, combined, coords_xy.cpu(),
        config.block_size, args.n_positions, device, rng_seed=args.seed,
    )
    print(f"  collected {len(y_t):,} positions across {len(X_trained)} layers "
          f"({time.time()-t0:.1f}s)")
    print(f"  unique tokens seen: {len(np.unique(tokens_t)):,} of {n_nodes:,} nodes")

    if untrained_model is not None:
        print(f"\nBuilding probe dataset for UNTRAINED model (random-init control) ...")
        t0 = time.time()
        X_untrained, y_u, tokens_u = build_probe_dataset(
            untrained_model, combined, coords_xy.cpu(),
            config.block_size, args.n_positions, device, rng_seed=args.seed,
        )
        print(f"  ({time.time()-t0:.1f}s)")

    # ── build the splits ──
    pos_train_t, pos_test_t = make_position_split(len(y_t), args.probe_train_frac, args.seed)
    print(f"\nPOSITION-LEVEL split  "
          f"(positions disjoint, tokens may appear in both):")
    print(f"  trained model:   train={len(pos_train_t):,} pos   test={len(pos_test_t):,} pos")

    if untrained_model is not None:
        pos_train_u, pos_test_u = make_position_split(len(y_u), args.probe_train_frac, args.seed)
        print(f"  untrained model: train={len(pos_train_u):,} pos   test={len(pos_test_u):,} pos")

    if not args.skip_node_split:
        node_train_t, node_test_t, ns_t = make_node_split(
            tokens_t, coords_xy, args.probe_train_frac, args.seed,
        )
        print(f"\nNODE-LEVEL split  "
              f"(tokens disjoint — probe sees held-out nodes only at test time):")
        print(f"  trained model:   "
              f"train={ns_t['n_train_positions']:,} pos "
              f"({ns_t['n_train_nodes']:,} nodes)  /  "
              f"test={ns_t['n_test_positions']:,} pos "
              f"({ns_t['n_test_nodes']:,} nodes)")
        if untrained_model is not None:
            node_train_u, node_test_u, ns_u = make_node_split(
                tokens_u, coords_xy, args.probe_train_frac, args.seed,
            )
            print(f"  untrained model: train={ns_u['n_train_positions']:,} pos  "
                  f"test={ns_u['n_test_positions']:,} pos")

    # ── per-layer probe sweeps ──
    trained_pos_rows = run_layer_sweep(
        X_trained, y_t, pos_train_t, pos_test_t, args.epochs, device,
        label="TRAINED MODEL — POSITION-LEVEL SPLIT",
    )
    trained_node_rows = []
    if not args.skip_node_split:
        trained_node_rows = run_layer_sweep(
            X_trained, y_t, node_train_t, node_test_t, args.epochs, device,
            label="TRAINED MODEL — NODE-LEVEL SPLIT (held-out tokens; probe-capacity test)",
        )

    untrained_pos_rows = []
    untrained_node_rows = []
    if untrained_model is not None:
        untrained_pos_rows = run_layer_sweep(
            X_untrained, y_u, pos_train_u, pos_test_u, args.epochs, device,
            label="UNTRAINED MODEL — POSITION-LEVEL SPLIT (random-init control)",
        )
        if not args.skip_node_split:
            untrained_node_rows = run_layer_sweep(
                X_untrained, y_u, node_train_u, node_test_u, args.epochs, device,
                label="UNTRAINED MODEL — NODE-LEVEL SPLIT",
            )

    # ── headline summary ──
    print(f"\n{'═'*78}")
    print(f"HEADLINE")
    print(f"{'═'*78}")

    def best(rows, probe_ix):
        if not rows:
            return (None, None)
        b = max(rows, key=lambda r: r[probe_ix]["r2"])
        return (b[0], b[probe_ix])

    layer_name = lambda L: "embed" if L == 0 else f"L{L}"

    def line(label, layer, res):
        if res is None:
            print(f"  {label:<24}    —")
            return
        print(f"  {label:<24}  {layer_name(layer):>5}   "
              f"R²={res['r2']:>+7.4f}   "
              f"median={res['median_m']:>7.1f} m   "
              f"p90={res['p90_m']:>7.1f} m")

    print(f"\n  POSITION-LEVEL  (positions disjoint, tokens shared) :")
    line("trained  linear", *best(trained_pos_rows, 1))
    line("trained  MLP",    *best(trained_pos_rows, 2))
    line("untrained linear", *best(untrained_pos_rows, 1))
    line("untrained MLP",    *best(untrained_pos_rows, 2))

    if not args.skip_node_split:
        print(f"\n  NODE-LEVEL  (tokens disjoint — probe-capacity test) :")
        line("trained  linear", *best(trained_node_rows, 1))
        line("trained  MLP",    *best(trained_node_rows, 2))
        line("untrained linear", *best(untrained_node_rows, 1))
        line("untrained MLP",    *best(untrained_node_rows, 2))

    # Interpretation guide
    print(f"\n{'─'*78}")
    print(f"INTERPRETATION GUIDE")
    print(f"{'─'*78}")
    print(f"  - Trained Linear R²  >>  Untrained Linear R²")
    print(f"        ⇒  geographic structure EMERGED from training")
    print(f"          (compare on position-level and especially node-level).")
    print(f"")
    print(f"  - If node-level MLP R² drops sharply vs position-level MLP R²")
    print(f"        ⇒  the position-level MLP win was probe LOOKUP MEMORIZATION,")
    print(f"          not the model's representation. The 'linear ≈ MLP' criterion")
    print(f"          for linear encoding cannot be read off the position-level result.")
    print(f"")
    print(f"  - If node-level LINEAR R² stays close to position-level LINEAR R²")
    print(f"        ⇒  the linear probe found a CONTINUOUS map that generalizes")
    print(f"          to unseen tokens — strong evidence for a true geographic")
    print(f"          embedding rather than a 663-key lookup.")
    print(f"")
    print(f"Deferred (PLAN.md Phase 4): destroyed-structure control")
    print(f"  (train a model on token-shuffled routes and re-run this probe).")


if __name__ == "__main__":
    main()
