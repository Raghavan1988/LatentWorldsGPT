"""
viz/overlay.py — Procrustes-aligned cities map overlay.

The headline visualization for the cities domain. For each trained
cities model (real / within-shuffled / global-shuffled), we:

1. Extract the token embeddings (wte) for every real cell token.
2. Fit a 2D linear projection from embedding space onto (lat, lon)
   space using a held-out fraction of cells as training data.
3. Apply the projection to ALL cells and overlay predicted (lat, lon)
   on the true (lat, lon) on the same axes.
4. Procrustes-align predicted points to true points (rotation +
   reflection + uniform scale) so the qualitative shape is
   immediately visible regardless of the linear projection's basis.

Output is a multi-panel figure: real London (true coords only) on the
left, and one decoded panel per checkpoint condition on the right. Per
panel, we also annotate median per-cell distance error (in degrees and
in approximate meters) and a Procrustes disparity score.

The expected story (per `update_phase1.md` and `STATUS_vs_OTHELLO-GPT.md`):
- Real-trained London: predicted overlays closely match the city grid.
- Within-shuffled: still shows geographic clustering (set-membership
  preserved within routes) but without the graph adjacency structure
  — coarser overlay.
- Global-shuffled: essentially noise — the embedding has nothing left
  to align with real coordinates.

Usage:
    python viz/overlay.py \\
        --ckpts checkpoints/best.pt \\
                checkpoints/london_shuffled/best.pt \\
                checkpoints/london_global_shuffled/best.pt \\
        --labels "real" "within-shuffled" "global-shuffled" \\
        --data_dir data/london_city \\
        --out figs/phase5_cities_overlay.png

(All three checkpoints are expected to share the same coords.csv, which
lives in --data_dir.)
"""
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import procrustes

from model.model import GPT, GPTConfig

DEVICE = "cpu"  # CPU is enough — only loading + slicing wte.


def load_coords(data_dir: Path):
    """Return (tokens, coords) — token IDs (>= 3) and corresponding
    (lat, lon) arrays from coords.csv."""
    tokens, coords = [], []
    with open(data_dir / "coords.csv") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            tok, lat, lon = int(parts[0]), float(parts[1]), float(parts[2])
            tokens.append(tok)
            coords.append((lat, lon))
    return np.array(tokens), np.array(coords)


def load_wte(ckpt_path: Path):
    """Return the token embedding matrix (vocab_size, n_embd) for the
    given checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.transformer.wte.weight.detach().numpy()


def fit_linear_decoder(embeddings: np.ndarray, coords: np.ndarray, train_frac=0.8, rng=None):
    """Fit a least-squares linear map from embedding → 2D coord on a
    train subset; return (W, b, train_idx, test_idx).

    embedding (n_cells, n_embd) → coord (n_cells, 2).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(embeddings)
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    X_train = embeddings[train_idx]
    Y_train = coords[train_idx]
    # Add bias column
    X_aug = np.hstack([X_train, np.ones((len(X_train), 1))])
    # Solve W_aug = (X_aug^T X_aug)^{-1} X_aug^T Y_train, with regularization for stability
    lam = 1e-3
    A = X_aug.T @ X_aug + lam * np.eye(X_aug.shape[1])
    W_aug = np.linalg.solve(A, X_aug.T @ Y_train)
    W, b = W_aug[:-1], W_aug[-1]
    return W, b, train_idx, test_idx


def apply_decoder(embeddings, W, b):
    return embeddings @ W + b


def procrustes_align(predicted, truth):
    """Apply Procrustes alignment from `predicted` onto `truth`.
    Returns the aligned `predicted_aligned` and a disparity score
    (lower is better; 0 means identical after rotation/reflection/scale)."""
    truth_std, pred_std, disparity = procrustes(truth, predicted)
    # `procrustes` from scipy returns standardized matrices (centered + scaled).
    # We rescale pred_std back to truth's original location and scale for plotting.
    truth_mean = truth.mean(axis=0)
    truth_scale = np.linalg.norm(truth - truth_mean)
    pred_aligned = pred_std * truth_scale + truth_mean
    return pred_aligned, disparity


def degrees_to_meters_factor(lat0):
    """At latitude lat0 (degrees), 1 degree of latitude ≈ 111,320 m;
    1 degree of longitude ≈ 111,320 * cos(lat0) m."""
    return 111_320.0, 111_320.0 * np.cos(np.radians(lat0))


def panel_decoder_overlay(ax, embeddings, coords, label, rng):
    """Fit linear decoder, apply, Procrustes-align, plot overlay."""
    W, b, train_idx, test_idx = fit_linear_decoder(embeddings, coords, rng=rng)
    pred = apply_decoder(embeddings, W, b)
    pred_aligned, disparity = procrustes_align(pred, coords)

    # Compute median error in meters
    errs_deg = np.linalg.norm(pred_aligned - coords, axis=1)
    lat0 = coords.mean(axis=0)[0]
    m_per_deg_lat, m_per_deg_lon = degrees_to_meters_factor(lat0)
    errs_m = np.sqrt(
        (errs_deg * m_per_deg_lat) ** 2  # crude (errs is already a euclidean norm)
    )
    # More carefully: per-axis errors in degrees, convert each
    dlat = (pred_aligned[:, 0] - coords[:, 0]) * m_per_deg_lat
    dlon = (pred_aligned[:, 1] - coords[:, 1]) * m_per_deg_lon
    errs_m = np.sqrt(dlat ** 2 + dlon ** 2)

    ax.scatter(coords[:, 1], coords[:, 0], s=4, c="black", alpha=0.5, label="real")
    ax.scatter(pred_aligned[:, 1], pred_aligned[:, 0],
               s=4, c="tab:orange", alpha=0.6, label="decoded")
    # connect each real with its prediction by a thin grey line
    for r, p in zip(coords, pred_aligned):
        ax.plot([r[1], p[1]], [r[0], p[0]], color="grey", alpha=0.15, linewidth=0.4)
    ax.set_title(
        f"{label}\nmedian err = {np.median(errs_m):.0f} m   "
        f"Procrustes disparity = {disparity:.4f}",
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return disparity, np.median(errs_m)


def panel_real_only(ax, coords, label):
    ax.scatter(coords[:, 1], coords[:, 0], s=4, c="black")
    ax.set_title(f"{label}\n(true coordinates)", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(True, alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="One or more checkpoint paths.")
    p.add_argument("--labels", nargs="+", required=True,
                   help="Label per checkpoint (e.g., 'real' 'within-shuffled' 'global-shuffled').")
    p.add_argument("--data_dir", required=True,
                   help="Cities data dir containing coords.csv.")
    p.add_argument("--out", required=True,
                   help="Output figure path.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    assert len(args.ckpts) == len(args.labels), \
        "--ckpts and --labels must have equal length"

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading coords from {data_dir / 'coords.csv'} ...")
    tokens, coords = load_coords(data_dir)
    print(f"  {len(tokens)} cells with coords")

    rng = np.random.default_rng(args.seed)
    n_panels = 1 + len(args.ckpts)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6.5))
    if n_panels == 1:
        axes = [axes]

    panel_real_only(axes[0], coords, "Real London (ground truth)")

    for i, (ckpt_path, label) in enumerate(zip(args.ckpts, args.labels)):
        print(f"\nLoading {ckpt_path} ({label}) ...")
        wte = load_wte(Path(ckpt_path))
        # Pull rows corresponding to real cell tokens
        embeddings = wte[tokens]
        print(f"  embeddings shape: {embeddings.shape}")
        disp, err_m = panel_decoder_overlay(
            axes[i + 1], embeddings, coords, label, rng,
        )
        print(f"  Procrustes disparity: {disp:.4f}  median error: {err_m:.0f} m")

    fig.suptitle(
        "Cities domain — Procrustes-aligned overlay of decoded coordinates on real London",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
