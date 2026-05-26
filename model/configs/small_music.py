"""
LatentWorldsGPT — Milestone 2 (music) config.

A smaller, more heavily regularized config than `small.py`, sized for the
music corpora where train.bin has ≤ ~150k tokens and the vocab is ~60.
The `small.py` config (10.7M params) catastrophically overfits on Bach
chorale data — val loss climbs while train loss drops past iter 500.
This config exists to:

  - Use ~3x fewer params (matches the smaller corpus)
  - Apply heavier dropout (0.3 vs 0.1)
  - Stop earlier (1000 iters vs 2000) since the corpus is small

  ┌─────────────────────────────────────────────────────────────────┐
  │  Purpose: clean fits on music corpora, no overfit.               │
  │  Target:  Bach chorales + expanded music_expanded corpus.        │
  │  Runtime: ~7-15 min on Apple MPS.                                │
  └─────────────────────────────────────────────────────────────────┘
"""

# ── Architecture (smaller than small.py to match corpus size) ──
block_size = 256
n_layer    = 3      # vs 6 in small.py
n_head     = 3      # vs 6 in small.py — n_embd must be divisible by this
n_embd     = 192    # vs 384 in small.py
dropout    = 0.3    # vs 0.1 in small.py — heavier regularization
bias       = False

# ── Training schedule ──
# 1000 iters at batch 64 × block 256 = 16k token-examples per iter, ~16M
# total. With ~150k train tokens that's ~100 visits/token; combined with
# stronger regularization this should fit without overfitting.
max_iters                   = 1_000
warmup_iters                = 100
lr_decay_iters              = 1_000
batch_size                  = 64
gradient_accumulation_steps = 1

# ── Optimizer ──
learning_rate = 3e-4
min_lr        = 3e-5
weight_decay  = 0.2       # heavier weight decay than small.py's 0.1
beta1         = 0.9
beta2         = 0.95
grad_clip     = 1.0

# ── Eval / logging ──
eval_interval = 100        # eval more frequently to catch overfit earlier
eval_iters    = 50
log_interval  = 10
