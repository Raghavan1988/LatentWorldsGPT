#!/usr/bin/env bash
# Post-Phase-2 pipeline: waits for the symgroup runner, then runs Phase
# 3-d/e/f (DLA + logit lens + zero-ablation) and Phase 4-c maze training
# on all 3 conditions.
#
# RESUMABLE via skip-if-complete sentinels. Safe to interrupt and re-run.
set -uo pipefail

PHASE2_PID="${1:-}"   # symgroup runner PID, optional
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ────────────────────────────────────────────────────────────────────
# (0) Wait for Phase 2 (symgroup) to finish
# ────────────────────────────────────────────────────────────────────
if [[ -n "$PHASE2_PID" ]]; then
    log "Waiting for Phase 2 (symgroup) runner PID=$PHASE2_PID to finish ..."
    while kill -0 "$PHASE2_PID" 2>/dev/null; do sleep 60; done
    log "Phase 2 runner finished."
fi

# ────────────────────────────────────────────────────────────────────
# (1) Phase 3-d/e/f — DLA + logit lens + zero-ablation
# ────────────────────────────────────────────────────────────────────
log "=== Phase 3-d/e/f: complementary causal-interp ==="
bash repro/phase3_complementary_interp.sh
log "=== Phase 3-d/e/f COMPLETE ==="

# ────────────────────────────────────────────────────────────────────
# (2) Phase 4-c — train maze models on 3 conditions
# ────────────────────────────────────────────────────────────────────
log "=== Phase 4-c: maze training on 3 conditions ==="

train_maze() {
    local data_dir="$1"
    local out_dir="$2"
    if [ -f "$out_dir/best.pt" ]; then
        log "  SKIP $out_dir (best.pt already exists)"
        return 0
    fi
    log "  TRAIN $data_dir -> $out_dir"
    python -u model/train.py \
        --config model/configs/small_maze.py \
        --data_dir "$data_dir" \
        --out_dir "$out_dir" \
        --seed 0 \
        > "$out_dir/train.log" 2>&1 || log "  WARN training failed for $data_dir"
}

mkdir -p checkpoints/maze_8x8 checkpoints/maze_8x8_within_shuffled checkpoints/maze_8x8_global_shuffled

train_maze data/maze_8x8 checkpoints/maze_8x8
train_maze data/maze_8x8_within_shuffled checkpoints/maze_8x8_within_shuffled
train_maze data/maze_8x8_global_shuffled checkpoints/maze_8x8_global_shuffled

log "=== Phase 4-c TRAINING COMPLETE ==="

# ────────────────────────────────────────────────────────────────────
# (3) Phase 4-d: maze probe + transplant (multi-seed) on 3 conditions
# ────────────────────────────────────────────────────────────────────
log "=== Phase 4-d: maze probe + transplant on 3 conditions ==="

OUT4D="checkpoints/multiseed_phase4d"
mkdir -p "$OUT4D"

run_unit() {
    local out="$1"; local sentinel="$2"; shift 2
    if [ -f "$out" ] && grep -q "$sentinel" "$out" 2>/dev/null; then
        log "  SKIP $out"
        return 0
    fi
    local tmp="${out}.tmp"
    log "  RUN  $out"
    python -u "$@" > "$tmp" 2>&1
    if [ $? -eq 0 ] && grep -q "$sentinel" "$tmp" 2>/dev/null; then
        mv "$tmp" "$out"
    else
        log "  WARN $out failed; leaving tmp at $tmp"
    fi
}

for cond in "" "_within_shuffled" "_global_shuffled"; do
    ckpt="checkpoints/maze_8x8${cond}/best.pt"
    data="data/maze_8x8${cond}"
    tag="maze${cond}"
    if [ ! -f "$ckpt" ]; then
        log "  SKIP $tag (no checkpoint)"
        continue
    fi
    # Probe (5 seeds, 4 targets, both splits)
    run_unit "$OUT4D/probe_${tag}.log" "HEADLINE" \
        eval/probe_maze.py --ckpt "$ckpt" --data_dir "$data" --seeds 0 1 2 3 4
    # Transplant at the predicted peak layer (L2-L3 region per predictions);
    # we sweep across all layers in a separate per-layer step below.
    for L in 0 1 2 3 4 5; do
        for S in 0 1 2 3 4; do
            run_unit "$OUT4D/transplant_${tag}_layer${L}_seed${S}.log" "Effect-size summary" \
                eval/transplant_maze.py --ckpt "$ckpt" --data_dir "$data" --layer "$L" --seed "$S"
        done
    done
done

log "=== Phase 4-d COMPLETE ==="
log "=== POST-PHASE-2 PIPELINE COMPLETE ==="
log "Next steps (manual):"
log "  - Aggregate Phase 4-d results"
log "  - Write results_maze_navigation.md confirm/falsify table comparing"
log "    observed values to predictions_maze_navigation.md"
