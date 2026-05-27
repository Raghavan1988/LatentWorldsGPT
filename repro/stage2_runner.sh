#!/usr/bin/env bash
# Stage 2-4 sequential pipeline: Othello -> Flight x3 -> Music x3
# Waits for cities runner (PID passed as first arg, optional) then proceeds.
set -euo pipefail

CITIES_PID="${1:-}"
OUT="checkpoints/multiseed_w1"
SEEDS="0 1 2 3 4"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Wait for cities probe to finish
if [[ -n "$CITIES_PID" ]]; then
    log "Waiting for cities runner PID=$CITIES_PID to finish..."
    while kill -0 "$CITIES_PID" 2>/dev/null; do sleep 30; done
    log "Cities runner finished."
fi

##########################################################################
# STAGE 2: Othello probe
##########################################################################
log "=== STAGE 2: Othello probe (5 seeds) ==="
python -u eval/probe_othello.py \
    --ckpt checkpoints/othello_50k/best.pt \
    --data_dir data/othello_50k \
    --seeds $SEEDS \
    2>&1 | tee "$OUT/probe_othello.log"
log "Othello probe DONE"

##########################################################################
# STAGE 3: Flight probe × 3 conditions
##########################################################################
log "=== STAGE 3: Flight probe × 3 conditions ==="

for cond in "" "_within_shuffled" "_global_shuffled"; do
    tag="probe_flight${cond}"
    log "  -> flight${cond}"
    python -u eval/probe_flight.py \
        --ckpt "checkpoints/adsb_5s${cond}/best.pt" \
        --data_dir "data/adsb_5s${cond}" \
        --seeds $SEEDS \
        2>&1 | tee "$OUT/${tag}.log"
    log "  flight${cond} DONE"
done

##########################################################################
# STAGE 4: Music probe × 3 conditions (epochs=50 to save time)
##########################################################################
log "=== STAGE 4: Music probe × 3 conditions ==="

for cond in "" "_within_shuffled" "_global_shuffled"; do
    tag="probe_music${cond}"
    log "  -> music_expanded${cond}"
    python -u eval/probe_music.py \
        --ckpt "checkpoints/music_expanded${cond}/best.pt" \
        --data_dir "data/music_expanded${cond}" \
        --seeds $SEEDS \
        --epochs 50 \
        2>&1 | tee "$OUT/${tag}.log"
    log "  music_expanded${cond} DONE"
done

log "=== ALL STAGES COMPLETE ==="
