#!/usr/bin/env bash
# Cities grid-classification probe — 5-seed retrofit.
#
# Runs eval/probe_cities_grid.py with --seeds 5 on all 5 city × condition
# combinations. Logs to checkpoints/multiseed_w1/probe_cities_grid_<tag>.log.
#
# Total wall time: ~3-5 hours on M-series MPS depending on city.

set -e
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-5}
N_POS=${N_POS:-20000}
EPOCHS=${EPOCHS:-50}
GRID=${GRID:-10}
OUT=checkpoints/multiseed_w1

mkdir -p "$OUT"

run() {
    local tag="$1"; local ckpt="$2"; local data="$3"
    local log="$OUT/probe_cities_grid_${tag}.log"
    echo "[$(date -Iseconds)] start $tag → $log"
    python -u eval/probe_cities_grid.py \
        --ckpt "$ckpt" --data_dir "$data" \
        --seeds "$SEEDS" --n_positions "$N_POS" \
        --epochs "$EPOCHS" --grid_size "$GRID" \
        2>&1 | tee "$log"
    echo "[$(date -Iseconds)] done  $tag"
}

run london_real            checkpoints/best.pt                       data/london_city
run london_within_shuffled checkpoints/london_shuffled/best.pt       data/london_shuffled
run london_global_shuffled checkpoints/london_global_shuffled/best.pt data/london_global_shuffled
run manhattan_real         checkpoints/manhattan/best.pt              data/manhattan
run boston_real            checkpoints/boston/best.pt                 data/boston

echo "[$(date -Iseconds)] all done"
