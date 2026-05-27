#!/usr/bin/env bash
# Phase 2 finishing pipeline — per-layer transplant ablation + symgroup multi-seed.
#
# RESUMABLE: each unit writes to its own log file with a sentinel marker
# at the end. On re-run we skip units whose log already ends with the
# sentinel. Safe to interrupt (Ctrl-C, lid close → sleep, kill bash, etc.)
# and re-run; will pick up from the last completed unit.
#
# Survivability stack:
#   - caffeinate -dimsu prevents system sleep (launched separately).
#   - nohup invocation detaches from terminal.
#   - Per-unit log writes are atomic (write to tmp, rename).
#   - Sentinel-marker check makes restart idempotent.
set -uo pipefail

PERLAYER_OUT="checkpoints/multiseed_w2_perlayer"
SYMGROUP_OUT="checkpoints/multiseed_w2_symgroup"
mkdir -p "$PERLAYER_OUT" "$SYMGROUP_OUT"

SEEDS="0 1 2 3 4"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Returns 0 if file exists and ends with the expected sentinel.
is_complete() {
    local f="$1"
    local sentinel="$2"
    [ -f "$f" ] && grep -q "$sentinel" "$f" 2>/dev/null
}

# Run a single unit with atomic writes + sentinel check.
run_unit() {
    local out="$1"; local sentinel="$2"; shift 2
    if is_complete "$out" "$sentinel"; then
        log "  SKIP $out (already complete)"
        return 0
    fi
    local tmp="${out}.tmp"
    log "  RUN  $out"
    python -u "$@" > "$tmp" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && grep -q "$sentinel" "$tmp" 2>/dev/null; then
        mv "$tmp" "$out"
    else
        log "  WARN $out failed (rc=$rc); leaving tmp at $tmp"
        # Don't return here — keep going so other units finish; user can retry.
    fi
}

##############################################################################
# (1) PER-LAYER TRANSPLANT ABLATION
#
# For each (domain, condition, layer, seed): run the appropriate transplant
# script with --layer L --seed S. Sentinels are the per-script "Effect
# sizes" or "Effect-size summary" or "Interpretation" lines.
##############################################################################

log "=== (1) PER-LAYER TRANSPLANT ABLATION ==="

# ---- Cities (n_layer=6, layers 0..5) ----
declare -a CITY_CONDS=(
    "cities_london_real|checkpoints/best.pt|data/london_city"
    "cities_london_within|checkpoints/london_shuffled/best.pt|data/london_shuffled"
    "cities_london_global|checkpoints/london_global_shuffled/best.pt|data/london_global_shuffled"
    "cities_manhattan|checkpoints/manhattan/best.pt|data/manhattan"
    "cities_boston|checkpoints/boston/best.pt|data/boston"
)
for entry in "${CITY_CONDS[@]}"; do
    IFS='|' read -r tag ckpt data <<< "$entry"
    for L in 0 1 2 3 4 5; do
        for S in $SEEDS; do
            out="$PERLAYER_OUT/${tag}_layer${L}_seed${S}.log"
            run_unit "$out" "Effect-size summary" \
                eval/transplant.py --ckpt "$ckpt" --data_dir "$data" --layer "$L" --seed "$S"
        done
    done
    log "$tag DONE (6 layers × 5 seeds)"
done

# ---- Othello (n_layer=4, layers 0..3) ----
for L in 0 1 2 3; do
    for S in $SEEDS; do
        out="$PERLAYER_OUT/othello_layer${L}_seed${S}.log"
        run_unit "$out" "Effect sizes" \
            eval/transplant_othello.py --ckpt checkpoints/othello_50k/best.pt --data_dir data/othello_50k --layer "$L" --seed "$S"
    done
done
log "othello per-layer DONE (4 layers × 5 seeds)"

# ---- Flight (n_layer=2, layers 0..1) ----
declare -a FLIGHT_CONDS=(
    "flight_real|checkpoints/adsb_5s/best.pt|data/adsb_5s"
    "flight_within|checkpoints/adsb_5s_within_shuffled/best.pt|data/adsb_5s_within_shuffled"
    "flight_global|checkpoints/adsb_5s_global_shuffled/best.pt|data/adsb_5s_global_shuffled"
)
for entry in "${FLIGHT_CONDS[@]}"; do
    IFS='|' read -r tag ckpt data <<< "$entry"
    for L in 0 1; do
        for S in $SEEDS; do
            out="$PERLAYER_OUT/${tag}_layer${L}_seed${S}.log"
            run_unit "$out" "P(B-phase tokens) gain" \
                eval/transplant_flight.py --ckpt "$ckpt" --data_dir "$data" --layer "$L" --seed "$S"
        done
    done
    log "$tag per-layer DONE (2 layers × 5 seeds)"
done

# ---- Music voice-leading (n_layer=3, layers 0..2) ----
declare -a MUSIC_CONDS=(
    "music_real|checkpoints/music_expanded/best.pt|data/music_expanded"
    "music_within|checkpoints/music_expanded_within_shuffled/best.pt|data/music_expanded_within_shuffled"
    "music_global|checkpoints/music_expanded_global_shuffled/best.pt|data/music_expanded_global_shuffled"
)
for entry in "${MUSIC_CONDS[@]}"; do
    IFS='|' read -r tag ckpt data <<< "$entry"
    for L in 0 1 2; do
        for S in $SEEDS; do
            out="$PERLAYER_OUT/${tag}_layer${L}_seed${S}.log"
            run_unit "$out" "Effect sizes" \
                eval/transplant_music.py --ckpt "$ckpt" --data_dir "$data" --layer "$L" --seed "$S"
        done
    done
    log "$tag per-layer DONE (3 layers × 5 seeds)"
done

log "=== PER-LAYER TRANSPLANT ABLATION COMPLETE ==="

##############################################################################
# (2) SYMGROUP MULTI-SEED AT PARITY
#
# Run probe_symgroup.py on (sa, sa_within_shuffled, sa_global_shuffled) with
# 5 seeds each. Sentinel: "HEADLINE".
##############################################################################

log "=== (2) SYMGROUP MULTI-SEED ==="

declare -a SYMGROUP_CONDS=(
    "symgroup_sa|checkpoints/symgroup_s8_sa/best.pt|data/symgroup_s8_sa"
    "symgroup_sa_within|checkpoints/symgroup_s8_sa_within_shuffled/best.pt|data/symgroup_s8_sa_within_shuffled"
    "symgroup_sa_global|checkpoints/symgroup_s8_sa_global_shuffled/best.pt|data/symgroup_s8_sa_global_shuffled"
)
for entry in "${SYMGROUP_CONDS[@]}"; do
    IFS='|' read -r tag ckpt data <<< "$entry"
    for S in $SEEDS; do
        out="$SYMGROUP_OUT/${tag}_seed${S}.log"
        run_unit "$out" "HEADLINE" \
            eval/probe_symgroup.py --ckpt "$ckpt" --data_dir "$data" --seed "$S"
    done
    log "$tag DONE (5 seeds)"
done

log "=== SYMGROUP MULTI-SEED COMPLETE ==="
log "=== ALL Phase 2 FINISH WORK COMPLETE ==="
