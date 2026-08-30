#!/usr/bin/env bash
# Experiment: SegmentAnyTree TENT with Adam and with SGD+momentum, the same
# optimizer FF3D uses, to separate method from implementation.
#
# Why it exists. The table in the paper calls "TENT" two runs that used
# DIFFERENT optimizers, Adam in SAT and SGD with momentum 0.9 in FF3D. That is
# a difference of experiment and not of method, and it invalidated the comparison between the
# two columns. Here we run SAT on both, with repetition, because the noise
# floor is ~2.5 F1 points and the AdaBN-TENT difference is 1.6 to 2.6.
#
# Usage: bash scripts/run_tent_otimizador.sh [n_repetitions]
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPS="${1:-3}"
IN="$R/work/tent_in"
[ -d "$IN" ] || { echo "ERROR: $IN missing. Run convert_ff3d_tiles.py first."; exit 1; }
for opt in adam sgd; do
  for i in $(seq 1 "$REPS"); do
    OUT="$R/work/tent_opt/${opt}_r${i}"
    LOG="$R/work/tent_opt/${opt}_r${i}.log"
    if [ -d "$OUT" ] && [ "$(ls -1 "$OUT"/*.laz 2>/dev/null | wc -l)" -eq 18 ]; then
      echo "[$opt r$i] already complete, skipping"; continue
    fi
    rm -rf "$OUT"; mkdir -p "$OUT" "$(dirname "$LOG")"
    echo "[$opt r$i] starting $(date +%H:%M:%S)"
    ( cd "$R/project/models/SegmentAnyTree_blackwell" && \
      SAT_TENT=1 SAT_TENT_OPT="$opt" SAT_IN="$IN" SAT_OUT="$OUT" \
        bash run_sat_locally.sh ) > "$LOG" 2>&1 || echo "[$opt r$i] FAILED, see $LOG"
    n=$(ls -1 "$OUT"/*.laz 2>/dev/null | wc -l)
    echo "[$opt r$i] end $(date +%H:%M:%S), $n of 18 tiles"
  done
done
echo "ALL RUNS FINISHED $(date +%H:%M:%S)"
