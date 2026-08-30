#!/usr/bin/env bash
# Runs the AdaBN condition on the SAME 18 tiles, IN PARALLEL with the TENT queue.
#
# Why it exists. The bootstrap over the 211 stems needs the per-stem outcome
# in BOTH conditions, AdaBN and TENT, on the same tiles. This script produces the
# AdaBN half.
#
# Warning: WHY IT NEEDS ITS OWN src AND work. The container mounts two shared
# mutable areas. `sat_pipeline.sh` does `rm -rf` on STAGE and DEST
# inside /home/datascience, and the pipeline also WRITES inside the code, in
# `src/processed_data_ready_for_training_sparse_*/treeinsfused`. Two runs
# pointing at the same folders destroy each other. `SAT_SRC` and `SAT_WORK` isolate them.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="$R/project/models/SegmentAnyTree_blackwell"
OUT="$R/work/tent_opt/adabn_r1"
[ -f "$D/src_par/model_file/PointGroup-PAPER.pt" ] || { echo "ERROR: src_par without checkpoint"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
echo "[adabn r1] starting $(date +%H:%M:%S), in parallel with the TENT queue"
( cd "$D" && SAT_ADABN=1 SAT_SRC="$D/src_par" SAT_WORK="$D/work_par" \
    SAT_IN="$R/work/tent_in" SAT_OUT="$OUT" SAT_MEM=8g \
    bash run_sat_locally.sh ) > "$R/work/tent_opt/adabn_r1.log" 2>&1 \
  || echo "[adabn r1] FAILED, see work/tent_opt/adabn_r1.log"
echo "[adabn r1] end $(date +%H:%M:%S), $(ls "$OUT"/*.laz 2>/dev/null | wc -l) of 18 tiles"
