#!/usr/bin/env bash
# BOTH arms of the FF3D TENT, SGD with momentum and Adam, on the SAME 18 tiles.
#
# Why it exists. The TENT column of Table II mixes optimizers: SAT runs
# Adam and FF3D runs SGD with momentum. SAT is measured on both
# (run_tent_otimizador.sh) and the difference does not survive the bootstrap. This
# supplies the FF3D side, without which the caveat in the paper can only be
# STATED, not resolved.
#
# Wang 2021 p4 says to follow the training hyperparameters of the source model.
# SAT trains with Adam, so its TENT FOLLOWS the recipe. FF3D trains with
# AdamW and our implementation uses SGD momentum 0.9 lr 1e-4, which is the
# ImageNet-C configuration of the TENT paper. The one that departs is FF3D.
#
# Note: THE SGD ARM IS NOT WASTE, it is the validation. It has to reproduce the
# published 0.848. If it does not reproduce it, the Adam number is worth nothing,
# because there is no way to know whether the difference comes from the optimizer
# or from this code path. The PLYs of the published run are not kept, so the only
# way to compare is to run both arms here.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GREENVISTA_TILE_BACKUP="${GREENVISTA_TILE_BACKUP:-$R/work/ff3d_tiles_overlap}"

for opt in sgd adam; do
  DEST="manual_match/ff3d_t001_tent_$opt"
  if [ "$(find "$R/$DEST" -name '*.ply' 2>/dev/null | wc -l)" -eq 18 ]; then
    echo "[$opt] already complete, skipping"; continue
  fi
  echo "=== FF3D TENT with $opt, start $(date +%H:%M:%S) ==="
  FF3D_TENT_OPT="$opt" bash "$R/scripts/run_ff3d_tta_t001.sh" tent "$DEST"
  echo "=== FF3D TENT with $opt, end $(date +%H:%M:%S) ==="
done

args=()
for opt in sgd adam; do
  d="$R/manual_match/ff3d_t001_tent_$opt"
  n=$(find "$d" -name '*.ply' 2>/dev/null | wc -l)
  if [ "$n" -eq 18 ]; then args+=(--ff3d "tent_$opt=$d")
  else echo "  skipping $opt, $n of 18 PLYs"; fi
done
[ ${#args[@]} -gt 0 ] || { echo "ERROR: no complete arm"; exit 1; }

# Note: TWO CORRECTIONS THE DEFAULT SCORER DOES NOT MAKE ON ITS OWN.
#  1. BACKUP. exp_tta_comparison.py resolves the tiles in
#     <GREENVISTA_LAZ_DIR>/../ff3d_tiles_overlap, and the default falls in /tmp, which
#     does not hold the tiles. Without this it prints "no backup tile" 18 times
#     and writes nothing.
#  2. MERGE RADIUS. The script fixes 1.5 m for both models, but the PUBLISHED
#     FF3D configuration uses 1.1 m (Table II). Scoring at 1.5 gives
#     0.823 against the 0.848 of the table and looks like a reproduction failure; at 1.1 it gives
#     0.854. Comparing arms at different radii does not measure any optimizer.
PYTHONPATH="$R" GREENVISTA_LAZ_DIR="$R/work/lazall" GV_REPO="$R" \
  "$HOME/miniforge3/envs/greenvista/bin/python" - "${args[@]}" <<'PYEOF'
import os, sys
R = os.environ["GV_REPO"]
sys.path.insert(0, os.path.join(R, "scripts"))
import exp_tta_comparison as m
m.RAIO_FUSAO = 1.1
sys.argv = ["exp_tta_comparison.py", *sys.argv[1:],
            "--out", os.path.join(R, "manual_match", "ff3d_otimizador.csv")]
m.main()
PYEOF
