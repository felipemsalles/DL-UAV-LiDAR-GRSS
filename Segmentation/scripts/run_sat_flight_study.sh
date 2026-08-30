#!/usr/bin/env bash
# SegmentAnyTree on the SAME 8 conditions of the FF3D flight study.
#
# WHY THIS RUN EXISTS. Figure 4 claimed that the drop in detection with
# density "is of the stand and not of the algorithm". That was not demonstrated: the
# comparison put FF3D in OUR closed stand against a classical method
# in the OPEN stand of da Cunha Neto, changing two things at the same time.
# Running a second algorithm on the SAME degraded tiles gives the missing control.
#
# If SegmentAnyTree also collapses, the explanation by algorithm falls and the sentence
# becomes a demonstration. If it holds up, our claim is wrong, and it is much
# better to find that out now.
#
# Warning: SAME ABLATION AS THE FF3D FLIGHT STUDY. One pass per tile, without the nine views
# and without adaptation. Running SAT in full mode here would compare different
# things and would not answer the question.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SATDIR="$REPO/project/models/SegmentAnyTree_blackwell"
DEG="$REPO/work/ff3d_degraded"
OUTBASE="$REPO/work/sat_flight_out"
IN="$SATDIR/bucket_in"
PROG=/tmp/satvoo_progress.txt
LOG=/tmp/satvoo.log
: > "$PROG"

marca(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$PROG" >> "$LOG"; }
mkdir -p "$OUTBASE"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate greenvista

CONDS=(dens_20 dens_50 dens_100 ang10_dens100 ang_10 dens_200 ang_20 full)
marca "START: ${#CONDS[@]} conditions x 13 tiles"
t0=$(date +%s)

for c in "${CONDS[@]}"; do
  OUT="$OUTBASE/$c"
  if [ -d "$OUT" ] && [ "$(find "$OUT" -name '*_out.laz' | wc -l)" -ge 13 ]; then
    marca "$c: already done, skipping"; continue
  fi
  rm -rf "$IN" "$OUT"; mkdir -p "$IN" "$OUT"
  # Warning: converts to the SAT format (point format 0, geometry only). The
  # degraded tiles were written for FF3D and carry attributes that break
  # the writing of their output.
  PYTHONPATH="$REPO" python "$SATDIR/convert_ff3d_tiles.py" --origem "$DEG/$c" --out "$IN" >> "$LOG" 2>&1
  q=$(ls "$IN"/*.laz 2>/dev/null | wc -l)
  marca "$c: $q tiles converted, running"
  # no SAT_ADABN: the same ablation as the FF3D flight study
  SAT_IN="$IN" SAT_OUT="$OUT" SAT_SEM_VIZ=1 \
    bash "$SATDIR/run_sat_locally.sh" >> "$LOG" 2>&1
  n=$(find "$OUT" -name '*_out.laz' | wc -l)
  el=$(( $(date +%s) - t0 ))
  marca "$c: $n/$q outputs  |  elapsed $((el/60)) min"
done

marca "END: $(find "$OUTBASE" -name '*_out.laz' | wc -l) of 104 tiles"
rm -rf "$IN"
