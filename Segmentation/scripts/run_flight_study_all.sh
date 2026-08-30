#!/usr/bin/env bash
# Full flight-design study queue, from scratch to result, unattended.
#
# Deliberate order: the `full` condition runs FIRST and is scored on its own before the others. It is the
# only one whose result we already know (the single-pass baseline, 60.5%), so it works as a smoke test
# of the whole pipeline. If it comes out far from that, something is wrong in the coordinate conversion
# or in the container, and we find out in ~40 min instead of finding out in 5 h.
#
# Usage:
#   nohup bash scripts/run_flight_study_all.sh > /dev/null 2>&1 &
#
# To follow it:
#   tail -f work/flight_study.log
#
# On finishing it writes work/FLIGHT_STUDY_DONE (or _FAILED) with the summary.

set -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="$HOME/miniforge3/envs/greenvista/bin/python"
LOG="$REPO/work/flight_study.log"
LAZ="$REPO/work/lazall"
DONE="$REPO/work/FLIGHT_STUDY_DONE"
FAILED="$REPO/work/FLIGHT_STUDY_FAILED"

mkdir -p "$REPO/work"
rm -f "$DONE" "$FAILED"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=========================================================="
log "flight-design study, start"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unavailable)"
log "=========================================================="

score() {
  PYTHONPATH=. GREENVISTA_LAZ_DIR="$LAZ" "$PY" scripts/exp_flight_study_score.py 2>&1 \
    | grep -v '^ERROR 1' | tee -a "$LOG"
}

# ---------- 0. container with the right mounts ----------
# A persistent container remembers the bind mounts it was created with. If the repository has been
# moved since then, an old container points at a path that no longer exists and fails on start.
# Recreating it once fixes that and costs nothing: the weights (220 MB) and the code come from the
# repository by bind mount, and the writable layer holds only NVIDIA driver configuration.
# Recreating also clears the meta_data state, which can corrupt a run.
STALE=$(docker inspect ff3d-blackwell-container --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}' 2>/dev/null \
        | grep -c "^${REPO%/eucalyptus-volume-prediction}" || true)
if docker ps -a --filter name=ff3d-blackwell-container --format '{{.Names}}' | grep -q .; then
  if [ "${STALE:-0}" -eq 0 ]; then
    log ">>> step 0: container points at an old path, recreating"
    docker rm -f ff3d-blackwell-container >/dev/null 2>&1 || true
  else
    log ">>> step 0: container already points at the current path"
  fi
fi
export RECREATE_CONTAINER="${RECREATE_CONTAINER:-auto}"

# ---------- 1. smoke test with the reference condition ----------
log ">>> step 1 of 3: condition 'full' (known reference, ~60.5%)"
if ! bash scripts/run_flight_study.sh full >>"$LOG" 2>&1; then
  log "FAILED on condition 'full'. Nothing else will be run."
  echo "failure on condition full at $(date -Is)" > "$FAILED"
  exit 1
fi

log ">>> scoring 'full' to check the pipeline"
# Take the FIRST cell of the line ending in '%', which is the detection. Do not use a fixed position: with
# the "range between tiles" column present, $4 is the "a" of "422 a 1311" and not the detection.
# Searching by pattern survives the next column somebody adds.
SMOKE=$(score | awk '/^full/ {for (i = 1; i <= NF; i++) if ($i ~ /%$/) { print $i; exit }}')
log "detection measured on 'full': ${SMOKE:-not read}  (expected close to 60.5%)"
if [ -z "$SMOKE" ]; then
  # Do not go on. Warning and continuing costs ~57 min of GPU on the reference condition and then hours
  # more on the other seven, all with the same defect: a permission error on the collection throws the
  # output away. A smoke test that does not stop is not a smoke test.
  log "FAILED: 'full' ran but produced no readable detection. The collection or the scoring is broken."
  log "Nothing else will be run. Check $LOG and work/ff3d_degraded_out/full/."
  echo "smoke test of condition full with no readable detection at $(date -Is)" > "$FAILED"
  exit 1
fi

# ---------- 2. the rest of the conditions ----------
log ">>> step 2 of 3: remaining conditions (~4 h)"
if ! bash scripts/run_flight_study.sh >>"$LOG" 2>&1; then
  log "FAILED on some condition. Scoring whatever exists anyway."
  score
  echo "partial failure at $(date -Is)" > "$FAILED"
  exit 1
fi

# ---------- 3. final scoring ----------
log ">>> step 3 of 3: final scoring"
score

log "=========================================================="
log "done"
log "=========================================================="
{
  echo "done at $(date -Is)"
  echo
  echo "results:"
  echo "  manual_match/flight_study.csv"
  echo "  manual_match/flight_study_per_plot.csv"
  echo "  manual_match/flight_study_manifest.json"
  echo "  raw outputs in work/ff3d_degraded_out/"
} > "$DONE"
cat "$DONE" | tee -a "$LOG"
