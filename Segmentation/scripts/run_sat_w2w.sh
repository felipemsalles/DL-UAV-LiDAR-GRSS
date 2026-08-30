#!/usr/bin/env bash
# Runs SegmentAnyTree over ALL wall-to-wall tiles, in batches.
#
# WHY IN BATCHES. sat_pipeline.sh processes the whole input in a single call
# and the output only materializes in step 6 of 6. A crash in step 4, which is
# the inference and takes 99% of the time, loses everything already computed. With 1405
# tiles that would be betting 19 hours on one throw. In batches of 60, a crash costs
# ~40 min and the rest is saved.
#
# BY INCREASING SIZE. The driver watchdog fires on the large tiles, which is what
# kills a run mid-queue. Sorting by size puts them last on purpose, so that the
# large ones end up isolated in the last batches, where the problem can be handled
# without repeating the small ones.
#
# Warning: EACH BATCH BECOMES CSV BEFORE BEING DELETED. The raw output is tens of GB
# that the disk does not have. The distiller runs between batches and fails loudly if the batch
# comes up empty, instead of warning and going on: deleting the raw output without the CSV in hand
# loses the batch for good.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SATDIR="$REPO/project/models/SegmentAnyTree_blackwell"
TILES="$REPO/work/tiles_w2w"
IN="$SATDIR/bucket_in"
OUTBASE="$REPO/work/sat_w2w_out"
CSV="$REPO/data/detections/sat_w2w_instancias.csv"
LOG=/tmp/w2w.log
PROG=/tmp/w2w_progress.txt
LOTE="${LOTE:-60}"

mkdir -p "$OUTBASE" "$(dirname "$CSV")"
marca(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PROG" >> "$LOG"; }

# Sorts by increasing size. -z/NUL because a file name is data, not text.
mapfile -d '' -t ORDEM < <(find "$TILES" -maxdepth 1 -name '*.laz' -printf '%s\t%p\0' \
  | sort -z -n -k1,1 | cut -z -f2-)
N=${#ORDEM[@]}
NLOTES=$(( (N + LOTE - 1) / LOTE ))
marca "START: $N tiles, $NLOTES batches of up to $LOTE"

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate greenvista

t0=$(date +%s)
feitos=0
for (( b=0; b<NLOTES; b++ )); do
  ini=$(( b * LOTE ))
  fim=$(( ini + LOTE )); (( fim > N )) && fim=$N
  OUT="$OUTBASE/lote$(printf '%02d' "$b")"

  rm -rf "$IN" "$OUT"; mkdir -p "$IN" "$OUT"
  for (( i=ini; i<fim; i++ )); do cp "${ORDEM[$i]}" "$IN/"; done
  q=$(( fim - ini ))
  marca "batch $b/$((NLOTES-1)): $q tiles, starting"

  # Warning: OWN log per batch. With a single log, the CUDA error `grep` finds the
  # error of a previous batch and misclassifies the failure of the current batch, for
  # good. Each attempt has to be judged only by what it wrote itself.
  ok=0
  for tentativa in 1 2; do
    BL="/tmp/w2w_lote${b}_t${tentativa}.log"
    SAT_IN="$IN" SAT_OUT="$OUT" SAT_ADABN=1 SAT_SEM_VIZ=1 \
      bash "$SATDIR/run_sat_locally.sh" > "$BL" 2>&1 && { ok=1; cat "$BL" >> "$LOG"; break; }
    cat "$BL" >> "$LOG"
    if grep -qa 'launch timed out\|CUDA error\|out of memory' "$BL"; then
      marca "batch $b: GPU crashed on attempt $tentativa, waiting 60 s and retrying"
      sleep 60
    else
      marca "batch $b: failed for another reason on attempt $tentativa ($(tail -1 "$BL" | cut -c1-90))"
      sleep 20
    fi
  done

  prod=$(find "$OUT" -name '*_out.laz' | wc -l)
  if [ "$prod" -gt 0 ]; then
    if PYTHONPATH="$REPO" python "$REPO/scripts/destila_lote_sat.py" "$OUT" "$CSV" >> "$LOG" 2>&1; then
      feitos=$(( feitos + prod ))
      rm -rf "$OUT"                      # only delete AFTER the CSV is closed
      marca "batch $b: OK, $prod/$q tiles distilled and raw output released"
    else
      marca "batch $b: DISTILLATION FAILED, keeping $OUT on disk"
    fi
  else
    marca "batch $b: NO OUTPUT (ok=$ok), $q tiles lost"
  fi

  el=$(( $(date +%s) - t0 ))
  if [ "$feitos" -gt 0 ]; then
    tx=$(( el / feitos ))
    falta=$(( (N - feitos) * tx ))
    marca "PROGRESS $feitos/$N tiles, ${tx}s/tile, elapsed $((el/3600))h$(( (el%3600)/60 ))m, ETA $((falta/3600))h$(( (falta%3600)/60 ))m"
  fi
done

marca "END: $feitos of $N tiles processed. CSV in $CSV"
rm -rf "$IN"
