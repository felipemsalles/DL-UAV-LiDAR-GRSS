#!/usr/bin/env bash
# Runs FF3D on the 18 tiles of stand 001 under one test-time adaptation
# condition, and keeps the panoptic PLYs in a folder of their own.
#
# Why it exists. `run_docker_locally.sh` calls the pipeline through `docker exec`,
# and `docker exec` does NOT carry the host environment variables into the
# container. Since `FF3D_TTA_ADAPT` is only read in there, exporting it on the
# host has no effect at all. Here we pass everything with `docker exec -e`.
#
# Precautions this script takes.
#   1. Step 2 of `run_oracle_pipeline.sh` converts the LAZ to PLY and DELETES the
#      input LAZ. So each condition re-stages the tiles from the backup
#      in /tmp/ff3d_tiles_overlap.
#   2. `CLEAR_OUTPUT_BEFORE_RUN` is always true inside the container, so the
#      output of the previous run dies at the start of the next one. We collect
#      between runs, and fail loudly if the collection comes up empty.
#   3. `bucket_out_folder` belongs to root, because the container is what writes it.
#      Moving out of there gives permission denied, because moving needs to write in the source.
#      We copy.
#
# Usage:
#   bash scripts/run_ff3d_tta_t001.sh off      manual_match/ff3d_t001_baseline
#   bash scripts/run_ff3d_tta_t001.sh adabn    manual_match/ff3d_t001_adabn
#   bash scripts/run_ff3d_tta_t001.sh tent     manual_match/ff3d_t001_tent
set -euo pipefail

MODO="${1:?give the mode: off | adabn | session_ema | tent}"
DEST_REL="${2:?give the destination folder}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FF3D="$REPO/project/models/FF3D_inference"
BUCKET_IN="$FF3D/FF3D_oracle/bucket_in_folder"
BUCKET_OUT="$FF3D/FF3D_oracle/bucket_out_folder"
BACKUP="${GREENVISTA_TILE_BACKUP:-/tmp/ff3d_tiles_overlap}"
CONTAINER="ff3d-blackwell-container"
DEST="$REPO/$DEST_REL"

log(){ echo "[$(date +%H:%M:%S)] $*"; }

# Which tiles to stage, and how many to expect. The default is stand 001 with its 18
# views, which is the only one with a stem map. For the 13 whole plots,
# GREENVISTA_TILE_GLOB='*' and GREENVISTA_TILE_N=117.
GLOB="${GREENVISTA_TILE_GLOB:-t001_*}"
ESPERADO="${GREENVISTA_TILE_N:-18}"

[[ -d "$BACKUP" ]] || { echo "tile backup not found in $BACKUP"; exit 1; }
N_TILES=$(find "$BACKUP" -maxdepth 1 -name "${GLOB}.laz" | wc -l)
(( N_TILES == ESPERADO )) || { echo "expected $ESPERADO tiles '$GLOB' in $BACKUP, found $N_TILES"; exit 1; }

############################################
# 1) Container up, with the right binds
############################################
STALE=$(docker inspect "$CONTAINER" --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}' 2>/dev/null \
        | grep -v '^$' | while read -r s; do [[ -d "$s" ]] || echo stale; done | head -1)
if [[ -n "${STALE:-}" ]]; then
  log "container points at a path that no longer exists, recreating"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
fi
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    log "starting existing container"
    docker start "$CONTAINER" >/dev/null
  else
    log "creating container"
    docker run -d --gpus all --shm-size=128g --name "$CONTAINER" \
      -v "$FF3D/ff3d_forestsens:/workspace" \
      --mount "type=bind,source=$BUCKET_IN,target=/workspace/data/ForAINetV2/test_data" \
      --mount "type=bind,source=$BUCKET_OUT,target=/workspace/work_dirs/output" \
      --entrypoint bash "ff3d-blackwell" -lc "sleep infinity" >/dev/null
  fi
fi

############################################
# 2) Stage the tiles again
############################################
log "staging the $N_TILES tiles '$GLOB' in bucket_in_folder"
mkdir -p "$BUCKET_IN"
find "$BUCKET_IN" -maxdepth 1 -type f \( -name '*.laz' -o -name '*.ply' \) -delete
find "$BACKUP" -maxdepth 1 -name "${GLOB}.laz" -exec cp {} "$BUCKET_IN"/ \;

############################################
# 3) Run, with the variable carried by -e
############################################
log "running FF3D with FF3D_TTA_ADAPT=$MODO"
docker exec \
  -e FF3D_TTA_ADAPT="$MODO" \
  -e FF3D_TENT_OPT="${FF3D_TENT_OPT:-sgd}" \
  -e FF3D_TTA_LR="${FF3D_TTA_LR:-1e-4}" \
  -e FF3D_TTA_VERBOSE=1 \
  -e FF3D_TTA_RESET=1 \
  -e FF3D_TTA_SESSION=t001 \
  -e KEEP_ONLY_ZIP=false \
  -e CLEAR_INPUT_AFTER_RUN=false \
  -i "$CONTAINER" bash -lc 'bash /workspace/run_oracle_pipeline.sh'

############################################
# 4) Collect, and fail loudly if it comes up empty
############################################
mkdir -p "$DEST"
rm -f "$DEST"/*.ply 2>/dev/null || true
log "collecting PLYs into $DEST_REL"
shopt -s nullglob
N=0
for f in "$BUCKET_OUT"/*round2*.ply "$BUCKET_OUT"/*_round_2*.ply; do
  cp "$f" "$DEST"/ && N=$((N+1))
done
if (( N == 0 )); then
  for z in "$BUCKET_OUT"/results_*.zip; do
    log "only a zip is left, extracting $(basename "$z")"
    python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$z" "$DEST"
  done
  N=$(find "$DEST" -name '*.ply' | wc -l)
fi
(( N > 0 )) || { echo "EMPTY COLLECTION: nothing in $BUCKET_OUT - the run failed"; exit 1; }
log "collected $N PLYs in $DEST_REL"
du -sh "$DEST"
