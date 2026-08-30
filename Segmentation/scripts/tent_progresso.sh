#!/usr/bin/env bash
# REAL progress of the TENT optimizer experiment, not "the process is alive".
# Counts output tiles per run, which is the only thing that actually advances.
set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "=== $(date +%H:%M:%S) ==="
feito=0
for opt in adam sgd; do
  for i in 1 2 3; do
    d="$R/work/tent_opt/${opt}_r${i}"; lg="$R/work/tent_opt/${opt}_r${i}.log"
    n=$(ls "$d"/*.laz 2>/dev/null | wc -l)
    feito=$((feito + n))
    if   [ ! -e "$lg" ];      then st="queued"
    elif [ "$n" -eq 18 ];     then st="COMPLETE"
    else st="running, last: $(grep -aoE '\[[0-9:]+\] [0-9]/6 [^ ]*' "$lg" 2>/dev/null | tail -1)"
    fi
    printf '  %-8s %2s/18  %s\n' "${opt}_r${i}" "$n" "$st"
  done
done
printf '\n  total %d of 108 tiles (%d%%)\n' "$feito" $((feito * 100 / 108))
echo "  container: $(docker ps --filter ancestor=sat-blackwell --format '{{.Status}}' 2>/dev/null || echo none)"
echo "  memory:    $(free -h | awk '/^Mem:/{print $7" available"}')   swap: $(free -h | awk '/^Swap:/{print $3}')"
