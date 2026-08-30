#!/usr/bin/env bash
# Compares TENT with Adam against TENT with SGD+momentum, three repetitions each.
#
# Why it exists. Table II of the paper calls "TENT" two columns that used
# DIFFERENT optimizers, Adam in SegmentAnyTree and SGD with momentum 0.9 in FF3D.
# That is a difference of experiment and not of method. Here the two conditions run
# on the SAME model, on the SAME 18 tiles, changing only the optimizer.
#
# Repetition matters: the noise floor between runs is ~2.5 F1 points and the
# AdaBN-TENT difference we want to judge is 1.6 to 2.6. One run per
# condition decides nothing.
#
# Uses the SAME protocol as Table II, via exp_tta_comparison.py: 211 stems from
# the TLS map, 26 m central square, Hungarian assignment, 2 m threshold.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
args=()
for opt in adam sgd; do
  for i in 1 2 3; do
    d="$R/work/tent_opt/${opt}_r${i}"
    n=$(ls "$d"/*.laz 2>/dev/null | wc -l)
    if [ "$n" -eq 18 ]; then args+=(--sat "${opt}_r${i}=$d")
    else echo "  skipping ${opt}_r${i}, $n of 18 tiles"; fi
  done
done
[ ${#args[@]} -gt 0 ] || { echo "ERROR: no complete run"; exit 1; }
echo "complete conditions: $((${#args[@]} / 2))"
PYTHONPATH="$R" "$HOME/miniforge3/envs/greenvista/bin/python" \
  "$R/scripts/exp_tta_comparison.py" "${args[@]}" \
  --out "$R/manual_match/tent_otimizador.csv"
