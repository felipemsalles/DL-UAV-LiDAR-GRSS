#!/usr/bin/env python3
"""Reduce the output of one SegmentAnyTree batch to one row per instance.

Runs between batches, not at the end: every tile returns an `_out.laz` of tens of MB
and there are 1405 tiles, which would accumulate tens of GB of raw output. Each batch
becomes a CSV as soon as it closes, and the CSV is what survives the output bucket
being cleared.

Records both positions per instance. The crown centroid is what the project had been
using; the stem base is estimated by the same rule in both models and is the right one
for merging neighbouring tiles, because a leaning crown displaces the centroid metres
from the trunk. Recording both avoids repeating 19 h of GPU time to switch criteria.

Usage: PYTHONPATH=. python scripts/destila_lote_sat.py <batch_dir> <target_csv>
"""
import sys
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_tta_comparison import posicao_da_instancia  # noqa: E402


def destila(pasta: Path):
    linhas = []
    arquivos = sorted(pasta.rglob("*_out.laz"))
    for f in arquivos:
        tile = f.name.replace("_out.laz", "")
        try:
            las = laspy.read(str(f))
            inst = np.asarray(las.PredInstance)
            sem = np.asarray(las.PredSemantic)
            xy = np.column_stack([np.asarray(las.x), np.asarray(las.y)])
            z = np.asarray(las.z)
        except Exception as e:                      # a corrupt tile does not sink the batch
            print(f"  {f.name}: {type(e).__name__} {e}")
            continue
        for i in np.unique(inst[inst > 0]):
            m = inst == i
            solo = sem[m] == 0
            bx, by = posicao_da_instancia(xy[m], z[m], solo, "base")
            cx, cy = xy[m].mean(0)
            linhas.append({
                "tile": tile, "talhao": int(tile[1:4]), "inst": int(i),
                "base_x": bx, "base_y": by, "cent_x": cx, "cent_y": cy,
                "z_max": float(z[m].max()), "n_pts": int(m.sum()),
            })
    return pd.DataFrame(linhas), len(arquivos)


def main():
    pasta, destino = Path(sys.argv[1]), Path(sys.argv[2])
    d, n_arq = destila(pasta)
    if d.empty:
        print(f"NOTHING DISTILLED from {pasta} ({n_arq} files read)")
        raise SystemExit(1)      # fail loudly: an empty batch is an error, not a warning
    destino.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(destino, mode="a", header=not destino.exists(), index=False)
    print(f"{n_arq} tiles -> {len(d)} instances appended to {destino.name}")


if __name__ == "__main__":
    main()
