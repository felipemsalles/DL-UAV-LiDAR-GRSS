#!/usr/bin/env python3
"""Regenerate `manual_match/sat_adabn_13parcelas.csv` from the distilled CSV.

The CSV this script produces has been versioned since `a9b7fa6`, and its numbers
(92.5% without adaptation, 102.8% with AdaBN, the 737) appear in the project notes
and in the spreadsheet, but the code that generated them was never committed: it was
a one-off command. This script closes that gap, using
`data/detections/sat_13parcelas_instancias.csv` as its source, the distillation of
the 34,772 instances of that run.

The source stores one position per instance, `cx,cy`. Cross-checking against
`tta_t001_instancias.csv`, which separates stem base from crown centroid, the
median distance is 0.053 m to the centroid and 0.166 m to the base, i.e. the
published numbers are crown-centroid ones. Comparing a centroid count with a
stem-base count is not a paired comparison, even though the difference in counts
inside an 11.28 m circle is small.

SegmentAnyTree noise floor: the 18 tiles do not run deterministically across
executions. Two runs of the same 18 tiles with AdaBN give 3,696 and 3,730
instances (0.9% difference), and the count of plot 1_1 goes from 63 to 67. This is
not batch-composition dependence, because `eval.yaml` fixes `batch_size: 1` and
each tile is normalised on its own; what remains is the non-determinism of the
sparse operations on the GPU. Differences of ~4 trees in a plot are within the
noise and should not be read as an effect.

Run: PYTHONPATH=. python scripts/exp_sat_13parcelas.py [--raio 1.5]
"""
import argparse
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_w2w_contagem import nms_malha  # noqa: E402

from greenvista import config  # noqa: E402

FONTE = config.REPO / "data/detections/sat_13parcelas_instancias.csv"
SAIDA = config.OUT_DIR / "sat_adabn_13parcelas.csv"
PR400 = math.sqrt(400 / math.pi)      # 11.28 m, the 400 m² the field crew measured
R12 = 12.0                            # the old radius, kept only to cross-check the historical numbers


def centros_e_censo():
    p = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    for c in ("talhao", "parcela"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    inv = pd.read_csv(config.DATA / "5-dados_campo/inventario_est.csv")
    p = p.dropna(subset=["talhao", "parcela"]).merge(
        inv[["talhao", "parcela", "n_arv"]], on=["talhao", "parcela"], how="left")
    return {f"{int(r.talhao)}_{int(r.parcela)}":
            (r.geometry.x, r.geometry.y, float(r.n_arv)) for _, r in p.iterrows()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raio", type=float, default=1.5, help="NMS fusion radius")
    ap.add_argument("--saida", type=Path, default=SAIDA)
    args = ap.parse_args()

    d = pd.read_csv(FONTE)
    centros = centros_e_censo()
    linhas = []
    for pid, (px, py, campo) in centros.items():
        sub = d[d.plot_id == pid]
        if sub.empty:
            continue
        linha = {"parcela": pid, "campo": int(campo)}
        for cond, rot in (("baseline", "base"), ("adabn", "adabn")):
            g = sub[sub.condicao == cond]
            xy = g[["cx", "cy"]].to_numpy(float)
            keep = nms_malha(xy, g.n_pts.to_numpy(float), args.raio)
            m = xy[keep]
            dist = np.hypot(m[:, 0] - px, m[:, 1] - py)
            linha[f"{rot}_r400"] = int((dist <= PR400).sum())
            linha[f"{rot}_r12"] = int((dist <= R12).sum())
        linhas.append(linha)

    r = pd.DataFrame(linhas)
    for rot in ("base", "adabn"):
        for raio in ("r400", "r12"):
            r[f"t_{rot}_{raio}"] = r[f"{rot}_{raio}"] / r.campo
    r = r[["parcela", "campo", "base_r400", "adabn_r400", "base_r12", "adabn_r12",
           "t_base_r400", "t_adabn_r400", "t_base_r12", "t_adabn_r12"]]
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(args.saida, index=False)

    C = int(r.campo.sum())
    print(r.to_string(index=False))
    print(f"\nfusion radius {args.raio} m, count inside the {PR400:.2f} m circle (400 m²)")
    print(f"  without adaptation {int(r.base_r400.sum())}/{C} = {100*r.base_r400.sum()/C:.1f}%")
    print(f"  with AdaBN         {int(r.adabn_r400.sum())}/{C} = {100*r.adabn_r400.sum()/C:.1f}%")
    print(f"\nwrote {args.saida}")


if __name__ == "__main__":
    main()
