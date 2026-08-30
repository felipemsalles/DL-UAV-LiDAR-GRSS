"""Fragment filter before the merge, in SegmentAnyTree.

SegmentAnyTree produces ~207 instances per tile and NMS at 1.5 m reduces that to ~110,
that is, NMS accumulates two roles: joining repeated views and hiding crown fragments.
A fragment is not a repeated view, it is a piece of a tree that became an instance of
its own, and filtering it beforehand, by instance attribute, is more direct than letting
NMS resolve it by proximity.

Concrete target: plot 7_14, census of 33, which under AdaBN scores 157.6% of the census.
It is the only one of the 13 without an explanation, and fragmentation would show up there.

Per-instance attributes, extracted straight from the outputs:
    n_pts        number of points
    z_max        maximum height above ground (the z of SegmentAnyTree already comes
                 normalised by the terrain)
    z_min        minimum height; a floating crown fragment has a high z_min
    extensao     z_max - z_min, how much the instance covers vertically

The judge is the matched metric of stand 001, the only place with a stem map. A filter
tuned there may not hold on 7_14, which belongs to another stand, so the script reports
the effect in both places.

Usage: PYTHONPATH=. python scripts/exp_filtro_fragmento.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_tta_comparison import (MAPA, PLOTS, RAIO_400, centros_e_censo,  # noqa: E402
                                no_quadrado, parear)
from ff3d_detection_overlap import nms_merge  # noqa: E402

from greenvista import config  # noqa: E402

SAT = config.REPO / "project/models/SegmentAnyTree_blackwell"
RAIO_FUSAO, LIMIAR = 1.5, 2.0
CORTES_PTS = [0, 50, 100, 200, 400, 800]
CORTES_ZMAX = [0.0, 3.0, 5.0, 8.0]
CORTES_EXT = [0.0, 1.0, 2.0, 4.0]


def instancias(pasta, prefixo="t001_"):
    """One record per instance, with the attributes the filter uses."""
    linhas = []
    for f in sorted(Path(pasta).glob(f"{prefixo}*_out.laz")):
        t, p = int(f.name[1:4]), int(f.name.split("_p")[1][:3])
        las = laspy.read(str(f))
        inst = np.asarray(las.PredInstance)
        x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        for i in np.unique(inst[inst > 0]):
            m = inst == i
            zi = z[m]
            linhas.append(dict(plot_id=f"{t}_{p}", cx=x[m].mean(), cy=y[m].mean(),
                               n_pts=int(m.sum()), z_max=float(zi.max()),
                               z_min=float(zi.min()), extensao=float(zi.max() - zi.min())))
    return pd.DataFrame(linhas)


def casada(df, ref, ctr, plots):
    tp = fp = fn = 0
    for pid in plots:
        s = df[df.plot_id == pid]
        cx, cy, _ = ctr[pid]
        if len(s) == 0:
            fn += len(no_quadrado(ref, cx, cy))
            continue
        pred = no_quadrado(nms_merge(s[["cx", "cy"]].to_numpy(float),
                                     s.n_pts.to_numpy(float), RAIO_FUSAO), cx, cy)
        r = no_quadrado(ref, cx, cy)
        t, _ = parear(r, pred, LIMIAR)
        tp += t; fp += len(pred) - t; fn += len(r) - t
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r, tp + fp


def contagem(df, ctr, pid):
    s = df[df.plot_id == pid]
    cx, cy, campo = ctr[pid]
    if len(s) == 0:
        return 0, campo
    fu = nms_merge(s[["cx", "cy"]].to_numpy(float), s.n_pts.to_numpy(float), RAIO_FUSAO)
    d = np.hypot(fu[:, 0] - cx, fu[:, 1] - cy)
    return int((d <= RAIO_400).sum()), campo


def main():
    ctr = centros_e_censo()
    ref_all = gpd.read_file(MAPA)
    REF = np.column_stack([ref_all.geometry.x, ref_all.geometry.y])

    print("reading the instances of stand 001 (judge: stem map)")
    t001 = instancias(SAT / "bucket_out_adabn2", "t001_")
    print(f"  {len(t001)} instances\n")
    print("reading plot 7_14 (judge: field census only)")
    t714 = instancias(SAT / "bucket_out_adabn13", "t007_p014")
    print(f"  {len(t714)} instances\n")

    base_f1, base_p, base_r, base_n = casada(t001, REF, ctr, PLOTS)
    base_714, campo_714 = contagem(t714, ctr, "7_14")
    print(f"no filter: stand 001 F1 {base_f1:.1%} (prec {base_p:.1%} recall {base_r:.1%}, "
          f"{base_n} pred) | 7_14 {base_714}/{campo_714} = {base_714/campo_714:.1%}\n")

    linhas = []
    for cpts in CORTES_PTS:
        for czmax in CORTES_ZMAX:
            for cext in CORTES_EXT:
                sel = ((t001.n_pts >= cpts) & (t001.z_max >= czmax) & (t001.extensao >= cext))
                f, p, r, n = casada(t001[sel], REF, ctr, PLOTS)
                sel7 = ((t714.n_pts >= cpts) & (t714.z_max >= czmax) & (t714.extensao >= cext))
                n7, _ = contagem(t714[sel7], ctr, "7_14")
                linhas.append(dict(min_pts=cpts, min_zmax=czmax, min_ext=cext,
                                   F1=f, precisao=p, revocacao=r, pred=n,
                                   n_714=n7, taxa_714=n7 / campo_714,
                                   sobrou=int(sel.sum()), de=len(t001)))
    df = pd.DataFrame(linhas).sort_values("F1", ascending=False)

    print("=== the 12 best by F1 in stand 001 ===")
    print(f"{'pts':>5} {'zmax':>5} {'ext':>5} | {'F1':>7} {'prec':>7} {'recall':>7} {'pred':>5} | "
          f"{'7_14':>5} {'rate':>7} | {'inst':>12}")
    for _, r in df.head(12).iterrows():
        print(f"{int(r.min_pts):>5} {r.min_zmax:>5.1f} {r.min_ext:>5.1f} | {r.F1:>6.1%} "
              f"{r.precisao:>6.1%} {r.revocacao:>6.1%} {int(r.pred):>5} | "
              f"{int(r.n_714):>5} {r.taxa_714:>6.1%} | {int(r.sobrou):>5}/{int(r.de)}")

    print("\n=== what brings 7_14 closest to the census, and what that costs in 001 ===")
    d7 = df.assign(erro=(df.taxa_714 - 1).abs()).sort_values("erro")
    for _, r in d7.head(6).iterrows():
        print(f"{int(r.min_pts):>5} {r.min_zmax:>5.1f} {r.min_ext:>5.1f} | 7_14 {r.taxa_714:>6.1%} | "
              f"F1 in 001 {r.F1:>6.1%} ({(r.F1-base_f1)*100:+.1f} against no filter)")

    out = config.OUT_DIR / "filtro_fragmento.csv"
    df.to_csv(out, index=False)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
